"""
helix_lm/nas_search_space.py

Neural Architecture Search space for HelixLM.

Design philosophy:
- The search algorithm samples HIGH-LEVEL architectural and training params.
- The config_builder deterministically expands these into NESTED node-level
  parameters that the search is "opaque to". This keeps the search dimensionality
  manageable while still exploring the full design space of the heterogeneous
  recurrent graph.
- Seq len (128/256/512) is an ECONOMICS grid, not an accuracy variable. We
  run studies across all three to predict scaling costs.
- Counter-intuitive HelixLM constraints are hard-coded:
    * grad_accum <= 2 for stability at high LR (3e-3+)
    * FP32 forced for seq_len <= 128 (FP16 overflows in short recurrent paths)
    * Native batch size preferred over grad_accum (recurrent loops create
      path-dependent gradients that clash when accumulated across micro-batches)
- All 3 attention modes supported: linear, hybrid, full
- AdamW hyperparameters (beta1, beta2, eps, weight_decay) are searchable
"""

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import optuna


def get_search_space_bounds() -> Dict[str, List[Any]]:
    """Return human-readable bounds for documentation."""
    return {
        "d_model": [128, 256, 384],
        "n_columns": [2, 3, 4],
        "nodes_per_column": ["(2,2)", "(2,3,2)", "(3,4,4,3)", "(3,4,4,4,3)"],
        "n_loops": [1, 2, 3, 4],
        "use_ssm": [ False ],
        "use_titans": [ False ],
        "attention_mode": ["linear", "hybrid", "full"],
        "lr": [1e-3, 3e-3, 5e-3, 1e-2],
        "seq_len": [128, 256, 512],
        "ffn_expansion": [2.0, 2.5, 3.0],
        "dropout": [0.0, 0.05, 0.1],
        "weight_decay": [0.0, 0.01, 0.05, 0.1],
        "beta1": [0.9, 0.95],
        "beta2": [0.999, 0.98],
        "adam_eps": [1e-8, 1e-6],
        "grad_clip": [0.5, 1.0, 2.0],
        "warmup_ratio": [0.01, 0.05, 0.1],
        "grad_accum": [1, 2],
    }


def sample_params(trial: optuna.Trial, seq_len: Optional[int] = None) -> Dict[str, Any]:
    """
    Sample a high-level configuration from the search space.

    Conditional / nested parameters are handled here so the main NAS script
    stays clean. Node-level details are NOT sampled — they are derived in
    build_helix_config().
    """
    params: Dict[str, Any] = {}

    # ========================================================================
    # 1. Core Architecture (all independent categoricals)
    # ========================================================================
    params["d_model"] = trial.suggest_categorical("d_model", [128, 256, 384])
    params["n_columns"] = trial.suggest_categorical("n_columns", [2, 3, 4])
    params["nodes_per_column"] = trial.suggest_categorical(
        "nodes_per_column", ["(2,2)", "(2,3,2)", "(3,4,4,3)", "(3,4,4,4,3)"]
    )
    params["n_loops"] = trial.suggest_categorical("n_loops", [1, 2, 3, 4])
    params["ffn_expansion"] = trial.suggest_categorical("ffn_expansion", [2.0, 2.5, 3.0])
    params["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])

    # ========================================================================
    # 2. Attention Topology (3 modes: linear, hybrid, full)
    # ========================================================================
    params["attention_mode"] = trial.suggest_categorical("attention_mode", ["linear", "hybrid", "full"])
    if params["attention_mode"] == "hybrid":
        params["hybrid_full_attention_interval"] = trial.suggest_categorical(
            "hybrid_full_attention_interval", [1, 2, 4]
        )
    else:
        params["hybrid_full_attention_interval"] = None

    # ========================================================================
    # 3. Optional Modules (conditional nested params)
    # ========================================================================
    params["use_ssm"] = trial.suggest_categorical("use_ssm", [False, True])
    if params["use_ssm"]:
        params["ssm_d_state"] = trial.suggest_categorical("ssm_d_state", [64, 128])
        params["ssm_dt_rank"] = trial.suggest_categorical("ssm_dt_rank", ["auto", 16, 32])
        params["ssm_d_conv"] = trial.suggest_categorical("ssm_d_conv", [3, 4])
        params["ssm_expand"] = trial.suggest_categorical("ssm_expand", [2, 3])
    else:
        params["ssm_d_state"] = None
        params["ssm_dt_rank"] = None
        params["ssm_d_conv"] = None
        params["ssm_expand"] = None

    params["use_titans"] = trial.suggest_categorical("use_titans", [False, True])
    if params["use_titans"]:
        params["titans_memory_dim"] = trial.suggest_categorical(
            "titans_memory_dim", [params["d_model"], params["d_model"] // 2]
        )
        params["titans_num_memories"] = trial.suggest_categorical("titans_num_memories", [4, 8, 16])
        params["titans_memory_lr"] = trial.suggest_categorical("titans_memory_lr", [1e-4, 3e-4, 1e-3])
    else:
        params["titans_memory_dim"] = None
        params["titans_num_memories"] = None
        params["titans_memory_lr"] = None

    # ========================================================================
    # 4. AdamW & Training Hyperparameters
    # ========================================================================
    params["lr"] = trial.suggest_categorical("lr", [1e-3, 3e-3, 5e-3, 1e-2])
    params["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.05, 0.1])
    params["beta1"] = trial.suggest_categorical("beta1", [0.9, 0.95])
    params["beta2"] = trial.suggest_categorical("beta2", [0.999, 0.98])
    params["adam_eps"] = trial.suggest_categorical("adam_eps", [1e-8, 1e-6])
    params["grad_clip"] = trial.suggest_categorical("grad_clip", [0.5, 1.0, 2.0])
    params["warmup_ratio"] = trial.suggest_categorical("warmup_ratio", [0.01, 0.05, 0.1])

    # ========================================================================
    # 5. Seq Len — economics grid
    # ========================================================================
    if seq_len is not None:
        params["seq_len"] = seq_len
    else:
        params["seq_len"] = trial.suggest_categorical("seq_len", [128, 256, 512])

    # ========================================================================
    # 6. Batch / Grad Accum — HelixLM-specific constraints
    # ========================================================================
    params["grad_accum"] = trial.suggest_categorical("grad_accum", [1, 2])

    # Batch size derived from VRAM estimate
    vram_mb = estimate_vram(params)
    if vram_mb > 22000:
        params["batch_size"] = trial.suggest_categorical("batch_size", [16, 24, 32])
    elif vram_mb > 14000:
        params["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 24])
    else:
        params["batch_size"] = trial.suggest_categorical("batch_size", [4, 8, 16])

    # ========================================================================
    # 7. Derived / Counter-intuitive flags
    # ========================================================================
    params["force_fp32"] = params["seq_len"] <= 128
    params["effective_batch"] = params["batch_size"] * params["grad_accum"]

    return params


def estimate_vram(params: Dict[str, Any]) -> float:
    """
    Rough VRAM estimate in MB for a given configuration.
    Used to auto-select batch size ranges.
    """
    d = params["d_model"]
    n_col = params["n_columns"]
    loops = params["n_loops"]
    seq = params["seq_len"]
    batch = params.get("batch_size", 32)
    vocab = 50257

    embed_mb = vocab * d * 4 / (1024 ** 2)
    nodes_tuple = eval(params["nodes_per_column"])
    total_nodes = sum(nodes_tuple)
    avg_nodes_per_col = total_nodes / len(nodes_tuple)
    graph_params_mb = n_col * avg_nodes_per_col * 2 * (d ** 2) * 4 / (1024 ** 2)
    activations_mb = batch * seq * d * loops * n_col * 4 / (1024 ** 2)

    if params["use_ssm"]:
        ssm_state = params.get("ssm_d_state", 64)
        ssm_expand = params.get("ssm_expand", 2)
        graph_params_mb *= 1.3
        activations_mb *= 1.2
        activations_mb += batch * seq * ssm_state * ssm_expand * 4 / (1024 ** 2)

    if params["use_titans"]:
        mem_dim = params.get("titans_memory_dim", d)
        n_mem = params.get("titans_num_memories", 8)
        graph_params_mb *= 1.1
        activations_mb += n_mem * mem_dim * seq * 4 / (1024 ** 2)

    optimizer_mb = 2 * (embed_mb + graph_params_mb)
    gradients_mb = embed_mb + graph_params_mb
    total_mb = (embed_mb + graph_params_mb + activations_mb + optimizer_mb + gradients_mb) * 1.2
    return total_mb


def estimate_training_cost(
    params: Dict[str, Any],
    dataset_tokens: int,
    epochs: int,
    instance_cost_per_hour: float,
    tok_per_sec_assumed: Optional[float] = None,
) -> Dict[str, Any]:
    """Predict wall-clock time and cloud cost for a full training run."""
    effective_batch = params["effective_batch"]
    seq_len = params["seq_len"]
    steps_per_epoch = max(1, dataset_tokens // (effective_batch * seq_len))
    total_steps = steps_per_epoch * epochs

    if tok_per_sec_assumed is None:
        base_tok_per_sec = 25000
        loop_penalty = 1.0 / max(1, params["n_loops"] ** 0.7)
        ssm_penalty = 0.7 if params["use_ssm"] else 1.0
        titans_penalty = 0.85 if params["use_titans"] else 1.0
        seq_penalty = 256 / seq_len
        tok_per_sec = base_tok_per_sec * loop_penalty * ssm_penalty * titans_penalty * seq_penalty
    else:
        tok_per_sec = tok_per_sec_assumed

    total_tokens = dataset_tokens * epochs
    wall_seconds = total_tokens / max(tok_per_sec, 1)
    wall_hours = wall_seconds / 3600
    cost = wall_hours * instance_cost_per_hour

    return {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "estimated_tok_per_sec": round(tok_per_sec, 1),
        "wall_hours": round(wall_hours, 2),
        "wall_days": round(wall_hours / 24, 2),
        "estimated_cost_usd": round(cost, 2),
        "instance_cost_per_hour": instance_cost_per_hour,
    }


def assign_node_type(
    col_idx: int,
    node_idx: int,
    n_nodes_in_col: int,
    attention_mode: str,
    use_ssm: bool,
    n_columns: int,
    use_titans: bool,
    hybrid_interval: Optional[int] = None,
) -> str:
    """
    Deterministically assign a node type based on global architecture settings.

    Supports 3 attention modes:
    - "linear": all attention nodes are LinearAttnNode (or Mamba2 if SSM enabled)
    - "hybrid": stagger FullAttnNode and LinearAttnNode by hybrid_interval
    - "full": all attention nodes are FullAttnNode
    """
    # Boundary gates
    if node_idx == n_nodes_in_col - 1 and (col_idx == 0 or col_idx == n_columns - 1):
        return "gate"

    # Titans memory slot
    if use_titans and node_idx == 0 and col_idx == n_columns // 2:
        return "titans_memory"

    if attention_mode == "full":
        # All attention nodes are FullAttnNode
        if node_idx % 2 == 0:
            return "full_attn"
        else:
            return "swiglu"

    elif attention_mode == "hybrid":
        interval = hybrid_interval or 2
        # Stagger full attention across columns
        if (col_idx + node_idx) % interval == 0:
            return "full_attn"
        else:
            return "linear_attn"
    else:
        # Linear-only mode
        if node_idx % 3 == 0 and use_ssm:
            return "mamba2"
        elif node_idx % 3 == 1:
            return "linear_attn"
        else:
            return "swiglu"

    return "swiglu"


def build_node_config(node_type: str, params: Dict[str, Any], col_idx: int, node_idx: int) -> Dict[str, Any]:
    """
    Build the nested parameter dict for a specific node instance.

    Counter-intuitive and topology-dependent:
    - LinearAttnNode feature_dim scales with n_loops (lower rank for deeper loops)
    - FullAttnNode num_heads scales inversely with n_columns
    - SwiGLUNode expansion adjusts per column depth
    """
    d = params["d_model"]
    n_col = params["n_columns"]
    ffn = params["ffn_expansion"]

    if node_type == "linear_attn":
        feature_dim = d // 2 if params["n_loops"] >= 3 else d
        return {
            "type": "LinearAttnNode",
            "num_heads": max(2, d // 64),
            "feature_dim": feature_dim,
            "dropout": params["dropout"],
            "use_rope": True,
            "feature_map": "elu",
        }

    elif node_type == "full_attn":
        heads = max(4, d // 32)
        if n_col >= 4:
            heads = max(2, heads // 2)
        return {
            "type": "FullAttnNode",
            "num_heads": heads,
            "attn_dropout": params["dropout"],
            "use_rope": True,
            "causal": True,
        }

    elif node_type == "swiglu":
        col_depth = eval(params["nodes_per_column"])[col_idx]
        adjusted_ffn = ffn * (1.0 - 0.05 * (col_depth - 2))
        adjusted_ffn = max(1.5, adjusted_ffn)
        return {
            "type": "SwiGLUNode",
            "expansion_factor": round(adjusted_ffn, 2),
            "dropout": params["dropout"],
            "activation": "silu",
        }

    elif node_type == "mamba2":
        d_state = params.get("ssm_d_state", 64)
        d_state = min(d_state, d // 2)
        return {
            "type": "Mamba2Node",
            "d_state": d_state,
            "d_conv": params.get("ssm_d_conv", 4),
            "dt_rank": params.get("ssm_dt_rank", "auto"),
            "expand": params.get("ssm_expand", 2),
            "use_mem_eff_path": True,
        }

    elif node_type == "gate":
        return {
            "type": "GateNode",
            "aggregation": "learned_softmax",
            "max_inputs": 4,
            "dropout": 0.0,
        }

    elif node_type == "titans_memory":
        return {
            "type": "TitansMemoryNode",
            "memory_dim": params.get("titans_memory_dim", d),
            "num_memories": params.get("titans_num_memories", 8),
            "memory_lr": params.get("titans_memory_lr", 3e-4),
        }

    else:
        return {"type": "SwiGLUNode", "expansion_factor": ffn}


def build_helix_config(
    params: Dict[str, Any],
    vocab_size: int,
    device: str,
    tokenizer_name: str = "gpt2",
) -> Any:
    """
    Build a HelixConfig from sampled high-level params.

    The NAS is "opaque" to node-level details — this function deterministically
    expands sampled params into the full nested graph spec.
    """
    from helix_lm import HelixConfig

    nodes_per_column: Tuple[int, ...] = eval(params["nodes_per_column"])

    column_specs: List[List[Dict[str, Any]]] = []
    for col_idx, n_nodes in enumerate(nodes_per_column):
        nodes: List[Dict[str, Any]] = []
        for node_idx in range(n_nodes):
            node_type = assign_node_type(
                col_idx=col_idx,
                node_idx=node_idx,
                n_nodes_in_col=n_nodes,
                attention_mode=params["attention_mode"],
                use_ssm=params["use_ssm"],
                n_columns=params["n_columns"],
                use_titans=params["use_titans"],
                hybrid_interval=params.get("hybrid_full_attention_interval"),
            )
            node_cfg = build_node_config(node_type, params, col_idx, node_idx)
            nodes.append(node_cfg)
        column_specs.append(nodes)

    # Precision rule
    if params["seq_len"] <= 128:
        dtype_str = "float32"
        use_amp = False
    else:
        dtype_str = "float16"
        use_amp = True

    effective_batch = params["batch_size"] * params["grad_accum"]
    steps_per_epoch = max(1, 20000 // effective_batch)
    warmup_steps = max(1, int(steps_per_epoch * params["warmup_ratio"]))

    # NOTE: If HelixConfig.tiny() does not accept all these kwargs,
    # adapt this function to use the correct constructor signature.
    cfg = HelixConfig.tiny(
        vocab_size=vocab_size,
        d_model=params["d_model"],
        n_columns=params["n_columns"],
        nodes_per_column=list(nodes_per_column),
        n_loops=params["n_loops"],
        seq_len=params["seq_len"],
        tokenizer_name=tokenizer_name,
        attention_mode=params["attention_mode"],
        hybrid_full_attention_interval=params.get("hybrid_full_attention_interval", 2),
        use_ssm=params["use_ssm"],
        use_titans_memory=params["use_titans"],
        use_rope=True,
        ffn_expansion=params["ffn_expansion"],
        dropout=params["dropout"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        grad_clip=params["grad_clip"],
        warmup_steps=warmup_steps,
        batch_size=params["batch_size"],
        grad_accum_steps=params["grad_accum"],
        use_amp=use_amp,
        dtype=dtype_str,
        device=device,
        # AdamW nested params
        adam_beta1=params["beta1"],
        adam_beta2=params["beta2"],
        adam_eps=params["adam_eps"],
        # SSM nested
        ssm_d_state=params.get("ssm_d_state"),
        ssm_dt_rank=params.get("ssm_dt_rank"),
        ssm_d_conv=params.get("ssm_d_conv"),
        ssm_expand=params.get("ssm_expand"),
        # Titans nested
        titans_memory_dim=params.get("titans_memory_dim"),
        titans_num_memories=params.get("titans_num_memories"),
        titans_memory_lr=params.get("titans_memory_lr"),
        # Graph topology (opaque to search)
        column_specs=column_specs,
    )

    return cfg


def params_to_flat_dict(params: Dict[str, Any]) -> Dict[str, str]:
    """Flatten nested params for MLflow logging (all values as strings)."""
    flat: Dict[str, str] = {}
    for k, v in params.items():
        if isinstance(v, (list, tuple, dict)):
            flat[k] = str(v)
        elif v is None:
            flat[k] = "null"
        else:
            flat[k] = str(v)
    return flat
