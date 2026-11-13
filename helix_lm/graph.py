"""
HelixGraph: Biological brain-inspired heterogeneous graph executor.

REMEDIATION (2026-08-22):
  - Issue 5: removed root-node cache pre-population so the first multi-scale
    node (and every root computation node) actually executes with raw x.
  - Issue 6: stopped passing output_ffn_dim to ErrorCorrectingMultiScaleAttnNode.
  - Passes new compressed-branch params (compressed_views, compressed_dim,
    compressed_heads, strict_nan_check) to the multi-scale node.
"""
import math
import random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HelixConfig
from .nodes import (
    HeteroNode, LinearAttnNode, FullAttnNode, DenseNode,
    SwiGLUNode, SSMNode, Mamba2Node, GateNode, TitansMemoryNode,
    RMSNorm, ErrorCorrectingMultiScaleAttnNode
)


class HelixGraph(nn.Module):
    """
    Randomly wired heterogeneous graph of neural nodes.

    - Topology: Biological-style neural columns with vertical and lateral connections
    - Aggregation: learned per-node merge (Linear bottleneck) or Gate
    - Stateful nodes (SSM/Mamba-2) expose state read/write across loops

    RNG safety: global torch and numpy RNG states are saved before graph
    construction and restored afterward.  Two HelixGraph instances built with
    the same seed at different times will produce identical topologies.
    """
    def __init__(self, cfg: HelixConfig, seed: int = None):
        super().__init__()
        self.cfg = cfg

        # ── save global RNG state ──────────────────────────────────
        _torch_rng = torch.get_rng_state()
        _numpy_rng = np.random.get_state()
        # ────────────────────────────────────────────────────────────

        seed = getattr(cfg, 'seed', 42) if seed is None else seed
        if seed is not None:
            rng = np.random.RandomState(seed)
            torch.manual_seed(seed)
        else:
            rng = np.random.RandomState()

        self.node_spec = self._build_node_spec()
        self.nodes = nn.ModuleDict()
        self.node_meta: Dict[str, Tuple[int, int, str]] = {}

        nid = 0
        for ci, column in enumerate(self.node_spec):
            for ni, (ntype, ncfg) in enumerate(column):
                name = f"n{nid}"
                self.node_meta[name] = (ci, ni, ntype)
                self.nodes[name] = self._create_node(ntype, ncfg)
                nid += 1

        # Build random wiring
        self.graph: Dict[str, List[str]] = {}
        names = list(self.nodes.keys())

        for name in names:
            ci, idx, ntype = self.node_meta[name]
            preds: List[str] = []

            # Vertical connections
            if ci > 0:
                for pc in range(max(0, ci - cfg.vertical_depth), ci):
                    above = [k for k, v in self.node_meta.items() if v[0] == pc]
                    if above and rng.rand() < cfg.vertical_p:
                        n_pick = rng.randint(1, len(above) + 1)
                        chosen = rng.choice(above, size=min(n_pick, len(above)), replace=False)
                        preds.extend(chosen.tolist())

            # Lateral connections
            same_column = [k for k, v in self.node_meta.items() if v[0] == ci and v[1] < idx]
            for s in same_column:
                if rng.rand() < cfg.lateral_p:
                    preds.append(s)

            # Dead-end failsafe
            if not preds and ci > 0:
                above = [k for k, v in self.node_meta.items() if v[0] == ci - 1]
                if above:
                    preds.append(rng.choice(above))

            self.graph[name] = preds

        # Size gate mixing weights from the ACTUAL wiring. n_preds=len(column)
        # from _build_node_spec is only a guess: random vertical/lateral draws
        # can leave a gate with a single predecessor (softmax over one logit is
        # identically 1 -> dead weights) or with more predecessors than n_preds
        # (weights[:n] truncation silently dropped the extras via zip).
        for name, preds in self.graph.items():
            node = self.nodes[name]
            if isinstance(node, GateNode):
                node.set_n_preds(max(1, len(preds)))

        # Merge layers for multi-predecessor non-gate nodes
        self.merges = nn.ModuleDict()
        self.merge_norms = nn.ModuleDict()
        for name, preds in self.graph.items():
            if len(preds) > 1 and self.node_meta[name][2] != "gate":
                self.merges[name] = nn.Linear(len(preds) * cfg.d_model, cfg.d_model)
                self.merge_norms[name] = RMSNorm(cfg.d_model)

        self.order = self._topsort()
        self.root_nodes = [n for n in names if len(self.graph[n]) == 0 or self.node_meta[n][0] == 0]
        last_col = max(v[0] for v in self.node_meta.values())
        self.sink_nodes = [k for k, v in self.node_meta.items() if v[0] == last_col]

        # Native CCA (Curriculum Component Activation) gates for attention nodes
        self.use_cca = getattr(cfg, 'use_cca', False)
        self.cca_warmup_steps = getattr(cfg, 'cca_warmup_steps', 5000)
        self.cca_ramp_mode = getattr(cfg, 'cca_ramp_mode', 'quadratic')
        self.cca_min_scale = getattr(cfg, 'cca_min_scale', 0.05)
        if self.use_cca:
            self.attention_gates = nn.ParameterDict()
            for name, (ci, idx, ntype) in self.node_meta.items():
                if ntype in ("linear_attn", "full_attn"):
                    self.attention_gates[name] = nn.Parameter(torch.tensor(2.0))
            self._cca_step = 0
            self._cca_total_steps = self.cca_warmup_steps

        # ── restore global RNG state ───────────────────────────────
        torch.set_rng_state(_torch_rng)
        np.random.set_state(_numpy_rng)
        # ────────────────────────────────────────────────────────────

    def _build_node_spec(self) -> List[List[Tuple[str, dict]]]:
        cfg = self.cfg
        spec = []
        for ci in range(cfg.n_columns):
            column = []
            use_full_attn = False
            use_multi_scale = False
            if cfg.attention_mode == "full":
                use_full_attn = True
            elif cfg.attention_mode == "hybrid":
                use_full_attn = (ci % cfg.hybrid_full_attention_interval == 0)
            elif cfg.attention_mode == "multi_scale_windowed":
                use_multi_scale = True

            attn_drop = getattr(cfg, 'attn_dropout', cfg.dropout)
            if use_multi_scale:
                column.append(("error_correcting_multi_scale_attn", {
                    "d_model": cfg.d_model,
                    "n_heads": cfg.n_heads,
                    "local_window": getattr(cfg, "local_window", 64),
                    "coarse_window": getattr(cfg, "coarse_window", 128),
                    "compressed_windows": getattr(cfg, "compressed_windows", 8),
                    "compressed_views": getattr(cfg, "compressed_views", 8),
                    "compressed_dim": getattr(cfg, "compressed_dim", None),
                    "compressed_heads": getattr(cfg, "compressed_heads", None),
                    "corrector_dim": getattr(cfg, "corrector_dim", None),
                    "consensus_type": getattr(cfg, "consensus_type", "cosine"),
                    "corrector_type": getattr(cfg, "corrector_type", "ffn"),
                    "dropout": cfg.dropout,
                    "attn_dropout": attn_drop,
                    "strict_nan_check": getattr(cfg, "strict_nan_check", False),
                    # Issue 6: output_ffn_dim intentionally NOT passed.
                }))
            elif use_full_attn:
                column.append(("full_attn", {
                    "d_model": cfg.d_model, "n_heads": cfg.n_heads,
                    "dropout": cfg.dropout, "use_rope": cfg.use_rope,
                    "attn_dropout": attn_drop,
                }))
            else:
                column.append(("linear_attn", {
                    "d_model": cfg.d_model, "n_heads": cfg.n_heads,
                    "feature_dim": cfg.linear_feature_dim, "dropout": cfg.dropout,
                    "attn_dropout": attn_drop,
                }))

            column.append(("swiglu", {
                "d_model": cfg.d_model, "expansion": cfg.ffn_expansion, "dropout": cfg.dropout,
            }))

            if cfg.use_ssm:
                if hasattr(cfg, 'ssm_d_state') and cfg.ssm_d_state >= 64:
                    column.append(("mamba2", {
                        "d_model": cfg.d_model, "d_state": cfg.ssm_d_state,
                        "d_conv": cfg.ssm_d_conv, "expand": cfg.ssm_expand,
                        "dt_rank": cfg.ssm_dt_rank if hasattr(cfg, 'ssm_dt_rank') else "auto",
                        "conv_bias": cfg.ssm_conv_bias if hasattr(cfg, 'ssm_conv_bias') else True,
                        "bias": cfg.ssm_bias if hasattr(cfg, 'ssm_bias') else False,
                        "dropout": cfg.dropout,
                    }))
                else:
                    column.append(("ssm", {
                        "d_model": cfg.d_model, "d_state": cfg.ssm_d_state,
                        "d_conv": cfg.ssm_d_conv, "expand": cfg.ssm_expand, "dropout": cfg.dropout,
                    }))

            if cfg.use_titans_memory:
                if cfg.titans_always_select and ci == 0:
                    column.append(("titans", {
                        "d_model": cfg.d_model,
                        "feature_dim": cfg.titans_feature_dim,
                        "eta_init": cfg.titans_eta_init,
                        "n_heads": cfg.titans_n_heads,
                        "dropout": cfg.titans_dropout,
                    }))

            if len(column) > 1 or ci > 0:
                column.append(("gate", {
                    "d_model": cfg.d_model, "n_preds": len(column), "dropout": cfg.dropout,
                }))

            spec.append(column)
        return spec

    def _create_node(self, ntype: str, ncfg: dict) -> HeteroNode:
        if ntype == "linear_attn":
            return LinearAttnNode(**ncfg)
        elif ntype == "full_attn":
            return FullAttnNode(**ncfg)
        elif ntype == "dense":
            return DenseNode(**ncfg)
        elif ntype == "swiglu":
            return SwiGLUNode(**ncfg)
        elif ntype == "ssm":
            return SSMNode(**ncfg)
        elif ntype == "mamba2":
            return Mamba2Node(**ncfg)
        elif ntype == "titans":
            return TitansMemoryNode(**ncfg)
        elif ntype == "gate":
            return GateNode(**ncfg)
        elif ntype == "error_correcting_multi_scale_attn":
            return ErrorCorrectingMultiScaleAttnNode(**ncfg)
        raise ValueError(f"Unknown node type: {ntype}")

    def _topsort(self) -> List[str]:
        in_deg = {n: 0 for n in self.nodes}
        adj = {n: [] for n in self.nodes}
        for n, preds in self.graph.items():
            for p in preds:
                adj[p].append(n)
                in_deg[n] += 1
        queue = [n for n, d in in_deg.items() if d == 0]
        out = []
        while queue:
            cur = queue.pop(0)
            out.append(cur)
            for nxt in adj[cur]:
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    queue.append(nxt)
        if len(out) != len(self.nodes):
            remaining = [n for n in self.nodes if n not in out]
            raise ValueError(f"Cycle detected! Remaining: {remaining}")
        return out

    def forward(self, x: torch.Tensor, states: Optional[Dict[str, Any]] = None, attention_mask: Optional[torch.Tensor] = None, cca_step: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if states is None:
            states = {}
        new_states = {}
        cache: Dict[str, torch.Tensor] = {}

        # Issue 5: REMOVED root-node cache pre-population.
        # Previously, every non-stateful root node was cached with raw x, which
        # caused the execution loop to skip it (``if name in cache: continue``).
        # The first multi-scale node in column 0 was therefore instantiated and
        # counted in parameters but never forwarded. Root nodes now execute with
        # merged = x via the ``len(feats) == 0`` branch below.

        for name in self.order:
            preds = self.graph[name]
            if not preds:
                feats = []
            else:
                feats = [cache[p] for p in preds]
            _, _, ntype = self.node_meta[name]

            if ntype == "gate":
                merged = feats if len(feats) > 0 else [x]
            elif len(feats) == 1:
                merged = feats[0]
            elif len(feats) == 0:
                merged = x
            else:
                merged = self.merges[name](torch.cat(feats, dim=-1))
                if name in self.merge_norms:
                    merged = self.merge_norms[name](merged)

            node = self.nodes[name]
            if isinstance(node, (SSMNode, Mamba2Node, TitansMemoryNode)):
                out, s = node(merged, state=states.get(name), attention_mask=attention_mask)
                new_states[name] = s
            elif isinstance(node, GateNode):
                out, _ = node(merged, attention_mask=attention_mask)
            else:
                out, _ = node(merged, attention_mask=attention_mask)

            if self.use_cca and ntype in ("linear_attn", "full_attn") and name in self.attention_gates:
                total = max(1, self._cca_total_steps)
                step = cca_step if cca_step is not None else self._cca_step
                progress = min(1.0, step / total)
                if self.cca_ramp_mode == 'quadratic':
                    scale = progress ** 2
                elif self.cca_ramp_mode == 'cubic_ease':
                    scale = progress ** 2 * (3 - 2 * progress)
                else:
                    scale = progress
                scale = self.cca_min_scale + (1.0 - self.cca_min_scale) * scale
                learned_gate = torch.sigmoid(self.attention_gates[name])
                curriculum_gate = learned_gate * scale
                out = curriculum_gate * out + (1 - curriculum_gate) * merged

            cache[name] = out

        if len(self.sink_nodes) == 1:
            out = cache[self.sink_nodes[0]]
        else:
            out = torch.stack([cache[s] for s in self.sink_nodes], dim=-1).mean(dim=-1)

        return out + x, new_states

    def get_graph_info(self) -> Dict[str, Any]:
        info = {
            "n_nodes": len(self.nodes),
            "n_columns": self.cfg.n_columns,
            "node_types": {},
            "n_edges": sum(len(p) for p in self.graph.values()),
            "roots": self.root_nodes,
            "sinks": self.sink_nodes,
        }
        for name, (ci, idx, ntype) in self.node_meta.items():
            info["node_types"][ntype] = info["node_types"].get(ntype, 0) + 1
        return info
