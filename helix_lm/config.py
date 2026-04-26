"""
HelixLM Configuration with HuggingFace PretrainedConfig integration.

Scales from tiny smoke-test models (128 d_model) up to multi-billion
parameter production models via a single dataclass.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import torch
from transformers import PretrainedConfig


class HelixConfig(PretrainedConfig):
    """
    Configuration class for HelixLM models.

    Inherits from PretrainedConfig for seamless HuggingFace integration.
    All dimensions are configurable to allow scaling from proof-of-concept
    to production (100M - 4B parameters).

    Preset recipes:
      - tiny:   d_model=128,  n_columns=2, nodes=(2,2),         ~0.5M
      - small:  d_model=256,  n_columns=3, nodes=(2,3,2),       ~5M
      - base:   d_model=512,  n_columns=4, nodes=(3,4,4,3),     ~25M
      - medium: d_model=768,  n_columns=5, nodes=(3,4,4,4,3),   ~100M
      - large:  d_model=1024, n_columns=6, nodes=(4,5,5,5,5,4),  ~300M
      - xl:     d_model=1536, n_columns=6, nodes=(5,6,6,6,6,5),  ~1B
      - xxl:    d_model=2048, n_columns=7, nodes=(5,6,6,6,6,6,5), ~4B
    """
    model_type = "helix"

    def __init__(
        self,
        # --- Token / Sequence ---
        vocab_size: int = 50257,
        seq_len: int = 2048,
        batch_size: int = 8,

        # --- Core dimensions ---
        d_model: int = 256,
        n_columns: int = 3,
        nodes_per_column: Tuple[int, ...] = (2, 3, 2),

        # --- Attention ---
        attention_mode: str = "hybrid",  # "linear", "full", "hybrid"
        hybrid_full_attention_interval: int = 4,
        n_heads: int = 4,
        k_proj_dim: int = 32,
        dropout: float = 0.05,

        # --- Linear Attention ---
        linear_feature_dim: int = 64,

        # --- SSM (Mamba-2 SSD) ---
        use_ssm: bool = False,
        ssm_d_state: int = 64,
        ssm_d_conv: int = 4,
        ssm_expand: int = 2,
        ssm_dt_rank: str = "auto",
        ssm_conv_bias: bool = True,
        ssm_bias: bool = False,

        # --- Recurrent depth ---
        n_loops: int = 2,
        act_threshold: float = 0.99,
        loop_dim_ratio: float = 0.125,

        # --- Graph topology ---
        lateral_p: float = 0.5,
        vertical_depth: int = 2,
        vertical_p: float = 0.7,
        gate_sinkhorn_iters: int = 5,

        # --- FFN ---
        ffn_expansion: float = 2.0,

        # --- Positional encoding ---
        rope_theta: float = 10000.0,
        use_rope: bool = True,

        # --- Training ---
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        epochs: int = 100,
        warmup_steps: int = 100,
        grad_clip: float = 1.0,

        # --- Initialization ---
        initializer_range: float = 0.02,

        # --- Device ---
        device: str = "auto",
        dtype: str = "float32",

        # --- Tokenizer ---
        tokenizer_name: str = "gpt2",  # "char", "gpt2", "qwen", "custom"
        pad_token_id: int = 0,
        eos_token_id: int = 0,
        bos_token_id: int = 0,

        # --- Generation ---
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,

        # --- Chat / Template ---
        chat_template: Optional[str] = None,
        stop_strings: Optional[List[str]] = None,

        # --- VLM ---
        is_vlm: bool = False,
        vision_encoder: Optional[str] = None,
        vision_hidden_size: int = 768,
        vision_patch_size: int = 16,
        vision_image_size: int = 448,
        vision_num_hidden_layers: int = 24,
        vision_intermediate_size: int = 3072,
        vision_num_attention_heads: int = 16,
        fusion_strategy: str = "perceiver",  # "perceiver", "simple_merge"

        # --- Misc ---
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        # --- HF PretrainedConfig expects these ---
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id or 0
        self.eos_token_id = eos_token_id or 0
        self.bos_token_id = bos_token_id or 0
        self.tie_word_embeddings = tie_word_embeddings

        # --- Core dims ---
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.d_model = d_model
        self.n_columns = n_columns
        self.nodes_per_column = nodes_per_column

        # --- Attention ---
        self.attention_mode = attention_mode
        self.hybrid_full_attention_interval = hybrid_full_attention_interval
        self.n_heads = n_heads
        self.k_proj_dim = k_proj_dim
        self.dropout = dropout
        self.linear_feature_dim = linear_feature_dim

        # --- SSM ---
        self.use_ssm = use_ssm
        self.ssm_d_state = ssm_d_state
        self.ssm_d_conv = ssm_d_conv
        self.ssm_expand = ssm_expand
        self.ssm_dt_rank = ssm_dt_rank
        self.ssm_conv_bias = ssm_conv_bias
        self.ssm_bias = ssm_bias

        # --- Recurrent ---
        self.n_loops = n_loops
        self.act_threshold = act_threshold
        self.loop_dim_ratio = loop_dim_ratio

        # --- Graph ---
        self.lateral_p = lateral_p
        self.vertical_depth = vertical_depth
        self.vertical_p = vertical_p
        self.gate_sinkhorn_iters = gate_sinkhorn_iters

        # --- FFN ---
        self.ffn_expansion = ffn_expansion

        # --- ROPE ---
        self.rope_theta = rope_theta
        self.use_rope = use_rope

        # --- Training ---
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.initializer_range = initializer_range
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        # --- Tokenizer ---
        self.tokenizer_name = tokenizer_name

        # --- Generation ---
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample

        # --- Chat ---
        self.chat_template = chat_template
        self.stop_strings = stop_strings or ["<|endoftext|>", "<|im_end|>", "</s>"]

        # --- VLM ---
        self.is_vlm = is_vlm
        self.vision_encoder = vision_encoder
        self.vision_hidden_size = vision_hidden_size
        self.vision_patch_size = vision_patch_size
        self.vision_image_size = vision_image_size
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_attention_heads = vision_num_attention_heads
        self.fusion_strategy = fusion_strategy

        # --- Validation ---
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.attention_mode in ["linear", "full", "hybrid"]

        # Ensure nodes_per_column matches n_columns
        npc = self.nodes_per_column
        if len(npc) != self.n_columns:
            if len(npc) < self.n_columns:
                npc = npc + (npc[-1],) * (self.n_columns - len(npc))
            else:
                npc = npc[:self.n_columns]
            self.nodes_per_column = npc

        # --- HF compat ---
        self.use_cache = kwargs.get("use_cache", True)

        super().__init__(
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            **kwargs,
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def loop_dim(self) -> int:
        return max(1, int(self.d_model * self.loop_dim_ratio))

    @property
    def ssm_dt_rank_value(self) -> int:
        if self.ssm_dt_rank == "auto":
            return max(1, self.d_model // 16)
        return int(self.ssm_dt_rank)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["dtype"] = str(self.dtype)
        return d

    @classmethod
    def tiny(cls, **kwargs):
        """~0.5M parameters — smoke test / debugging."""
        defaults = dict(
            d_model=128, n_columns=2, nodes_per_column=(2, 2),
            n_heads=4, n_loops=1, seq_len=256, use_ssm=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def small(cls, **kwargs):
        """~5M parameters — experiments and small-scale training."""
        defaults = dict(
            d_model=256, n_columns=3, nodes_per_column=(2, 3, 2),
            n_heads=4, n_loops=2, seq_len=512, use_ssm=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def base(cls, **kwargs):
        """~25M parameters — serious pretraining."""
        defaults = dict(
            d_model=512, n_columns=4, nodes_per_column=(3, 4, 4, 3),
            n_heads=8, n_loops=2, seq_len=1024, use_ssm=True,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def medium(cls, **kwargs):
        """~100M parameters — production small model."""
        defaults = dict(
            d_model=768, n_columns=5, nodes_per_column=(3, 4, 4, 4, 3),
            n_heads=12, n_loops=3, seq_len=2048, use_ssm=True,
            ssm_d_state=64,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def large(cls, **kwargs):
        """~300M parameters — competitive with popular small LLMs."""
        defaults = dict(
            d_model=1024, n_columns=6, nodes_per_column=(4, 5, 5, 5, 5, 4),
            n_heads=16, n_loops=3, seq_len=4096, use_ssm=True,
            ssm_d_state=128, ssm_expand=2,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def xl(cls, **kwargs):
        """~1B parameters — frontier small model."""
        defaults = dict(
            d_model=1536, n_columns=6, nodes_per_column=(5, 6, 6, 6, 6, 5),
            n_heads=24, n_loops=4, seq_len=8192, use_ssm=True,
            ssm_d_state=128, ssm_expand=2,
            ffn_expansion=2.5,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def xxl(cls, **kwargs):
        """~4B parameters — approaching frontier territory."""
        defaults = dict(
            d_model=2048, n_columns=7, nodes_per_column=(5, 6, 6, 6, 6, 6, 5),
            n_heads=32, n_loops=4, seq_len=16384, use_ssm=True,
            ssm_d_state=256, ssm_expand=2,
            ffn_expansion=3.0,
        )
        defaults.update(kwargs)
        return cls(**defaults)
