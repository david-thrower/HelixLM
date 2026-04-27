"""
HelixLM Configuration.

Defines model hyperparameters, presets, and architectural wiring rules.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any


@dataclass
class HelixConfig:
    """
    Configuration dataclass for HelixLM.

    All architectural and training hyperparameters are centralized here.
    Preset classmethods provide sensible defaults for common model sizes.
    """

    # -------------------- Core dimensions --------------------
    vocab_size: int = 50257          # GPT-2 vocab size default
    seq_len: int = 256               # Maximum sequence length
    d_model: int = 128               # Hidden dimension
    n_columns: int = 2               # Number of graph columns (depth)
    nodes_per_column: Tuple[int, ...] = (2, 2)  # Nodes in each column
    n_heads: int = 4                 # Attention heads
    n_loops: int = 1                 # Depth-wise recurrence loops
    dropout: float = 0.0             # Dropout rate

    # -------------------- Attention --------------------
    attention_mode: str = "linear"   # "linear", "full", "hybrid"
    hybrid_full_attention_interval: int = 3  # Every Nth column uses full attention in hybrid mode
    use_rope: bool = True              # Rotary positional embeddings
    rope_theta: float = 10000.0        # RoPE base frequency
    use_spectral_init: bool = True     # Spectral normalization for LTI stability

    # -------------------- Recurrent block --------------------
    use_act_halting: bool = False      # Adaptive Computation Time halting
    act_threshold: float = 0.5         # ACT confidence threshold
    use_lti_injection: bool = True     # Linear Time-Invariant stability injection
    lti_spectral_radius: float = 0.95  # Spectral radius cap for LTI matrix A

    # -------------------- Mamba-2 SSM --------------------
    use_mamba2: bool = False           # Enable Mamba-2 SSD nodes
    mamba2_d_state: int = 64           # Mamba-2 state dimension
    mamba2_d_conv: int = 4             # Mamba-2 conv kernel size
    mamba2_expand: int = 2             # Mamba-2 expansion factor
    mamba2_dt_rank: str = "auto"       # Mamba-2 dt projection rank

    # -------------------- Titans Neural Memory --------------------
    use_titans_memory: bool = False    # Enable Titans-style neural memory nodes
    titans_feature_dim: int = 64       # Memory feature dimension (keys)
    titans_eta_init: float = 0.01      # Initial learning rate for memory updates
    titans_n_heads: int = 4            # Number of memory retrieval heads
    titans_dropout: float = 0.0          # Dropout on memory output
    titans_always_select: bool = True  # Guarantee at least one Titans node in graph

    # -------------------- Graph wiring --------------------
    vertical_p: float = 0.5            # Probability of vertical (feedforward) edge
    lateral_p: float = 0.3             # Probability of lateral (skip) edge
    vertical_depth: int = 2            # How many previous columns can connect
    random_seed: Optional[int] = None  # Graph wiring RNG seed (None = random)

    # -------------------- Training --------------------
    lr: float = 5e-4                   # Learning rate
    weight_decay: float = 0.01           # Weight decay
    grad_clip: float = 1.0               # Gradient clipping norm
    epochs: int = 10                     # Training epochs (for smoke tests)
    batch_size: int = 4                  # Batch size
    grad_accum_steps: int = 1            # Gradient accumulation steps
    use_amp: bool = False                # Automatic Mixed Precision (disabled by default for stability)
    warmup_steps: int = 500              # LR warmup steps
    min_tail_len: int = 1                # Minimum document tail length for chunking

    # -------------------- Tokenizer --------------------
    tokenizer_name: str = "gpt2"       # "gpt2", "qwen", "char", or HF name
    pad_token_id: int = 0                # Pad token ID
    eos_token_id: int = 1                # EOS token ID

    # -------------------- Initialization --------------------
    initializer_range: float = 0.02      # Stddev for weight initialization
    dtype: str = "float32"               # Model dtype string

    # -------------------- HF compatibility --------------------
    return_dict: bool = True             # Return dict from forward (HF style)
    use_cache: bool = True               # Use KV cache during generation

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HelixConfig":
        """Deserialize from dictionary."""
        # Filter to known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    # -------------------- Presets --------------------
    @classmethod
    def tiny(cls, **kwargs) -> "HelixConfig":
        """Tiny config for smoke tests."""
        defaults = dict(
            vocab_size=0,  # set after tokenizer
            seq_len=64,
            batch_size=4,
            d_model=128,
            n_columns=2,
            nodes_per_column=(2, 2),
            attention_mode="linear",
            n_heads=4,
            n_loops=1,
            dropout=0.0,
            lr=5e-4,
            weight_decay=0.01,
            epochs=30,
            grad_clip=1.0,
            tokenizer_name="char",
            use_titans_memory=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def small(cls, **kwargs) -> "HelixConfig":
        """Small config for experiments."""
        defaults = dict(
            seq_len=512,
            batch_size=4,
            d_model=256,
            n_columns=3,
            nodes_per_column=(2, 3, 2),
            attention_mode="hybrid",
            n_heads=4,
            n_loops=2,
            dropout=0.05,
            lr=3e-4,
            weight_decay=0.01,
            epochs=10,
            grad_clip=1.0,
            tokenizer_name="gpt2",
            use_titans_memory=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def base(cls, **kwargs) -> "HelixConfig":
        """Base config for pretraining."""
        defaults = dict(
            seq_len=1024,
            batch_size=8,
            d_model=512,
            n_columns=4,
            nodes_per_column=(3, 4, 4, 3),
            attention_mode="hybrid",
            n_heads=8,
            n_loops=2,
            dropout=0.1,
            lr=2e-4,
            weight_decay=0.01,
            epochs=5,
            grad_clip=1.0,
            tokenizer_name="gpt2",
            use_mamba2=True,
            use_titans_memory=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def medium(cls, **kwargs) -> "HelixConfig":
        """Medium config for production small."""
        defaults = dict(
            seq_len=2048,
            batch_size=8,
            d_model=768,
            n_columns=5,
            nodes_per_column=(3, 4, 4, 4, 3),
            attention_mode="hybrid",
            n_heads=12,
            n_loops=3,
            dropout=0.1,
            lr=1.5e-4,
            weight_decay=0.01,
            epochs=3,
            grad_clip=1.0,
            tokenizer_name="gpt2",
            use_mamba2=True,
            use_titans_memory=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def large(cls, **kwargs) -> "HelixConfig":
        """Large config for competitive models."""
        defaults = dict(
            seq_len=4096,
            batch_size=4,
            d_model=1024,
            n_columns=6,
            nodes_per_column=(4, 5, 5, 5, 5, 4),
            attention_mode="hybrid",
            n_heads=16,
            n_loops=3,
            dropout=0.1,
            lr=1e-4,
            weight_decay=0.01,
            epochs=2,
            grad_clip=1.0,
            tokenizer_name="gpt2",
            use_mamba2=True,
            use_titans_memory=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def xl(cls, **kwargs) -> "HelixConfig":
        """XL config for frontier small."""
        defaults = dict(
            seq_len=8192,
            batch_size=2,
            d_model=1536,
            n_columns=6,
            nodes_per_column=(5, 6, 6, 6, 6, 5),
            attention_mode="hybrid",
            n_heads=24,
            n_loops=4,
            dropout=0.1,
            lr=8e-5,
            weight_decay=0.01,
            epochs=2,
            grad_clip=1.0,
            tokenizer_name="gpt2",
            use_mamba2=True,
            use_titans_memory=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def xxl(cls, **kwargs) -> "HelixConfig":
        """XXL config for near-frontier."""
        defaults = dict(
            seq_len=16384,
            batch_size=1,
            d_model=2048,
            n_columns=7,
            nodes_per_column=(5, 6, 6, 6, 6, 6, 5),
            attention_mode="hybrid",
            n_heads=32,
            n_loops=4,
            dropout=0.1,
            lr=5e-5,
            weight_decay=0.01,
            epochs=1,
            grad_clip=1.0,
            tokenizer_name="gpt2",
            use_mamba2=True,
            use_titans_memory=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)
