"""
HelixLM Heterogeneous Graph Nodes.

Each node type represents a distinct "neural column" cell type.
All nodes implement the HeteroNode interface: forward(x, state, cache) -> (output, new_state).
"""
import math
import random
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rotary_emb
from .mamba2 import Mamba2SSD


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class HeteroNode(nn.Module):
    """
    Base class for all heterogeneous graph nodes.

    Args:
        d_model: Hidden dimension.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# RMSNorm (taken from LLaMA / Mistral)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ---------------------------------------------------------------------------
# Linear Attention Node (Katharopoulos et al.)
# ---------------------------------------------------------------------------

class LinearAttnNode(HeteroNode):
    """
    Linear attention using random Fourier features.
    Complexity O(T * d^2) instead of O(T^2 * d).
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_features: int = 64,
        use_rope: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(d_model)
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_features = n_features
        self.use_rope = use_rope
        self.dropout = nn.Dropout(dropout)

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Random Fourier feature matrix (non-trainable)
        self.register_buffer(
            "omega",
            torch.randn(n_features, self.head_dim) * math.sqrt(2.0),
        )

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """Random Fourier feature map."""
        # x: (..., head_dim)
        proj = x @ self.omega.T  # (..., n_features)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        state: Any = None,
        cache: Any = None,
    ) -> Tuple[torch.Tensor, Any]:
        B, T, D = x.shape
        x = self.norm(x)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope and cache is not None and "freqs_cis" in cache:
            q = apply_rotary_emb(q, cache["freqs_cis"])
            k = apply_rotary_emb(k, cache["freqs_cis"])

        # Feature maps
        q_feat = self._phi(q)  # (B, H, T, 2*n_features)
        k_feat = self._phi(k)  # (B, H, T, 2*n_features)

        # Causal linear attention via cumulative sum
        kv_cum = torch.cumsum(torch.einsum("bhTf,bhTd->bhTfd", k_feat, v), dim=2)
        z_cum = torch.cumsum(k_feat, dim=2)

        # Numerator and denominator
        num = torch.einsum("bhTf,bhTfd->bhTd", q_feat, kv_cum)
        den = torch.einsum("bhTf,bhTf->bhT", q_feat, z_cum).clamp(min=1e-6)

        out = num / den.unsqueeze(-1)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.dropout(self.out_proj(out))
        return x + out, None


# ---------------------------------------------------------------------------
# Full Softmax Attention Node
# ---------------------------------------------------------------------------

class FullAttnNode(HeteroNode):
    """
    Standard multi-head causal softmax attention.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        use_rope: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(d_model)
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        state: Any = None,
        cache: Any = None,
    ) -> Tuple[torch.Tensor, Any]:
        B, T, D = x.shape
        x = self.norm(x)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope and cache is not None and "freqs_cis" in cache:
            q = apply_rotary_emb(q, cache["freqs_cis"])
            k = apply_rotary_emb(k, cache["freqs_cis"])

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.dropout(self.out_proj(out))
        return x + out, None


# ---------------------------------------------------------------------------
# SwiGLU Feedforward Node
# ---------------------------------------------------------------------------

class SwiGLUNode(HeteroNode):
    """
    SwiGLU feedforward (Shazeer 2020).
    """
    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__(d_model)
        self.hidden_dim = expansion * d_model
        self.w1 = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(
        self,
        x: torch.Tensor,
        state: Any = None,
        cache: Any = None,
    ) -> Tuple[torch.Tensor, Any]:
        x = self.norm(x)
        hidden = F.silu(self.w1(x)) * self.w2(x)
        out = self.dropout(self.w3(hidden))
        return x + out, None


# ---------------------------------------------------------------------------
# Gate / Merge Node (input-dependent aggregation)
# ---------------------------------------------------------------------------

class GateNode(HeteroNode):
    """
    Input-dependent learned merge of predecessor activations.
    """
    def __init__(
        self,
        d_model: int,
        n_inputs: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__(d_model)
        self.n_inputs = n_inputs
        self.gate_proj = nn.Linear(d_model * n_inputs, n_inputs)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.gate_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        state: Any = None,
        cache: Any = None,
    ) -> Tuple[torch.Tensor, Any]:
        # x is expected to be a concatenation of predecessor features along the last dim
        # But in the graph execution, we merge by averaging then apply gate
        # For simplicity, we treat x as the merged input and apply a learned transform
        x = self.norm(x)
        # Gate is a simple MLP for now; in full implementation it would receive all predecessors
        gate = torch.sigmoid(self.gate_proj(x.repeat(1, 1, self.n_inputs)))
        gate = gate.mean(dim=-1, keepdim=True)
        out = self.dropout(x * gate)
        return x + out, None


# ---------------------------------------------------------------------------
# Mamba-2 SSM Node
# ---------------------------------------------------------------------------

class Mamba2Node(HeteroNode):
    """
    Mamba-2 SSD node wrapping the Mamba2SSD layer.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dropout: float = 0.0,
    ):
        super().__init__(d_model)
        self.ssm = Mamba2SSD(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Any = None,
        cache: Any = None,
    ) -> Tuple[torch.Tensor, Any]:
        x = self.norm(x)
        out, new_state = self.ssm(x, state=state)
        out = self.dropout(out)
        return x + out, new_state


# ---------------------------------------------------------------------------
# Titans Neural Memory Node (NEW)
# ---------------------------------------------------------------------------

class TitansMemoryNode(HeteroNode):
    """
    Titans-style neural long-term memory node for HelixLM.

    Maintains persistent memory across forward passes using a surprise-gated
    delta rule. Compatible with the existing heterogeneous graph interface.

    Architecture (based on Behrouz et al. 2025 "Titans: Learning to Memorize
    at Test Time" MAC variant):
        - Keys and values are projected from the input hidden states.
        - A persistent memory tensor M (batch, feature_dim, d_model) stores
          the long-term key->value mapping via outer-product updates.
        - Surprise metric = ||v_pred - v_true|| drives update magnitude.
        - Retrieval uses query projection + ELU feature map.

    The memory tensor is returned as ``state`` and can be persisted across
    chunks by the caller (HelixRecurrentBlock / HelixLMCore).
    """
    def __init__(
        self,
        d_model: int,
        feature_dim: int = 64,
        eta_init: float = 0.01,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__(d_model)
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Projections for memory keys, values, and retrieval queries
        self.k_proj = nn.Linear(d_model, feature_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, feature_dim, bias=False)

        # Learnable per-coordinate learning rate for memory updates
        self.eta = nn.Parameter(torch.ones(feature_dim) * eta_init)

        # Feature map: ELU + 1 (standard in Titans / linear attention literature)
        self.phi = lambda x: F.elu(x, alpha=1.0) + 1.0

        # Output projection and dropout
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Persistent memory buffer (lazily initialized per batch)
        # Shape will be (B, feature_dim, d_model)
        self.register_buffer("_memory_template", None, persistent=False)

        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _init_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initialize a zero memory tensor for a new batch."""
        return torch.zeros(batch_size, self.feature_dim, self.d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: Any = None,
        cache: Any = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            x: (B, T, D) input hidden states.
            state: Previous persistent memory tensor (B, feature_dim, D) or None.
            cache: Unused (for API compatibility).

        Returns:
            (output, updated_memory) where updated_memory has shape
            (B, feature_dim, D) and should be passed back on the next chunk.
        """
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        # ------------------------------------------------------------------
        # 1. Retrieve or initialize persistent memory
        # ------------------------------------------------------------------
        if state is not None:
            M = state  # (B, feature_dim, D)
        else:
            M = self._init_memory(B, device, dtype)

        # ------------------------------------------------------------------
        # 2. Pre-norm and project to keys / values
        # ------------------------------------------------------------------
        x_norm = self.norm(x)

        k = self.phi(self.k_proj(x_norm))  # (B, T, feature_dim)
        v = self.v_proj(x_norm)             # (B, T, d_model)

        # ------------------------------------------------------------------
        # 3. Memory update loop (test-time learning)
        # ------------------------------------------------------------------
        # Clamp eta to be positive and non-zero
        eta = self.eta.abs().clamp(min=1e-6)  # (feature_dim,)

        for t in range(T):
            k_t = k[:, t, :]          # (B, feature_dim)
            v_t = v[:, t, :]          # (B, d_model)

            # Surprise metric: deviation of memory prediction from true value
            # v_pred = k_t @ M^T  -> (B, d_model)
            v_pred = torch.einsum('bf,bfd->bd', k_t, M)
            surprise = torch.norm(v_t - v_pred, dim=-1, keepdim=True)  # (B, 1)

            # Delta rule update: outer product of key and value
            # delta = k_t^T @ v_t  -> (B, feature_dim, d_model)
            delta = torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(1))

            # Surprise-gated update with learnable per-coordinate LR
            M = M + eta.view(1, -1, 1) * surprise.unsqueeze(-1) * delta

            # Layer norm over the feature_dim dimension to prevent explosion
            M = F.layer_norm(M, M.shape[-2:])

        # ------------------------------------------------------------------
        # 4. Memory retrieval for output
        # ------------------------------------------------------------------
        q = self.phi(self.q_proj(x_norm))  # (B, T, feature_dim)
        mem_out = torch.einsum('btf,bfd->btd', q, M)  # (B, T, d_model)

        # ------------------------------------------------------------------
        # 5. Output projection + residual
        # ------------------------------------------------------------------
        out = self.dropout(self.out_proj(mem_out))
        output = x + out

        return output, M
