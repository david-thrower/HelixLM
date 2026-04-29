"""
Heterogeneous neural nodes for HelixLM.

Each node type implements a different computational mechanism inspired by
diverse neural structures in biological brains.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class HeteroNode(nn.Module):
    """Base class for all heterogeneous nodes."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError


class LinearAttnNode(HeteroNode):
    """
    Causal linear attention node using feature maps.
    O(n) in sequence length during training via prefix sums.
    """
    def __init__(self, d_model: int, n_heads: int = 4, feature_dim: int = 64, dropout: float = 0.0):
        super().__init__(d_model)
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.feature_dim = feature_dim

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.q_feat = nn.Linear(self.head_dim, feature_dim, bias=False)
        self.k_feat = nn.Linear(self.head_dim, feature_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None) -> Tuple[torch.Tensor, Any]:
        B, T, D = x.shape
        x = self.norm(x)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # AMP-safe: compute feature maps and cumulatives in fp32 to prevent
        # float16 overflow. The einsum over feature_dim can exceed 65504.
        q_fp32 = self._feature_map(self.q_feat(q.float()))
        k_fp32 = self._feature_map(self.k_feat(k.float()))
        v_fp32 = v.float()

        kv = torch.einsum('bhTf,bhTd->bhTfd', k_fp32, v_fp32)
        kv_cum = torch.cumsum(kv, dim=2)
        z = torch.cumsum(k_fp32, dim=2).sum(dim=-1, keepdim=True).clamp(min=1e-6)

        out = torch.einsum('bhTf,bhTfd->bhTd', q_fp32, kv_cum) / z
        out = out.to(x.dtype)  # cast back to fp16/bf16
        # --------------------------------------------------------------

        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.dropout(self.out_proj(out))
        return out, None


class FullAttnNode(HeteroNode):
    """Standard causal softmax attention with multi-head support and optional RoPE."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0, use_rope: bool = True):
        super().__init__(d_model)
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None) -> Tuple[torch.Tensor, Any]:
        B, T, D = x.shape
        x = self.norm(x)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.resid_dropout(self.out_proj(out))
        return out, None


class DenseNode(HeteroNode):
    """Dense processing node with GELU activation."""
    def __init__(self, d_model: int, expansion: float = 2.0, dropout: float = 0.0):
        super().__init__(d_model)
        h = int(d_model * expansion)
        self.w1 = nn.Linear(d_model, h)
        self.w2 = nn.Linear(h, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None) -> Tuple[torch.Tensor, Any]:
        x = self.norm(x)
        h = F.gelu(self.w1(x))
        out = self.w2(h)
        return self.dropout(out), None


class SwiGLUNode(HeteroNode):
    """SwiGLU activation node as used in modern LLMs."""
    def __init__(self, d_model: int, expansion: float = 2.0, dropout: float = 0.0):
        super().__init__(d_model)
        h = int(d_model * expansion)
        self.gate = nn.Linear(d_model, h, bias=False)
        self.up = nn.Linear(d_model, h, bias=False)
        self.down = nn.Linear(h, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.xavier_uniform_(self.down.weight)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None) -> Tuple[torch.Tensor, Any]:
        x = self.norm(x)
        h = F.silu(self.gate(x)) * self.up(x)
        out = self.down(h)
        return self.dropout(out), None


class SSMNode(HeteroNode):
    """
    Simplified SSM node (Mamba-style) with efficient batched sequential scan.
    For production, replace with Mamba2SSD from mamba2.py.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__(d_model)
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=False
        )
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log.data = torch.log(A)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None) -> Tuple[torch.Tensor, Any]:
        B, T, D = x.shape
        x = self.norm(x)

        x_and_gate = self.in_proj(x)
        x_inner, gate = x_and_gate.chunk(2, dim=-1)
        x_inner = x_inner * F.silu(gate)

        x_conv = self.conv(x_inner.transpose(1, 2))[:, :, :T].transpose(1, 2)

        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log.float())
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        dt = dt.float()
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(2)

        if state is None:
            h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        else:
            h = state

        ys = []
        for t in range(T):
            h = A_bar[:, t] * h + B_bar[:, t] * x_conv[:, t].unsqueeze(-1)
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            y = y + self.D * x_conv[:, t]
            ys.append(y)

        out = torch.stack(ys, dim=1)
        out = self.dropout(self.out_proj(out.to(x.dtype)))
        return out, h.detach()


class Mamba2Node(HeteroNode):
    """Mamba-2 SSD node wrapper."""
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2,
                 dt_rank: str = "auto", conv_bias: bool = True, bias: bool = False, dropout: float = 0.0):
        super().__init__(d_model)
        from .mamba2 import Mamba2SSD
        self.mamba = Mamba2SSD(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand=expand, dt_rank=dt_rank, conv_bias=conv_bias, bias=bias,
            use_fast_path=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state: Any = None, cache: Any = None) -> Tuple[torch.Tensor, Any]:
        x = self.norm(x)
        out, new_state = self.mamba(x, state=state)
        return self.dropout(out), new_state


class GateNode(HeteroNode):
    """Aggregation node with learned softmax weighted sum."""
    def __init__(self, d_model: int, n_preds: int = 2, dropout: float = 0.0):
        super().__init__(d_model)
        self.n_preds = n_preds
        self.weights = nn.Parameter(torch.ones(n_preds) / n_preds)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x_list, state: Any = None, cache: Any = None) -> Tuple[torch.Tensor, Any]:
        if not isinstance(x_list, list):
            raise TypeError("GateNode expects a list of predecessor tensors")
        n = len(x_list)
        if n == 0:
            raise ValueError("GateNode received no predecessors")
        weights = F.softmax(self.weights[:n], dim=0)
        out = sum(w * x for w, x in zip(weights, x_list))
        out = self.out_proj(out)
        return self.dropout(out), None



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

        # 1. Retrieve or initialize persistent memory
        if state is not None:
            M = state  # (B, feature_dim, D)
        else:
            M = self._init_memory(B, device, dtype)

        # 2. Pre-norm and project to keys / values
        x_norm = self.norm(x)

        k = self.phi(self.k_proj(x_norm))  # (B, T, feature_dim)
        v = self.v_proj(x_norm)             # (B, T, d_model)

        # 3. Memory update loop (test-time learning)
        eta = self.eta.abs().clamp(min=1e-6)  # (feature_dim,)

        for t in range(T):
            k_t = k[:, t, :]          # (B, feature_dim)
            v_t = v[:, t, :]          # (B, d_model)

            # Surprise metric: deviation of memory prediction from true value
            v_pred = torch.einsum('bf,bfd->bd', k_t, M)
            surprise = torch.norm(v_t - v_pred, dim=-1, keepdim=True)  # (B, 1)

            # Delta rule update: outer product of key and value
            delta = torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(1))

            # Surprise-gated update with learnable per-coordinate LR
            M = M + eta.view(1, -1, 1) * surprise.unsqueeze(-1) * delta

            # Layer norm over the feature_dim dimension to prevent explosion
            M = F.layer_norm(M, M.shape[-2:])

        # 4. Memory retrieval for output
        q = self.phi(self.q_proj(x_norm))  # (B, T, feature_dim)
        mem_out = torch.einsum('btf,bfd->btd', q, M)  # (B, T, d_model)

        # 5. Output projection + residual
        out = self.dropout(self.out_proj(mem_out))
        output = x + out

        return output, M
