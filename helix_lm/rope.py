"""
Rotary Positional Embedding (RoPE) for HelixLM.
Supports standard RoPE, extended sequences, and configurable theta.
"""
import math
import torch
import torch.nn as nn


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Precompute RoPE frequency tensor [cos, sin] pairs."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return torch.stack([cos, sin], dim=-1)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to tensor x of shape (..., seq_len, n_heads, head_dim)."""
    *leading_dims, seq_len, n_heads, head_dim = x.shape
    x_reshaped = x.reshape(*leading_dims, seq_len, n_heads, head_dim // 2, 2)
    cos = freqs_cis[:seq_len, :, 0]
    sin = freqs_cis[:seq_len, :, 1]
    for _ in range(len(leading_dims) + 1):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    x0 = x_reshaped[..., 0]
    x1 = x_reshaped[..., 1]
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    y = torch.stack([y0, y1], dim=-1)
    return y.reshape(*leading_dims, seq_len, n_heads, head_dim)


class RoPE(nn.Module):
    """Rotary Positional Embedding module with precomputed frequencies."""
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        return apply_rotary_emb(x, self.freqs_cis)

    def refresh(self, max_seq_len: int, theta: float = None):
        self.max_seq_len = max_seq_len
        if theta is not None:
            self.theta = theta
        freqs_cis = precompute_freqs_cis(self.dim, max_seq_len, self.theta)
        self.freqs_cis = freqs_cis.to(self.freqs_cis.device)
