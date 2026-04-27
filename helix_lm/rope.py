"""
Rotary Positional Embeddings (RoPE) for HelixLM.
"""
import torch
import math


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute rotary frequency cis matrix.

    Args:
        dim: Head dimension (must be even).
        end: Maximum sequence length.
        theta: RoPE base frequency.

    Returns:
        Complex tensor of shape (end, dim//2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply rotary embeddings to query/key tensors.

    Args:
        xq: (B, H, T, D) tensor.
        freqs_cis: (T, D//2) complex tensor.

    Returns:
        Rotated tensor of same shape as xq.
    """
    B, H, T, D = xq.shape
    xq_ = xq.float().reshape(B, H, T, D // 2, 2)
    xq_complex = torch.view_as_complex(xq_)
    freqs_cis = freqs_cis[:T].unsqueeze(0).unsqueeze(0)
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
    return xq_out.type_as(xq)
