"""
Mamba-2 SSD layer wrapper for HelixLM.

This is a simplified reference implementation.  For production use, swap in
the official mamba_ssm package.
"""
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba2SSD(nn.Module):
    """
    Simplified Mamba-2 State Space Model (SSD) layer.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        d_conv: Convolution kernel size.
        expand: Expansion factor.
        dt_rank: Rank for delta projection ("auto" -> d_model//16).
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        d_inner = int(expand * d_model)

        if dt_rank == "auto":
            dt_rank = max(1, d_model // 16)
        self.dt_rank = dt_rank

        # Input projection (x -> x, z, B, C, delta)
        self.in_proj = nn.Linear(d_model, d_inner * 2 + d_state * 2 + dt_rank, bias=False)

        # Short convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )

        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        state: Any = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            x: (B, T, D)
            state: Previous SSM state (B, d_inner, d_state) or None.

        Returns:
            (output, new_state)
        """
        B, T, D = x.shape
        d_inner = int(self.expand * self.d_model)

        # Project
        projected = self.in_proj(x)
        x_and_z = projected[..., :d_inner * 2]
        B_param = projected[..., d_inner * 2:d_inner * 2 + self.d_state]
        C_param = projected[..., d_inner * 2 + self.d_state:d_inner * 2 + self.d_state * 2]
        delta_raw = projected[..., -self.dt_rank:]

        x_inner, z = x_and_z.chunk(2, dim=-1)

        # Short convolution
        x_conv = self.conv1d(x_inner.transpose(1, 2))[..., :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Discretization
        delta = F.softplus(self.dt_proj(delta_raw))
        A = -torch.exp(self.A_log.float())

        # Simplified SSM scan (sequential for clarity)
        if state is None:
            h = torch.zeros(B, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        else:
            h = state

        ys = []
        for t in range(T):
            # Discretized update
            dA = torch.exp(A.unsqueeze(0) * delta[:, t, :].unsqueeze(-1))
            dB = delta[:, t, :].unsqueeze(-1) * B_param[:, t, :].unsqueeze(1)
            h = dA * h + dB * x_conv[:, t, :].unsqueeze(-1)
            y = torch.sum(h * C_param[:, t, :].unsqueeze(1), dim=-1)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, T, d_inner)
        y = y + self.D * x_conv

        # Gating
        z = F.silu(z)
        out = y * z

        # Output projection
        out = self.out_proj(out)
        return out, h
