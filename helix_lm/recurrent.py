"""
HelixLM Recurrent block with LTI stable injection and Adaptive Computation Time halting.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HelixConfig
from .graph import HelixGraph
from .nodes import RMSNorm


class LTIInjection(nn.Module):
    """Linear Time-Invariant state update for stable recurrent loops."""
    def __init__(self, dim: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self):
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(self, h, e, trans_out):
        A = self.get_A().view(1, 1, -1)
        return A * h + self.B.view(1, 1, -1) * e + trans_out


class ACTHalting(nn.Module):
    """Adaptive Computation Time halting mechanism."""
    def __init__(self, dim: int, threshold: float = 0.99):
        super().__init__()
        self.halt = nn.Linear(dim, 1)
        self.threshold = threshold
        nn.init.xavier_uniform_(self.halt.weight)

    def forward(self, h: torch.Tensor):
        return torch.sigmoid(self.halt(h)).squeeze(-1)


def loop_index_embedding(h: torch.Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0):
    """Add sinusoidal loop-index embedding to hidden state."""
    device = h.device
    dtype = h.dtype
    freqs = 1.0 / (theta ** (torch.arange(0, loop_dim, 2, device=device, dtype=torch.float32) / loop_dim))
    angles = loop_t * freqs
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb.size(0) < loop_dim:
        emb = F.pad(emb, (0, loop_dim - emb.size(0)))
    buf = torch.zeros(h.size(-1), device=device, dtype=dtype)
    buf[:loop_dim] = emb[:loop_dim]
    return h + buf.view(1, 1, -1)


class HelixRecurrentBlock(nn.Module):
    """Recurrent block that loops over the HelixGraph with LTI stability and ACT halting."""
    def __init__(self, cfg: HelixConfig):
        super().__init__()
        self.cfg = cfg
        self.graph = HelixGraph(cfg)
        self.norm = RMSNorm(cfg.d_model)
        self.injection = LTIInjection(cfg.d_model)
        self.act = ACTHalting(cfg.d_model, cfg.act_threshold)
        self.loop_dim = cfg.loop_dim

    def forward(self, h: torch.Tensor, e: torch.Tensor, freqs_cis=None):
        B, T, D = h.shape
        device = h.device

        h_out = torch.zeros_like(h)
        cum_p = torch.zeros(B, T, device=device)
        halted = torch.zeros(B, T, device=device, dtype=torch.bool)
        node_states = {}

        for t in range(self.cfg.n_loops):
            h_loop = loop_index_embedding(h, t, self.loop_dim)
            combined = self.norm(h_loop + e)
            trans_out, node_states = self.graph(combined, states=node_states)
            h = self.injection(h, e, trans_out)
            p = self.act(h)

            still = ~halted
            rem = (1.0 - cum_p).clamp(min=0)
            weight = torch.where(cum_p + p >= self.cfg.act_threshold, rem, p)
            weight = weight * still.float()
            h_out = h_out + weight.unsqueeze(-1) * h
            cum_p = cum_p + p * still.float()
            halted = halted | (cum_p >= self.cfg.act_threshold)

            if halted.all() and not self.training:
                break

        return h_out
