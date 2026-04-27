"""
HelixLM Recurrent Block.

Wraps the heterogeneous graph with depth-wise recurrence (n_loops),
LTI stability injection, and optional ACT halting.
"""
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

from .config import HelixConfig
from .graph import HelixGraph


class HelixRecurrentBlock(nn.Module):
    """
    A single recurrent block that loops over the HelixGraph.

    Args:
        cfg: HelixConfig instance.
    """
    def __init__(self, cfg: HelixConfig):
        super().__init__()
        self.cfg = cfg
        self.graph = HelixGraph(cfg)

        # LTI stability injection matrices
        if cfg.use_lti_injection:
            self.lti_A = nn.Parameter(torch.randn(cfg.d_model, cfg.d_model) * 0.02)
            self.lti_B = nn.Parameter(torch.randn(cfg.d_model, cfg.d_model) * 0.02)
            # Spectral normalization to ensure stability
            if cfg.use_spectral_init:
                with torch.no_grad():
                    u, s, v = torch.svd(self.lti_A)
                    s = s.clamp(max=cfg.lti_spectral_radius)
                    self.lti_A.copy_(u @ torch.diag(s) @ v.T)

        # ACT halting probability
        if cfg.use_act_halting:
            self.act_proj = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        persistent_states: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Args:
            x: (B, T, D) input.
            persistent_states: Node states from previous chunk (for SSM / Titans).

        Returns:
            (output, updated_persistent_states)
        """
        cfg = self.cfg
        h = x
        act_loss = 0.0

        # Initialize persistent states if None
        if persistent_states is None:
            persistent_states = {}

        updated_states = {}

        for loop in range(cfg.n_loops):
            # Run graph with persistent states
            h_new, loop_states = self.graph(h, states=persistent_states)

            # LTI stability injection
            if cfg.use_lti_injection and loop > 0:
                h = torch.matmul(h, self.lti_A) + torch.matmul(h_new, self.lti_B)
            else:
                h = h_new

            # Merge loop states into updated_states (accumulate across loops)
            for k, v in loop_states.items():
                updated_states[k] = v

            # ACT halting (simplified: just check mean activation)
            if cfg.use_act_halting:
                halt_prob = torch.sigmoid(self.act_proj(h.mean(dim=1, keepdim=True)))
                if halt_prob.mean().item() > cfg.act_threshold:
                    break

        return h, updated_states
