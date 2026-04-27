"""
HelixLM Core Model.

Wraps token embeddings, the recurrent block, and the language modeling head.
"""
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HelixConfig
from .recurrent import HelixRecurrentBlock
from .rope import precompute_freqs_cis


class HelixLMCore(nn.Module):
    """
    Main HelixLM model.

    Args:
        cfg: HelixConfig instance.
    """
    def __init__(self, cfg: HelixConfig):
        super().__init__()
        self.cfg = cfg

        # Token embeddings
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.token_emb.weight, std=cfg.initializer_range)

        # Recurrent block
        self.recurrent = HelixRecurrentBlock(cfg)

        # LM head
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=cfg.initializer_range)

        # Tie weights if desired (optional)
        self.lm_head.weight = self.token_emb.weight

        # Precompute RoPE frequencies
        if cfg.use_rope:
            freqs_cis = precompute_freqs_cis(
                cfg.d_model // cfg.n_heads,
                cfg.seq_len * 2,
                theta=cfg.rope_theta,
            )
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        else:
            self.freqs_cis = None

    def forward(
        self,
        input_ids: torch.Tensor,
        persistent_states: Optional[Dict[str, Any]] = None,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Args:
            input_ids: (B, T) token IDs.
            persistent_states: Optional persistent node states.
            return_states: If True, return updated persistent states.

        Returns:
            logits: (B, T, vocab_size)
            updated_states: (optional) Dict of persistent states.
        """
        B, T = input_ids.shape
        x = self.token_emb(input_ids)

        # Add RoPE frequencies to cache
        freqs_cis = None
        if self.freqs_cis is not None:
            freqs_cis = self.freqs_cis[:T]

        # Pass through recurrent block
        h, updated_states = self.recurrent(x, persistent_states=persistent_states)

        # LM head
        logits = self.lm_head(h)

        if return_states:
            return logits, updated_states
        return logits

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
