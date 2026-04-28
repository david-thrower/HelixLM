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
    _tied_weights_keys = {"lm_head.weight": "token_emb.weight"}

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

        # Tie weights when used as standalone (smoke tests).
        # HelixForCausalLM passes tie_word_embeddings=False and handles
        # tying itself to stay compatible with HF save_pretrained.
        if getattr(cfg, "tie_word_embeddings", True):
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

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Standard autoregressive generation."""
        self.eval()
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(generated)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated
