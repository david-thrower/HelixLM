"""
HuggingFace-compatible wrapper for HelixLM.

Provides generate() and generate_ext() with support for persistent memory
across chunks, enabling arbitrarily long coherent generation.
"""
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn

from .config import HelixConfig
from .model import HelixLMCore


class HelixForCausalLM(nn.Module):
    """
    HF-style causal LM wrapper.

    Args:
        cfg: HelixConfig instance.
    """
    def __init__(self, cfg: HelixConfig):
        super().__init__()
        self.cfg = cfg
        self.core = HelixLMCore(cfg)

    @property
    def config(self):
        """Duck-type HF config."""
        return self.cfg

    def forward(
        self,
        input_ids: torch.Tensor,
        persistent_states: Optional[Dict[str, Any]] = None,
        return_states: bool = False,
    ):
        return self.core(input_ids, persistent_states=persistent_states, return_states=return_states)

    @torch.no_grad()
    def generate_ext(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_persistent_memory: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens with optional persistent memory for long sequences.

        When ``use_persistent_memory`` is True and the sequence exceeds
        ``seq_len``, the model processes in overlapping chunks and carries
        forward the Titans / SSM persistent states, avoiding the hard
        truncation that occurs in the baseline sliding-window approach.

        Args:
            input_ids: (B, T) prompt token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling filter.
            use_persistent_memory: If True, use chunk-wise generation with
                persistent memory states.

        Returns:
            generated: (B, T + max_new_tokens) token IDs.
        """
        self.eval()
        generated = input_ids.clone()
        persistent_states = None
        cfg = self.cfg

        for _ in range(max_new_tokens):
            # Use last seq_len tokens as input chunk
            if generated.size(1) <= cfg.seq_len:
                chunk = generated
            else:
                chunk = generated[:, -cfg.seq_len:]

            if use_persistent_memory:
                logits, persistent_states = self.core(
                    chunk,
                    persistent_states=persistent_states,
                    return_states=True,
                )
            else:
                logits = self.core(chunk)

            # Take logits for the last position
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def count_parameters(self) -> int:
        return self.core.count_parameters()
