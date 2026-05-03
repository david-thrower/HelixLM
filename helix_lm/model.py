"""
HelixLM Core model: embeddings, recurrent heterogeneous graph, output head, generation.
"""
import math
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HelixConfig
from .recurrent import HelixRecurrentBlock
from .rope import precompute_freqs_cis
from .nodes import RMSNorm


class HelixLMCore(nn.Module):
    """
    Core HelixLM model (non-HF wrapper).

    Used internally; for HF compatibility see hf_model.py HelixForCausalLM.
    """
    def __init__(self, cfg: HelixConfig, tie_weights: bool = True, create_output_head: bool = True):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.recurrent = HelixRecurrentBlock(cfg)
        self.out_norm = RMSNorm(cfg.d_model)

        self.head = None
        if create_output_head:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if tie_weights:
                self.head.weight = self.embed.weight  # weight tying

        # RoPE frequencies
        self.register_buffer("freqs_cis", None, persistent=False)
        if cfg.use_rope:
            freqs = precompute_freqs_cis(
                cfg.head_dim,
                cfg.seq_len * 4,
                cfg.rope_theta,
                dtype=cfg.dtype,
            )
            self.register_buffer("freqs_cis", freqs, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        e = self.embed(token_ids)
        h = self.recurrent(e, e.detach())
        if self.head is not None:
            logits = self.head(self.out_norm(h))
        else:
            raise RuntimeError(
                "HelixLMCore was created with create_output_head=False, "
                "but forward() was called without an output head. "
                "Either create the core with create_output_head=True (default), "
                "or ensure the caller provides its own output projection."
            )
        return logits

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        self.eval()
        device = token_ids.device
        generated = token_ids.clone()

        for _ in range(max_new_tokens):
            logits = self(generated)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "non_trainable": total - trainable}
