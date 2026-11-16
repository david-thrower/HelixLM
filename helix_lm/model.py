"""
HelixLM Core model: embeddings, recurrent heterogeneous graph, output head, generation.
"""
import math
import random
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HelixConfig
from .recurrent import HelixRecurrentBlock
from .nodes import RMSNorm


class HelixLMCore(nn.Module):
    """
    Core HelixLM model (non-HF wrapper).

    Used internally; for HF compatibility see hf_model.py HelixForCausalLM.
    """
    def __init__(self, cfg: HelixConfig, tie_weights: bool = True, create_output_head: bool = True):
        super().__init__()
        self.cfg = cfg

        # --- Fix A: Global RNG self-seeding (fresh builds only) ---
        # from_pretrained() builds the module under a meta-device context and
        # then loads the weights, so seeding there buys nothing and only resets
        # the caller's global torch/numpy/random state (sampling, data order).
        loading_pretrained = torch.empty(0).device.type == "meta"
        if getattr(cfg, 'seed', None) is not None and not loading_pretrained:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
            random.seed(cfg.seed)
            print(f"[HelixLM] Global RNG seed set to {cfg.seed} "
                  f"(set HelixConfig(seed=None) to disable auto-seeding)")

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.recurrent = HelixRecurrentBlock(cfg)
        self.out_norm = RMSNorm(cfg.d_model)

        self.head = None
        if create_output_head:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if tie_weights:
                self.head.weight = self.embed.weight  # weight tying

        # RoPE: applied inside the attention nodes (nodes.apply_rotary, driven by
        # cfg.use_rope / cfg.rope_theta). The old precomputed freqs_cis buffer
        # was never read, and rope.py used a different (interleaved) pairing
        # than nodes.apply_rotary, so both were removed.

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, cca_step: Optional[int] = None) -> torch.Tensor:
        e = self.embed(token_ids)
        h = self.recurrent(e, e, attention_mask=attention_mask, cca_step=cca_step)
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
