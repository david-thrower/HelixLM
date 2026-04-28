"""
HuggingFace-compatible wrapper for HelixLM.

Provides HelixForCausalLM as a PreTrainedModel subclass with full HF integration:
  - save_pretrained / from_pretrained
  - generate() with support for persistent memory across chunks
  - labels support for training with Trainer / standard HF pipelines
"""
from typing import Optional, Dict, Any, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import HelixConfig
from .model import HelixLMCore


class HelixForCausalLM(PreTrainedModel):
    """
    HuggingFace PreTrainedModel wrapper for HelixLM.

    Integrates seamlessly with the HF ecosystem:
      - ``AutoModelForCausalLM`` registration
      - ``save_pretrained`` / ``from_pretrained``
      - Training with ``Trainer`` (labels support)
      - Generation with ``generate()`` or ``generate_ext()``

    Args:
        config: HelixConfig instance.
    """

    config_class = HelixConfig
    base_model_prefix = "helix"
    supports_gradient_checkpointing = False
    _tied_weights_keys = {}  # set after core is built

    def __init__(self, config: HelixConfig):
        # PreTrainedModel init handles config, device_map, etc.
        super().__init__(config)
        self.cfg = config
        # Create core with weight tying disabled — we'll handle it ourselves
        orig_tie = getattr(config, "tie_word_embeddings", True)
        config.tie_word_embeddings = False
        self.core = HelixLMCore(config)
        config.tie_word_embeddings = orig_tie
        # Manually tie embeddings the HF-compatible way
        if getattr(config, "tie_word_embeddings", True):
            self.core.lm_head.weight = self.core.token_emb.weight


    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        persistent_states: Optional[Dict[str, Any]] = None,
        return_states: bool = False,
        **kwargs,
    ) -> Union[CausalLMOutputWithPast, Tuple[torch.Tensor, Optional[Dict[str, Any]]]]:
        """
        Forward pass with optional labels for loss computation.

        Args:
            input_ids: (B, T) token IDs.
            attention_mask: (B, T) attention mask (currently unused but kept for
                            HF compatibility).
            labels: (B, T) target token IDs for next-token prediction loss.
                    If provided, returns a CausalLMOutputWithPast with loss.
            persistent_states: Optional persistent node states (for SSM/Titans).
            return_states: If True, return updated persistent states.

        Returns:
            CausalLMOutputWithPast when labels are provided (HF Trainer
            compatible), otherwise (logits, updated_states) or logits.
        """
        logits, updated_states = self.core(
            input_ids,
            persistent_states=persistent_states,
            return_states=True,
        )

        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if labels is not None:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=None,
            )

        if return_states:
            return logits, updated_states
        return logits

    @torch.no_grad()
    def generate_ext(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_persistent_memory: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens with optional persistent memory for long sequences.

        When ``use_persistent_memory`` is True and the sequence exceeds
        ``seq_len``, the model processes in overlapping chunks and carries
        forward the Titans / SSM persistent states.

        Args:
            input_ids: (B, T) prompt token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling filter.
            top_p: Nucleus sampling filter.
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

    def count_parameters(self) -> Dict[str, int]:
        """Return total trainable parameter count."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": total}
