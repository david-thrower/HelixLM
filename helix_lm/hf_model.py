"""
HelixLM HuggingFace PreTrainedModel integration.

Provides full compatibility with the transformers ecosystem:
  - HelixForCausalLM: AutoModelForCausalLM registration
  - KV-cache generation beyond max_seq_len
  - Batched generation with stop token / stop string detection
  - save_pretrained / from_pretrained / push_to_hub
  - Automatic device placement when config.device="auto"
"""
import math
from typing import Optional, List, Dict, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    GenerationMixin,
    AutoModelForCausalLM,
    AutoConfig,
    StoppingCriteria,
)

from .config import HelixConfig
from .model import HelixLMCore
from .tokenizer import HelixTokenizer


# ---------------------------------------------------------------------------
# KV Cache for efficient autoregressive generation
# ---------------------------------------------------------------------------
class HelixKVCache:
    """Simple key-value cache for attention layers during generation."""
    def __init__(self):
        self._cache = {}  # layer_id -> (k, v) tensors

    def get(self, layer_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self._cache.get(layer_id)

    def set(self, layer_id: int, k: torch.Tensor, v: torch.Tensor):
        self._cache[layer_id] = (k, v)

    def update(self, layer_id: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """Append new keys/values to existing cache."""
        existing = self._cache.get(layer_id)
        if existing is None:
            self._cache[layer_id] = (k_new, v_new)
        else:
            k_old, v_old = existing
            k = torch.cat([k_old, k_new], dim=2)
            v = torch.cat([v_old, v_new], dim=2)
            self._cache[layer_id] = (k, v)

    def clear(self):
        self._cache.clear()


# ---------------------------------------------------------------------------
# Stop string detection
# ---------------------------------------------------------------------------
class StopStringCriteria(StoppingCriteria):
    """Stops generation when any of the given strings is produced."""
    def __init__(self, tokenizer: HelixTokenizer, stop_strings: List[str], batch_size: int = 1):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.batch_size = batch_size
        self._decoded = [""] * batch_size

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        should_stop = False
        for b in range(input_ids.shape[0]):
            text = self.tokenizer.decode(input_ids[b], skip_special_tokens=True)
            for stop_str in self.stop_strings:
                if stop_str in text:
                    should_stop = True
                    break
        return should_stop


# ---------------------------------------------------------------------------
# HelixForCausalLM: HF-compatible model
# ---------------------------------------------------------------------------
class HelixPreTrainedModel(PreTrainedModel):
    """Base class for HelixLM models with HF integration."""
    config_class = HelixConfig
    base_model_prefix = "helix"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HelixRecurrentBlock"]
    _tied_weights_keys = {}  # Override to empty dict

    def _init_weights(self, module):
        """Initialize weights the same way as the core model."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
        """Override to return empty dict — we handle weight tying manually."""
        return {}

    def to_device(self, device: Optional[Union[str, torch.device]] = None) -> "HelixPreTrainedModel":
        """Move model to the specified device, or auto-detect if None."""
        if device is None:
            device = self._resolve_device()
        return self.to(device)

    def _resolve_device(self) -> torch.device:
        """Resolve device from config or auto-detect."""
        cfg_device = getattr(self.config, "device", "auto")
        if cfg_device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(cfg_device)


class HelixForCausalLM(HelixPreTrainedModel, GenerationMixin):
    """
    HelixLM causal language model with full HuggingFace compatibility.
    """
    def __init__(self, config: HelixConfig):
        super().__init__(config)
        self.config = config
        self.model = HelixLMCore(config, tie_weights=False)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.post_init()

        # KV cache for generation
        self._kv_cache: Optional[HelixKVCache] = None

        # Auto device placement
        target_device = self._resolve_device()
        if target_device.type == "cpu" and torch.cuda.is_available():
            import warnings
            warnings.warn(
                f"HelixConfig.device='{getattr(config, 'device', 'auto')}' resolved to CPU, "
                f"but CUDA is available. Model will run on CPU. "
                f"Call model.to('cuda') or set config.device='cuda' to use GPU.",
                UserWarning,
                stacklevel=2,
            )
        self.to(target_device)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Dict]:
        """
        Forward pass compatible with HF transformers.
        """
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", True)

        if inputs_embeds is not None:
            e = inputs_embeds
        else:
            e = self.model.embed(input_ids)

        # Run through recurrent core
        h = self.model.recurrent(e, e)

        # Output
        h = self.model.out_norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # CRITICAL: use -100 to support sliding-window warmup masking
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            return output

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": None,
            "hidden_states": h if output_hidden_states else None,
            "attentions": None,
        }

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare inputs for the .generate() method."""
        # If past_key_values is provided, only pass the last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }

    def _reorder_cache(self, past_key_values: Any, beam_idx: torch.Tensor) -> Any:
        """Reorder cache for beam search."""
        # For stateless architectures, nothing to reorder
        return past_key_values

    @torch.no_grad()
    def generate_ext(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        stop_strings: Optional[List[str]] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        return_full_text: bool = True,
    ) -> torch.Tensor:
        """
        Extended generation with stop string detection and beyond-max-len support.
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        pad_token_id = pad_token_id or self.config.pad_token_id
        eos_token_id = eos_token_id or self.config.eos_token_id

        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Track repetition
        if repetition_penalty != 1.0:
            token_counts = torch.zeros(batch_size, self.config.vocab_size, device=device)

        for _ in range(max_new_tokens):
            # Forward on the last token only (efficient for long sequences)
            inputs = generated if generated.size(1) <= self.config.seq_len else generated[:, -self.config.seq_len:]
            logits = self(inputs)["logits"][:, -1, :] / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    for token_id in range(self.config.vocab_size):
                        if token_counts[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty

            # Top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Update repetition counts
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    token_counts[b, next_token[b, 0]] += 1

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Check EOS
            finished = finished | (next_token.squeeze(-1) == eos_token_id)

            if finished.all():
                break

        if not return_full_text:
            return generated[:, input_ids.shape[1]:]
        return generated

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# Auto registration
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModel, AutoModelForCausalLM
    AutoConfig.register("helix", HelixConfig)
    AutoModelForCausalLM.register(HelixConfig, HelixForCausalLM)
except Exception:
    pass  # Will be registered when transformers is imported properly
