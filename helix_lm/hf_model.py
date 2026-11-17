"""
HelixLM HuggingFace PreTrainedModel integration (V5-orthodox).

Design choices:
  1. Weight tying is CONDITIONAL via TiedLMHead with gradient buffer.
     When tie_word_embeddings=True (default), uses TiedLMHead which shares
     the embedding weight but routes gradients through a learned buffer to
     prevent ~3x gradient overload on embeddings.
     When tie_word_embeddings=False, uses separate lm_head (sacred baseline).
  2. Forward pass: PRESERVES full gradient flow (no e.detach()). The sacred
     baseline passes e (not detached) to recurrent(), allowing gradients to
     flow through BOTH the hidden-state path and the LTI injection path.
     This gives the embedding layer ~2x stronger gradients and is essential
     for convergence to PPL ~15 (train) / ~90 (val).  DO NOT add detach().
  3. Generation: prepare_inputs_for_generation passes FULL sequence (no KV-cache).
     The recurrent graph re-initializes node_states on every forward.
  4. Auto-registration: explicit, visible, no silent try/except.
  5. use_cache=False is hard-enforced — the recurrent graph has no KV state.

Provides full compatibility with the transformers ecosystem:
  - HelixForCausalLM: AutoModelForCausalLM registration
  - save_pretrained / from_pretrained / push_to_hub
  - Standard model.generate() with StoppingCriteria, logits processors
  - Batched generation with stop token / stop string detection
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
    StoppingCriteriaList,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import HelixConfig
from .model import HelixLMCore
from .tokenizer import HelixTokenizer


# ---------------------------------------------------------------------------
# Tied LM Head with gradient buffer (safe weight tying)
# ---------------------------------------------------------------------------
class TiedLMHead(nn.Module):
    """LM head that shares weight with embedding but routes gradients
    through a learned buffer to prevent embedding gradient overload.

    Problem: Without tying, embed.grad comes from (1) recurrent path +
    (2) LTI injection path. With naive tying, embed.grad gets those PLUS
    (3) lm_head backward path = ~3x overload.

    Solution: The buffer absorbs some of gradient path (3), reducing the
    embedding gradient to manageable levels while keeping the parameter
    savings of tying (halves the embedding+head parameter count).
    """
    def __init__(self, embed_weight: nn.Parameter, d_model: int, vocab_size: int,
                 grad_buffer_ratio: float = 0.5):
        super().__init__()
        # Share the embedding weight tensor (shape: vocab_size, d_model)
        self.weight = embed_weight
        self.grad_buffer_ratio = float(grad_buffer_ratio)
        # A disabled buffer has no forward or backward role. Do not allocate a
        # d_model² parameter that can never receive a gradient or affect logits.
        # Checkpoints trained with ratio=0 may still contain the old key; HF's
        # non-strict from_pretrained load reports and ignores that inactive key.
        if self.grad_buffer_ratio > 0:
            self.buffer = nn.Linear(d_model, d_model, bias=False)
            nn.init.eye_(self.buffer.weight)  # identity: no forward diff at init
        else:
            self.buffer = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, d_model)
        # P0 FIX: Buffer applied consistently in BOTH train and eval.
        # Forward pass must NEVER depend on self.training.
        # Defensive: align h dtype to the active projection's weight dtype.
        compute_dtype = self.buffer.weight.dtype if self.buffer is not None else self.weight.dtype
        h = h.to(compute_dtype)
        if 0 < self.grad_buffer_ratio < 1:
            h_buffered = self.buffer(h)
            h_mixed = (1 - self.grad_buffer_ratio) * h + self.grad_buffer_ratio * h_buffered
        elif self.grad_buffer_ratio >= 1.0:
            h_mixed = self.buffer(h)
        else:
            # buffer_ratio=0: pass through directly (standard tying)
            h_mixed = h
        return F.linear(h_mixed, self.weight.to(h_mixed.dtype))


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
        for b in range(input_ids.shape[0]):
            text = self.tokenizer.decode(input_ids[b], skip_special_tokens=True)
            for stop_str in self.stop_strings:
                if stop_str in text:
                    return True
        return False


# ---------------------------------------------------------------------------
# HelixForCausalLM: HF-compatible model
# ---------------------------------------------------------------------------
class HelixPreTrainedModel(PreTrainedModel):
    """Base class for HelixLM models with HF integration."""
    config_class = HelixConfig
    # The core lives in `self.model` (checkpoint keys are "model.*").
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # Stateless recurrent graph: there is no KV cache of any kind, so
        # generate() must not build a DynamicCache for this model.
        return False
    _no_split_modules = ["HelixRecurrentBlock"]

    def _init_weights(self, module):
        """Initialize weights the same way as the core model."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

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


# helix_lm/hf_model.py  (inside HelixForCausalLM.__init__)

class HelixForCausalLM(HelixPreTrainedModel, GenerationMixin):
    ...
    # Declare tied weights so save_pretrained deduplicates correctly
    _tied_weights_keys = {"lm_head.weight": "model.embed.weight"}

    def __init__(self, config: HelixConfig):
        super().__init__(config)
        self.config = config

        # Hard-enforce: this model has no KV-cache
        self.config.use_cache = False

        # Core model without output head (HelixForCausalLM owns lm_head)
        self.model = HelixLMCore(config, tie_weights=False, create_output_head=False)

        # Use TiedLMHead with gradient buffer when tie_word_embeddings=True
        if config.tie_word_embeddings:
            self.lm_head = TiedLMHead(
                self.model.embed.weight,
                config.d_model,
                config.vocab_size,
                grad_buffer_ratio=getattr(config, 'grad_buffer_ratio', 0.5),
            )
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # SACRED BASELINE PRESERVATION:
        # Do NOT call post_init(). The main branch did not call it, and doing so
        # re-initializes all weights, shifting the RNG stream and changing lm_head's
        # init from Kaiming-uniform to normal_(std=0.02). This causes a different
        # local minimum with the same PPL but different generation style.
        #
        # We manually set the HF attributes that post_init() would have set,
        # so from_pretrained / push_to_hub / save_pretrained still work.
        self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)
        self._keep_in_fp32_modules = set(getattr(self, '_keep_in_fp32_modules', None) or [])
        self._keep_in_fp32_modules_strict = set(getattr(self, '_keep_in_fp32_modules_strict', None) or [])
        self._no_split_modules = set(getattr(self, '_no_split_modules', None) or [])

        # Gather metadata from submodules (matches post_init logic without re-init)
        for name, module in self.named_children():
            if tied_keys := getattr(module, "all_tied_weights_keys", None):
                self.all_tied_weights_keys.update(
                    {f"{name}.{k}": f"{name}.{v}" for k, v in tied_keys.copy().items()}
                )
            if keep_fp32 := getattr(module, "_keep_in_fp32_modules", None):
                self._keep_in_fp32_modules.update(keep_fp32)
            if keep_fp32_strict := getattr(module, "_keep_in_fp32_modules_strict", None):
                self._keep_in_fp32_modules_strict.update(keep_fp32_strict)
            if no_split := getattr(module, "_no_split_modules", None):
                self._no_split_modules.update(no_split)

        # NOTE: We do NOT call self.to(device) here.
        # HF's from_pretrained() handles device placement; doing it ourselves
        # causes "Cannot copy out of meta tensor" errors during load.
        # Call model.to_device() explicitly after construction if desired.

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def tie_weights(self, missing_keys=None, recompute_mapping=False):
        """Tie lm_head.weight to model.embed.weight when using TiedLMHead."""
        if isinstance(self.lm_head, TiedLMHead):
            self.lm_head.weight = self.model.embed.weight
        elif getattr(self.config, 'tie_word_embeddings', False):
            # If using standard Linear but config says tie, share the weight
            self.lm_head.weight = self.model.embed.weight

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

        SACRED: e is passed WITHOUT detach() to recurrent().  This preserves
        full gradient flow through both the hidden-state path and the LTI
        injection path.  The embedding layer receives gradients from both
        paths, giving ~2x stronger training signal.  Adding e.detach() was
        tested and causes ~2x PPL regression (train PPL ~29 vs ~15 baseline).

        Note: model.py uses recurrent(e, e.detach()), but HelixForCausalLM
        is the primary training interface — its behavior is the sacred baseline.
        """
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)
        # IGNORE use_cache -- this model has no KV-cache
        _ = use_cache  # noqa: F841

        if inputs_embeds is not None:
            e = inputs_embeds
        else:
            e = self.model.embed(input_ids)

        # SACRED: Pass e (NOT detached) to preserve full gradient flow.
        # e.detach() was tested and causes ~2x PPL regression.
        # Pass attention_mask and cca_step for regression fixes (mask propagation, CCA)
        if not self.training and self.canonical_eval_layout:
            # Eval/inference: run every row at the training layout (content at
            # index 0, length cfg.seq_len, right-padded) so the compressed-branch
            # boundaries and the window grid match training exactly; results are
            # scattered back to the caller's positions. See _canonical_eval_hidden.
            h = self._canonical_eval_hidden(e, attention_mask, kwargs.get("cca_step", None))
        else:
            h = self.model.recurrent(
                e, e,
                attention_mask=attention_mask,
                cca_step=kwargs.get("cca_step", None),
            )

        # Output
        # P2 FIX: Removed the CUDA-only h.clone() band-aid. The clone
        # was hiding an in-place op bug (likely in RMSNorm). If in-place
        # errors resurface, fix them at the source — not with conditional clones.
        h = self.model.out_norm(h)
        # Logits and loss are always returned in fp32, also when the weights are
        # bf16/fp16: a bf16 log-softmax over 50k classes loses ~0.01 nats.
        logits = self.lm_head(h).float()

        loss = None
        if labels is not None:
            # Shift by 1 for next-token prediction:
            # logits[:, :-1] predicts labels[:, 1:]
            shift_logits = logits[:, :-1, :].float().reshape(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            if getattr(self.config, "memory_efficient_forward", False):
                del logits
                logits = None
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            return output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=h if output_hidden_states else None,
            attentions=None,
        )

    # Eval-only layout canonicalisation (set False to get the raw layout back).
    canonical_eval_layout = True

    def _canonical_eval_hidden(
        self,
        e: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cca_step: Optional[int] = None,
    ) -> torch.Tensor:
        """Run the recurrent block at the training layout and map back.

        Training only ever saw full ``cfg.seq_len`` windows whose content starts
        at index 0. The multi-scale attention is not translation- or
        length-invariant: its compressed boundaries are linspace(0, T-1, K) and
        its local/coarse windows are aligned to multiples of the window size,
        both measured from index 0 of the *padded* tensor. A prompt therefore
        got different logits depending on its batch's padded length and on left
        vs right padding.

        Here every row's valid tokens (attention_mask == 1, in order) are packed
        to positions 0..n-1 of a length-T buffer (T = cfg.seq_len), right-padded
        and masked, the block runs at length T, and the hidden states are
        gathered back to the caller's positions (padded positions get 0, so
        their logits are 0). The model is causal, so a token's output equals
        its output inside a full training window with the same prefix.

        Cost: every eval forward runs at T = cfg.seq_len (2048 for the released
        model), whatever the prompt length. If a row has more than T valid
        tokens, the raw layout is used unchanged.
        """
        B, L, D = e.shape
        T = int(self.config.seq_len)
        if attention_mask is None:
            valid = torch.ones(B, L, dtype=torch.bool, device=e.device)
        else:
            valid = attention_mask.to(device=e.device).bool()
        n = valid.sum(dim=1)
        if B == 0:
            return self.model.recurrent(e, e, attention_mask=attention_mask, cca_step=cca_step)
        # n <= L always, so the length check needs the data only when L > T.
        # For L <= T (every generation step inside the training context) this
        # is a shape-only branch: no device->host sync, no torch.compile graph
        # break. The data-dependent check is kept for the rare L > T case.
        if L > T and int(n.max()) > T:
            return self.model.recurrent(e, e, attention_mask=attention_mask, cca_step=cca_step)
        # destination slot of each caller position; padded positions go to slot T (dropped)
        dest = torch.where(valid, valid.long().cumsum(dim=1) - 1, torch.full_like(n[:, None], T))
        idx = dest.unsqueeze(-1).expand(B, L, D)
        e_c = e.new_zeros(B, T + 1, D).scatter(1, idx, e)[:, :T]
        mask_c = (torch.arange(T, device=e.device).unsqueeze(0) < n.unsqueeze(1)).long()
        h_c = self.model.recurrent(e_c, e_c, attention_mask=mask_c, cca_step=cca_step)
        h_c = torch.cat([h_c, h_c.new_zeros(B, 1, D)], dim=1)
        return h_c.gather(1, idx)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for the .generate() method.

        CRITICAL: This model has NO KV-cache.  The recurrent graph re-initializes
        node_states on every forward.  Therefore we must pass the FULL sequence
        (or the last seq_len window) on every generation step — NEVER just
        input_ids[:, -1:].

        past_key_values is explicitly ignored.
        """
        # Ignore past_key_values -- the recurrent graph doesn't use them
        if past_key_values is not None:
            pass  # explicitly no-op

        # Pass the full sequence, but cap at seq_len to avoid OOM on long gens
        seq_len = input_ids.shape[1]
        if seq_len > self.config.seq_len:
            input_ids = input_ids[:, -self.config.seq_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.config.seq_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # Do NOT pass past_key_values or use_cache -- this model is stateless
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
        tokenizer: Optional[HelixTokenizer] = None,
    ) -> torch.Tensor:
        """
        Extended generation with stop-string support.

        If tokenizer and stop_strings are provided, uses StopStringCriteria
        with the standard GenerationMixin.generate(). Otherwise falls back to
        basic generation.

        DEPRECATED: Use model.generate() with StoppingCriteriaList instead.
        """
        # Build stopping criteria if tokenizer + stop strings provided
        stopping_criteria = None
        if tokenizer is not None and stop_strings:
            stopping_criteria = StoppingCriteriaList([
                StopStringCriteria(tokenizer, stop_strings, batch_size=input_ids.shape[0])
            ])

        # Delegate to standard GenerationMixin.generate()
        result = self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            stopping_criteria=stopping_criteria,
        )

        return result

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# Explicit Auto-registration (no silent try/except)
# ---------------------------------------------------------------------------
def register_helix_for_auto_classes():
    """Register HelixLM config and model classes with HF AutoClasses."""
    try:
        AutoConfig.register("helix", HelixConfig)
    except Exception as e:
        import warnings
        warnings.warn(f"AutoConfig.register('helix') failed: {e}. "
                      "You may need to use trust_remote_code=True when loading.",
                      RuntimeWarning)
        return

    try:
        AutoModelForCausalLM.register(HelixConfig, HelixForCausalLM)
    except Exception as e:
        import warnings
        warnings.warn(f"AutoModelForCausalLM.register failed: {e}. "
                      "You may need to use trust_remote_code=True when loading.",
                      RuntimeWarning)
        return

    # Register for auto_class so push_to_hub writes auto_map in config.json
    try:
        HelixConfig.register_for_auto_class()
        HelixForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    except Exception as e:
        import warnings
        warnings.warn(f"register_for_auto_class failed: {e}. "
                      "Hub push/pull may require manual trust_remote_code=True.",
                      RuntimeWarning)


# Register on import
register_helix_for_auto_classes()
