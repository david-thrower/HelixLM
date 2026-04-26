"""
HelixLM Tokenizer Abstraction.

Supports multiple backends:
  - character:  character-level (for smoke tests, tiny models)
  - gpt2:       GPT-2 BPE via transformers
  - qwen:       Qwen3 family tokenizer via transformers
  - custom:     Any AutoTokenizer from HF

All backends expose a unified interface for encode / decode / vocab_size / pad_id / eos_id / bos_id.
"""
from typing import List, Optional, Union, Dict, Any
import torch


class HelixTokenizer:
    """
    Unified tokenizer wrapper for HelixLM.
    Automatically loads the correct backend based on tokenizer_name.
    """
    def __init__(self, tokenizer_name: str = "gpt2", **kwargs):
        self.tokenizer_name = tokenizer_name
        self._backend = None
        self._char_to_id: Optional[Dict[str, int]] = None
        self._id_to_char: Optional[Dict[int, str]] = None

        if tokenizer_name == "char":
            # Character-level: must call build_vocab(texts) before use
            pass
        elif tokenizer_name.startswith("gpt2") or tokenizer_name.startswith("openai"):
            from transformers import AutoTokenizer
            self._backend = AutoTokenizer.from_pretrained("gpt2", **kwargs)
            self._backend.pad_token = self._backend.eos_token
        elif tokenizer_name.startswith("qwen"):
            from transformers import AutoTokenizer
            self._backend = AutoTokenizer.from_pretrained(
                tokenizer_name if "/" in tokenizer_name else "Qwen/Qwen2.5-0.5B",
                trust_remote_code=True,
                **kwargs,
            )
            if self._backend.pad_token is None:
                self._backend.pad_token = self._backend.eos_token
        else:
            # Custom HF tokenizer
            from transformers import AutoTokenizer
            self._backend = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, **kwargs)
            if self._backend.pad_token is None:
                self._backend.pad_token = self._backend.eos_token

    # ------------------------------------------------------------------
    # Character-level vocab builder
    # ------------------------------------------------------------------
    def build_char_vocab(self, texts: Union[str, List[str]], special_tokens: Optional[List[str]] = None):
        """
        Build character vocabulary from text(s).
        Must be called before encode/decode when tokenizer_name == 'char'.
        """
        if isinstance(texts, str):
            texts = [texts]
        all_chars = set()
        for t in texts:
            all_chars.update(t)
        chars = sorted(all_chars)

        # Reserve 0 for pad, 1 for eos, 2 for bos
        offset = 3
        if special_tokens:
            for i, tok in enumerate(special_tokens):
                chars = [tok] + chars if tok not in chars else chars

        self._char_to_id = {c: i + offset for i, c in enumerate(chars)}
        self._id_to_char = {i + offset: c for i, c in enumerate(chars)}
        self._char_to_id["<pad>"] = 0
        self._char_to_id["<eos>"] = 1
        self._char_to_id["<bos>"] = 2
        self._id_to_char[0] = "<pad>"
        self._id_to_char[1] = "<eos>"
        self._id_to_char[2] = "<bos>"

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = False, **kwargs) -> List[int]:
        if self.tokenizer_name == "char":
            if self._char_to_id is None:
                raise RuntimeError("Call build_char_vocab() before encode()")
            ids = [self._char_to_id.get(c, 0) for c in text]
            if add_special_tokens:
                ids = [2] + ids + [1]
            return ids
        return self._backend.encode(text, add_special_tokens=add_special_tokens, **kwargs)

    def decode(self, ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True, **kwargs) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if self.tokenizer_name == "char":
            if self._id_to_char is None:
                raise RuntimeError("Call build_char_vocab() before decode()")
            return "".join(self._id_to_char.get(i, "") for i in ids if not (skip_special_tokens and i in (0, 1, 2)))
        return self._backend.decode(ids, skip_special_tokens=skip_special_tokens, **kwargs)

    def __call__(self, text: Union[str, List[str]], return_tensors: Optional[str] = None, padding: bool = False,
                 truncation: bool = False, max_length: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Batch tokenization returning dict with input_ids, attention_mask."""
        if self.tokenizer_name == "char":
            if isinstance(text, str):
                text = [text]
            input_ids = [self.encode(t) for t in text]
            max_len = max(len(ids) for ids in input_ids) if not max_length else max_length
            attention_mask = []
            padded_ids = []
            for ids in input_ids:
                if truncation and max_length and len(ids) > max_length:
                    ids = ids[:max_length]
                mask = [1] * len(ids) + [0] * (max_len - len(ids))
                ids = ids + [0] * (max_len - len(ids))
                padded_ids.append(ids)
                attention_mask.append(mask)
            result = {"input_ids": padded_ids, "attention_mask": attention_mask}
            if return_tensors == "pt":
                result["input_ids"] = torch.tensor(result["input_ids"], dtype=torch.long)
                result["attention_mask"] = torch.tensor(result["attention_mask"], dtype=torch.long)
            return result
        return self._backend(text, return_tensors=return_tensors, padding=padding, truncation=truncation,
                             max_length=max_length, **kwargs)

    def batch_encode(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """Alias for __call__ with list input."""
        return self(texts, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        if self.tokenizer_name == "char":
            return len(self._char_to_id) if self._char_to_id else 0
        return len(self._backend)

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def pad_token_id(self) -> int:
        if self.tokenizer_name == "char":
            return 0
        return self._backend.pad_token_id

    @property
    def eos_token_id(self) -> int:
        if self.tokenizer_name == "char":
            return 1
        return self._backend.eos_token_id

    @property
    def bos_token_id(self) -> int:
        if self.tokenizer_name == "char":
            return 2
        return getattr(self._backend, "bos_token_id", self.eos_token_id)

    @property
    def pad_token(self) -> str:
        if self.tokenizer_name == "char":
            return "<pad>"
        return self._backend.pad_token

    @property
    def eos_token(self) -> str:
        if self.tokenizer_name == "char":
            return "<eos>"
        return self._backend.eos_token

    @property
    def bos_token(self) -> str:
        if self.tokenizer_name == "char":
            return "<bos>"
        return getattr(self._backend, "bos_token", self.eos_token)

    @property
    def special_tokens_map(self) -> Dict[str, Any]:
        if self.tokenizer_name == "char":
            return {"pad_token": "<pad>", "eos_token": "<eos>", "bos_token": "<bos>"}
        return self._backend.special_tokens_map

    def apply_chat_template(self, messages: List[Dict[str, Any]], tokenize: bool = True,
                            add_generation_prompt: bool = False, return_dict: bool = False,
                            return_tensors: Optional[str] = None, **kwargs) -> Any:
        """
        Apply chat template to messages.
        Falls back to manual formatting for char tokenizer or missing template.
        """
        if self.tokenizer_name == "char":
            # Simple manual format
            formatted = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    content = " ".join(text_parts)
                formatted += f"[{role}]: {content}\n"
            if add_generation_prompt:
                formatted += "[assistant]: "
            if not tokenize:
                return formatted
            result = self(formatted, return_tensors=return_tensors)
            if return_dict:
                return result
            return result["input_ids"]

        if hasattr(self._backend, "apply_chat_template") and self._backend.chat_template is not None:
            return self._backend.apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
                return_dict=return_dict, return_tensors=return_tensors, **kwargs,
            )

        # Fallback: manual Qwen-style format
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                content = " ".join(text_parts)
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n"

        if not tokenize:
            return formatted
        result = self._backend(formatted, return_tensors=return_tensors, padding=True, truncation=True)
        if return_dict:
            return result
        return result["input_ids"]
