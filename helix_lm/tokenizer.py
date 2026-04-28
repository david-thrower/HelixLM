"""
HelixLM Tokenizer — multi-backend wrapper with HuggingFace integration.

Supports GPT-2 BPE, Qwen, character-level, and any HuggingFace tokenizer.
Provides a unified interface so the rest of the codebase doesn't need to
care which backend is in use.
"""
from typing import List, Optional, Union
import torch


class CharTokenizer:
    """Character-level tokenizer for smoke tests and tiny models."""

    def __init__(self):
        self.vocab = ["<pad>", "<eos>", "<unk>"] + [chr(i) for i in range(32, 127)]
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.bos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [self.stoi.get(c, self.unk_token_id) for c in text]

    def decode(self, ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = False) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if skip_special_tokens:
            special = {self.pad_token_id, self.eos_token_id, self.unk_token_id}
            ids = [i for i in ids if i not in special]
        return "".join(self.itos.get(i, "<unk>") for i in ids)

    def __call__(self, text: str, return_tensors: Optional[str] = None, add_special_tokens: bool = False):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def __len__(self):
        return self.vocab_size

    def build_char_vocab(self, text: str):
        """(Re)build vocabulary from a corpus — call before training."""
        chars = sorted(set(text))
        self.vocab = ["<pad>", "<eos>", "<unk>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)


class HelixTokenizer:
    """
    Unified tokenizer wrapper for HelixLM.

    Backends:
      - ``"char"``        : Character-level (tiny models / debugging)
      - ``"gpt2"``        : GPT-2 BPE  (English, general)
      - ``"qwen"``        : Qwen3 family (multilingual, code)
      - any HF model name : Auto-loaded from HuggingFace Hub

    The wrapper provides a uniform API regardless of backend:
      ``encode``, ``decode``, ``__call__``, ``__len__``,
      ``vocab_size``, ``pad_token_id``, ``eos_token_id``,
      ``apply_chat_template`` (if supported).

    Args:
        name: Backend identifier (see above).
        max_length: If > 0, passed through to the HF backend as
                    ``model_max_length``.
    """

    def __init__(self, name: str = "gpt2", max_length: int = 0):
        self._backend_name = name
        self._tokenizer = None
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = None
        self.vocab_size = 0

        if name == "char":
            self._backend = CharTokenizer()
            self.vocab_size = self._backend.vocab_size
            self.pad_token_id = self._backend.pad_token_id
            self.eos_token_id = self._backend.eos_token_id
            self.bos_token_id = getattr(self._backend, "bos_token_id", None)
        else:
            from transformers import AutoTokenizer

            # Resolve shortcuts
            model_id = name
            if name == "gpt2":
                model_id = "gpt2"
            elif name == "qwen":
                model_id = "Qwen/Qwen3-0.6B"  # smallest Qwen3 for tokenizer

            self._backend = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                model_max_length=max_length if max_length > 0 else None,
            )

            # Ensure pad token exists
            if self._backend.pad_token is None:
                if self._backend.eos_token is not None:
                    self._backend.pad_token = self._backend.eos_token
                    self._backend.pad_token_id = self._backend.eos_token_id
                else:
                    self._backend.add_special_tokens({"pad_token": "[PAD]"})

            self.vocab_size = len(self._backend)
            self.pad_token_id = self._backend.pad_token_id
            self.eos_token_id = self._backend.eos_token_id
            self.bos_token_id = getattr(self._backend, "bos_token_id", None)

    # ------------------------------------------------------------------
    # Passthrough API
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if self._backend_name == "char":
            return self._backend.encode(text, add_special_tokens=add_special_tokens)
        return self._backend.encode(text, add_special_tokens=add_special_tokens)

    def decode(
        self,
        ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = False,
    ) -> str:
        if self._backend_name == "char":
            return self._backend.decode(ids, skip_special_tokens=skip_special_tokens)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._backend.decode(ids, skip_special_tokens=skip_special_tokens)

    def __call__(self, text: str, return_tensors: Optional[str] = None, add_special_tokens: bool = False):
        if self._backend_name == "char":
            return self._backend(text, return_tensors=return_tensors, add_special_tokens=add_special_tokens)
        return self._backend(text, return_tensors=return_tensors, add_special_tokens=add_special_tokens)

    def __len__(self) -> int:
        if self._backend_name == "char":
            return len(self._backend)
        return len(self._backend)

    # ------------------------------------------------------------------
    # HF-specific passthroughs
    # ------------------------------------------------------------------

    @property
    def backend(self):
        """Access the underlying HF tokenizer (for save_pretrained etc.)."""
        return self._backend

    def apply_chat_template(self, messages, tokenize: bool = True, add_generation_prompt: bool = False, return_tensors: Optional[str] = None):
        """Apply chat template if the backend supports it."""
        if hasattr(self._backend, "apply_chat_template"):
            kwargs = {"tokenize": tokenize, "add_generation_prompt": add_generation_prompt}
            if return_tensors:
                kwargs["return_tensors"] = return_tensors
            return self._backend.apply_chat_template(messages, **kwargs)
        raise RuntimeError(f"Backend '{self._backend_name}' does not support chat templates")

    def save_pretrained(self, save_directory: str):
        """Save the underlying tokenizer in HuggingFace format."""
        if self._backend_name == "char":
            import json
            import os
            os.makedirs(save_directory, exist_ok=True)
            data = {
                "vocab": self._backend.vocab,
                "stoi": self._backend.stoi,
                "itos": {int(k): v for k, v in self._backend.itos.items()},
                "pad_token_id": self._backend.pad_token_id,
                "eos_token_id": self._backend.eos_token_id,
                "unk_token_id": self._backend.unk_token_id,
            }
            with open(os.path.join(save_directory, "char_tokenizer.json"), "w") as f:
                json.dump(data, f)
        else:
            self._backend.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        """Load a previously saved tokenizer."""
        import os
        char_path = os.path.join(save_directory, "char_tokenizer.json")
        if os.path.exists(char_path):
            tok = cls("char")
            import json
            with open(char_path) as f:
                data = json.load(f)
            tok._backend.vocab = data["vocab"]
            tok._backend.stoi = data["stoi"]
            tok._backend.itos = {int(k): v for k, v in data["itos"].items()}
            tok._backend.vocab_size = len(tok._backend.vocab)
            return tok
        return cls(save_directory)

    def build_char_vocab(self, text: str):
        """(Re)build character vocabulary — only valid for ``'char'`` backend."""
        if self._backend_name == "char":
            self._backend.build_char_vocab(text)
            self.vocab_size = self._backend.vocab_size
            self.pad_token_id = self._backend.pad_token_id
            self.eos_token_id = self._backend.eos_token_id
        else:
            raise RuntimeError("build_char_vocab is only valid for 'char' backend")
