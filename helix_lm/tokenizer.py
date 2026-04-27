"""
Tokenizer wrapper for HelixLM.
"""
from typing import List, Optional
import torch


class CharTokenizer:
    """Character-level tokenizer for smoke tests."""
    def __init__(self):
        self.vocab = ["<pad>", "<eos>", "<unk>"] + [chr(i) for i in range(32, 127)]
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, self.unk_token_id) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, "<unk>") for i in ids)

    def __call__(self, text: str, return_tensors: Optional[str] = None):
        ids = self.encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}


def get_tokenizer(name: str = "gpt2"):
    """Get tokenizer by name."""
    if name == "char":
        return CharTokenizer()
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name)
    except Exception:
        return CharTokenizer()
