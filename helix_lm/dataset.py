"""
HelixLM Dataset with rolling text chunking and natural stop detection.

Key features:
  - Rolling window for text > MAX_SEQ_LEN: overlapping chunks with stride
  - Padding for text < MAX_SEQ_LEN
  - Distinguishes natural stop (end of document) from EOS token
  - Supports both raw text and pre-tokenized inputs
  - Compatible with HF datasets streaming interface
  - Document-aware chunking: no cross-document boundaries, 100% token utilization
  - Optional lazy loading: O(1) init, tokenize on-the-fly
"""
import random
from typing import List, Optional, Iterator, Dict, Any, Union, Tuple

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from tqdm import tqdm


class HelixDataset(Dataset):
    """
    Index-based Dataset with rolling chunking for language model pretraining.
    Compatible with DataLoader(shuffle=True) natively.

    Handles three scenarios:
      1. Text >> seq_len: rolling window with configurable stride
      2. Text == seq_len: exact match
      3. Text < seq_len: padding to seq_len (with attention mask)

    For each chunk, produces:
      - input_ids: (seq_len,)
      - labels: (seq_len,) — shifted by 1 for next-token prediction
      - attention_mask: (seq_len,) — 1 for real tokens, 0 for padding
      - is_natural_stop: scalar bool — True if chunk ends at document boundary

    Args:
        texts: List of document texts.
        tokenizer: Tokenizer instance.
        seq_len: Target sequence length.
        stride: Rolling window stride. Default: seq_len // 2.
        lazy: If True, store raw texts and tokenize on-the-fly in __getitem__.
              If False, eagerly tokenize all documents in __init__.
        add_eos: Whether to append EOS token to each document.
        natural_stop_threshold: Fraction of document length to consider as natural stop.
    """
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        seq_len: int = 2048,
        stride: Optional[int] = None,
        lazy: bool = True,
        add_eos: bool = True,
        natural_stop_threshold: float = 0.8,
    ):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or max(1, seq_len // 2)
        self.lazy = lazy
        self.add_eos = add_eos
        self.natural_stop_threshold = natural_stop_threshold

        if lazy:
            # O(1) init: just store texts
            self._tokenized_docs = None
            self._chunk_index = None
        else:
            # Eager tokenization with progress bar
            self._tokenized_docs = self._tokenize_all()
            self._chunk_index = self._build_chunk_index()

    def _tokenize_all(self) -> List[Dict[str, Any]]:
        """Pre-tokenize all documents, storing per-document token lists."""
        docs = []
        iterable = tqdm(self.texts, desc="Tokenizing", unit="doc", disable=len(self.texts) < 1000)
        for text in iterable:
            if not text.strip():
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if self.add_eos and hasattr(self.tokenizer, 'eos_token_id'):
                ids.append(self.tokenizer.eos_token_id)
            docs.append({"ids": ids, "length": len(ids)})
        return docs

    def _build_chunk_index(self) -> List[Tuple[int, int, bool]]:
        """Build (doc_idx, start_idx, is_natural_stop) for every chunk."""
        index = []
        for doc_idx, doc in enumerate(self._tokenized_docs):
            length = doc["length"]
            if length == 0:
                continue
            ids = doc["ids"]
            if length >= self.seq_len:
                for start_idx in range(0, length - self.seq_len + 1, self.stride):
                    end_idx = start_idx + self.seq_len
                    is_natural_stop = end_idx >= length * self.natural_stop_threshold
                    index.append((doc_idx, start_idx, is_natural_stop))
                    if end_idx >= length:
                        break
            else:
                index.append((doc_idx, 0, True))
        return index

    def _build_lazy_chunk_index(self) -> List[Tuple[int, int, int, bool]]:
        """Build chunk index from raw texts on-the-fly.
        Returns (doc_idx, start_idx, effective_len, is_natural_stop).
        """
        index = []
        for doc_idx, text in enumerate(self.texts):
            text = text.strip()
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if self.add_eos and hasattr(self.tokenizer, 'eos_token_id'):
                ids.append(self.tokenizer.eos_token_id)
            length = len(ids)
            if length == 0:
                continue
            if length >= self.seq_len:
                for start_idx in range(0, length - self.seq_len + 1, self.stride):
                    end_idx = start_idx + self.seq_len
                    is_natural_stop = end_idx >= length * self.natural_stop_threshold
                    index.append((doc_idx, start_idx, length, is_natural_stop))
                    if end_idx >= length:
                        break
            else:
                index.append((doc_idx, 0, length, True))
        return index

    def __len__(self) -> int:
        if self._chunk_index is not None:
            return len(self._chunk_index)
        # Lazy mode: build index on first len() call and cache it
        if not hasattr(self, '_lazy_index') or self._lazy_index is None:
            self._lazy_index = self._build_lazy_chunk_index()
        return len(self._lazy_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._chunk_index is not None:
            doc_idx, start_idx, is_natural_stop = self._chunk_index[idx]
            ids = self._tokenized_docs[doc_idx]["ids"]
            length = self._tokenized_docs[doc_idx]["length"]
        else:
            if not hasattr(self, '_lazy_index') or self._lazy_index is None:
                self._lazy_index = self._build_lazy_chunk_index()
            doc_idx, start_idx, length, is_natural_stop = self._lazy_index[idx]
            text = self.texts[doc_idx].strip()
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if self.add_eos and hasattr(self.tokenizer, 'eos_token_id'):
                ids.append(self.tokenizer.eos_token_id)

        if length >= self.seq_len:
            end_idx = start_idx + self.seq_len
            chunk = ids[start_idx:end_idx]
            labels = list(chunk)
            if start_idx > 0 and self.stride < self.seq_len:
                warmup_len = self.seq_len - self.stride
                labels[:warmup_len] = [-100] * warmup_len
            return self._make_sample(chunk, labels, is_natural_stop)
        else:
            chunk = ids[:length]
            pad_len = self.seq_len - length
            chunk = chunk + [self.tokenizer.pad_token_id] * pad_len
            labels = list(chunk)
            if pad_len > 0:
                labels[-pad_len:] = [-100] * pad_len
            return self._make_sample(chunk, labels, is_natural_stop=True)

    def _make_sample(self, chunk: List[int], labels: List[int], is_natural_stop: bool) -> Dict[str, torch.Tensor]:
        """Create a sample dictionary from chunk."""
        input_ids = torch.tensor(chunk[:self.seq_len], dtype=torch.long)
        labels_tensor = torch.tensor(labels[:self.seq_len], dtype=torch.long)
        labels_tensor = torch.where(
            labels_tensor == self.tokenizer.pad_token_id,
            torch.tensor(-100, dtype=torch.long),
            labels_tensor,
        )
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels_tensor,
            "attention_mask": attention_mask,
            "is_natural_stop": torch.tensor(is_natural_stop, dtype=torch.bool),
        }


class DocumentAwareDataset(Dataset):
    """
    Per-document chunking with no cross-document boundaries.

    For each document:
      - Long documents: split into non-overlapping seq_len chunks.
      - Short documents: kept as-is, padded to seq_len.
      - Final chunk of each document is marked is_natural_stop=True.
      - Only padding positions are masked in labels (-100).
      - No label masking for overlap regions (100% token utilization).

    Args:
        texts: List of document texts.
        tokenizer: Tokenizer with encode(), pad_token_id, eos_token_id.
        seq_len: Target sequence length for all chunks.
        min_tail_len: Minimum length for a tail chunk to be kept.
                      Default: seq_len // 4.
                      Set to 1 to keep all tails (e.g., for instruct data).
        add_eos: Whether to append EOS token to each document.
        lazy: If True, store raw texts and build chunks on-the-fly.
              If False, eagerly tokenize all documents in __init__.
    """
    def __init__(self, texts, tokenizer, seq_len, min_tail_len=None, add_eos=True, lazy=True):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.pad_id = getattr(tokenizer, "pad_token_id", 0)
        self.eos_id = getattr(tokenizer, "eos_token_id", self.pad_id)
        self.lazy = lazy

        if min_tail_len is None:
            min_tail_len = seq_len // 4
        self.min_tail_len = min_tail_len
        self.add_eos = add_eos

        if lazy:
            self.texts = texts
            self.chunks = None
            self._stats = None
        else:
            self.texts = None
            self.chunks, self._stats = self._build_chunks(texts)

    def _build_chunks(self, texts):
        """Eagerly build chunks from texts. Returns (chunks, stats)."""
        chunks = []
        dropped_short = 0
        dropped_tail = 0
        kept = 0

        iterable = tqdm(texts, desc="Chunking", unit="doc", disable=len(texts) < 1000)
        for text in iterable:
            text = text.strip()
            if not text:
                continue

            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < self.min_tail_len:
                dropped_short += 1
                continue

            if self.add_eos and self.eos_id is not None:
                ids.append(self.eos_id)

            n_full = len(ids) // self.seq_len
            remainder_len = len(ids) % self.seq_len

            for i in range(n_full):
                start = i * self.seq_len
                chunk = ids[start:start + self.seq_len]
                is_last_chunk = (i == n_full - 1)
                no_tail_kept = (remainder_len < self.min_tail_len)
                chunks.append((chunk, is_last_chunk and no_tail_kept))
                kept += 1

            if remainder_len >= self.min_tail_len:
                tail = ids[n_full * self.seq_len:]
                tail = tail + [self.pad_id] * (self.seq_len - remainder_len)
                chunks.append((tail, True))
                kept += 1
            elif remainder_len > 0:
                dropped_tail += 1

        stats = {
            "kept": kept,
            "dropped_short": dropped_short,
            "dropped_tail": dropped_tail,
        }
        return chunks, stats

    def _ensure_chunks(self):
        """Lazy build chunks on first access."""
        if self.chunks is None:
            self.chunks, self._stats = self._build_chunks(self.texts)
            self.texts = None  # free memory

    def __len__(self):
        self._ensure_chunks()
        return len(self.chunks)

    def __getitem__(self, idx):
        self._ensure_chunks()
        chunk, is_natural = self.chunks[idx]
        x = torch.tensor(chunk, dtype=torch.long)
        labels = x.clone()
        labels[x == self.pad_id] = -100
        return {
            "input_ids": x,
            "labels": labels,
            "attention_mask": (x != self.pad_id).long(),
            "is_natural_stop": torch.tensor(is_natural, dtype=torch.bool),
        }

    def get_stats(self):
        self._ensure_chunks()
        return self._stats


class HelixDatasetFromTokens(Dataset):
    """
    Dataset from pre-tokenized token stream (e.g., from HF datasets).
    Handles rolling chunking over a long token sequence.
    """
    def __init__(
        self,
        tokens: Union[List[int], torch.Tensor],
        seq_len: int = 2048,
        stride: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride or max(1, seq_len // 2)

        # Pre-compute indices for all valid chunks
        n = len(self.tokens)
        self.indices = list(range(0, max(1, n - seq_len), self.stride))
        if n >= seq_len and (n - seq_len) % self.stride != 0:
            self.indices.append(n - seq_len)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = x.clone()
        return {
            "input_ids": x,
            "labels": y,
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "is_natural_stop": torch.tensor(end >= len(self.tokens) * 0.9, dtype=torch.bool),
        }


class HelixHFDataset(Dataset):
    """
    Wrapper for HuggingFace datasets with streaming and non-streaming support.
    Handles text > seq_len via rolling chunking. Works with both Dataset
    and IterableDataset sources.

    Args:
        hf_dataset: A HuggingFace datasets.Dataset or IterableDataset,
                    or a dataset name string.
        tokenizer: Tokenizer instance.
        seq_len: Target sequence length.
        text_column: Column name containing text.
        stride: Rolling window stride.
        max_samples: Maximum number of samples to use. For streaming datasets,
                     uses .take(); for map datasets, uses .select().
        lazy: Whether to build chunks lazily.
        **kwargs: Additional arguments passed to load_dataset if hf_dataset is a string.
    """
    def __init__(
        self,
        hf_dataset: Union[str, Any],
        tokenizer,
        seq_len: int = 2048,
        text_column: str = "text",
        stride: Optional[int] = None,
        max_samples: Optional[int] = None,
        lazy: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or max(1, seq_len // 2)
        self.text_column = text_column
        self.lazy = lazy
        self.max_samples = max_samples

        if isinstance(hf_dataset, str):
            from datasets import load_dataset
            dataset_name = hf_dataset
            # Only pass trust_remote_code for known script-based datasets
            known_script_datasets = {"openai_humaneval", "bigcode/the-stack", "bigcode/the-stack-v2"}
            load_kwargs = dict(kwargs)
            if dataset_name not in known_script_datasets:
                load_kwargs.pop("trust_remote_code", None)
            self.dataset = load_dataset(dataset_name, **load_kwargs)
        else:
            self.dataset = hf_dataset

        # Handle DatasetDict: use "train" split by default
        if hasattr(self.dataset, "keys") and hasattr(self.dataset, "__getitem__"):
            if "train" in self.dataset:
                self.dataset = self.dataset["train"]
            else:
                # Pick first available split
                self.dataset = self.dataset[list(self.dataset.keys())[0]]

        # Apply max_samples
        if max_samples is not None:
            if hasattr(self.dataset, "take") and hasattr(self.dataset, "__iter__"):
                # IterableDataset
                self.dataset = self.dataset.take(max_samples)
            elif hasattr(self.dataset, "select"):
                # Map Dataset
                indices = list(range(min(max_samples, len(self.dataset))))
                self.dataset = self.dataset.select(indices)

        # Extract texts
        if hasattr(self.dataset, "__iter__") and not hasattr(self.dataset, "__getitem__"):
            # IterableDataset: consume to list (streaming)
            self._texts = []
            iterable = tqdm(self.dataset, desc="Loading HF dataset", unit="sample",
                            total=max_samples, disable=max_samples is not None and max_samples < 1000)
            for example in iterable:
                text = example.get(self.text_column, "")
                if text:
                    self._texts.append(text)
                if max_samples is not None and len(self._texts) >= max_samples:
                    break
            self._dataset_type = "list"
        else:
            # Map Dataset or list
            if hasattr(self.dataset, "__getitem__") and hasattr(self.dataset, "__len__"):
                iterable = tqdm(range(len(self.dataset)), desc="Loading HF dataset", unit="sample",
                                disable=len(self.dataset) < 1000)
                self._texts = [self.dataset[i].get(self.text_column, "") for i in iterable]
                self._texts = [t for t in self._texts if t]
                self._dataset_type = "map"
            else:
                self._texts = list(self.dataset)
                self._dataset_type = "list"

        # Wrap with DocumentAwareDataset for chunking
        self._doc_dataset = DocumentAwareDataset(
            self._texts, tokenizer, seq_len,
            min_tail_len=seq_len // 4, add_eos=True, lazy=lazy,
        )

    def __len__(self) -> int:
        return len(self._doc_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._doc_dataset[idx]

    def get_stats(self):
        return self._doc_dataset.get_stats()


def create_helix_dataloader(
    texts: List[str],
    tokenizer,
    seq_len: int = 2048,
    batch_size: int = 8,
    stride: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    lazy: bool = True,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader from text list with rolling chunking.
    """
    dataset = HelixDataset(texts, tokenizer, seq_len, stride, lazy=lazy, **kwargs)

    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        is_natural_stop = torch.stack([b["is_natural_stop"] for b in batch])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "is_natural_stop": is_natural_stop,
        }

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def create_document_loader(
    texts: List[str],
    tokenizer,
    seq_len: int = 2048,
    batch_size: int = 8,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    min_tail_len: Optional[int] = None,
    add_eos: bool = True,
    lazy: bool = True,
) -> DataLoader:
    """
    Create a DataLoader using DocumentAwareDataset (no boundary crossings).

    Args:
        texts: List of document texts.
        tokenizer: Tokenizer instance.
        seq_len: Sequence length for chunks.
        batch_size: Batch size.
        shuffle: Whether to shuffle chunks.
        drop_last: Whether to drop last incomplete batch.
        num_workers: DataLoader workers.
        min_tail_len: Minimum tail length to keep. Default seq_len//4.
        add_eos: Whether to append EOS to documents.
        lazy: If True, tokenize on-the-fly. If False, eagerly tokenize.

    Returns:
        DataLoader yielding batches with input_ids, labels, attention_mask,
        and is_natural_stop tensors.
    """
    ds = DocumentAwareDataset(
        texts, tokenizer, seq_len,
        min_tail_len=min_tail_len, add_eos=add_eos, lazy=lazy,
    )

    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        is_natural_stop = torch.stack([b["is_natural_stop"] for b in batch])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "is_natural_stop": is_natural_stop,
        }

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last,
    )
