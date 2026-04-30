"""
HelixLM Dataset with rolling text chunking and natural stop detection.

Key features:
  - Rolling window for text > MAX_SEQ_LEN: overlapping chunks with stride
  - Padding for text < MAX_SEQ_LEN
  - Distinguishes natural stop (end of document) from EOS token
  - Supports both raw text and pre-tokenized inputs
  - Compatible with HF datasets streaming interface
  - Document-aware chunking: no cross-document boundaries, 100% token utilization
"""
import random
from typing import List, Optional, Iterator, Dict, Any, Union, Tuple

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader


class HelixDataset(IterableDataset):
    """
    Streaming dataset with rolling chunking for language model pretraining.

    Handles three scenarios:
      1. Text >> seq_len: rolling window with configurable stride
      2. Text == seq_len: exact match
      3. Text < seq_len: padding to seq_len (with attention mask)

    For each chunk, produces:
      - input_ids: (seq_len,)
      - labels: (seq_len,) — shifted by 1 for next-token prediction
      - attention_mask: (seq_len,) — 1 for real tokens, 0 for padding
      - is_natural_stop: scalar bool — True if chunk ends at document boundary
    """
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        seq_len: int = 2048,
        stride: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = True,
        add_eos: bool = True,
        natural_stop_threshold: float = 0.8,
    ):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or max(1, seq_len // 2)  # Default 50% overlap
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.add_eos = add_eos
        self.natural_stop_threshold = natural_stop_threshold

        # Pre-tokenize all texts
        self._tokenized_docs = self._tokenize_all()

    def _tokenize_all(self) -> List[Dict[str, Any]]:
        """Pre-tokenize all documents, storing per-document token lists."""
        docs = []
        for text in self.texts:
            if not text.strip():
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if self.add_eos and hasattr(self.tokenizer, 'eos_token_id'):
                ids.append(self.tokenizer.eos_token_id)
            docs.append({
                "ids": ids,
                "length": len(ids),
            })
        return docs

    def __len__(self) -> int:
        """Approximate number of samples (for progress bars)."""
        total = 0
        for doc in self._tokenized_docs:
            length = doc["length"]
            if length >= self.seq_len:
                total += (length - self.seq_len) // self.stride + 1
            else:
                total += 1
        return total

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield samples with rolling chunking."""
        worker_info = torch.utils.data.get_worker_info()
        docs = self._tokenized_docs

        if worker_info is not None:
            per_worker = len(docs) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(docs)
            docs = docs[start:end]

        if self.shuffle:
            random.shuffle(docs)

        for doc in docs:
            ids = doc["ids"]
            length = doc["length"]

            if length == 0:
                continue

            if length >= self.seq_len:
                # Rolling window: yield overlapping chunks
                for start_idx in range(0, length - self.seq_len + 1, self.stride):
                    end_idx = start_idx + self.seq_len
                    chunk = ids[start_idx:end_idx]
                    labels = list(chunk)   # unshifted; HF model handles causal shift
                    # Sliding-window masking: predictions for overlap tokens are
                    # evaluated in the previous window; mask them with -100.
                    if start_idx > 0 and self.stride < self.seq_len:
                        warmup_len = self.seq_len - self.stride
                        labels[:warmup_len] = [-100] * warmup_len

                    # Natural stop: is this the last chunk of this document?
                    is_natural_stop = end_idx >= length * self.natural_stop_threshold

                    yield self._make_sample(chunk, labels, is_natural_stop)

                    if end_idx >= length:
                        break
            else:
                # Short document: pad to seq_len
                chunk = ids[:length]
                pad_len = self.seq_len - length
                chunk = chunk + [self.tokenizer.pad_token_id] * pad_len
                labels = list(chunk)
                if pad_len > 0:
                    labels[-pad_len:] = [-100] * pad_len

                yield self._make_sample(chunk, labels, is_natural_stop=True)

    def _make_sample(self, chunk: List[int], labels: List[int], is_natural_stop: bool) -> Dict[str, torch.Tensor]:
        """Create a sample dictionary from chunk."""
        input_ids = torch.tensor(chunk[:self.seq_len], dtype=torch.long)
        labels_tensor = torch.tensor(labels[:self.seq_len], dtype=torch.long)
        # Convert any remaining pad tokens to -100 so CrossEntropyLoss ignores them
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

    This eliminates the "islands of good PPL in a sea of bad PPL" problem
    caused by document-boundary crossings in contiguous-stream chunking.

    Args:
        texts: List of document texts.
        tokenizer: Tokenizer with encode(), pad_token_id, eos_token_id.
        seq_len: Target sequence length for all chunks.
        min_tail_len: Minimum length for a tail chunk to be kept.
                      Tails shorter than this are dropped.
                      Default: seq_len // 4 (drop very short tails).
                      Set to 1 to keep all tails (e.g., for instruct data).
        add_eos: Whether to append EOS token to each document.
    """
    def __init__(self, texts, tokenizer, seq_len, min_tail_len=None, add_eos=True):
        self.seq_len = seq_len
        self.pad_id = getattr(tokenizer, "pad_token_id", 0)
        self.eos_id = getattr(tokenizer, "eos_token_id", self.pad_id)
        self.chunks = []

        if min_tail_len is None:
            min_tail_len = seq_len // 4

        dropped_short = 0
        dropped_tail = 0
        kept = 0

        for text in texts:
            text = text.strip()
            if not text:
                continue

            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < min_tail_len:
                dropped_short += 1
                continue

            if add_eos and self.eos_id is not None:
                ids.append(self.eos_id)

            n_full = len(ids) // seq_len
            remainder_len = len(ids) % seq_len

            for i in range(n_full):
                start = i * seq_len
                chunk = ids[start:start + seq_len]
                is_last_chunk = (i == n_full - 1)
                no_tail_kept = (remainder_len < min_tail_len)
                self.chunks.append((chunk, is_last_chunk and no_tail_kept))
                kept += 1

            if remainder_len >= min_tail_len:
                tail = ids[n_full * seq_len:]
                tail = tail + [self.pad_id] * (seq_len - remainder_len)
                self.chunks.append((tail, True))
                kept += 1
            elif remainder_len > 0:
                dropped_tail += 1

        print(f"DocumentAwareDataset: {kept} chunks "
              f"({dropped_short} short docs dropped, {dropped_tail} tails dropped)")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
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
        y = x.clone()   # unshifted for HF causal-LM shift

        return {
            "input_ids": x,
            "labels": y,
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "is_natural_stop": torch.tensor(end >= len(self.tokens) * 0.9, dtype=torch.bool),
        }


class HelixHFDataset(IterableDataset):
    """
    Wrapper for HuggingFace datasets with streaming support.
    Handles text > seq_len via rolling chunking.
    """
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        seq_len: int = 2048,
        text_column: str = "text",
        split: str = "train",
        streaming: bool = True,
        stride: Optional[int] = None,
        shuffle_buffer_size: int = 10000,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or max(1, seq_len // 2)
        self.text_column = text_column

        try:
            from datasets import load_dataset
            self.dataset = load_dataset(dataset_name, split=split, streaming=streaming, **kwargs)
            if streaming:
                self.dataset = self.dataset.shuffle(seed=42, buffer_size=shuffle_buffer_size)
        except ImportError:
            raise ImportError("Please install `datasets` library: pip install datasets")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer = []
        buffer_text = ""

        for example in self.dataset:
            text = example.get(self.text_column, "")
            if not text:
                continue

            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if hasattr(self.tokenizer, 'eos_token_id'):
                ids.append(self.tokenizer.eos_token_id)

            buffer.extend(ids)

            # Yield complete sequences from buffer
            while len(buffer) >= self.seq_len + 1:
                x = torch.tensor(buffer[:self.seq_len], dtype=torch.long)
                y = x.clone()   # unshifted; HF model handles causal shift
                yield {
                    "input_ids": x,
                    "labels": y,
                    "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                    "is_natural_stop": torch.tensor(False, dtype=torch.bool),
                }


def create_helix_dataloader(
    texts: List[str],
    tokenizer,
    seq_len: int = 2048,
    batch_size: int = 8,
    stride: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader from text list with rolling chunking.
    """
    dataset = HelixDataset(texts, tokenizer, seq_len, stride, shuffle, drop_last, **kwargs)

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
                      Use seq_len//4 for pretrain, 1 for instruct.
        add_eos: Whether to append EOS to documents.

    Returns:
        DataLoader yielding batches with input_ids, labels, attention_mask,
        and is_natural_stop tensors.
    """
    ds = DocumentAwareDataset(
        texts, tokenizer, seq_len,
        min_tail_len=min_tail_len, add_eos=add_eos,
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
