"""
HelixLM Dataset with rolling text chunking and natural stop detection.

REMEDIATION (2026-08-22) — Issue 4:
  Attention mask and loss mask are now SEPARATE across all four sample paths:
    1. HelixDataset._make_sample
    2. DocumentAwareDataset.__getitem__
    3. HelixPrechunkedDataset.__getitem__
    4. HelixShardedDataset._item_from_chunk

  attention_mask: 1 for every REAL token (including overlap warmup), 0 only for
                 exact trailing padding.
  labels:         -100 for overlap warmup AND exact trailing padding; token id
                 otherwise.

  Previously, attention_mask was derived from (labels != -100), which made
  overlap tokens invisible to attention even though they are legitimate context.
  Overlap tokens now remain visible to attention while still excluded from loss.

Key fixes retained from prior revision
  * DocumentAwareDataset now tracks exact pad_len in every chunk tuple.
    It NEVER scans backwards for pad_token_id, so GPT-2 (pad_id == eos_id)
    cannot accidentally mask a real EOS.
  * Optional within-document overlap (stride) added to DocumentAwareDataset.
    stride == seq_len  -> non-overlapping (default).
    stride <  seq_len  -> overlapping windows; overlap is masked in labels.
  * No cross-document boundaries are ever crossed.

Compatible with both eager and lazy loading.
"""
import random
import threading
from typing import List, Optional, Iterable, Iterator, Dict, Any, Union, Tuple
from collections import OrderedDict

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from tqdm import tqdm


def _collate_batch(batch):
    """Module-level collate function for pickling with multiprocessing."""
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


def _build_attention_mask(seq_len: int, pad_len: int) -> torch.Tensor:
    """
    Issue 4: attention_mask marks every REAL token as visible (1), including
    overlap warmup tokens, and only exact trailing padding as 0.
    """
    am = torch.ones(seq_len, dtype=torch.long)
    if pad_len > 0:
        am[-pad_len:] = 0
    return am


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
      - attention_mask: (seq_len,) — 1 for real tokens (incl. overlap), 0 for padding
      - is_natural_stop: scalar bool — True if chunk ends at document boundary
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
            self._tokenized_docs = None
            self._chunk_index = None
        else:
            self._tokenized_docs = self._tokenize_all()
            self._chunk_index = self._build_chunk_index()

    def _tokenize_all(self) -> List[Dict[str, Any]]:
        docs = []
        iterable = tqdm(self.texts, desc="Tokenizing", unit="doc", disable=len(self.texts) < 1000)
        for text in iterable:
            if not text.strip():
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if self.add_eos and hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                ids.append(self.tokenizer.eos_token_id)
            docs.append({"ids": ids, "length": len(ids)})
        return docs

    def _build_chunk_index(self) -> List[Tuple[int, int, bool]]:
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
        index = []
        for doc_idx, text in enumerate(self.texts):
            text = text.strip()
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if self.add_eos and hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
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
            if self.add_eos and hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                ids.append(self.tokenizer.eos_token_id)

        if length >= self.seq_len:
            end_idx = start_idx + self.seq_len
            chunk = ids[start_idx:end_idx]
            labels = list(chunk)
            overlap_mask = 0
            if start_idx > 0 and self.stride < self.seq_len:
                warmup_len = self.seq_len - self.stride
                labels[:warmup_len] = [-100] * warmup_len
                overlap_mask = warmup_len
            # Issue 4: pad_len=0 for full chunks; overlap stays visible to attention.
            return self._make_sample(chunk, labels, is_natural_stop,
                                     pad_len=0, overlap_mask=overlap_mask)
        else:
            chunk = ids[:length]
            pad_len = self.seq_len - length
            chunk = chunk + [self.tokenizer.pad_token_id] * pad_len
            labels = list(chunk)
            if pad_len > 0:
                labels[-pad_len:] = [-100] * pad_len
            return self._make_sample(chunk, labels, is_natural_stop=True,
                                     pad_len=pad_len, overlap_mask=0)

    def _make_sample(self, chunk, labels, is_natural_stop, pad_len=0, overlap_mask=0):
        input_ids = torch.tensor(chunk[:self.seq_len], dtype=torch.long)
        labels_t = torch.tensor(labels[:self.seq_len], dtype=torch.long)
        # Issue 4: attention_mask is derived from pad_len, NOT from labels.
        # Overlap warmup tokens are real context -> visible to attention (1),
        # but excluded from loss via labels=-100.
        attention_mask = _build_attention_mask(self.seq_len, pad_len)
        return {
            "input_ids": input_ids,
            "labels": labels_t,
            "attention_mask": attention_mask,
            "is_natural_stop": torch.tensor(is_natural_stop, dtype=torch.bool),
        }


class DocumentAwareDataset(Dataset):
    """
    Per-document chunking with no cross-document boundaries.

    Chunk tuple format (built in _build_chunks):
        (token_ids, is_natural_stop, pad_len, overlap_mask_len)

      * pad_len: exact number of padded positions at the TAIL.
                 Used to set labels[-pad_len:] = -100.
                 Always 0 for full chunks.
      * overlap_mask_len: number of positions at the HEAD to mask with -100.
                 Used when stride < seq_len so overlapping tokens are not
                 double-counted in loss. Always 0 when stride == seq_len.

    For each document:
      - Long documents: split into seq_len chunks (optionally overlapping).
      - Short documents: kept as-is, padded to seq_len.
      - Only padding positions are masked in labels (-100).
      - No label masking for overlap regions except the explicit overlap head.

    Issue 4: attention_mask now marks overlap tokens as visible (1) and only
    exact trailing padding as 0. Previously it was (labels != -100), which hid
    overlap context from attention.
    """
    def __init__(
        self,
        texts,
        tokenizer,
        seq_len,
        min_tail_len=None,
        add_eos=True,
        lazy=True,
        stride: Optional[int] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # Robust pad_id fallback: if unset, fall back to eos_id, then 0.
        self.pad_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_id is None:
            self.pad_id = getattr(tokenizer, "eos_token_id", 0)
        self.eos_id = getattr(tokenizer, "eos_token_id", None)

        self.lazy = lazy
        self.stride = stride if stride is not None else seq_len
        if not (1 <= self.stride <= self.seq_len):
            raise ValueError(f"stride must be in [1, seq_len], got {self.stride}")

        if min_tail_len is None:
            min_tail_len = 1  # Keep all documents; was seq_len//4 which dropped most docs at 512
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
        """
        Build chunk tuples:
          (token_ids: List[int], is_natural: bool, pad_len: int, overlap_mask: int)
        """
        chunks = []
        dropped_short = dropped_tail = kept = 0
        stride = self.stride

        for text in tqdm(texts, desc="Chunking", unit="doc", disable=len(texts) < 1000):
            text = text.strip()
            if not text:
                continue

            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < self.min_tail_len:
                dropped_short += 1
                continue

            if self.add_eos and self.eos_id is not None:
                ids.append(self.eos_id)

            length = len(ids)

            if length >= self.seq_len:
                # Sliding-window chunks fully inside the document
                starts = list(range(0, length - self.seq_len + 1, stride))
                for i, start in enumerate(starts):
                    chunk = ids[start:start + self.seq_len]
                    is_last_start = (i == len(starts) - 1)
                    reaches_end = (start + self.seq_len == length)

                    # Natural stop logic: if this is the last chunk we will emit
                    # and there is no tail (or tail is too short), mark it.
                    if not is_last_start:
                        is_natural = False
                    else:
                        tail_len = length - (start + self.seq_len)
                        has_tail = tail_len >= self.min_tail_len
                        is_natural = reaches_end or (not has_tail)

                    overlap_mask = 0
                    if start > 0 and stride < self.seq_len:
                        overlap_mask = self.seq_len - stride

                    chunks.append((chunk, is_natural, 0, overlap_mask))
                    kept += 1

                # Tail: tokens after the last sliding chunk
                last_covered_end = starts[-1] + self.seq_len if starts else 0
                remainder_len = length - last_covered_end

                if remainder_len >= self.min_tail_len:
                    tail = ids[last_covered_end:]
                    pad_len = self.seq_len - remainder_len
                    tail = tail + [self.pad_id] * pad_len
                    chunks.append((tail, True, pad_len, 0))
                    kept += 1
                elif remainder_len > 0:
                    # Tail too short to keep: last sliding chunk becomes the doc end
                    if starts:
                        last_chunk, _, last_pad, last_overlap = chunks[-1]
                        chunks[-1] = (last_chunk, True, last_pad, last_overlap)
                    dropped_tail += 1

            else:
                # Short document: pad once, everything is a natural stop
                pad_len = self.seq_len - length
                chunk = ids + [self.pad_id] * pad_len
                chunks.append((chunk, True, pad_len, 0))
                kept += 1

        stats = {
            "kept": kept,
            "dropped_short": dropped_short,
            "dropped_tail": dropped_tail,
        }
        return chunks, stats

    def _ensure_chunks(self):
        if self.chunks is None:
            self.chunks, self._stats = self._build_chunks(self.texts)
            self.texts = None  # free memory

    def __len__(self):
        self._ensure_chunks()
        return len(self.chunks)

    def __getitem__(self, idx):
        self._ensure_chunks()
        chunk, is_natural, pad_len, overlap_mask = self.chunks[idx]

        x = torch.tensor(chunk, dtype=torch.long)
        labels = x.clone()

        # 1. Mask overlapping head (only when stride < seq_len) — loss mask only
        if overlap_mask > 0:
            labels[:overlap_mask] = -100

        # 2. Mask exact trailing padding (robust to pad_id == eos_id) — loss mask only
        if pad_len > 0:
            labels[-pad_len:] = -100

        # Issue 4: attention_mask marks real tokens (incl. overlap) as visible.
        # Only exact trailing padding is masked from attention.
        attention_mask = _build_attention_mask(self.seq_len, pad_len)
        return {
            "input_ids": x,
            "labels": labels,
            "attention_mask": attention_mask,
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

##########

class ContinuousWindowDataset(IterableDataset):
    """
    Yields fixed-length seq_len windows from a continuous token stream.
    Documents are concatenated with the tokenizer's eos_token.
    No padding, no overlap, no label masking.

    Args:
        texts: Iterable of raw strings (List[str], Column, IterableColumn, etc.)
        tokenizer: HelixTokenizer instance.
        seq_len: Sequence length of the model.
        buffer_size: Size of the shuffle buffer (number of windows). Larger = better shuffle, more RAM.
        seed: Seed for the shuffle buffer.
        shuffle: If True, apply a shuffle buffer before yielding samples (for training). If False, yield in deterministic order (for validation).
    """
    def __init__(self, texts, tokenizer, seq_len, buffer_size=50000, seed=42, shuffle=True):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.buffer_size = buffer_size
        self.seed = seed
        self.shuffle = shuffle

    def _token_stream(self):
        for text in self.texts:
            if not text.strip():
                continue
            ids = self.tokenizer.encode(text.strip(), add_special_tokens=False)
            yield from ids
            if self.eos_token_id is not None:
                yield self.eos_token_id

    def _windowed_stream(self, shard: int = 0, num_shards: int = 1):
        # Window w goes to shard w % num_shards. Every worker reads the whole
        # token stream (the window boundaries depend on all earlier documents)
        # but materializes only its own windows, so N workers yield exactly the
        # windows that 0 workers yield, each once.
        buf = []
        window_index = 0
        for token_id in self._token_stream():
            buf.append(token_id)
            if len(buf) == self.seq_len:
                if window_index % num_shards == shard:
                    chunk = torch.tensor(buf, dtype=torch.long)
                    yield {
                        "input_ids": chunk,
                        "labels": chunk.clone(),
                        "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                    }
                window_index += 1
                buf = []
        # Drop incomplete tail

    def __iter__(self):
        # DataLoader workers each get a copy of this dataset. Without sharding,
        # num_workers=N yields every window N times.
        worker = torch.utils.data.get_worker_info()
        shard, num_shards = (worker.id, worker.num_workers) if worker is not None else (0, 1)
        if not self.shuffle:
            yield from self._windowed_stream(shard, num_shards)
            return

        rng = random.Random(self.seed + shard)  # shard 0 keeps the single-process order
        buf = []
        for sample in self._windowed_stream(shard, num_shards):
            if len(buf) < self.buffer_size:
                buf.append(sample)
            else:
                idx = rng.randint(0, len(buf) - 1)
                yield buf[idx]
                buf[idx] = sample
        rng.shuffle(buf)
        for sample in buf:
            yield sample


def collate_continuous(batch):
    """Stack dicts of tensors for continuous-window batches."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }

##########


class HelixHFDataset(Dataset):
    """
    Wrapper for HuggingFace datasets with streaming and non-streaming support.
    Uses DocumentAwareDataset internally, so no cross-document boundaries.
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
        self.text_column = text_column
        self.lazy = lazy
        self.max_samples = max_samples

        if isinstance(hf_dataset, str):
            from datasets import load_dataset
            dataset_name = hf_dataset
            known_script_datasets = {"openai_humaneval", "bigcode/the-stack", "bigcode/the-stack-v2"}
            load_kwargs = dict(kwargs)
            if dataset_name not in known_script_datasets:
                load_kwargs.pop("trust_remote_code", None)
            self.dataset = load_dataset(dataset_name, **load_kwargs)
        else:
            self.dataset = hf_dataset

        if hasattr(self.dataset, "keys") and hasattr(self.dataset, "__getitem__"):
            if "train" in self.dataset:
                self.dataset = self.dataset["train"]
            else:
                self.dataset = self.dataset[list(self.dataset.keys())[0]]

        if max_samples is not None:
            if hasattr(self.dataset, "take") and hasattr(self.dataset, "__iter__"):
                self.dataset = self.dataset.take(max_samples)
            elif hasattr(self.dataset, "select"):
                indices = list(range(min(max_samples, len(self.dataset))))
                self.dataset = self.dataset.select(indices)

        # Extract texts
        if hasattr(self.dataset, "__iter__") and not hasattr(self.dataset, "__getitem__"):
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
            if hasattr(self.dataset, "__getitem__") and hasattr(self.dataset, "__len__"):
                iterable = tqdm(range(len(self.dataset)), desc="Loading HF dataset", unit="sample",
                                disable=len(self.dataset) < 1000)
                self._texts = [self.dataset[i].get(self.text_column, "") for i in iterable]
                self._texts = [t for t in self._texts if t]
                self._dataset_type = "map"
            else:
                self._texts = list(self.dataset)
                self._dataset_type = "list"

        self._doc_dataset = DocumentAwareDataset(
            self._texts, tokenizer, seq_len,
            min_tail_len=1, add_eos=True, lazy=lazy, stride=stride,
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
    dataset = HelixDataset(texts, tokenizer, seq_len, stride, lazy=lazy, **kwargs)

    # Use module-level collate_fn for pickling with multiprocessing
    collate_fn = _collate_batch

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
    stride: Optional[int] = None,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader using DocumentAwareDataset (no boundary crossings).

    Args:
        stride: If < seq_len, enables within-document overlap (default: seq_len).
                This restores more optimizer steps per epoch without ever
                crossing document boundaries.
        seed: RNG seed for shuffling (uses torch.Generator for determinism).
    """
    ds = DocumentAwareDataset(
        texts, tokenizer, seq_len,
        min_tail_len=min_tail_len, add_eos=add_eos, lazy=lazy, stride=stride,
    )

    # Use module-level collate_fn for pickling with multiprocessing
    collate_fn = _collate_batch

    # Use torch.Generator for deterministic shuffling with given seed
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=drop_last,
            generator=generator,
        )
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=drop_last,
        )


# =============================================================================
# Streaming Dataset Support (New API for 87M/3B Token Scale-Up)
# =============================================================================

def _is_iterable_column(data) -> bool:
    """
    Detect if data should use streaming path (vs materialized List[str]).
    
    Streaming data:
        - Has __iter__ (can be iterated)
        - Does NOT have working __len__ (len() raises or doesn't exist)
    
    Materialized data:
        - Has working __len__ (len() returns a number)
    
    Examples:
        - List[str]: materialized (has len)
        - IterableColumn: streaming (no len, has iter)
        - Generator: streaming (no len, has iter)
    """
    # Must be iterable
    if not hasattr(data, '__iter__'):
        return False
    
    # Check if len() works
    if hasattr(data, '__len__'):
        try:
            len(data)  # If this works, it's materialized
            return False
        except (TypeError, NotImplementedError):
            pass  # No working len, continue to check
    
    # Has iter but no working len → streaming
    return True


def _chunk_text_streaming(
    texts_iter: Iterator[str],
    tokenizer,
    seq_len: int,
    stride: Optional[int] = None,
    min_tail_len: int = 1,
    add_eos: bool = True,
) -> Iterator[Tuple[List[int], bool, int, int]]:
    """
    Streaming text chunker that yields (token_ids, is_natural_stop, pad_len, overlap_mask).
    
    Memory bounded: processes one document at a time.
    """
    stride = stride if stride is not None else seq_len
    if not (1 <= stride <= seq_len):
        raise ValueError(f"stride must be in [1, seq_len], got {stride}")
    
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    
    for text in texts_iter:
        text = text.strip() if text else ""
        if not text:
            continue
        
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < min_tail_len:
            continue
        
        if add_eos and eos_id is not None:
            ids.append(eos_id)
        
        length = len(ids)
        
        if length >= seq_len:
            # Sliding-window chunks
            starts = list(range(0, length - seq_len + 1, stride))
            for i, start in enumerate(starts):
                chunk = ids[start:start + seq_len]
                is_last_start = (i == len(starts) - 1)
                reaches_end = (start + seq_len == length)
                
                if not is_last_start:
                    is_natural = False
                else:
                    tail_len = length - (start + seq_len)
                    has_tail = tail_len >= min_tail_len
                    is_natural = reaches_end or (not has_tail)
                
                overlap_mask = 0
                if start > 0 and stride < seq_len:
                    overlap_mask = seq_len - stride
                
                yield (chunk, is_natural, 0, overlap_mask)
            
            # Tail handling
            last_covered_end = starts[-1] + seq_len if starts else 0
            remainder_len = length - last_covered_end
            
            if remainder_len >= min_tail_len:
                tail = ids[last_covered_end:]
                pad_len = seq_len - remainder_len
                tail = tail + [pad_id] * pad_len
                yield (tail, True, pad_len, 0)
            elif remainder_len > 0 and starts:
                # Tail too short: last chunk becomes natural stop
                # (handled by setting is_natural=True in last yielded)
                pass
        else:
            # Short document
            pad_len = seq_len - length
            chunk = ids + [pad_id] * pad_len
            yield (chunk, True, pad_len, 0)


class HelixPrechunkedDataset(Dataset):
    """
    Dataset from pre-chunked token sequences (for sharded preprocessing).
    
    Each sample is a tuple: (token_ids, is_natural_stop, pad_len, overlap_mask)
    """
    def __init__(self, chunks: List[Tuple[List[int], bool, int, int]], seq_len: int):
        super().__init__()
        self.chunks = chunks
        self.seq_len = seq_len
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk, is_natural, pad_len, overlap_mask = self.chunks[idx]
        
        x = torch.tensor(chunk, dtype=torch.long)
        labels = x.clone()
        
        # Mask overlapping head (stride < seq_len) — loss mask only
        if overlap_mask > 0:
            labels[:overlap_mask] = -100
        
        # Mask trailing padding — loss mask only
        if pad_len > 0:
            labels[-pad_len:] = -100
        
        # Issue 4: attention_mask marks real tokens (incl. overlap) as visible.
        attention_mask = _build_attention_mask(self.seq_len, pad_len)
        
        return {
            "input_ids": x,
            "labels": labels,
            "attention_mask": attention_mask,
            "is_natural_stop": torch.tensor(is_natural, dtype=torch.bool),
        }


class HelixShardedDataset(Dataset):
    """
    Dataset that reads pre-chunked token sequences from multiple shard files on disk.
    
    This allows handling datasets larger than memory by:
    - Loading only one shard at a time into memory
    - Maintaining a global index across all shards
    - Random access via shard + local_index calculation
    - Deterministic shuffle via index permutation (indices stay in memory, data on disk)
    
    Each shard file contains pickled List[Tuple[...]] saved by _handle_streaming_iterable.
    
    The shuffle implementation uses a Torch Generator (like List[str] path) for deterministic
    reproducibility and identical behavior to non-streaming datasets.
    
    OPTIMIZED: Multi-shard LRU cache for high-throughput training
    """
    def __init__(
        self,
        shard_paths: List[str],
        seq_len: int,
        shuffle: bool = False,
        seed: int = 42,
        cache_size: int = 8,  # Number of shards to keep in memory
    ):
        super().__init__()
        self.shard_paths = shard_paths
        self.seq_len = seq_len
        self.cache_size = cache_size
        self._cache_lock = threading.RLock()
        
        # Build shard index: cumulative offsets for O(1) __getitem__
        self.shard_sizes = []
        self.shard_offsets = [0]
        self._index_map = []  # (shard_idx, local_idx) for each global index
        total = 0
        
        # Get sizes and build index map without loading full data into memory
        import pickle
        for path in shard_paths:
            with open(path, 'rb') as f:
                chunks = pickle.load(f)
                size = len(chunks)
                self.shard_sizes.append(size)
                # Record (shard_idx, local_idx) for each position
                for local_idx in range(size):
                    self._index_map.append((len(self.shard_sizes) - 1, local_idx))
                total += size
                self.shard_offsets.append(total)
        
        self.total_size = total
        self.num_shards = len(shard_paths)
        
        # Apply deterministic shuffle to index_map (only 2*int per sample, not full data)
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(seed)
            # Generate permutation indices
            perm = torch.randperm(total, generator=generator).tolist()
            # Reorder index_map according to permutation
            self._index_map = [self._index_map[i] for i in perm]
        
        # Multi-shard LRU cache: OrderedDict for O(1) move_to_end
        # Keys: shard_idx, Values: shard_data (List of chunks)
        self._shard_cache: OrderedDict[int, List] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def __len__(self) -> int:
        return self.total_size
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache hit/miss statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cached_shards": len(self._shard_cache),
            "total_shards": self.num_shards,
        }
    
    def _load_shard(self, shard_idx: int) -> List:
        """Load shard data with multi-shard LRU caching."""
        with self._cache_lock:
            # Check cache first
            if shard_idx in self._shard_cache:
                # Move to end (most recently used)
                self._shard_cache.move_to_end(shard_idx)
                self._cache_hits += 1
                return self._shard_cache[shard_idx]
            
            self._cache_misses += 1
        
        # Load from disk (outside lock to allow concurrent loads)
        import pickle
        with open(self.shard_paths[shard_idx], 'rb') as f:
            shard_data = pickle.load(f)
        
        # Add to cache with LRU eviction
        with self._cache_lock:
            # Evict oldest if at capacity
            while len(self._shard_cache) >= self.cache_size:
                self._shard_cache.popitem(last=False)
            
            self._shard_cache[shard_idx] = shard_data
            self._shard_cache.move_to_end(shard_idx)
        
        return shard_data
    
    def _prefetch_shard(self, shard_idx: int):
        """Background prefetch hint - loads shard if not in cache."""
        if shard_idx < 0 or shard_idx >= self.num_shards:
            return
        
        with self._cache_lock:
            if shard_idx in self._shard_cache:
                return  # Already cached
        
        # Load without blocking current access
        try:
            import pickle
            with open(self.shard_paths[shard_idx], 'rb') as f:
                shard_data = pickle.load(f)
            
            with self._cache_lock:
                if shard_idx not in self._shard_cache:
                    while len(self._shard_cache) >= self.cache_size:
                        self._shard_cache.popitem(last=False)
                    self._shard_cache[shard_idx] = shard_data
        except Exception:
            pass  # Silently fail prefetch
    
    def _item_from_chunk(self, chunk_data: Tuple) -> Dict[str, torch.Tensor]:
        """Convert chunk tuple to sample dict (identical to HelixPrechunkedDataset and List[str] path)."""
        chunk, is_natural, pad_len, overlap_mask = chunk_data
        
        x = torch.tensor(chunk, dtype=torch.long)
        labels = x.clone()
        
        # Mask overlapping head (stride < seq_len) — loss mask only
        if overlap_mask > 0:
            labels[:overlap_mask] = -100
        
        # Mask trailing padding — loss mask only
        if pad_len > 0:
            labels[-pad_len:] = -100
        
        # Issue 4: attention_mask marks real tokens (incl. overlap) as visible.
        attention_mask = _build_attention_mask(self.seq_len, pad_len)
        
        return {
            "input_ids": x,
            "labels": labels,
            "attention_mask": attention_mask,
            "is_natural_stop": torch.tensor(is_natural, dtype=torch.bool),
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Handle negative indices
        if idx < 0:
            idx = self.total_size + idx
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range [0, {self.total_size})")
        
        # Lookup (shard_idx, local_idx) from shuffled index map
        shard_idx, local_idx = self._index_map[idx]
        shard_data = self._load_shard(shard_idx)
        chunk_data = shard_data[local_idx]
        
        # Prefetch next shard (sequential access pattern optimization)
        # In shuffled mode, next access is random, but prefetch may help
        next_shard_idx = shard_idx + 1
        if next_shard_idx < self.num_shards:
            # Use a thread for non-blocking prefetch
            try:
                t = threading.Thread(target=self._prefetch_shard, args=(next_shard_idx,))
                t.daemon = True
                t.start()
            except Exception:
                pass  # Silently fail
        
        return self._item_from_chunk(chunk_data)


def _process_and_shard_batch(
    texts: List[str],
    tokenizer,
    seq_len: int,
    stride: Optional[int] = None,
    min_tail_len: int = 1,
    add_eos: bool = True,
) -> List[Tuple[List[int], bool, int, int]]:
    """
    Process a batch of texts and return pre-chunked samples.
    
    This is used during streaming preprocessing to build shards.
    """
    chunks = []
    stride = stride if stride is not None else seq_len
    
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    
    for text in texts:
        text = text.strip() if text else ""
        if not text:
            continue
        
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < min_tail_len:
            continue
        
        if add_eos and eos_id is not None:
            ids.append(eos_id)
        
        length = len(ids)
        
        if length >= seq_len:
            # Sliding-window chunks
            starts = list(range(0, length - seq_len + 1, stride))
            for i, start in enumerate(starts):
                chunk = ids[start:start + seq_len]
                is_last_start = (i == len(starts) - 1)
                reaches_end = (start + seq_len == length)
                
                if not is_last_start:
                    is_natural = False
                else:
                    tail_len = length - (start + seq_len)
                    has_tail = tail_len >= min_tail_len
                    is_natural = reaches_end or (not has_tail)
                
                # BUG FIX: Calculate overlap_mask correctly
                overlap_mask = 0
                if start > 0 and stride < seq_len:
                    overlap_mask = seq_len - stride
                
                chunks.append((chunk, is_natural, 0, overlap_mask))
            
            # Tail handling
            last_covered_end = starts[-1] + seq_len if starts else 0
            remainder_len = length - last_covered_end
            
            if remainder_len >= min_tail_len:
                tail = ids[last_covered_end:]
                pad_len = seq_len - remainder_len
                tail = tail + [pad_id] * pad_len
                chunks.append((tail, True, pad_len, 0))
        else:
            # Short document
            pad_len = seq_len - length
            chunk = ids + [pad_id] * pad_len
            chunks.append((chunk, True, pad_len, 0))
    
    return chunks


def _handle_streaming_iterable(
    iterable,
    tokenizer,
    seq_len: int = 2048,
    batch_size: int = 8,
    stride: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    min_tail_len: Optional[int] = None,
    add_eos: bool = True,
    seed: int = 42,
    text_column: str = "text",
    shard_cache_dir: Optional[str] = None,
    preprocess_num_proc: int = 5,
    preprocess_batch_size: int = 1000,
    cleanup_shards: bool = True,
) -> Tuple[DataLoader, str]:
    """
    Handle streaming IterableColumn by preprocessing to sharded Dataset.
    
    Strategy: Stream -> Shards -> Fast Loader
    1. Stream data in batches to shards on disk (multi-thread preprocessing)
    2. Pre-tokenize each shard using thread pool
    3. Return DataLoader from concatenated shards
    
    MEMORY-EFFICIENT: Processes batches incrementally, never materializes full dataset.
    
    Returns:
        Tuple of (DataLoader, shard_cache_dir). Caller is responsible for cleaning up
        shard_cache_dir after training completes.
    """
    import tempfile
    import os
    import pickle
    from datetime import datetime
    from concurrent.futures import ThreadPoolExecutor
    
    if min_tail_len is None:
        min_tail_len = 1
    
    # Create a cache directory that belongs to this loader only. A fixed name
    # (or a caller dir used as-is) lets a second loader overwrite the shard
    # files a live loader still reads (same shard_0000.pkl names).
    if shard_cache_dir is None:
        shard_cache_dir = tempfile.mkdtemp(prefix="helix_shards_")
    else:
        os.makedirs(shard_cache_dir, exist_ok=True)
        shard_cache_dir = tempfile.mkdtemp(prefix="helix_shards_", dir=shard_cache_dir)
    
    # Determine text extraction method
    def extract_text(example):
        if isinstance(example, dict):
            return example.get(text_column, "")
        return str(example)
    
    # Thread-safe batch processor function (closure captures tokenizer)
    def process_batch(batch):
        return _process_and_shard_batch(batch, tokenizer, seq_len, stride, min_tail_len, add_eos)
    
    # MEMORY-EFFICIENT: Stream and process batches incrementally with DETERMINISTIC ORDER
    # Use a bounded queue approach with ThreadPoolExecutor, but buffer results
    # to ensure shards are written in strict submission order
    all_chunks = []
    shard_idx = 0
    batch = []
    max_pending_batches = preprocess_num_proc * 2  # Limit pending work
    
    # For deterministic ordering: buffer completed results and write in sequence order
    completed_results = {}  # batch_idx -> chunks
    next_batch_to_write = 0  # Next batch index that should be written
    
    # Process batches incrementally using a sliding window of futures
    with ThreadPoolExecutor(max_workers=preprocess_num_proc) as executor:
        pending_futures = {}
        
        def drain_completed_futures():
            """Process completed futures and buffer them for ordered writing."""
            nonlocal all_chunks, shard_idx, next_batch_to_write
            completed = [f for f in pending_futures if f.done()]
            for f in completed:
                batch_idx = pending_futures.pop(f)
                try:
                    chunks = f.result()
                    completed_results[batch_idx] = chunks
                except Exception as e:
                    raise RuntimeError(f"Failed to process batch {batch_idx}: {e}")
            
            # Write completed batches in strict order
            while next_batch_to_write in completed_results:
                chunks = completed_results.pop(next_batch_to_write)
                all_chunks.extend(chunks)
                next_batch_to_write += 1
                
                # Save shard if it gets large
                if len(all_chunks) >= 10000:  # ~10k sequences per shard
                    shard_path = os.path.join(shard_cache_dir, f"shard_{shard_idx:04d}.pkl")
                    with open(shard_path, 'wb') as f_save:
                        pickle.dump(all_chunks, f_save)
                    all_chunks = []
                    shard_idx += 1
        
        # Stream examples and submit batches for processing
        batch_idx = 0
        for example in iterable:
            text = extract_text(example)
            if text:
                batch.append(text)
            
            if len(batch) >= preprocess_batch_size:
                # Wait if we have too many pending batches
                while len(pending_futures) >= max_pending_batches:
                    drain_completed_futures()
                    if len(pending_futures) >= max_pending_batches:
                        import time
                        time.sleep(0.001)  # Brief yield
                
                # Submit batch for processing
                future = executor.submit(process_batch, batch)
                pending_futures[future] = batch_idx
                batch_idx += 1
                batch = []
        
        # Handle remaining batch
        if batch:
            while len(pending_futures) >= max_pending_batches:
                drain_completed_futures()
            future = executor.submit(process_batch, batch)
            pending_futures[future] = batch_idx
            batch_idx += 1
        
        # Drain remaining futures
        while pending_futures:
            drain_completed_futures()
        
        # Final flush: ensure all ordered results are written
        while next_batch_to_write in completed_results:
            chunks = completed_results.pop(next_batch_to_write)
            all_chunks.extend(chunks)
            next_batch_to_write += 1
            
            if len(all_chunks) >= 10000:
                shard_path = os.path.join(shard_cache_dir, f"shard_{shard_idx:04d}.pkl")
                with open(shard_path, 'wb') as f_save:
                    pickle.dump(all_chunks, f_save)
                all_chunks = []
                shard_idx += 1
    
    # Save final shard
    if all_chunks:
        shard_path = os.path.join(shard_cache_dir, f"shard_{shard_idx:04d}.pkl")
        with open(shard_path, 'wb') as f:
            pickle.dump(all_chunks, f)
        shard_idx += 1
    
    # Build list of shard paths (do NOT load them into memory)
    shard_paths = []
    for i in range(shard_idx):
        shard_path = os.path.join(shard_cache_dir, f"shard_{i:04d}.pkl")
        if os.path.exists(shard_path):
            shard_paths.append(shard_path)
    
    # Create sharded dataset that reads from disk on-demand
    # Note: We do NOT shuffle the dataset itself - shuffle is handled by DataLoader
    # to match the List[str] path behavior exactly
    dataset = HelixShardedDataset(shard_paths, seq_len, shuffle=False, seed=seed)
    
    # Use module-level collate_fn for pickling with multiprocessing
    collate_fn = _collate_batch
    
    # DataLoader with prefetching and persistent workers for high throughput
    prefetch_factor = 4 if num_workers > 0 else None
    persistent_workers = num_workers > 0
    
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        # Linux optimization: use fork for faster worker spawning
        import sys
        if sys.platform == 'linux' and num_workers > 0:
            loader_kwargs['multiprocessing_context'] = 'fork'
        loader = DataLoader(**loader_kwargs)
    else:
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        import sys
        if sys.platform == 'linux' and num_workers > 0:
            loader_kwargs['multiprocessing_context'] = 'fork'
        loader = DataLoader(**loader_kwargs)
    
    return loader, shard_cache_dir


def create_unified_data_loader(
    data: Union[List[str], Any],
    tokenizer,
    seq_len: int = 2048,
    batch_size: int = 8,
    stride: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    min_tail_len: Optional[int] = None,
    add_eos: bool = True,
    lazy: bool = True,
    seed: int = 42,
    text_column: str = "text",
    # Sharding options (for IterableColumn path)
    shard_cache_dir: Optional[str] = None,
    preprocess_num_proc: int = 5,
    preprocess_batch_size: int = 1000,
    cleanup_shards: bool = True,
) -> Tuple[DataLoader, str]:
    """
    Create DataLoader that automatically detects data type:
    - List[str] -> DocumentAwareDataset
    - Column[str] (with __getitem__) -> DocumentAwareDataset  
    - IterableColumn[str] (streaming) -> Sharded preprocessing -> HelixShardedDataset
    
    For streaming data, returns (DataLoader, shard_cache_dir) tuple.
    The caller MUST clean up shard_cache_dir after training completes.
    
    Args:
        data: Input data - List[str], Column[str], or IterableColumn[str]
        tokenizer: Tokenizer instance
        seq_len: Sequence length for model
        batch_size: Batch size for DataLoader
        stride: If < seq_len, enables within-document overlap
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop last incomplete batch
        num_workers: Number of worker processes for DataLoader
        min_tail_len: Minimum tail length for document handling
        add_eos: Whether to add EOS token
        lazy: Whether to use lazy loading for DocumentAwareDataset
        seed: Random seed for shuffling
        text_column: Column name for text extraction from dicts
        shard_cache_dir: Directory for temporary shard storage (streaming only)
        preprocess_num_proc: Number of threads for parallel preprocessing (default: 5, streaming only)
        preprocess_batch_size: Batch size for streaming preprocessing
        cleanup_shards: If True, caller should cleanup shards after training
    
    Returns:
        DataLoader for List[str]/Column[str] inputs
        Tuple[DataLoader, str] for IterableColumn inputs (includes shard_cache_dir for cleanup)
    """
    # Detect data type
    if _is_iterable_column(data):
        # Streaming path
        return _handle_streaming_iterable(
            data, tokenizer, seq_len, batch_size, stride,
            shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,
            min_tail_len=min_tail_len, add_eos=add_eos, seed=seed,
            text_column=text_column,
            shard_cache_dir=shard_cache_dir,
            preprocess_num_proc=preprocess_num_proc,
            preprocess_batch_size=preprocess_batch_size,
            cleanup_shards=cleanup_shards,
        )
    
    # List[str] or Column[str] path
    # Convert Column to list if needed (for non-iterable columns with __getitem__)
    if hasattr(data, '__getitem__') and hasattr(data, '__len__') and not isinstance(data, list):
        data = list(data)
    
    loader = create_document_loader(
        data, tokenizer, seq_len, batch_size,
        shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,
        min_tail_len=min_tail_len, add_eos=add_eos, lazy=lazy, stride=stride,
    )
    
    return loader
