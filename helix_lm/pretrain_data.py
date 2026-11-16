"""Deterministic, disk-backed data preparation for causal pretraining.

This module deliberately does not participate in the document-aware SFT path.
It compiles an ordered text stream into exact, non-overlapping token windows,
then applies a persisted epoch permutation at load time.  In-memory and
disk-backed courts can therefore consume the same sample order instead of
merely sharing a random seed.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


FORMAT_VERSION = "helix.pretrain.samples.v1"
PERMUTATION_VERSION = "helix.pretrain.permutation.v1"
TOKEN_DTYPE = np.dtype("<u2")
PERMUTATION_DTYPE = np.dtype("<u4")


def _canonical_json(value: Dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


_TOKENIZER_PROBES = ("Hello world.", "0 1 2 3 42", "The quick brown fox, 1984; café.")


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Stable identity of a tokenizer: class, name, size, special ids and probe encodings.

    Used to refuse reusing a pretraining store that was compiled with another
    tokenizer (the store keeps only token ids, so the mismatch is otherwise silent).
    """
    try:
        size = len(tokenizer)
    except Exception:
        size = getattr(tokenizer, "vocab_size", None)
    probes = []
    for text in _TOKENIZER_PROBES:
        try:
            probes.append([int(t) for t in tokenizer.encode(text, add_special_tokens=False)])
        except Exception as exc:  # e.g. an integer-only test tokenizer
            probes.append(type(exc).__name__)
    value = {
        "class": type(tokenizer).__qualname__,
        "name": str(getattr(tokenizer, "name_or_path", getattr(tokenizer, "tokenizer_name", ""))),
        "size": size,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "probes": probes,
    }
    return "tok-sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def corpus_fingerprint(texts: Any, max_rows: int = 256) -> Optional[str]:
    """sha256 of the first `max_rows` non-empty texts, or None if `texts` is a
    one-shot iterator (peeking would consume rows the compiler needs)."""
    try:
        iterator = iter(texts)
    except TypeError:
        return None
    if iterator is texts:
        return None
    digest = hashlib.sha256()
    rows = 0
    for raw_text in iterator:
        text = str(raw_text).strip()
        if not text:
            continue
        payload = text.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        rows += 1
        if rows >= max_rows:
            break
    return f"first-{rows}-rows-sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class PretrainDatasetManifest:
    root: Path
    value: Dict[str, Any]

    @classmethod
    def load(cls, root: os.PathLike[str] | str) -> "PretrainDatasetManifest":
        root_path = Path(root)
        path = root_path / "manifest.json"
        value = json.loads(path.read_text())
        if value.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported pretraining sample format: {value.get('format_version')!r}")
        return cls(root=root_path, value=value)

    @property
    def seq_len(self) -> int:
        return int(self.value["seq_len"])

    @property
    def sample_count(self) -> int:
        return int(self.value["sample_count"])

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.value)).hexdigest()

    def verify(self) -> None:
        expected_start = 0
        observed_samples = 0
        for shard in self.value["shards"]:
            if int(shard["start_sample"]) != expected_start:
                raise ValueError("Pretraining shard sample ranges are not contiguous")
            path = self.root / shard["file"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(shard["bytes"]):
                raise ValueError(f"Pretraining shard size mismatch: {path}")
            expected_bytes = int(shard["sample_count"]) * self.seq_len * TOKEN_DTYPE.itemsize
            if path.stat().st_size != expected_bytes:
                raise ValueError(f"Pretraining shard shape mismatch: {path}")
            if _sha256_file(path) != shard["sha256"]:
                raise ValueError(f"Pretraining shard hash mismatch: {path}")
            count = int(shard["sample_count"])
            expected_start += count
            observed_samples += count
        if observed_samples != self.sample_count:
            raise ValueError("Pretraining manifest sample count does not match its shards")


class PretrainSampleCompiler:
    """Compile an ordered text iterable into fixed-size uint16 token shards."""

    def __init__(
        self,
        tokenizer: Any,
        seq_len: int,
        output_dir: os.PathLike[str] | str,
        *,
        samples_per_shard: int = 65_536,
        source: Optional[Dict[str, Any]] = None,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if samples_per_shard <= 0:
            raise ValueError("samples_per_shard must be positive")
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.output_dir = Path(output_dir)
        self.samples_per_shard = int(samples_per_shard)
        self.source = dict(source or {})
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def compile(
        self,
        texts: Iterable[str],
        *,
        max_source_rows: Optional[int] = None,
        max_samples: Optional[int] = None,
    ) -> PretrainDatasetManifest:
        self.output_dir.mkdir(parents=True, exist_ok=False)

        token_tail: List[int] = []
        pending_windows: List[List[int]] = []
        shards: List[Dict[str, Any]] = []
        source_rows = 0
        raw_utf8_bytes = 0
        encoded_tokens = 0
        sample_count = 0
        shard_start = 0
        shard_handle = None
        shard_path: Optional[Path] = None
        shard_digest = None
        shard_samples = 0
        write_block_samples = min(self.samples_per_shard, 256)

        def open_shard() -> None:
            nonlocal shard_handle, shard_path, shard_digest, shard_samples
            if shard_handle is not None:
                return
            shard_path = self.output_dir / f"samples-{len(shards):05d}.u16"
            shard_handle = shard_path.open("wb")
            shard_digest = hashlib.sha256()
            shard_samples = 0

        def flush_block() -> None:
            nonlocal pending_windows, shard_samples
            if not pending_windows:
                return
            open_shard()
            matrix = np.asarray(pending_windows, dtype=TOKEN_DTYPE)
            if matrix.shape != (len(pending_windows), self.seq_len):
                raise RuntimeError("Compiler produced a malformed pretraining window matrix")
            payload = matrix.tobytes(order="C")
            shard_handle.write(payload)
            shard_digest.update(payload)
            shard_samples += len(pending_windows)
            pending_windows = []

        def close_shard() -> None:
            nonlocal shard_handle, shard_path, shard_digest, shard_samples, shard_start
            flush_block()
            if shard_handle is None:
                return
            shard_handle.flush()
            os.fsync(shard_handle.fileno())
            shard_handle.close()
            shards.append(
                {
                    "file": shard_path.name,
                    "start_sample": shard_start,
                    "sample_count": shard_samples,
                    "bytes": shard_path.stat().st_size,
                    "sha256": shard_digest.hexdigest(),
                }
            )
            shard_start += shard_samples
            shard_handle = None
            shard_path = None
            shard_digest = None
            shard_samples = 0

        for raw_text in texts:
            if max_source_rows is not None and source_rows >= max_source_rows:
                break
            source_rows += 1
            text = str(raw_text).strip()
            if not text:
                continue
            raw_utf8_bytes += len(text.encode("utf-8"))
            token_ids = list(self.tokenizer.encode(text, add_special_tokens=False))
            if self.eos_token_id is not None:
                token_ids.append(int(self.eos_token_id))
            if token_ids and (min(token_ids) < 0 or max(token_ids) > np.iinfo(TOKEN_DTYPE).max):
                raise ValueError("Tokenizer vocabulary does not fit the uint16 pretraining format")
            encoded_tokens += len(token_ids)
            token_tail.extend(token_ids)

            available_samples = len(token_tail) // self.seq_len
            if max_samples is not None:
                available_samples = min(available_samples, max_samples - sample_count)
            consumed_tokens = available_samples * self.seq_len
            for start in range(0, consumed_tokens, self.seq_len):
                pending_windows.append(token_tail[start : start + self.seq_len])
                sample_count += 1
                if shard_samples + len(pending_windows) >= self.samples_per_shard:
                    flush_block()
                    close_shard()
                elif len(pending_windows) >= write_block_samples:
                    flush_block()
            if consumed_tokens:
                token_tail = token_tail[consumed_tokens:]
            if max_samples is not None and sample_count >= max_samples:
                break

        close_shard()
        manifest_value = {
            "format_version": FORMAT_VERSION,
            "token_dtype": TOKEN_DTYPE.str,
            "seq_len": self.seq_len,
            "eos_token_id": self.eos_token_id,
            "source_rows": source_rows,
            "raw_utf8_bytes": raw_utf8_bytes,
            "encoded_tokens": encoded_tokens,
            "sample_count": sample_count,
            "causal_target_count": sample_count * (self.seq_len - 1),
            "dropped_tail_tokens": len(token_tail),
            "samples_per_shard": self.samples_per_shard,
            "source": self.source,
            "shards": shards,
        }
        _atomic_write(self.output_dir / "manifest.json", _canonical_json(manifest_value))
        manifest = PretrainDatasetManifest.load(self.output_dir)
        manifest.verify()
        return manifest


class PretrainIndexedDataset(Dataset):
    """Random-access view over compiled pretraining sample shards."""

    def __init__(self, root: os.PathLike[str] | str, *, verify: bool = False) -> None:
        self.manifest = PretrainDatasetManifest.load(root)
        if verify:
            self.manifest.verify()
        self.root = self.manifest.root
        self.seq_len = self.manifest.seq_len
        self._shards = list(self.manifest.value["shards"])
        self._starts = [int(shard["start_sample"]) for shard in self._shards]
        self._maps: Dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self.manifest.sample_count

    def _shard_map(self, shard_index: int) -> np.memmap:
        mapping = self._maps.get(shard_index)
        if mapping is None:
            shard = self._shards[shard_index]
            mapping = np.memmap(
                self.root / shard["file"],
                mode="r",
                dtype=TOKEN_DTYPE,
                shape=(int(shard["sample_count"]), self.seq_len),
            )
            self._maps[shard_index] = mapping
        return mapping

    def __getitem__(self, sample_id: int) -> Dict[str, torch.Tensor]:
        if sample_id < 0:
            sample_id += len(self)
        if sample_id < 0 or sample_id >= len(self):
            raise IndexError(sample_id)
        shard_index = bisect.bisect_right(self._starts, sample_id) - 1
        shard = self._shards[shard_index]
        local_index = sample_id - int(shard["start_sample"])
        tokens = torch.from_numpy(np.array(self._shard_map(shard_index)[local_index], dtype=np.int64))
        return {
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
            "input_ids": tokens,
            "labels": tokens.clone(),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
        }


@dataclass(frozen=True)
class PretrainPermutation:
    path: Path
    metadata: Dict[str, Any]

    @classmethod
    def create(
        cls,
        path: os.PathLike[str] | str,
        sample_count: int,
        seed: int,
        *,
        epoch: int = 0,
    ) -> "PretrainPermutation":
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if sample_count > np.iinfo(PERMUTATION_DTYPE).max:
            raise ValueError("sample_count exceeds the uint32 permutation format")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), int(epoch)])))
        values = rng.permutation(sample_count).astype(PERMUTATION_DTYPE, copy=False)
        temporary = path.with_suffix(path.suffix + ".tmp")
        values.tofile(temporary)
        os.replace(temporary, path)
        metadata = {
            "format_version": PERMUTATION_VERSION,
            "algorithm": "numpy.pcg64.seedsequence-v1",
            "seed": int(seed),
            "epoch": int(epoch),
            "sample_count": int(sample_count),
            "dtype": PERMUTATION_DTYPE.str,
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        _atomic_write(path.with_suffix(path.suffix + ".json"), _canonical_json(metadata))
        return cls(path=path, metadata=metadata)

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> "PretrainPermutation":
        path = Path(path)
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        if metadata.get("format_version") != PERMUTATION_VERSION:
            raise ValueError(f"Unsupported pretraining permutation: {metadata.get('format_version')!r}")
        if metadata.get("algorithm") != "numpy.pcg64.seedsequence-v1":
            raise ValueError("Unsupported pretraining permutation algorithm")
        if metadata.get("file") != path.name:
            raise ValueError("Pretraining permutation metadata path mismatch")
        if metadata.get("dtype") != PERMUTATION_DTYPE.str:
            raise ValueError("Pretraining permutation dtype mismatch")
        if int(metadata.get("epoch", -1)) < 0:
            raise ValueError("Pretraining permutation epoch must be non-negative")
        if path.stat().st_size != int(metadata["bytes"]):
            raise ValueError("Pretraining permutation size mismatch")
        expected_bytes = int(metadata["sample_count"]) * PERMUTATION_DTYPE.itemsize
        if path.stat().st_size != expected_bytes:
            raise ValueError("Pretraining permutation shape mismatch")
        if _sha256_file(path) != metadata["sha256"]:
            raise ValueError("Pretraining permutation hash mismatch")
        return cls(path=path, metadata=metadata)

    @property
    def sample_count(self) -> int:
        return int(self.metadata["sample_count"])

    def values(self) -> np.memmap:
        return np.memmap(self.path, mode="r", dtype=PERMUTATION_DTYPE, shape=(self.sample_count,))


class PretrainPermutationSampler(Sampler[int]):
    """Replay an exact persisted sample order with a fixed exclusion set."""

    def __init__(
        self,
        permutation: PretrainPermutation,
        *,
        cursor: int = 0,
        excluded_sample_ids: Optional[Sequence[int]] = None,
    ) -> None:
        excluded = frozenset(int(sample_id) for sample_id in (excluded_sample_ids or ()))
        if any(sample_id < 0 or sample_id >= permutation.sample_count for sample_id in excluded):
            raise ValueError("excluded sample ID is outside the persisted permutation")
        available_samples = permutation.sample_count - len(excluded)
        if cursor < 0 or cursor > available_samples:
            raise ValueError("cursor is outside the persisted permutation")
        self.permutation = permutation
        self.cursor = int(cursor)
        self.excluded_sample_ids = excluded
        self.available_samples = available_samples

    def __iter__(self) -> Iterator[int]:
        values = self.permutation.values()
        included_position = 0
        for value in values:
            sample_id = int(value)
            if sample_id in self.excluded_sample_ids:
                continue
            if included_position >= self.cursor:
                yield sample_id
            included_position += 1

    def __len__(self) -> int:
        return self.available_samples - self.cursor


def collate_pretrain_samples(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "sample_id": torch.stack([sample["sample_id"] for sample in batch]),
        "input_ids": torch.stack([sample["input_ids"] for sample in batch]),
        "labels": torch.stack([sample["labels"] for sample in batch]),
        "attention_mask": torch.stack([sample["attention_mask"] for sample in batch]),
    }


def create_pretrain_indexed_loader(
    dataset: PretrainIndexedDataset,
    permutation: PretrainPermutation,
    batch_size: int,
    *,
    cursor: int = 0,
    excluded_sample_ids: Optional[Sequence[int]] = None,
    num_workers: int = 0,
    drop_last: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    seed: int = 0,
) -> DataLoader:
    if len(dataset) != permutation.sample_count:
        raise ValueError("Permutation sample count does not match the pretraining dataset")
    # A DataLoader without its own generator draws its worker base seed from the
    # GLOBAL torch RNG on every iter(): a resumed run then restores the RNG,
    # creates the iterator, and its dropout stream no longer matches the
    # uninterrupted run. The order itself comes from the persisted permutation.
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "sampler": PretrainPermutationSampler(
            permutation,
            cursor=cursor,
            excluded_sample_ids=excluded_sample_ids,
        ),
        "shuffle": False,
        "drop_last": drop_last,
        "collate_fn": collate_pretrain_samples,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory and torch.cuda.is_available()),
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)
