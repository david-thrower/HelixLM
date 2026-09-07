#!/usr/bin/env python3
"""Canonical Branch 62 continuous-pretraining launcher.

``Trainer`` remains the document-aware SFT path. This launcher uses only
``PretrainTrainer``: exact persisted sample order, local resumable checkpoints,
and a local JSONL metric record projected to MLflow.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from datasets import load_dataset

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, PretrainTrainer
from helix_lm.experiment_tracking import ExperimentTracker


SUTRA_DATASET = "codelion/sutra-10B"
SUTRA_REVISION = "415549cff1a92b69df8b88c6108faa6097457068"
REFERENCE_STORE_MANIFEST = "f1ebfff2f2dfa7396ab5b78c4fbd3cbd0bb1fca0cfb1ff7ace64a17858c907b2"
REFERENCE_PERMUTATION = "8df7f8c24708ef6507ada2c87f7f2cb6ab9f681417e57fa073d4d854f87f6680"
REFERENCE_VALIDATION_IDS = "0e4471ec0dd5a3dbab4a941d2d64ef646b01919d3594000a58059915409eacad"


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    d_model: int
    n_heads: int
    batch_size: int
    grad_accum: int
    max_optimizer_steps: int
    reference_run_id: str
    claim: str

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum


PROFILES = {
    "rtx5080-relative": TrainingProfile(
        name="rtx5080-relative",
        d_model=768,
        n_heads=12,
        batch_size=3,
        grad_accum=28,
        max_optimizer_steps=3082,
        reference_run_id="0abd42410cce4fc880009320c4663287",
        claim=(
            "Branch 62 width using the admitted RTX 5080 microbatch shape; "
            "not an exact replay of the 250-step reference"
        ),
    ),
    "branch60-exact-shape": TrainingProfile(
        name="branch60-exact-shape",
        d_model=1024,
        n_heads=16,
        batch_size=2,
        grad_accum=42,
        max_optimizer_steps=3082,
        reference_run_id="6ad46206ff1d49a3a96d71fd7723f16b",
        claim=(
            "Exact model/data/optimizer shape of the Branch 60 overnight run; "
            "Branch 62 source is a new executable subject"
        ),
    ),
}


@dataclass(frozen=True)
class RunSettings:
    profile: TrainingProfile
    dataset: str
    dataset_revision: str
    dataset_split: str
    text_column: str
    tokenizer_name: str
    train_store_dir: Optional[Path]
    compile_store_dir: Optional[Path]
    resume_training_state: Optional[Path]
    output_root: Path
    epochs: int
    learning_rate: float
    warmup_microbatches: int
    validation_samples: int
    validation_batches: int
    checkpoint_every_steps: int
    eval_every_steps: int
    max_optimizer_steps: Optional[int]
    num_workers: int
    seed: int
    mlflow_uri: str
    mlflow_experiment: str
    require_mlflow: bool
    push_to_hub: bool
    hf_username: str


def _optional_path(value: Optional[str]) -> Optional[Path]:
    return Path(value).expanduser() if value and value.strip() else None


def _env_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    value = environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def resolve_settings(environ: Mapping[str, str] = os.environ) -> RunSettings:
    profile_name = environ.get("HELIX_PROFILE", "rtx5080-relative")
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown HELIX_PROFILE: {profile_name}")
    profile = PROFILES[profile_name]
    max_steps_value = int(
        environ.get("HELIX_MAX_OPTIMIZER_STEPS", str(profile.max_optimizer_steps))
    )
    timestamp = datetime.now(timezone.utc).strftime("%y%m%d-%H%M")
    train_store_dir = _optional_path(environ.get("HELIX_PRETRAIN_STORE_DIR"))
    compile_store_dir = _optional_path(
        environ.get(
            "HELIX_PRETRAIN_COMPILE_DIR",
            "" if train_store_dir else "./pretrain_store",
        )
    )
    settings = RunSettings(
        profile=profile,
        dataset=environ.get("HELIX_DATASET", SUTRA_DATASET),
        dataset_revision=environ.get("HELIX_DATASET_REVISION", SUTRA_REVISION),
        dataset_split=environ.get("HELIX_DATASET_SPLIT", "train"),
        text_column=environ.get("HELIX_TEXT_COLUMN", "text"),
        tokenizer_name=environ.get("HELIX_TOKENIZER", "gpt2"),
        train_store_dir=train_store_dir,
        compile_store_dir=compile_store_dir,
        resume_training_state=_optional_path(environ.get("HELIX_RESUME_TRAINING_STATE")),
        output_root=Path(
            environ.get(
                "HELIX_OUTPUT_DIR",
                f"production_runs/hlx-b62-{profile.name}-{timestamp}",
            )
        ).expanduser(),
        epochs=int(environ.get("HELIX_EPOCHS", "1")),
        learning_rate=float(environ.get("HELIX_LEARNING_RATE", "0.0002")),
        warmup_microbatches=int(environ.get("HELIX_WARMUP_MICROBATCHES", "500")),
        validation_samples=int(environ.get("HELIX_VALIDATION_SAMPLES", "252")),
        validation_batches=int(environ.get("HELIX_VALIDATION_BATCHES", "8")),
        checkpoint_every_steps=int(environ.get("HELIX_CHECKPOINT_EVERY", "250")),
        eval_every_steps=int(environ.get("HELIX_EVAL_EVERY", "250")),
        max_optimizer_steps=max_steps_value or None,
        num_workers=int(environ.get("HELIX_NUM_WORKERS", "4")),
        seed=int(environ.get("HELIX_SEED", "42")),
        mlflow_uri=environ.get("HELIX_MLFLOW_URI", "https://mlflow.thunderline.net"),
        mlflow_experiment=environ.get(
            "HELIX_MLFLOW_EXPERIMENT", "helix-branch60-rtx5080-v0"
        ),
        require_mlflow=_env_bool(environ, "HELIX_REQUIRE_MLFLOW", True),
        push_to_hub=_env_bool(environ, "HELIX_PUSH_TO_HUB", False),
        hf_username=environ.get("HELIX_HF_USERNAME", "david-thrower"),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: RunSettings) -> None:
    if settings.profile.effective_batch != 84:
        raise ValueError("Comparison profiles must preserve effective batch 84")
    if settings.profile.d_model % settings.profile.n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    if settings.epochs <= 0 or settings.learning_rate <= 0:
        raise ValueError("epochs and learning rate must be positive")
    positive = {
        "warmup microbatches": settings.warmup_microbatches,
        "validation samples": settings.validation_samples,
        "validation batches": settings.validation_batches,
        "checkpoint interval": settings.checkpoint_every_steps,
        "evaluation interval": settings.eval_every_steps,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError("All count and interval settings must be positive")
    if settings.resume_training_state and not settings.train_store_dir:
        raise ValueError("Exact resume requires HELIX_PRETRAIN_STORE_DIR")
    if settings.train_store_dir and settings.compile_store_dir:
        raise ValueError("Existing and new pretraining stores are mutually exclusive")
    if not settings.dataset_revision:
        raise ValueError("HELIX_DATASET_REVISION must be pinned")


def source_identity() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()

    return {
        "source_head": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_branch": git("branch", "--show-current") or "DETACHED",
        "source_dirty": bool(git("status", "--porcelain")),
    }


def gpu_utilization_percent() -> Optional[float]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits", "--id=0"],
            check=True, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=5,
        )
        return float(completed.stdout.strip().splitlines()[0])
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return None


def build_config(settings: RunSettings, tokenizer: HelixTokenizer) -> HelixConfig:
    profile = settings.profile
    cfg = HelixConfig.small_v2(
        vocab_size=len(tokenizer), tokenizer_name=settings.tokenizer_name,
        d_model=profile.d_model, n_columns=3, nodes_per_column=(3, 3, 3),
        n_heads=profile.n_heads, n_loops=4, seq_len=1024,
        dropout=0.1, attn_dropout=0.05, ffn_expansion=3.0,
        weight_decay=0.05, grad_clip=1.0, grad_buffer_ratio=0.0,
        batch_size=profile.batch_size, lr=settings.learning_rate,
        warmup_steps=settings.warmup_microbatches, epochs=settings.epochs,
        use_cca=False, use_ssm=False, use_titans_memory=False,
        seed=settings.seed, device="auto", dtype="float32",
        amp_dtype="bfloat16", lateral_p=0.8, vertical_p=0.9,
        vertical_depth=2, attention_mode="multi_scale_windowed",
        local_window=64, coarse_window=128, compressed_windows=8,
        compressed_views=8, consensus_type="cosine", corrector_type="ffn",
        tie_word_embeddings=True, strict_nan_check=True,
    )
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.bos_token_id = tokenizer.bos_token_id
    return cfg


def model_name(settings: RunSettings, timestamp: str) -> str:
    profile = settings.profile
    value = (
        f"hlx-b62-{timestamp}-d{profile.d_model}-c3-n333-l4-"
        f"f30-s1024-e{settings.epochs}"
    )
    if len(value) > 96:
        raise ValueError("Generated Hugging Face model name exceeds 96 characters")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def training_texts(settings: RunSettings):
    if settings.train_store_dir:
        return None
    dataset = load_dataset(
        settings.dataset, streaming=True, revision=settings.dataset_revision
    )
    return dataset[settings.dataset_split][settings.text_column]


def main() -> None:
    settings = resolve_settings()
    if os.environ.get("HELIX_PRINT_CONTRACT") == "1":
        print(json.dumps(asdict(settings), default=str, indent=2))
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("UNAVAILABLE: CUDA with BF16 support is required")

    timestamp = datetime.now(timezone.utc).strftime("%y%m%d-%H%M")
    settings.output_root.mkdir(parents=True, exist_ok=True)
    subject = source_identity()
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    torch.cuda.reset_peak_memory_stats()

    tokenizer = HelixTokenizer(settings.tokenizer_name)
    cfg = build_config(settings, tokenizer)
    model = HelixForCausalLM(cfg)
    counts = model.count_parameters()
    graph_info = model.model.recurrent.graph.get_graph_info()
    run_name = model_name(settings, timestamp)
    pretrain_source = {
        "dataset": settings.dataset, "revision": settings.dataset_revision,
        "split": settings.dataset_split, "text_column": settings.text_column,
        "tokenizer": settings.tokenizer_name,
    }
    params = {
        **subject, "profile": settings.profile.name,
        "profile_claim": settings.profile.claim,
        "reference_run_id": settings.profile.reference_run_id,
        **pretrain_source, "vocab_size": len(tokenizer),
        "d_model": cfg.d_model, "n_heads": cfg.n_heads,
        "n_columns": cfg.n_columns, "nodes_per_column": "3,3,3",
        "nodes_per_column_graph_effective": False, "n_loops": cfg.n_loops,
        "ffn_expansion": cfg.ffn_expansion, "sequence_length": cfg.seq_len,
        "local_window": cfg.local_window, "coarse_window": cfg.coarse_window,
        "compressed_windows": cfg.compressed_windows,
        "compressed_views": cfg.compressed_views,
        "lateral_p": cfg.lateral_p, "vertical_p": cfg.vertical_p,
        "vertical_depth": cfg.vertical_depth,
        "batch_size": settings.profile.batch_size,
        "grad_accum": settings.profile.grad_accum,
        "effective_batch": settings.profile.effective_batch,
        "learning_rate": settings.learning_rate,
        "warmup_microbatches": settings.warmup_microbatches,
        "epochs": settings.epochs,
        "max_optimizer_steps": settings.max_optimizer_steps,
        "validation_samples": settings.validation_samples,
        "validation_batches": settings.validation_batches,
        "checkpoint_every_steps": settings.checkpoint_every_steps,
        "eval_every_steps": settings.eval_every_steps, "seed": settings.seed,
        "amp": "bfloat16", "strict_nan_check": True,
        "parameter_count": counts["total"],
        "graph_nodes": graph_info["n_nodes"], "graph_edges": graph_info["n_edges"],
        "gpu_name": torch.cuda.get_device_name(0),
    }
    tracker = ExperimentTracker(
        tracking_uri=settings.mlflow_uri, experiment=settings.mlflow_experiment,
        run_name=run_name, spool_path=settings.output_root / "mlflow-events.jsonl",
        params=params,
        tags={"branch": subject["source_branch"],
              "source_head": subject["source_head"],
              "comparison_reference": settings.profile.reference_run_id,
              "data_contract": "eos_joined_nonoverlap_persisted_permutation_v1"},
        require_remote=settings.require_mlflow,
    )
    data_stats: dict[str, float] = {}
    trainer: Optional[PretrainTrainer] = None

    def on_step(metrics: dict[str, float]) -> None:
        cursor = metrics["sample_cursor"]
        sample_count = max(data_stats.get("sample_count", 0.0), 1.0)
        raw_bytes = data_stats.get("raw_utf8_bytes", 0.0) * cursor / sample_count
        tracker.log_metrics(
            {"train/loss": metrics["loss"],
             "train/ppl": metrics["perplexity"],
             "train/lr": metrics["lr"],
             "train/global_step": metrics["global_step"],
             "train/grad_norm": metrics["grad_norm"],
             "train/causal_targets_total": metrics["causal_targets_total"],
             "train/causal_targets_per_second_session": metrics["causal_targets_per_second_session"],
             "train/causal_targets_per_second_step": metrics["causal_targets_per_second_step"],
             "train/step_seconds": metrics["step_seconds"],
             "train/skipped_batches": metrics["skipped_batches"],
             "data/sample_cursor": cursor,
             "data/sample_store_bytes_consumed": cursor * cfg.seq_len * 2,
             "data/raw_utf8_bytes_exposure_estimated": raw_bytes,
             "system/vram_allocated_bytes": metrics["vram_allocated_bytes"],
             "system/vram_reserved_bytes": metrics["vram_reserved_bytes"],
             "system/peak_vram_bytes": metrics["peak_vram_bytes"],
             "system/gpu_utilization_percent": gpu_utilization_percent()},
            step=int(metrics["global_step"]), phase="train",
        )

    def on_validation(metrics: dict[str, float]) -> None:
        assert trainer is not None
        tracker.log_metrics(
            {"val/loss": metrics["loss"], "val/ppl": metrics["perplexity"],
             "val/causal_targets": metrics["causal_targets"],
             "val/sample_count": metrics["sample_count"]},
            step=trainer.global_step, phase="validation",
        )

    terminal_status = "FAILED"
    terminal: Optional[dict[str, Any]] = None
    started = time.time()
    try:
        trainer = PretrainTrainer(
            model=model, cfg=cfg, train_texts=training_texts(settings),
            train_store_dir=settings.train_store_dir,
            pretrain_store_dir=settings.compile_store_dir,
            pretrain_source=pretrain_source,
            resume_training_state=settings.resume_training_state,
            validation_sample_count=settings.validation_samples,
            tokenizer=tokenizer, output_dir=settings.output_root / "checkpoints",
            grad_accum_steps=settings.profile.grad_accum, use_amp=True,
            amp_dtype="bfloat16", verbose=True, num_workers=settings.num_workers,
            total_optimizer_steps=settings.max_optimizer_steps,
            max_optimizer_steps=settings.max_optimizer_steps, min_lr_ratio=1.0,
            checkpoint_every_steps=settings.checkpoint_every_steps,
            checkpoint_slots=2, step_callback=on_step,
            eval_every_steps=settings.eval_every_steps,
            validation_batches=settings.validation_batches,
            evaluation_callback=on_validation,
        )
        manifest = trainer._train_dataset.manifest
        data_stats.update(sample_count=float(manifest.sample_count),
                          raw_utf8_bytes=float(manifest.value["raw_utf8_bytes"]))
        contract = {
            "schema": "helix.branch62-pretrain-run.v1", **params,
            "run_name": run_name, "output_root": str(settings.output_root.resolve()),
            "sample_store": str(manifest.root.resolve()),
            "sample_manifest_sha256": manifest.manifest_sha256,
            "permutation_sha256": trainer._train_permutation.metadata["sha256"],
            "validation_policy": "tail_of_epoch_zero_persisted_permutation_v0",
            "validation_sample_ids_sha256": trainer._validation_sample_ids_sha256,
            "reference_store_manifest_sha256": REFERENCE_STORE_MANIFEST,
            "reference_permutation_sha256": REFERENCE_PERMUTATION,
            "reference_validation_ids_sha256": REFERENCE_VALIDATION_IDS,
        }
        write_json(settings.output_root / "run_contract.json", contract)
        tracker.params.update({"sample_manifest_sha256": manifest.manifest_sha256,
                               "permutation_sha256": trainer._train_permutation.metadata["sha256"],
                               "validation_sample_ids_sha256": trainer._validation_sample_ids_sha256})
        mlflow_run_id = tracker.start()
        contract["mlflow_run_id"] = mlflow_run_id
        write_json(settings.output_root / "run_contract.json", contract)

        history = trainer.train(num_epochs=settings.epochs, eval_every=1)
        final_dir = settings.output_root / "final-model"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        hub_repo = ""
        if settings.push_to_hub:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError("HELIX_PUSH_TO_HUB=1 requires HF_TOKEN")
            hub_repo = f"{settings.hf_username}/{run_name}"
            model.push_to_hub(hub_repo, token=token)
            tokenizer.push_to_hub(hub_repo, token=token)

        train_loss = (history.get("train_loss") or [float("nan")])[-1]
        val_loss = (history.get("val_loss") or [float("nan")])[-1]
        terminal = {
            "status": "PASS", "mlflow_run_id": mlflow_run_id,
            "mlflow_errors": tracker.errors, "global_step": trainer.global_step,
            "sample_cursor": trainer._train_cursor, "final_train_loss": train_loss,
            "final_train_ppl": math.exp(min(train_loss, 20)),
            "final_val_loss": val_loss, "final_val_ppl": math.exp(min(val_loss, 20)),
            "elapsed_seconds": time.time() - started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "local_model": str(final_dir.resolve()), "hub_repo": hub_repo,
        }
        tracker.log_metrics(
            {"final/train_loss": terminal["final_train_loss"],
             "final/train_ppl": terminal["final_train_ppl"],
             "final/val_loss": terminal["final_val_loss"],
             "final/val_ppl": terminal["final_val_ppl"],
             "final/elapsed_seconds": terminal["elapsed_seconds"],
             "final/peak_vram_bytes": terminal["peak_vram_bytes"]},
            step=trainer.global_step, phase="terminal",
        )
        terminal_status = "FINISHED"
    finally:
        projected_status = tracker.finish(terminal_status)
        if terminal is not None:
            terminal["mlflow_errors"] = list(tracker.errors)
            terminal["status"] = (
                "PASS" if projected_status == "FINISHED"
                else "PASS_WITH_MLFLOW_ERRORS"
            )
            write_json(settings.output_root / "run_terminal.json", terminal)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; resume from the latest local training-state checkpoint", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
