"""
HelixLM Trainer with gradient accumulation, configurable AMP, and progress bars.

Key features:
  - Gradient accumulation for effective larger batch sizes
  - Configurable AMP (default: off for stability on small models)
  - NaN/Inf detection and batch skipping
  - Scheduler steps count optimizer steps, not raw batches
  - Uses DocumentAwareDataset (no cross-document boundary crossings)
  - Modern torch.amp API (not deprecated torch.cuda.amp)
  - Live tqdm progress bars with loss, PPL, LR, and throughput metrics
  - Optional train/val DataLoader injection for custom dataset pipelines
"""
import os
import math
import time
import warnings
import tempfile
import hashlib
import json
from typing import Optional, List, Dict, Any, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import HelixConfig
from .hf_model import HelixForCausalLM
from .dataset import create_document_loader, create_unified_data_loader, _is_iterable_column


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.1,
):
    """Cosine learning rate schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (
            1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        )
        return min_lr_ratio + (1.0 - min_lr_ratio) * max(0.0, cosine)

    return LambdaLR(optimizer, lr_lambda)


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss, capping at exp(20) to avoid overflow."""
    return math.exp(min(loss, 20))


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


class Trainer:
    """Trainer for HelixLM with gradient accumulation, AMP, and progress bars."""

    def __init__(
        self,
        model: HelixForCausalLM,
        cfg: HelixConfig,
        train_texts: Optional[Union[List[str], Any]] = None,
        val_texts: Optional[Union[List[str], Any]] = None,
        tokenizer=None,
        output_dir: str = "./checkpoints",
        example_prompts: Optional[List[str]] = None,
        generated_example_length: int = 15,
        grad_accum_steps: int = 1,
        use_amp: bool = False,
        amp_dtype: Optional[str] = None,
        min_tail_len: Optional[int] = None,
        stride: Optional[int] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True,
        # Sharding options for IterableColumn stream
        shard_cache_dir: Optional[str] = None,
        preprocess_num_proc: int = 5,
        preprocess_batch_size: int = 1000,
        cleanup_shards: bool = True,
        # DataLoader performance options
        num_workers: int = 4,  # Number of DataLoader workers for prefetching
    ):
        """
        Initialize Trainer.

        Args:
            model: HelixForCausalLM instance.
            cfg: HelixConfig with training hyperparameters.
            train_texts: List of training document texts (used if train_loader not provided).
                         Can also be IterableColumn from streaming dataset.
            val_texts: List of validation document texts (used if val_loader not provided).
                       Can also be IterableColumn from streaming dataset.
            tokenizer: Tokenizer instance.
            output_dir: Directory to save checkpoints.
            example_prompts: Prompts for generation samples during training.
            generated_example_length: Number of tokens to generate for samples.
            grad_accum_steps: Gradient accumulation steps (default: 1).
            use_amp: Whether to use torch.amp automatic mixed precision.
            amp_dtype: AMP autocast dtype: "float16" or "bfloat16" (default: "float16").
            min_tail_len: Minimum tail length for DocumentAwareDataset.
            stride: Chunking stride for document sliding window. If None:
                    - Defaults to seq_len if seq_len <= 512 (no overlap)
                    - Defaults to 512 if seq_len > 512 (~50% overlap for longer contexts)
            train_loader: Optional custom DataLoader to override built-in dataset creation.
            val_loader: Optional custom DataLoader to override built-in dataset creation.
            verbose: Whether to show tqdm progress bars and print logs.
            shard_cache_dir: Directory for temporary shard storage (streaming only).
            preprocess_num_proc: Number of processes for preprocessing (streaming only).
            preprocess_batch_size: Batch size for streaming preprocessing.
            cleanup_shards: Whether to auto-cleanup shards after training.
            num_workers: Number of DataLoader worker processes for background data loading.
                        Higher values enable more prefetching but use more CPU/RAM.
                        Set to 0 for single-threaded loading (default: 4).
        """
        # Apply intelligent stride default based on seq_len
        if stride is None:
            if cfg.seq_len > 512:
                stride = 512  # Standard: 50% overlap for longer contexts
            else:
                stride = cfg.seq_len  # No overlap for shorter contexts
        self._stride = stride

        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.use_amp = use_amp and torch.cuda.is_available()
        _amp_dtype = amp_dtype if amp_dtype is not None else getattr(cfg, "amp_dtype", "float16")
        self.amp_dtype = getattr(torch, _amp_dtype) if isinstance(_amp_dtype, str) else _amp_dtype
        self.verbose = verbose

        # Store shard cleanup settings
        self._shard_cache_dir = shard_cache_dir
        self._cleanup_shards = cleanup_shards
        self._train_texts_is_streaming = False
        self._val_texts_is_streaming = False

        if example_prompts:
            self.example_prompts = example_prompts
        else:
            self.example_prompts = [
                "In the beginning",
                "And God said",
                "The sky was",
            ]
        self.generated_example_length = generated_example_length

        self.device = self._get_device()
        self.model = self.model.to(self.device)

        # Validate config
        self.validate_config()

        # Data loaders: use injected loaders if provided, otherwise build from texts
        if train_loader is not None:
            self.train_loader = train_loader
        else:
            if train_texts is None:
                raise ValueError("Either train_loader or train_texts must be provided.")
            
            # Check if streaming data
            if _is_iterable_column(train_texts):
                self._train_texts_is_streaming = True
                result = create_unified_data_loader(
                    train_texts,
                    tokenizer,
                    cfg.seq_len,
                    cfg.batch_size,
                    stride=stride,
                    shuffle=True,
                    drop_last=True,
                    num_workers=num_workers,
                    min_tail_len=min_tail_len,
                    seed=getattr(cfg, 'seed', 42),  # Use cfg.seed for determinism
                    shard_cache_dir=shard_cache_dir,
                    preprocess_num_proc=preprocess_num_proc,
                    preprocess_batch_size=preprocess_batch_size,
                    cleanup_shards=cleanup_shards,
                )
                if isinstance(result, tuple):
                    self.train_loader, self._train_shard_dir = result
                else:
                    self.train_loader = result
                    self._train_shard_dir = None
            else:
                self.train_loader = create_document_loader(
                    train_texts,
                    tokenizer,
                    cfg.seq_len,
                    cfg.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    min_tail_len=min_tail_len,
                    seed=getattr(cfg, 'seed', 42),  # Use cfg.seed for determinism
                    lazy=True,
                    stride=stride,
                )
                self._train_shard_dir = None

        self.val_loader = None
        if val_loader is not None:
            self.val_loader = val_loader
        elif val_texts is not None:
            # Check if streaming data
            if _is_iterable_column(val_texts):
                self._val_texts_is_streaming = True
                result = create_unified_data_loader(
                    val_texts,
                    tokenizer,
                    cfg.seq_len,
                    cfg.batch_size,
                    stride=stride,
                    shuffle=False,
                    drop_last=False,
                    num_workers=num_workers,
                    min_tail_len=min_tail_len,
                    seed=getattr(cfg, 'seed', 42),  # Use cfg.seed for determinism
                    shard_cache_dir=shard_cache_dir,
                    preprocess_num_proc=preprocess_num_proc,
                    preprocess_batch_size=preprocess_batch_size,
                    cleanup_shards=cleanup_shards,
                )
                if isinstance(result, tuple):
                    self.val_loader, self._val_shard_dir = result
                else:
                    self.val_loader = result
                    self._val_shard_dir = None
            else:
                self.val_loader = create_document_loader(
                    val_texts,
                    tokenizer,
                    cfg.seq_len,
                    cfg.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=num_workers,
                    min_tail_len=min_tail_len,
                    seed=getattr(cfg, 'seed', 42),  # Use cfg.seed for determinism
                    lazy=True,
                    stride=stride,
                )
                self._val_shard_dir = None

        # AdamW with standard betas (0.9, 0.999)
        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler steps count optimizer steps, not raw batches
        # DEFERRED: avoid len() on lazy datasets to prevent eager chunking at init time.
        # The scheduler is built on the first train_epoch() call when length is known.
        self._scheduler_warmup = max(1, cfg.warmup_steps // self.grad_accum_steps)
        self._scheduler_cycles = 0.5
        self._scheduler_min_lr = 0.1
        self.scheduler = None

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "perplexity": []}

        # GradScaler for AMP (only if use_amp=True and CUDA available and dtype is float16)
        # BFloat16 does not need/ support GradScaler — it has sufficient range natively.
        self.scaler = None
        if self.use_amp and self.amp_dtype == torch.float16:
            try:
                from torch.amp import GradScaler
                self.scaler = GradScaler("cuda")
            except Exception:
                pass  # scaler stays None, AMP still works without scaling

    def _get_device(self) -> torch.device:
        """Get device from config."""
        if self.cfg.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.cfg.device)

    def validate_config(self) -> None:
        """Validate training config and emit warnings for suboptimal settings."""
        total_params = getattr(self.model, "count_parameters", lambda: {"total": 0})()["total"]
        use_titans = getattr(self.cfg, "use_titans_memory", False)
        seq_len = getattr(self.cfg, "seq_len", 2048)

        if use_titans and total_params < 50_000_000 and seq_len < 512:
            warnings.warn(
                f"use_titans_memory=True on a small model ({total_params:,} params) "
                f"with seq_len={seq_len} may not provide substantial benefit, "
                f"as Titans state resets per batch at this scale. "
                f"Consider disabling Titans for faster training or increasing seq_len.",
                UserWarning,
                stacklevel=2,
            )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation and progress bar."""
        self.model.train()
        total_loss = 0.0
        raw_count = 0
        accum_count = 0
        skipped_batches = 0
        epoch_start = time.time()
        tokens_seen = 0

        self.optimizer.zero_grad()

        # Lazily initialize scheduler now that we know real loader length.
        # This avoids forcing DocumentAwareDataset(lazy=True) to chunk all
        # documents during Trainer construction.
        if self.scheduler is None:
            steps_per_epoch = math.ceil(
                len(self.train_loader) / self.grad_accum_steps
            )
            total_optimizer_steps = steps_per_epoch * self.cfg.epochs
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self._scheduler_warmup,
                num_training_steps=total_optimizer_steps,
                num_cycles=self._scheduler_cycles,
                min_lr_ratio=self._scheduler_min_lr,
            )

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            unit="batch",
            disable=not self.verbose,
        )

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            tokens_seen += input_ids.numel()

            # Get attention_mask from batch
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Build cca_step from global optimizer step (not batch index)
            cca_step = None
            if getattr(self.cfg, "use_cca", False):
                cca_step = self.global_step

            # Forward pass — autocast whenever AMP is enabled (independent of scaler)
            if self.use_amp:
                with torch.amp.autocast(
                    device_type="cuda", dtype=self.amp_dtype
                ):
                    outputs = self.model(
                        input_ids, labels=labels,
                        attention_mask=attention_mask,
                        cca_step=cca_step,
                    )
                    loss = outputs["loss"]
            else:
                outputs = self.model(
                    input_ids, labels=labels,
                    attention_mask=attention_mask,
                    cca_step=cca_step,
                )
                loss = outputs["loss"]

            # Skip NaN/Inf losses (numerical instability)
            if torch.isnan(loss) or torch.isinf(loss):
                skipped_batches += 1
                if skipped_batches <= 5 and self.verbose:
                    print(
                        f"  WARNING: NaN/Inf loss at batch {batch_idx}. "
                        f"Skipping. (Try disabling AMP: use_amp=False)"
                    )
                continue

            # Scale loss for gradient accumulation
            divisor = 1
            if self.grad_accum_steps > 1:
                is_last = (batch_idx + 1) == len(self.train_loader)
                if is_last and accum_count < self.grad_accum_steps - 1:
                    divisor = accum_count + 1
                else:
                    divisor = self.grad_accum_steps
                loss = loss / divisor

            # Backward pass — scale only if scaler exists
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_count += 1
            total_loss += loss.item() * divisor
            raw_count += 1

            # Optimizer step after accumulation
            is_last = (batch_idx + 1) == len(self.train_loader)
            if accum_count >= self.grad_accum_steps or is_last:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                accum_count = 0
                self.global_step += 1

            # Live progress bar update
            avg = total_loss / max(raw_count, 1)
            lr = self.scheduler.get_last_lr()[0]
            elapsed = time.time() - epoch_start
            tok_per_sec = tokens_seen / max(elapsed, 1e-6)
            pbar.set_postfix({
                "loss": f"{avg:.4f}",
                "ppl": f"{compute_perplexity(avg):.2f}",
                "lr": f"{lr:.2e}",
                "tok/s": f"{tok_per_sec:,.0f}",
            })

        avg_loss = total_loss / max(raw_count, 1)
        return {
            "loss": avg_loss,
            "perplexity": compute_perplexity(avg_loss),
            "time": time.time() - epoch_start,
            "skipped_batches": skipped_batches,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set with progress bar.
        
        Uses token-weighted averaging instead of simple batch averaging
        for more accurate perplexity calculation.
        """
        if self.val_loader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        pbar = tqdm(
            self.val_loader,
            desc="Validation",
            unit="batch",
            disable=not self.verbose,
        )
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            if self.use_amp:
                with torch.amp.autocast(
                    device_type="cuda", dtype=self.amp_dtype
                ):
                    outputs = self.model(input_ids, labels=labels, attention_mask=attention_mask)
            else:
                outputs = self.model(input_ids, labels=labels, attention_mask=attention_mask)

            loss = outputs["loss"]
            if not (torch.isnan(loss) or torch.isinf(loss)):
                # Count valid (non -100) tokens for weighting
                valid_tokens = (labels != -100).sum().item()
                
                # Weight loss by token count
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
                num_batches += 1
                
                # Use token-weighted average for display
                avg = total_loss / max(total_tokens, 1)
                pbar.set_postfix({
                    "loss": f"{avg:.4f}",
                    "ppl": f"{compute_perplexity(avg):.2f}",
                })

        # Token-weighted average
        avg_loss = total_loss / max(total_tokens, 1)
        return {"loss": avg_loss, "perplexity": compute_perplexity(avg_loss), "total_tokens": total_tokens}

    @torch.no_grad()
    def generate_sample(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> str:
        """Generate text from a prompt."""
        if self.tokenizer is None:
            return ""
        self.model.eval()
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt)], dtype=torch.long
        ).to(self.device)
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        generated = self.model.generate_ext(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
        )
        new_tokens = generated[0][input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def save_checkpoint(self, epoch: int, filename: Optional[str] = None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"helixlm_epoch_{epoch}.pt"
        path = os.path.join(self.output_dir, filename)
        self.model.save_pretrained(path)
        if self.verbose:
            print(f"Checkpoint saved to {path}")

    def train(
        self, num_epochs: Optional[int] = None, eval_every: int = 1
    ) -> Dict[str, Any]:
        """Train for specified number of epochs."""
        epochs = num_epochs or self.cfg.epochs
        effective_batch = self.cfg.batch_size * self.grad_accum_steps

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training HelixLM on {self.device}")
            print(f"Parameters: {self.model.count_parameters()['total']:,}")
            print(
                f"Epochs: {epochs} | Batch: {self.cfg.batch_size} | "
                f"Accum: {self.grad_accum_steps} | Effective: {effective_batch}"
            )
            print(f"LR: {self.cfg.lr} | AMP: {self.use_amp}")
            print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            if self.verbose:
                print(f"\nEpoch {epoch}/{epochs}")
                print("-" * 40)

            train_metrics = self.train_epoch(epoch)
            skip_info = ""
            if train_metrics.get("skipped_batches", 0) > 0:
                skip_info = f" | Skipped: {train_metrics['skipped_batches']}"
            if self.verbose:
                print(
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"PPL: {train_metrics['perplexity']:.2f} | "
                    f"Time: {format_time(train_metrics['time'])}"
                    f"{skip_info}"
                )
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["perplexity"].append(train_metrics["perplexity"])

            if self.val_loader is not None and epoch % eval_every == 0:
                val_metrics = self.evaluate()
                if self.verbose:
                    print(
                        f"Val Loss: {val_metrics['loss']:.4f} | "
                        f"Val PPL: {val_metrics['perplexity']:.2f}"
                    )
                self.history["val_loss"].append(val_metrics["loss"])
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint(epoch, "best_model")

            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

            if self.tokenizer and epoch % eval_every == 0 and self.verbose:
                print("\nGeneration samples:")
                for prompt in self.example_prompts:
                    if self.generated_example_length:
                        try:
                            generated = self.generate_sample(
                                prompt,
                                max_new_tokens=self.generated_example_length,
                            )
                            print(f"  '{prompt}' -> '{generated}'")
                        except Exception as e:
                            print(f"  '{prompt}' -> [Error: {e}]")
                    else:
                        print(
                            "Parameter 'generated_example_length' set to 0. "
                            "Skipping generation samples."
                        )
                print()

        self.save_checkpoint(epochs, "final_model")
        if self.verbose:
            print(f"\nTraining complete!")
        
        # Cleanup shards if requested (for streaming datasets)
        if self._cleanup_shards:
            import shutil
            for shard_dir in [self._train_shard_dir, self._val_shard_dir]:
                if shard_dir is not None and os.path.exists(shard_dir):
                    try:
                        shutil.rmtree(shard_dir)
                        if self.verbose:
                            print(f"Cleaned up shard cache: {shard_dir}")
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: Failed to cleanup shard cache {shard_dir}: {e}")
        
        return self.history


class PretrainTrainer(Trainer):
    """
    Trainer for causal LM pretraining using continuous token windows.
    Mirrors the data regime of the cofounder's script:
      - Documents concatenated with eos_token
      - Fixed seq_len windows, no padding
      - labels = input_ids (no loss masking)
      - Non-overlapping windows
    Works with both List[str] / Column and IterableColumn inputs.

    If `train_texts` is a streaming iterable (no `__len__`), it will be automatically
    compiled to a disk-backed sample store using `PretrainSampleCompiler`.
    The compiled store is reused across runs if `pretrain_store_dir` is provided;
    otherwise a temporary directory is used (not automatically deleted).

    Args:
        total_optimizer_steps: Optional known number of total optimizer steps.
            If None, a constant LR after warmup is used (min_lr_ratio=1.0).
        min_lr_ratio: Minimum learning rate ratio for cosine decay (only used if total_optimizer_steps is not None).
        buffer_size: Size of shuffle buffer for continuous window dataset.
        count_first: If True, perform a deterministic counting pass before the first
            training epoch to obtain the total number of batches. This enables an exact
            progress bar from epoch 1. If False, the count is learned after epoch 1.
    """
    TRAINING_STATE_VERSION = "helix.pretrain.training-state.v3"
    TRAINING_STATE_FIELDS = frozenset(
        {
            "epoch",
            "global_step",
            "sample_cursor",
            "usable_sample_count",
            "dataset_manifest_sha256",
            "permutation_sha256",
            "permutation_epoch",
            "permutation_seed",
            "validation_source_root",
            "validation_sample_ids",
            "validation_sample_ids_sha256",
            "model",
            "optimizer",
            "best_val_loss",
            "history",
            "scheduler",
            "scheduler_config",
            "training_config",
            "torch_rng_state",
            "cuda_rng_state_all",
            "scaler",
        }
    )

    @staticmethod
    def load_training_state(path) -> Dict[str, Any]:
        """Load and structurally validate a trusted local indexed checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Indexed pretraining checkpoint must contain a state mapping")
        if state.get("format_version") != PretrainTrainer.TRAINING_STATE_VERSION:
            raise ValueError("Unsupported indexed pretraining checkpoint format")
        missing_fields = sorted(PretrainTrainer.TRAINING_STATE_FIELDS - state.keys())
        if missing_fields:
            raise ValueError(
                "Indexed pretraining checkpoint is missing fields: "
                + ", ".join(missing_fields)
            )
        history = state["history"]
        expected_history_keys = {"train_loss", "val_loss", "perplexity"}
        if (
            not isinstance(history, dict)
            or set(history) != expected_history_keys
            or not all(isinstance(values, list) for values in history.values())
        ):
            raise ValueError("Indexed pretraining checkpoint history is malformed")
        return state

    @staticmethod
    def training_state_epoch_complete(state: Dict[str, Any]) -> bool:
        """Return whether a structurally validated state consumed its usable epoch."""
        cursor = int(state["sample_cursor"])
        usable_samples = int(state["usable_sample_count"])
        if cursor < 0 or cursor > usable_samples:
            raise ValueError("Indexed pretraining checkpoint cursor is outside its epoch")
        return cursor == usable_samples

    @staticmethod
    def resume_stage_plan(state: Dict[str, Any], stage_count: int) -> tuple[int, bool]:
        """Return the next LR-stage index and whether exact in-stage restore is needed."""
        stage_index = int(state["permutation_epoch"])
        resume_current_stage = not PretrainTrainer.training_state_epoch_complete(state)
        if not resume_current_stage:
            stage_index += 1
        if stage_index < 0 or stage_index > int(stage_count):
            raise ValueError("Resume checkpoint permutation epoch is outside LR_STAGES")
        return stage_index, resume_current_stage

    def __init__(
        self,
        model,
        cfg,
        train_texts=None,
        val_texts=None,
        val_loader=None,
        val_store_dir=None,
        tokenizer=None,
        output_dir="./checkpoints",
        grad_accum_steps=1,
        use_amp=False,
        amp_dtype="bfloat16",
        buffer_size=50000,
        seed=42,
        num_workers=0,
        total_optimizer_steps=None,
        max_optimizer_steps=None,
        min_lr_ratio=1.0,
        checkpoint_every_steps=None,
        checkpoint_slots: int = 2,
        step_callback: Optional[Callable[[Dict[str, float]], None]] = None,
        eval_every_steps=None,
        validation_batches=None,
        evaluation_callback: Optional[Callable[[Dict[str, float]], None]] = None,
        count_first: bool = False,
        train_store_dir=None,
        pretrain_store_dir=None,
        train_permutation_path=None,
        train_permutation_epoch: int = 0,
        train_cursor: int = 0,
        resume_training_state=None,
        validation_sample_count: int = 0,
        validation_sample_ids=None,
        pretrain_source: Optional[Dict[str, Any]] = None,
        verify_train_store: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        **kwargs,
    ):

      
        verbose = kwargs.pop('verbose', True)
        self.verbose = verbose

        import tempfile
        from torch.utils.data import DataLoader
        from .dataset import ContinuousWindowDataset, collate_continuous, _is_iterable_column
        from .pretrain_data import (
            PretrainIndexedDataset,
            PretrainPermutation,
            PretrainSampleCompiler,
            collate_pretrain_samples,
            create_pretrain_indexed_loader,
        )

        # ── Determine if we need to auto-compile streaming train_texts ──
        auto_compiled_store_dir = None
        if train_store_dir is None and train_texts is not None:
            if _is_iterable_column(train_texts):
                # Streaming iterable -> compile to disk automatically
                # IMPORTANT: the store path must NOT exist yet; the atomic
                # publication helper will create it via rename.
                if pretrain_store_dir:
                    store_dir = pretrain_store_dir
                else:
                    import uuid
                    store_dir = os.path.join(
                        tempfile.gettempdir(),
                        f"helix_pretrain_{uuid.uuid4().hex}"
                    )
                # Atomic store publication
                store_dir = self._ensure_atomic_pretrain_store(
                    texts=train_texts,
                    store_dir=store_dir,
                    tokenizer=tokenizer,
                    seq_len=cfg.seq_len,
                    source=dict(pretrain_source or {"auto": "true"}),
                    verify=verify_train_store,
                )
                train_store_dir = store_dir
                auto_compiled_store_dir = store_dir

        self._auto_compiled_store_dir = auto_compiled_store_dir

        # ── Indexed path ──
        resume_state = None
        if resume_training_state is not None:
            resume_state = self.load_training_state(resume_training_state)
            if train_store_dir is None:
                raise ValueError("resume_training_state requires train_store_dir")
            train_permutation_epoch = int(resume_state["permutation_epoch"])
            train_cursor = int(resume_state["sample_cursor"])

        self._indexed_train = train_store_dir is not None
        self._train_permutation = None
        self._train_cursor = int(train_cursor)
        self._train_store_dir = os.fspath(train_store_dir) if train_store_dir is not None else None
        self._train_seed = int(seed)
        self._initial_train_permutation_epoch = int(train_permutation_epoch)
        resume_train_epoch = int(resume_state["epoch"]) if resume_state is not None else None
        self._train_permutation_base_epoch = (
            int(train_permutation_epoch) - (resume_train_epoch - 1)
            if resume_train_epoch is not None
            else int(train_permutation_epoch)
        )
        if self._train_permutation_base_epoch < 0:
            raise ValueError("Indexed pretraining checkpoint epoch identity mismatch")
        self._train_permutation_epoch = int(train_permutation_epoch)
        self._start_train_epoch = resume_train_epoch or 1
        self._initial_train_permutation_path = train_permutation_path
        self._indexed_num_workers = int(num_workers)
        self._indexed_pin_memory = bool(pin_memory)
        self._indexed_prefetch_factor = int(prefetch_factor)
        self._resume_scheduler_state = None
        self._validation_source_root = self._compute_validation_source_root(
            val_texts=val_texts,
            val_store_dir=val_store_dir,
        )
        self._validation_sample_ids = tuple()
        self._validation_sample_ids_sha256 = None

        if self._indexed_train:
            self._train_dataset = PretrainIndexedDataset(
                train_store_dir,
                verify=verify_train_store,
            )
            if pretrain_source:
                stored_source = self._train_dataset.manifest.value.get("source", {})
                for key, value in pretrain_source.items():
                    if stored_source.get(key) != value:
                        raise ValueError(
                            f"Source identity mismatch for key '{key}': "
                            f"stored {stored_source.get(key)!r} != requested {value!r}"
                        )
            if self._train_dataset.seq_len != cfg.seq_len:
                raise ValueError(
                    "Compiled pretraining seq_len does not match the model config: "
                    f"{self._train_dataset.seq_len} != {cfg.seq_len}"
                )
            if train_permutation_path is None:
                train_permutation_path = os.path.join(
                    train_store_dir,
                    "permutations",
                    f"epoch-{int(train_permutation_epoch):04d}-seed-{int(seed)}.u32",
                )
            if os.path.exists(train_permutation_path):
                self._train_permutation = PretrainPermutation.load(train_permutation_path)
            else:
                self._train_permutation = PretrainPermutation.create(
                    train_permutation_path,
                    len(self._train_dataset),
                    seed,
                    epoch=train_permutation_epoch,
                )
            self._validate_permutation_identity(
                self._train_permutation,
                expected_epoch=train_permutation_epoch,
                expected_seed=seed,
            )
            if resume_state is not None and validation_sample_ids is None:
                validation_sample_ids = resume_state["validation_sample_ids"]
            if validation_sample_ids is not None:
                resolved_validation_ids = tuple(int(value) for value in validation_sample_ids)
                if validation_sample_count not in (0, len(resolved_validation_ids)):
                    raise ValueError(
                        "validation_sample_count does not match validation_sample_ids"
                    )
            elif validation_sample_count:
                validation_sample_count = int(validation_sample_count)
                if validation_sample_count <= 0:
                    raise ValueError("validation_sample_count must be non-negative")
                if validation_sample_count >= len(self._train_dataset):
                    raise ValueError(
                        "validation_sample_count must leave at least one training sample"
                    )
                resolved_validation_ids = tuple(
                    int(value)
                    for value in self._train_permutation.values()[
                        len(self._train_dataset) - validation_sample_count :
                    ]
                )
            else:
                resolved_validation_ids = tuple()
            if len(set(resolved_validation_ids)) != len(resolved_validation_ids):
                raise ValueError("validation_sample_ids must be unique")
            if any(
                sample_id < 0 or sample_id >= len(self._train_dataset)
                for sample_id in resolved_validation_ids
            ):
                raise ValueError("validation_sample_ids contains an out-of-range sample")
            self._validation_sample_ids = resolved_validation_ids
            if resolved_validation_ids:
                validation_id_bytes = np.asarray(
                    resolved_validation_ids,
                    dtype=np.dtype("<u4"),
                ).tobytes(order="C")
                self._validation_sample_ids_sha256 = hashlib.sha256(
                    validation_id_bytes
                ).hexdigest()
                self._validation_source_root = (
                    "indexed-permutation-tail:"
                    + self._validation_sample_ids_sha256
                )
            self.train_loader = create_pretrain_indexed_loader(
                self._train_dataset,
                self._train_permutation,
                cfg.batch_size,
                cursor=self._train_cursor,
                excluded_sample_ids=self._validation_sample_ids,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
            )
            available_training_samples = (
                len(self._train_dataset) - len(self._validation_sample_ids)
            )
            self._usable_sample_count = (
                available_training_samples // int(cfg.batch_size)
            ) * int(cfg.batch_size)
            if self._train_cursor < 0 or self._train_cursor > self._usable_sample_count:
                raise ValueError("Indexed pretraining cursor is outside the usable sample range")
            resume_epoch_complete = (
                resume_state is not None
                and self.training_state_epoch_complete(resume_state)
            )
            if (
                resume_state is not None
                and int(resume_state["usable_sample_count"]) != self._usable_sample_count
            ):
                raise ValueError("Indexed pretraining checkpoint usable sample count mismatch")
            if len(self.train_loader) == 0 and not resume_epoch_complete:
                raise ValueError("Indexed pretraining cursor leaves no complete training batch")
        else:
            if train_texts is None:
                raise ValueError("train_texts or train_store_dir must be provided")
            self._train_dataset = ContinuousWindowDataset(
                train_texts, tokenizer, cfg.seq_len,
                buffer_size=buffer_size, seed=seed, shuffle=True
            )
            self.train_loader = self._make_loader(
                self._train_dataset, cfg.batch_size, num_workers, collate_continuous
            )

        # ── Validation ──
        self.val_loader = None
        if val_loader is not None:
            self.val_loader = val_loader
        elif val_store_dir is not None:
            val_dataset = PretrainIndexedDataset(val_store_dir, verify=verify_train_store)
            if val_dataset.seq_len != cfg.seq_len:
                raise ValueError("Validation store seq_len does not match config")
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=int(cfg.batch_size),
                shuffle=False,
                drop_last=False,
                collate_fn=collate_pretrain_samples,
                num_workers=int(num_workers),
                pin_memory=bool(pin_memory and torch.cuda.is_available()),
            )
        elif self._indexed_train and self._validation_sample_ids:
            from torch.utils.data import Subset

            val_dataset = Subset(
                self._train_dataset,
                list(self._validation_sample_ids),
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=int(cfg.batch_size),
                shuffle=False,
                drop_last=False,
                collate_fn=collate_pretrain_samples,
                num_workers=int(num_workers),
                pin_memory=bool(pin_memory and torch.cuda.is_available()),
            )
        elif val_texts is not None:
            # Always use ContinuousWindowDataset for validation; it works for
            # both materialized and streaming inputs without __len__ issues.
            val_dataset = ContinuousWindowDataset(
                val_texts, tokenizer, cfg.seq_len,
                buffer_size=buffer_size, seed=seed, shuffle=False
            )
            # IMPORTANT: Force 0 workers for validation from val_texts.
            # Iterable continuous‑window validation cannot be sharded safely.
            # Use val_store_dir if you need multi‑worker validation.
            self.val_loader = self._make_loader(
                val_dataset, cfg.batch_size, 0, collate_continuous
            )

        # Call parent with our pre-built loaders
        super().__init__(
            model=model,
            cfg=cfg,
            tokenizer=tokenizer,
            output_dir=output_dir,
            grad_accum_steps=grad_accum_steps,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            **kwargs,
        )

        # Ensure shard cleanup attributes exist
        self._train_shard_dir = None
        self._val_shard_dir = None

        self.total_optimizer_steps = total_optimizer_steps
        self.max_optimizer_steps = (
            int(max_optimizer_steps) if max_optimizer_steps is not None else None
        )
        if self.max_optimizer_steps is not None and self.max_optimizer_steps <= 0:
            raise ValueError("max_optimizer_steps must be positive")
        self.checkpoint_every_steps = (
            int(checkpoint_every_steps)
            if checkpoint_every_steps is not None
            else None
        )
        if self.checkpoint_every_steps is not None and self.checkpoint_every_steps <= 0:
            raise ValueError("checkpoint_every_steps must be positive")
        self.checkpoint_slots = int(checkpoint_slots)
        if self.checkpoint_slots <= 0:
            raise ValueError("checkpoint_slots must be positive")
        self.step_callback = step_callback
        self.eval_every_steps = (
            int(eval_every_steps) if eval_every_steps is not None else None
        )
        if self.eval_every_steps is not None and self.eval_every_steps <= 0:
            raise ValueError("eval_every_steps must be positive")
        self.validation_batches = (
            int(validation_batches) if validation_batches is not None else None
        )
        if self.validation_batches is not None and self.validation_batches <= 0:
            raise ValueError("validation_batches must be positive")
        self.evaluation_callback = evaluation_callback
        self._scheduler_min_lr = min_lr_ratio if total_optimizer_steps is not None else 1.0

        self.count_first = count_first
        self._known_train_batches = len(self.train_loader) if self._indexed_train else None
        self._known_val_batches = None

        if resume_state is not None:
            if resume_state["scheduler_config"] != self._scheduler_config():
                raise ValueError("Indexed pretraining checkpoint scheduler configuration mismatch")
            if resume_state["training_config"] != self._training_config():
                raise ValueError("Indexed pretraining checkpoint training configuration mismatch")
            if (
                resume_state["dataset_manifest_sha256"]
                != self._train_dataset.manifest.manifest_sha256
            ):
                raise ValueError("Indexed pretraining checkpoint dataset manifest mismatch")
            if (
                resume_state["permutation_sha256"]
                != self._train_permutation.metadata["sha256"]
            ):
                raise ValueError("Indexed pretraining checkpoint permutation mismatch")
            if resume_state["validation_source_root"] != self._validation_source_root:
                raise ValueError("Indexed pretraining checkpoint validation source mismatch")
            if (
                resume_state["validation_sample_ids_sha256"]
                != self._validation_sample_ids_sha256
            ):
                raise ValueError("Indexed pretraining checkpoint validation identity mismatch")
            self.model.load_state_dict(resume_state["model"])
            self.optimizer.load_state_dict(resume_state["optimizer"])
            self.global_step = int(resume_state["global_step"])
            self.best_val_loss = float(resume_state["best_val_loss"])
            self.history = {
                key: list(values)
                for key, values in resume_state["history"].items()
            }
            torch.set_rng_state(resume_state["torch_rng_state"])
            saved_cuda_rng = resume_state["cuda_rng_state_all"]
            current_cuda = self.device.type == "cuda"
            if current_cuda != (saved_cuda_rng is not None):
                raise ValueError("Indexed pretraining checkpoint CUDA RNG configuration mismatch")
            if current_cuda:
                torch.cuda.set_rng_state_all(saved_cuda_rng)
            saved_scaler = resume_state["scaler"]
            if (self.scaler is not None) != (saved_scaler is not None):
                raise ValueError("Indexed pretraining checkpoint AMP scaler configuration mismatch")
            if self.scaler is not None:
                self.scaler.load_state_dict(saved_scaler)
            self._resume_scheduler_state = resume_state.get("scheduler")
            if resume_epoch_complete:
                self._activate_indexed_epoch(
                    int(resume_state["permutation_epoch"]) + 1,
                )
                self._start_train_epoch = int(resume_state["epoch"]) + 1

    # ------------------------------------------------------------------
    # Atomic store publication helpers
    # ------------------------------------------------------------------
    def _ensure_atomic_pretrain_store(
        self,
        texts,
        store_dir: str,
        tokenizer,
        seq_len: int,
        source: Dict[str, Any],
        verify: bool,
    ) -> str:
        """
        Ensure that `store_dir` contains a valid, verified pretrain sample store.
        If it already exists and is valid, reuse it. If it doesn't exist, compile
        atomically via a temporary sibling directory and rename.
        """
        from .pretrain_data import PretrainDatasetManifest, PretrainSampleCompiler

        store_path = os.fspath(store_dir)

        # Check if store_dir already exists
        if os.path.exists(store_path):
            try:
                manifest = PretrainDatasetManifest.load(store_path)
                if verify:
                    manifest.verify()
                if manifest.seq_len != seq_len:
                    raise ValueError(
                        f"Existing store seq_len {manifest.seq_len} != requested {seq_len}"
                    )
                # Optionally: verify source identity if provided
                if source:
                    stored_source = manifest.value.get("source", {})
                    for k, v in source.items():
                        if stored_source.get(k) != v:
                            raise ValueError(
                                f"Source identity mismatch for key '{k}': "
                                f"stored {stored_source.get(k)!r} != requested {v!r}"
                            )
                if self.verbose:
                    print("[PretrainStore] REUSING_VERIFIED_STORE:", store_path)
                return store_path
            except Exception as e:
                if self.verbose:
                    print("[PretrainStore] REFUSED_STORE_IDENTITY_MISMATCH:", store_path, "-", str(e))
                raise

        # Store doesn't exist -> compile atomically
        if self.verbose:
            print("[PretrainStore] COMPILING_NEW_STORE:", store_path)
        parent_dir = os.path.dirname(store_path) or "."
        os.makedirs(parent_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp(
            prefix=os.path.basename(store_path) + ".building-",
            dir=parent_dir,
        )
        # The compiler expects to create its output directory itself, so remove
        # the empty staging directory we just created.
        os.rmdir(temp_dir)
        try:
            compiler = PretrainSampleCompiler(
                tokenizer, seq_len, temp_dir,
                source=source,
            )
            compiler.compile(texts)
            if verify:
                manifest = PretrainDatasetManifest.load(temp_dir)
                manifest.verify()
            os.rename(temp_dir, store_path)
            if self.verbose:
                print("[PretrainStore] Store published atomically to:", store_path)
            return store_path
        except Exception as e:
            if self.verbose:
                print("[PretrainStore] INCOMPLETE_PRETRAIN_STORE: compilation failed, leaving temp dir:", temp_dir)
                print("                  Error:", str(e))
            # Do not delete temp_dir; leave for inspection
            raise

    @staticmethod
    def _compute_validation_source_root(
        val_texts=None,
        val_store_dir=None,
    ) -> Optional[str]:
        """
        Compute a stable identity string for the validation source.
        Used for checkpoint binding. Returns None if no validation source.
        """
        if val_store_dir is not None:
            from .pretrain_data import PretrainDatasetManifest
            manifest = PretrainDatasetManifest.load(val_store_dir)
            return f"store:{manifest.manifest_sha256}"
        if val_texts is not None:
            # Hash the repr of the list (or iterable? For list-like only)
            if hasattr(val_texts, "__len__") or isinstance(val_texts, list):
                # Simple deterministic hash
                try:
                    text_repr = repr(list(val_texts))
                except TypeError:
                    text_repr = repr(val_texts)
                digest = hashlib.sha256(text_repr.encode("utf-8")).hexdigest()
                return f"texts:{digest}"
            else:
                # For streaming validation, no stable hash without consuming;
                # we cannot bind precisely, but we can use a placeholder.
                return "streaming:unknown"
        return None

    @staticmethod
    def _make_loader(dataset, batch_size, num_workers, collate_fn):
        from torch.utils.data import DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            shuffle=False,  # dataset handles shuffling
        )

    def _count_batches(self, loader):
        """Returns the number of batches in the loader without training."""
        count = 0
        for _ in loader:
            count += 1
        return count

    def _scheduler_config(self) -> Dict[str, Any]:
        return {
            "total_optimizer_steps": (
                int(self.total_optimizer_steps)
                if self.total_optimizer_steps is not None
                else None
            ),
            "warmup_steps": int(self._scheduler_warmup),
            "cycles": float(self._scheduler_cycles),
            "min_lr_ratio": float(self._scheduler_min_lr),
        }

    def _training_config(self) -> Dict[str, Any]:
        return {
            "learning_rate": float(self.cfg.lr),
            "weight_decay": float(self.cfg.weight_decay),
            "optimizer_betas": [0.9, 0.999],
            "batch_size": int(self.cfg.batch_size),
            "grad_accum_steps": int(self.grad_accum_steps),
            "grad_clip": float(self.cfg.grad_clip),
            "use_amp": bool(self.use_amp),
            "amp_dtype": str(self.amp_dtype),
            "device_type": self.device.type,
            "max_optimizer_steps": self.max_optimizer_steps,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "eval_every_steps": self.eval_every_steps,
            "validation_batches": self.validation_batches,
        }

    @staticmethod
    def _validate_permutation_identity(
        permutation,
        *,
        expected_epoch: int,
        expected_seed: int,
    ) -> None:
        if int(permutation.metadata["epoch"]) != int(expected_epoch):
            raise ValueError("Indexed pretraining permutation epoch mismatch")
        if int(permutation.metadata["seed"]) != int(expected_seed):
            raise ValueError("Indexed pretraining permutation seed mismatch")

    def _activate_indexed_epoch(
        self,
        permutation_epoch: int,
        *,
        cursor: int = 0,
    ) -> None:
        """Activate one persisted global order for an indexed pretraining epoch."""
        from .pretrain_data import PretrainPermutation, create_pretrain_indexed_loader

        if not self._indexed_train:
            raise ValueError("Indexed epoch activation requires train_store_dir")
        if (
            permutation_epoch == self._initial_train_permutation_epoch
            and self._initial_train_permutation_path
        ):
            permutation_path = os.fspath(self._initial_train_permutation_path)
        else:
            permutation_path = os.path.join(
                self._train_store_dir,
                "permutations",
                f"epoch-{int(permutation_epoch):04d}-seed-{self._train_seed}.u32",
            )
        if os.path.exists(permutation_path):
            permutation = PretrainPermutation.load(permutation_path)
        else:
            permutation = PretrainPermutation.create(
                permutation_path,
                len(self._train_dataset),
                self._train_seed,
                epoch=permutation_epoch,
            )
        self._validate_permutation_identity(
            permutation,
            expected_epoch=permutation_epoch,
            expected_seed=self._train_seed,
        )
        self._train_permutation = permutation
        self._train_permutation_epoch = int(permutation_epoch)
        self._train_cursor = int(cursor)
        self.train_loader = create_pretrain_indexed_loader(
            self._train_dataset,
            permutation,
            self.cfg.batch_size,
            cursor=self._train_cursor,
            excluded_sample_ids=self._validation_sample_ids,
            num_workers=self._indexed_num_workers,
            drop_last=True,
            pin_memory=self._indexed_pin_memory,
            prefetch_factor=self._indexed_prefetch_factor,
        )
        if len(self.train_loader) == 0:
            raise ValueError("Indexed pretraining cursor leaves no complete training batch")
        self._known_train_batches = len(self.train_loader)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Override train_epoch to avoid relying on len(self.train_loader).
        Handles gradient accumulation correctly for any number of micro-batches.
        """
        if self._indexed_train:
            desired_epoch = self._train_permutation_base_epoch + int(epoch) - 1
            if desired_epoch != self._train_permutation_epoch:
                self._activate_indexed_epoch(desired_epoch)

        # Determine total batches for progress bar
        total_batches = None
        if self._known_train_batches is not None:
            total_batches = self._known_train_batches
        elif self.count_first:
            # Counting pass before first epoch
            if self.verbose:
                print("Counting training batches (first pass)...")
            count = self._count_batches(self.train_loader)
            # Recreate loader for actual training
            self.train_loader = self._make_loader(
                self._train_dataset,
                self.cfg.batch_size,
                self.train_loader.num_workers,
                self.train_loader.collate_fn,
            )
            self._known_train_batches = count
            total_batches = count

        self.model.train()
        total_loss = 0.0
        raw_count = 0
        accum_count = 0
        skipped_batches = 0
        epoch_start = time.time()
        tokens_seen = 0
        causal_targets_seen = 0
        group_loss_weighted = 0.0
        group_targets = 0
        group_started = time.perf_counter()
        step_limit_reached = False
        batch_idx = -1

        self.optimizer.zero_grad(set_to_none=True)

        # Initialize scheduler without relying on len
        if self.scheduler is None:
            total_steps = self.total_optimizer_steps if self.total_optimizer_steps is not None else 10**9
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self._scheduler_warmup,
                num_training_steps=total_steps,
                num_cycles=self._scheduler_cycles,
                min_lr_ratio=self._scheduler_min_lr,
            )
            if self._resume_scheduler_state is not None:
                self.scheduler.load_state_dict(self._resume_scheduler_state)
                self._resume_scheduler_state = None

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            unit="batch",
            disable=not self.verbose,
            total=total_batches,   # None if unknown, exact if known
        )

        for batch_idx, batch in enumerate(pbar):
            if (
                self.max_optimizer_steps is not None
                and self.global_step >= self.max_optimizer_steps
            ):
                step_limit_reached = True
                break
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            if self._indexed_train:
                self._train_cursor += int(input_ids.shape[0])
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            cca_step = None
            if getattr(self.cfg, "use_cca", False):
                cca_step = self.global_step

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids, labels=labels,
                        attention_mask=attention_mask,
                        cca_step=cca_step,
                    )
                    loss = outputs["loss"]
            else:
                outputs = self.model(
                    input_ids, labels=labels,
                    attention_mask=attention_mask,
                    cca_step=cca_step,
                )
                loss = outputs["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                skipped_batches += 1
                if skipped_batches <= 5 and self.verbose:
                    print(f"  WARNING: NaN/Inf loss at batch {batch_idx}. Skipping.")
                continue

            # Backward without scaling; we will divide gradients manually
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_count += 1
            total_loss += loss.item()
            raw_count += 1
            tokens_seen += input_ids.numel()
            causal_targets = int((labels[:, 1:] != -100).sum().item())
            causal_targets_seen += causal_targets
            group_targets += causal_targets
            group_loss_weighted += float(loss.detach()) * causal_targets

            # Optimizer step after accumulation
            if accum_count == self.grad_accum_steps:
                lr = float(self.optimizer.param_groups[0]["lr"])
                grad_norm = self._step(accum_count)
                step_seconds = time.perf_counter() - group_started
                elapsed = max(time.time() - epoch_start, 1e-6)
                if self.step_callback is not None:
                    self.step_callback(
                        {
                            "global_step": float(self.global_step),
                            "loss": group_loss_weighted / max(group_targets, 1),
                            "perplexity": compute_perplexity(
                                group_loss_weighted / max(group_targets, 1)
                            ),
                            "lr": lr,
                            "grad_norm": grad_norm,
                            "step_seconds": step_seconds,
                            "causal_targets_step": float(group_targets),
                            "causal_targets_total": float(causal_targets_seen),
                            "causal_targets_per_second_step": (
                                group_targets / max(step_seconds, 1e-6)
                            ),
                            "causal_targets_per_second_session": (
                                causal_targets_seen / elapsed
                            ),
                            "tokens_total": float(tokens_seen),
                            "sample_cursor": float(self._train_cursor),
                            "skipped_batches": float(skipped_batches),
                            "vram_allocated_bytes": float(
                                torch.cuda.memory_allocated()
                                if self.device.type == "cuda"
                                else 0
                            ),
                            "vram_reserved_bytes": float(
                                torch.cuda.memory_reserved()
                                if self.device.type == "cuda"
                                else 0
                            ),
                            "peak_vram_bytes": float(
                                torch.cuda.max_memory_allocated()
                                if self.device.type == "cuda"
                                else 0
                            ),
                        }
                    )
                if (
                    self.checkpoint_every_steps is not None
                    and self.global_step % self.checkpoint_every_steps == 0
                ):
                    slot = (
                        self.global_step // self.checkpoint_every_steps
                    ) % self.checkpoint_slots
                    self.save_checkpoint(epoch, f"latest-{slot}")
                if (
                    self.val_loader is not None
                    and self.eval_every_steps is not None
                    and self.global_step % self.eval_every_steps == 0
                ):
                    validation_metrics = self.evaluate(
                        max_batches=self.validation_batches
                    )
                    self.model.train()
                    if self.evaluation_callback is not None:
                        self.evaluation_callback(validation_metrics)
                accum_count = 0
                group_loss_weighted = 0.0
                group_targets = 0
                group_started = time.perf_counter()
                if (
                    self.max_optimizer_steps is not None
                    and self.global_step >= self.max_optimizer_steps
                ):
                    step_limit_reached = True
                    break

            # Progress bar update
            avg = total_loss / max(raw_count, 1)
            lr = self.scheduler.get_last_lr()[0]
            elapsed = time.time() - epoch_start
            tok_per_sec = tokens_seen / max(elapsed, 1e-6)
            pbar.set_postfix({
                "loss": f"{avg:.4f}",
                "ppl": f"{compute_perplexity(avg):.2f}",
                "lr": f"{lr:.2e}",
                "tok/s": f"{tok_per_sec:,.0f}",
            })

        # End of epoch: handle leftover accumulation (partial group)
        if accum_count > 0 and not step_limit_reached:
            self._step(accum_count)

        # After first epoch, store actual batch count
        if self._known_train_batches is None:
            self._known_train_batches = max(batch_idx + 1, 0)

        avg_loss = total_loss / max(raw_count, 1)
        return {
            "loss": avg_loss,
            "perplexity": compute_perplexity(avg_loss),
            "time": time.time() - epoch_start,
            "skipped_batches": skipped_batches,
            "causal_targets": causal_targets_seen,
            "step_limit_reached": step_limit_reached,
        }

    def save_checkpoint(self, epoch: int, filename: Optional[str] = None):
        """Save model state plus the exact indexed-data cursor when applicable."""
        super().save_checkpoint(epoch, filename)
        if not self._indexed_train:
            return
        checkpoint_name = filename or f"helixlm_epoch_{epoch}.pt"
        checkpoint_dir = os.path.join(self.output_dir, checkpoint_name)
        state = {
            "format_version": "helix.pretrain.cursor.v1",
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "sample_cursor": int(self._train_cursor),
            "usable_sample_count": int(self._usable_sample_count),
            "dataset_manifest_sha256": self._train_dataset.manifest.manifest_sha256,
            "permutation_sha256": self._train_permutation.metadata["sha256"],
            "permutation_epoch": int(self._train_permutation.metadata["epoch"]),
            "permutation_seed": int(self._train_permutation.metadata["seed"]),
            "validation_source_root": self._validation_source_root,
            "validation_sample_ids": list(self._validation_sample_ids),
            "validation_sample_ids_sha256": self._validation_sample_ids_sha256,
        }
        path = os.path.join(checkpoint_dir, "pretrain_data_state.json")
        with open(path + ".tmp", "w", encoding="utf-8") as handle:
            json.dump(state, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(path + ".tmp", path)

        training_state = {
            **state,
            "format_version": self.TRAINING_STATE_VERSION,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_loss": float(self.best_val_loss),
            "history": self.history,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scheduler_config": self._scheduler_config(),
            "training_config": self._training_config(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
            ),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
        }
        training_state_path = os.path.join(checkpoint_dir, "pretrain_training_state.pt")
        torch.save(training_state, training_state_path + ".tmp")
        os.replace(training_state_path + ".tmp", training_state_path)

    def evaluate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Override evaluate to avoid len(self.val_loader) (IterableDataset has no len).
        Uses token-weighted averaging and a tqdm progress bar.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_samples = 0
        num_batches = 0
        batch_idx = -1

        # Determine total batches for progress bar (None if unknown)
        total_batches = self._known_val_batches

        pbar = tqdm(
            self.val_loader,
            desc="Validation",
            unit="batch",
            disable=not self.verbose,
            total=total_batches,
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                        outputs = self.model(input_ids, labels=labels, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_ids, labels=labels, attention_mask=attention_mask)

                loss = outputs["loss"]
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    valid_tokens = (labels[:, 1:] != -100).sum().item()
                    total_loss += loss.item() * valid_tokens
                    total_tokens += valid_tokens
                    total_samples += int(input_ids.shape[0])
                    num_batches += 1

                # Update progress bar
                avg = total_loss / max(total_tokens, 1)
                pbar.set_postfix({
                    "loss": f"{avg:.4f}",
                    "ppl": f"{compute_perplexity(avg):.2f}",
                })

        # Store actual batch count for future evaluations
        if self._known_val_batches is None:
            self._known_val_batches = max(batch_idx + 1, 0)

        avg_loss = total_loss / max(total_tokens, 1)
        return {
            "loss": avg_loss,
            "perplexity": compute_perplexity(avg_loss),
            "causal_targets": total_tokens,
            "sample_count": total_samples,
        }

    def _step(self, group_size: int):
        """
        Divide accumulated gradients by the actual number of micro-batches in the group,
        then clip, optimizer step, scheduler step, zero_grad.
        """
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.div_(group_size)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.grad_clip,
        )

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        return float(grad_norm)

    def train(self, num_epochs: Optional[int] = None, eval_every: int = 1) -> Dict[str, Any]:
        """
        Override train to avoid using boolean context on self.val_loader.
        """
        epochs = num_epochs or self.cfg.epochs
        start_epoch = self._start_train_epoch if self._indexed_train else 1
        if start_epoch > epochs:
            return self.history
        effective_batch = self.cfg.batch_size * self.grad_accum_steps

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training HelixLM on {self.device}")
            print(f"Parameters: {self.model.count_parameters()['total']:,}")
            print(
                f"Epochs: {epochs} | Batch: {self.cfg.batch_size} | "
                f"Accum: {self.grad_accum_steps} | Effective: {effective_batch}"
            )
            print(f"LR: {self.cfg.lr} | AMP: {self.use_amp}")
            print(f"{'='*60}\n")

        final_epoch = start_epoch
        for epoch in range(start_epoch, epochs + 1):
            final_epoch = epoch
            if self.verbose:
                print(f"\nEpoch {epoch}/{epochs}")
                print("-" * 40)

            train_metrics = self.train_epoch(epoch)
            skip_info = ""
            if train_metrics.get("skipped_batches", 0) > 0:
                skip_info = f" | Skipped: {train_metrics['skipped_batches']}"
            if self.verbose:
                print(
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"PPL: {train_metrics['perplexity']:.2f} | "
                    f"Time: {format_time(train_metrics['time'])}"
                    f"{skip_info}"
                )
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["perplexity"].append(train_metrics["perplexity"])

            if self.val_loader is not None and epoch % eval_every == 0:
                val_metrics = self.evaluate()
                if self.verbose:
                    print(
                        f"Val Loss: {val_metrics['loss']:.4f} | "
                        f"Val PPL: {val_metrics['perplexity']:.2f}"
                    )
                self.history["val_loss"].append(val_metrics["loss"])
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint(epoch, "best_model")

            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

            if self.tokenizer and epoch % eval_every == 0 and self.verbose:
                print("\nGeneration samples:")
                for prompt in self.example_prompts:
                    if self.generated_example_length:
                        try:
                            generated = self.generate_sample(
                                prompt,
                                max_new_tokens=self.generated_example_length,
                            )
                            print(f"  '{prompt}' -> '{generated}'")
                        except Exception as e:
                            print(f"  '{prompt}' -> [Error: {e}]")
                print()

            if train_metrics.get("step_limit_reached"):
                break

        self.save_checkpoint(final_epoch, "final_model")
        if self.verbose:
            print(f"\nTraining complete!")
        return self.history


# Alias for supervised fine-tuning (legacy Trainer)
SFTTrainer = Trainer
