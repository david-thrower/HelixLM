"""
HelixLM Trainer with gradient accumulation, configurable AMP, and progress bars.

Key fixes in this revision:
  * Added num_workers and pin_memory args (defaults: num_workers=min(4, cpu_count),
    pin_memory=True when CUDA available).
  * All .to(device) calls use non_blocking=True for async H2D transfers.
  * AMP loss reporting fixed: raw_loss captured BEFORE scaler.scale(loss).
    total_loss accumulates raw_loss.item() WITHOUT * divisor.
  * REMOVED all len(self.train_loader) calls to prevent eager chunking of lazy
    map-style datasets (DocumentAwareDataset) at epoch start.
  * tqdm wraps iter(self.train_loader) / enumerate(self.train_loader) to suppress
    implicit len() calls that would trigger the same eager chunking.
  * Scheduler uses cfg.steps_per_epoch if provided, otherwise a conservative default.
  * End-of-epoch gradient flush handles incomplete accumulation windows.
  * Uses DocumentAwareDataset (no cross-document boundary crossings).
  * Modern torch.amp API (not deprecated torch.cuda.amp).
  * Live tqdm progress bars with loss, PPL, LR, and throughput metrics.
  * Optional train/val DataLoader injection for custom dataset pipelines.
"""
import os
import math
import time
import warnings
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import HelixConfig
from .hf_model import HelixForCausalLM
from .dataset import create_document_loader, create_streaming_loader


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
        train_texts: Optional[List[str]] = None,
        val_texts: Optional[List[str]] = None,
        tokenizer=None,
        output_dir: str = "./checkpoints",
        example_prompts: Optional[List[str]] = None,
        generated_example_length: int = 15,
        grad_accum_steps: int = 1,
        use_amp: bool = False,
        min_tail_len: Optional[int] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True,
        num_workers: int = -1,
        pin_memory: bool = True,
    ):
        """
        Initialize Trainer.

        Args:
            model: HelixForCausalLM instance.
            cfg: HelixConfig with training hyperparameters.
            train_texts: List of training document texts (used if train_loader not provided).
            val_texts: List of validation document texts (used if val_loader not provided).
            tokenizer: Tokenizer instance.
            output_dir: Directory to save checkpoints.
            example_prompts: Prompts for generation samples during training.
            generated_example_length: Number of tokens to generate for samples.
            grad_accum_steps: Gradient accumulation steps (default: 1).
            use_amp: Whether to use torch.amp automatic mixed precision.
            min_tail_len: Minimum tail length for DocumentAwareDataset.
            train_loader: Optional custom DataLoader to override built-in dataset creation.
            val_loader: Optional custom DataLoader to override built-in dataset creation.
            verbose: Whether to show tqdm progress bars and print logs.
            num_workers: DataLoader workers. -1 = auto (min(4, cpu_count)).
            pin_memory: Use pinned memory for async CPU->GPU transfer.
        """
        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.verbose = verbose

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

        # Determine num_workers
        if num_workers < 0:
            import multiprocessing
            num_workers = min(4, multiprocessing.cpu_count())
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()

        # Validate config
        self.validate_config()

        # Data loaders: use injected loaders if provided, otherwise build from texts
        if train_loader is not None:
            self.train_loader = train_loader
        else:
            if train_texts is None:
                raise ValueError("Either train_loader or train_texts must be provided.")
            self.train_loader = create_document_loader(
                train_texts,
                tokenizer,
                cfg.seq_len,
                cfg.batch_size,
                shuffle=True,
                min_tail_len=min_tail_len,
                lazy=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        self.val_loader = None
        if val_loader is not None:
            self.val_loader = val_loader
        elif val_texts is not None:
            self.val_loader = create_document_loader(
                val_texts,
                tokenizer,
                cfg.seq_len,
                cfg.batch_size,
                shuffle=False,
                drop_last=False,
                min_tail_len=min_tail_len,
                lazy=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        # AdamW with standard betas (0.9, 0.999)
        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler steps count optimizer steps, not raw batches.
        # DEFERRED: avoid len() on IterableDataset or lazy datasets at init time.
        self._scheduler_warmup = max(1, cfg.warmup_steps // self.grad_accum_steps)
        self._scheduler_cycles = 0.5
        self._scheduler_min_lr = 0.1
        self.scheduler = None

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "perplexity": []}

        # GradScaler for AMP (only if use_amp=True and CUDA available)
        self.scaler = None
        if self.use_amp:
            try:
                from torch.amp import GradScaler
                self.scaler = GradScaler("cuda")
            except Exception:
                self.use_amp = False

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

    def _init_scheduler(self, steps_per_epoch: Optional[int] = None) -> None:
        """Lazy scheduler initialization. Never calls len(train_loader)."""
        if self.scheduler is not None:
            return

        if steps_per_epoch is None:
            # No len() call — avoids eager chunking on lazy map datasets
            # and is required for IterableDataset.
            # Use cfg attribute if available, otherwise conservative default.
            steps_per_epoch = getattr(self.cfg, "steps_per_epoch", None)
            if steps_per_epoch is None:
                steps_per_epoch = 1000

        total_optimizer_steps = steps_per_epoch * self.cfg.epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self._scheduler_warmup,
            num_training_steps=total_optimizer_steps,
            num_cycles=self._scheduler_cycles,
            min_lr_ratio=self._scheduler_min_lr,
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

        # Lazily initialize scheduler. We NEVER call len(self.train_loader) here
        # to prevent eager chunking of lazy map-style datasets at epoch start.
        if self.scheduler is None:
            self._init_scheduler(steps_per_epoch=None)

        # Use iter() to suppress tqdm's implicit len() call, which would
        # trigger DocumentAwareDataset.__len__() -> _ensure_chunks() eagerly.
        pbar = tqdm(
            iter(self.train_loader),
            desc=f"Epoch {epoch}",
            unit="batch",
            disable=not self.verbose,
        )

        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device, non_blocking=self.pin_memory)
            labels = batch["labels"].to(self.device, non_blocking=self.pin_memory)
            tokens_seen += input_ids.numel()

            # Forward pass
            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs["loss"]
            else:
                outputs = self.model(input_ids, labels=labels)
                loss = outputs["loss"]

            # Capture raw loss BEFORE any scaling/division for accurate reporting.
            # This is the TRUE, unscaled loss value.
            raw_loss = loss.detach().clone()

            # Skip NaN/Inf losses (numerical instability)
            if torch.isnan(raw_loss) or torch.isinf(raw_loss):
                skipped_batches += 1
                if skipped_batches <= 5 and self.verbose:
                    print(
                        f"  WARNING: NaN/Inf loss at batch {raw_count}. "
                        f"Skipping. (Try disabling AMP: use_amp=False)"
                    )
                continue

            # Scale loss for gradient accumulation (gradients are averaged over window)
            loss = loss / self.grad_accum_steps

            # Backward pass
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_count += 1
            # raw_loss is the FULL loss (before division), so do NOT multiply by divisor.
            total_loss += raw_loss.item()
            raw_count += 1

            # Optimizer step after accumulation window is full
            if accum_count >= self.grad_accum_steps:
                if self.use_amp and self.scaler is not None:
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

        # End-of-epoch flush: if any gradients remain unstepped, step now.
        # The gradients were divided by grad_accum_steps; if accum_count < grad_accum_steps,
        # the update is slightly weaker than a full window. This is standard practice.
        if accum_count > 0:
            if self.use_amp and self.scaler is not None:
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

        avg_loss = total_loss / max(raw_count, 1)
        return {
            "loss": avg_loss,
            "perplexity": compute_perplexity(avg_loss),
            "time": time.time() - epoch_start,
            "skipped_batches": skipped_batches,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set with progress bar."""
        if self.val_loader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # iter() suppresses tqdm's implicit len() call
        pbar = tqdm(
            iter(self.val_loader),
            desc="Validation",
            unit="batch",
            disable=not self.verbose,
        )
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device, non_blocking=self.pin_memory)
            labels = batch["labels"].to(self.device, non_blocking=self.pin_memory)

            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    outputs = self.model(input_ids, labels=labels)
            else:
                outputs = self.model(input_ids, labels=labels)

            loss = outputs["loss"]
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1
                avg = total_loss / max(num_batches, 1)
                pbar.set_postfix({
                    "loss": f"{avg:.4f}",
                    "ppl": f"{compute_perplexity(avg):.2f}",
                })

        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss, "perplexity": compute_perplexity(avg_loss)}

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

            if self.val_loader and epoch % eval_every == 0:
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
        return self.history
