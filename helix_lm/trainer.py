"""
HelixLM Trainer with gradient accumulation and configurable AMP.

Key features:
  - Gradient accumulation for effective larger batch sizes
  - Configurable AMP (default: off for stability on small models)
  - NaN/Inf detection and batch skipping
  - Scheduler steps count optimizer steps, not raw batches
  - Uses DocumentAwareDataset (no cross-document boundary crossings)
  - Modern torch.amp API (not deprecated torch.cuda.amp)
"""
import os
import math
import time
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import HelixConfig
from .hf_model import HelixForCausalLM
from .dataset import create_document_loader


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
    """Trainer for HelixLM with gradient accumulation and configurable AMP."""

    def __init__(
        self,
        model: HelixForCausalLM,
        cfg: HelixConfig,
        train_texts: List[str],
        val_texts: Optional[List[str]] = None,
        tokenizer=None,
        output_dir: str = "./checkpoints",
        example_prompts: Optional[List[str]] = None,
        generated_example_length: int = 15,
        grad_accum_steps: int = 1,
        use_amp: bool = False,
        min_tail_len: Optional[int] = None,
    ):
        """
        Initialize Trainer.

        Args:
            model: HelixForCausalLM instance.
            cfg: HelixConfig with training hyperparameters.
            train_texts: List of training document texts.
            val_texts: List of validation document texts (optional).
            tokenizer: Tokenizer instance.
            output_dir: Directory to save checkpoints.
            example_prompts: Prompts for generation samples during training.
            generated_example_length: Number of tokens to generate for samples.
            grad_accum_steps: Gradient accumulation steps (default: 1).
                              Effective batch size = batch_size * grad_accum_steps.
            use_amp: Whether to use torch.amp automatic mixed precision.
                     Default: False (recommended for models < 100M params).
                     Enable for XL/XXL on Ampere+ GPUs.
            min_tail_len: Minimum tail length for DocumentAwareDataset.
                          Default: cfg.seq_len // 4 for pretraining.
                          Use 1 for instruction tuning.
        """
        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.use_amp = use_amp and torch.cuda.is_available()

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

        # Use DocumentAwareDataset (no boundary crossings, 100% token utilization)
        self.train_loader = create_document_loader(
            train_texts,
            tokenizer,
            cfg.seq_len,
            cfg.batch_size,
            shuffle=True,
            min_tail_len=min_tail_len,
        )
        self.val_loader = None
        if val_texts:
            self.val_loader = create_document_loader(
                val_texts,
                tokenizer,
                cfg.seq_len,
                cfg.batch_size,
                shuffle=False,
                drop_last=False,
                min_tail_len=min_tail_len,
            )

        # AdamW with standard betas (0.9, 0.999)
        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler steps count optimizer steps, not raw batches
        steps_per_epoch = math.ceil(
            len(self.train_loader) / self.grad_accum_steps
        )
        total_optimizer_steps = steps_per_epoch * cfg.epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=max(1, cfg.warmup_steps // self.grad_accum_steps),
            num_training_steps=total_optimizer_steps,
        )

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

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        raw_count = 0
        accum_count = 0
        skipped_batches = 0
        epoch_start = time.time()

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            else:
                outputs = self.model(input_ids, labels=labels)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

            # Skip NaN/Inf losses (numerical instability)
            if torch.isnan(loss) or torch.isinf(loss):
                skipped_batches += 1
                if skipped_batches <= 5:  # Only warn first few
                    print(
                        f"  WARNING: NaN/Inf loss at batch {batch_idx}. "
                        f"Skipping. (Try disabling AMP: use_amp=False)"
                    )
                continue

            # Scale loss for gradient accumulation
            if self.grad_accum_steps > 1:
                loss = loss / self.grad_accum_steps

            # Backward pass
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_count += 1
            total_loss += loss.item() * max(1, self.grad_accum_steps)
            raw_count += 1

            # Optimizer step after accumulation
            is_last = (batch_idx + 1) == len(self.train_loader)
            if accum_count >= self.grad_accum_steps or is_last:
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

            # Logging
            if batch_idx % 10 == 0:
                lr = self.scheduler.get_last_lr()[0]
                avg = total_loss / max(raw_count, 1)
                print(
                    f"  Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {avg:.4f} | PPL: {compute_perplexity(avg):.2f} | "
                    f"LR: {lr:.2e}"
                )

        avg_loss = total_loss / max(raw_count, 1)
        return {
            "loss": avg_loss,
            "perplexity": compute_perplexity(avg_loss),
            "time": time.time() - epoch_start,
            "skipped_batches": skipped_batches,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    outputs = self.model(input_ids, labels=labels)
            else:
                outputs = self.model(input_ids, labels=labels)

            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1

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
            filename = f"helixlm_epoch_{epoch}"
        path = os.path.join(self.output_dir, filename)
        self.model.save_pretrained(path)
        print(f"Checkpoint saved to {path}")

    def train(
        self, num_epochs: Optional[int] = None, eval_every: int = 1
    ) -> Dict[str, Any]:
        """Train for specified number of epochs."""
        epochs = num_epochs or self.cfg.epochs
        effective_batch = self.cfg.batch_size * self.grad_accum_steps

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
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 40)

            train_metrics = self.train_epoch(epoch)
            skip_info = ""
            if train_metrics.get("skipped_batches", 0) > 0:
                skip_info = f" | Skipped: {train_metrics['skipped_batches']}"
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

            if self.tokenizer and epoch % eval_every == 0:
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
        print(f"\nTraining complete!")
        return self.history
