"""
HelixLM Trainer with best practices.
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
from .dataset import create_helix_dataloader


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int,
                                     num_cycles: float = 0.5, min_lr_ratio: float = 0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)


def compute_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20))


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"



class Trainer:
    """Trainer for HelixLM with comprehensive logging and evaluation."""
    def __init__(self,
                 model: HelixForCausalLM,
                 cfg: HelixConfig,
                 train_texts: List[str],
                 val_texts: Optional[List[str]] = None,
                 tokenizer=None,
                 output_dir: str = "./checkpoints",
                 example_prompts: Optional[List[str]] = None,
                 generated_example_length: int = 15):

        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Prints examples of generation in the generation loop on checkpoint: 
        # Prompts for generation samples during training
        if example_prompts: 
            self.example_prompts = example_prompts
        else:
            self.example_prompts = [
                "In the beginning",
                "And God said",
                "The sky was"
            ]
        self.generated_example_length = generated_example_length


        self.device = self._get_device()
        self.model = self.model.to(self.device)

        self.train_loader = create_helix_dataloader(
            train_texts, tokenizer, cfg.seq_len, cfg.batch_size, shuffle=True,
        )
        self.val_loader = None
        if val_texts:
            self.val_loader = create_helix_dataloader(
                val_texts, tokenizer, cfg.seq_len, cfg.batch_size, shuffle=False,
            )

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )

        total_steps = len(self.train_loader) * cfg.epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=total_steps,
        )

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {"train_loss": [], "val_loss": [], "perplexity": []}

        self.scaler = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
            except ImportError:
                pass

    def _get_device(self) -> torch.device:
        if self.cfg.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.cfg.device)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs["loss"]
            else:
                outputs = self.model(input_ids, labels=labels)
                loss = outputs["loss"]

            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if batch_idx % 10 == 0:
                lr = self.scheduler.get_last_lr()[0]
                ppl = compute_perplexity(loss.item())
                print(f"  Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss.item():.4f} | PPL: {ppl:.2f} | LR: {lr:.2e}")

        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start
        return {"loss": avg_loss, "perplexity": compute_perplexity(avg_loss), "time": epoch_time}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            outputs = self.model(input_ids, labels=labels)
            total_loss += outputs["loss"].item()
            num_batches += 1
        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss, "perplexity": compute_perplexity(avg_loss)}

    @torch.no_grad()
    def generate_sample(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        if self.tokenizer is None:
            return ""
        self.model.eval()
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long).to(self.device)
        max_tokens = max_new_tokens or self.cfg.max_new_tokens
        generated = self.model.generate_ext(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
        )
        new_tokens = generated[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def save_checkpoint(self, epoch: int, filename: Optional[str] = None):
        if filename is None:
            filename = f"helixlm_epoch_{epoch}.pt"
        path = os.path.join(self.output_dir, filename)
        self.model.save_pretrained(path)
        print(f"Checkpoint saved to {path}")

    def train(self, num_epochs: Optional[int] = None, eval_every: int = 1):
        epochs = num_epochs or self.cfg.epochs
        print(f"\n{'='*60}")
        print(f"Training HelixLM on {self.device}")
        print(f"Parameters: {self.model.count_parameters()['total']:,}")
        print(f"Epochs: {epochs} | Batch: {self.cfg.batch_size} | LR: {self.cfg.lr}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 40)

            train_metrics = self.train_epoch(epoch)
            print(f"Train Loss: {train_metrics['loss']:.4f} | PPL: {train_metrics['perplexity']:.2f} | Time: {format_time(train_metrics['time'])}")
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["perplexity"].append(train_metrics["perplexity"])

            if self.val_loader and epoch % eval_every == 0:
                val_metrics = self.evaluate()
                print(f"Val Loss: {val_metrics['loss']:.4f} | Val PPL: {val_metrics['perplexity']:.2f}")
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
                                    max_new_tokens=self.generated_example_length)
                            print(f"  '{prompt}' -> '{generated}'")
                        except Exception as e:
                            print(f"  '{prompt}' -> [Error: {e}]")
                    else:
                      print(
                          "Parameter 'generated_example_length' set to 0. ",
                          "Skipping generating samples. ",
                          "To have samples generated in the trainign loop, set ",
                          "the parameter 'generated_example_length' to an non-zero integer."
                      )
                print()

        self.save_checkpoint(epochs, "final_model")
        print(f"\nTraining complete!")
        return self.history
