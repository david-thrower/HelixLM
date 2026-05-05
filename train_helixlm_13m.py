#!/usr/bin/env python3
"""
train_helixlm_13m.py

Production rewrite with three critical fixes:
  1. torch.compile()  -> fuses Python loops in Mamba2/Titans
  2. Shard streaming  -> caps host RAM to ~2-4 GB per shard
  3. Epoch checkpoint -> full resume (model + optimizer + scheduler + RNG)

Stages:
  1A: 3 epochs @ 3e-3  (high-LR descent)
  1B: 5 epochs @ 3e-4  (grokking)
  2:  2 epochs @ 3e-4  (instruction tuning)
"""

import argparse
import gc
import glob
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# HelixLM imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HELIXLM_PATH = os.environ.get("HELIXLM_PATH", os.path.join(SCRIPT_DIR, "..", "HelixLM"))
if HELIXLM_PATH not in sys.path:
    sys.path.insert(0, HELIXLM_PATH)

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer
from helix_lm.dataset import DocumentAwareDataset


DATASET_REPO = "david-thrower/HelixLM-tiny-400.0Mt-730000pt-57143it-20260430"

PRETRAIN_PROMPTS = [
    "After seeing something unexpected, Peter shouted",
    "It was a rainy day, so the children decided to",
    "The cat sat on the mat and looked at",
    "To make a sandwich, you need bread and",
    "The old car sputtered once and then",
]

INSTRUCT_PROMPTS = [
    "<|user|>\nWhat is the derivative of x^2?\n<|assistant|>\n",
    "<|user|>\nExplain quantum computing in simple terms.\n<|assistant|>\n",
    "<|user|>\nWrite a Python function to calculate fibonacci numbers.\n<|assistant|>\n",
]


# ---------------------------------------------------------------------------
# Scheduler (inline to avoid import fragility)
# ---------------------------------------------------------------------------
def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * max(0.0, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# GPU / preset helpers
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        print(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f}GB")
        return dev
    print("WARNING: No CUDA available, falling back to CPU")
    return torch.device("cpu")


def detect_gpu_preset(seq_len: int):
    if not torch.cuda.is_available():
        return {"batch_size": 8, "grad_accum": 4, "shard_size": 5000, "label": "CPU"}
    name = torch.cuda.get_device_properties(0).name.lower()
    if "a100" in name:
        # 80 GB HBM2e — massive headroom for a 14 M model
        return {"batch_size": 64, "grad_accum": 2, "shard_size": 50000, "label": "A100"}
    if "l4" in name:
        # 24 GB
        return {"batch_size": 24, "grad_accum": 4, "shard_size": 50000, "label": "L4"}
    # Generic GPU fallback
    return {"batch_size": 16, "grad_accum": 4, "shard_size": 25000, "label": "GPU"}


# ---------------------------------------------------------------------------
# Shard streaming
# ---------------------------------------------------------------------------
def shard_iterator(
    repo_id: str,
    split: str,
    shard_size: int,
    max_samples: Optional[int] = None,
):
    """Yield lists of `shard_size` texts from a streaming HF dataset."""
    print(f"  Streaming '{split}' in shards of {shard_size} ...")
    ds = load_dataset(repo_id, split=split, streaming=True)
    buffer: List[str] = []
    count = 0
    for item in ds:
        text = item.get("text", "")
        if text:
            buffer.append(text)
            count += 1
        if len(buffer) >= shard_size:
            yield buffer
            buffer = []
            gc.collect()
        if max_samples and count >= max_samples:
            break
    if buffer:
        yield buffer


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------
def build_loader(
    texts: List[str],
    tokenizer,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
):
    ds = DocumentAwareDataset(
        texts, tokenizer, seq_len, min_tail_len=1, lazy=True
    )

    def collate_fn(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        }

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
        num_workers=0,  # safer in container jobs
    )


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------
def save_resume(
    output_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    stage: str,
):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "global_step": global_step,
        "stage": stage,
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path = os.path.join(output_dir, f"resume_{stage}_epoch_{epoch:03d}.pt")
    torch.save(ckpt, path)
    print(f"  [CKPT] Resume state saved: {path}")
    return path


def try_resume(
    output_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    stage: str,
):
    """Return (start_epoch, global_step). If nothing found, returns (1, 0)."""
    pattern = os.path.join(output_dir, f"resume_{stage}_epoch_*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        return 1, 0
    latest = files[-1]
    print(f"  [RESUME] Loading state from {latest}")
    ckpt = torch.load(latest, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("rng") is not None:
        torch.set_rng_state(ckpt["rng"])
    if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt.get("global_step", 0)
    print(f"  [RESUME] Stage {stage} resuming at epoch {start_epoch} (step {global_step})")
    return start_epoch, global_step


def fast_forward_scheduler(scheduler, steps: int):
    """Step scheduler `steps` times to restore LR position after resume."""
    for _ in range(steps):
        scheduler.step()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_model(model, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in tqdm(loader, desc="Val", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            num_batches += 1
    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Core shard-aware training stage
# ---------------------------------------------------------------------------
def train_stage(
    model: HelixForCausalLM,
    cfg: HelixConfig,
    tokenizer,
    device: torch.device,
    stage_name: str,
    lr: float,
    epochs: int,
    train_split: str,
    val_split: str,
    shard_size: int,
    batch_size: int,
    grad_accum: int,
    output_dir: str,
    example_prompts: List[str],
    max_samples: Optional[int] = None,
):
    os.makedirs(output_dir, exist_ok=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    # Scheduler: estimate total steps for the stage
    # (over-estimate is safe; under-estimate just means LR stays higher longer)
    if max_samples:
        est_steps_per_epoch = math.ceil(
            (max_samples * 1.5) / (batch_size * grad_accum)
        )
    else:
        # ~730 k samples in full dataset; 1.5x fudge for chunking overlap
        est_steps_per_epoch = math.ceil((730_000 * 1.5) / (batch_size * grad_accum))
    total_est_steps = est_steps_per_epoch * epochs
    warmup_steps = max(1, cfg.warmup_steps // grad_accum)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_est_steps,
        min_lr_ratio=0.1,
    )

    # Resume
    start_epoch, global_step = try_resume(output_dir, model, optimizer, stage_name)
    if global_step > 0:
        fast_forward_scheduler(scheduler, global_step)

    # Load validation once (small — cap at 10 k docs)
    print(f"\nLoading validation split '{val_split}' (max 10k docs) ...")
    val_texts: List[str] = []
    for shard in shard_iterator(DATASET_REPO, val_split, shard_size=10000, max_samples=10000):
        val_texts.extend(shard)
        if len(val_texts) >= 10000:
            break
    val_loader = build_loader(val_texts, tokenizer, cfg.seq_len, batch_size, shuffle=False, drop_last=False)
    print(f"  Val docs: {len(val_texts)} | Val batches: {len(val_loader)}")

    # Epoch loop
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*60}")
        print(f"{stage_name.upper()} | Epoch {epoch}/{epochs} | LR={lr}")
        print(f"{'='*60}")

        model.train()
        epoch_loss = 0.0
        num_batches = 0
        accum_count = 0
        optimizer.zero_grad()

        shard_idx = 0
        for train_texts in shard_iterator(
            DATASET_REPO, train_split, shard_size, max_samples
        ):
            shard_idx += 1
            print(f"  Shard {shard_idx:02d} | {len(train_texts):,} docs")

            train_loader = build_loader(
                train_texts, tokenizer, cfg.seq_len, batch_size, shuffle=True, drop_last=True
            )

            pbar = tqdm(train_loader, desc=f"Ep{epoch} Sh{shard_idx}", leave=False)
            for batch_idx, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"] / grad_accum
                loss.backward()

                accum_count += 1
                epoch_loss += loss.item() * grad_accum
                num_batches += 1

                is_last = (batch_idx + 1) == len(train_loader)
                if accum_count >= grad_accum or is_last:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    accum_count = 0
                    global_step += 1

                # Live postfix
                avg = epoch_loss / max(num_batches, 1)
                pbar.set_postfix(
                    {
                        "loss": f"{avg:.4f}",
                        "ppl": f"{math.exp(min(avg, 20)):.1f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            # AGGRESSIVE CLEANUP between shards
            del train_loader, train_texts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # End of epoch
        avg_loss = epoch_loss / max(num_batches, 1)
        ppl = math.exp(min(avg_loss, 20))
        print(f"\nEpoch {epoch} done | loss={avg_loss:.4f} | ppl={ppl:.2f}")

        # Validation
        val_loss = eval_model(model, val_loader, device)
        val_ppl = math.exp(min(val_loss, 20))
        print(f"Val loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")

        # Save everything
        save_resume(output_dir, model, optimizer, scheduler, epoch, global_step, stage_name)
        model.save_pretrained(os.path.join(output_dir, f"{stage_name}_epoch_{epoch:03d}"))

        # Generation sanity check
        if example_prompts:
            model.eval()
            print("\nGeneration samples:")
            for prompt in example_prompts[:3]:
                try:
                    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
                    out = model.generate_ext(ids, max_new_tokens=40, temperature=0.8, top_k=50)
                    text = tokenizer.decode(out[0], skip_special_tokens=True)
                    print(f"  '{prompt}' -> '{text[len(prompt):].strip()}'")
                except Exception as e:
                    print(f"  [gen error] {e}")
            model.train()

    # Cleanup
    del val_loader, val_texts
    gc.collect()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="./helixlm-13m-fast")
    parser.add_argument("--hf-org", type=str, default="david-thrower")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--no-titans", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    # Auto-detect preset
    preset = detect_gpu_preset(args.seq_len)
    batch_size = args.batch_size or preset["batch_size"]
    grad_accum = args.grad_accum or preset["grad_accum"]
    shard_size = args.shard_size or preset["shard_size"]

    print(f"\n{'='*60}")
    print(f"HelixLM 13M Fast | GPU={preset['label']} | Compile=ON")
    print(f"Batch={batch_size} | Accum={grad_accum} | Shard={shard_size} | Seq={args.seq_len}")
    print(f"{'='*60}")

    # Tokenizer
    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)
    print(f"Vocab: {vocab_size}")

    # Config: 13 M with all unique features ON
    cfg = HelixConfig.tiny(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        tokenizer_name="gpt2",
        attention_mode="hybrid",
        hybrid_full_attention_interval=2,
        n_loops=2,
        use_ssm=True,
        ssm_d_state=64,
        use_titans_memory=not args.no_titans,
        use_rope=True,
        ffn_expansion=2.0,
        dropout=0.0,
        lr=3e-3,
        weight_decay=0.05,
        grad_clip=1.0,
        warmup_steps=2000,
        device=str(device),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

    # Model
    model = HelixForCausalLM(cfg).to(device)
    param_count = model.count_parameters()["total"]
    print(f"Parameters: {param_count:,} (~{param_count / 1e6:.1f}M)")

    # CRITICAL: compile the recurrent core to fuse Python loops
    print("\nCompiling model with torch.compile (reduce-overhead)...")
    try:
        model.model = torch.compile(model.model, mode="reduce-overhead")
        print("  torch.compile OK.")
    except Exception as e:
        print(f"  torch.compile warning (continuing): {e}")

    HF_TOKEN = os.getenv("HF_TOKEN")
    if args.push_to_hub and not HF_TOKEN:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M")

    # =====================================================================
    # STAGE 1A: High-LR descent (3 epochs @ 3e-3)
    # =====================================================================
    model = train_stage(
        model=model,
        cfg=cfg,
        tokenizer=tokenizer,
        device=device,
        stage_name="stage1a",
        lr=3.0e-3,
        epochs=3,
        train_split="pretrain_train",
        val_split="pretrain_val",
        shard_size=shard_size,
        batch_size=batch_size,
        grad_accum=grad_accum,
        output_dir=os.path.join(args.output_dir, "stage1a"),
        example_prompts=PRETRAIN_PROMPTS,
        max_samples=args.max_samples,
    )

    stage1a_final = os.path.join(args.output_dir, "stage1a", "final")
    model.save_pretrained(stage1a_final)
    if args.push_to_hub:
        repo = f"{args.hf_org}/HelixLM-13M-stage1a-{TIMESTAMP}"
        print(f"\nPushing Stage 1A to {repo}")
        model.push_to_hub(repo, token=HF_TOKEN, private=False)
        tokenizer._backend.push_to_hub(repo, token=HF_TOKEN)

    # =====================================================================
    # STAGE 1B: Grokking (5 epochs @ 3e-4)
    # =====================================================================
    model = train_stage(
        model=model,
        cfg=cfg,
        tokenizer=tokenizer,
        device=device,
        stage_name="stage1b",
        lr=3.0e-4,
        epochs=5,
        train_split="pretrain_train",
        val_split="pretrain_val",
        shard_size=shard_size,
        batch_size=batch_size,
        grad_accum=grad_accum,
        output_dir=os.path.join(args.output_dir, "stage1b"),
        example_prompts=PRETRAIN_PROMPTS,
        max_samples=args.max_samples,
    )

    stage1_final = os.path.join(args.output_dir, "stage1b", "final")
    model.save_pretrained(stage1_final)
    if args.push_to_hub:
        repo = f"{args.hf_org}/HelixLM-13M-stage1b-{TIMESTAMP}"
        print(f"\nPushing Stage 1B to {repo}")
        model.push_to_hub(repo, token=HF_TOKEN, private=False)
        tokenizer._backend.push_to_hub(repo, token=HF_TOKEN)

    # =====================================================================
    # STAGE 2: Instruction tuning (2 epochs @ 3e-4)
    # =====================================================================
    model = train_stage(
        model=model,
        cfg=cfg,
        tokenizer=tokenizer,
        device=device,
        stage_name="stage2",
        lr=3.0e-4,
        epochs=2,
        train_split="instruct_train",
        val_split="instruct_val",
        shard_size=shard_size,
        batch_size=batch_size,
        grad_accum=grad_accum,
        output_dir=os.path.join(args.output_dir, "stage2"),
        example_prompts=INSTRUCT_PROMPTS,
        max_samples=args.max_samples,
    )

    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer._backend.save_pretrained(final_path)

    # Metadata
    metadata = {
        "parameters": param_count,
        "seq_len": args.seq_len,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "shard_size": shard_size,
        "gpu": preset["label"],
        "compiled": True,
        "timestamp": TIMESTAMP,
    }
    with open(os.path.join(final_path, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if args.push_to_hub:
        repo = f"{args.hf_org}/HelixLM-13M-final-{TIMESTAMP}"
        print(f"\nPushing final to {repo}")
        model.push_to_hub(repo, token=HF_TOKEN, private=False)
        tokenizer._backend.push_to_hub(repo, token=HF_TOKEN)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"Final model: {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
