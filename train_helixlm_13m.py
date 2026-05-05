#!/usr/bin/env python3
"""
train_helixlm_13m_resume.py

Proven A100 training with epoch-level resume.
- NO torch.compile (harmful for this architecture)
- NO sharded streaming (unnecessary on A100 142GB RAM)
- Uses existing Trainer class directly
- Saves full state (model + optimizer + scheduler + RNG) every epoch
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
from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HELIXLM_PATH = os.environ.get("HELIXLM_PATH", os.path.join(SCRIPT_DIR, "..", "HelixLM"))
if HELIXLM_PATH not in sys.path:
    sys.path.insert(0, HELIXLM_PATH)

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer


DATASET_REPO = "david-thrower/HelixLM-tiny-400.0Mt-730000pt-57143it-20260430"

PRESETS = {
    256: {
        "batch_size": 24,
        "grad_accum": 4,
        "seq_len": 256,
        "epochs_stage1a": 3,
        "epochs_stage1b": 5,
        "epochs_stage2": 2,
        "lr_stage1": 3.0e-3,
        "lr_grok": 3.0e-4,
        "lr_stage2": 3.0e-4,
        "warmup_steps": 2000,
        "weight_decay": 0.05,
        "grad_clip": 1.0,
        "eval_every": 1,
        "generated_example_length": 50,
    },
}

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
    "<|user|>\nHow does photosynthesis work?\n<|assistant|>\n",
]


# ---------------------------------------------------------------------------
# Scheduler fallback (in case helix_lm.trainer export is flaky)
# ---------------------------------------------------------------------------
def _get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                     num_cycles=0.5, min_lr_ratio=0.1):
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
# Dataset loading (same as original working script)
# ---------------------------------------------------------------------------
def load_texts(repo_id: str, split_name: str, max_samples: Optional[int] = None) -> List[str]:
    print(f"  Streaming '{split_name}' ...")
    ds = load_dataset(repo_id, split=split_name, streaming=True)
    texts = []
    for i, item in enumerate(tqdm(ds, desc=f"  {split_name}", unit="smpl", leave=False)):
        if max_samples and i >= max_samples:
            break
        texts.append(item["text"])
    print(f"    -> {len(texts):,} samples")
    return texts


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        print(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f}GB")
        return dev
    print("WARNING: No CUDA available, falling back to CPU")
    return torch.device("cpu")


def print_model_report(cfg, model, preset, phase="INIT"):
    params = model.count_parameters()
    try:
        graph = model.model.recurrent.graph.get_graph_info()
    except Exception:
        graph = {"n_nodes": "N/A", "n_edges": "N/A", "node_types": {}, "sinks": []}

    print(f"\n{'='*70}")
    print(f"  HELIXLM MODEL REPORT  [{phase}]")
    print(f"{'='*70}")
    print(f"  Total params      : {params['total']:,}  (~{params['total']/1e6:.1f}M)")
    print(f"  d_model           : {cfg.d_model}")
    print(f"  n_columns         : {cfg.n_columns}")
    print(f"  n_loops           : {cfg.n_loops}")
    print(f"  seq_len           : {cfg.seq_len}")
    print(f"  batch_size        : {cfg.batch_size}")
    print(f"  grad_accum        : {preset['grad_accum']}")
    print(f"  Effective batch   : {cfg.batch_size * preset['grad_accum']} samples")
    print(f"  attention_mode    : {cfg.attention_mode}")
    print(f"  use_titans        : {cfg.use_titans_memory}")
    print(f"  use_ssm           : {cfg.use_ssm}")
    print(f"  Graph nodes       : {graph.get('n_nodes', 'N/A')}")
    print(f"  Graph edges       : {graph.get('n_edges', 'N/A')}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------
def _find_latest_resume_dir(output_dir: str, stage_name: str) -> Optional[str]:
    pattern = os.path.join(output_dir, stage_name, "resume_epoch_*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        return None
    latest = dirs[-1]
    if os.path.exists(os.path.join(latest, "trainer_state.pt")):
        return latest
    return None


def _find_global_resume(output_dir: str) -> Optional[str]:
    for stage in ["stage2", "stage1b", "stage1a"]:
        d = _find_latest_resume_dir(output_dir, stage)
        if d:
            return d
    return None


def _save_resume_checkpoint(trainer: Trainer, model: torch.nn.Module, epoch: int,
                            stage_name: str, output_dir: str, tokenizer):
    ckpt_dir = os.path.join(output_dir, stage_name, f"resume_epoch_{epoch:03d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "stage": stage_name,
        "model_state_dict": model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict() if trainer.scheduler else None,
        "global_step": trainer.global_step,
        "best_val_loss": trainer.best_val_loss,
        "history": trainer.history,
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))
    model.save_pretrained(ckpt_dir)
    tokenizer._backend.save_pretrained(ckpt_dir)
    print(f"  [CKPT] Resume saved: {ckpt_dir}")


def _resume_stage(trainer: Trainer, model: torch.nn.Module, stage_name: str,
                  output_dir: str, num_epochs: int) -> int:
    """Rebuild scheduler and load state. Returns next epoch to run (1 if fresh)."""
    ckpt_dir = _find_latest_resume_dir(output_dir, stage_name)
    if not ckpt_dir:
        return 1

    print(f"  [RESUME] Found checkpoint: {ckpt_dir}")
    state = torch.load(os.path.join(ckpt_dir, "trainer_state.pt"), map_location="cpu")

    # Load model
    model.load_state_dict(state["model_state_dict"])

    # Rebuild scheduler BEFORE loading (train_epoch skips init if scheduler exists)
    steps_per_epoch = math.ceil(len(trainer.train_loader) / trainer.grad_accum_steps)
    total_steps = steps_per_epoch * num_epochs
    warmup = max(1, trainer.cfg.warmup_steps // trainer.grad_accum_steps)

    try:
        from helix_lm.trainer import get_cosine_schedule_with_warmup
        trainer.scheduler = get_cosine_schedule_with_warmup(
            trainer.optimizer, warmup, total_steps
        )
    except Exception:
        trainer.scheduler = _get_cosine_schedule_with_warmup(
            trainer.optimizer, warmup, total_steps
        )

    # Load trainer state
    trainer.optimizer.load_state_dict(state["optimizer"])
    if state.get("scheduler") and trainer.scheduler:
        trainer.scheduler.load_state_dict(state["scheduler"])
    trainer.global_step = state.get("global_step", 0)
    trainer.best_val_loss = state.get("best_val_loss", float("inf"))
    if state.get("history"):
        trainer.history = state["history"]

    # RNG
    if state.get("rng") is not None:
        torch.set_rng_state(state["rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    start_epoch = state["epoch"] + 1
    print(f"  [RESUME] Starting {stage_name} at epoch {start_epoch}/{num_epochs}")
    return start_epoch


def _stage_is_complete(output_dir: str, stage_name: str, num_epochs: int) -> bool:
    d = _find_latest_resume_dir(output_dir, stage_name)
    if not d:
        return False
    state = torch.load(os.path.join(d, "trainer_state.pt"), map_location="cpu")
    return state["epoch"] >= num_epochs


# ---------------------------------------------------------------------------
# Stage runner (wraps Trainer, calls train_epoch directly for resume control)
# ---------------------------------------------------------------------------
def run_stage(model, cfg, tokenizer, device, stage_name, lr, num_epochs,
              train_texts, val_texts, output_dir, preset, example_prompts,
              push_to_hub=False, hf_org=None, hf_token=None, timestamp=None):
    os.makedirs(os.path.join(output_dir, stage_name), exist_ok=True)

    cfg.lr = lr
    cfg.epochs = num_epochs

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        output_dir=os.path.join(output_dir, stage_name),
        example_prompts=example_prompts,
        generated_example_length=preset["generated_example_length"],
        grad_accum_steps=preset["grad_accum"],
        use_amp=False,
        min_tail_len=1,
    )

    # Resume or start fresh
    start_epoch = _resume_stage(trainer, model, stage_name, output_dir, num_epochs)

    # Run epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"{stage_name.upper()} | Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")

        train_m = trainer.train_epoch(epoch)
        print(f"  Train loss: {train_m['loss']:.4f} | PPL: {train_m['perplexity']:.2f} | "
              f"Time: {train_m['time']:.1f}s | tok/s: {train_m.get('tok_per_sec', 'N/A')}")

        if trainer.val_loader and epoch % preset["eval_every"] == 0:
            val_m = trainer.evaluate()
            print(f"  Val loss:   {val_m['loss']:.4f} | Val PPL: {val_m['perplexity']:.2f}")
            if val_m["loss"] < trainer.best_val_loss:
                trainer.best_val_loss = val_m["loss"]
                model.save_pretrained(os.path.join(trainer.output_dir, "best_model"))

        # Generation samples
        if tokenizer and epoch % preset["eval_every"] == 0:
            print("\n  Generation samples:")
            for prompt in example_prompts[:3]:
                try:
                    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
                    out = model.generate_ext(ids, max_new_tokens=40, temperature=0.8, top_k=50)
                    text = tokenizer.decode(out[0], skip_special_tokens=True)
                    print(f"    '{prompt}' -> '{text[len(prompt):].strip()}'")
                except Exception as e:
                    print(f"    Error: {e}")

        # Save resume every epoch
        _save_resume_checkpoint(trainer, model, epoch, stage_name, output_dir, tokenizer)

    # Final stage export
    final_dir = os.path.join(output_dir, f"{stage_name}-final")
    model.save_pretrained(final_dir)
    print(f"\n  Stage {stage_name} complete. Saved to {final_dir}")

    if push_to_hub and hf_token and timestamp:
        repo = f"{hf_org}/HelixLM-13M-{stage_name}-{timestamp}"
        print(f"  Pushing to {repo}")
        model.push_to_hub(repo, token=hf_token, private=False)
        tokenizer._backend.push_to_hub(repo, token=hf_token)

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="./helixlm-13m-resume")
    parser.add_argument("--hf-org", type=str, default="david-thrower")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--use-ssm", action="store_true", help="Enable Mamba2 (NOT recommended, kills throughput)")
    parser.add_argument("--no-titans", action="store_true", help="Disable Titans memory")
    return parser.parse_args()


def main():
    args = parse_args()
    preset = PRESETS[args.seq_len]

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = get_device()
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M")

    print(f"\n{'='*60}")
    print(f"HelixLM 13M Resume Job | SeqLen={preset['seq_len']} | A100 Proven Config")
    print(f"Batch={args.batch_size} | Accum={args.grad_accum} | Titans={not args.no_titans} | SSM={args.use_ssm}")
    print(f"{'='*60}")

    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)
    print(f"Vocab: {vocab_size}")

    cfg = HelixConfig.tiny(
        vocab_size=vocab_size,
        seq_len=preset["seq_len"],
        tokenizer_name="gpt2",
        attention_mode="hybrid",
        hybrid_full_attention_interval=2,
        n_loops=2,
        use_ssm=args.use_ssm,
        use_titans_memory=not args.no_titans,
        use_rope=True,
        ffn_expansion=2.0,
        dropout=0.0,
        lr=preset["lr_stage1"],
        weight_decay=preset["weight_decay"],
        grad_clip=preset["grad_clip"],
        warmup_steps=preset["warmup_steps"],
        device=str(device),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    cfg.batch_size = args.batch_size

    model = HelixForCausalLM(cfg).to(device)
    param_count = model.count_parameters()["total"]
    print(f"Parameters: {param_count:,} (~{param_count/1e6:.1f}M)")

    if not (12_000_000 <= param_count <= 14_000_000):
        print(f"WARNING: Expected ~13M params, got {param_count:,}")

    print_model_report(cfg, model, preset, "INIT")

    # GPU warmup
    print("\nWarming up GPU...")
    dummy = torch.randint(0, vocab_size, (args.batch_size, preset["seq_len"]), device=device)
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()
    print("Warmup complete.")

    HF_TOKEN = os.getenv("HF_TOKEN")
    if args.push_to_hub and not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set")

    # ------------------------------------------------------------------
    # Global resume: load latest model weights across all stages
    # ------------------------------------------------------------------
    global_resume_dir = _find_global_resume(args.output_dir)
    if global_resume_dir:
        print(f"\n[INIT] Resuming from global checkpoint: {global_resume_dir}")
        state = torch.load(os.path.join(global_resume_dir, "trainer_state.pt"), map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        if state.get("rng") is not None:
            torch.set_rng_state(state["rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print("[INIT] Model weights restored.")

    # ------------------------------------------------------------------
    # Stage 1A + 1B (pretrain data)
    # ------------------------------------------------------------------
    run_1a = not _stage_is_complete(args.output_dir, "stage1a", preset["epochs_stage1a"])
    run_1b = not _stage_is_complete(args.output_dir, "stage1b", preset["epochs_stage1b"])

    if run_1a or run_1b:
        print("\nLoading pretrain splits...")
        pretrain_train = load_texts(DATASET_REPO, "pretrain_train", args.max_samples)
        pretrain_val = load_texts(DATASET_REPO, "pretrain_val", args.max_samples)

        if run_1a:
            model = run_stage(
                model, cfg, tokenizer, device,
                stage_name="stage1a",
                lr=preset["lr_stage1"],
                num_epochs=preset["epochs_stage1a"],
                train_texts=pretrain_train,
                val_texts=pretrain_val,
                output_dir=args.output_dir,
                preset=preset,
                example_prompts=PRETRAIN_PROMPTS,
                push_to_hub=args.push_to_hub,
                hf_org=args.hf_org,
                hf_token=HF_TOKEN,
                timestamp=TIMESTAMP,
            )

        if run_1b:
            model = run_stage(
                model, cfg, tokenizer, device,
                stage_name="stage1b",
                lr=preset["lr_grok"],
                num_epochs=preset["epochs_stage1b"],
                train_texts=pretrain_train,
                val_texts=pretrain_val,
                output_dir=args.output_dir,
                preset=preset,
                example_prompts=PRETRAIN_PROMPTS,
                push_to_hub=args.push_to_hub,
                hf_org=args.hf_org,
                hf_token=HF_TOKEN,
                timestamp=TIMESTAMP,
            )

        del pretrain_train, pretrain_val
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Stage 2 (instruct data)
    # ------------------------------------------------------------------
    if not _stage_is_complete(args.output_dir, "stage2", preset["epochs_stage2"]):
        print("\nLoading instruct splits...")
        instruct_train = load_texts(DATASET_REPO, "instruct_train", args.max_samples)
        instruct_val = load_texts(DATASET_REPO, "instruct_val", args.max_samples)

        model = run_stage(
            model, cfg, tokenizer, device,
            stage_name="stage2",
            lr=preset["lr_stage2"],
            num_epochs=preset["epochs_stage2"],
            train_texts=instruct_train,
            val_texts=instruct_val,
            output_dir=args.output_dir,
            preset=preset,
            example_prompts=INSTRUCT_PROMPTS,
            push_to_hub=args.push_to_hub,
            hf_org=args.hf_org,
            hf_token=HF_TOKEN,
            timestamp=TIMESTAMP,
        )

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------
    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer._backend.save_pretrained(final_path)

    metadata = {
        "parameters": param_count,
        "seq_len": preset["seq_len"],
        "batch_size": args.batch_size,
        "timestamp": TIMESTAMP,
    }
    with open(os.path.join(final_path, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"Final model: {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
