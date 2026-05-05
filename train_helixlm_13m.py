#!/usr/bin/env python3
"""
train_helixlm_13m.py

13M-param HelixLM training job with full architecture enabled:
  - Recurrent heterogeneous graph (n_loops=2)
  - Hybrid linear + full attention
  - Mamba-2 SSD (ssm_d_state=64)
  - Titans neural memory
  - RoPE, SwiGLU, RMSNorm

Saves a checkpoint every epoch. No stride / no within-document overlap.
Stage 1A: 3 epochs @ 3e-3  -> checkpoint every epoch
Stage 1B: 5 epochs @ 3e-4  -> checkpoint every epoch (grokking)
Stage 2:  2 epochs @ 3e-4  -> checkpoint every epoch (instruction tuning)
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HELIXLM_PATH = os.environ.get("HELIXLM_PATH", os.path.join(SCRIPT_DIR, "..", "HelixLM"))
if HELIXLM_PATH not in sys.path:
    sys.path.insert(0, HELIXLM_PATH)

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer


DATASET_REPO = "david-thrower/HelixLM-tiny-400.0Mt-730000pt-57143it-20260430"

# Presets for the 13M model.  No stride (overlap disabled).
JOB_PRESETS = {
    128: {
        "batch_size": 24,
        "grad_accum": 4,
        "seq_len": 128,
        "epochs_stage1a": 3,
        "epochs_stage1b": 5,
        "epochs_stage2": 2,
        "lr_stage1": 3.0e-3,
        "lr_grok": 3.0e-4,
        "lr_stage2": 3.0e-4,
        "warmup_steps": 1600,
        "weight_decay": 0.05,
        "grad_clip": 1.0,
        "eval_every": 1,
        "generated_example_length": 40,
        "label": "13M-128sl-loops2-fullarch",
    },
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
        "label": "13M-256sl-loops2-fullarch",
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


def print_memory(label: str):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  [VRAM] {label}: {alloc:.2f}GB")


def print_model_report(cfg, model, preset, phase="INIT"):
    params = model.count_parameters()
    try:
        graph = model.model.recurrent.graph.get_graph_info()
    except Exception:
        graph = {"n_nodes": "N/A", "n_edges": "N/A", "node_types": {}, "sinks": []}

    print(f"\n{'='*70}")
    print(f"  HELIXLM MODEL REPORT  [{phase}]")
    print(f"{'='*70}")
    print(f"  Preset            : {preset['label']}")
    print(f"  Total params      : {params['total']:,}  (~{params['total']/1e6:.1f}M)")
    print(f"  Trainable params  : {params['trainable']:,}")
    print(f"  d_model           : {cfg.d_model}")
    print(f"  n_columns         : {cfg.n_columns}")
    print(f"  n_loops           : {cfg.n_loops}")
    print(f"  n_heads           : {cfg.n_heads}")
    print(f"  head_dim          : {cfg.head_dim}")
    print(f"  seq_len           : {cfg.seq_len}")
    print(f"  batch_size        : {cfg.batch_size}")
    print(f"  grad_accum        : {preset['grad_accum']}")
    print(f"  Effective batch   : {cfg.batch_size * preset['grad_accum']} samples")
    print(f"  LR (current)      : {cfg.lr}")
    print(f"  weight_decay      : {cfg.weight_decay}")
    print(f"  grad_clip         : {cfg.grad_clip}")
    print(f"  warmup_steps      : {cfg.warmup_steps}")
    print(f"  attention_mode    : {cfg.attention_mode}")
    print(f"  hybrid_interval   : {cfg.hybrid_full_attention_interval}")
    print(f"  use_rope          : {cfg.use_rope}")
    print(f"  use_titans        : {cfg.use_titans_memory}")
    print(f"  use_ssm           : {cfg.use_ssm}")
    print(f"  ssm_d_state       : {cfg.ssm_d_state}")
    print(f"  ffn_expansion     : {cfg.ffn_expansion}")
    print(f"  Graph nodes       : {graph.get('n_nodes', 'N/A')}")
    print(f"  Graph edges       : {graph.get('n_edges', 'N/A')}")
    print(f"  Node type counts  : {graph.get('node_types', {})}")
    print(f"  Sink nodes        : {graph.get('sinks', [])}")
    print(f"{'='*70}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=256, choices=[128, 256])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="./helixlm-13m-job")
    parser.add_argument("--hf-org", type=str, default="david-thrower")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--no-titans", action="store_true")
    parser.add_argument("--push-every-epoch", action="store_true", help="Also push every epoch to HF Hub (slow)")
    return parser.parse_args()


def run_stage_epochwise(trainer, num_epochs, eval_every=1, stage_name="stage",
                        push_every_epoch=False, hf_org=None, repo_base=None,
                        tokenizer=None, hf_token=None):
    """
    Epoch-by-epoch training loop.
    Saves a local checkpoint after EVERY epoch.
    """
    history = {"train_loss": [], "val_loss": [], "perplexity": []}

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'-'*50}")
        print(f"{stage_name} | Epoch {epoch}/{num_epochs}")
        print(f"{'-'*50}")

        train_metrics = trainer.train_epoch(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["perplexity"].append(train_metrics["perplexity"])

        # --- SAVE CHECKPOINT EVERY EPOCH ---
        ckpt_name = f"{stage_name}_epoch_{epoch}"
        trainer.save_checkpoint(epoch, ckpt_name)
        print(f"  [CKPT] Saved {ckpt_name}")

        # Optional: push every epoch to Hub
        if push_every_epoch and hf_org and repo_base and hf_token:
            epoch_repo = f"{hf_org}/{repo_base}-epoch{epoch}"
            try:
                trainer.model.push_to_hub(epoch_repo, token=hf_token, private=False)
                if tokenizer:
                    tokenizer._backend.push_to_hub(epoch_repo, token=hf_token)
                print(f"  [HUB]  Pushed to {epoch_repo}")
            except Exception as e:
                print(f"  [HUB]  Push failed: {e}")

        # Validation
        if trainer.val_loader and epoch % eval_every == 0:
            val_metrics = trainer.evaluate()
            history["val_loss"].append(val_metrics["loss"])
            if val_metrics["loss"] < trainer.best_val_loss:
                trainer.best_val_loss = val_metrics["loss"]
                trainer.save_checkpoint(epoch, f"{stage_name}_best")
                print(f"  [CKPT] New best model saved")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val PPL: {val_metrics['perplexity']:.2f}")

        # Generation samples
        if trainer.tokenizer and trainer.verbose and epoch % eval_every == 0:
            print("\n  Generation samples:")
            for prompt in trainer.example_prompts:
                try:
                    gen = trainer.generate_sample(prompt, max_new_tokens=trainer.generated_example_length)
                    print(f"    '{prompt}' -> '{gen}'")
                except Exception as e:
                    print(f"    '{prompt}' -> [Error: {e}]")
            print()

    return history


def main():
    args = parse_args()
    preset = JOB_PRESETS[args.seq_len]
    os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = get_device()
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M")

    print(f"\n{'='*60}")
    print(f"HelixLM 13M HF Job | SeqLen={preset['seq_len']} | n_loops=2 | Full Arch")
    print(f"Titans: {'OFF' if args.no_titans else 'ON'} | SSM: ON | Stride: OFF")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"{'='*60}\n")

    print("Initializing tokenizer...")
    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)
    print(f"  Vocab: {vocab_size}")

    use_titans = not args.no_titans

    # ------------------------------------------------------------------
    # 13M config with ALL unique architecture features enabled
    # ------------------------------------------------------------------
    cfg = HelixConfig.tiny(
        vocab_size=vocab_size,
        seq_len=preset["seq_len"],
        tokenizer_name="gpt2",
        attention_mode="hybrid",
        hybrid_full_attention_interval=2,
        n_loops=2,                          # recurrent depth
        use_ssm=True,                       # Mamba-2 SSD
        ssm_d_state=64,                     # triggers optimized Mamba-2 path
        ssm_d_conv=4,
        ssm_expand=2,
        use_titans_memory=use_titans,       # neural long-term memory
        titans_feature_dim=64,
        titans_eta_init=0.01,
        titans_n_heads=4,
        use_rope=True,                      # rotary positional embeddings
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

    model = HelixForCausalLM(cfg)
    if str(model.device) != str(device):
        model = model.to(device)

    param_count = model.count_parameters()["total"]
    p_m = round(param_count / 1e6)

    # Guardrail: ensure we are in 13M territory
    if not (12_000_000 <= param_count <= 14_000_000):
        print(f"WARNING: Expected ~13M params, got {param_count:,}. Check HelixConfig.tiny defaults.")
    else:
        print(f"  Parameters: {param_count:,} (~{p_m}M) ✓")

    print_memory("After model init")
    print_model_report(cfg, model, preset, "INIT")

    # GPU warmup
    print("\nWarming up GPU...")
    dummy = torch.randint(0, vocab_size, (preset["batch_size"], preset["seq_len"]), device=device)
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()
    print("  Warmup complete.")
    print_memory("After warmup")

    # ------------------------------------------------------------------
    # Dataset (streaming, NO stride / no overlap)
    # ------------------------------------------------------------------
    print(f"\nStreaming dataset splits...")
    t0 = time.time()
    pretrain_train = load_texts(DATASET_REPO, "pretrain_train", args.max_samples)
    pretrain_val = load_texts(DATASET_REPO, "pretrain_val", args.max_samples)
    instruct_train = load_texts(DATASET_REPO, "instruct_train", args.max_samples)
    instruct_val = load_texts(DATASET_REPO, "instruct_val", args.max_samples)
    print(f"\nAll splits loaded in {time.time() - t0:.1f}s")

    if args.max_samples:
        pretrain_train = pretrain_train[:args.max_samples]
        instruct_train = instruct_train[:args.max_samples]

    print(f"  Pretrain train: {len(pretrain_train):,}")
    print(f"  Pretrain val:   {len(pretrain_val):,}")
    print(f"  Instruct train: {len(instruct_train):,}")
    print(f"  Instruct val:   {len(instruct_val):,}")

    HF_TOKEN = os.getenv("HF_TOKEN")
    if args.push_to_hub and not HF_TOKEN:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    # =====================================================================
    # STAGE 1A: High-LR Rapid Descent (3 epochs @ 3e-3)
    # =====================================================================
    print(f"\n{'='*60}")
    print("STAGE 1A: High-LR Pretraining (3e-3)")
    print(f"{'='*60}")

    cfg.lr = preset["lr_stage1"]

    trainer1a = Trainer(
        model=model,
        cfg=cfg,
        train_texts=pretrain_train,
        val_texts=pretrain_val,
        tokenizer=tokenizer,
        output_dir=os.path.join(args.output_dir, "stage1a"),
        example_prompts=PRETRAIN_PROMPTS,
        generated_example_length=preset["generated_example_length"],
        grad_accum_steps=preset["grad_accum"],
        use_amp=False,
        min_tail_len=1,
        # NOTE: no stride=... passed -> defaults to seq_len (no overlap)
    )

    t0 = time.time()
    history1a = run_stage_epochwise(
        trainer1a,
        num_epochs=preset["epochs_stage1a"],
        eval_every=preset["eval_every"],
        stage_name="stage1a",
        push_every_epoch=args.push_every_epoch,
        hf_org=args.hf_org,
        repo_base=f"HelixLM-13M-stage1a-{TIMESTAMP}",
        tokenizer=tokenizer,
        hf_token=HF_TOKEN,
    )
    stage1a_time = time.time() - t0
    print(f"\nStage 1A done in {stage1a_time / 3600:.1f}h")

    stage1a_ckpt = os.path.join(args.output_dir, "stage1a-final")
    model.save_pretrained(stage1a_ckpt)
    print(f"Final Stage 1A checkpoint: {stage1a_ckpt}")
    print_model_report(cfg, model, preset, "POST-1A")

    if args.push_to_hub:
        stage1a_name = f"HelixLM-13M-stage1a-3e3lr-{preset['seq_len']}sl-400Mt-{TIMESTAMP}"
        stage1a_repo = f"{args.hf_org}/{stage1a_name}"
        print(f"\nPushing Stage 1A to {stage1a_repo}")
        model.push_to_hub(stage1a_repo, token=HF_TOKEN, private=False)
        tokenizer._backend.push_to_hub(stage1a_repo, token=HF_TOKEN)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # =====================================================================
    # STAGE 1B: Grokking Phase (5 epochs @ 3e-4)
    # =====================================================================
    print(f"\n{'='*60}")
    print("STAGE 1B: Grokking Phase (3e-4)")
    print(f"{'='*60}")

    cfg.lr = preset["lr_grok"]

    trainer1b = Trainer(
        model=model,
        cfg=cfg,
        train_texts=pretrain_train,
        val_texts=pretrain_val,
        tokenizer=tokenizer,
        output_dir=os.path.join(args.output_dir, "stage1b"),
        example_prompts=PRETRAIN_PROMPTS,
        generated_example_length=preset["generated_example_length"],
        grad_accum_steps=preset["grad_accum"],
        use_amp=False,
        min_tail_len=1,
    )

    t0 = time.time()
    history1b = run_stage_epochwise(
        trainer1b,
        num_epochs=preset["epochs_stage1b"],
        eval_every=preset["eval_every"],
        stage_name="stage1b",
        push_every_epoch=args.push_every_epoch,
        hf_org=args.hf_org,
        repo_base=f"HelixLM-13M-stage1b-{TIMESTAMP}",
        tokenizer=tokenizer,
        hf_token=HF_TOKEN,
    )
    stage1b_time = time.time() - t0
    print(f"\nStage 1B done in {stage1b_time / 3600:.1f}h")

    stage1_ckpt = os.path.join(args.output_dir, "stage1-final")
    model.save_pretrained(stage1_ckpt)
    print(f"Final Stage 1 checkpoint: {stage1_ckpt}")
    print_model_report(cfg, model, preset, "POST-1B")

    if args.push_to_hub:
        stage1_name = f"HelixLM-13M-stage1-grok-{preset['seq_len']}sl-400Mt-{TIMESTAMP}"
        stage1_repo = f"{args.hf_org}/{stage1_name}"
        print(f"\nPushing Stage 1 to {stage1_repo}")
        model.push_to_hub(stage1_repo, token=HF_TOKEN, private=False)
        tokenizer._backend.push_to_hub(stage1_repo, token=HF_TOKEN)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # =====================================================================
    # STAGE 2: Instruction Fine-Tuning
    # =====================================================================
    print(f"\n{'='*60}")
    print("STAGE 2: Instruction Fine-Tuning")
    print(f"{'='*60}")

    cfg.lr = preset["lr_stage2"]

    trainer2 = Trainer(
        model=model,
        cfg=cfg,
        train_texts=instruct_train,
        val_texts=instruct_val,
        tokenizer=tokenizer,
        output_dir=os.path.join(args.output_dir, "stage2"),
        example_prompts=INSTRUCT_PROMPTS,
        generated_example_length=preset["generated_example_length"],
        grad_accum_steps=preset["grad_accum"],
        use_amp=False,
        min_tail_len=1,
    )

    t0 = time.time()
    history2 = run_stage_epochwise(
        trainer2,
        num_epochs=preset["epochs_stage2"],
        eval_every=preset["eval_every"],
        stage_name="stage2",
        push_every_epoch=args.push_every_epoch,
        hf_org=args.hf_org,
        repo_base=f"HelixLM-13M-stage2-{TIMESTAMP}",
        tokenizer=tokenizer,
        hf_token=HF_TOKEN,
    )
    stage2_time = time.time() - t0
    print(f"\nStage 2 done in {stage2_time / 3600:.1f}h")

    final_name = f"HelixLM-13M-stage2-instruct-{preset['seq_len']}sl-400Mt-{TIMESTAMP}"
    local_path = os.path.join(args.output_dir, final_name)
    model.save_pretrained(local_path)
    tokenizer._backend.save_pretrained(local_path)

    # ------------------------------------------------------------------
    # Metadata (same structure as your prior script)
    # ------------------------------------------------------------------
    metadata = {
        "model_size": "tiny",
        "parameters": param_count,
        "seq_len": preset["seq_len"],
        "batch_size": preset["batch_size"],
        "precision": "fp32",
        "titans_memory": use_titans,
        "use_ssm": cfg.use_ssm,
        "ssm_d_state": cfg.ssm_d_state,
        "n_loops": cfg.n_loops,
        "attention_mode": cfg.attention_mode,
        "stride": "disabled (no overlap)",
        "stage1a": {
            "epochs": preset["epochs_stage1a"],
            "lr": preset["lr_stage1"],
            "train_samples": len(pretrain_train),
            "time_hours": round(stage1a_time / 3600, 2),
            "final_train_loss": history1a["train_loss"][-1] if history1a.get("train_loss") else None,
            "final_val_loss": history1a["val_loss"][-1] if history1a.get("val_loss") else None,
        },
        "stage1b": {
            "epochs": preset["epochs_stage1b"],
            "lr": preset["lr_grok"],
            "train_samples": len(pretrain_train),
            "time_hours": round(stage1b_time / 3600, 2),
            "final_train_loss": history1b["train_loss"][-1] if history1b.get("train_loss") else None,
            "final_val_loss": history1b["val_loss"][-1] if history1b.get("val_loss") else None,
        },
        "stage2": {
            "epochs": preset["epochs_stage2"],
            "lr": preset["lr_stage2"],
            "train_samples": len(instruct_train),
            "time_hours": round(stage2_time / 3600, 2),
            "final_train_loss": history2["train_loss"][-1] if history2.get("train_loss") else None,
            "final_val_loss": history2["val_loss"][-1] if history2.get("val_loss") else None,
        },
        "total_wall_time_hours": round((stage1a_time + stage1b_time + stage2_time) / 3600, 2),
    }

    with open(os.path.join(local_path, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if args.push_to_hub:
        final_repo = f"{args.hf_org}/{final_name}"
        print(f"\nPushing Stage 2 (final) to {final_repo}")
        model.push_to_hub(final_repo, token=HF_TOKEN, private=False)
        tokenizer._backend.push_to_hub(final_repo, token=HF_TOKEN)

    # Final generation test
    print(f"\n{'='*60}")
    print("Final Generation Test")
    print(f"{'='*60}")
    model.eval()
    with torch.no_grad():
        print("\n-- Pretraining style --")
        for prompt in PRETRAIN_PROMPTS[:3]:
            ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            out = model.generate_ext(ids, max_new_tokens=40, temperature=0.8, top_k=50)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"  '{prompt}' -> '{text[len(prompt):].strip()}'")

        print("\n-- Instruction style --")
        for prompt in INSTRUCT_PROMPTS[:2]:
            ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            out = model.generate_ext(ids, max_new_tokens=60, temperature=0.7, top_k=50, top_p=0.95,
                                     stop_strings=["<|user|>", "<|endoftext|>"])
            full = tokenizer.decode(out[0], skip_special_tokens=True)
            resp = full.split("<|assistant|>")[-1].split("<|user|>")[0].strip() if "<|assistant|>" in full else full[len(prompt):].strip()
            q = prompt.replace("<|user|>\n", "").replace("\n<|assistant|>\n", "")
            print(f"  Q: {q}")
            print(f"  A: {resp}")
            print()

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"Total wall time: {metadata['total_wall_time_hours']:.1f}h")
    print(f"Checkpoints: every epoch saved under {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
