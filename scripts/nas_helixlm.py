#!/usr/bin/env python3
"""
scripts/nas_helixlm.py

Neural Architecture Search for HelixLM using Optuna.

3-round escalation protocol:
  Screening   : 5K samples, 2 epochs, ~80 trials, 5 parallel  -> eliminate losers
  Validation  : 50K samples, 5 epochs, ~15 trials, 3 parallel -> confirm top configs
  Final       : Full dataset, 10 epochs, 3 trials, 1 parallel  -> convergence

Features:
  - All 3 attention modes: linear, hybrid, full
  - AdamW hyperparameters searchable (beta1, beta2, eps, weight_decay)
  - Node-level params derived deterministically (opaque to search)
  - HelixLM-specific constraints hard-coded:
      * grad_accum <= 2 for stability at high LR (3e-3+)
      * FP32 forced for seq_len <= 128 (FP16 overflows in short recurrent paths)
      * Native batch size preferred over grad_accum
  - torch.compile flag: default OFF, only enables for static configs (n_loops<=2, no ACT)
  - Streaming support: when max_samples is None, uses StreamingDocumentDataset
    to avoid materializing the full corpus in memory.

Usage:
  # Screening round (fast, many trials, 5 parallel on g6e.2xlarge spot)
  python scripts/nas_helixlm.py --round screening --n-jobs 5 --seq-len 256

  # Validation round (top configs from screening, 3 parallel)
  python scripts/nas_helixlm.py --round validation --n-jobs 3 --seq-len 256

  # Final round (best config, full dataset, single A100)
  python scripts/nas_helixlm.py --round final --n-jobs 1 --seq-len 256

  # With torch.compile (only for fixed-loop configs)
  python scripts/nas_helixlm.py --round screening --compile

  # Cost-prediction study: search across all 3 seq_lens in one screening
  python scripts/nas_helixlm.py --round screening --search-seq-len --n-jobs 5
"""

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import optuna
import torch
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HELIXLM_PATH = os.environ.get("HELIXLM_PATH", os.path.join(SCRIPT_DIR, ".."))
if HELIXLM_PATH not in sys.path:
    sys.path.insert(0, HELIXLM_PATH)

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer
from helix_lm.nas_search_space import (
    sample_params,
    build_helix_config,
    estimate_vram,
    estimate_training_cost,
    params_to_flat_dict,
)
from helix_lm.dataset import create_streaming_loader

# ---------------------------------------------------------------------------
# Round configurations
# ---------------------------------------------------------------------------
ROUNDS: Dict[str, Dict[str, Any]] = {
    "screening": {
        "max_samples": 5000,         # 5K samples for fast elimination
        "epochs": 3,
        "n_trials": 80,
        "n_parallel": 5,
        "instance": "g6e.2xlarge",
        "instance_cost_hr": 1.52,
    },
    "validation": {
        "max_samples": 50000,      # 50K samples
        "epochs": 5,
        "n_trials": 15,
        "n_parallel": 3,
        "instance": "g6e.2xlarge",
        "instance_cost_hr": 1.52,
    },
    "final": {
        "max_samples": None,       # full dataset
        "epochs": 10,
        "n_trials": 3,
        "n_parallel": 1,
        "instance": "p4d.24xlarge",
        "instance_cost_hr": 32.00,
    },
}

# Default to tiny dataset for fast screening
DATASET_REPO = "david-thrower/HelixLM-small-50.0Mt-91250pt-7143it-20260427"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_texts(repo_id: str, split_name: str, max_samples: Optional[int] = None) -> List[str]:
    print(f"  Streaming '{split_name}' ...")
    ds = load_dataset(repo_id, split=split_name, streaming=True)
    texts = []
    for i, item in enumerate(tqdm(ds, desc=f"  {split_name}", unit="smpl", leave=False)):
        if max_samples is not None and i >= max_samples:
            break
        texts.append(item["text"])
    print(f"  -> {len(texts):,} samples loaded")
    return texts


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        print(f"  GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f}GB")
        return dev
    print("  WARNING: No CUDA available, falling back to CPU")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# torch.compile helper
# ---------------------------------------------------------------------------
def maybe_compile_model(model: torch.nn.Module, params: Dict[str, Any], compile_enabled: bool) -> torch.nn.Module:
    """
    Conditionally apply torch.compile based on HelixLM-specific safety rules.

    Rules:
    - Default: OFF (compile is harmful for dynamic recurrent graphs)
    - Only compile if:
        1. --compile flag is passed AND
        2. n_loops <= 2 (shallow loops are more traceable) AND
        3. ACT is disabled (fixed loop depth, static graph)
    """
    if not compile_enabled:
        print("  [COMPILE] torch.compile disabled (default for HelixLM)")
        return model

    # Safety checks
    if params["n_loops"] > 2:
        print(f"  [COMPILE] SKIPPED: n_loops={params['n_loops']} > 2 (dynamic graph too deep)")
        return model

    # Check if ACT is enabled in the model config
    has_act = getattr(model, "use_act", False) or getattr(model.cfg, "use_act", True)
    if has_act:
        print(f"  [COMPILE] SKIPPED: ACT halting enabled (dynamic loop depth)")
        return model

    # Check attention mode: full attention is more static than hybrid
    if params["attention_mode"] == "hybrid" and params.get("hybrid_full_attention_interval", 2) <= 1:
        print(f"  [COMPILE] SKIPPED: hybrid interval=1 (too much graph variation)")
        return model

    print(f"  [COMPILE] ENABLED: n_loops={params['n_loops']}, no ACT, mode={params['attention_mode']}")
    compiled = torch.compile(model, mode="reduce-overhead")
    return compiled


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial, args: argparse.Namespace, round_cfg: Dict[str, Any]) -> float:
    """
    Train one HelixLM configuration and return best validation perplexity.

    Returns:
        float: best validation perplexity (lower is better).
               Returns inf on failure/NaN so Optuna prunes the trial.
    """
    mlflow.set_experiment(f"helixlm_nas_{args.round}")

    # -----------------------------------------------------------------------
    # 1. Sample configuration
    # -----------------------------------------------------------------------
    if args.search_seq_len:
        seq_len = trial.suggest_categorical("seq_len", [128, 256, 512])
    else:
        seq_len = args.seq_len

    params = sample_params(trial, seq_len=seq_len)

    # -----------------------------------------------------------------------
    # 2. Build model config
    # -----------------------------------------------------------------------
    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)
    device = get_device()

    try:
        cfg = build_helix_config(params, vocab_size, str(device))
    except Exception as e:
        warnings.warn(f"Config build failed for trial {trial.number}: {e}")
        return float("inf")

    # -----------------------------------------------------------------------
    # 3. Instantiate model & count params
    # -----------------------------------------------------------------------
    try:
        model = HelixForCausalLM(cfg).to(device)
        param_count = model.count_parameters()["total"]
    except Exception as e:
        warnings.warn(f"Model instantiation failed for trial {trial.number}: {e}")
        return float("inf")

    # -----------------------------------------------------------------------
    # 3b. Conditionally apply torch.compile
    # -----------------------------------------------------------------------
    model = maybe_compile_model(model, params, args.compile)

    # -----------------------------------------------------------------------
    # 4. MLflow run
    # -----------------------------------------------------------------------
    run_name = f"trial_{trial.number:03d}_seq{params['seq_len']}_d{params['d_model']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params_to_flat_dict(params))
        mlflow.log_param("round", args.round)
        mlflow.log_param("trial_number", trial.number)
        mlflow.log_param("param_count", param_count)
        mlflow.log_param("estimated_vram_mb", round(estimate_vram(params), 1))
        mlflow.log_param("torch_compile", args.compile)

        cost_pred = estimate_training_cost(
            params,
            dataset_tokens=5_000_000 if args.round == "screening" else 400_000_000,
            epochs=round_cfg["epochs"],
            instance_cost_per_hour=round_cfg["instance_cost_hr"],
        )
        mlflow.log_params({f"cost_pred_{k}": str(v) for k, v in cost_pred.items()})

        print(f"\n{'='*60}")
        print(f"  TRIAL {trial.number} | {args.round.upper()}")
        print(f"  d_model={params['d_model']}  loops={params['n_loops']}  "
              f"seq={params['seq_len']}  lr={params['lr']}")
        print(f"  attention={params['attention_mode']}  "
              f"ssm={params['use_ssm']}  titans={params['use_titans']}")
        print(f"  adam: beta1={params['beta1']}  beta2={params['beta2']}  "
              f"wd={params['weight_decay']}  eps={params['adam_eps']}")
        print(f"  params={param_count:,}  est_vram={estimate_vram(params):.0f}MB  "
              f"batch={params['batch_size']}  accum={params['grad_accum']}")
        print(f"  est_cost=${cost_pred['estimated_cost_usd']}  "
              f"est_wall={cost_pred['wall_days']} days")
        print(f"{'='*60}")

        # -------------------------------------------------------------------
        # 5. Load data
        # -------------------------------------------------------------------
        train_max = round_cfg["max_samples"]
        val_max = max(500, train_max // 10) if train_max else 5000

        try:
            if train_max is None:
                # Full dataset: use streaming to avoid RAM explosion
                print("  [DATA] Using streaming loader (full dataset, no RAM materialization)")
                train_stream = load_dataset(args.dataset_repo, split="pretrain_train", streaming=True)
                val_stream = load_dataset(args.dataset_repo, split="pretrain_val", streaming=True)

                train_loader = create_streaming_loader(
                    train_stream, tokenizer, cfg.seq_len,
                    batch_size=cfg.batch_size,
                    max_samples=None,
                    num_workers=2,
                    pin_memory=True,
                )
                val_loader = create_streaming_loader(
                    val_stream, tokenizer, cfg.seq_len,
                    batch_size=cfg.batch_size,
                    max_samples=val_max,
                    num_workers=2,
                    pin_memory=True,
                )

                trainer = Trainer(
                    model=model,
                    cfg=cfg,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    tokenizer=tokenizer,
                    output_dir=os.path.join(args.output_dir, f"trial_{trial.number:03d}"),
                    example_prompts=["The next day", "In 1492,"],
                    generated_example_length=30,
                    grad_accum_steps=params["grad_accum"],
                    use_amp=params["seq_len"] > 128,
                    min_tail_len=1,
                    num_workers=0,  # workers are in the DataLoader, not Trainer
                    pin_memory=True,
                )
            else:
                # Subset: load texts into memory (fast for small subsets)
                train_texts = load_texts(args.dataset_repo, "pretrain_train", train_max)
                val_texts = load_texts(args.dataset_repo, "pretrain_val", val_max)

                trainer = Trainer(
                    model=model,
                    cfg=cfg,
                    train_texts=train_texts,
                    val_texts=val_texts,
                    tokenizer=tokenizer,
                    output_dir=os.path.join(args.output_dir, f"trial_{trial.number:03d}"),
                    example_prompts=["The next day", "In 1492,"],
                    generated_example_length=30,
                    grad_accum_steps=params["grad_accum"],
                    use_amp=params["seq_len"] > 128,
                    min_tail_len=1,
                )
        except Exception as e:
            warnings.warn(f"Data loading failed: {e}")
            mlflow.log_param("failed", "data_loading")
            return float("inf")

        # -------------------------------------------------------------------
        # 6. Training loop
        # -------------------------------------------------------------------
        best_val_ppl = float("inf")
        tok_per_sec_list: List[float] = []
        start_time = time.time()

        for epoch in range(1, round_cfg["epochs"] + 1):
            epoch_start = time.time()

            try:
                train_m = trainer.train_epoch(epoch)
            except Exception as e:
                warnings.warn(f"Train epoch failed: {e}")
                mlflow.log_param("failed", f"train_epoch_{epoch}")
                return float("inf")

            epoch_time = time.time() - epoch_start
            tok_per_sec = train_m.get("tok_per_sec", 0.0) or train_m.get("tok/s", 0.0)
            tok_per_sec_list.append(tok_per_sec)

            train_loss = train_m.get("loss", float("inf"))
            train_ppl = train_m.get("perplexity", float("inf"))
            if not math.isfinite(train_loss) or train_loss > 50000 or train_ppl > 50000:
                print(f"  [FAIL] Trial {trial.number} exploded at epoch {epoch} "
                      f"(loss={train_loss:.2f}, ppl={train_ppl:.2f})")
                mlflow.log_param("failed", f"exploded_epoch_{epoch}")
                mlflow.log_param("exploded_loss", train_loss)
                return float("inf")

            # Validation
            val_ppl = float("inf")
            if trainer.val_loader and epoch % max(1, round_cfg["epochs"] // 2) == 0:
                try:
                    val_m = trainer.evaluate()
                    val_loss = val_m.get("loss", float("inf"))
                    val_ppl = val_m.get("perplexity", float("inf"))
                    best_val_ppl = min(best_val_ppl, val_ppl)

                    mlflow.log_metrics({
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                    }, step=epoch)
                except Exception as e:
                    warnings.warn(f"Validation failed: {e}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_ppl": train_ppl,
                "tok_per_sec": tok_per_sec,
                "epoch_time_sec": epoch_time,
            }, step=epoch)

            report_value = best_val_ppl if math.isfinite(best_val_ppl) else train_ppl
            trial.report(report_value, epoch)
            if trial.should_prune():
                print(f"  [PRUNE] Trial {trial.number} pruned at epoch {epoch}")
                mlflow.log_param("pruned_at_epoch", epoch)
                raise optuna.TrialPruned()

        # -------------------------------------------------------------------
        # 7. Finalize
        # -------------------------------------------------------------------
        wall_time = time.time() - start_time
        avg_tok_per_sec = float(np.mean(tok_per_sec_list)) if tok_per_sec_list else 0.0

        mlflow.log_metrics({
            "best_val_ppl": best_val_ppl if math.isfinite(best_val_ppl) else 99999.0,
            "avg_tok_per_sec": avg_tok_per_sec,
            "wall_time_sec": wall_time,
        })

        actual_cost = estimate_training_cost(
            params,
            dataset_tokens=5_000_000 if args.round == "screening" else 400_000_000,
            epochs=round_cfg["epochs"],
            instance_cost_per_hour=round_cfg["instance_cost_hr"],
            tok_per_sec_assumed=avg_tok_per_sec,
        )
        mlflow.log_params({f"actual_cost_{k}": str(v) for k, v in actual_cost.items()})

        print(f"  [DONE] Trial {trial.number} | best_val_ppl={best_val_ppl:.2f} | "
              f"avg_tok/s={avg_tok_per_sec:.0f} | wall={wall_time/60:.1f}min")

        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_val_ppl if math.isfinite(best_val_ppl) else 99999.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HelixLM NAS with Optuna")
    parser.add_argument("--round", choices=["screening", "validation", "final"], required=True)
    parser.add_argument("--output-dir", default="./nas_results")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--search-seq-len", action="store_true")
    parser.add_argument("--study-name", default="helixlm_nas")
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--enqueue-top", type=int, default=None)
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for static configs only (n_loops<=2, no ACT). Default: OFF.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    round_cfg = ROUNDS[args.round]
    n_trials = args.n_trials or round_cfg["n_trials"]
    n_jobs = args.n_jobs or round_cfg["n_parallel"]

    os.makedirs(args.output_dir, exist_ok=True)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(f"helixlm_nas_{args.round}")

    storage_path = os.path.join(args.output_dir, f"{args.study_name}_{args.round}.db")
    storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=f"{args.study_name}_{args.round}",
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            n_startup_trials=min(10, n_trials // 4),
            seed=42,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=1,
            interval_steps=1,
        ),
        load_if_exists=True,
    )

    # Enqueue top configs from previous round
    if args.round in ("validation", "final"):
        prev_round = "screening" if args.round == "validation" else "validation"
        prev_storage = os.path.join(args.output_dir, f"{args.study_name}_{prev_round}.db")
        if os.path.exists(prev_storage):
            try:
                prev_study = optuna.load_study(
                    study_name=f"{args.study_name}_{prev_round}",
                    storage=f"sqlite:///{prev_storage}",
                )
                completed = [t for t in prev_study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
                top_n = args.enqueue_top or round_cfg["n_trials"]
                top_trials = sorted(completed, key=lambda t: t.value)[:top_n]

                for t in top_trials:
                    if t.params:
                        study.enqueue_trial(t.params)
                print(f"Enqueued {len(top_trials)} top configs from {prev_round}")
            except Exception as e:
                print(f"Warning: could not load previous study for enqueuing: {e}")

    print(f"\n{'='*70}")
    print(f" HELIXLM NAS — {args.round.upper()} ROUND")
    print(f" Trials: {n_trials}  |  Parallel: {n_jobs}  |  SeqLen: {'searched' if args.search_seq_len else args.seq_len}")
    print(f" Dataset: {args.dataset_repo}")
    print(f" torch.compile: {'enabled (conditional)' if args.compile else 'disabled'}")
    print(f" Storage: {storage_path}")
    print(f"{'='*70}\n")

    study.optimize(
        lambda trial: objective(trial, args, round_cfg),
        n_trials=n_trials,
        n_jobs=n_jobs,
        catch=(Exception,),
        show_progress_bar=True,
    )

    print(f"\n{'='*70}")
    print(f" NAS {args.round.upper()} COMPLETE")
    print(f"{'='*70}")

    if study.best_trial is not None:
        bt = study.best_trial
        print(f"Best trial    : #{bt.number}")
        print(f"Best val PPL  : {bt.value:.2f}")
        print(f"Best params   :")
        for k, v in bt.params.items():
            print(f"  {k:30s} = {v}")
    else:
        print("No successful trials completed.")

    results: List[Dict[str, Any]] = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
            results.append({
                "trial": t.number,
                "value": t.value,
                "params": t.params,
                "datetime_start": str(t.datetime_start) if t.datetime_start else None,
                "datetime_complete": str(t.datetime_complete) if t.datetime_complete else None,
                "duration_sec": (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_complete and t.datetime_start else None,
            })

    json_path = os.path.join(args.output_dir, f"nas_{args.round}_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "round": args.round,
            "best_trial": study.best_trial.number if study.best_trial else None,
            "best_value": study.best_trial.value if study.best_trial else None,
            "best_params": study.best_trial.params if study.best_trial else None,
            "n_trials_completed": len(results),
            "trials": results,
        }, f, indent=2)
    print(f"\nJSON results : {json_path}")

    csv_path = os.path.join(args.output_dir, f"nas_{args.round}_results.csv")
    if results:
        all_keys = sorted(set().union(*(r["params"].keys() for r in results)))
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "val_ppl", "duration_sec"] + all_keys)
            for r in results:
                writer.writerow([
                    r["trial"],
                    r["value"],
                    r.get("duration_sec", ""),
                ] + [r["params"].get(k, "") for k in all_keys])
        print(f"CSV results  : {csv_path}")

    if results and args.search_seq_len:
        print(f"\nCost-ranked summary (seq_len grid):")
        for seq in [128, 256, 512]:
            seq_results = [r for r in results if r["params"].get("seq_len") == seq]
            if seq_results:
                best = min(seq_results, key=lambda r: r["value"])
                print(f"  seq={seq:3d} | best_ppl={best['value']:.2f} | trial={best['trial']}")


if __name__ == "__main__":
    main()
