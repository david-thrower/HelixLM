"""
Integration test: DocumentAwareDataset + Trainer with mixed-length texts.
CICD-ready with live stdout logging.
"""
import sys
import os
import math
import random
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from datasets import load_dataset

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer
from helix_lm.dataset import DocumentAwareDataset, create_document_loader


SEED = 42
SEQ_LEN = 32
N_SAMPLES = 500
N_EPOCHS = 15
BATCH_SIZE = 8
DATASET_SLICES = 5_000


def banner(text):
    line = "=" * 60
    print(f"\n{line}\n{text}\n{line}", flush=True)


def get_token_counts(texts, tokenizer):
    return [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]


def select_long_mixed(texts, counts, seq_len, n=N_SAMPLES):
    barely, considerable = [], []
    for t, c in zip(texts, counts):
        if c < seq_len:
            continue
        if c <= int(seq_len * 1.5):
            barely.append(t)
        elif c >= seq_len * 2:
            considerable.append(t)

    n_b = min(len(barely), n // 3)
    n_c = min(len(considerable), n * 2 // 3)
    chosen = []
    if n_b:
        chosen.extend(random.sample(barely, n_b))
    if n_c:
        chosen.extend(random.sample(considerable, n_c))

    need = n - len(chosen)
    pool = [t for t, c in zip(texts, counts) if c >= seq_len and t not in chosen]
    if need > 0 and pool:
        chosen.extend(random.sample(pool, min(need, len(pool))))

    random.shuffle(chosen)
    return chosen[:n]


def select_short(texts, counts, seq_len, n=N_SAMPLES):
    pool = [t for t, c in zip(texts, counts) if 0 < c < seq_len]
    if len(pool) >= n:
        return random.sample(pool, n)
    banner(f"WARNING: only {len(pool)} short texts found (requested {n})")
    return pool


def build_cfg(vocab_size):
    return HelixConfig.tiny(
        vocab_size=vocab_size,
        seq_len=SEQ_LEN,
        tokenizer_name="gpt2",
        use_titans_memory=False,
        batch_size=BATCH_SIZE,
        lr=3e-4,
        weight_decay=0.1,
        epochs=N_EPOCHS,
        warmup_steps=50,
        grad_clip=1.0,
    )


def run_trainer(model, cfg, tokenizer, train_texts, val_texts, stride, min_tail_len, label):
    """
    Run the official Trainer for N_EPOCHS with live verbose output.
    Returns (final_train_loss, final_val_loss, final_ppl).
    """
    banner(f"[{label}] Starting {N_EPOCHS}-epoch training | stride={stride} | {len(train_texts)} train / {len(val_texts)} val")

    train_loader = create_document_loader(
        train_texts, tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        stride=stride, shuffle=True, drop_last=True,
        min_tail_len=min_tail_len, lazy=True,
    )
    val_loader = create_document_loader(
        val_texts, tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        stride=SEQ_LEN, shuffle=False, drop_last=False,
        min_tail_len=min_tail_len, lazy=True,
    )

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        output_dir=f"./checkpoints_smoke_{label}_s{stride}",
        example_prompts=["Once upon a time", "The cat sat"],
        generated_example_length=20,
        grad_accum_steps=1,
        use_amp=False,
        min_tail_len=min_tail_len,
        verbose=True,   # ← LIVE PRINTING
    )

    history = trainer.train(num_epochs=N_EPOCHS, eval_every=max(1, N_EPOCHS // 3))

    train_loss = history["train_loss"][-1] if history.get("train_loss") else float("inf")
    val_loss = history["val_loss"][-1] if history.get("val_loss") else float("inf")
    ppl = math.exp(min(val_loss, 20)) if val_loss != float("inf") else float("inf")

    print(f"[{label}] RESULT | train={train_loss:.4f} | val={val_loss:.4f} | ppl={ppl:.2f}", flush=True)
    return train_loss, val_loss, ppl


def test_long_stride_equivalence(texts, tokenizer):
    banner("TEST: Long texts (>= seq_len) — stride equivalence")
    split = int(len(texts) * 0.9)
    train, val = texts[:split], texts[split:]

    cfg_a = build_cfg(len(tokenizer))
    cfg_a.pad_token_id = tokenizer.pad_token_id
    cfg_a.eos_token_id = tokenizer.eos_token_id

    # Run A: no overlap
    print("\n>>> Run A: stride=32 (no overlap)", flush=True)
    model_a = HelixForCausalLM(cfg_a)
    loss_a, _, ppl_a = run_trainer(
        model_a, cfg_a, tokenizer, train, val,
        stride=32, min_tail_len=SEQ_LEN // 4, label="longA"
    )

    # Fresh config copy for Run B (defensive against mutation from save_pretrained)
    cfg_b = copy.deepcopy(cfg_a)

    # Run B: 50 % overlap
    print("\n>>> Run B: stride=16 (50% overlap)", flush=True)
    model_b = HelixForCausalLM(cfg_b)
    loss_b, _, ppl_b = run_trainer(
        model_b, cfg_b, tokenizer, train, val,
        stride=16, min_tail_len=SEQ_LEN // 4, label="longB"
    )

    baseline = math.log(len(tokenizer))
    assert loss_a < baseline * 0.95, f"stride=32 not learning ({loss_a:.4f} >= {baseline:.4f})"
    assert loss_b < baseline * 0.95, f"stride=16 not learning ({loss_b:.4f} >= {baseline:.4f})"

    ppl_ratio = max(ppl_a, ppl_b) / min(ppl_a, ppl_b) if min(ppl_a, ppl_b) > 0 else 1.0
    assert ppl_ratio < 2.0, f"stride equivalence broken (ppl ratio {ppl_ratio:.2f})"

    banner(f"[PASS] stride=32 vs stride=16 equivalent (ppl ratio {ppl_ratio:.2f})")
    return True


def test_short_equivalence(texts, tokenizer):
    banner("TEST: Short texts (< seq_len) — Dataset vs raw list equivalence")
    split = int(len(texts) * 0.9)
    train, val = texts[:split], texts[split:]

    cfg = build_cfg(len(tokenizer))
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id

    # Path A: explicit DocumentAwareDataset + DataLoader
    print("\n>>> Path A: Explicit DocumentAwareDataset (custom loaders)", flush=True)
    train_loader_a = create_document_loader(
        train, tokenizer, SEQ_LEN, BATCH_SIZE, stride=SEQ_LEN,
        shuffle=True, drop_last=True, min_tail_len=1, lazy=True,
    )
    val_loader_a = create_document_loader(
        val, tokenizer, SEQ_LEN, BATCH_SIZE, stride=SEQ_LEN,
        shuffle=False, drop_last=False, min_tail_len=1, lazy=True,
    )

    model_a = HelixForCausalLM(cfg)
    trainer_a = Trainer(
        model=model_a, cfg=cfg,
        train_loader=train_loader_a, val_loader=val_loader_a,
        tokenizer=tokenizer,
        output_dir="./checkpoints_smoke_short_explicit",
        example_prompts=["Once upon a time"],
        generated_example_length=20,
        grad_accum_steps=1, use_amp=False, min_tail_len=1,
        verbose=True,
    )
    hist_a = trainer_a.train(num_epochs=N_EPOCHS, eval_every=5)
    loss_a = hist_a["train_loss"][-1] if hist_a.get("train_loss") else float("inf")

    # Path B: raw list (Trainer internally creates DocumentAwareDataset)
    print("\n>>> Path B: Raw text list (Trainer internal dataset creation)", flush=True)
    model_b = HelixForCausalLM(copy.deepcopy(cfg))
    trainer_b = Trainer(
        model=model_b, cfg=copy.deepcopy(cfg),
        train_texts=train, val_texts=val,
        tokenizer=tokenizer,
        output_dir="./checkpoints_smoke_short_raw",
        example_prompts=["Once upon a time"],
        generated_example_length=20,
        grad_accum_steps=1, use_amp=False, min_tail_len=1,
        verbose=True,
    )
    hist_b = trainer_b.train(num_epochs=N_EPOCHS, eval_every=5)
    loss_b = hist_b["train_loss"][-1] if hist_b.get("train_loss") else float("inf")

    print(f"\n[explicit] train_loss={loss_a:.4f}")
    print(f"[raw_list] train_loss={loss_b:.4f}")

    assert not math.isinf(loss_a) and not math.isnan(loss_a), "Path A diverged"
    assert not math.isinf(loss_b) and not math.isnan(loss_b), "Path B diverged"

    ratio = max(loss_a, loss_b) / min(loss_a, loss_b) if min(loss_a, loss_b) > 0 else 1.0
    assert ratio < 1.5, f"Dataset vs raw diverged (ratio {ratio:.2f})"

    banner(f"[PASS] explicit vs raw list equivalent (loss ratio {ratio:.2f})")
    return True


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    banner("SETUP")
    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}", flush=True)

    print(f"Loading dataset (first {DATASET_SLICES} samples)...", flush=True)
    ds = load_dataset("david-thrower/tiny-stories-mini-96-seq-len-50000-samples")
    texts_all = ds["train"]["text"][:DATASET_SLICES]

    # Tokenize via pandas (as requested)
    df = pd.DataFrame({"text": texts_all})
    print("Tokenizing for cohort selection...", flush=True)
    df["n_tokens"] = df["text"].apply(
        lambda t: len(tokenizer.encode(t, add_special_tokens=False))
    )
    counts = df["n_tokens"].tolist()

    texts_long = select_long_mixed(texts_all, counts, SEQ_LEN, N_SAMPLES)
    texts_short = select_short(texts_all, counts, SEQ_LEN, N_SAMPLES)

    long_counts = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts_long]
    short_counts = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts_short]

    print(f"Long cohort:  {len(texts_long)} samples  (tokens: {min(long_counts) if long_counts else 'n/a'}–{max(long_counts) if long_counts else 'n/a'})", flush=True)
    print(f"Short cohort: {len(texts_short)} samples  (tokens: {min(short_counts) if short_counts else 'n/a'}–{max(short_counts) if short_counts else 'n/a'})", flush=True)

    # Guard against empty cohorts
    assert len(texts_long) >= 10, f"Need >=10 long texts, got {len(texts_long)}"
    assert len(texts_short) >= 10, f"Need >=10 short texts, got {len(texts_short)}"

    results = {
        "long_stride_equiv": test_long_stride_equivalence(texts_long, tokenizer),
        "short_ds_equiv": test_short_equivalence(texts_short, tokenizer),
    }

    banner("FINAL SUMMARY")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}", flush=True)

    all_ok = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_ok else 'SOME FAILED'}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
