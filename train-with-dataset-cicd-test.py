"""
Integration test: DocumentAwareDataset + Trainer with mixed-length texts.

Validates:
1. Long texts (>= seq_len): stride=32 (no overlap) vs stride=16 (50% overlap)
   trained for 15 epochs via the official Trainer — no substantial degradation.
2. Short texts (< seq_len): explicit DocumentAwareDataset vs raw text list
   passed to Trainer — equivalent results.
3. 15-epoch training stability end-to-end.
"""
import sys
import os
import math
import random

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer
from helix_lm.dataset import DocumentAwareDataset, create_document_loader


SEED = 42
SEQ_LEN = 32
N_SAMPLES = 500
N_EPOCHS = 15
BATCH_SIZE = 8
DATASET_SLICES = 5_000


def get_token_counts(texts, tokenizer):
    """Return list of token counts for each raw text."""
    return [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]


def select_long_mixed(texts, counts, seq_len, n=N_SAMPLES):
    """
    Select a mix of barely-longer (seq_len … 1.5×) and considerably-longer
    (> 2× seq_len) samples.
    """
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

    # back-fill if pool was lopsided
    need = n - len(chosen)
    pool = [t for t, c in zip(texts, counts) if c >= seq_len and t not in chosen]
    if need > 0 and pool:
        chosen.extend(random.sample(pool, min(need, len(pool))))

    random.shuffle(chosen)
    return chosen[:n]


def select_short(texts, counts, seq_len, n=N_SAMPLES):
    """Select texts strictly shorter than seq_len."""
    pool = [t for t, c in zip(texts, counts) if 0 < c < seq_len]
    if len(pool) >= n:
        return random.sample(pool, n)
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


def run_trainer(
    model,
    cfg,
    tokenizer,
    train_texts,
    val_texts,
    stride,
    min_tail_len,
    label,
):
    """
    Run the official Trainer for N_EPOCHS.
    Returns (final_train_loss, final_val_loss, final_ppl).
    """
    # Explicit DataLoader path (at-scale style)
    train_loader = create_document_loader(
        train_texts,
        tokenizer,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        stride=stride,
        shuffle=True,
        drop_last=True,
        min_tail_len=min_tail_len,
        lazy=True,
    )
    val_loader = create_document_loader(
        val_texts,
        tokenizer,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        stride=SEQ_LEN,  # no overlap for eval
        shuffle=False,
        drop_last=False,
        min_tail_len=min_tail_len,
        lazy=True,
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
        verbose=False,
    )

    history = trainer.train(num_epochs=N_EPOCHS, eval_every=max(1, N_EPOCHS // 3))

    train_loss = history["train_loss"][-1] if history.get("train_loss") else float("inf")
    val_loss = history["val_loss"][-1] if history.get("val_loss") else float("inf")
    ppl = math.exp(min(val_loss, 20)) if val_loss != float("inf") else float("inf")

    print(f"[{label}] stride={stride} | train={train_loss:.4f} | val={val_loss:.4f} | ppl={ppl:.2f}")
    return train_loss, val_loss, ppl


def test_long_stride_equivalence(texts, tokenizer, cfg):
    """Compare stride=32 vs stride=16 on long texts."""
    print("\n" + "=" * 60)
    print("TEST: Long texts (>= seq_len) — stride equivalence")
    print("=" * 60)

    split = int(len(texts) * 0.9)
    train, val = texts[:split], texts[split:]

    # Run A: no overlap
    model_a = HelixForCausalLM(cfg)
    loss_a, _, ppl_a = run_trainer(
        model_a, cfg, tokenizer, train, val, stride=32, min_tail_len=SEQ_LEN // 4, label="long"
    )

    # Run B: 50 % overlap
    model_b = HelixForCausalLM(cfg)
    loss_b, _, ppl_b = run_trainer(
        model_b, cfg, tokenizer, train, val, stride=16, min_tail_len=SEQ_LEN // 4, label="long"
    )

    # Assertions
    baseline = math.log(cfg.vocab_size)
    assert loss_a < baseline * 0.9, "stride=32 not learning"
    assert loss_b < baseline * 0.9, "stride=16 not learning"

    ppl_ratio = max(ppl_a, ppl_b) / min(ppl_a, ppl_b) if min(ppl_a, ppl_b) > 0 else 1.0
    assert ppl_ratio < 2.0, f"stride equivalence broken (ppl ratio {ppl_ratio:.2f})"

    print(f"[PASS] stride=32 vs stride=16 equivalent (ppl ratio {ppl_ratio:.2f})")
    return True


def test_short_equivalence(texts, tokenizer, cfg):
    """
    Compare:
      A) Explicit DocumentAwareDataset passed as DataLoader
      B) Raw text list passed to Trainer (internal dataset creation)
    """
    print("\n" + "=" * 60)
    print("TEST: Short texts (< seq_len) — Dataset vs raw list equivalence")
    print("=" * 60)

    split = int(len(texts) * 0.9)
    train, val = texts[:split], texts[split:]

    # Path A: explicit DocumentAwareDataset + DataLoader
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
        verbose=False,
    )
    hist_a = trainer_a.train(num_epochs=N_EPOCHS, eval_every=5)
    loss_a = hist_a["train_loss"][-1] if hist_a.get("train_loss") else float("inf")

    # Path B: raw list (Trainer internally creates DocumentAwareDataset)
    model_b = HelixForCausalLM(cfg)
    trainer_b = Trainer(
        model=model_b, cfg=cfg,
        train_texts=train, val_texts=val,
        tokenizer=tokenizer,
        output_dir="./checkpoints_smoke_short_raw",
        example_prompts=["Once upon a time"],
        generated_example_length=20,
        grad_accum_steps=1, use_amp=False, min_tail_len=1,
        verbose=False,
    )
    hist_b = trainer_b.train(num_epochs=N_EPOCHS, eval_every=5)
    loss_b = hist_b["train_loss"][-1] if hist_b.get("train_loss") else float("inf")

    print(f"[explicit] train_loss={loss_a:.4f}")
    print(f"[raw_list] train_loss={loss_b:.4f}")

    assert not math.isinf(loss_a) and not math.isnan(loss_a), "explicit path diverged"
    assert not math.isinf(loss_b) and not math.isnan(loss_b), "raw list path diverged"

    ratio = max(loss_a, loss_b) / min(loss_a, loss_b) if min(loss_a, loss_b) > 0 else 1.0
    assert ratio < 1.5, f"Dataset vs raw diverged (ratio {ratio:.2f})"

    print(f"[PASS] explicit vs raw list equivalent (loss ratio {ratio:.2f})")
    return True


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. Tokenizer
    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")

    # 2. Load dataset slice
    print(f"Loading dataset (first {DATASET_SLICES} samples)...")
    ds = load_dataset("david-thrower/tiny-stories-mini-96-seq-len-50000-samples")
    texts_all = ds["train"]["text"][:DATASET_SLICES]

    # 3. Tokenize via pandas (as requested)
    df = pd.DataFrame({"text": texts_all})
    df["n_tokens"] = df["text"].apply(
        lambda t: len(tokenizer.encode(t, add_special_tokens=False))
    )
    counts = df["n_tokens"].tolist()

    # 4. Select cohorts
    texts_long = select_long_mixed(texts_all, counts, SEQ_LEN, N_SAMPLES)
    texts_short = select_short(texts_all, counts, SEQ_LEN, N_SAMPLES)

    long_counts = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts_long]
    short_counts = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts_short]

    print(f"\nLong cohort:  {len(texts_long)} samples  (tokens: {min(long_counts)}–{max(long_counts)})")
    print(f"Short cohort: {len(texts_short)} samples  (tokens: {min(short_counts) if short_counts else 0}–{max(short_counts) if short_counts else 0})")

    cfg = build_cfg(vocab_size)
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id

    # 5. Run tests
    results = {
        "long_stride_equiv": test_long_stride_equivalence(texts_long, tokenizer, cfg),
        "short_ds_equiv": test_short_equivalence(texts_short, tokenizer, cfg),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
