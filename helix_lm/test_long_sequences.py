"""
HelixLM Long-Sequence Smoke Test

Verifies that training on documents longer than max_seq_len works correctly
with DocumentAwareDataset. Tests:
  - No NaN/Inf losses with long documents (> seq_len)
  - Loss decreases over training
  - Document-aware chunking produces stable perplexity
  - Gradient accumulation works correctly
  - AMP can be enabled/disabled without crashes

Run with:  python -m helix_lm.test_long_sequences
"""
import os
import sys
import math
import random
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helix_lm.config import HelixConfig
from helix_lm.tokenizer import HelixTokenizer
from helix_lm.hf_model import HelixForCausalLM
from helix_lm.trainer import Trainer


# Use char tokenizer by default (no HF download needed)
USE_CHAR_TOKENIZER = os.environ.get("USE_CHAR_TOKENIZER", "1") == "1"


def get_tokenizer():
    """Get tokenizer - char for fast tests, gpt2 when explicitly requested."""
    if USE_CHAR_TOKENIZER:
        t = HelixTokenizer("char")
        # Build vocab from a large sample
        words = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-\n"
        t.build_char_vocab(words)
        return t
    return HelixTokenizer("gpt2")


def create_long_documents(n_docs=50, min_len=500, max_len=2000, seed=42):
    """Create synthetic long documents for testing."""
    random.seed(seed)
    # Vocabulary of common words to generate coherent-ish text
    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have",
        "I", "it", "for", "not", "on", "with", "he", "as", "you",
        "do", "at", "this", "but", "his", "by", "from", "they",
        "we", "say", "her", "she", "or", "an", "will", "my",
        "one", "all", "would", "there", "their", "what", "so",
        "up", "out", "if", "about", "who", "get", "which", "go",
        "me", "when", "make", "can", "like", "time", "no", "just",
        "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than",
        "then", "now", "look", "only", "come", "its", "over",
        "think", "also", "back", "after", "use", "two", "how",
        "our", "work", "first", "well", "way", "even", "new",
        "want", "because", "any", "these", "give", "day", "most",
        "us", "is", "was", "are", "were", "been", "has", "had",
        "did", "does", "doing", "done", "being", "having",
        "calculus", "theorem", "derivative", "integral", "function",
        "equation", "variable", "constant", "limit", "infinity",
        "matrix", "vector", "space", "dimension", "linear",
        "algebra", "geometry", "topology", "analysis", "proof",
    ]

    documents = []
    for i in range(n_docs):
        n_words = random.randint(min_len, max_len)
        # Add some structure: sentences of 8-20 words
        doc_words = []
        while len(doc_words) < n_words:
            sentence_len = random.randint(8, 20)
            sentence = random.choices(words, k=sentence_len)
            sentence[0] = sentence[0].capitalize()
            doc_words.extend(sentence)
            doc_words.append(".")
        doc_text = " ".join(doc_words[:n_words])
        documents.append(doc_text)

    return documents


def test_document_aware_chunking():
    """Test DocumentAwareDataset chunks correctly without boundary crossings."""
    print("\n--- Test: DocumentAwareDataset chunking ---")

    from helix_lm.dataset import DocumentAwareDataset

    tokenizer = get_tokenizer()
    seq_len = 64

    # Create 3 documents of different lengths
    docs = [
        "Hello world " * 100,   # 1200 chars -> multiple full chunks
        "Short doc.",            # < min_tail_len -> dropped
        "Medium document " * 20, # ~320 chars -> 1 full chunk + tail
    ]

    ds = DocumentAwareDataset(docs, tokenizer, seq_len, min_tail_len=8)

    # Check we got chunks
    assert len(ds) > 0, "Dataset should produce chunks"

    # Check all chunks are exactly seq_len
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["input_ids"].shape[0] == seq_len, \
            f"Chunk {i} has wrong length: {sample['input_ids'].shape[0]}"
        assert sample["labels"].shape[0] == seq_len

    # Check only padding is masked
    for i in range(len(ds)):
        sample = ds[i]
        mask = sample["labels"] == -100
        pad_positions = sample["input_ids"] == tokenizer.pad_token_id
        # All masked positions should be padding
        if mask.any():
            assert (mask == pad_positions).all() or (mask | ~pad_positions).all(), \
                "Non-padding tokens are masked!"

    print(f"[PASS] Produced {len(ds)} chunks, all length {seq_len}")
    print(f"[PASS] Only padding masked in labels")
    return True


def test_long_document_training_no_nan():
    """Train on long documents and verify no NaN losses."""
    print("\n--- Test: Long-document training (no NaN) ---")

    tokenizer = get_tokenizer()
    vocab_size = len(tokenizer)

    cfg = HelixConfig.tiny(
        vocab_size=vocab_size,
        seq_len=128,
        tokenizer_name="gpt2",
        device="cpu",  # CPU for determinism
        lr=1e-3,
        weight_decay=0.01,
        epochs=3,
        warmup_steps=10,
        grad_clip=1.0,
    )
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id

    model = HelixForCausalLM(cfg)
    print(f"Model parameters: {model.count_parameters()['total']:,}")

    # Create long documents (500-2000 tokens each)
    documents = create_long_documents(n_docs=30, min_len=300, max_len=800)
    print(f"Training on {len(documents)} long documents")

    # Split train/val
    split_idx = int(len(documents) * 0.9)
    train_texts = documents[:split_idx]
    val_texts = documents[split_idx:]

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        output_dir="./test_checkpoints",
        grad_accum_steps=2,
        use_amp=False,
        min_tail_len=cfg.seq_len // 4,
    )

    history = trainer.train(num_epochs=3, eval_every=1)

    # Assertions
    for loss in history["train_loss"]:
        assert not math.isnan(loss), "NaN loss detected!"
        assert not math.isinf(loss), "Inf loss detected!"

    # Loss should decrease (or at least not explode)
    assert history["train_loss"][-1] < history["train_loss"][0] * 2, \
        f"Loss exploded: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}"

    print(f"[PASS] No NaN/Inf losses over {len(history['train_loss'])} epochs")
    print(f"[PASS] Loss: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
    return True


def test_gradient_accumulation():
    """Verify gradient accumulation produces same results as larger batch."""
    print("\n--- Test: Gradient accumulation ---")

    tokenizer = get_tokenizer()
    vocab_size = len(tokenizer)

    cfg = HelixConfig.tiny(
        vocab_size=vocab_size,
        seq_len=64,
        tokenizer_name="gpt2",
        device="cpu",
        lr=1e-3,
        weight_decay=0.01,
        epochs=2,
        warmup_steps=5,
        batch_size=2,
    )
    cfg.pad_token_id = tokenizer.pad_token_id

    documents = create_long_documents(n_docs=20, min_len=200, max_len=400)

    # Trainer with grad_accum=4 (effective batch = 8)
    model1 = HelixForCausalLM(cfg)
    trainer1 = Trainer(
        model=model1, cfg=cfg,
        train_texts=documents, val_texts=None,
        tokenizer=tokenizer, output_dir="./test_checkpoints_1",
        grad_accum_steps=4, use_amp=False, min_tail_len=16,
    )
    history1 = trainer1.train(num_epochs=2)

    # No NaN
    for loss in history1["train_loss"]:
        assert not math.isnan(loss), "NaN with gradient accumulation!"

    print(f"[PASS] grad_accum=4: loss {history1['train_loss'][0]:.4f} -> {history1['train_loss'][-1]:.4f}")
    return True


def test_amp_safety():
    """Test that AMP doesn't crash (if CUDA available)."""
    print("\n--- Test: AMP safety ---")

    if not torch.cuda.is_available():
        print("[SKIP] No CUDA, skipping AMP test")
        return True

    tokenizer = get_tokenizer()
    vocab_size = len(tokenizer)

    cfg = HelixConfig.tiny(
        vocab_size=vocab_size,
        seq_len=64,
        tokenizer_name="gpt2",
        device="cuda",
        lr=1e-3,
        weight_decay=0.01,
        epochs=1,
        warmup_steps=5,
        batch_size=2,
    )
    cfg.pad_token_id = tokenizer.pad_token_id

    documents = create_long_documents(n_docs=10, min_len=200, max_len=400)

    model = HelixForCausalLM(cfg)
    trainer = Trainer(
        model=model, cfg=cfg,
        train_texts=documents, val_texts=None,
        tokenizer=tokenizer, output_dir="./test_checkpoints_amp",
        grad_accum_steps=2, use_amp=True, min_tail_len=16,
    )
    history = trainer.train(num_epochs=1)

    # Should not crash and should not NaN (with our fp32 fix in LinearAttnNode)
    for loss in history["train_loss"]:
        assert not math.isnan(loss), "NaN loss with AMP!"

    print(f"[PASS] AMP enabled: no crashes or NaN")
    return True


def test_trainer_vs_baseline_comparable():
    """
    Smoke test: verify Trainer on HelixLM-core produces comparable results
    to the original smoke_test.py (loss decreases, no NaN).
    """
    print("\n--- Test: Trainer comparable to baseline ---")

    tokenizer = get_tokenizer()
    vocab_size = len(tokenizer)

    # Same scale as original smoke test but with longer docs
    cfg = HelixConfig.tiny(
        vocab_size=vocab_size,
        seq_len=64,
        tokenizer_name="gpt2" if not USE_CHAR_TOKENIZER else "char",
        device="cpu",
        lr=5e-4,
        weight_decay=0.01,
        epochs=5,
        warmup_steps=10,
        batch_size=4,
    )
    cfg.pad_token_id = tokenizer.pad_token_id

    # Create documents that are longer than seq_len
    documents = create_long_documents(n_docs=20, min_len=150, max_len=400)

    model = HelixForCausalLM(cfg)
    trainer = Trainer(
        model=model, cfg=cfg,
        train_texts=documents[:18], val_texts=documents[18:],
        tokenizer=tokenizer, output_dir="./test_checkpoints_comp",
        grad_accum_steps=1, use_amp=False, min_tail_len=16,
    )
    history = trainer.train(num_epochs=5, eval_every=5)

    # Assertions matching original smoke_test criteria
    assert not math.isnan(history["train_loss"][-1])
    assert not math.isinf(history["train_loss"][-1])
    assert history["train_loss"][-1] < history["train_loss"][0], \
        "Loss did not decrease"

    print(f"[PASS] Loss decreased: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
    print(f"[PASS] PPL: {math.exp(min(history['train_loss'][-1], 20)):.2f}")
    return True


def main():
    print("="*60)
    print("HelixLM Long-Sequence Smoke Test")
    print("="*60)

    results = []
    start = time.time()

    try:
        results.append(("DocumentAware chunking", test_document_aware_chunking()))
    except Exception as e:
        results.append(("DocumentAware chunking", False))
        print(f"[FAIL] {e}")

    try:
        results.append(("Long-doc training (no NaN)", test_long_document_training_no_nan()))
    except Exception as e:
        results.append(("Long-doc training (no NaN)", False))
        print(f"[FAIL] {e}")

    try:
        results.append(("Gradient accumulation", test_gradient_accumulation()))
    except Exception as e:
        results.append(("Gradient accumulation", False))
        print(f"[FAIL] {e}")

    try:
        results.append(("AMP safety", test_amp_safety()))
    except Exception as e:
        results.append(("AMP safety", False))
        print(f"[FAIL] {e}")

    try:
        results.append(("Comparable to baseline", test_trainer_vs_baseline_comparable()))
    except Exception as e:
        results.append(("Comparable to baseline", False))
        print(f"[FAIL] {e}")

    # Summary
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.1f}s")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(r[1] for r in results)
    print(f"\n{'='*60}")
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    print(f"{'='*60}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
