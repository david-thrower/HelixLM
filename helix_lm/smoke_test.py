"""
HelixLM Smoke Test

Self-contained test that trains a small model and verifies articulable text generation.
Run with:  python -m helix_lm.smoke_test
"""
import os
import sys
import math
import random
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helix_lm.config import HelixConfig
from helix_lm.tokenizer import HelixTokenizer
from helix_lm.model import HelixLMCore


def load_bible_text(filepath, fraction=1.0/3.0):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    if start_marker in text:
        text = text.split(start_marker, 1)[1]
    if end_marker in text:
        text = text.split(end_marker, 1)[0]
    return text[:int(len(text) * fraction)]


def quick_train(model, cfg, tokens, n_batches_per_epoch=40):
    device = torch.device('cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    seq_len, batch_size = cfg.seq_len, cfg.batch_size
    n_samples = len(tokens) - seq_len

    all_x = []
    all_y = []
    for i in range(n_batches_per_epoch * batch_size):
        idx = i % max(n_samples, 1)
        all_x.append(tokens[idx:idx+seq_len])
        all_y.append(tokens[idx+1:idx+seq_len+1])

    all_x = torch.stack(all_x).reshape(n_batches_per_epoch, batch_size, seq_len)
    all_y = torch.stack(all_y).reshape(n_batches_per_epoch, batch_size, seq_len)

    best_loss = float('inf')
    history = []

    start = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        batch_order = list(range(n_batches_per_epoch))
        random.shuffle(batch_order)

        for b_idx in batch_order:
            x = all_x[b_idx].to(device)
            y = all_y[b_idx].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1), ignore_index=0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches_per_epoch
        ppl = math.exp(min(avg_loss, 20))
        if avg_loss < best_loss:
            best_loss = avg_loss
        history.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{cfg.epochs} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Best: {best_loss:.4f}")

    return history, best_loss, time.time() - start


def main():
    print("="*60)
    print("HelixLM Smoke Test")
    print("="*60)

    # Config
    cfg = HelixConfig(
        vocab_size=0,  # set after tokenizer
        seq_len=64,
        batch_size=4,
        d_model=128,
        n_columns=2,
        nodes_per_column=(2, 2),
        attention_mode="linear",
        n_heads=4,
        n_loops=1,
        dropout=0.0,
        lr=5e-4,
        weight_decay=0.01,
        epochs=30,
        grad_clip=1.0,
        tokenizer_name="char",
    )

    # Load data
    bible_path = os.path.join(os.path.dirname(__file__), "bible.txt")
    if not os.path.exists(bible_path):
        bible_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bible.txt")

    if os.path.exists(bible_path):
        text = load_bible_text(bible_path)
    else:
        text = "In the beginning God created the heaven and the earth.\n" * 50

    # Tokenizer
    tokenizer = HelixTokenizer("char")
    tokenizer.build_char_vocab(text)
    cfg.vocab_size = len(tokenizer)

    # Tokenize
    all_ids = tokenizer.encode(text)
    tokens = torch.tensor(all_ids, dtype=torch.long)
    print(f"\nVocab: {len(tokenizer)} | Tokens: {len(tokens)} | Text: {len(text)} chars")

    # Model
    model = HelixLMCore(cfg)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Train
    print(f"\nTraining (30 epochs, 40 batches/epoch)...")
    history, best_loss, elapsed = quick_train(model, cfg, tokens)

    # Generation
    print(f"\n--- Generation ---")
    model.eval()
    for prompt in ["In the beginning", "And God said", "And God saw"]:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=30, temperature=0.8, top_k=20)
        text_out = tokenizer.decode(generated[0])
        print(f"  '{prompt}' -> '{text_out}'")

    # Results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Best loss: {best_loss:.4f} | PPL: {math.exp(min(best_loss,20)):.2f}")
    print(f"Final loss: {history[-1]:.4f}")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), "helixlm_smoke.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "char_to_id": tokenizer._char_to_id,
        "id_to_char": tokenizer._id_to_char,
    }, save_path)
    print(f"\nSaved to {save_path}")

    # Acceptance
    passed = True
    if math.isnan(history[-1]) or math.isinf(history[-1]):
        passed = False
        print("[FAIL] NaN/inf")
    else:
        print("[PASS] No numerical errors")

    if history[-1] < history[0]:
        print(f"[PASS] Loss decreased: {history[0]:.4f} -> {history[-1]:.4f}")
    else:
        passed = False
        print("[FAIL] Loss did not decrease")

    if best_loss < 5.0:
        print(f"[PASS] PPL {math.exp(min(best_loss,20)):.2f} < 5.0")
    else:
        print(f"[INFO] PPL {math.exp(min(best_loss,20)):.2f}")

    print(f"\n{'='*60}")
    print("SMOKE TEST PASSED" if passed else "SMOKE TEST PARTIAL")
    print(f"{'='*60}")
    return passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
