#!/usr/bin/env python3
"""
Minimal CPU demo for ErrorCorrectingMultiScaleAttnNode.
Tests multi-scale windowed attention on a small model (seq_len=96).
"""
import random
import os
import sys
from math import ceil

from datasets import load_dataset
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer


EPOCHS = 10
MAX_SEQ_LEN = 96
NUM_SAMPLES = 1_500
VAL_SPLIT = 0.2

EXAMPLE_PROMPTS = [
    "The next day, something unexpected",
    "I have an idea, Ben. Let\'s build a",
    "The oyster and its friends decided to make"
]
GENERATED_EXAMPLE_LENGTH = 50


def main():
    random.seed(42)
    torch.manual_seed(42)

    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")

    # small_v2 defaults: d_model=256, n_heads=4
    # Scale windows to fit 96-length sequences
    cfg = HelixConfig.small_v2(
        lr=1.5e-4,
        vocab_size=vocab_size,
        seq_len=MAX_SEQ_LEN,
        tokenizer_name="gpt2",
        use_titans_memory=False,
        n_loops=3,
        attention_mode="multi_scale_windowed",
        local_window=32,
        coarse_window=48,
        compressed_windows=16,
        compressed_views=8,
        corrector_dim=128,      # d_model // 2
        output_ffn_dim=1024,    # 4 * d_model
        consensus_type="cosine",
        corrector_type="ffn",
        dropout=0.1,
        attn_dropout=0.1,
        ffn_expansion=4.0,
    )

    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.bos_token_id = tokenizer.bos_token_id

    model = HelixForCausalLM(cfg)
    params = model.count_parameters()
    print(f"Parameters: {params['total']:,}")

    # Data
    ds = load_dataset("david-thrower/tiny-stories-mini-96-seq-len-50000-samples")
    texts = ds['train']['text'][:NUM_SAMPLES]

    random.seed(42)
    random.shuffle(texts)

    split_idx = ceil(NUM_SAMPLES * (1 - VAL_SPLIT))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        output_dir="./checkpoints_multiscale",
        example_prompts=EXAMPLE_PROMPTS,
        generated_example_length=GENERATED_EXAMPLE_LENGTH,
    )

    history = trainer.train(num_epochs=EPOCHS)

    model.save_pretrained("./helix-multiscale-demo")
    print(f"\nModel saved to ./helix-multiscale-demo")

    print("\n--- Generation ---")
    prompt = "In 1492,"
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
    generated = model.generate_ext(input_ids, max_new_tokens=25, temperature=0.8)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"  '{prompt}' -> '{text}'")


if __name__ == "__main__":
    main()
