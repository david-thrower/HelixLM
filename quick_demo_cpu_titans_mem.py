"""
Example: Training HelixLM with HuggingFace integration.

Demonstrates:
  - GPT-2 tokenizer
  - HelixForCausalLM HF model
  - Streaming dataset
  - save_pretrained / push_to_hub
"""
import random
import os
import sys
from math import ceil

from datasets import load_dataset
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer


EPOCHS = 25
MAX_SEQ_LEN = 96

NUM_SAMPLES = 1_000
VAL_SPLIT = 0.2

EXAMPLE_PROMPTS = [
        "The next day, something unexpected",
        "I have an idea, Ben. Let\'s build a",
        "The oyster and its friends decided to make"
]
GENERATED_EXAMPLE_LENGTH = 50


def main():


    # ------------------------------------------------------------------
    # Tokenizer setup
    # ------------------------------------------------------------------

    tokenizer = HelixTokenizer("gpt2")
    VOCABULARY_SIZE = len(tokenizer)
    print(f"Vocab size: {VOCABULARY_SIZE}")

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    # Config
    cfg = HelixConfig.tiny(
        vocab_size=VOCABULARY_SIZE,
        seq_len=MAX_SEQ_LEN,
        tokenizer_name="gpt2",
        use_titans_memory=True)

    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.bos_token_id = tokenizer.bos_token_id
  
    model = HelixForCausalLM(cfg)
    print(f"Parameters: {model.count_parameters()['total']:,}")

    # Data
    # Load data
    ds = load_dataset("david-thrower/tiny-stories-mini-96-seq-len-50000-samples")
    texts = ds['train']['text'][:NUM_SAMPLES]

    # Shuffle before splitting (set seed for reproducibility)
    random.seed(42)
    random.shuffle(texts)

    # Split
    SPLIT_IDX = ceil(NUM_SAMPLES  * (1 - VAL_SPLIT))     
    train_texts = texts[:SPLIT_IDX]
    val_texts = texts[SPLIT_IDX:]

    # Train
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        output_dir="./checkpoints",
        example_prompts=EXAMPLE_PROMPTS,
        generated_example_length=GENERATED_EXAMPLE_LENGTH
    )

    history = trainer.train(num_epochs=EPOCHS)

    # Save to HF format
    model.save_pretrained("./helix-gpt2-small")
    print(f"\nModel saved to ./helix-gpt2-small")

    # Generate
    print("\n--- Generation ---")
    prompt = "In 1492,"
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
    generated = model.generate_ext(input_ids, max_new_tokens=25, temperature=0.8)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"  '{prompt}' -> '{text}'")


if __name__ == "__main__":
    main()
