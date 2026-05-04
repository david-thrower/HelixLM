"""
Example: Training HelixLM with HuggingFace integration.

Demonstrates:
  - GPT-2 tokenizer
  - HelixForCausalLM HF model
  - Document-aware dataset with lazy loading
  - save_pretrained / push_to_hub
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer


def main():
    # Config
    cfg = HelixConfig.small(
        vocab_size=50257,
        seq_len=256,
        tokenizer_name="gpt2",
        use_titans_memory=False,
    )

    # Tokenizer
    tokenizer = HelixTokenizer("gpt2")
    cfg.vocab_size = tokenizer.vocab_size
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.bos_token_id = tokenizer.bos_token_id

    # Model
    model = HelixForCausalLM(cfg)
    print(f"Parameters: {model.count_parameters()['total']:,}")

    # Data
    texts = [
        "In 1492, Christopher Columbus sailed across the Atlantic Ocean.",
        "The exchange of plants and animals transformed societies.",
        "The Renaissance was a period of great cultural change.",
    ] * 100

    # Train
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_texts=texts,
        val_texts=texts[-10:],
        tokenizer=tokenizer,
        output_dir="./checkpoints",
    )

    history = trainer.train(num_epochs=10)

    # Save to HF format
    model.save_pretrained("./helix-gpt2-small")
    print(f"\nModel saved to ./helix-gpt2-small")

    # Generate
    print("\n--- Generation ---")
    prompt = "In 1492,"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    generated = model.generate_ext(input_ids, max_new_tokens=20, temperature=0.8)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"  '{prompt}' -> '{text}'")


if __name__ == "__main__":
    import torch
    main()
