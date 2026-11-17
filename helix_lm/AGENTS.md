# `helix_lm` module guide

## Where changes belong

| Concern | Module |
| --- | --- |
| Model-wide configuration | `config.py` |
| Graph construction and topology | `graph.py` |
| Node implementations | `nodes.py` |
| Hugging Face model adapter | `hf_model.py` |
| Tokenizer adapter | `tokenizer.py` |
| Supervised/document-aware data | `dataset.py` |
| Pretraining sample store and order | `pretrain_data.py` |
| Training loops | `trainer.py` |

## Trainer boundary

`Trainer` is the legacy document-aware SFT implementation. Preserve its
document boundaries, padding masks, overlap semantics, and public call shape.
Do not rename it here; David owns the later `Trainer` to `SFTTrainer` migration.

`PretrainTrainer` is the only trainer that may consume the continuous
pretraining sample store. Its canonical data contract is:

```text
ordered source rows
-> tokenize without tokenizer-added specials
-> append one EOS per nonempty document
-> concatenate
-> exact seq_len windows
-> discard the incomplete tail
-> persist stable sample IDs
-> replay a persisted global permutation
```

Checkpoint data state must identify both the compiled sample manifest and the
permutation. A model checkpoint without those roots is not an exact mid-epoch
resume receipt.

For indexed training, `pretrain_training_state.pt` is the exact local recovery
object at an optimizer-step boundary: model, optimizer, scheduler, AMP scaler
when present, Torch RNG, global step, sample cursor, manifest root, and
permutation root. A new epoch must activate a new persisted epoch permutation;
reusing the prior epoch's DataLoader is not a global-shuffle implementation.

## Deferred topology work

Do not add a fourth column or raise vertical depth to three in this lane. The
current comparison freezes three columns, `(3, 3, 3)` nodes, and vertical depth
two. Future topology work gets its own matched ablation after the pretraining
pipeline is proven.

## Verification gate — run BEFORE any paid job

Every failure in the 5B run was a verification failure, not a modelling failure.
These cost seconds; each skipped one cost a GPU job.

1. **dtype.** Load in the FINAL dtype with NO autocast — fp32 *and* bf16 — and run
   one forward. Training always uses autocast, so training cannot catch dtype
   bugs; `_safe_attention` mixed an fp32 mask with bf16 values and raised on the
   first bf16 forward.
2. **save/load.** `save_pretrained(tmp)` and `from_pretrained(tmp)` must both pass.
   transformers >= 5 validates the config on save.
3. **train returns.** `trainer.train()` must RETURN, not raise after
   `save_checkpoint`. A raise there looks like a failure on a model that is fine
   and silently skips every line after the call — including `push_to_hub`.
   `Trainer._val_shard_dir` was exactly this.
4. **throughput.** Measure on fixed GPU tensors at the production shape, never
   through a small corpus. A 1,200-document corpus gave 2,710 tok/s where compute
   is 66,904 tok/s: the loader was the bottleneck, not the GPU.
5. **budget vs store.** `EPOCHS = ceil(tokens / store_tokens)`; assert
   `global_step` reaches `TOTAL_STEPS`, or the epoch loop ends quietly mid-decay
   and still prints "Training complete".
6. **smoke in the job's shape.** Same dtype, batch, seq length, hardware. Shrink
   the step count and nothing else.

## Evaluation (lighteval 0.13.0)

Verified pin: `lighteval==0.13.0 transformers==5.23.0 datasets==5.2.0
huggingface_hub==2.4.0 xxhash==3.6.0`.

Three runtime patches are required. Treat them as a shim to be replaced by pins
or upstream fixes, not something to re-derive each time:

- `metrics_sample.py`: `HfApi.list_models(model_name=…)` → `search=` (hub 2.x
  removed the kwarg; it raises while the task registry loads, so nothing runs).
- `logging/info_loggers.py`: current xxhash rejects `str`. The shim must tolerate
  `xxh64(str)` *and* the no-argument form `datasets` fingerprinting uses.
- `_loglikelihood_tokens` (~line 1012): when a continuation is longer than the
  available logits the slice is empty and lighteval compares a 0-length argmax to
  an N-token target. **Filter those documents out.** Never return a sentinel
  score: a `-1e9` "fix" fabricated a 100% BLiMP that was published by mistake.

Task notes:
- `hellaswag` is GENERATIVE in 0.13 (exact match on a letter). A base LM scores
  ~3%, below the 25% chance line — it measures format compliance, not knowledge.
  Use a log-likelihood variant.
- `piqa` unavailable: `ybisk/piqa` is script-only and datasets 5.x refuses scripts.
- `truthfulqa:mc` reports both `truthfulqa_mc1` and `truthfulqa_mc2`.
- Results land at `<out>/results/<org>/<model>/results_*.json`, not `<out>/`.
- Sanity band for ~50M: ARC-Easy ≈ 0.48, WinoGrande ≈ chance. A number far above
  these means a harness bug, not a strong model.

## Known-good anchors

| artifact | value |
| --- | --- |
| 5B WSD run, full 59.8K-doc val | loss 3.6020 / ppl 36.67 |
| training throughput, bf16+compile, batch 8 × seq 2048 | 66,904 tok/s, 0.98 s/step, 24.0 GiB peak |
| ARC-Easy, 0-shot, log-likelihood | 0.4811 ± 0.0103 |
| params, production shape | 53,592,343 |
