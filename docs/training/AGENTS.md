# Training documentation guide

## API and Hardware Compatibility Pointers

- `helix_lm.trainer.Trainer` remains the legacy document-aware SFT path. Do not
  change or rename it here; David owns the later `SFTTrainer` rename.
- Continuous, globally ordered causal pretraining belongs to
  `PretrainTrainer` and `pretrain_data.py`. Passing an `IterableColumn` to
  `PretrainTrainer` invokes the compiler automatically; do not reintroduce a
  second preprocessing entry point without a distinct operator requirement.
- For a single 16 GB consumer GPU, `113M_param_train.py` defaults to the
  `rtx5080-relative` profile: `d_model=768`, 12 heads, three columns,
  configured `(3, 3, 3)` nodes, four loops, FFN expansion 3.0, `seq_len=1024`,
  and vertical depth two. Other launchers or external run
  contracts may use different widths or topology; do not describe any of them
  as the single active experiment.
- Hardware scaling must be explicit in the launcher or run contract. Use the
  768-width baseline for constrained GPUs, and record any wider or deeper
  configuration as a separate matched run rather than silently changing the
  baseline.
- For a 16 GB GPU, prefer the checked-in 768-width configuration with the
  smallest supported microbatch, gradient accumulation, activation
  checkpointing, and reduced ablation scope. Keep sequence length, optimizer,
  tokenizer, data order, and evaluator fixed when comparing ablations.
- For an L40S or H200, consider a wider configuration such as `d_model=1024`
  with 16 heads. A four-column topology and `vertical_depth=3` may also be
  evaluated, but only as an explicitly named hardware-scaled ablation with its
  own launcher or run contract. Report memory, throughput, effective batch,
  and convergence separately from the 768-width baseline.
- Do not treat additional GPU memory as evidence that a wider or deeper model
  is comparable to the baseline. A topology change needs a matched source,
  data order, optimizer, tokenizer, sequence length, evaluator, and documented
  compute or token budget.
- Save checkpoints locally before any optional Hugging Face upload. Publication
  is never a prerequisite for local training or recovery.

## Hardware-scaled ablation guidance

Use these as starting points, not implicit defaults. Every selected value must
be recorded in the launcher and run contract.

| Hardware class | Starting configuration | Recommended ablations |
|---|---|---|
| 16 GB GPU | `d_model=768`, 12 heads, three columns, vertical depth two | Baseline versus data-order, optimizer, or regularization changes; use gradient accumulation and activation checkpointing as needed |
| L40S-class GPU | `d_model=1024`, 16 heads, three or four columns, vertical depth two | Compare width first; then test four columns and `vertical_depth=3` independently |
| H200-class GPU | `d_model=1024`, 16 heads, four columns, vertical depth two or three | Compare width, column count, and vertical depth under matched token and optimizer budgets; measure scaling efficiency |

## Pretraining data contract

```text
pinned ordered source rows
-> tokenize without tokenizer-added special tokens
-> append one EOS to each nonempty document
-> concatenate the token stream
-> emit exact non-overlapping seq\_len windows
-> discard and count the incomplete tail
-> assign stable sample IDs
-> persist one global epoch permutation
-> replay that exact order from disk
```
