# AGENTS.md — HelixLM architecture, training, and export governance

This file applies to the HelixLM source repository and every launcher, training script,
data builder, evaluation script, and export path maintained here. Exported model and
dataset repositories must also carry their own `AGENTS.md`; the stricter applicable rule
wins.

These are not style preferences. They are safeguards against wasted spend, lost training
state, false provenance, misleading model claims, and unapproved experiments.

This file tightens the 2026-11-20 Hub launch protocol in two respects: there is no
low-cost/short-run approval loophole, and token-corrected models require validated ID-based
stopping rather than a `stop_strings` field alone. If an older Hub protocol conflicts, flag
the discrepancy and follow the stricter rule until the user reviews an update.

Do not edit, reinterpret, or weaken this policy or an approved spec merely to make a desired
action compliant. Policy changes are separate user-reviewed documentation changes.

## 0. Prime directives

1. **No unapproved execution.** Do not submit or start any training job, paid smoke test,
   paid sandbox, GPU worker, or retry without explicit user approval for that exact run.
2. **The launcher is reviewable before launch.** A spec-sheet summary alone is insufficient.
   Provide the complete launcher or a reviewable diff, its SHA-256, and the approved spec
   sheet before submission.
3. **Approval is bound to an immutable run definition.** Any material change after approval
   voids it. A failed run does not authorize a paid retry.
4. **No silent scope reduction.** Never shrink epochs, data, context, model size, evaluation,
   or quality goals merely to fit budget or time. Present options and wait.
5. **A model repository is an export, not a working tree.** Change source, move the branch,
   regenerate the artifact. Never falsify an existing artifact's lineage.
6. **Observed, expected, inferred, and planned are different states.** Label them accurately.
   Never turn a hope, forecast, or plausible mechanism into an achieved result.
7. **“Context only,” “FYI,” “quote only,” and past requests are not authorization.** The
   current explicit user instruction controls the action.
8. **When semantics or spend are ambiguous, stop before acting.** This includes “instruct,”
   “prototype,” “faithful scale-down,” “resume,” “shippable,” and “cheap test.”

### Authority order

Current explicit user instruction/correction > run-specific approved spec sheet > this file
> handoffs, README claims, agent memory, and comments. Silence is not approval. An uploaded
conversation is evidence and context, not permission to execute its proposed next step.

### Status vocabulary

Use these terms literally:

- **Written:** text or code exists; it has not necessarily run.
- **Parked:** written for review and deliberately unrun.
- **Smoke-tested:** the stated gate actually executed in the stated environment.
- **Completed:** the run reached its asserted terminal step and returned normally.
- **Shipped:** the intended files were pushed to a durable repository.
- **Reload-verified:** a fresh load from the durable repository passed the stated gates.
- **Reviewed:** a human or explicitly described review inspected the named evidence.
- **Recommended:** a bounded judgment based on named comparisons, not a quality guarantee.

Never collapse “written” into “tested,” “pushed” into “reloaded,” or “best validation loss”
into “best model.”

## 1. Hard launch gate and signed spec sheet

Before **any** training submission, regardless of price or duration:

1. Post the spec sheet in the visible reply, not only in a file.
2. Show the complete launcher or reviewable diff and its SHA-256.
3. Ask for approval for the named run and its expected/max cost through the available
   explicit confirmation mechanism.
4. Submit only after approval naming that run. A generic “go ahead” counts only when it
   unambiguously repeats the run name and no material detail has changed.
5. Embed the final approved spec sheet in the launcher docstring and ship it with provenance.

The old “more than $1 or longer than 15 minutes” threshold is **not** a loophole. Even a
$0.02 GPU smoke test requires approval when the user has forbidden spend.

### Approval is void if any of these change

- Launcher contents or hash; source branch/commit; bundled code.
- Base/parent repository or revision; optimizer/RNG initialization mode.
- Dataset, configuration, split, revision, tokenizer, tokenization, packing, or replay mix.
- Special tokens, vocabulary size, embedding resize, parameter count, or architecture.
- Epochs, steps, token budget, seed, data order, batch, accumulation, or sequence length.
- LR, warmup, schedule, floor, optimizer, weight decay, clipping, precision, or compile mode.
- Hardware, timeout, expected cost, hold/max cost, destination, recovery repo, or export gates.
- Evaluation prompt set, generation path, benchmark suite, or claimed deliverable.

Post the revised spec and launcher diff and request approval again. Do not quietly patch a
running or queued job.

### Required spec sheet

```text
RUN: <name and session label> -> <destination repo>
STATUS: <quote only / parked / approved / running / complete>
LAUNCHER: <path or URL> sha256=<hash>  REVIEW LINK: <visible diff or file>
SOURCE: <repo> branch=<branch> commit=<full commit>  HASH AUDIT: <status/link>
BASE/PARENT: <repo> @ <full revision>  AUDITED: <yes/no + evidence>
RESUME MODE: <full-state resume / fresh-optimizer continuation / new run>
DATA: <repo> @ <revision> config=<cfg> split=<split>
DATA GATES: raw rows=<n> packed rows=<n> real tokens=<n> eval rows=<n> leakage=0
TOKENIZER: vocab=<n> markers=<atomic/multi-token> special tokens=<list>
EMBEDDING: resized=<y/n> params=<before> -> <after> tied=<y/n>
STOPPING: <ID-based/string-based/none>  VALIDATED: <gate and result>
METHOD: <full fine-tune/LoRA/other>  EPOCHS=<n> STEPS=<derived from actual loader>
LR: peak=<x> floor=<y> warmup=<n> schedule=<shape> first-step LR=<x>
BATCH: <bs> x accum=<g> x seq=<n> = <scheduled positions/step>
HARDWARE: <flavor/image>  TIMEOUT=<t including queue>  COMPILE=<y/n>
BUDGET: remaining=<x> expected=<y> hold=<h> hard max=<m> retry buffer=<r>
GATES: <exact pre-flight, training, export, and reload gates>
EVIDENCE: full-state snapshots every <n> steps + greedy generations + full validation
OUTPUT: <weights, state, metrics, launchers, hashes, samples, card, recovery location>
DELIVERABLE MEANING: <mechanics test / trajectory probe / research model / prototype>
NON-GOALS: <what this run cannot establish>
```

A quote-only request ends after this document and any requested launcher draft. It does not
authorize package installation on paid infrastructure, smoke tests, repository creation,
Hub writes, or job submission.

## 2. Define what the experiment can prove

Before implementation, classify the deliverable:

| Deliverable | Legitimate claim | Forbidden leap |
| --- | --- | --- |
| Mechanics check | Tokenization, resize, tying, save/reload, stopping, and export work | “Useful assistant” or “recipe scales” |
| Trajectory probe | Named behavior changed at named checkpoints under controlled decoding | Quality at a different size/token budget |
| Research checkpoint | This exact artifact produced these metrics/samples | General reliability or deployment readiness |
| Prototype | The agreed recipe dimensions were faithfully represented | Fidelity based only on parameter/token ratios |
| Shippable model | Explicit acceptance criteria were met | A claim based on loss, one prompt, or one epoch |

For a scale-down prototype, specify which dimensions are faithful: base→instruct path,
tokenizer and embedding treatment, data mixture, supervised token exposure, epochs, LR
schedule, architecture, stopping, and evaluation. Tokens-per-parameter alone does not prove
recipe fidelity or quality scaling.

Do not state that a run “will format better,” “will be shippable at $X,” or “only needs more
data/epochs/LR.” Use hypotheses and decision criteria. Failure by step N does not isolate
data as the cause: optimization, cold embeddings, architecture, masking, tokenization,
decoding, and evaluation remain possible explanations unless controls exclude them.

## 3. Module boundaries and architecture lane

The contents of helix_lm is sacred and should not be changed without explicit user agreement. Any artifact where this is changed needs to note the changes, as this would otherwise violate provenance of the source code -> artifact, leading me to "I don't know" being the answer when a member of the research community asks "How do I reproduce this | I ran this launcher against the most upstream source code and failed to reproduce, what did you do different?".

| Concern | Canonical module |
| --- | --- |
| Model-wide configuration and presets | `helix_lm/config.py` |
| Graph construction and topology | `helix_lm/graph.py` |
| Node implementations and attention math | `helix_lm/nodes.py` |
| Hugging Face adapter, tying, generation, save/load | `helix_lm/hf_model.py` |
| Tokenizer adapter and template serialization | `helix_lm/tokenizer.py` |
| Supervised/document-aware data and collation | `helix_lm/dataset.py` |
| Pretraining sample store, manifest, and order | `helix_lm/pretrain_data.py` |
| Training loops and checkpoint state | `helix_lm/trainer.py` |

Keep changes in their owning module. Do not bury training-policy changes in export scripts
or architecture changes in launchers.

### Trainer boundary

- `Trainer` is the legacy document-aware SFT implementation. Preserve document boundaries,
  padding masks, overlap semantics, and public call shape.
- Do not rename `Trainer` to `SFTTrainer` in this lane; David owns that migration.
- `PretrainTrainer` is the only trainer allowed to consume the continuous pretraining store.
- Do not silently filter kwargs through `inspect.signature`. Assert accepted parameters and
  prove that callbacks/hooks fire. Install step hooks with the verified wrapper pattern when
  kwargs would be dropped.
- `trainer.train()` must return its history normally after checkpoint saving. Assert the
  returned history, terminal `global_step`, and expected snapshot/export side effects.

### Canonical pretraining data contract

```text
ordered source rows
-> tokenize without tokenizer-added specials
-> append one EOS per nonempty document
-> concatenate
-> exact seq_len windows
-> discard incomplete tail
-> persist stable sample IDs and compiled manifest
-> activate and persist a global epoch permutation
```

A new epoch must activate a new persisted permutation. Reusing the previous DataLoader is
not a global shuffle. Exact resume requires the sample cursor, manifest root, and epoch
permutation root in addition to model and optimizer state.

### Deferred topology work

Do not add a fourth column or raise vertical depth to three in the current comparison lane.
The frozen comparison uses three columns, `(3, 3, 3)` nodes, and vertical depth two. Future
topology work requires a separate matched ablation after the data pipeline is proven.

Do not mix unrelated KV-cache, GQA, tying, topology, or performance work into an SFT bug-fix
PR. The eval-path `int(n.max())` graph break, for example, is a separate benchmark/PR unless
it blocks correctness.

Distributed or kernel scaling is its own reviewed lane. Do not apply pipeline parallelism to
the recurrent loop architecture without a design that explicitly handles recurrence. Tensor
parallelism must divide attention heads exactly. FlashAttention is not a drop-in replacement
for custom linear/multi-scale attention math. Preserve required FP32 accumulation and norm
gradient reductions, and justify CP/SP rather than enabling it for short sequences by default.

## 4. Tokenizer, embedding, chat-template, and stopping contracts

### Existing tokenless lineage

The 20261118 SFT, original v3, and `ep-4-ct` lineage used GPT-2 vocabulary **50,257** and
**53,592,343 parameters**. `<|im_start|>` and `<|im_end|>` are ordinary multi-token BPE
strings; no embedding rows were added. A chat-template round-trip proves only consistent
rendering/tokenization. It does not prove atomic markers, learned turn-end embeddings, or
reliable stopping.

Do not retroactively edit these exports to imply otherwise. Describe them as tokenless SFT
artifacts. For these artifacts, `stop_strings=["<|im_end|>"]` plus an explicitly supplied
tokenizer is only a candidate mitigation; test it before claiming it works. It does not fix
instruction quality or make the lineage token-corrected.

### New token-corrected instruct/prototype runs

Unless the user explicitly approves a different contract in writing:

1. Add `<|im_start|>` and `<|im_end|>` as special tokens **before training and before save**.
2. Resize input embeddings through the model's actual
   `get_input_embeddings`/`set_input_embeddings`/`resize_token_embeddings` path.
3. Re-tie weights using the shipped implementation and verify that the tied head and input
   embedding still reference the intended weight after resize and after reload.
4. Update `config.vocab_size`; loss reshaping and output logits depend on it.
5. Assert each marker encodes to exactly one ID with `add_special_tokens=False` and
   round-trips without splitting.
6. Print vocabulary size, embedding shape, and exact total parameters before resize, after
   resize, before training, after save, and after Hub reload.
7. Set the chat template in the live process before `save_pretrained`; ship
   `chat_template.jinja`; verify a fresh reload without importing `helix_lm`.
8. Validate generation stopping end to end on the `<|im_end|>` **token ID**. Record the
   relevant EOS/stop IDs and generation configuration using the pinned Transformers API.
9. Test both stopping behavior and emitted text. A configuration field without a behavioral
   test is not a stop gate.

For the unchanged d512 shape, two new tied rows imply **50,259 vocabulary entries** and
**53,593,367 parameters** (`53,592,343 + 2 × 512`). This exact count is shape-specific, not
a universal HelixLM constant. Parameter counts are measured gates, never marketing claims.

**Do not rely solely on `stop_strings` once the marker is special.** Decode paths that use
`skip_special_tokens=True` can strip the marker before string matching. Use and test an
ID-based stopping mechanism. Conversely, do not apply an ID for a token that does not exist
in the old tokenless artifacts.

Adding special tokens to an already tokenlessly trained model is a different experiment from
adding them to the base before SFT. The former tests adaptation/embedding robustness; it is
not evidence for the base→instruct recipe.

## 5. Dataset and token-accounting gates

No GPU job may be queued until every applicable gate passes and its evidence is recorded.

### Repository and split integrity

- Resolve the dataset to a full revision. Verify that the repository and requested configs
  are non-empty by reading actual rows, not by trusting its name, card, or build plan.
- Assert raw train/eval row counts against the build report. An empty or partial repository
  blocks launch.
- Assert split disjointness, duplicate/replica handling, and zero evaluation leakage.
- For packed data, assert every member fingerprint belongs to train and preserve the chain
  packed chunk → raw row → original source.
- Assert sequence/token caps, EOS behavior, padding/label masks, and decode→encode round
  trips on representative boundary cases.
- Build/push large datasets incrementally. Do not wait until the final push to discover a
  timeout left an empty destination.

### Use the real pipeline

Never substitute `text[i:i+N]` single-file chunking for `helix_lm.dataset` or
`helix_lm.pretrain_data`. Tests must include:

- Multi-paragraph documents longer than the training context.
- Documents much shorter than the context that require padding or packing.
- Multiple documents with explicit EOS/end-of-document boundaries.
- Empty documents, exact-context-length documents, tails, and packed chunks crossing rows.
- Label-shift and `ignore_index=-100` checks proving padding is excluded from loss while
  intended EOS supervision is retained.

If a Hub download fails, use generous timeouts and report the access problem. Do not silently
substitute synthetic data or a different pipeline.

### Accounting discipline

Report separately:

- Raw conversations/documents and unique source examples.
- Packed chunks and real packed tokens, including separators.
- Replay fraction **by rows** and, if relevant, by real tokens; they need not match.
- Scheduled token positions (`batch × accumulation × sequence × steps`), including layout
  padding.
- Optimizer steps and micro-batches derived from the **actual loader after tokenization**.
- File bytes, which are not a token count.

Changing ChatML markers from multiple BPE tokens to one ID changes tokenization and may
change packed chunk counts and steps. Recompute; never carry over 152,668 chunks, 19,397
micro-batches, or 4,850 steps by assumption.

Verify library data-order assumptions empirically with a tiny loader on the same Torch
version. Do not assume “one randperm per epoch”: worker base-seed draws and sampler/permutation
remainder behavior can consume additional generator state. Assert meaningful invariants such
as `n // batch_size == batches`; do not invent false gates such as divisibility when a
remainder batch is valid.

## 6. Environment and dependency discipline

1. Before package operations, run the required system update/install step when system
   dependencies are needed and upgrade `pip` according to the repository setup protocol.
2. Install exact pins from `requirements.txt`/the approved image. If a new unpinned
   dependency is necessary, resolve it, record the actual installed version, and update the
   lock/requirements through a reviewed source change.
3. Never choose versions from model memory or call a remembered version “latest.” Inspect
   the approved environment and current package metadata.
4. Verify Torch against the actual container, CUDA, and worker image. Do not downgrade
   Transformers to v4.x as a workaround for an integration failure.
5. Give multi-gigabyte installs a generous timeout and a persistent log. Do not repeatedly
   interrupt and resubmit short-timeout installs.
6. Record the fully resolved environment in run provenance. “Same script” with different
   Torch/Transformers/datasets versions is not the same environment.
7. For current custom-code Hub artifacts, `trust_remote_code=True` executes code. Audit it,
   pin the full revision, and never print credentials. Treat unexpected remote-code loading
   as an integration issue to investigate, not an excuse to downgrade.

Previously supplied reproducibility pins for the 2026 SFT/lighteval stack include
`transformers==5.23.0`, `datasets==5.2.0`, `huggingface_hub==2.4.0`, and
`lighteval==0.13.0`; the supplied training record also lists `torch==2.15.0`,
`numpy==2.5.5`, `jinja2==3.1.6`, `tqdm==4.70.1`, and `trackio==0.45.1`.
These are historical environment pins, not instructions to mutate another approved image.
Re-resolve and record the environment for each new run.

## 7. Free pre-flight verification before paid training

Run every feasible local/free gate before requesting a paid job. If the user has forbidden
all spend, even a paid smoke test remains blocked.

1. **Final dtypes:** instantiate and run a forward in FP32 and BF16 with **no autocast**.
   Training autocast cannot catch every final-load dtype bug.
2. **Save/load:** `save_pretrained(tmp)` and `from_pretrained(tmp)` pass; config validation,
   tying, tokenizer, chat template, vocabulary, and parameter counts survive.
3. **Stopping:** generate past a short answer and prove the intended stop fires; inspect both
   raw IDs and decoded text.
4. **Data:** all section-5 gates pass against the actual dataset revision and loader.
5. **Trainer return:** a shortened run returns history after checkpoint saving and executes
   every line that follows, including durable pushes.
6. **Hooks:** callbacks and snapshot wrappers demonstrably fire; no silent kwarg filtering.
7. **Step math:** actual loader steps match the spec; terminal `global_step` reaches
   `TOTAL_STEPS`. `EPOCHS = ceil(token_budget / store_tokens)` where that contract applies.
8. **Same job shape:** smoke with the approved dtype, batch, sequence length, model shape,
   and hardware class. Shrink steps, not the semantics being tested.
9. **Tokenizer realism:** character-tokenizer tests are rudimentary forward/gradient checks,
   not production-viability evidence. Use the approved GPT-2 path for current realistic SFT
   tests. Do not substitute Qwen3 or run it in a CPU sandbox unless the user explicitly
   approves that tokenizer and environment.
10. **Throughput:** measure fixed GPU tensors at production shape to distinguish loader
    bottleneck from compute. Do not quote small-corpus throughput as GPU capacity.
11. **Failure reproduction:** after a failed job, reproduce the failing step on the same
    image/package flavor before proposing another paid submission. No resubmit loops.

For the current production SFT lane, `model.forward = torch.compile(model.forward)` is the
approved performance configuration. Assert that compile was applied and record it in
`training_info`; any exception requires explicit spec approval. A measured historical anchor
at BF16+compile, batch 8 × sequence 2048, is 66,904 tok/s, 0.98 s/step, and 24.0 GiB peak.
It is a regression reference, not a quote for different hardware or a guarantee.

## 8. Checkpoint, resume, and continuation contract

### Full-state checkpoint

At every snapshot interval and at the end of training, save one versioned bundle containing,
as applicable:

- Model weights and config.
- Optimizer state, including AdamW moments and internal step counter.
- Scheduler state and global/epoch step.
- AMP scaler state when present.
- Torch CPU RNG, CUDA RNG when used, Python/NumPy RNG when used, and DataLoader generator.
- Sample/loader cursor, compiled sample manifest root, epoch permutation root, and worker
  seeding information needed to reproduce order.
- Resolved environment, launcher hash, source commit, base/data revisions, and metrics.

Push the bundle to a durable recovery location with the weights, not in a later best-effort
step. Local container storage is ephemeral. Weights-only snapshots are useful exports but
are **not** exact resume checkpoints.

For the current roughly 4,850-step SFT lane, the default evidence cadence is a durable
snapshot every 1,000 steps and at EOF, early-abort monitoring over the first 500 steps, and
controlled greedy generations with each diagnostic checkpoint. Any different cadence must be
stated and approved in the spec. A 200-row diagnostic validation slice remains distinct from
final full validation.

### Resume

`resume_from=<repo/full revision>` must restore all required state before the first update.
Assert global-step continuity, scheduler position/LR, optimizer state presence, data cursor,
manifest/permutation roots, tokenizer/vocabulary, and parameter count. Perform a small
state-restoration test before relying on a paid resume.

### Fresh-state continuation

If optimizer moments, RNG, or data state are unavailable:

- Call the run a **fresh-state continuation**, not an exact resume or bit-identical recovery.
- Start Adam's internal counter at zero; do not fake optimizer moments.
- Pre-advance only the scheduler when that is the approved design.
- Record `optimizer_state`, `rng_state`, `permutation`, parent revision, and first-step LR in
  `training_info`.
- Measure and publish the continuation's own metrics. Similar validation loss does not prove
  weight or behavior equivalence to the lost run.

Push final weights and full state **before** launcher capture, card rendering, or any other
fallible export work. Export must be idempotent and as non-fatal as possible.

## 9. Export, provenance, and security

1. Pin full source and artifact commits, not branch HEAD or a short prefix in final records.
2. Hash-audit every shipped `.py` against the approved source archive. Store the audit in the
   run record and model card. Hash matching proves bundled-code identity, not which code
   trained the weights; keep that distinction.
3. Record `training_info.code_commit`, base/parent revision, dataset revision, launcher
   hashes, and run ID. Numbers measured under another code/artifact state must be labeled
   reference-only.
4. Never rely on `__file__` in a launcher that may run via `uv run -`/stdin. Ship launchers
   from a real source path or fetch their pinned repository copies.
5. Do not hand-edit exported `.py`, `config.json`, `generation_config.json`,
   `tokenizer_config.json`, tokenizer data, or weights. Change source → review → regenerate.
6. README and `AGENTS.md` documentation corrections may fix forward with a clear commit, but
   must not rewrite weight provenance or make an old artifact claim a new capability.
7. Never delete or rewrite history to conceal an error. Preserve the old artifact, mark its
   status, and publish a corrected successor or explanatory commit.
8. Do not reproduce access tokens, credentials, signed URLs, or secrets in launchers, logs,
   cards, issues, memory, or summaries. Use secret handles/environment injection.
9. Reserve destination and recovery repositories and the metrics dashboard before launch.
   Verify that the first durable write succeeds.

## 10. Budget, submission, and monitoring

### Preflight cost quote

State expected actual cost, provider hold, hard maximum, remaining budget, timeout including
queue variance, and a cost-to-complete range that accounts for known smoke/retry risk. Treat
budget balances in old conversations as historical; verify the current balance before quoting
or submitting. A run is not affordable merely because expected spend fits if its required
hold does not.

Never raise the budget or invoke a budget-setting control on the user's behalf. Never evade a
hold by changing destination, splitting jobs, shrinking the approved run, or starting “just a
smoke.” Present the blockage and wait.

Set timeout above the measured estimate plus reasonable queue buffer, not at the estimate and
not at an unjustified multiple that holds funds hostage. Name every job and carry the session
label. Reserve the destination and trackio dashboard before submission.

Production training is submitted through Hugging Face Jobs by default rather than run as a
long training process in the interactive sandbox. This is still a submission requiring full
approval; using `hf jobs run` does not make it an unmonitored or low-risk action.

### Monitoring and success

- Read trackio rows on the first log inspection. “Saved locally” metrics die with ephemeral
  storage.
- Confirm snapshot pushes and generation files at their intervals; do not wait until EOF.
- Job logs may truncate their middle. Judge completion from terminal step, returned history,
  repository commits, `metrics/`, `config.json`, and reload gates—not a final log line alone.
- Compare observed throughput and spend with the quote. Report queue time separately from
  train time.
- Report failed runs, partial artifacts, and their cost honestly. Do not hide them from the
  cost-to-complete history.

## 11. Evaluation and generation review

### Checkpoint selection

Validation loss is not the release acceptance test. Training and validation losses may fall
monotonically while reviewed generation regresses. Select canonical revisions only after
controlled generation review; run expensive benchmark evaluation on the selected candidate
unless the user approves a broader evaluation matrix.

Use a fixed, documented prompt suite covering factual completion, concise explanation,
arithmetic, formatting, poetry/instruction constraints, turn termination, repetition, and
topic drift. “The capital of France is” is a useful probe, not an acceptance suite.

For controlled comparisons:

- Pin exact model/tokenizer/code revisions.
- Use greedy decoding (`do_sample=False`) or an explicitly fixed sampler and seed.
- Keep chat template, prompt text, stopping, maximum new tokens, dtype, and device path fixed.
- Decode and inspect newly generated IDs/text; check turn termination and leaked markup.
- Do not compare trainer samples from one generation path with gate-7 samples from another
  and attribute every difference to weights.
- A few identical greedy outputs or one better sampled response do not prove collapse or
  recovery. State the limited evidence.

Keep full 1,000-row validation separate from 200-row diagnostic slices. Compute
`val_ppl = exp(val_loss)` once in a shared metric function. Never label training PPL as
validation PPL. Do not compare losses across changed eval splits without qualification.

### lighteval 0.13.0 harness safeguards

Use the approved pins and preserve the documented compatibility shim as a replaceable,
versioned artifact. In particular:

- Hub 2.x uses `search=` where the old task-registry code used `model_name=`.
- The xxhash shim must tolerate string and no-argument dataset-fingerprint calls.
- If continuation logits are shorter than the target, filter/report the affected documents.
  **Never return a sentinel score.** A sentinel previously fabricated a perfect BLiMP result.
- HellaSwag in lighteval 0.13.0 generative mode measures letter-format compliance; use a
  suitable log-likelihood variant for capability claims.
- Script-only PIQA loading is incompatible with the stated datasets 5.x setup; report it as
  unavailable rather than fabricating a substitute.
- `truthfulqa:mc` yields MC1 and MC2; preserve metric identity.
- Locate results under `<out>/results/<org>/<model>/results_*.json` and archive raw JSON.

Treat implausibly strong small-model scores as a harness-audit trigger, not a release win.
Record evaluated full revision, harness/backend/dtype/shots/task revisions, metric type, and
raw output. Do not compare lighteval `acc` directly with lm-evaluation-harness `acc_norm`.
External model-card scores are indicative context, not controlled head-to-head evidence.

WeightWatcher/spectral results must be measured on the exact artifact with identical settings
for comparisons. Never reuse base or earlier-SFT spectra for a new continuation.

## 12. Documentation and claims

Every model card must disclose, as applicable:

- Exact parameter count, architecture shape, base/parent and dataset revisions.
- Atomic versus multi-token chat markers, vocabulary/embedding resize, and stopping behavior.
- Full-state resume versus fresh-state continuation and lost-state limitations.
- Real tokens versus scheduled positions and actual evaluation split.
- Train and validation metrics with correct PPL and artifact source.
- Generation failures, decoding controls, benchmark status, and unevaluated areas.
- Current recommendation scope and stronger alternative revision, if any.
- No-KV-cache or other serving limitations and `trust_remote_code` security implications.
- License, out-of-scope uses, and lack of authoritative/factual reliability.

Do not claim:

- “Shippable” or “useful assistant” without explicit acceptance evidence.
- “More epochs/data will fix it” or “LR cannot fix it” without controlled diagnosis.
- “Data is the problem” from one failed trajectory probe.
- “Equivalent to the lost checkpoint” because losses match.
- “Prototype of the 300M recipe” when tokenizer, parent path, or training exposure differs.
- “Zero cost” when compute, provider holds, or paid agent infrastructure is involved.

Correct mistakes briefly and visibly, preserve evidence, and fix forward.

## 13. Current historical landmines — verify status before acting

As of 2026-11-21:

- The altered/corrupted `20261115` artifact is do-not-use for attributable comparisons.
- The empty v2 dataset repository is do-not-use.
- Do not rerun replay8 as a duplicate of the completed 20261118 run.
- Original v3 epoch 2, revision `621484017b3a0a6b697649b52d24cd8c96e7c8d1`, is only the
  current **relative research recommendation within the tokenless lineage**.
- Original v3 epoch-4 weights and full training state were lost. `ep-4-ct` is a separate
  fresh-AdamW/RNG continuation from full parent revision
  `e89ca39ab4c7b9412cb4e4ed5631b84c638711a1`.
- Do not execute `resume_v3_br72.py` unchanged for a new epoch.
- `ep-5-ct-xd` and any further epoch are unlaunched unless the user provides newer verified
  status. A previous quote or parked launcher is not approval.
- The separately discussed token-corrected launcher is parked and not smoke-tested unless a
  newer verified record says otherwise. It produces no quality claim by existing.

These are historical safeguards, not substitutes for checking the current user instruction
and repository state.

## 14. Definition of done

Before saying “ready,” identify which level is meant and satisfy its checklist:

### Parked launcher

- [ ] Spec sheet and NON-GOALS visible.
- [ ] Full launcher/diff and SHA-256 visible.
- [ ] Tokenizer/embedding/stopping semantics explicit.
- [ ] Actual-loader step derivation implemented, not assumed.
- [ ] Clearly labeled unrun/untested where applicable.
- [ ] No job, paid resource, or unauthorized Hub write.

### Training run

- [ ] Explicit approval still valid; no material diff.
- [ ] Data/environment/pre-flight gates archived.
- [ ] Durable full-state snapshots verified during training.
- [ ] Terminal step and normal `train()` return asserted.
- [ ] Final weights/state pushed before fallible export work.
- [ ] Cost and failures reported.

### Shipped model

- [ ] Fresh Hub reload passes parameter, tying, tokenizer, template, stopping, and generation gates.
- [ ] Metrics and raw evaluation artifacts belong to this exact revision.
- [ ] Provenance and launcher hashes archived.
- [ ] Model card states token contract, continuation mode, limitations, and recommendation scope.
- [ ] No claim exceeds the evidence.

If any box is unchecked, report the precise blocked/pending state. Do not improvise around it.
