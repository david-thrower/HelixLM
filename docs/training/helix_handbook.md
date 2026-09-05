# Helix Handbook

~~~text
DOCUMENT=HELIX_HANDBOOK
STATUS=LIVING
EDIT_AUTHORITY=David | Porthos | Mo
OTHER_CONTRIBUTORS=evidence packets and proposed diffs only
THUNDERLINE_RUNTIME_AUTHORITY=none
~~~

This is the operating handbook for Helix model research conducted alongside
Thunderline. It records the rules that must survive handoffs: how data reaches
the model, which trainer owns which workload, how experiments remain
comparable, what a checkpoint proves, and which claims are not yet earned.

It does not activate Python, Hugging Face, MLflow, external training, model
promotion, or GPU execution inside Thunderline clean trunk. Those remain
external research surfaces until separately admitted.

## 1. Editing and contribution law

This is a living document, but it is not a communal scratchpad.

Only the following authorities may directly edit its normative content:

- **David** — Helix implementation and scientific-method authority.
- **Porthos** — independent model, evidence, and admission reviewer.
- **Mo** — founder authority, experiment sponsor, and final scope arbiter.

Everyone else, including automated agents, may:

- collect and verify evidence;
- identify contradictions or stale instructions;
- prepare a proposed patch in a review comment or evidence packet;
- link exact commits, run IDs, hashes, logs, and court results.

They may not directly rewrite this file. A Porthos-authored change may be
applied by David or Mo when Porthos has no repository identity, but the commit
or pull request must identify Porthos as the originating authority.

The header expresses governance; it is not a substitute for repository review
controls. Any change to this file requires an authorized editor and review by
at least one of the other two authorities.

Normative changes append an entry to the decision register in section 16 with:

~~~text
date
author
status
decision or caveat
evidence
affected runs or code
revisit trigger
~~~

Do not replace an earlier decision silently. Mark it SUPERSEDED, preserve the
old language in Git history, and point to its successor.

## 2. Core boundaries

Keep these objects separate:

~~~text
source code          = executable implementation at an exact commit and tree
run contract         = declared inputs and intended configuration
runtime observation  = what the process and hardware actually did
local custody        = logs, checkpoints, manifests, hashes, and terminals
MLflow projection    = searchable experiment telemetry
evaluation           = downstream measurement under a frozen evaluator
promotion            = a separate human-authorized decision
~~~

A green training process is not a promoted model. An MLflow page is not
checkpoint custody. A lower perplexity is not automatically a downstream win.
A configured topology value is not evidence that graph construction used it.

## 3. Trainer ownership

### SFT path

The legacy Trainer behavior remains the document-aware supervised fine-tuning
path. It preserves variable-length examples, padding, attention masks, and
document boundaries appropriate to instruction tuning.

Do not silently replace or repurpose it for continuous pretraining. Preserve
compatibility while the codebase transitions toward the clearer name
SFTTrainer. An alias and deprecation path are acceptable; an abrupt breaking
rename is not.

### Pretraining path

PretrainTrainer owns continuous causal pretraining:

~~~text
ordered documents
-> tokenize without implicit special-token surprises
-> append one EOS to each nonempty document
-> concatenate the token stream
-> emit exact, non-overlapping sequence-length windows
-> count and discard the incomplete tail
-> assign stable sample IDs
-> persist a global seeded permutation
-> replay exact samples from disk
~~~

The pretraining path uses fixed windows and no padding. Its labels equal its
input tokens; causal loss is produced by the model's one-token shift.

### Input dispatch

The caller should not need to know a private preprocessing ceremony.

- A reiterable list of strings, narrow dataset text column, or object that
  safely duck-types as a reiterable sequence of strings may use the live
  continuous path.
- A Hugging Face IterableColumn, one-shot iterator, or streaming input must be
  compiled automatically to a verified disk store before training.
- An explicit store path remains supported for offline preparation, reuse,
  exact replay, and restart.
- Ambiguous or mixed input types are refused with a useful typed error; they
  are never guessed into a training run.

prepare_pretrain_dataset.py may remain as an explicit batch-job interface until
automatic trainer compilation is implemented and proven. It must not be the
only undocumented way to obtain the correct large-corpus behavior. Do not
delete it before feature parity exists.

Do not monkey-patch private Trainer or datasets internals. Integrate the
compiler through the public Helix pretraining boundary, ideally under
helix_lm/dataset/ or its eventual canonical successor.

## 4. Compilation, chunking, and shuffle law

PretrainSampleCompiler must produce a content-addressed, verifiable sample
store. PretrainPermutation must identify the exact global presentation order
for each epoch.

Required store identity:

~~~text
dataset name and immutable revision
split and text column
tokenizer name, revision, and vocabulary hash
sequence length
EOS policy
source-row count
raw-byte count
causal-target count
sample count
dropped-tail token count
sample-manifest root
compiler version
~~~

Required ordering identity:

~~~text
epoch
seed
permutation algorithm and version
permutation root
validation-ID root
first and last admitted sample IDs
~~~

A seed alone does not identify sample order.

The globally shuffled order must match the order that the equivalent in-memory
path presents to the model. Disk throughput may be improved with compiled
physical layout, page-local reads, prefetching, pinned host memory, and staged
device transfer. Those optimizations may not redefine the permutation.

GPU memory may stage already-selected batches. It must not become the authority
for global shuffle order. The persisted CPU/disk permutation remains the
replayable source of truth.

## 5. Validation law

Validation selection is explicit. The trainer must not hide a
validation_sample_count policy that silently changes the data population.

The caller supplies one of:

- an explicit validation text input;
- an explicit validation loader;
- a separate compiled validation store; or
- a declared, persisted validation-ID set excluded from training.

Never call len() on an iterable dataset, iterable column, or loader unless the
object explicitly supports it. Evaluation loops terminate by exhaustion or an
explicit bounded contract, not an assumed length.

Validation IDs and their root are frozen before examining results. Changing
validation membership creates a new comparison family.

## 6. Reproducible experiment contract

Every run freezes these inputs before the first optimizer step.

### Source and environment

~~~text
repository
branch
source head
source tree
dirty status
launcher path and SHA-256
dependency lock SHA-256
Python version
PyTorch version
Transformers version
CUDA and driver versions
GPU model and memory
~~~

The project baseline for new Hugging Face integrations is Transformers 5.4 or
newer, pinned to an exact tested version in the dependency lock. “At least
5.4” is not reproducibility; the lock is.

### Launcher and repository hygiene

The mergeable Helix repository should contain the production implementation,
CI/CD, maintained examples, required datasets or tiny fixtures, README and
agent guidance, plus one clearly designated launcher for the admitted winning
configuration.

Non-winning ablation launchers, temporary JSON summaries, and exploratory
manifests remain available through pull-request discussion, MLflow, and local
custody packets unless one is required for maintenance or exact replay. Do not
turn the production tree into an experiment attic.

The canonical launcher must be readable without wrapper archaeology:

- give every environment-controlled option a documented default;
- keep dataset revision optional but observable;
- make resume state and push-to-Hub explicitly opt-in;
- avoid try/rescue logic that silently changes the data path;
- record the actual parameter count instead of trusting a historical filename;
- identify one profile as the default and label exact-replay profiles honestly.

### Data and tokenizer

~~~text
dataset and immutable revision
split
text column
sample-store root
permutation root
validation-ID root
tokenizer name and immutable revision
tokenizer vocabulary and configuration hashes
EOS/BOS/padding policy
raw UTF-8 bytes admitted
causal targets admitted
~~~

### Model and optimization

~~~text
parameter count
vocabulary size
sequence length
d_model
heads
columns
nodes_per_column
nodes_per_column_graph_effective
loops
FFN expansion
lateral_p
vertical_p
vertical_depth
local and coarse windows
compressed keys and views
optimizer
learning rate
scheduler
warmup
weight decay
precision
microbatch
gradient accumulation
effective batch
gradient clipping
activation checkpointing
seed
strict NaN policy
~~~

If nodes_per_column_graph_effective=false, the run may record the configured
value but may not claim it tested that topology. Wiring the setting into graph
construction is a prerequisite to a node-count ablation.

## 7. Metric semantics

All operators must use the same definitions.

For next-token causal language modeling, count supervised targets after the
shift:

~~~python
causal_targets = (labels[:, 1:] != -100).sum()
~~~

Do not use (labels != -100).sum() as the causal-target count. That measures raw
non-ignored label positions and is one position larger per full sequence.

Record both when useful, with distinct names:

~~~text
raw_sequence_positions
causal_targets
~~~

Required MLflow parameters include every field in the experiment contract,
especially:

~~~text
lateral_p
vertical_p
vertical_depth
nodes_per_column
nodes_per_column_graph_effective
graph_nodes
graph_edges
~~~

Required MLflow metrics:

~~~text
train loss and perplexity
validation loss and perplexity
learning rate
gradient norm
optimizer step and microbatch step
step duration
tokens per second
causal targets seen and per second
raw bytes seen and per second
global sample cursor
skipped batches
NaN/Inf count
checkpoint duration
evaluation duration
~~~

When directly observed and collected without destabilizing training, also log
GPU utilization, allocated and reserved VRAM, temperature, power, host RAM and
swap pressure, and data-loader wait time.

MLflow is the searchable projection. The local append-only event log,
checkpoints, manifests, and terminal packet are custody. A missing or delayed
MLflow metric must not erase locally captured evidence, and a visible MLflow
metric must not replace it.

## 8. Checkpoints, restart, and Hugging Face

Multi-epoch runs write periodic local checkpoints before any external upload.

Each resumable checkpoint contains or binds:

~~~text
model weights and config
tokenizer files and hashes
optimizer state
scheduler state
AMP scaler state
Python, NumPy, Torch, and CUDA RNG states
epoch and optimizer step
global sample cursor
sample-manifest root
permutation root
validation-ID root
source head and launcher hash
metric terminal through the checkpoint
~~~

A checkpoint is not certified resumable until a deliberate stop-and-restore
court reproduces the expected next sample and continues training without
counter or schedule drift. Until then, describe it as saved, not resume-proven.

Hugging Face upload is opt-in. Save locally, close files, hash the artifact,
write the local receipt, and only then upload. Never make successful upload a
precondition for preserving the local checkpoint.

### Model naming

Names must be at most 96 characters and use observable configuration, not
promotional claims.

Recommended pattern:

~~~text
hlx-<branch>-<yymmdd>-d<width>-c<cols>-n<nodes>-l<loops>-f<ffn>-s<seq>-e<epoch>
~~~

Example:

~~~text
hlx-b62-260905-d768-c3-n333-l4-f30-s1024-e1
~~~

Legend:

~~~text
d = model width
c = column count
n = nodes per column, concatenated in column order
l = recurrent loops
f = FFN expansion without decimal point
s = sequence length
e = completed epoch represented by the artifact
~~~

The base run name may omit e while training. A checkpoint or exported model
includes only the epoch it actually represents. Never name an unfinished model
after a future epoch.

## 9. Scientific comparison law

A controlled ablation changes one independent variable. If several variables
change together, label the run a bundle experiment or pilot, not an ablation.

“One complete pass over the dataset” means one full epoch. A token, sample, or
optimizer-step cap changes the run into a bounded pilot and must be stated in
the run name and contract before launch.

Matched comparisons freeze:

~~~text
source implementation except the named change
initial weights or initialization seed
corpus revision
sample order
validation set
tokenizer unless tokenizer is the independent variable
causal-target budget
optimizer and schedule
precision
checkpoint-selection rule
evaluator
~~~

For tokenizer comparisons, token counts are not comparable by themselves.
Record raw UTF-8 byte exposure, causal-target exposure, compression ratio,
throughput, validation results, and downstream evaluation. The GPT-2 tokenizer
remains the current default unless a matched experiment earns a change.

For architecture comparisons, parameter count and active parameter count must
both be reported. A larger model that sees fewer targets is not a matched run
without an explicit budget rationale.

## 10. Current architecture priorities

The present RTX 5080 research profile targets sequence length 1024 and an
approximately 100M-parameter model. The production-oriented priority is model
width before additional columns or vertical nodes:

~~~text
first:  d_model=768 class throughput and quality
later:  matched lateral_p / vertical_p / vertical_depth experiments
later:  additional column or vertical-depth topology
~~~

The current reference baseline family uses FFN expansion 3.0 and learning rate
2e-4. FFN 2.5 versus 3.0 must be reopened only under a frozen evaluator and
matched run contract.

Historical nodes_per_column=(2,3,2) and experimental (3,3,3) settings are not
scientific factors until graph construction demonstrably consumes them.

Avoid extreme gradient accumulation merely to preserve an old effective batch.
On a 16 GB RTX 5080, increase the microbatch as far as stable memory allows and
reduce accumulation correspondingly while preserving the declared effective
batch and data order. Gradient accumulation reduces optimizer frequency; it
does not reduce per-microbatch peak activation memory.

Gradient accumulation 42 is retained only when an explicitly named exact-shape
replay requires it. It is not the default 5080 operating posture. The current
768-width comparison family uses accumulation 28 until a matched throughput and
memory court earns a better value.

Do not add a column or raise vertical depth during the current pretraining-data
integration repair. Those are later, matched topology experiments.

## 11. Hardware profiles and failure response

Hardware guidance is profile-based, not universal.

### RTX 5080, 16 GB class

- Prefer the 768-width, 1024-sequence family for the current throughput court.
- Find the largest stable microbatch before selecting accumulation.
- Keep strict NaN/Inf checks enabled and require zero skipped batches.
- Observe browser compositors, desktop processes, and other GPU occupants before
  launch.
- Treat swap exhaustion as a system-stability risk even when VRAM is available.

### L40S/H200, high-memory class

- Wider models, larger microbatches, another column, or vertical depth 3 are
  candidates, not inherited defaults.
- Re-establish throughput, numerical, and data-order courts on that hardware.
- Never describe an H200 result as proof of an RTX 5080 operating envelope, or
  vice versa.

### OOM and CUDA failures

Before changing the model:

1. Identify every process using the GPU.
2. Confirm the failed process has terminated.
3. Record allocated and reserved VRAM and the exact failure.
4. Clear only the terminated process's reclaimable cache between runs.
5. Reduce microbatch first; then consider activation checkpointing or model
   shape. Accumulation is not the primary peak-memory control.

Never kill an unidentified process or delete an experiment directory to “make
room.” Preserve its custody packet first.

A DataLoader IndexError caused by a shard/index mismatch is a data-integrity
HOLD. It is not a model-quality verdict and must not be counted as a completed
training comparison.

## 12. Evaluation freeze

Checkpoint evaluation is a separate governed stage.

Freeze before comparison:

~~~text
Lighteval 0.13.0
task names and task revisions
dataset revisions
prompt and template versions
model adapter
tokenization and truncation behavior
few-shot policy
generation parameters
scoring semantics
dependency lock
checkpoint preparation procedure
~~~

Evaluate both checkpoints through the same environment and adapter. Training or
validation perplexity determines which checkpoints are interesting; it does not
establish downstream superiority.

## 13. Experimental long-memory program

The automata and DAG proposal is a research experiment, not a production claim.

Freeze one Sutra baseline checkpoint and evaluator, then compare:

~~~text
A: local-context transformer
B: transformer plus retrieval
C: transformer plus automata state and DAG retrieval
~~~

Use identical source observations and question sets. Measure distant recall by
history distance, causal reasoning accuracy, active context length, KV-cache
bytes, retrieval-index bytes, automata-state bytes, transition-log bytes,
latency, inference FLOPs where measurable, and total and active parameters.

The desired result is improved effective historical capacity, not the claim
that tokens are stored inside parameters. Parameters encode learned laws;
tokens supply observations; automata maintain compressed state; the DAG retains
addressable history.

## 14. Third-party model and dataset intake

Treat every external model, tokenizer, dataset, and code-bearing repository as
an untrusted supply-chain input regardless of its country, author, popularity,
or hosting platform.

Required intake:

~~~text
license and commercial-use review
immutable repository or dataset revision
artifact hashes
model card and dataset card capture
custom-code inventory
dependency and native-extension inventory
malware and secret scan
dataset provenance and personal-data review
tokenizer special-token audit
isolated environment smoke
network observation during load
~~~

Defaults:

~~~text
trust_remote_code=false
safetensors preferred
no production credentials in the environment
no write-capable Hugging Face token during inspection
no arbitrary pickle load from an untrusted source
no external network during evaluation after artifacts are frozen
~~~

If custom code is necessary, pin and inspect it before execution in an isolated
environment. A benchmark win does not cure an unknown license, provenance gap,
or executable supply-chain risk.

## 15. Completion courts

### Data-path court

~~~text
list[str] and equivalent IterableColumn
-> identical compiled samples
-> identical validation exclusions
-> identical persisted permutation
-> identical first optimizer batches
~~~

The minimum CI boundary includes:

~~~text
python -m unittest test_pretrain_data.py test_pretrain_data_court.py
python pretrain_data_court.py
~~~

The streaming-equivalence test must compare a list of strings with an
IterableColumn through PretrainTrainer. CPU demos must use the trainer matching
their workload rather than relying on the legacy Trainer by accident.

### Restart court

~~~text
train to checkpoint
-> stop process
-> reopen exact store and checkpoint
-> verify next sample ID
-> verify optimizer, scheduler, and scaler counters
-> continue without restore-induced loss discontinuity
~~~

### MLflow and custody court

~~~text
local event count and terminal
-> project to MLflow
-> read back run identity and required parameters
-> verify missing projection cannot erase local custody
~~~

### Topology court

~~~text
change nodes_per_column
-> graph node and edge identity changes as declared
-> hostile no-op wiring mutation turns court red
~~~

### Checkpoint publication court

~~~text
local write completes
-> local hash and manifest verify
-> optional upload begins
-> remote readback matches local root
~~~

## 16. Decision and caveat register

This register is append-only. Live progress metrics belong in run receipts, not
in this handbook.

| Date | Authority | Status | Decision or caveat | Evidence or revisit trigger |
| --- | --- | --- | --- | --- |
| 2026-09-04 | David, Mo | ACTIVE | Preserve the legacy document-aware Trainer; develop continuous pretraining through PretrainTrainer. | Revisit only through an explicit compatibility migration. |
| 2026-09-04 | David, Mo | ACTIVE | IterableColumn and other one-shot inputs must auto-compile through the public pretraining boundary; users should not need a hidden manual script. | Remove the preparation CLI only after auto-compilation and equivalence courts pass. |
| 2026-09-04 | David | ACTIVE | Validation membership is caller-declared; remove hidden validation-sample-count behavior. | Revisit with a versioned validation-policy contract. |
| 2026-09-04 | David, Mo | ACTIVE | The RTX 5080 priority is a stable approximately 100M model at sequence length 1024, with width 768 ahead of extra columns or vertical depth. | Revisit after the Branch 62 data path and matched throughput court are admitted. |
| 2026-09-04 | David, Mo | ACTIVE | Use FFN 3.0 and learning rate 2e-4 as the current comparison baseline. | Reopen FFN 2.5 versus 3.0 only after evaluator freeze. |
| 2026-09-04 | David, Mo | ACTIVE | Track lateral and vertical probabilities and depth on every run; vary them later from the best matched baseline. | Requires same data order, evaluator, precision, and target budget. |
| 2026-09-04 | David, Mo | ACTIVE | nodes_per_column is not an ablated variable until wired into graph construction and proven by a hostile topology court. | Revisit after graph-effective flag becomes true. |
| 2026-09-04 | Mo | ACTIVE | GPT-2 remains the default tokenizer; alternatives require raw-byte and causal-target matched evidence. | Revisit after a matched downstream evaluation win. |
| 2026-09-04 | David, Mo | ACTIVE | Multi-epoch runs require local periodic checkpoints and optional post-hash Hugging Face upload. | Resume claims require a stop-and-restore court. |
| 2026-09-04 | Porthos, Mo | ACTIVE | MLflow is telemetry projection; local append-only evidence and checkpoints retain custody. | Revisit only if a durable bidirectional custody protocol is admitted. |
| 2026-09-04 | Porthos, Mo | ACTIVE | Helix research does not block or authorize Thunderline Beta 1 runtime or checkpoint promotion. | Revisit through an explicit HC admission packet. |
| 2026-09-05 | Porthos, Mo | EXPERIMENTAL | Preserve the exact Branch 62 anchor for every example; permit Trident supplemental paths and fusion only for uncertain examples; use whole-node structured gating on the RTX 5080. | Revisit after a fixed-checkpoint CUDA benchmark establishes oracle gain, positive net repair, and real wall-clock behavior. |

## 17. Run handoff template

Every operator leaves the next operator this packet:

~~~text
RUN_NAME=
MLFLOW_EXPERIMENT=
MLFLOW_RUN_ID=

SOURCE_BRANCH=
SOURCE_HEAD=
SOURCE_TREE=
SOURCE_DIRTY=

LAUNCHER=
LAUNCHER_SHA256=
RUN_CONTRACT_SHA256=

DATASET=
DATASET_REVISION=
SAMPLE_MANIFEST_SHA256=
PERMUTATION_SHA256=
VALIDATION_IDS_SHA256=
TOKENIZER=
TOKENIZER_REVISION=

MODEL_SHAPE=
PARAMETER_COUNT=
SEQUENCE_LENGTH=
MICROBATCH=
GRAD_ACCUM=
EFFECTIVE_BATCH=
CAUSAL_TARGET_BUDGET=

CURRENT_STEP=
LAST_CHECKPOINT=
LAST_CHECKPOINT_SHA256=
LAST_VALIDATION=
NAN_INF_COUNT=
SKIPPED_BATCHES=

LOCAL_CUSTODY_ROOT=
MLFLOW_PROJECTION_STATUS=
RESTORE_PROOF=

KNOWN_LIMITATIONS=
NEXT_BOUNDED_ACTION=
~~~

If any identity field is unknown, write UNAVAILABLE. Never backfill it from
memory and call the packet exact.

## 18. Adaptive structured compute and Trident

Trident tests whether one shared Helix model can spend additional computation
selectively instead of applying the same full multi-path budget to every
example. It remains an experimental path and does not replace the canonical
Branch 62 model or trainer.

### Normative boundary

```text
anchor=
exact Branch 62 model object
always available
always executed for every example
never approximated by a lane adapter

supplemental execution=
uncertain examples only
dense selected-example sub-batch
complete graph-node selection

exploration=
conditional on a declared uncertainty policy

fusion=
uncertain examples only
confident logits remain exact anchor logits

RTX 5080 sparsity posture=
whole-node structured gating only
```

Do not introduce irregular parameter, channel, attention-head, or activation-
element masks on the RTX 5080 merely because they reduce nominal FLOPs. They
require a separate matched benchmark proving improved wall-clock performance,
preserved quality, and stable kernels against the whole-node baseline.

The M0 uncertainty observation is normalized entropy at the final attended
token. A threshold is part of the run contract. Per-example decisions may form
a smaller dense sub-batch; they may not create elementwise sparse computation
and describe it as GPU acceleration.

### Truth posture

`stability` and `exploration` are hypotheses about future learned behavior, not
presently established cognitive properties. The M0 prototype selects
complementary structural paths through the shared graph. Those names become
earned only if fixed-evaluator evidence shows complementary errors and useful
repair.

Because the anchor always runs, adaptive Trident is not expected to outperform
the anchor's raw throughput. Its first performance comparison is against an
always-on three-path system. Its first scientific question is whether selective
supplemental compute improves quality enough to justify its average cost.

### Required experiment

Freeze one checkpoint, sample order, evaluator, and scoring contract. Compare:

```text
A. exact Branch 62 anchor
B. always-on three-path Trident
C. adaptive whole-node Trident
```

Required measurements:

```text
oracle gain
fusion repair rate
fusion harm rate
net repair
error correlation and calibration
supplemental-example rate
validation and downstream quality
causal targets per second
latency p50/p95
executed node calls
GPU utilization, power, and VRAM
active and total parameters
```

Stop if the supplemental paths show no oracle gain, fusion has non-positive net
repair, confident output drifts from the anchor, or the structured scheduler
does not reduce measured cost relative to always-on Trident. A positive result
authorizes a specialization-training design; it does not itself admit Trident
into the production model.

### Current executable evidence

The isolated M0 prototype is bound to:

```text
branch=experiment/helix-trident-rule30-m0
commit=8864b5d48f2c67d53d21cf7fb7a7643d86787951
tree=e63446ec1e4c89da740e7ebe08250a157e51e712
hostile_courts=18 PASS
cpu_smoke=PASS
anchor_max_abs_diff=0.0
bypassed_original_node_calls=zero
gpu_benchmark=NOT_RUN
training_specialization=NOT_ESTABLISHED
```
