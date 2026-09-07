# Helix Experiment Clipboard

> Living experiment note. This file records decisions, controls, and next-run
> contracts; it is not a substitute for MLflow artifacts or immutable run
> receipts.

## Current question

Can a small neural-cellular-automata curriculum improve the sample efficiency
of the Branch 62 language model enough to preserve quality at a smaller
`d_model`, allowing the RTX 5080 budget to be reallocated to larger
microbatches, additional recurrent loops, or—after its construction contract is
repaired—richer graph topology?

The answer is not established yet. The current experiment tests curriculum
transfer at the unchanged Branch 62 width. A successful result would authorize
a width-rescue experiment, not an immediate architecture rewrite.

## Bound experiment

### Control

```text
MLflow run:
b53c2364ce234a64bc48216acbdfac9c

Configuration:
d_model=768
n_heads=12
n_columns=3
nodes_per_column=(3,3,3)
n_loops=4
ffn_expansion=3.0
sequence_length=1024
lateral_p=0.8
vertical_p=0.9
vertical_depth=2
learning_rate=2e-4
microbatch=3
gradient_accumulation=28
effective_batch=84
```

### NCA curriculum stage

```text
MLflow run:
87c68b4b90604cfe87b15b6330e6ebd2

Terminal:
PASS, 672/672 optimizer steps

Final validation loss:
8.7201387

Final target accuracy:
0.0099557

Transferred artifact:
/home/mo/DEV/Thunderline/.tmp/helix-nca-branch62-v0/full-treatment-v0/nca-run/nca-core-transfer.pt

Artifact SHA-256:
5b659f24a5bf01754e3b5a333c5dbe5abdf89f95485abeaf6176f43015ef5827
```

The NCA stage trained a synthetic-vocabulary model and transferred only the
non-vocabulary neural core. The GPT-2 embedding, tied output head, optimizer,
scheduler, RNG state, and sample cursor did not transfer. The treatment's
language embedding and output head therefore begin from the same seeded state
as the control.

This is curriculum transfer. It is not yet online cellular-automata state, DAG
memory, retrieval, or additional inference-time machinery.

### Sutra treatment

```text
MLflow run:
1ad8d124051a4098ab28aaa1654d3724

Source head:
e729a6f8372289eb4839f15245fa5342b7135846

Source tree:
682c990aab815f183d355cf481bb4384686a4837

Dataset:
codelion/sutra-10B

Status at the 2026-09-05 checkpoint:
active, stable, zero skipped batches
```

At optimizer step 92, the treatment reported loss `7.73399` and approximately
`8,639` causal targets per second for the session. Earlier matched-step evidence
showed the treatment and control effectively tied in loss while the treatment
was approximately six percent slower. These are interim observations, not an
admission verdict.

## Immediate next steps

1. Allow the current Sutra treatment to finish without source or runtime
   mutation.
2. Freeze its final checkpoint and run the same fixed Lighteval suite used for
   the Branch 62 control.
3. Compare at equal causal-target counts:
   - training-loss trajectory;
   - validation trajectory;
   - time and targets required to reach fixed loss thresholds;
   - causal targets per second;
   - peak VRAM;
   - skipped/non-finite batches;
   - every frozen Lighteval task and scoring semantic.
4. Decide whether NCA improved final quality, learning speed, neither, or both.
5. Open the width-rescue experiment only if the paired evidence is promising.

Final loss alone is insufficient. The main curriculum question is whether the
treatment reaches the same quality with fewer causal targets or preserves more
quality after width is removed.

## `NCA_WIDTH_RESCUE_V0`

The first width-rescue comparison changes only model width.

```text
d_model=672
n_heads=12
head_dim=56
n_columns=3
nodes_per_column=(3,3,3)
n_loops=4
ffn_expansion=3.0
sequence_length=1024
lateral_p=0.8
vertical_p=0.9
vertical_depth=2
learning_rate=2e-4
microbatch=3
gradient_accumulation=28
effective_batch=84
```

Run two fresh subjects:

```text
A. d672 control, no NCA curriculum
B. d672 treatment, identical NCA curriculum and transfer law
```

Both subjects must share the same seed, corpus revision, compiled sample store,
permutation, optimizer schedule, causal-target budget, checkpoint cadence, and
evaluator contract. Do not change batch geometry during this comparison.

`d_model=672` is a 12.5% width reduction and remains divisible by twelve heads.
It creates a large enough intervention to expose a real rescue effect while
preserving the current head count. The exact parameter count must be measured
from the instantiated model and recorded; it must not be inferred from width
alone.

### Width-rescue interpretation

```text
d672 NCA beats d672 control but both trail d768 materially:
curriculum helps, width not rescued

d672 NCA approaches d768 control while d672 control trails:
credible width-rescue candidate

d672 NCA and d672 control are equivalent:
no evidence NCA bought width

d672 NCA harms stability or downstream scores:
reject this transfer posture
```

One seed produces a candidate, not a general claim. Any promoted width-rescue
result should be repeated with additional declared seeds.

## Reallocating the recovered budget

Only after width rescue is established should the recovered memory or compute
be spent. Each step is a separate ablation.

### 1. Throughput and occupancy

```text
microbatch: 3 -> 4
gradient_accumulation: 28 -> 21
effective_batch: remains 84
```

This is the cleanest RTX 5080 optimization because it can improve device
occupancy while preserving the optimizer's effective batch. It must be tested
after, not inside, the width-rescue comparison.

### 2. Recurrent computation

```text
n_loops: 4 -> 5
```

Loops reuse weights, so this is the most direct way to exchange model width for
additional computation without immediately restoring the removed parameter
count. Compare it against the admitted `d672, loops=4` subject with all other
settings fixed.

### 3. Graph topology

Do not spend the budget on `nodes_per_column` yet. The current graph builder
does not use that configuration value to construct its node specification, and
the Branch 62 launcher truthfully records:

```text
nodes_per_column_graph_effective=false
```

Before any node-count ablation:

1. wire `nodes_per_column` into graph construction;
2. add shape, node-count, execution, and hostile mutation tests;
3. serialize the exact generated topology and its canonical root;
4. prove the requested node count changes executed computation;
5. measure the resulting parameter and merge-layer changes.

Likewise, `vertical_depth=3` cannot express three preceding-column levels in a
three-column graph. Testing depth three requires at least four columns. Adding
a column and increasing vertical depth are therefore a later topology
experiment, not part of width rescue.

### 4. Connectivity probabilities

Do not raise `lateral_p` and `vertical_p` together with another architectural
change. Those values alter the seeded graph, predecessor fan-in, merge layers,
parameter count, and execution cost. Freeze and record the graph manifest for
every probability subject, then compare probabilities independently.

### 5. FFN expansion

The current admitted subject already uses `ffn_expansion=3.0`. A later increase
must be treated as its own parameter-allocation experiment. Do not combine it
with the initial width cut or claim that an NCA result caused an FFN result.

## Evaluation and admission posture

Every candidate must record:

```text
source head and tree
dataset name and immutable revision
compiled-store and permutation roots
tokenizer identity and vocabulary size
model parameter count
d_model, heads, head dimension
columns and executed nodes per column
loops and FFN expansion
lateral_p, vertical_p, vertical_depth
sequence length
microbatch, accumulation, effective batch
learning rate and scheduler
causal targets processed
raw UTF-8 bytes exposed
throughput and step time
peak allocated and reserved VRAM
gradient norm and skipped batches
checkpoint identities
frozen evaluator identity and task revisions
all downstream scores
```

Promotion requires stable training, zero unexplained skipped/non-finite batches,
a reproducible checkpoint, and fixed-evaluator evidence. A lower training loss
does not supersede a downstream regression.

## Planned larger A/B/C experiment

The curriculum-transfer experiment is deliberately smaller than the eventual
stateful semantic-computation test. After the baseline is stable, freeze one
checkpoint and evaluator and compare:

```text
A. local-context transformer
B. transformer plus retrieval
C. transformer plus automata state and DAG retrieval
```

Measure distant recall, causal reasoning, KV-cache bytes, automata-state bytes,
transition-log bytes, latency, active parameters, and historical-token distance.
This later experiment asks whether automata and DAG state improve effective
token capacity at runtime; the current curriculum experiment does not establish
that claim.

## David-ready summary

> We are testing whether the NCA curriculum can rescue a 12.5% width reduction
> from `d_model=768` to `672`. If that is proven, we will preserve effective
> batch while testing microbatch `4` with accumulation `21`, then test recurrent
> depth `5`. Additional nodes and `vertical_depth=3` wait until
> `nodes_per_column` genuinely controls graph construction and the topology is
> hash-bound. We will not change width, loops, batching, topology, and
> connectivity in one run because that would destroy causal attribution.

## Nonclaims

- The active treatment has not yet established an NCA advantage.
- NCA curriculum does not automatically replace model width.
- Current work does not provide online automata memory or DAG retrieval.
- `nodes_per_column=(3,3,3)` is recorded configuration, not currently proven
  graph-effective configuration.
- An interim loss advantage does not equal a downstream-model win.
- No candidate becomes canonical without identical evaluation and checkpoint
  evidence.

## 2026-09-05 council: Branch 62 100x training-speed challenge

### Question

Hypothetically, if the team had to improve Branch 62 training speed by at least
100x without violating David's Branch 62 caveats, what would constitute an
honest path and what should be tested first?

This is a research challenge and performance backlog, not a claim that a 100x
speedup has been achieved.

### Bound subject

```text
source_head=
adb5efd93c83d226a238d954be15d87893d7a8df

source_tree=
5043998cc8b763bfebb3282012167e846ab6b639

dataset=
codelion/sutra-10B

tokenizer=
GPT-2

sequence_length=
1024

d_model=
768

n_columns=
3

nodes_per_column=
(3,3,3), graph-effective in this subject

n_loops=
4

ffn_expansion=
3.0

lateral_p=
0.8

vertical_p=
0.9

vertical_depth=
2

learning_rate=
2e-4

microbatch=
2

gradient_accumulation=
42

effective_batch=
84

strict_nan_check=
true

parameter_count=
148,511,267
```

The observed reference throughput was approximately `4,111-4,114` causal
targets per second. A literal 100x target is therefore approximately `411,400`
causal targets per second.

### Exact-source performance finding

The graph constructor repeats the base attention/FFN pattern to satisfy
`nodes_per_column=(3,3,3)`. Each column therefore contains two multiscale
attention nodes, one SwiGLU node, and a gate. Across three columns and four
recurrent loops, one training forward executes:

```text
6 multiscale attention nodes per graph
x 4 recurrent loops
= 24 multiscale attention executions

each multiscale execution:
1 local attention
+ 1 coarse attention
+ 8 independent compressed views
= 10 attention paths

total attention paths per forward=
24 x 10 = 240
```

In strict mode, every path can call both `_safe_attention/6` and
`_check_finite_output/3`. Each currently performs
`torch.isfinite(...).all().item()`. On an ordinary non-empty training batch,
the source-derived upper bound is therefore:

```text
240 attention finite checks
+ 240 output finite checks
= up to 480 GPU-to-host synchronizing scalar reads per forward
```

This count is source-derived, not profiler-measured. Its wall-clock cost remains
to be established. The strict checks are safety behavior and must not simply be
removed.

Other source-visible fragmentation candidates are:

- causal masks, compressed boundaries, and expansion indices rebuilt inside
  forward paths;
- eight independent compressed views executed serially in Python;
- Python dictionary/list graph routing on every recurrent loop;
- manually composed matmul/softmax/matmul attention rather than an admitted
  scaled-dot-product-attention kernel;
- static-shape work not yet proven safe under `torch.compile` or CUDA Graphs.

### Council disposition

The challenge must use three separate scoreboards:

```text
RAW_THROUGHPUT=
causal targets per wall-clock second on a fixed subject

TIME_TO_TARGET_QUALITY=
wall time and causal targets required to reach a frozen validation terminal

DECISIONS_PER_GPU_DAY=
correctly admitted experimental decisions per unit of GPU time
```

Council conclusion:

```text
100x raw throughput on one RTX 5080 while preserving the exact subject=
not presently credible

single-card systems optimization target=
3x to 10x is aggressive but testable; no promise

100x time-to-quality=
possible only as a separate scientific claim involving curriculum,
progressive training, retrieval, automata, or data selection

100x experimental decision efficiency=
plausible through paired low-cost rungs, sequential stopping,
frozen evaluators, and full-run confirmation

literal 100x raw-throughput route=
single-GPU optimization plus distributed hardware,
with deterministic global-batch and sample-order courts
```

Multipliers must not be added naively. Every optimization changes the remaining
bottleneck. The unnamed competing team's `50x` or `50%` result is unusable
without its denominator, subject, hardware, quality terminal, and accounting
rules.

### First bounded performance experiment

```text
experiment=
HELIX_B62_STRICT_FINITE_AGGREGATION_V0

goal=
measure whether strict per-path host synchronization is a dominant Branch 62
bottleneck while retaining fail-closed numerical behavior
```

Variants, applied sequentially rather than blended:

```text
A=current per-path strict .item() checks

B=device-side finite-flag aggregation with one host synchronization at the
  complete-forward boundary; any observed non-finite value must still refuse
  the optimizer step

C=B plus cached causal masks, boundaries, and expansion indices

D=C plus exact-weight-preserving vectorization of the eight compressed views

E=D plus torch.compile after graph-break inspection

F=E plus scaled-dot-product attention only after causal, padding, output,
  gradient, and injected-corruption equivalence courts pass
```

Required controls:

```text
initial checkpoint and parameters=
identical

dataset revision, compiled store, and permutation=
identical

batch order and causal-target accounting=
identical

optimizer, schedule, and effective batch=
identical

evaluator and checkpoint preparation=
identical
```

Required measurements:

```text
causal targets per second
microbatch and optimizer-step latency p50/p95
CUDA synchronization count
kernel-launch count
CPU launch gaps
GPU utilization
allocated and reserved VRAM
validation NLL
skipped and non-finite batches
```

Hostile numerical courts:

```text
inject non-finite value into local path
→ hard failure before optimizer.step()

inject non-finite value into coarse path
→ hard failure before optimizer.step()

inject non-finite value into every compressed-view position
→ hard failure before optimizer.step()

detected corruption followed by parameter mutation
→ RED
```

If aggregation does not materially reduce measured synchronization or latency,
that is a useful negative result: proceed according to profiler attribution
rather than stacking speculative optimizations.

### Distributed arithmetic, not an authorization

The current effective batch of 84 has useful exact divisors. Candidate logical
layouts include:

```text
6 workers x microbatch 2 x accumulation 7 = 84
7 workers x microbatch 3 x accumulation 4 = 84
14 workers x microbatch 3 x accumulation 2 = 84
```

These equations do not prove equivalent training. Distributed admission still
requires a deterministic global permutation, rank assignment, global-batch
identity, all-reduce semantics, failure/restart behavior, and fixed-quality
comparison. If David's caveat requires identical serial presentation rather
than identical logical global batches, ordinary data parallelism does not
satisfy it.

### Relationship to automata and DAG work

Automata, DAG retrieval, curriculum transfer, sparse activation, and progressive
training remain promising for time-to-quality and effective token capacity.
They do not count as exact Branch 62 raw-throughput improvements if they change
the model, sample order, activated computation, or quality target.

The immediate performance lane is deliberately less glamorous:

> Preserve the computation and safety law, remove unnecessary synchronization
> and dispatch overhead, measure the result, and only then decide whether the
> 100x objective requires distributed hardware or a separately admitted model.

### Live-run observation at capture time

The active `n333` Sutra run was not interrupted during this analysis. A
five-second heartbeat advanced from batch `11,929` to `11,940`; loss moved from
`7.4537` to `7.4529`; observed throughput remained approximately `4,111` causal
targets per second. The host GPU readback reported approximately 94% utilization,
14.32 GiB used, 166.8 W, and 57 C. This established active progress at that
instant, not final model quality or completion.

### Early-stop terminal and width/topology successor

The NCA-transferred `d768/c3/n333` treatment was intentionally stopped on
operator direction after the session throughput stabilized near `4,052` causal
targets per second. Its last MLflow metric was optimizer step `392`; the durable
local checkpoint remains step `250`. MLflow run:

```text
acf1869bc9f64b45832aeef86163db9e
```

The successor is a no-NCA throughput probe that changes only the model width,
head count, node cardinality, and memory-safe microbatch shape:

```text
source_head=6b4eb3ad269ed317e0de4c82066360eb9788e62c
source_tree=30c10992f9a7bc069809a14d1b9fca842cb896e0
d_model=384
n_heads=6
head_width=64
nodes_per_column=2,2,2
graph_nodes=9
graph_edges=27
parameters=35,795,156
batch_size=6
grad_accum=14
effective_batch=84
sequence_length=1024
ffn_expansion=3.0
learning_rate=0.0002
nca_curriculum=false
mlflow_run_id=82b9e0907c52453fa88524a560b1d8fa
```

Initial readback at optimizer step `10` showed approximately `18,145` causal
targets per second with zero skipped batches. This is about `4.48x` the stopped
`n333` treatment's session throughput, but it is not a quality result: the model
has roughly one quarter as many parameters and remains inside warmup.

### Correction: retain the width probe as control and restore the NCA treatment

The preceding `d384/n222` launch did not use the NCA curriculum. It was therefore
stopped after its step-250 checkpoint became durable and is retained only as the
matched no-NCA control, not as the intended successor treatment.

```text
control_mlflow_run_id=82b9e0907c52453fa88524a560b1d8fa
last_observed_optimizer_step=254
durable_checkpoint_step=250
model_sha256=fb5e7852502d3afd655b1afd8267868782a6122b91ef08ee465ff96763feb406
pretrain_data_state_sha256=9db6f0a563d5771d170f823d070d674817e385fa56f8a0b50b6a37a0e75d3745
terminal=operator_interrupted_after_checkpoint
scientific_posture=matched_no_nca_control
```

The first attempted NCA restart exposed a profile-binding defect:
`train_nca_stage.py` hardcoded `rtx5080-relative`, so an invocation intended for
`d384/n222` constructed `d768/n333`. The run was stopped as soon as its emitted
contract revealed the mismatch. Its evidence was preserved at
`nca-treatment-v0-misbound-d768-n333`; no artifact was deleted.

The NCA stage now requires and records an explicit profile. Focused verification
passed 20 unit tests plus Python compilation and `git diff --check` at:

```text
source_head=8539904c89bbad233359fe10037ac8418871cd3d
source_tree=224fee1487446127efac41d34bc994f71f30d0db
```

The corrected NCA treatment is running with the intended compact topology:

```text
mlflow_run_id=1921fa87cef547d88310550d44af705e
run_name=hlx-b62-nca11m-s1024-d384-n222-260905-1407
d_model=384
n_heads=6
nodes_per_column=2,2,2
graph_nodes=9
graph_edges=27
parameter_count=20,337,236
sequence_length=1024
ffn_expansion=3.0
nca_train_samples=10,752
nca_validation_samples=256
nca_causal_targets_per_sample=935
microbatch=4
grad_accum=4
effective_batch=16
learning_rate=0.0001
curriculum_manifest_sha256=b1e1aa574cf0b6c7085817013342fd9bae702d6e03d8d54d26df4a50eee76bfe
run_contract_sha256=8c468e73aab6392a51f0b307b72207400a7f756d4625fafa7aa59f67408dbfd7
```

At optimizer step `77`, the NCA stage had processed `1,151,920` causal targets;
session optimizer-step time was approximately `1.259` seconds, peak VRAM was
approximately `5.89 GB`, and no non-finite or skipped-batch failure had been
reported. Loss was still an early curriculum-training observation (`8.8347`),
not a downstream Sutra quality comparison.

After the NCA stage reaches step `672`, the launcher admits only a weights-only
transfer into an otherwise matching Sutra `d384/n222` run. Optimizer, scheduler,
RNG, and data-cursor state are not transferred. That Sutra treatment is the
comparison against the preserved no-NCA control; the NCA-stage loss itself is
not directly comparable to the control's Sutra loss.

### References

- PyTorch compiler: https://docs.pytorch.org/docs/2.14/torch.compiler.html
- PyTorch scaled dot-product attention:
  https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch CUDA Graphs:
  https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- PyTorch DistributedDataParallel:
  https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html

## 2026-09-05 Trident adaptive whole-node compute prototype

### Thesis

Helix should not spend identical computation on every example. The exact
Branch 62 model remains an always-available anchor. Only examples whose anchor
output crosses a declared uncertainty threshold earn supplemental stability
and exploration passes, and only those examples may receive fused logits.

This is an adaptive-quality hypothesis, not a claim that Trident is faster than
the Branch 62 anchor. Because the anchor always runs, the initial performance
target is:

```text
better quality per unit of additional compute
and
lower average cost than an always-on three-path system
```

The longer-range hypothesis is that Helix can scale useful computation through
conditional traversal of shared heterogeneous graph weights rather than only
through a wider dense model.

### Exact prototype subject

```text
repository=
Thunderline isolated experiment worktree

branch=
experiment/helix-trident-rule30-m0

commit=
8864b5d48f2c67d53d21cf7fb7a7643d86787951

tree=
e63446ec1e4c89da740e7ebe08250a157e51e712

production_path_changes=
none

gpu_launch=
none
```

Implementation lives under:

```text
experiments/helix_trident_rule30_m0/
```

### Executable contract

```text
ANCHOR=
exact Branch 62 HelixForCausalLM object
always executes for every example
never gated or lane-conditioned

UNCERTAINTY=
normalized entropy at the last attended token
evaluated independently per example
default threshold 0.85
supplemental route when score > threshold

STABILITY=
earliest declared compute nodes retained per column
complete remaining compute nodes bypassed

EXPLORATION=
latest declared compute nodes retained per column
complete remaining compute nodes bypassed
executes only for uncertain examples

FUSION=
default weights stability=0.25, anchor=0.50, exploration=0.25
applied only to uncertain examples
confident anchor logits remain unchanged

GPU SHAPE=
uncertain examples form a dense supplemental sub-batch
gating selects complete module names
gate plumbing remains active
no parameter, channel, head, or activation-element mask
```

The stability and exploration names describe intended future specialization.
The present selector only chooses complementary structural paths; it has not
yet proved that either path learned a stable cognitive role.

The M0 implementation temporarily replaces bypassed modules with whole-node
identity modules and restores the originals after each serialized adaptive
forward. That is admissible for an isolated prototype and not yet a production
concurrency design.

### Verified terminal

```text
hostile_unittest_courts=
18 PASS

cpu_smoke=
PASS

anchor_max_abs_diff=
0.0

confident_executed_lanes=
anchor only

confident_fusion_count=
zero

mixed_batch_uncertain_examples=
one of two

supplemental_active_nodes_per_lane_on_n333=
6

bypassed_original_node_calls=
zero

finite_backward_to_lane_adapters_and_shared_trunk=
PASS

active_rtx5080_run_modified_or_stopped=
false
```

The courts additionally refuse malformed uncertainty thresholds, malformed
per-example overrides, fractional node budgets, and invalid fusion weights.
Invalid fusion weights are rejected before supplemental execution begins.

### Required RTX 5080 benchmark

Freeze one Branch 62 checkpoint, data order, evaluator, and quality terminal.
Compare:

```text
A. exact Branch 62 anchor
B. all three Trident paths on every example
C. adaptive whole-node Trident at declared escalation rates
```

Calibrate or sweep thresholds that yield approximately 5%, 10%, 20%, and 50%
supplemental-example rates. Record:

```text
validation NLL and perplexity
fixed downstream evaluator scores
anchor error and fusion error
oracle gain
fusion repair rate
fusion harm rate
net repair = repair rate - harm rate
escalation rate
causal targets per second
batch and example latency p50/p95
executed original-node calls
GPU utilization and power
allocated and reserved VRAM
active and total parameters
```

### Admission and kill criteria

```text
oracle gain approximately zero
-> supplemental paths are not complementary; stop

net repair <= 0
-> fusion harms at least as often as it repairs; stop

confident output differs from exact anchor
-> contract violation; stop

structured gating produces no measured wall-clock saving versus always-on Trident
-> do not claim adaptive efficiency

downstream gain disappears under the frozen evaluator
-> do not promote
```

Irregular element masks remain forbidden on the RTX 5080 unless a separate
matched benchmark proves they improve real wall-clock performance and preserve
quality against this structured whole-node baseline. Reduced theoretical FLOPs
alone are insufficient.

### David-ready explanation

> We preserved Branch 62 as an exact always-available anchor. Trident adds two
> weight-sharing partial graph traversals that run only for uncertain examples,
> using complete-node gating and dense supplemental sub-batches instead of
> irregular tensor masks. The hypothesis is that Helix's heterogeneous graph
> contains complementary computation, and uncertainty routing can purchase
> that computation only when its expected repair value exceeds its cost. We
> will measure oracle gain, error diversity, net repair, escalation rate, and
> actual CUDA latency before training specialization or claiming a speedup.
