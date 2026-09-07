
# HelixLM: Recurrent Heterogeneous Graph Neural Language Model

> **Why "Helix"?** A helix coils back on itself, just as our recurrent graph reuses its weights across depth iterations, refining understanding with each loop. Biological, elegant, memorable.

HelixLM is an optimized hybrid architecture for small-scale language modeling, designed for **hyperpersonalization** and **on-device AI**. It combines biological brain-inspired random graph wiring with modern SOTA primitives (hybrid attention, Mamba-2 SSD, RoPE, SwiGLU, RMSNorm, optional Titans neural memory) and full HuggingFace integration.

---

## Why HelixLM in Practice?

### Hyperpersonalization

| Requirement | How HelixLM Delivers |
|-------------|---------------------|
| Small enough to train per-user | **33M–1.2B params** = trainable on single GPU in hours |
| Rich enough to capture user style | **Heterogeneous graph** routes different info types through optimal processing paths |
| Fast inference for tool calls | **Linear attention** gives O(n) complexity; **ACT** skips unnecessary depth |
| Queryable by frontier model | **HF integration** = standard API; **chat templates** = structured responses |

### On-Device AI

| Requirement | How HelixLM Delivers |
|-------------|---------------------|
| Runs on CPU | **Linear attention** + **lightweight graph** = fast without GPU |
| Good quality for size | **Recurrent depth** reuses weights across loops = more capacity per parameter |
| Long context without forgetting | **LTI-stable recurrence** preserves state; **RoPE** generalizes length |
| Optional log-less operation | **No cloud dependency**; weights run locally |
| Responsive to long messages | **Rolling dataset** training = learns to handle arbitrary-length inputs |

## Parameter Efficiency & Scaling

Recurrent loops reuse weights (depth without parameter growth). Graph wiring creates expressive pathways without wide uniform layers. Heterogeneous nodes specialize, so no capacity is wasted on uniform operations.

---

## Use Cases

1. **Hyperpersonalization:** Train a small model (from tens of milions of parameters to ~1B parameters) from cold start or from an early - stage **partially** pre-trained model checkpoint on a **personalized** corpus enriched in it's representation of content relevant to your domain knowledge, **before** it was full fine-tunes on your own data. The model becomes an expert in the one thing generic frontier models don't know anything about: **you**, your domain knowledge, style, notes, emails, work patterns, and personal voice. A personalized small language model drafts responses to your requests with full awareness of the entire data you encoded in its weights, including the secondary details, tertiary details, and edge cases that vector DB RAG systems usually miss because they are semantically dissimilar to the prompt. This oversight leaves the loose ends that hang you if you don't catch them. Evolve to a better adapted approach.

2. **On-device AI:** Efficient inference on CPU/GPU for desktops, laptops, tablets, and mobile. Strong quality per parameter, with optional fully-local (log-free) operation for sensitive use cases.

3. **Economical and ecologically sound AI:** The brute force approach that previous generation [used to be] frontier labs use is unsustainable: It burn cash at a rate that is economically unsustainable, causes electricity prices to surge for consumers competing with them for their share of electric grid's limited capacity, and compromises water supplies. Take an exponential cut out of the problem with more efficient AI: A model architecture with better parameter efficinecy, one that can use hybrid or even flat out linear attention only, and that can saturate the model's weights / grock on as little as **one** epoch.

---

## Architecture

Standard transformers force information through a fixed-depth stack of identical layers. Biological brains don't work that way: cortical columns contain diverse cell types, lateral connections enable short-circuit pathways, and feedback loops allow iterative refinement.

HelixLM mimics this structure inside a single recurrent block:

## Architecture

Standard transformers force information through a fixed-depth stack of identical layers. Biological brains don't work that way: cortical columns contain diverse cell types, lateral connections enable short-circuit pathways, and feedback loops allow iterative refinement.

HelixLM mimics this structure inside a single recurrent block:

```
Input Tokens (B, T)
    ↓
Embedding [vocab_size × d_model]
    ↓
    +--→ e (preserved for LTI injection — full gradient flow)
    ↓
┌──────────────────────────────────────────────────────────────────┐
│  Recurrent Block (n_loops × shared graph weights)                 │
│  ├── Loop Index Embedding (sinusoidal)                            │
│  ├── HelixGraph: Randomly wired heterogeneous DAG                 │
│  │   ├── Neural columns with vertical + lateral wiring            │
│  │   ├── LinearAttnNode     : O(n) causal linear attention       │
│  │   ├── FullAttnNode       : Causal softmax (periodic, hybrid)   │
│  │   ├── SwiGLUNode         : Modern gated FFN (silu·gate × up)   │
│  │   ├── Mamba2Node         : Mamba-2 SSD (long-range, optional)  │
│  │   ├── TitansMemoryNode   : Surprise-gated persistent memory   │
│  │   ├── GateNode           : Learned softmax multi-input merge  │
│  │   ├── CCA gates          : Curriculum Component Activation    │
│  │   └── Merge RMSNorm      : Normalization after multi-predecessor merges │
│  ├── LTI Injection          : Stable recurrent state mixing       │
│  │   h_new = A·h + B·e + graph_output   (A < 1, learnable decay) │
│  └── ACT Halting            : Dynamic per-token depth allocation │
└──────────────────────────────────────────────────────────────────┘
    ↓
RMSNorm
    ↓
TiedLMHead (gradient-buffered weight tying with embedding)
    ↓
Logits / Loss
```

### Key Design Choices

| Component | What it does | Why it matters |
|-----------|--------------|----------------|
| **Neural Columns & Heterogeneous Nodes** | Each column holds diverse node types (attention variants, SwiGLU, Mamba-2, gating, optional Titans memory) instead of identical transformer blocks. | Different information pathways for different computations, like biological cortical columns. |
| **Recurrent Depth (LTI + ACT)** | The same graph weights are looped `n_loops` times. LTI injection keeps the recurrent state stable. ACT halting dynamically allocates compute per token. | Iterative refinement without parameter growth; easy tokens use 1 loop, hard reasoning uses more. |
| **Hybrid Attention** | Linear attention (O(n) complexity) in most columns, with periodic full-attention columns for exact retrieval. | Long-context efficiency without losing precise copy/lookup capability. |
| **Mamba-2 SSD (optional)** | State Space Duality with chunked parallel scan. Auto-activates when `ssm_d_state >= 64`. | Handles very long-range dependencies efficiently on CPU, CUDA, or MPS. |
| **Titans Neural Memory (optional)** | Persistent surprise-gated memory via outer-product updates (first column only by default). | Test-time memory that can retain patterns across long documents without growing KV cache. |
| **CCA (Curriculum Component Activation)** | Attention nodes start gated, gradually open over warmup steps via learned sigmoid gates. | Prevents random attention from drowning FFN signal at initialization. ~30% PPL improvement observed. |
| **TiedLMHead with Gradient Buffer** | LM head shares embedding weight. A learned linear buffer (init as identity) routes part of the gradient to prevent ~3× embedding gradient overload. | Halves parameter count (embedding table not duplicated). Safe weight tying without destabilizing training. |
| **Modern Primitives** | RoPE, SwiGLU, RMSNorm. | Proven SOTA components for convergence and generation quality. |

---

### HelixLM and upstream aspects from OpenMythos

OpenMythos published a recurrent depth with LTI stability and ACT halting. HelixLM takes that insight and makes it work inside a **heterogeneous** graph that mimics neural column and random topology connectivity found in biological brains.

| Capability | OpenMythos | HelixLM |
|-----------|-----------|---------|
| **Recurrent depth** | ✅ Same block looped | ✅ Same block looped |
| **LTI stability** | ✅ Spectral radius < 1 | ✅ Spectral radius < 1, configurable `init_A` (default `1/e`) |
| **ACT halting** | ✅ Dynamic per-token depth | ✅ Dynamic per-token depth |
| **Architecture inside loop** | Standard transformer block | **Heterogeneous random graph** |
| **Attention** | Standard full attention only | **Linear + full hybrid** |
| **Node types** | Single (transformer block) | **7+ active types (attention variants, FFN, SSM, gate, neural memory)** |
| **Positional encoding** | Standard learned | **RoPE** |
| **Activation** | GELU | **SwiGLU** |
| **Normalization** | LayerNorm | **RMSNorm** |
| **Open source** | ✅ Yes | ✅ Yes |
| **HF integration** | ❌ No | ✅ **Full PreTrainedModel, AutoModelForCausalLM** |
| **Weight tying** | Standard | **Gradient-buffered TiedLMHead** |

---

### HelixLM's Cerebros legacy

[Cerebros](https://github.com/david-thrower/cerebros-core-algorithm-alpha/) showed that biological random hyperdense vertical and lateral topology of Dense layers could outperform rigid layer stacks. It generated text without attention on small data, but required elaborate integration and clashed with standard model-structure paradigms. HelixLM smoothly integrates that topological insight into a modern, HF-compatible LLM backbone.

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Smoke Test (CPU, character-level)

```bash
cd helix_lm
python smoke_test.py
```

### Minimal Demo (CPU, GPT-2 tokenizer)

```bash
python quick_demo_cpu.py
```

## HuggingFace integration

This minimal example shows the document-aware SFT path. See
`docs/training/BRANCH62_PRETRAIN_HANDOFF.md` for indexed pretraining and its
exact sample-order, recovery, checkpoint, and MLflow contracts.

```python
from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer

tokenizer = HelixTokenizer("gpt2")
cfg = HelixConfig.small_v2(
    vocab_size=tokenizer.vocab_size,
    seq_len=64,
    batch_size=2,
    epochs=1,
)

cfg.pad_token_id = tokenizer.pad_token_id
cfg.eos_token_id = tokenizer.eos_token_id
cfg.bos_token_id = tokenizer.bos_token_id
model = HelixForCausalLM(cfg)

trainer = Trainer(
    model=model,
    cfg=cfg,
    train_texts=["First training document.", "Second training document."],
    val_texts=["Held-out validation document."],
    tokenizer=tokenizer,
    output_dir="./sft-checkpoints",
    grad_accum_steps=1,
    use_amp=False,
    min_tail_len=16,
)
history = trainer.train(num_epochs=1)

# Save/load in standard HF format
model.save_pretrained("./my-helix-model")
tokenizer.save_pretrained("./my-helix-model")

# Load later with standard transformers Auto classes
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./my-helix-model")
tokenizer = AutoTokenizer.from_pretrained("./my-helix-model")

# Optional publication: authenticate first, and push only after the local
# save_pretrained output above exists and has been read back.
# model.push_to_hub("your-hf-username/your-repo-id")
```

---

## Training

### Continuous pretraining with a globally ordered disk-backed store

Full-corpus causal pretraining uses `PretrainTrainer` with an EOS-joined,
non-overlapping sample store and a persisted epoch permutation. This is separate
from the document-aware SFT behavior of `Trainer`.

The normal API accepts a Hugging Face `IterableColumn` as `train_texts` and
compiles or reuses the verified store automatically:

```python
from datasets import load_dataset
from helix_lm import PretrainTrainer

texts = load_dataset(
    "codelion/sutra-10B",
    revision="415549cff1a92b69df8b88c6108faa6097457068",
    split="train",
    streaming=True,
)["text"]

trainer = PretrainTrainer(
    model=model,
    cfg=cfg,
    tokenizer=tokenizer,
    train_texts=texts,
    pretrain_store_dir="./pretrain_store",
    pretrain_source={
        "dataset": "codelion/sutra-10B",
        "revision": "415549cff1a92b69df8b88c6108faa6097457068",
        "split": "train",
        "text_column": "text",
        "tokenizer": "gpt2",
    },
)
```

The trainer owns compilation; callers do not need a separate preprocessing
script. For long-lived full-corpus runs, pass an explicit
`pretrain_store_dir` so the verified store can be reused and audited.

See `docs/training/BRANCH62_PRETRAIN_HANDOFF.md` for the canonical launcher,
comparison profiles, exact resume boundary, and MLflow metric vocabulary.

Run the independent fixture equivalence court with:

```bash
python pretrain_data_court.py
```

After training is idle, replay the complete compiled store and enforce the
Branch 60 storage-throughput floor with:

```bash
python pretrain_data_court.py \
  --sample-store /data/sutra-gpt2-t1024 \
  --permutation /data/sutra-gpt2-t1024/permutations/epoch-0000-seed-42.u32 \
  --batch-size 2 \
  --num-workers 4 \
  --output /path/to/pretrain-data-court-terminal.json
```

The full-store court compares every observed sample ID with the persisted
permutation, rejects duplicates or omissions, verifies labels and masks, binds
ordered token and sample-ID roots, and reports measured storage-only sample and
causal-target throughput. Run it outside an active training window so the
measurement does not perturb the model run it is meant to qualify.

### Document-Aware Chunking

`DocumentAwareDataset` splits documents into non-overlapping chunks without crossing document boundaries. Only padding positions are masked in labels, giving 100% token utilization on real text.

```python
from helix_lm import create_document_loader, Trainer

loader = create_document_loader(
    texts, tokenizer,
    seq_len=512,
    batch_size=8,
    stride=512,    # default: no overlap. See warning above if stride < seq_len
)

trainer = Trainer(
    model=model, cfg=cfg,
    train_loader=loader,
    val_loader=val_loader,
    tokenizer=tokenizer,
    output_dir="./checkpoints",
)
trainer.train(num_epochs=3)
```

### Or let the Trainer handle everything (Generally recommneded)

```python
trainer = Trainer(
    model=model, cfg=cfg,
    train_texts=train_texts,
    val_texts=val_texts,
    tokenizer=tokenizer,
)
trainer.train(num_epochs=3)
```

---

## Generation

```python
# Standard generation
generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=50)

# Extended generation beyond training seq_len
generated = model.generate_ext(
    input_ids,
    max_new_tokens=500,
    temperature=0.6,
    top_k=40,
    top_p=0.95,
    stop_strings=["</s>", "\n\n"],
)
```

### Recipe for vaible - scale training: See 400M_production_trainer.py

---

## Project Structure

```
helix_lm/
  __init__.py           - Package exports
  config.py             - HelixConfig with presets (tiny through xxl)
  tokenizer.py          - Multi-backend tokenizer (char / GPT-2 / Qwen / custom)
  rope.py               - Rotary positional embeddings
  nodes.py              - Heterogeneous node types (7+ active types)
  graph.py              - HelixGraph executor with random wiring, CCA, attention mask propagation
  recurrent.py          - Recurrent block (LTI injection with configurable init_A + ACT halting)
  model.py              - HelixLMCore (non-HF)
  hf_model.py           - HelixForCausalLM (HF PreTrainedModel), TiedLMHead with gradient buffer
  dataset.py            - DocumentAwareDataset, HelixDataset, HF streaming integration
  trainer.py            - Production training loop with gradient accumulation, AMP
  mamba2.py             - Mamba-2 SSD with parallel scan
  smoke_test.py         - Self-contained CPU test
```
---

## Configuration Reference

### Core Structural Dimensions

| Parameter | Effect | Practical Tip |
|-----------|--------|---------------|
| `d_model` | Width of the model | 128 for smoke tests; 192–256 for small experiments; 512+ for production |
| `n_columns` | Number of neural columns | 2 for fast experiments; 4–7 for large models |
| `n_loops` | Recurrent iterations | 1 for speed; 2–4 for iterative reasoning depth |
| `n_heads` | Attention heads | Must divide `d_model`. 4–8 for small models; 16–32 for large |

### Attention & Memory

| Parameter | Effect | Default |
|-----------|--------|---------|
| `attention_mode` | `"linear"`, `"full"`, or `"hybrid"` (recommended) | `"hybrid"` |
| `hybrid_full_attention_interval` | Every Nth column gets full attention | `4` |
| `use_ssm` | Enable Mamba-2 SSD nodes | `False` |
| `ssm_d_state` | Mamba-2 state dimension (≥64 for optimized path) | `64` |
| `use_titans_memory` | Titans neural memory in first column | `False` |

### Weight Tying

| Parameter | Effect | Default |
|-----------|--------|---------|
| `tie_word_embeddings` | Share embedding weight with LM head | `True` |
| `grad_buffer_ratio` | Fraction of LM head gradient routed through buffer. `0.0` = standard tying. `1/e` = balanced. | `1 / e` |

### Training & Stability

| Parameter | Effect | Default |
|-----------|--------|---------|
| `lti_init_A` | Initial decay for recurrent state. `None` → `1/e ≈ 0.368` (mathematically grounded) | `None` |
| `use_cca` | Curriculum Component Activation for attention warmup | `False` |
| `cca_warmup_steps` | Steps over which attention gates gradually open | `5000` |
| `dropout` | Regularization rate | `0.05` |
| `lr` | Learning rate | `3e-4` (optimal often higher, e-3 - 2e-3) |
| `weight_decay` | AdamW weight decay | `0.1` |
| `grad_clip` | Max gradient norm | `1.0` |

---

## Existing Research We Build On

| Idea | Source / Inspiration | How we use it |
|------|---------------------|---------------|
| **Hybrid Attention** | Linear + full attention interleaving | 3:1 or 4:1 linear-to-full ratio; O(n) training with exact copy layers |
| **Recurrent Depth** | OpenMythos / Universal Transformers | Same graph weights looped `n_loops` times; fewer params, iterative refinement |
| **LTI Stability** | OpenMythos | Log-parameterized state decay to keep spectral radius < 1; configurable `init_A` |
| **Mamba-2 SSD** | Dao & Gu (2024) | Chunked associative scan for selective SSM; auto-enabled at `d_state >= 64` |
| **Biological Graph Wiring** | Cerebros / Random DAGs | Random vertical & lateral edges instead of strict feedforward stacks |
| **Titans Neural Memory** | Behrouz et al. (2025) | Optional first-column persistent memory with surprise-gated outer-product updates |
| **CCA** | Curriculum learning literature | Gradual attention wake-up: learned sigmoid gates ramp from 0.05 to 1.0 over warmup |
| **SwiGLU + RMSNorm + RoPE** | Llama / PaLM / GPT-NeoX | Modern SOTA primitives throughout |

---

## ⚠️ Known Interaction: Stride < seq_len + Gradient Buffer

### The Issue

When `stride < seq_len` (overlapping chunks), `DocumentAwareDataset` masks the overlapping token positions with `labels = -100`. These "context-only" positions flow through the `TiedLMHead` buffer in the forward pass but contribute **zero gradient** in the backward pass. Over many steps, this forward/backward asymmetry can cause the buffer to drift, degrading model quality.

### Impact

| Training mode | Affected? | Severity |
|--------------|-----------|----------|
| `stride = seq_len` (default, any sequence length) | ❌ No | Safe — all positions contribute to loss |
| `stride < seq_len` (overlap enabled) | ⚠️ Yes | Risk scales with overlap ratio. PPL degradation of ~2.5× observed at 50% overlap on tiny models |
| `grad_buffer_ratio = 0.0` (standard tying, no buffer) | ❌ No | Always safe, at any stride |

### Guidelines

- **Default training (stride = seq_len):** Any `grad_buffer_ratio` is safe. No action needed.
- **Overlap training (stride < seq_len):** Either set `grad_buffer_ratio=0.0`, or accept the quality trade-off. This combination should only be used for ablations, not production training.
- **If you need overlap for data efficiency:** Consider using `grad_buffer_ratio=0.0` (standard weight tying without the learned buffer) or revert to untied embeddings for overlap training runs.

```python
# Safe: default stride, any buffer ratio
cfg = HelixConfig.micro(grad_buffer_ratio=1/e)  # fine

# Safe: overlap training, buffer disabled
cfg = HelixConfig.micro(grad_buffer_ratio=0.0)
loader = create_document_loader(texts, tokenizer, seq_len=512, stride=256)  # overlap OK

# ⚠️ Risky: overlap training + buffer enabled
cfg = HelixConfig.micro(grad_buffer_ratio=0.5)
loader = create_document_loader(texts, tokenizer, seq_len=512, stride=256)  # avoid
```
Gradient buffer is generally not indicated above small ablation trials. At scale it may be best set to 0 anyway. 

## ⚠️ Other caveats

### Crystallization in smallest model configurations and how to work with it:

On small model configurations (e.g. d_model of 256, maybe 384) you may experience crystallization (aka the optimizer gets lost in a local minima rabbit hole and overfits on subset of the data having repetitive tokens).

Raising d_model to 512 + may be your cleanest fix. If you don't have the data or hardware to support increasing d_model:

- Set a high learnng rate (usually 1e-3 to 2e-3)
- Use the 'KITA' scheduler / spiking learning rate scheduler

Cosine anealing does not appear optimal on small data sets (400M and below, however, a contant learning rate or constant rate with periodic spikes in the rate both work well).

Muon optimizer may be structurally incompatible with this architecture.

---


## License

This project is open-source under a modified Apache 2.0 license. See [license.md](license.md) for full terms.

## Cite this architecture:

```bibtex
@software{helixlm2026,
  title = {HelixLM: Recurrent Heterogeneous Graph Neural Language Model},
  year = {2026},
  note = {Open-source small language model architecture}
}
