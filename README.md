
# HelixLM: Recurrent Heterogeneous Graph Neural Language Model

> **Why "Helix"?** A helix coils back on itself — just as our recurrent graph reuses its weights across depth iterations, refining understanding with each loop. Biological, elegant, memorable.

HelixLM is an optimized hybrid architecture for small-scale language modeling, designed for **hyperpersonalization** and **on-device AI**. It combines biological brain-inspired random graph wiring with modern SOTA primitives (hybrid attention, Mamba-2 SSD, RoPE, SwiGLU, RMSNorm, optional Titans neural memory) and full HuggingFace integration.

---

## Use Cases

1. **Hyperpersonalization** — Train a small model (from sub-million to ~1B parameters) from cold start on a personal corpus, then full fine-tune on the user's own data. The model becomes an expert in the one thing generic frontier models don't know: **you** — your domain knowledge, style, notes, emails, and work patterns.

2. **On-device AI** — Efficient inference on CPU/GPU for desktops, laptops, tablets, and mobile. Strong quality per parameter, with optional fully-local (log-less) operation for sensitive use cases.

---

## Architecture

Standard transformers force information through a fixed-depth stack of identical layers. Biological brains don't work that way: cortical columns contain diverse cell types, lateral connections enable short-circuit pathways, and feedback loops allow iterative refinement.

HelixLM mimics this structure inside a single recurrent block:

```
Input Tokens
    ↓
Embedding (d_model)
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Recurrent Block (n_loops × shared graph weights)            │
│  ├── HelixGraph: Randomly wired heterogeneous neural columns │
│  │   ├── LinearAttnNode   : O(n) causal linear attention   │
│  │   ├── FullAttnNode     : Causal softmax (periodic)        │
│  │   ├── SwiGLUNode       : Modern gated activation          │
│  │   ├── Mamba2Node       : Mamba-2 SSD (long-range)         │
│  │   ├── TitansMemoryNode : Neural long-term memory (opt.)   │
│  │   ├── GateNode         : Learned multi-input aggregation  │
│  │   └── Random wiring    : Vertical + lateral connections   │
│  ├── LTI Injection        : Recurrent stability (A < 1)      │
│  └── ACT Halting          : Dynamic per-token depth         │
└─────────────────────────────────────────────────────────────┘
    ↓
RMSNorm + LM Head (tied embeddings)
    ↓
Logits / Generation
```

### Key Design Choices

| Component | What it does | Why it matters |
|-----------|--------------|----------------|
| **Neural Columns & Heterogeneous Nodes** | Each column holds diverse node types (attention variants, SwiGLU, Mamba-2, gating, optional Titans memory) instead of identical transformer blocks. | Different information pathways for different computations, like biological cortical columns. |
| **Recurrent Depth (LTI + ACT)** | The same graph weights are looped `n_loops` times. LTI injection keeps the recurrent state stable. ACT halting dynamically allocates compute per token. | Iterative refinement without parameter growth; easy tokens use 1 loop, hard reasoning uses more. |
| **Hybrid Attention** | Linear attention (O(n) complexity) in most columns, with periodic full-attention columns for exact retrieval. | Long-context efficiency without losing precise copy/lookup capability. |
| **Mamba-2 SSD** | State Space Duality with chunked parallel scan. Auto-activates when `ssm_d_state >= 64`. | Handles very long-range dependencies efficiently on CPU, CUDA, or MPS. |
| **Titans Neural Memory (optional)** | Persistent surprise-gated memory via outer-product updates (first column only by default). | Test-time memory that can retain patterns across long documents without growing KV cache. |
| **Modern Primitives** | RoPE, SwiGLU, RMSNorm, and weight tying. | Proven SOTA components for convergence and generation quality. |

> **Note on column layout:** The `nodes_per_column` config field is reserved for future extensibility. In the current graph builder, every column automatically instantiates **Attention + SwiGLU + optional Mamba-2 SSD + optional Titans Memory + Gate**. `DenseNode` is defined in the codebase but not used in the default wiring recipe.

---

## Quick Start

### Installation

```bash
pip install torch transformers datasets accelerate
```

### Smoke Test (CPU, character-level)

```bash
cd helix_lm
python smoke_test.py
```

### Minimal Demo (CPU, HuggingFace tokenizer)

```bash
python quick_demo_cpu.py
```

---

## Comparison: Where does HelixLM fit?

### The Landscape

| Project | What They Do | Their Core Tech | Their Weakness |
|---------|-------------|-----------------|----------------|
| **OpenMythos** | Recurrent depth for LLMs | LTI stability + ACT halting | Standard transformer blocks; no heterogeneous graph |
| **Cerebros** | Biological brain-inspired wiring | Random DAG of dense layers | No real attention; no positional encoding; no recurrence |
| **Phi-2 (Microsoft)** | Small but capable | Standard transformer + quality data | Nothing novel architecturally; just good data curation |
| **Qwen2.5 (Alibaba)** | Multilingual small LLM | Standard transformer | No graph wiring; no recurrent depth; no SSM integration |
| **Mamba-2 (Tri Dao)** | Efficient SSM | State Space Duality | Not a complete LLM architecture; needs integration |
| **Kimi Linear** | Linear attention at scale | Hybrid linear + full attention | Not open source; no graph wiring |
| **HelixLM (Ours)** | **All of the above, integrated** | **Graph + Recurrent + Hybrid Attention + Mamba-2 + Titans** | **Small team; needs community compute for 1B+ scaling** |

### HelixLM vs. OpenMythos

OpenMythos pioneered recurrent depth with LTI stability and ACT halting. HelixLM takes that insight and makes it work inside a **heterogeneous** graph that mimics neural column and random topology connectivity found in biological brains.

| Capability | OpenMythos | HelixLM |
|-----------|-----------|---------|
| **Recurrent depth** | ✅ Same block looped | ✅ Same block looped |
| **LTI stability** | ✅ Spectral radius < 1 | ✅ Spectral radius < 1 |
| **ACT halting** | ✅ Dynamic per-token depth | ✅ Dynamic per-token depth |
| **Architecture inside loop** | Standard transformer block | **Heterogeneous random graph** |
| **Attention** | Standard full attention only | **Linear + full hybrid** |
| **Node types** | Single (transformer block) | **7+ active types (attention variants, FFN, SSM, gate, neural memory)** |
| **Positional encoding** | Standard learned | **RoPE** |
| **Activation** | GELU | **SwiGLU** |
| **Normalization** | LayerNorm | **RMSNorm** |
| **Open source** | ✅ Yes | ✅ Yes |
| **HF integration** | ❌ No | ✅ **Full PreTrainedModel** |

### HelixLM vs. Cerebros

[Cerebros](https://github.com/david-thrower/cerebros-core-algorithm-alpha/) showed that biological random hyperdense vertical and lateral topology of Dense layers could outperform rigid layer stacks. It generated text without attention on small data, but required elaborate integration and clashed with standard model-structure paradigms. HelixLM integrates that topological insight into a modern, HF-compatible LLM backbone.

---

## Why HelixLM in Practice?

### Hyperpersonalization

| Requirement | How HelixLM Delivers |
|-------------|---------------------|
| Small enough to train per-user | **0.4M–1.2B params** = trainable on single GPU in hours |
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

---

## Parameter Efficiency & Scaling

Recurrent loops reuse weights (depth without parameter growth). Graph wiring creates expressive pathways without wide uniform layers. Heterogeneous nodes specialize, so no capacity is wasted on uniform operations.

Because the graph is the bulk of the model, parameter counts are shown for **two common vocabularies**: a minimal character-level vocab (~100 tokens) and the GPT-2 BPE vocab (50,257 tokens). Both assume **tied embeddings/head**.

## Parameter Scaling

| Preset | d_model | ~Params (Char) | ~Params (GPT-2) | Status |
|--------|---------|---------------|-----------------|--------|
| **tiny** | 128 | 0.5 M | **13 M** | Smoke tests + demos |
| **small** | 256 | 3.2 M | 29 M | Experiments |
| **base** | 512 | 32 M | 83 M | Serious pretraining |
| **medium** | 768 | ~134 M | ~211 M | Production small |
| **large** | 1 024 | ~286 M | ~389 M | Competitive |
| **xl** | 1 536 | ~1.1 B | ~1.2 B | Frontier small |
| **xxl** | 2 048 | ~2.7 B | ~2.9 B | Frontier |


```python
# One-liner scaling
cfg = HelixConfig.medium(vocab_size=151936, tokenizer_name="qwen")
model = HelixForCausalLM(cfg)
```

---

## HuggingFace Integration

```python
from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer

cfg = HelixConfig.small(vocab_size=50257)
model = HelixForCausalLM(cfg)
tokenizer = HelixTokenizer("gpt2")

# Save/load in standard HF format
model.save_pretrained("./my-helix-model")
tokenizer._backend.save_pretrained("./my-helix-model")

# Load later with standard transformers Auto classes
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./my-helix-model")
tokenizer = AutoTokenizer.from_pretrained("./my-helix-model")
```

---

## Tokenizer Support

HelixLM supports multiple tokenizer backends, switchable via `tokenizer_name`:

| Name | Backend | Vocab Size | Best For |
|------|---------|-----------|----------|
| `char` | Character-level | ~100 | Debugging, tiny models |
| `gpt2` | GPT-2 BPE | 50,257 | English, general text |
| `qwen` | Qwen3 family | 151,936 | Multilingual, code |
| `custom` | Any HF tokenizer | Any | Your own tokenizer |

```python
# GPT-2
tokenizer = HelixTokenizer("gpt2")

# Qwen3 multilingual
tokenizer = HelixTokenizer("qwen")

# Custom HF model
tokenizer = HelixTokenizer("meta-llama/Llama-3.2-1B")

# Character (for debugging)
tokenizer = HelixTokenizer("char")
tokenizer.build_char_vocab("some text here")
```

---

## Forward Pass & Generation

```python
import torch

# Forward
input_ids = torch.tensor([tokenizer.encode("Hello world")])
logits = model(input_ids)

# Standard generation
generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=50)
text = tokenizer.decode(generated[0])

# Extended generation with stop-string detection
generated = model.generate_ext(
    input_ids,
    max_new_tokens=2000,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    stop_strings=["</s>", "\n\n"],
)
```

---

## Dataset & Training

### Rolling Chunking

The `HelixDataset` handles three scenarios automatically:

1. **Text >> seq_len**: Rolling window with configurable stride (default 50% overlap)
2. **Text == seq_len**: Exact fit
3. **Text < seq_len**: Padding with attention mask

```python
from helix_lm import HelixDataset, create_helix_dataloader

dataset = HelixDataset(
    texts=train_texts,
    tokenizer=tokenizer,
    seq_len=2048,
    stride=1024,  # 50% overlap
)

loader = create_helix_dataloader(
    texts=train_texts,
    tokenizer=tokenizer,
    seq_len=2048,
    batch_size=8,
    stride=1024,
)
```

### Natural Stop Detection

Each sample includes `is_natural_stop`: `True` if the chunk ends at a document boundary. This lets the model learn to distinguish natural endings from forced truncations.

### Document-Aware Chunking (No Boundary Crossings)

`DocumentAwareDataset` splits documents into non-overlapping chunks without crossing document boundaries. Only padding positions are masked in labels, giving 100% token utilization on real text.

### HuggingFace Streaming

```python
from helix_lm import HelixHFDataset

hf_dataset = HelixHFDataset(
    dataset_name="HuggingFaceTB/smoltalk",
    tokenizer=tokenizer,
    seq_len=2048,
    streaming=True,
)
```

---

## Configuration & Parameters

HelixLM scales from **0.4M to 1.3B+ parameters** through a single configuration class. Below are the key knobs and their practical effects.

### Core Dimensions

| Parameter | Effect | Practical Tip |
|-----------|--------|---------------|
| `d_model` | Width of the model; determines embedding and hidden dimension. | 128 for smoke tests; 768–1536 for production-quality small models. |
| `n_columns` | Number of neural columns (graph depth). | 2–4 for fast experiments; 6–7 for large models. |
| `n_loops` | How many times the graph is recurrently executed. | 1 for speed; 3–4 for iterative reasoning depth. |
| `n_heads` | Attention heads for attention nodes. | Must divide `d_model`. 4–8 for small models; 16–32 for large. |

### Attention & Memory

| Parameter | Effect | Practical Tip |
|-----------|--------|---------------|
| `attention_mode` | `linear` (O(n)), `full` (exact softmax), or `hybrid` (both). | `hybrid` is recommended for almost all use cases. |
| `hybrid_full_attention_interval` | How often to place a full-attention column in hybrid mode. | 3 or 4 (i.e., every 3rd/4th column is full attention). |
| `use_ssm` | Enables Mamba-2 SSD nodes in the graph. | Turn on for long-context or stateful tasks. |
| `ssm_d_state` | State dimension for Mamba-2. | Must be `>= 64` to activate the optimized Mamba-2 path; smaller values use a simplified SSM. |
| `use_titans_memory` | Adds a Titans neural memory node to the first column. | Useful for very long documents; still experimental. |
| `seq_len` | Training context window. | Can generate far beyond this at inference time thanks to recurrent state. |

### Training & Stability

| Parameter | Effect | Practical Tip |
|-----------|--------|---------------|
| `ffn_expansion` | Hidden dimension multiplier for SwiGLU nodes. | 2.0 is standard; 2.5–3.0 for XL/XXL presets. |
| `act_threshold` | Confidence threshold for Adaptive Computation Time halting. | 0.99 is conservative. Reduce to 0.95 if you want deeper per-token thinking. |
| `dropout` | Regularization rate. | 0.0–0.05 for small models; higher for large datasets. |
| `lr` / `weight_decay` | AdamW optimizer settings. | 3e-4 with 0.1 decay works well for small models. |
| `grad_clip` | Max gradient norm. | Keep at 1.0 to prevent spikes in graph architectures. |

---

## Generation Features

### Extended Generation Beyond max_seq_len

```python
# Standard generation (uses full model each step)
generated = model.generate(input_ids, max_new_tokens=200)

# Extended generation via HelixForCausalLM (uses only last token, with truncation)
generated = model.generate_ext(
    input_ids,
    max_new_tokens=2000,  # Far beyond training seq_len
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    stop_strings=["</s>", "\n\n"],
)
```

### Stop String Detection

```python
model.generate_ext(
    input_ids,
    max_new_tokens=100,
    stop_strings=["Question:", "User:", "\n\n\n"],
)
```

### Batched Generation

```python
prompts = ["The weather is", "In 1492,", "Once upon a time"]
batch = torch.stack([torch.tensor(tokenizer.encode(p)) for p in prompts])
results = model.generate(batch, max_new_tokens=30)
```

---

## Chat Template Support

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)

output = model.generate_ext(input_ids, max_new_tokens=50)
response = tokenizer.decode(output[0][input_ids.shape[-1]:])
```

---

## Mamba-2 SSD Integration

For models with `use_ssm=True` and `ssm_d_state >= 64`, HelixLM automatically uses **Mamba-2 SSD** instead of the simplified SSM:

- **State Space Duality**: Unifies attention and SSM perspectives
- **Parallel Scan**: Efficient associative scan in PyTorch (chunked for large sequences)
- **Selective Parameters**: Input-dependent B, C, and Δ (time step)

```python
cfg = HelixConfig.medium(
    use_ssm=True,
    ssm_d_state=128,   # Use Mamba-2 when >= 64
    ssm_d_conv=4,
    ssm_expand=2,
)
```

---

## VLM Extensibility (Future-Ready)

HelixLM includes architecture hooks for vision-language integration:

```python
cfg = HelixConfig.base(
    is_vlm=True,
    vision_encoder="siglip",  # or "clip", "custom"
    vision_hidden_size=768,
    vision_patch_size=16,
    vision_image_size=448,
    fusion_strategy="perceiver",  # "perceiver" or "simple_merge"
)

# Future:
#   HelixForImageTextToText
#   processor.apply_chat_template with images
```

---

## Training Best Practices

1. **Warmup + Cosine Decay**: Linear warmup followed by cosine decay to 0.1× base LR.
2. **Gradient Clipping**: Max norm 1.0 prevents spikes in graph architectures.
3. **Weight Decay**: 0.1 for small models, 0.01 for large.
4. **Mixed Precision**: Use `torch.cuda.amp` only on Ampere+ GPUs for XL/XXL; keep it off for tiny/small to avoid NaNs.
5. **Streaming**: `HelixDataset` supports corpora larger than RAM.
6. **No Cross-Document Boundaries**: Use `DocumentAwareDataset` to avoid pollution of gradient signal at document seams.
7. **Data Quality**: For small models, data quality > quantity. Curate aggressively.

---

## Project Structure

```
helix_lm/
  __init__.py           - Package exports
  config.py             - HelixConfig with 7 presets
  tokenizer.py          - Multi-backend tokenizer (char / GPT-2 / Qwen / custom)
  rope.py               - Rotary positional embeddings
  nodes.py              - Heterogeneous node types (7+ active, Dense defined)
  graph.py              - HelixGraph executor with random wiring
  recurrent.py          - Recurrent block (LTI + ACT)
  model.py              - HelixLMCore (non-HF)
  hf_model.py           - HelixForCausalLM (HF PreTrainedModel)
  dataset.py            - Rolling chunking, streaming, HF integration
  trainer.py            - Production training loop
  mamba2.py             - Mamba-2 SSD with parallel scan
  smoke_test.py         - Self-contained CPU test
examples/
  train_hf_full.py      - HF tokenizer + training example
  quick_demo_cpu.py     - Minimal CPU demo
```

---

## Existing Research We Build On

| Idea | Source / Inspiration | How we use it |
|------|---------------------|---------------|
| **Hybrid Attention** | Linear + full attention interleaving | 3:1 or 4:1 linear-to-full ratio; O(n) training with exact copy layers |
| **Recurrent Depth** | OpenMythos / Universal Transformers | Same graph weights looped `n_loops` times; fewer params, iterative refinement |
| **LTI Stability** | OpenMythos | Log-parameterized state decay to keep spectral radius < 1 |
| **Mamba-2 SSD** | Dao & Gu (2024) | Chunked associative scan for selective SSM; auto-enabled at `d_state >= 64` |
| **Biological Graph Wiring** | Cerebros / Random DAGs | Random vertical & lateral edges instead of strict feedforward stacks |
| **Titans Neural Memory** | Behrouz et al. (2025) | Optional first-column persistent memory with surprise-gated outer-product updates |

---

## License

This project is open-source under a modified Apache 2.0 license. Please see the [license.md](license.md) file for full terms and conditions.

```bibtex
@software{helixlm2026,
  title = {HelixLM: Recurrent Heterogeneous Graph Neural Language Model},
  year = {2026},
  note = {Open-source small language model architecture}
}
```
