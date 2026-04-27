
# HelixLM: Recurrent Heterogeneous Graph Neural Language Model

> **Why "Helix"?** A helix coils back on itself — just as our recurrent graph reuses its weights across depth iterations, refining understanding with each loop. Biological, elegant, memorable.

HelixLM is an optimized hybrid architecture for small-scale language modeling, designed for **hyperpersonalization** and **on-device AI**. It combines biological brain-inspired random graph wiring with modern SOTA primitives (hybrid attention, Mamba-2 SSD, RoPE, SwiGLU, RMSNorm) and full HuggingFace integration.

---

## Use Caases:

1. **Hyperpersonalization**: A small model (100M–4B parameters) that can be trained from cold start on a personalized personal corpus, then fine tuned on anuser's own data, making the model an expert in the one thing generic frontier models just don't know: **you**: your specialized domain knowledge, personalized context, work patterns, and communication style, email history, work notes, documents, briefs and dossiers, etc.

2. **On-device AI**: Efficient inference on CPU/GPU for desktops, laptops, tablets, and mobile — strong quality for its parameter count, with optional log-less operation for sensitive use cases.


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
│  Recurrent Block (n_loops × same graph weights)              │
│  ├── HelixGraph: Randomly wired heterogeneous neural columns │
│  │   ├── LinearAttnNode  : O(n) causal linear attention      │
│  │   ├── FullAttnNode    : Causal softmax (periodic)         │
│  │   ├── SwiGLUNode      : Modern gated activation           │
│  │   ├── Mamba2Node      : Mamba-2 SSD (optional)           │
│  │   ├── GateNode        : Learned multi-input aggregation    │
│  │   └── Random wiring   : Vertical + lateral connections    │
│  ├── LTI Injection       : Spectral-radius < 1 stability     │
│  └── ACT Halting         : Dynamic per-token depth           │
└─────────────────────────────────────────────────────────────┘
    ↓
RMSNorm + LM Head (tied embeddings)
    ↓
Logits / Generation
```

### Key Design Choices

| Component | What it does | Why it matters |
|-----------|--------------|----------------|
| **Neural Columns & Heterogeneous Nodes** | Each column holds diverse node types (attention variants, FFN, SSM, gate) instead of identical transformer blocks. | Different information pathways for different computations, like biological cortical columns. |
| **Recurrent Depth (LTI + ACT)** | The same graph weights are looped `n_loops` times. LTI injection keeps the spectral radius < 1 for stable recurrence. ACT halting dynamically allocates compute per token. | Iterative refinement without parameter growth; easy tokens use 1 loop, hard reasoning uses more. |
| **Hybrid Attention** | 80–90% linear attention (O(n) complexity) + periodic full attention layers for exact retrieval. | Long-context efficiency without losing precise copy/lookup capability. |
| **Mamba-2 SSD** | State Space Duality implementation with chunked parallel scan. Auto-activates when `ssm_d_state >= 64`. | Handles very long-range dependencies efficiently on CPU, CUDA, or MPS. |
| **Modern Primitives** | RoPE, SwiGLU, RMSNorm, and weight tying. | Proven SOTA components for convergence and generation quality. |

---

## Quick Start

### Installation

```bash
pip install torch transformers datasets accelerate
```



```bash
cd helix_lm
python smoke_test.py
```


### 'Minimal Scalable' Example  (Train a Language Model on CPU)

```sh
python quick_demo_cpu.py

```

---

## Comparison: Where does HelixLM fit in the ecosystem and landscape

---

## The Landscape

| Project | What They Do | Their Core Tech | Their Weakness |
|---------|-------------|-----------------|----------------|
| **OpenMythos** | Recurrent depth for LLMs | LTI stability + ACT halting | Standard transformer blocks; no heterogeneous graph |
| **Cerebros** | Biological brain-inspired wiring | Random DAG of dense layers | No real attention; no positional encoding; no recurrence |
| **Phi-2 (Microsoft)** | Small but capable | Standard transformer + quality data | Nothing novel architecturally; just good data curation |
| **Qwen2.5 (Alibaba)** | Multilingual small LLM | Standard transformer | No graph wiring; no recurrent depth; no SSM integration |
| **Mamba-2 (Tri Dao)** | Efficient SSM | State Space Duality | Not a complete LLM architecture; needs integration |
| **Kimi Linear** | Linear attention at scale | Hybrid linear + full attention | Not open source; no graph wiring |
| **HelixLM (Ours)** | **All of the above, integrated** | **Graph + Recurrent + Hybrid Attention + Mamba-2** | **Small team; needs compute for 1B+ scaling** |

---

## HelixLM vs. its community predecessor, OpenMythos

OpenMythos pioneered recurrent depth (looping over the same attention block) with LTI stability and ACT halting. We took that insight and **made it work inside a **heterogeneous** graph mimicking neural column and random topology connectiviety found in biological neurons.

| Capability | OpenMythos | HelixLM |
|-----------|-----------|---------|
| **Recurrent depth** | ✅ Same block looped | ✅ Same block looped |
| **LTI stability** | ✅ Spectral radius < 1 | ✅ Spectral radius < 1 |
| **ACT halting** | ✅ Dynamic per-token depth | ✅ Dynamic per-token depth |
| **Architecture inside loop** | Standard transformer block | **Heterogeneous random graph** |
| **Attention** | Standard full attention only | **Linear + full hybrid** |
| **Node types** | Single (transformer block) | **7 types (attention variants, FFN, SSM, gate)** |
| **Positional encoding** | Standard learned | **RoPE** |
| **Activation** | GELU | **SwiGLU** |
| **Normalization** | LayerNorm | **RMSNorm** |
| **Open source** | ✅ Yes | ✅ Yes |
| **HF integration** | ❌ No | ✅ **Full PreTrainedModel** |

## HelixLM vs. its inernal predecessor, CerebrosNotGPT 

Cerebros (the original inspiration https://github.com/david-thrower/cerebros-core-algorithm-alpha/) had the insight that biological neural connectivity, random hyperdense vertical and lateral topology of Dense layyers could outperform rigid layer stacks or sparse sequential connectivity and proved to be a promising component of an LLM. It was able to generate text without attention layers with a small model and small trainig corpus, boostable with attention hybridization, but required elaborate integration and clashed in model structure paradigm.

---

## HelixLM + Titans - Inspired Neural Memory Integration: Titans-style Neural Memory Node**

The implementation adds persistent, surprise-gated neural memory to HelixLM's
heterogeneous graph architecture, enabling:
1. **Cross-sequence memory**: Memory persists across forward passes/chunks
2. **Test-time learning**: Memory updates via delta rule during inference
3. **Guaranteed inclusion**: `titans_always_select=True` ensures at least one
   Titans node is always present in the graph


## Architecture: TitansMemoryNode

Based on Behrouz et al. (2025) "Titans: Learning to Memorize at Test Time" MAC variant:

```
Input x (B, T, D)
  → k_proj → φ(k)  (keys for memory update)
  → v_proj → v     (values for memory update)
  → q_proj → φ(q)  (queries for memory retrieval)

Memory M (B, feature_dim, D) — persistent across chunks:
  For each token t:
    v_pred = k_t @ M
    surprise = ||v_t - v_pred||
    delta = outer_product(k_t, v_t)
    M = M + η * surprise * delta
    M = layer_norm(M)

Output = x + proj(φ(q) @ M)
```

---

## The "Why HelixLM?" in Practice?

### For Hyperpersonalization (Use Case 1)

| Requirement | How HelixLM Delivers |
|-------------|---------------------|
| Small enough to train per-user | **0.5M–1B params** = trainable on single GPU in hours / full weights fine tuning foe the cost of higher - rank LoRA |
| Rich enough to capture user style | **Heterogeneous graph** routes different info types through optimal processing paths |
| Fast inference for tool calls | **Linear attention** gives O(n) complexity; **ACT** skips unnecessary depth |
| Queryable by frontier model | **HF integration** = standard API; **chat templates** = structured responses |

### For On-Device AI (Use Case 2)

| Requirement | How HelixLM Delivers |
|-------------|---------------------|
| Runs on CPU | **Linear attention** + **lightweight graph** = fast without GPU |
| Good quality for size | **Recurrent depth** reuses weights across loops = more capacity per parameter |
| Long context without forgetting | **LTI-stable recurrence** preserves state; **RoPE** generalizes length |
| Optional log-less operation | **No cloud dependency**; weights run locally |
| Responsive to long messages | **Rolling dataset** training = learns to handle arbitrary-length inputs |

---

## Parameter Efficiency Comparison

1. Recurrent loops reuse weights (depth without parameter growth)
2. Graph wiring creates expressive pathways without wide layers
3. Heterogeneous nodes specialize (no wasted capacity)

---

## Summary Matrix

| Feature | OpenMythos | Cerebros | Phi-2 | Qwen | Mamba-2 | HelixLM |
|---------|-----------|----------|-------|------|---------|---------|
| Recurrent depth | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Graph-based state | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Biological graph | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Heterogeneous nodes | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Hybrid attention | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| SSM backbone | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Stable recurrent dynamics | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| LTI stability | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| ACT halting | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| RoPE | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| SwiGLU | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| RMSNorm | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| HF Integration | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| Multi-tokenizer | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Rolling dataset | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Multimodal hooks | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |


**HelixLM is the only architecture that combines all of these.**

---


## Key Differentiators

| Feature | Typical Small LLM | HelixLM |
|---------|------------------|---------|
| Architecture | Stack of identical blocks | **Random heterogeneous graph** |
| Depth | Fixed N layers | **Dynamic (ACT halting)** + recurrent loops |
| Attention | All full or all linear | **Hybrid: linear + periodic full** |
| State model | None or pure Mamba | **LTI-stable recurrence** |
| Node types | Uniform | **6 heterogeneous types** |
| Scalability | Fixed recipes | **Continuous: 0.5M → 4B+ via config** |

---

### HuggingFace Integration

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

## Example Forward pass

```
import torch
input_ids = torch.tensor([tokenizer.encode("Hello world")])
logits = model(input_ids)
```

## Examle Generate

```
generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=50)
text = tokenizer.decode(generated[0])
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

Each sample includes `is_natural_stop`: True if the chunk ends at a document boundary (as opposed to an artificial slice). This lets the model learn to distinguish natural endings from forced truncations.

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

HelixLM scales from **0.5M to 4B+ parameters** through a single configuration class. Below are the key knobs and their practical effects.

### Core Dimensions

| Parameter | Effect | Practical Tip |
|-----------|--------|---------------|
| `d_model` | Width of the model; determines embedding and hidden dimension. | 128 for smoke tests; 768–1536 for production-quality small models. |
| `n_columns` | Number of neural columns (graph depth). | 2–4 for fast experiments; 6–7 for large models. |
| `nodes_per_column` | Nodes placed in each column. | More nodes = denser lateral connectivity and richer pathways. |
| `n_loops` | How many times the graph is recurrently executed. | 1 for speed; 3–4 for iterative reasoning depth. |
| `n_heads` | Attention heads for attention nodes. | Must divide `d_model`. 4–8 for small models; 16–32 for large. |

### Attention & Memory

| Parameter | Effect | Practical Tip |
|-----------|--------|---------------|
| `attention_mode` | `linear` (O(n)), `full` (exact softmax), or `hybrid` (both). | `hybrid` is recommended for almost all use cases. |
| `hybrid_full_attention_interval` | How often to place a full-attention column in hybrid mode. | 3 or 4 (i.e., every 3rd/4th column is full attention). |
| `use_ssm` | Enables Mamba-2 SSD nodes in the graph. | Turn on for long-context or stateful tasks. |
| `ssm_d_state` | State dimension for Mamba-2. | Must be `>= 64` to activate the optimized Mamba-2 path; smaller values use a simplified SSM. |
| `seq_len` | Training context window. | Can generate far beyond this at inference time thanks to recurrent state. |

### Training & Stability

| Parameter | Effect | Practical Tip |
|-----------|--------|---------------|
| `ffn_expansion` | Hidden dimension multiplier for FFN/SwiGLU nodes. | 2.0 is standard; 2.5–3.0 for XL/XXL presets. |
| `act_threshold` | Confidence threshold for Adaptive Computation Time halting. | 0.99 is conservative. Reduce to 0.95 if you want deeper per-token thinking. |
| `dropout` | Regularization rate. | 0.0–0.05 for small models; higher for large datasets. |
| `lr` / `weight_decay` | AdamW optimizer settings. | 3e-4 with 0.1 decay works well for small models. |
| `grad_clip` | Max gradient norm. | Keep at 1.0 to prevent spikes in graph architectures. |

---

## Scaling Guide / Preset recipes

| Preset | d_model | Columns | Nodes | Heads | Loops | SSM | ~Params | Seq Len | Use Case |
|--------|---------|---------|-------|-------|-------|-----|---------|---------|----------|
| `tiny` | 128 | 2 | (2,2) | 4 | 1 | No | 0.5M | 256 | Smoke test |
| `small` | 256 | 3 | (2,3,2) | 4 | 2 | No | 5M | 512 | Experiments |
| `base` | 512 | 4 | (3,4,4,3) | 8 | 2 | Yes | 25M | 1024 | Pretraining |
| `medium` | 768 | 5 | (3,4,4,4,3) | 12 | 3 | Yes | 100M | 2048 | Production small |
| `large` | 1024 | 6 | (4,5,5,5,5,4) | 16 | 3 | Yes | 300M | 4096 | Competitive |
| `xl` | 1536 | 6 | (5,6,6,6,6,5) | 24 | 4 | Yes | 1B | 8192 | Frontier small |
| `xxl` | 2048 | 7 | (5,6,6,6,6,6,5) | 32 | 4 | Yes | 4B | 16384 | Near-frontier |

```python
# One-liner scaling
cfg = HelixConfig.medium(vocab_size=151936, tokenizer_name="qwen")
```

## Generation Features

### Extended Generation Beyond max_seq_len

```python
# Standard generation (uses full model each step)
generated = model.generate(input_ids, max_new_tokens=200)

# Extended generation via HelixForCausalLM (uses only last token, with cache)
generated = model.generate_ext(
    input_ids,
    max_new_tokens=2000,  # Far beyond training seq_len
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    stop_strings=["</s>", "\n\n"],  # Stop on strings, not just tokens
)
```

### Stop String Detection

```python
# Stop on specific strings, not just EOS tokens
model.generate_ext(
    input_ids,
    max_new_tokens=100,
    stop_strings=["Question:", "User:", "\n\n\n"],
)
```

### Batched Generation

```python
# Generate multiple sequences in parallel
prompts = ["The weather is", "In 1492,", "Once upon a time"]
batch = torch.stack([torch.tensor(tokenizer.encode(p)) for p in prompts])
results = model.generate(batch, max_new_tokens=30)
```

## Chat Template Support

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

# With Qwen-style tokenizer
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)

# Generate response
output = model.generate_ext(input_ids, max_new_tokens=50)
response = tokenizer.decode(output[0][input_ids.shape[-1]:])
```

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

# Will expose:
#   HelixForImageTextToText  (future)
#   processor.apply_chat_template with images
```

## Training Best Practices

1. **Warmup + Cosine Decay**: Linear warmup (default 100 steps) followed by cosine decay to 0.1× base LR
2. **Gradient Clipping**: Max norm 1.0 prevents spikes in graph architectures
3. **Weight Decay**: 0.1 for small models, 0.01 for large
4. **Mixed Precision**: Automatic `torch.cuda.amp` when CUDA available
5. **Streaming**: `HelixDataset` supports corpora larger than RAM
6. **Data Quality**: For small models, data quality > quantity. Curate aggressively.

## Project Structure

```
helix_lm/
  __init__.py           - Package exports
  config.py             - HelixConfig with 7 presets
  tokenizer.py          - Multi-backend tokenizer (char / GPT-2 / Qwen / custom)
  rope.py               - Rotary positional embeddings
  nodes.py              - 7 heterogeneous node types
  graph.py              - HelixGraph executor with random wiring
  recurrent.py          - Recurrent block (LTI + ACT)
  model.py              - HelixLMCore (non-HF)
  hf_model.py           - HelixForCausalLM (HF PreTrainedModel)
  dataset.py            - Rolling chunking, streaming, HF datasets
  trainer.py            - Production training loop
  mamba2.py             - Mamba-2 SSD with parallel scan
  smoke_test.py         - Self-contained CPU test
examples/
  train_hf_full.py      - HF tokenizer + training example
```

---

## Acknowledgements

This project would not be possible without the support, insight, and contributions of many people and communities.

- **Family:** My Jennifer and my step-kids. My son **Aidyn** (also a cofounder), daughter **Jenna**, and the rest of my family for their patience and encouragement.
- **Cofounders:** **Aidyn Lopez**, **Moises Perez**, **Alexander Kolpakov**, and **Jeffly Archellus** — thank you for building alongside me.
- **Colleagues:** My colleagues who I work with every day — your collaboration pushes this forward.
- **Open Source Communities:** The TensorFlow, Keras, PyTorch, MLflow, Kubeflow, Kale, Optuna, Keras Tuner, and Ray open source communities and contributors and many others have been instrumental to this. Your tools are the foundation we build on.
- **AI Research Community** Kye Gomez who published OpenMythos (https://github.com/kyegomez/OpenMythos and his team), the teams at OpenAI, Qwen, Kimi, Google, MIT AI Lab, and many others who crafted many of the foundational bricks used to build this.
- **Infrastructure Partners, past and present:** AWS, Innovative Solutions, Google Cloud Platform, Arrikto, Canonical, and Paperspace and their support staff — for past and present operational contributions.
- **Investors:** **AI Forge; Jennifer**, our pre-seed round investors, for backing this vision early.

---

```
helix_lm/
  __init__.py           - Package exports
  config.py             - HelixConfig (PretrainedConfig) with 7 presets
  tokenizer.py          - Multi-backend tokenizer (char/gpt2/qwen/custom)
  rope.py               - Rotary positional embeddings
  nodes.py              - 7 heterogeneous node types
  graph.py              - HelixGraph executor with random wiring
  recurrent.py          - Recurrent block (LTI + ACT)
  model.py              - HelixLMCore (non-HF)
  hf_model.py           - HelixForCausalLM (HF PreTrainedModel)
  dataset.py            - Rolling chunking, streaming, HF integration
  trainer.py            - Production training loop
  mamba2.py             - Mamba-2 SSD with parallel scan
  smoke_test.py         - Self-contained CPU test
```

---

## Existing Research We Build On Top Of

### Hybrid Attention

Pure linear attention cannot do exact retrieval. Interleaving linear layers with periodic full attention layers gives O(n) efficiency + exact copying. Default ratio: 3:1 or 4:1 linear-to-full.

### Recurrent Depth (OpenMythos / Universal Transformers)

Reusing the same block weights across depth iterations:
- Reduces parameter count (same weights do more work)
- Improves generalization (iterative refinement)
- Enables dynamic depth via ACT halting

### LTI Stability

The `LTIInjection` module constrains the spectral radius to < 1 using log-parameterized A, preventing exploding/vanishing gradients in deep recurrence.

### Mamba-2 SSD

Mamba-2 unifies attention and SSM through structured state space duality. Our implementation uses PyTorch associative scan (chunked for large sequences) for efficient training.

### Biological Graph Wiring (Cerebros)

Random vertical and lateral connections create multiple information pathways of varying depth. Lateral connections preserve fine-grained details that would be lost in a strict feedforward stack.

---

## License

This project is open-source under a modified Apache 2.0 license. Please see the [license.md](license.md) file for full terms and conditions.


```bibtex
@software{helixlm2025,
  title = {HelixLM: Recurrent Heterogeneous Graph Neural Language Model},
  year = {2025},
  note = {Open-source small language model architecture}
}
```
