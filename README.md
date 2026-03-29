# Practical-Decoder

Practical-Decoder is a decoder-only Transformer playground for implementing and comparing core LLM building blocks in a small, readable codebase. The repo is organized around swappable attention modules, positional encoding schemes, feed-forward variants, and inference-time efficiency features.

This README is structured as an implementation log. The high-level sections are in place so additional notes, equations, benchmarks, and design decisions can be filled in later.

## What This Repo Covers

- Decoder-only GPT-style language model
- Swappable attention backends: `mha`, `gqa`, `mla`
- Inference optimizations: KV cache and cached causal masks
- Positional encoding options: learned absolute embeddings and RoPE
- Feed-forward variants: dense FFN and MoE
- Activation variants: `GELU` and `SwiGLU`
- Training, generation, and unit tests for core components

## Current Implementation Map

| Area | Implemented in repo | Main config switches |
| --- | --- | --- |
| Attention | Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Latent Attention (MLA) | `model.attn_type`, `model.n_heads`, `model.n_kv_heads`, `model.dimckv`, `model.dimcq`, `model.head_dim_r` |
| Efficient decoding | KV cache, cached causal attention masks | generation path + `src/attention/cache.py` |
| Positional encoding | Learned positional embeddings, RoPE | `model.rope` |
| Feed-forward | Dense MLP, MoE | `model.mlp_type`, `model.moe.*` |
| Activations | `GELU`, `SwiGLU` | `model.ffn`, `model.moe.ffn` |
| Normalization | `LayerNorm`, `RMSNorm` | `model.norm_type` |

## Repository Structure

```text
src/
  attention/
    mha.py
    gqa.py
    mla.py
    cache.py
  models/
    gpt.py
    blocks.py
    rope.py
    mlp.py
  moe/
    moe_block.py
  train/
    train.py
  utils/
    generate.py
    bench_compare.py
tests/
  test_mha.py
  test_gqa.py
  test_kv_cache.py
  test_rope.py
  test_rope_cache.py
  test_moe.py
  test_gpt_losses.py
```

## Getting Started

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python -m src.train.train
```

The default training entrypoint uses Hydra config from `src/config/mac_tinyshakespeare.yaml`.

### Generate

```bash
python -m src.utils.generate \
  --checkpoint checkpoints/final.pt \
  --data-path data/raw/shakespeare.txt \
  --prompt "To be, or not to be" \
  --device mps
```

To enable incremental decoding with cache:

```bash
python -m src.utils.generate \
  --checkpoint checkpoints/final.pt \
  --data-path data/raw/shakespeare.txt \
  --prompt "To be, or not to be" \
  --device mps \
  --enable-cache
```

### Run Tests

```bash
pytest
```

Current test suite covers 19 unit tests across MHA, GQA, RoPE, KV cache, MoE, and GPT loss handling.

## Configuration Surface

Most experiments are controlled through Hydra config under `src/config`.

Common knobs:

- `model.attn_type`: `mha`, `gqa`, `mla`
- `model.n_heads`: number of attention heads
- `model.n_kv_heads`: grouped KV heads for GQA
- `model.rope`: `0` for learned positional embeddings, `1` for RoPE
- `model.mlp_type`: `dense` or `moe`
- `model.ffn`: dense block activation choice
- `model.moe.num_experts`
- `model.moe.shared_num_experts`
- `model.moe.top_k`
- `model.moe.ffn`
- `model.norm_type`: `layernorm` or `rmsnorm`

## Implementation Walkthrough

The rest of this README is organized around five implementation axes that evolve the decoder stack from a simple baseline to more modern LLM components.

### 1. Attention Mechanisms

This section compares the attention backends implemented behind the shared Transformer block interface in `src/models/blocks.py`.

#### 1.1 Multi-Head Attention (MHA)

- Implementation: `src/attention/mha.py`
- Role in the repo: baseline attention module
- Notes to add:
  - Why this serves as the reference implementation
  - Tensor shapes for `q`, `k`, `v`
  - How the vectorized implementation differs from the naive version
  - Observed training or inference behavior

#### 1.2 Grouped-Query Attention (GQA)

- Implementation: `src/attention/gqa.py`
- Main idea: share KV heads across multiple query heads
- Notes to add:
  - Why reducing KV heads helps decoding efficiency
  - Relationship between `n_heads`, `n_kv_heads`, and `group_factor`
  - Tradeoff between memory use and quality
  - What matched or differed from MHA in experiments

#### 1.3 Multi-Latent Attention (MLA)

- Implementation: `src/attention/mla.py`
- Main idea: compress parts of the attention state into latent projections
- Notes to add:
  - High-level intuition for latent compression
  - Meaning of `dimckv`, `dimcq`, and `head_dim_r`
  - How rotary dimensions are handled in this variant
  - What you found harder to implement or reason about

#### 1.4 Attention Summary

Suggested content for later:

- A comparison table for parameter count, KV cache size, and generation speed
- When you would choose MHA vs GQA vs MLA
- Failure cases, surprising behaviors, or unresolved questions

### 2. Efficient Attention

This section focuses on inference-time optimizations rather than architecture changes.

#### 2.1 KV Cache

- Implementation: `src/attention/cache.py`
- Generation path: `src/utils/generate.py`
- Notes to add:
  - Why autoregressive decoding benefits from KV caching
  - Cache layout for MHA/GQA vs MLA
  - How `start_pos` is threaded through the model
  - Measured speedup and memory overhead

#### 2.2 Cached Causal Attention Mask

- Implemented inside each attention module
- Notes to add:
  - Why rebuilding a lower-triangular mask each step is wasteful
  - How the mask is registered and sliced
  - Whether this mattered noticeably in profiling

#### 2.3 Decoding Path

Suggested content for later:

- Prefill vs token-by-token decode
- Benchmark setup
- Throughput and latency comparison with and without cache

### 3. Positional Encoding

This repo supports both learned absolute positional embeddings and RoPE.

#### 3.1 Learned Positional Embeddings

- Used when `model.rope = 0`
- Integrated in `src/models/gpt.py`
- Notes to add:
  - Baseline implementation details
  - Limitations for extrapolation beyond training length
  - Any behavioral differences you observed

#### 3.2 Rotary Positional Embedding (RoPE)

- Implementation: `src/models/rope.py`
- Integration point: `src/models/blocks.py`
- Notes to add:
  - Why RoPE is appealing for decoder-only models
  - How frequencies are precomputed and reused
  - How RoPE interacts with MHA, GQA, and MLA
  - Any edge cases around head dimension parity

#### 3.3 Positional Encoding Summary

Suggested content for later:

- Comparison between learned embeddings and RoPE in this repo
- Practical reasons to switch to RoPE
- Whether RoPE changed training stability or generation quality

### 4. FFN to MoE

This section documents the feed-forward path evolution from a standard dense MLP to a routed sparse expert block.

#### 4.1 Dense FFN

- Defined through the Transformer block in `src/models/blocks.py`
- Notes to add:
  - Baseline dense FFN structure
  - Hidden dimension choice
  - Activation variants supported in dense mode

#### 4.2 Mixture-of-Experts (MoE)

- Implementation: `src/moe/moe_block.py`
- Notes to add:
  - Router design and top-k dispatch
  - Shared experts vs routed experts
  - Auxiliary load-balancing loss
  - Expert usage patterns observed during training

#### 4.3 Dense vs MoE Summary

Suggested content for later:

- Parameter-efficiency vs compute-efficiency discussion
- When MoE became worthwhile in this codebase
- Any routing collapse or imbalance issues encountered

### 5. Activation Functions

This repo currently exposes activation choices in both dense FFN and MoE experts.

#### 5.1 GELU

- Used in dense and MoE expert blocks
- Notes to add:
  - Why GELU is a reasonable baseline
  - Where it appears in the code
  - How it behaved relative to SwiGLU

#### 5.2 SwiGLU

- Implementation helper: `src/models/mlp.py`
- Notes to add:
  - Motivation for adding SwiGLU
  - How the gating path is structured
  - Whether it improved quality, stability, or efficiency

#### 5.3 Activation Summary

Suggested content for later:

- Side-by-side comparison of GELU vs SwiGLU
- Whether the activation change mattered more in dense FFN or MoE experts

## Testing and Validation

The current tests focus on implementation correctness for the core architectural pieces.

- Attention correctness against reference behavior:
  - `tests/test_mha.py`
  - `tests/test_gqa.py`
- Cache correctness:
  - `tests/test_kv_cache.py`
  - `tests/test_rope_cache.py`
- Positional encoding:
  - `tests/test_rope.py`
- MoE and loss handling:
  - `tests/test_moe.py`
  - `tests/test_gpt_losses.py`

Suggested additions for later:

- Benchmark results for each attention backend
- Training curves or validation perplexity snapshots
- Expert routing histograms
- Memory and latency plots with and without cache

## Results and Notes

Use this section later for:

- experiment table
- model configurations tried
- generation samples
- training logs worth keeping
- implementation lessons learned

## Next Steps

Possible future extensions:

- add benchmark tables to compare MHA, GQA, and MLA
- document training results per config
- expand test coverage for MLA cache parity
- add references to papers or external implementation notes
