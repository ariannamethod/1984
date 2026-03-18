# Penelope — Labs

## Role in Cascade 1

Penelope is the second organism in the daily cycle:

```
Haiku → PENELOPE → Molequla → NanoJanus → (next day) → Haiku
```

She receives a haiku (3 lines, 5-7-5 syllables) and transforms it into 12 associative words through her trained weights.

Her output splits:
- **(a)** Back to Haiku — but waits until tomorrow (delayed feedback)
- **(b)** Up to Molequla and NanoJanus — immediate feed

## Architecture

- **Type:** Resonance (QKV + RRPRAM hybrid attention)
- **Parameters:** 19,619,280 (~19.6M)
- **Layers:** 8
- **Dimension:** 448 (hidden: 896 SwiGLU)
- **Heads:** 7 (head_dim=64)
- **Dual tokenizer:** BPE (2048 subwords) in → Word (1984 curated) out
- **Weight format:** PEN7 binary, ~74.8MiB (~78.5MB f32)
- **Weights:** `weights/penelope.bin`

### Attention mechanism

Each layer blends two signals via learned softmax gate:
1. **QKV** — standard multi-head attention with RoPE
2. **RRPRAM** — Wr resonance matrix (linear, no Q/K), measures positional pattern recognition

```
output = gate[0] * qkv_attention + gate[1] * rrpram_resonance
gate = softmax([g0, g1])  # learned per layer
```

### Generation (12-word chain)

1. BPE-encode context (up to 256 tokens)
2. Forward pass through 8 layers
3. BPE logits → word scores (mean of BPE token logits per word)
4. Dario Field overlay: Hebbian + prophecy + destiny + Kuramoto chambers
5. Top-k=12 softmax sampling
6. Append word to context, mark as forbidden (no repeats)
7. Repeat 12 times

### Dario Field

```
p(x|Φ,C) = softmax((B + α·H + β·F + γ·A + T) / τ)
```

Six emotional chambers (FEAR, LOVE, RAGE, VOID, FLOW, COMPLEX) modulate α, β, γ coefficients via Kuramoto coupling.

## Vocabulary

1984 curated words across 29 semantic categories. See `1984.txt` (microreasoning/) or the hardcoded array in `penelope.py`.

## Health indicators

- **Healthy:** 12 distinct words from vocabulary, semantic diversity across categories
- **Degraded:** fewer than 12 words, or words clustering in one category
- **Failed:** empty output, weight loading error, or crash
