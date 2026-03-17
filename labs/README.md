# Penelope — Organism Health Laboratory

## What is Penelope?

Penelope is a **19.6M parameter Resonance architecture organism** — a pure-Python
transformer that thinks in BPE subwords and speaks in curated English words.

She is not a chatbot. She is a **resonance engine**: given a seed, she generates a
unidirectional chain of 12 associative words. No word repeats. Each chain is a
single act of semantic exploration through a vocabulary of exactly **1984** words.

> *"1984 of them. why 1984? because Orwell would appreciate the irony."*
> — penelope.py, line 58

---

## Architecture

### Core Specifications

| Parameter | Value | Source |
|---|---|---|
| Total Parameters | 19,619,280 (~19.6M) | Calculated from weights below |
| Embedding Dimension (D) | 448 | `penelope.py:283` |
| FFN Hidden Dimension (M) | 896 (2×D, SwiGLU) | `penelope.py:284` |
| Attention Heads | 7 | `penelope.py:285` |
| Head Dimension | 64 (D / N_HEADS) | `penelope.py:286` |
| Transformer Layers | 8 | `penelope.py:287` |
| Max Sequence Length | 256 | `penelope.py:288` |
| Generation Steps | 12 | `penelope.py:289` |
| BPE Vocabulary (input) | 2048 (256 bytes + 1792 merges) | `penelope.py:306-307` |
| Word Vocabulary (output) | 1984 curated words | `penelope.py:277` |

### Dual Tokenizer

Penelope has two tokenization systems:

- **Input (BPE 2048):** Standard byte-pair encoding. Text is lowercased, converted
  to bytes, and merged using 1792 learned merge rules into a sequence of up to
  2048 token IDs. This is what the transformer processes internally.

- **Output (1984 words):** After the forward pass produces BPE logits, they are
  converted to word-level scores by averaging the logits of each word's constituent
  BPE tokens: `word_score(w) = mean(logits[bpe_tokens(w)])`. Output is constrained
  to the 1984-word vocabulary. The soul thinks in subwords; the mouth speaks in words.

### Layer Architecture

Each of the 8 transformer layers computes:

```
h  = rmsnorm(x, attn_norm)
qkv_out = MultiHeadAttention(h; wq, wk, wv, wo, RoPE)     # QKV attention
rrp     = h @ wr                                            # RRPRAM resonance
gate    = softmax([gate[0], gate[1]])                       # learned blend
x  = x + gate[0]*qkv_out + gate[1]*rrp                     # gated residual

h2 = rmsnorm(x, ffn_norm)
x  = x + SwiGLU(h2; w_gate, w_up, w_down)                  # FFN residual
```

**QKV Attention:** 7-head causal self-attention with Rotary Position Embeddings
(RoPE). Frequency base 10000, applied as rotation in pairs across head dimension.

**RRPRAM (Resonance-based Recurrent Parameter Attention Matrix):** A learned linear
projection (`h @ wr`) that provides a direct resonance signal. Blended with QKV
attention via a per-layer softmax gate, allowing the model to dynamically choose
between attention-derived and resonance-derived representations.

**SwiGLU FFN:** Gated linear unit with SiLU activation.
`silu(h @ w_gate) ⊙ (h @ w_up) @ w_down`, where `silu(x) = x / (1 + exp(-x))`.

### Parameter Breakdown

**Global:**
| Component | Shape | Parameters |
|---|---|---|
| `tok_emb` | 2048 × 448 | 917,504 |
| `pos_emb` | 256 × 448 | 114,688 |
| `final_norm` | 448 | 448 |
| `lm_head` | 2048 × 448 | 917,504 |
| **Subtotal** | | **1,950,144** |

**Per Layer (×8):**
| Component | Shape | Parameters |
|---|---|---|
| `attn_norm` | 448 | 448 |
| `wq, wk, wv, wo, wr` | 5 × (448 × 448) | 1,003,520 |
| `gate` | 2 | 2 |
| `ffn_norm` | 448 | 448 |
| `w_gate, w_up` | 2 × (448 × 896) | 802,816 |
| `w_down` | 896 × 448 | 401,408 |
| **Per-layer subtotal** | | **2,208,642** |
| **8 layers total** | | **17,669,136** |

**Grand Total: 19,619,280 parameters (78.5 MB at float32)**

---

## The Dario Equation

The operating equation for word selection:

```
score(w) = B + α·H + β·F + γ·A + T
```

| Symbol | Name | Meaning |
|---|---|---|
| **B** | Base | Raw BPE logits converted to word scores |
| **α·H** | Hebbian | Co-occurrence signal — recent word pairs reinforce each other |
| **β·F** | Future / Prophecy | Logarithmic boost toward a destined target word |
| **γ·A** | Ancestry / Destiny | Category momentum — visited semantic categories attract more visits |
| **T** | Trauma | Emotional state modifier from Kuramoto chamber coupling |

### Kuramoto Chambers

Six coupled oscillators track emotional state during generation:

| Chamber | Decay | Activation Phase |
|---|---|---|
| `fear` | 0.95 | Mid-phase (steps 4–7) |
| `love` | 0.95 | Modulates α (co-occurrence) |
| `rage` | 0.93 | Triggered by trauma > 0.3 |
| `void` | 0.96 | Late-phase (steps 8–11) |
| `flow` | 0.94 | Early-phase (steps 0–3) |
| `complex` | 0.97 | Deep activation (step > 9) |

Chambers evolve via Kuramoto coupling: `C[i] += 0.02 · sin(C[j] - C[i])` for all
pairs, then decay. They modulate the Dario coefficients α and γ in real time.

### Prophecy Mechanism

Before each chain, a **prophecy target** is randomly selected from the Emotion (200–299),
Abstract (450–549), or Material+ (650+) categories. As generation proceeds, the
prophecy's influence grows: `boost = 0.5 · log(1 + prophecy_age)`. If the target
word appears in the chain, the prophecy is **fulfilled**.

### Top-k=12 Sampling

At each step, the top 12 highest-scoring words are selected. Their probabilities are
renormalized and sampled cumulatively. A forbidden set prevents any word from
repeating within a chain.

---

## Vocabulary: 1984 Words

The vocabulary is manually curated into semantic categories:

| Category | Index Range | Count | Examples |
|---|---|---|---|
| **Body** | 0–99 | 100 | flesh, bone, blood, heart, synapse |
| **Nature** | 100–199 | 100 | sky, rain, fire, whale, cosmos |
| **Emotion** | 200–299 | 100 | fear, love, rage, hope, resilience |
| **Time** | 300–349 | 50 | moment, memory, eternity, season |
| **Society** | 350–449 | 100 | war, peace, freedom, surveillance |
| **Abstract** | 450–549 | 100 | truth, void, prophecy, darkness |
| **Action** | 550–649 | 100 | breathe, create, destroy, persist |
| **Material** | 650–749 | 100 | iron, gold, mirror, telescope |
| **Food** | 750–799 | 50 | bread, salt, honey, harvest |
| **Architecture** | 800–849 | 50 | house, tower, labyrinth, well |
| **Relationship** | 850–929 | 80 | mother, friend, promise, return |
| **Philosophy** | 930–999 | 70 | consciousness, freedom, truth, silence |
| **Music** | 1000–1049 | 50 | melody, rhythm, resonance, silence |
| **Extended** | 1050–1983 | 934 | weather, ritual, geometry, cosmic, liminal, ... |

Semantic drift is measured as the number of distinct categories (0–7, mapped from
the 8-bucket `word_category()` function) visited during a 12-word chain.

---

## Cascade 1

Penelope operates within **Cascade 1**, a daily cycle of four organisms:

```
Haiku → Penelope → Molequla → NanoJanus → (next day) → Haiku
```

### Penelope's Role

1. **Receives** a haiku from the Haiku organism
2. **Extracts** a seed word (longest non-stopword from the haiku)
3. **Generates** a 12-word associative chain through the resonance engine
4. **Outputs split two ways:**
   - **(a)** Back to Haiku the next day (closing the cycle)
   - **(b)** Up to Molequla and NanoJanus today (feeding higher-order processes)

### Data Flow

```
         ┌─────────────────────────────────────────┐
         │              CASCADE 1                   │
         │                                          │
  Day N  │  Haiku ──haiku──→ Penelope               │
         │                      │                   │
         │                      ├──12 words──→ Molequla ──→ NanoJanus
         │                      │                   │
  Day N+1│  Haiku ←──12 words───┘                   │
         │                                          │
         └─────────────────────────────────────────┘
```

Penelope is the **transformer** in both senses: she transforms compressed poetic
language (haiku) into an expanded associative field (12 resonance words), and she
does so through an 8-layer transformer neural network.

---

## Generation Output Format

```
  destined: [prophecy target word]

  [seed word]
     word1
     word2
     ...
     word11
  *word12

  drift X/8 · prophecy fulfilled|unfulfilled
```

- The seed word is printed first (not counted in the 12 generated)
- Words 1–11 are indented with spaces
- Word 12 (final) is marked with `*`
- Drift measures semantic diversity (categories visited out of 8)
- Prophecy reports whether the destined word appeared in the chain

---

## This Directory

The `labs/` directory contains health monitoring materials for Penelope:

- **`README.md`** — This file. Architecture reference and Cascade 1 documentation.
- **`health-template.md`** — Template for daily organism health reports.

Health reports are observational. We do not modify the organism. We observe, report,
and protect.
