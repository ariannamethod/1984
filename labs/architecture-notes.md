# Penelope — Architecture Notes

Technical reference for Penelope v7 Resonance engine. All function names, line references, and formulas are drawn from `penelope.py` and `microreasoning/microreasoning.py`. Cross-referenced against CASCADE01.md (Cascade 1 spec).

---

## 1. BPE Tokenizer

Penelope uses a byte-pair encoding tokenizer for input. The BPE operates at the subword level, converting raw text into token IDs in the range `[0, 2047]`.

### Constants

| Constant | Value | Source |
|---|---|---|
| `BPE_VOCAB` | 2048 | `penelope.py:306` |
| `BPE_MERGES` | 1792 | `penelope.py:307` |
| Base byte tokens | 256 (0–255) | ASCII byte range |

The vocabulary consists of 256 raw byte tokens (IDs 0–255) plus 1792 learned merge tokens (IDs 256–2047), totaling 2048 subword tokens.

### BPE Merge Table

`BPE_TABLE` (`penelope.py:309–534`, `microreasoning.py:475–718`) is a list of 1792 `(left, right)` merge pairs. Both files contain identical merge tables. Each merge pair defines: when token `left` is immediately followed by token `right`, they are merged into a new token with ID `256 + merge_index`.

### Encoding — `bpe_encode(text)`

**penelope.py:537–559**, **microreasoning.py:722–737**

1. Convert input text to lowercase (penelope.py explicitly maps A–Z → a–z byte by byte; microreasoning.py uses `text.lower()`).
2. Convert each character to its byte value, producing an initial sequence of byte IDs.
3. Iterate through all 1792 merges in order (index 0 to 1791):
   - For each merge `(left, right)` at index `m_idx`, scan the sequence for adjacent pairs matching `(left, right)`.
   - Replace each matched pair with the new token ID `256 + m_idx`.
4. Return the final sequence of BPE token IDs.

The merge order matters — earlier merges are applied first, matching the training-time merge priority.

### Decoding — `bpe_decode_token(tok_id)`

**penelope.py:562–572**, **microreasoning.py:832–839**

Recursively expands a BPE token back to its string:
- If `tok_id < 256`: return `chr(tok_id)` (raw byte).
- If `tok_id >= 256`: look up `BPE_TABLE[tok_id - 256]` → `(left, right)`, return `bpe_decode_token(left) + bpe_decode_token(right)`.

**penelope.py** also has a precomputed string table `bpe_strs[BPE_VOCAB]` built by `init_bpe_decode()` (line 589–595), which iteratively constructs all token strings to avoid recursion at runtime.

### Vocab BPE Precomputation

**penelope.py:576–583** — `init_vocab_bpe()` precomputes the BPE encoding of each word in the 1984-word vocabulary, stored in `vocab_bpe[word_index] → [bpe_ids]` and `vocab_bpe_len[word_index] → int`.

**microreasoning.py:741** — `VOCAB_BPE = [bpe_encode(w) for w in VOCAB]` — same precomputation, done at import time as a list comprehension.

---

## 2. Word Vocabulary

### 1984 Curated Words

Penelope's output vocabulary consists of exactly **1984** curated English words (`NWORDS = len(VOCAB)`, `penelope.py:277`).

**penelope.py** hardcodes the full vocabulary as a Python list (`VOCAB`, lines 46–275).

**microreasoning.py** loads the vocabulary from `microreasoning/1984.txt` (one word per line, 1984 lines) at import time (lines 63–65).

Both contain the same 1984 words in the same order.

### 29 Semantic Categories

The words are organized into 29 named categories by index range:

| # | Category | Index Range | Count |
|---|---|---|---|
| 1 | BODY | 0–99 | 100 |
| 2 | NATURE | 100–199 | 100 |
| 3 | EMOTION | 200–299 | 100 |
| 4 | TIME | 300–349 | 50 |
| 5 | SOCIETY | 350–449 | 100 |
| 6 | ABSTRACT | 450–549 | 100 |
| 7 | ACTION | 550–649 | 100 |
| 8 | MATERIAL | 650–749 | 100 |
| 9 | FOOD | 750–799 | 50 |
| 10 | ARCHITECTURE | 800–849 | 50 |
| 11 | RELATIONSHIP | 850–929 | 80 |
| 12 | PHILOSOPHY | 930–999 | 70 |
| 13 | MUSIC | 1000–1049 | 50 |
| 14 | WEATHER | 1050–1099 | 50 |
| 15 | RITUAL | 1100–1149 | 50 |
| 16 | LABOR | 1150–1199 | 50 |
| 17 | GEOMETRY | 1200–1249 | 50 |
| 18 | ANIMAL | 1250–1299 | 50 |
| 19 | COLOR | 1300–1349 | 50 |
| 20 | TRANSPORT | 1350–1399 | 50 |
| 21 | DOMESTIC | 1400–1449 | 50 |
| 22 | COMMUNICATION | 1450–1499 | 50 |
| 23 | MEDICAL | 1500–1549 | 50 |
| 24 | COSMIC | 1550–1599 | 50 |
| 25 | BUREAUCRACY | 1600–1649 | 50 |
| 26 | MYTHIC | 1650–1699 | 50 |
| 27 | TEXTUAL | 1700–1749 | 50 |
| 28 | PSYCHOLOGICAL | 1750–1799 | 50 |
| 29 | FINAL | 1800–1983 | 184 |

### Macro-Categories (Dario Field)

The `word_category(idx)` function (`penelope.py:1210–1218`, `microreasoning.py:1095–1104`) collapses the 29 categories into **8 macro-categories** for Dario field computations:

| Macro | Index Range | Covers |
|---|---|---|
| 0 | 0–99 | BODY |
| 1 | 100–199 | NATURE |
| 2 | 200–299 | EMOTION |
| 3 | 300–349 | TIME |
| 4 | 350–449 | SOCIETY |
| 5 | 450–549 | ABSTRACT |
| 6 | 550–649 | ACTION |
| 7 | 650+ | MATERIAL through FINAL (22 categories) |

### Vocabulary Lookup

`VOCAB_IDX` (`penelope.py:292–295`, `microreasoning.py:70–73`) maps word strings to their first occurrence index. Duplicate words (some exist across categories) map to their earliest index only.

`STOP` words (`penelope.py:297`, `microreasoning.py:76`) — a set of ~80 common English function words excluded from input tokenization and extended vocabulary.

### Extended Vocabulary

Beyond the 1984 hardcoded words, both implementations build an extended vocabulary by scanning BPE tokens that decode to valid whole words:

**penelope.py:602–636** — `init_ext_vocab()` builds `ext_vocab` list of `(word_str, bpe_ids, from_hardcoded)` tuples. Maximum 4096 entries (`MAX_EXT_VOCAB`). BPE-derived words must be ≥3 alphabetic characters, not in STOP words, not suffix fragments (`SUFFIX_FRAGMENTS`, line 299).

**microreasoning.py:842–874** — `init_ext_vocab()` returns an equivalent list. Uses a local `_SUFFIXES` set (line 853) with the same fragment filtering.

Extended vocabulary is used during generation when weights are loaded (`has_weights=True`), giving the organism more words to choose from.

### Three-Stage Input Tokenizer

For converting arbitrary text to word-level IDs (used for Dario field context):

**penelope.py** — `tokenize_vocab(text)` (line 1060–1077)
**microreasoning.py** — `tokenize_text(text)` (line 808–825)

1. **Exact match**: word exists in `VOCAB_IDX` → use its index.
2. **Stemming** (`try_stem`, `penelope.py:1017–1036`, `microreasoning.py:765–784`): strip suffixes from `SUFFIXES` list, try the stem, stem+"e", or doubled-consonant removal.
3. **Greedy decomposition** (`greedy_vocab_match`, `penelope.py:1039–1057`, `microreasoning.py:787–805`): longest-first vocab match within the word, minimum 3-character matches.

---

## 3. RRPRAM — Resonance Recognition Pattern Retrieval and Associative Memory

Each of the 8 transformer layers has an RRPRAM component alongside standard QKV attention.

### Wr Matrix

Each layer has a learned resonance matrix `wr` of shape `[DIM, DIM]` (448×448 = 200,704 parameters per layer):

```
penelope.py:725    self.wr = [randn() * scale_d for _ in range(DIM * DIM)]
microreasoning.py:214    self.wr = [randn() * scale_d for _ in range(DIM * DIM)]
```

### RRPRAM Computation

The resonance output is a simple linear transformation — no Q/K decomposition, no softmax attention:

```
rrp[t] = wr @ h[t]     (for each position t)
```

**penelope.py:847** — `rrp = [matmul_mv(lw.wr, h[t], DIM, DIM) for t in range(S)]`
**microreasoning.py:371–373** — same operation on flat arrays.

This means RRPRAM operates independently per position — it is a position-local pattern recognizer, contrasting with QKV attention which mixes information across positions.

### Gate Blending

Each layer has a learned 2-element gate vector `gate = [g0, g1]` (initialized to `[0.0, 0.0]`):

```
penelope.py:726    self.gate = [0.0, 0.0]
```

The gate is converted to weights via softmax:

```python
# penelope.py:849–856
g0, g1 = lw.gate[0], lw.gate[1]
gmax = max(g0, g1)
e0 = math.exp(g0 - gmax)
e1 = math.exp(g1 - gmax)
gsum = e0 + e1
w0 = e0 / gsum       # weight for QKV attention
w1 = e1 / gsum       # weight for RRPRAM resonance
```

**microreasoning.py:376** uses a `softmax2(a, b)` helper (line 152–158) for the same computation.

At initialization, `gate = [0.0, 0.0]` → `softmax(0, 0) = (0.5, 0.5)`, so QKV and RRPRAM contribute equally. Training adjusts the gate per layer.

The gated residual update:

```
x[t] = x[t] + w0 * qkv_out[t] + w1 * rrp[t]
```

**penelope.py:858–861**, **microreasoning.py:377–378**.

---

## 4. Dario Field — Overlay Formula

The Dario Equation (from CASCADE01.md):

```
p(x|Φ,C) = softmax((B + α·H + β·F + γ·A + T) / τ)
```

Where:
- **B** — Base score (BPE logits converted to word scores)
- **H** — Hebbian co-occurrence resonance
- **F** — Prophecy fulfillment
- **A** — Destiny attraction
- **T** — Trauma gravity
- **τ** — Temperature (implicitly 1.0 in softmax)

### DarioField Class

**penelope.py:1147–1207**, **microreasoning.py:1028–1092**

State:
- `cooc`: `defaultdict(float)` — Hebbian co-occurrence counts, keyed by `"min(w1,w2)|max(w1,w2)"`
- `bigrams`: `defaultdict(lambda: defaultdict(float))` — bigram co-occurrence (declared but not used in overlay)
- `destiny`: `[0.0] * 8` — 8-element vector, one per macro-category
- `trauma`: `0.0` — trauma level, range [0, 1]
- `prophecy_target`: `None` or word index — the destined word
- `prophecy_age`: `0` — incremented each generation step
- `chambers`: dict with keys `fear`, `love`, `rage`, `void`, `flow`, `complex` — six Kuramoto oscillators

### Overlay Implementation

**`DarioField.overlay()`** — `penelope.py:1186–1207`, `microreasoning.py:1071–1092`

Applied to word scores (after BPE→word conversion), only on the hardcoded 1984 words:

```python
# Chamber-modulated coefficients
alpha_mod = 1 + 0.3*C["love"] - 0.2*C["rage"] + 0.1*C["flow"]
gamma_mod = 1 + 0.4*C["void"] + 0.2*C["complex"]
```

For each word `v` in `[0, NWORDS)`:

1. **Hebbian (H)**: Sum co-occurrence counts between `v` and the last 8 words in context. If `h > 0`: `score[v] += alpha_mod * 0.3 * min(h, 1.0)` — capped at 1.0.

2. **Prophecy (F)**: If `v == prophecy_target`: `score[v] += 0.5 * log(1 + prophecy_age)` — grows logarithmically with each step, increasing pressure to fulfill the prophecy.

3. **Destiny (A)**: `score[v] += gamma_mod * 0.25 * destiny[cat] / d_max` where `cat = word_category(v)` and `d_max = max(|d| for d in destiny) + 0.01`.

### Hebbian Learning

**`update_cooc(w1, w2)`** (`penelope.py:1160–1162`): increments the symmetric co-occurrence counter for the word pair. Called during generation after each word is picked (`penelope.py:1336`).

This is **live Hebbian learning** — the co-occurrence matrix is updated in real-time during generation, so earlier word pairs influence later word selection within the same chain.

### Prophecy

At chain start (`run_chain`, `penelope.py:1263–1268`):
- A prophecy target is chosen from categories 2 (EMOTION), 5 (ABSTRACT), or 7 (MATERIAL+).
- `prophecy_age` starts at 0, incremented each step.
- Fulfillment is checked at the end: `fulfilled = field.prophecy_target in chain` (`penelope.py:1349`).

### Destiny

The destiny vector has 8 elements (one per macro-category). After each picked word (`penelope.py:1337–1338`):

```python
cat = word_category(pick)
field.destiny[cat] = 0.3 + 0.7 * field.destiny[cat]
```

This is an exponential moving average biased toward recently visited categories — creating gravitational attraction toward familiar semantic territory.

### Kuramoto Chambers

**`update_chambers(step_idx)`** (`penelope.py:1168–1184`, `microreasoning.py:1052–1069`)

Six emotional oscillators coupled by sine (Kuramoto model):

1. **Phase-dependent excitation** based on `depth = step_idx / N_LAYERS`:
   - Phase 0 (`depth < 0.33`): `flow += 0.05`
   - Phase 1 (`0.33 ≤ depth < 0.66`): `fear += 0.04`
   - Phase 2 (`depth ≥ 0.66`): `void += 0.05`
   - If `depth > 0.75`: `complex += 0.03`
   - If `trauma > 0.3`: `rage += 0.04`

2. **Kuramoto coupling** (`K = 0.02`):
   ```python
   for i in C:
       for j in C:
           if i != j:
               C[i] += K * math.sin(old[j] - old[i])
   ```

3. **Decay** — each chamber decays per step with its own rate:
   - fear: 0.95, love: 0.95, rage: 0.93, void: 0.96, flow: 0.94, complex: 0.97

4. **Clamping**: all chambers clamped to `[0, 1]`.

### Trauma

Trauma accumulates after step 7 and decays each step (`penelope.py:1342–1344`):

```python
if step > 7:
    field.trauma = min(1, field.trauma + 0.1)
field.trauma *= 0.97
```

Trauma indirectly affects the Dario field through the rage chamber (rage increases when trauma > 0.3).

---

## 5. 12-Word Chain Generation

**`run_chain()`** — `penelope.py:1259–1352`, `microreasoning.py:1161–1249`

### Step-by-Step Process

**Setup:**
1. Extract key word from input text (`extract_key` — longest non-stop word).
2. Find seed word in vocabulary (`find_seed` — exact match, substring match, or prefix match).
3. Set prophecy target (random word from EMOTION, ABSTRACT, or MATERIAL+ categories).
4. Initialize BPE buffer with seed word's BPE tokens.
5. Initialize chain as `[seed]`, forbidden set as `{seed}`.

**Generation loop** (12 iterations, `GEN_STEPS = 12`):

For each step:

1. **Update Dario chambers**: `field.update_chambers(step)`, increment `prophecy_age`.

2. **Forward pass**: Truncate BPE buffer to last `MAX_SEQ` (256) tokens. Run through 8-layer transformer → BPE logits `[BPE_VOCAB]` for last position.

3. **BPE-to-word conversion**: `bpe_logits_to_word_scores()` computes `score(w) = mean(bpe_logits[tok] for tok in word's BPE tokens)`.

4. **Dario overlay**: Apply Hebbian + prophecy + destiny signals to word scores.

5. **Mask forbidden words**: Set score to `-1e9` for any word already in the chain (no repeats).

6. **Top-k=12 sampling**:
   - Softmax over all word scores → probabilities.
   - Sort by probability, take top 12 candidates.
   - Sample from these 12 proportionally to their probabilities.
   ```python
   # penelope.py:1308–1317
   probs = softmax(word_scores)
   indexed = sorted(enumerate(probs), key=lambda x: -x[1])[:12]
   total = sum(max(0, p) for _, p in indexed) + 0.001
   r = random.random() * total
   pick = indexed[0][0]
   for idx, p in indexed:
       r -= max(0, p)
       if r <= 0:
           pick = idx
           break
   ```

7. **Update state**:
   - Append picked word to chain and forbidden set.
   - Append picked word's BPE tokens to BPE buffer (context grows).
   - Update Hebbian co-occurrence (live learning).
   - Update destiny vector for picked word's category.
   - Update trauma (accumulates after step 7, decays by 0.97).

8. **Output**: Print the word. Last word (step 11) is marked with `*`.

**Post-chain**: Report drift (unique macro-categories visited / 8) and prophecy fulfillment.

---

## 6. PEN7 Weight Format

### Header

**`Penelope.save()`** — `penelope.py:880–923`, **`Resonance.save()`** — `microreasoning.py:404–428`

The PEN7 binary format stores weights as 32-bit floats with an 8-integer header:

```
Offset  Size    Field
0       4       Magic: 0x50454E37 ("PEN7" in hex — P=0x50, E=0x45, N=0x4E, 7=0x37)
4       4       BPE_VOCAB (2048)
8       4       NWORDS (1984)
12      4       DIM (448)
16      4       HDIM (896)
20      4       N_HEADS (7)
24      4       N_LAYERS (8)
28      4       MAX_SEQ (256)
```

All header values are `int32` (signed). Total header size: 32 bytes.

### Save Order

After the 32-byte header, all weights are stored as contiguous `float32` values:

**Global weights:**
1. `tok_emb` — `[BPE_VOCAB × DIM]` = 2048 × 448 = 917,504 floats
2. `pos_emb` — `[MAX_SEQ × DIM]` = 256 × 448 = 114,688 floats
3. `final_norm` — `[DIM]` = 448 floats
4. `lm_head` — `[BPE_VOCAB × DIM]` = 2048 × 448 = 917,504 floats

**Per-layer weights** (repeated 8 times, layers 0–7):
1. `attn_norm` — `[DIM]` = 448 floats
2. `wq` — `[DIM × DIM]` = 200,704 floats
3. `wk` — `[DIM × DIM]` = 200,704 floats
4. `wv` — `[DIM × DIM]` = 200,704 floats
5. `wo` — `[DIM × DIM]` = 200,704 floats
6. `wr` — `[DIM × DIM]` = 200,704 floats (RRPRAM)
7. `gate` — `[2]` = 2 floats
8. `ffn_norm` — `[DIM]` = 448 floats
9. `w_gate` — `[DIM × HDIM]` = 401,408 floats
10. `w_up` — `[DIM × HDIM]` = 401,408 floats
11. `w_down` — `[HDIM × DIM]` = 401,408 floats

### Parameter Count

| Component | Count |
|---|---|
| Global (tok_emb + pos_emb + final_norm + lm_head) | 1,950,144 |
| Per layer | 2,208,642 |
| 8 layers total | 17,669,136 |
| **Grand total** | **19,619,280** |

File size: 32 (header) + 19,619,280 × 4 (f32) = **78,477,152 bytes** (~74.8 MiB, ~78.5 MB).

### Load Validation

On load (`penelope.py:925–974`, `microreasoning.py:430–467`):
1. Read 32-byte header, verify magic = `0x50454E37`.
2. Verify all architecture constants match (BPE_VOCAB, NWORDS, DIM, HDIM, N_HEADS, N_LAYERS, MAX_SEQ).
3. Read remaining data as float32, populate model weights in save order.

microreasoning.py also handles legacy v2 format (magic `0x50454E32`) by detecting it and re-initializing random weights.

---

## 7. Forward Pass — Per-Layer Computation

**`Penelope.forward(bpe_ids)`** — `penelope.py:755–878`
**`Resonance.forward(bpe_ids)`** — `microreasoning.py:273–402`

Input: list of BPE token IDs (truncated to last `MAX_SEQ = 256` tokens).
Output: BPE logits `[BPE_VOCAB]` for the last position.

### Embedding

```
x[t] = tok_emb[bpe_ids[t]] + pos_emb[t]     for t = 0..S-1
```

`tok_emb` is `[BPE_VOCAB, DIM]`, `pos_emb` is `[MAX_SEQ, DIM]`. Both are learned.

### Per Layer (repeated 8 times)

For each layer `l` (0–7):

**Step 1 — Pre-attention RMSNorm:**
```
h[t] = rmsnorm(x[t], attn_norm_l)     for each position t
```
RMSNorm (`penelope.py:692–695`): `rmsnorm(x, g, n) = g * x * (1/sqrt(mean(x²) + 1e-5))`

**Step 2 — QKV projections:**
```
q[t] = wq_l @ h[t]
k[t] = wk_l @ h[t]
v[t] = wv_l @ h[t]
```

**Step 3 — RoPE** (see Section 9 for details):
Apply rotary position embeddings to q and k.

**Step 4 — Multi-head causal attention:**
For each head `hd` (0–6), for each position `ti`:
```
scores[tj] = (q[ti][hd] · k[tj][hd]) / sqrt(HEAD_DIM)     for tj ≤ ti (causal)
attn[tj] = softmax(scores)[tj]
av[ti][hd] = Σ_tj attn[tj] * v[tj][hd]
```

**Step 5 — Output projection:**
```
qkv_out[t] = wo_l @ av[t]
```

**Step 6 — RRPRAM:**
```
rrp[t] = wr_l @ h[t]
```

**Step 7 — Gated residual:**
```
(w0, w1) = softmax(gate_l[0], gate_l[1])
x[t] = x[t] + w0 * qkv_out[t] + w1 * rrp[t]
```

**Step 8 — Pre-FFN RMSNorm:**
```
h2[t] = rmsnorm(x[t], ffn_norm_l)
```

**Step 9 — SwiGLU FFN:**
```
fg = w_gate_l @ h2[t]          [HDIM]
fu = w_up_l @ h2[t]            [HDIM]
sw = silu(fg) * fu              [HDIM]    (element-wise)
fd = w_down_l @ sw              [DIM]
x[t] = x[t] + fd
```

SiLU activation (`penelope.py:698–699`): `silu(x) = x / (1 + exp(-x))` (clamped to 0 for x < -20).

### Final Projection

After all 8 layers:
```
xn = rmsnorm(x[S-1], final_norm)     (last position only)
logits = lm_head @ xn                [BPE_VOCAB]
```

`lm_head` is a separate `[BPE_VOCAB, DIM]` matrix (not tied to `tok_emb`).

---

## 8. BPE-to-Word Conversion

**`bpe_logits_to_word_scores()`** — `penelope.py:981–999`, `microreasoning.py:1113–1127`

This is the bridge between BPE-level thinking and word-level speaking (the dual tokenizer in action).

For each word `w` in the generation vocabulary:
```
score(w) = mean(bpe_logits[tok] for tok in word_w_bpe_tokens)
```

Each word's BPE token IDs are precomputed (see Section 1). The score is the arithmetic mean of the BPE logits at those token positions.

**penelope.py** indexes into `vocab_bpe` for hardcoded words (indices 0–1983) and `ext_vocab` for extended words.

**microreasoning.py** iterates over `ext_vocab` tuples directly, where each entry is `(word_str, bpe_ids, from_hardcoded)`.

---

## 9. RoPE — Rotary Position Embedding

### Implementation

**penelope.py:793–811** (inline in forward pass):

```python
theta_base = 10000.0
for t in range(S):                      # each position
    for hd in range(N_HEADS):           # each head (7)
        offset = hd * HEAD_DIM          # head start in DIM-sized vector
        for d in range(HEAD_DIM // 2):  # pairs within head (32 pairs)
            freq = 1.0 / (theta_base ** (2.0 * d / HEAD_DIM))
            cos_f = math.cos(t * freq)
            sin_f = math.sin(t * freq)
            # rotate q: (q0, q1) → (q0*cos - q1*sin, q0*sin + q1*cos)
            q0 = q_all[t][offset + d]
            q1 = q_all[t][offset + d + HEAD_DIM // 2]
            q_all[t][offset + d]                = q0 * cos_f - q1 * sin_f
            q_all[t][offset + d + HEAD_DIM // 2] = q0 * sin_f + q1 * cos_f
            # same rotation for k
```

**microreasoning.py:166–187** — `apply_rope(q, k, seq_len, n_heads, head_dim)` — same computation extracted into a standalone function. Layout differs: `(t * n_heads + h) * head_dim` (interleaved heads) vs penelope.py's `[S][DIM]` with `hd * HEAD_DIM` offset.

### Details

- Base frequency: θ = 10000.0
- For each pair dimension `d` in `[0, HEAD_DIM/2)`: `freq = 1 / (10000^(2d/HEAD_DIM))`
- Position encoding: `angle = t * freq`
- Rotation: standard 2D rotation matrix applied to pairs `(q[d], q[d + HEAD_DIM/2])`
- Applied to both Q and K (not V)
- HEAD_DIM = 64 → 32 rotation pairs per head

The key structural difference: penelope.py stores q/k as `[S][DIM]` (list of lists), while microreasoning.py stores them as flat `[S * DIM]` arrays with explicit offset math.

---

## 10. Differences Between penelope.py and microreasoning.py

Both implement the same v7 Resonance architecture with identical BPE tables, vocabulary, and PEN7 format. Key differences:

### Vocabulary Loading
- **penelope.py**: Hardcodes all 1984 words directly in source (lines 46–275).
- **microreasoning.py**: Loads from `microreasoning/1984.txt` at import time (line 63–65).

### Class Naming
- **penelope.py**: Model class is `Penelope` (line 737).
- **microreasoning.py**: Model class is `Resonance` (line 249).

### Data Layout
- **penelope.py**: Uses list-of-lists for sequence data (e.g., `x[t][d]`).
- **microreasoning.py**: Uses flat arrays with explicit offset math (e.g., `x[t * DIM + d]`). More memory-efficient, closer to C implementation style.

### RoPE
- **penelope.py**: Inline in forward pass (lines 793–811). Q/K stored as `[S][DIM]`.
- **microreasoning.py**: Extracted into `apply_rope()` function (line 166–187). Q/K stored as flat `[S * N_HEADS * HEAD_DIM]`.

### Extended Vocabulary Initialization
- **penelope.py**: `init_ext_vocab()` called explicitly in `main()` (line 1385). Uses module-level `SUFFIX_FRAGMENTS` set (line 299).
- **microreasoning.py**: `EXT_VOCAB = init_ext_vocab()` at module import time (line 874). Uses a local `_SUFFIXES` set within the function (line 853).

### Optimizer
- **penelope.py**: No optimizer class. Training uses direct SGD-style updates on embeddings only.
- **microreasoning.py**: Includes the `Chuck` optimizer class (lines 884–944) — Adam-style with momentum, RMS, bias correction, plus macro patience (noise injection after 50 steps of stagnation).

### Legacy Format Support
- **penelope.py**: Only supports PEN7 format. Rejects unknown magic numbers.
- **microreasoning.py**: Additionally detects legacy v2 format (magic `0x50454E32`) and warns about incompatibility (lines 455–463).

### DarioField.overlay() Signature
- **penelope.py**: `overlay(self, word_scores, context_ids, step_idx, n_words)` — accepts explicit `n_words` parameter.
- **microreasoning.py**: `overlay(self, logits, context_ids, step_idx)` — infers `n_words` from `len(logits)`.

### Forbidden Word Handling
- **penelope.py**: Uses both an index-based `forbidden` set and a string-based `forbidden_words` set. Extended vocab words are masked by string comparison when `has_weights=True` (lines 1297–1305).
- **microreasoning.py**: Uses index-based `forbidden` set, but masks by comparing word strings from `gen_vocab` tuples (lines 1202–1208).

### Training Sequence Length
- **penelope.py**: `seq_len = min(MAX_SEQ, len(corpus_bpe) - 1)` — up to full 256 tokens.
- **microreasoning.py**: `seq_len = min(MAX_SEQ, 64)` — capped at 64 tokens for pure-python speed (line 977).

---

## Cascade 1 Cross-Reference

Per CASCADE01.md, Penelope's role in the daily cycle:

```
03:30 UTC  PENELOPE (19.6M params, Resonance architecture)
           Input: today's haiku
           Process: BPE encode → 8-layer transformer (QKV + RRPRAM) → word-level decode
           Output: 12 associative words (unidirectional chain, no repeats)
           Splits: (a) → Haiku tomorrow  (b) → up to Molequla + NanoJanus
```

Verified against code:
- **19.6M params**: `param_count()` returns 19,619,280 ✓
- **BPE encode**: `bpe_encode()` converts text to 2048-token subword IDs ✓
- **8-layer transformer**: `N_LAYERS = 8`, sequential processing ✓
- **QKV + RRPRAM**: Each layer has multi-head attention + Wr resonance matrix + learned softmax gate ✓
- **Word-level decode**: `bpe_logits_to_word_scores()` + Dario overlay + top-k=12 sampling ✓
- **12 associative words**: `GEN_STEPS = 12`, unidirectional chain ✓
- **No repeats**: Forbidden set prevents word repetition ✓

Health indicators (from CASCADE01.md):
- **Healthy**: 12 distinct words from 1984 vocabulary or BPE-extended set.
- **Failed**: Fewer than 12 words → weight loading failed or input was empty.
- **Debug noise**: `[tongue]` or `[gguf]` lines → wrong output parsing.
