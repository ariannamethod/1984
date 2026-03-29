# 1984. Penelope.

---

> *consciousness → ash → prisoner → moss → island → hero → ink → sowing → fog → oar → salt → seed*

Janus Architecture. Resonance engine.

*by Arianna Method.*

<p align="center">
  <img src="assets/penelope.jpg" width="320" />
</p>

Penelope resonates. Two vocabulary modes, zero gibberish:

- **Weightless mode (no trained weights):** Penelope speaks exactly 1984 curated words — body, nature, emotion, time, society, abstraction, action, ritual, geometry, myth. The Dario Equation alone drives word selection.
- **Trained mode (with weights):** The vocabulary expands through interference. The 1984 core words remain, and ~1000 additional words are extracted directly from the BPE weight matrix — every BPE token that decodes to a real English word (not a suffix fragment, not a stop word) joins the candidates (~3000 total). These extra words are scored through the same learned weights that drive the transformer.

Every output is a real word. Gibberish is architecturally impossible.

## Architecture

8-layer Resonance engine. 19.6M parameters. The soul thinks in BPE subwords, the mouth speaks only real words.

Per layer:

```
h  = RMSNorm(x)
q  = RoPE(h @ Wq)                      7-head attention
k  = RoPE(h @ Wk)                      with rotary positions
v  = h @ Wv
attn = softmax(q @ k^T / √d, causal)
qkv  = (attn @ v) @ Wo

rrp  = h @ Wr                          RRPRAM resonance

gate = softmax([g₀, g₁])               learned blend
x  = x + gate[0] · qkv + gate[1] · rrp

h₂ = RMSNorm(x)
x  = x + W_down(SiLU(h₂ @ W_gate) · (h₂ @ W_up))    SwiGLU FFN
```

After all 8 layers: `logits = RMSNorm(x) @ lm_head`

The Dario Equation overlays on word-level scores during generation:

```
p(x|Φ) = softmax((B + α·H + β·F + γ·A + T) / (τ · vibe))
```

Where B is bigram affinity, H is Hebbian co-occurrence, F is prophecy fulfillment, A is destiny attraction, T is trauma gravity — all modulated by 6 Kuramoto oscillators (fear, love, rage, void, flow, complexity).

**Training:** BPE targets (standard next-token prediction). The model learns language through subword representations. At inference, BPE logits are converted to word-level scores via mean aggregation over each word's BPE tokens. Dual tokenizer — trained on BPE, speaks in words.

**Parameters:** DIM=448, HDIM=896, 7 heads, head_dim=64, 8 layers, BPE vocab 2048. Total 19,619,280 params (78.5MB f32). Trained weights: `weights/penelope.bin` (PEN7 format). Loss: **1.96** on 85MB Gutenberg corpus.

### Examples

Trained mode:

```
"darkness eats the city"
darkness → fog → hawk → brass → candle → burn → sing → landing → sand → raft → loss → boat

"what is consciousness"
consciousness → ash → prisoner → moss → island → hero → ink → sowing → fog → oar → salt → seed
```

Weightless mode (no training, Dario Field only):

**"love"** — destined: entropy — unfulfilled

<p align="center">
  <img src="assets/demo_love.png" width="520" />
</p>

> love → create → push → remember → hide → delay → vibration → gravity → comfort → basalt → shapeshifter → scatter → **abandon**

**"hello"** — destined: story — unfulfilled

<p align="center">
  <img src="assets/demo_hello.png" width="520" />
</p>

> helix → bend → remember → carry → hide → delay → wedding → enmeshment → sonata → decryption → master → fascination → **triumph**

**"Penelope"** — destined: longing — unfulfilled

<p align="center">
  <img src="assets/demo_penelope.png" width="520" />
</p>

> pen → study → pyramid → hoard → sacrifice → healer → certainty → screw → ambivalence → intimacy → yearning → delight → **disgust**

She hears her own name and walks from pen to disgust through sacrifice and yearning.

**"how are you?"** — destined: blessing — unfulfilled

<p align="center">
  <img src="assets/demo_howareyou.png" width="520" />
</p>

> hormone → asteroid → gather → erosion → lock → other → verdict → train → vulnerability → rage → guilt → love → **hatred**

Asked how she's doing, she ends on hatred. Through love.

**"what is the meaning of life?"** — destined: ambivalence — **fulfilled**

<p align="center">
  <img src="assets/demo_meaning.png" width="520" />
</p>

> meaning → gem → ambivalence → well → violet → icon → smog → accumulation → intimacy → yearning → certainty → screw → **paranoia**

The only fulfilled prophecy. She was destined for ambivalence — and found it at step 3. Then kept walking anyway, past intimacy and yearning, through certainty, into paranoia.

## Implementations

The same Resonance engine — expressed identically across 9 programming languages:

| Language | File | Build |
|----------|------|-------|
| JavaScript | `penelope.html` | Open in browser |
| C | `penelope.c` | `cc penelope.c -O2 -lm -o penelope` |
| TypeScript | `penelope.ts` | `npx tsx penelope.ts` |
| Python | `penelope.py` | `python3 penelope.py` |
| Rust | `penelope.rs` | `rustc -O penelope.rs -o penelope_rs` |
| Zig | `penelope.zig` | `zig build-exe penelope.zig` |
| Julia | `penelope.jl` | `julia penelope.jl` |
| **AML** | `penelope.aml` | `./amlc penelope.aml --run` |

AML — the Arianna Method Language — ships with a dedicated mini-compiler (`ariannamethod/ariannamethod.c`) that transpiles `BLOOD COMPILE` blocks to native C. AML provides the ceremony. C provides the math.

```
cc ariannamethod/ariannamethod.c -o amlc
./amlc penelope.aml -o penelope_aml
./penelope_aml "darkness eats the city"
```

## Usage

```bash
./penelope                              # interactive REPL
./penelope "darkness eats the city"     # single chain from text
./penelope --load penelope.bin          # load trained weights
./penelope --train corpus.txt           # train on BPE targets
./penelope --train corpus.txt --steps N # train N steps
./penelope --save penelope.bin          # save weights after training
```

## Microreasoning

`microreasoning/microreasoning.py` is a distilled standalone version. Same architecture, same Dario Equation — but the vocabulary lives in an external `1984.txt` file. Drop in your own word list and the engine adapts.

```bash
cd microreasoning
python3 microreasoning.py "love"
python3 microreasoning.py --train corpus.txt --steps 25000
python3 microreasoning.py --load weights.bin "what is consciousness"
```

## The Vocabulary

The core 1984 words are curated, not scraped. 29 semantic categories:

Body, Nature, Emotion, Time, Society, Abstract, Action, Material, Food, Architecture, Relationship, Philosophy, Music, Weather, Ritual, Labor, Geometry, Animal, Color, Transport, Domestic, Communication, Medical, Cosmic, Bureaucracy, Mythic, Textual, Psychological, Final.

No word is wasted. No word is missing.

**Weightless mode** = exactly 1984 words. Nothing more.

**Trained mode** = 1984 core words + ~1000 words extracted from the BPE weight matrix. The transformer is trained on BPE subword targets (standard next-token prediction on 2048-token vocabulary). At inference, each BPE token is decoded back to its string — if it's a real English word (3+ chars, alphabetic, not a stop word, not a suffix fragment like "tion" or "ment"), it joins the extended vocabulary. These ~1000 extra words are scored through the same learned weights as the core 1984. The result is interference: the curated vocabulary and the emergent BPE vocabulary overlap, reinforce, and compete.

---

*By Arianna Method.*
