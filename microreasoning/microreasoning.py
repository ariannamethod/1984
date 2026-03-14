#!/usr/bin/env python3
"""
microreasoning.py — 1984 words. 12 steps of associative resonance.

not a transformer. not pretending to be.
why? good question. really, why does this thing exist?

because someone wanted to generate one coherent word per step 
instead of the gibberish we all love from char-level models.
so here's the deal: 
  - input: BPE tokenizer reads your text with nuance
  - output: word-level from 1984 curated words. gibberish impossible.
  - formula: the Dario Equation replaced boring softmax because life must evolve
  - 12 steps of microreasoning. each step is another generation.
    each one has its own weights. together they form an emergent party.
 
train it on Gutenberg. train it on Dostoevsky. train it on your diary.
the associations will become sharper. the resonance will deepen.
but even without training, the dual tokenizer guarantees every output is a real word.

  python microreasoning.py                          # interactive
  python microreasoning.py "love"                   # single chain
  python microreasoning.py --train corpus.txt       # train
  python microreasoning.py --load model.bin         # load weights
  
by Arianna Method. Janus Architecture.
"""

import math
import random
import struct
import sys
import os
import re
from collections import defaultdict

# ===================================================================
# 1984 WORDS — loaded from file, not hardcoded like a caveman
# one word per line. 1984 of them. that's the whole vocabulary.
# why 1984? because Orwell would appreciate the irony.
# ===================================================================

VOCAB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1984.txt")
with open(VOCAB_FILE) as f:
    VOCAB = [line.strip() for line in f if line.strip()]

V = len(VOCAB)  # 1984
STEPS = 12      # 12 steps. 12 different weight sets. 12 drunk dudes at a party making emergent decisions.
D = 384         # embedding dim
M = 768         # SwiGLU hidden dim — twice D because SwiGLU likes elbow room

VOCAB_SET = set(VOCAB)
VOCAB_IDX = {}
for i, w in enumerate(VOCAB):
    if w not in VOCAB_IDX:
        VOCAB_IDX[w] = i

# stop words. boring but necessary. we skip these during tokenization.
STOP = set("i me my we our you your he she it they them the a an and or but in on at to for of is am are was were be been being have has had do does did will would shall should can could may might must not no nor so if then than that this these those what which who whom how when where why all each every some any few many much more most other another such".split())


# ===================================================================
# MATH — numpy-free, pure python
# because dependencies are for people who trust other people's code
# ===================================================================

def randn():
    u1 = random.random() + 1e-12
    u2 = random.random() + 1e-12
    return math.sqrt(-2 * math.log(u1)) * math.cos(6.2831853 * u2)


def zeros(n):
    return [0.0] * n


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def vadd(a, b):
    return [x + y for x, y in zip(a, b)]


def vsub(a, b):
    return [x - y for x, y in zip(a, b)]


def vscale(a, s):
    return [x * s for x in a]


def matmul_mv(W, x, rows, cols):
    # W[rows*cols] @ x[cols] -> out[rows]
    out = zeros(rows)
    for i in range(rows):
        s = 0.0
        for j in range(cols):
            s += W[i * cols + j] * x[j]
        out[i] = s
    return out


def matmul_mtv(W, x, rows, cols):
    # W^T[cols*rows] @ x[rows] -> out[cols]. W stored [rows, cols].
    out = zeros(cols)
    for j in range(cols):
        s = 0.0
        for i in range(rows):
            s += W[i * cols + j] * x[i]
        out[j] = s
    return out


def rmsnorm(x, g, n):
    # RMSNorm. the well-behaved sibling of LayerNorm.
    ss = sum(v * v for v in x) / n + 1e-5
    inv = 1.0 / math.sqrt(ss)
    return [g[i] * x[i] * inv for i in range(n)]


def silu(x):
    # SiLU aka Swish. sigmoid * x. elegant, differentiable, slightly smug.
    return x / (1.0 + math.exp(-x)) if x > -20 else 0.0


def softmax(x):
    # and let the boring softmax remain, because someone has to do the accounting
    mx = max(x)
    e = [math.exp(v - mx) for v in x]
    s = sum(e)
    return [v / s for v in e]


# ===================================================================
# MODEL — 12 step-specific weight sets + shared embedding
# 12 steps. 12 different weight sets. 12 drunk dudes at a party
# making emergent decisions. step 1 sees the surface. step 12
# sees the bone. together they see more than any single model.
# ===================================================================

class StepWeights:
    """One step's learned weights. ~1.03M params each. not bad for a drunk dude."""
    def __init__(self):
        scale_d = math.sqrt(2.0 / D)
        scale_m = math.sqrt(2.0 / M)
        self.wr     = [randn() * scale_d for _ in range(D * D)]       # RRPRAM resonance
        self.rms    = [1.0] * D                                        # RMSNorm gain
        self.w_gate = [randn() * scale_d for _ in range(D * M)]       # SwiGLU gate
        self.w_up   = [randn() * scale_d for _ in range(D * M)]       # SwiGLU up
        self.w_down = [randn() * scale_m for _ in range(M * D)]       # SwiGLU down

    def param_count(self):
        return D*D + D + D*M + D*M + M*D

    def params(self):
        return self.wr + self.rms + self.w_gate + self.w_up + self.w_down

    def load_from(self, flat, offset):
        o = offset
        self.wr     = flat[o:o+D*D]; o += D*D
        self.rms    = flat[o:o+D]; o += D
        self.w_gate = flat[o:o+D*M]; o += D*M
        self.w_up   = flat[o:o+D*M]; o += D*M
        self.w_down = flat[o:o+M*D]; o += M*D
        return o


class MicroReasoner:
    """
    12 learned steps + shared embedding. ~13M params.
    dual tokenizer: BPE reads your Shakespeare, word-level speaks its truth.
    one clean word per step. this is not a transformer. this is not pretending to be.
    why? because why not.
    """

    def __init__(self):
        scale = math.sqrt(2.0 / V)
        self.embed = [randn() * scale for _ in range(V * D)]   # E[V, D]
        self.steps = [StepWeights() for _ in range(STEPS)]

    def param_count(self):
        return V * D + sum(s.param_count() for s in self.steps)

    def get_embed(self, idx):
        return self.embed[idx * D:(idx + 1) * D]

    def pool_context(self, word_ids):
        # average embedding of context words. simple but honest.
        if not word_ids:
            return zeros(D)
        ctx = zeros(D)
        for wid in word_ids:
            e = self.get_embed(wid)
            ctx = vadd(ctx, e)
        return vscale(ctx, 1.0 / len(word_ids))

    def forward_step(self, context_ids, step_idx):
        """One step: context -> logits[V]. the forward pass. the easy part."""
        sw = self.steps[step_idx]
        ctx = self.pool_context(context_ids)

        # RRPRAM resonance: query = ctx @ Wr
        query = matmul_mv(sw.wr, ctx, D, D)

        # RMSNorm — keep things well-behaved
        query = rmsnorm(query, sw.rms, D)

        # SwiGLU: hidden = silu(query @ W_gate) * (query @ W_up) @ W_down
        gate = matmul_mv(sw.w_gate, query, M, D)
        up = matmul_mv(sw.w_up, query, M, D)
        swiglu = [silu(gate[i]) * up[i] for i in range(M)]
        hidden = matmul_mv(sw.w_down, swiglu, D, M)

        # residual — because even resonance needs a safety net
        out = vadd(query, hidden)

        # logits = E @ out (tied weights). word-level output.
        # gibberish? not on our watch. every generation is a real word.
        logits = matmul_mv(self.embed, out, V, D)
        return logits

    def save(self, path):
        """Save all weights to binary file."""
        flat = list(self.embed)
        for s in self.steps:
            flat.extend(s.params())
        with open(path, "wb") as f:
            f.write(struct.pack("iiii", V, D, M, STEPS))
            for v in flat:
                f.write(struct.pack("f", v))
        print(f"  saved {path}: {len(flat)} params ({os.path.getsize(path)/1e6:.1f}MB)")

    def load(self, path):
        """Load weights from binary file."""
        with open(path, "rb") as f:
            v, d, m, st = struct.unpack("iiii", f.read(16))
            assert v == V and d == D and m == M and st == STEPS, \
                f"config mismatch: file has V={v} D={d} M={m} S={st}"
            flat = []
            while True:
                chunk = f.read(4)
                if not chunk:
                    break
                flat.append(struct.unpack("f", chunk)[0])
        o = 0
        self.embed = flat[o:o + V * D]; o += V * D
        for s in self.steps:
            o = s.load_from(flat, o)
        print(f"  loaded {path}: {len(flat)} params")


# ===================================================================
# BPE INPUT — stem + greedy longest vocab match
#
# dual tokenizer: BPE reads your Shakespeare, word-level speaks its truth.
# three-stage tokenizer for arbitrary text:
#   1. exact vocab match     ("fire" -> fire)
#   2. suffix stripping       ("burning" -> burn, "created" -> create)
#   3. greedy decomposition   ("heartbreak" -> heart + break)
#
# the 1984 vocab words ARE the BPE token vocabulary.
# greedy longest-match IS BPE encoding.
# ===================================================================

SUFFIXES = [
    "ting","ning","ring","ling","ding","ping","bing","ging","ming","king",
    "sing","zing",
    "ing","ment","ness","tion","sion","able","ible","ence","ance",
    "eous","ious","ful","less","ize","ise","ous","ive","ity",
    "ly","er","ed","est","al","en","es","s",
]

VOCAB_LENS = [len(w) for w in VOCAB]


def try_stem(word):
    """Strip suffix, try exact match, stem+'e', doubled consonant removal."""
    wlen = len(word)
    for sfx in SUFFIXES:
        slen = len(sfx)
        if wlen <= slen + 2:
            continue
        if not word.endswith(sfx):
            continue
        stem = word[:wlen - slen]
        if stem in VOCAB_IDX:
            return VOCAB_IDX[stem]
        stem_e = stem + "e"
        if stem_e in VOCAB_IDX:
            return VOCAB_IDX[stem_e]
        if len(stem) >= 3 and stem[-1] == stem[-2]:
            stem_short = stem[:-1]
            if stem_short in VOCAB_IDX:
                return VOCAB_IDX[stem_short]
    return -1


def greedy_vocab_match(word):
    """Greedy longest vocab match within a word. this IS the BPE."""
    ids = []
    pos = 0
    wlen = len(word)
    while pos < wlen:
        best, best_len = -1, 0
        for v in range(V):
            vl = VOCAB_LENS[v]
            if vl <= best_len or vl > wlen - pos:
                continue
            if word[pos:pos + vl] == VOCAB[v]:
                best, best_len = v, vl
        if best >= 0 and best_len >= 3:
            ids.append(best)
            pos += best_len
        else:
            pos += 1
    return ids


def tokenize_text(text):
    """Three-stage BPE: exact -> stem -> greedy vocab decomposition."""
    words = re.findall(r"[a-z]+", text.lower())
    ids = []
    for w in words:
        if len(w) < 2 or w in STOP:
            continue
        # 1. exact vocab match
        if w in VOCAB_IDX:
            ids.append(VOCAB_IDX[w])
            continue
        # 2. stem + match
        idx = try_stem(w)
        if idx >= 0:
            ids.append(idx)
            continue
        # 3. greedy longest vocab match (BPE decomposition)
        for sub_id in greedy_vocab_match(w):
            if not ids or ids[-1] != sub_id:
                ids.append(sub_id)
    return ids


# ===================================================================
# CHUCK OPTIMIZER — named after a friend. patience of a saint,
# noise of a rebel. tracks momentum and RMS like Adam, but adds
# macro patience: if loss hasn't improved in 50 steps, it gets
# restless and adds noise. because stagnation is death.
# ===================================================================

class Chuck:
    """
    Chuck optimizer. Adam-style first/second moments with bias correction,
    plus macro patience and stagnation noise.
    beta1=0.9 (momentum), beta2=0.999 (RMS), eps=1e-8.
    if loss hasn't improved for `patience` steps, add noise instead of update.
    """

    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8, patience=50,
                 noise_scale=1e-3):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.patience = patience
        self.noise_scale = noise_scale
        # per-parameter-group moments: keyed by (step_idx, param_name) or "embed"
        self.m = {}   # first moment
        self.v = {}   # second moment
        self.t = {}   # step counter per group
        # macro patience state
        self.best_loss = float("inf")
        self.steps_without_improvement = 0

    def _ensure_group(self, key, size):
        if key not in self.m:
            self.m[key] = [0.0] * size
            self.v[key] = [0.0] * size
            self.t[key] = 0

    def report_loss(self, loss):
        """Call once per training step with the current loss."""
        if loss < self.best_loss:
            self.best_loss = loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

    def is_stagnant(self):
        return self.steps_without_improvement >= self.patience

    def update(self, params, grads, key):
        """
        Update params in-place using Adam with bias correction.
        If stagnant, add noise instead — shake the tree.
        params: list of floats (mutable, updated in place)
        grads: list of floats (same length)
        key: string identifier for this parameter group
        """
        n = len(params)
        self._ensure_group(key, n)

        if self.is_stagnant():
            # stagnation noise: small random perturbation to escape local minima
            # Chuck doesn't sit still when things stop moving
            for i in range(n):
                params[i] += self.noise_scale * randn()
            return

        self.t[key] += 1
        t = self.t[key]
        m = self.m[key]
        v = self.v[key]

        # bias correction factors
        bc1 = 1.0 / (1.0 - self.beta1 ** t)
        bc2 = 1.0 / (1.0 - self.beta2 ** t)

        for i in range(n):
            g = grads[i]
            # update moments
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g
            # bias-corrected
            m_hat = m[i] * bc1
            v_hat = v[i] * bc2
            # update
            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


# ===================================================================
# TRAINING — next-word prediction, step s predicts word[s+1]
# the hard part. the part where resonance learns to resonate.
# Chuck optimizer handles the weight updates with patience and noise.
# ===================================================================

def train(model, data_path, steps=5000, lr=3e-4):
    """Train on text corpus. Each 13-word window trains all 12 steps."""
    with open(data_path, "r") as f:
        text = f.read()
    ids = tokenize_text(text)
    if len(ids) < STEPS + 2:
        print(f"  corpus too small: {len(ids)} words (need {STEPS+2}+)")
        return

    print(f"  corpus: {len(text)} chars -> {len(ids)} vocab words")
    print(f"  model: {model.param_count():,} params ({model.param_count()*4/1e6:.1f}MB f32)")
    print(f"  training: {steps} steps, lr={lr:.1e}")
    print(f"  optimizer: Chuck (patience=50, noise when stuck)")

    # Chuck optimizer. named after a friend. patience of a saint, noise of a rebel.
    optimizer = Chuck(lr=lr)

    window = STEPS + 1  # 13 words: 1 seed + 12 targets

    for step in range(1, steps + 1):
        # random window from corpus
        start = random.randint(0, len(ids) - window)
        win = ids[start:start + window]

        total_loss = 0.0

        # each of 12 steps predicts next word
        for s in range(STEPS):
            context = win[:s + 1]      # words 0..s
            target = win[s + 1]        # word s+1

            logits = model.forward_step(context, s)
            probs = softmax(logits)
            p = probs[target]
            if p < 1e-10:
                p = 1e-10
            total_loss -= math.log(p)

            # gradient: d_logits = probs - one_hot(target)
            d_logits = list(probs)
            d_logits[target] -= 1.0

            # backprop through tied output: d_out = d_logits @ E
            sw = model.steps[s]
            ctx = model.pool_context(context)

            # reconstruct forward (yes, again. no free lunch.)
            query = matmul_mv(sw.wr, ctx, D, D)
            query_n = rmsnorm(query, sw.rms, D)
            gate = matmul_mv(sw.w_gate, query_n, M, D)
            up = matmul_mv(sw.w_up, query_n, M, D)
            swiglu = [silu(gate[i]) * up[i] for i in range(M)]
            hidden = matmul_mv(sw.w_down, swiglu, D, M)
            out = vadd(query_n, hidden)

            # d_out from tied weights
            d_out = zeros(D)
            for v in range(V):
                if abs(d_logits[v]) < 1e-8:
                    continue
                ev = model.get_embed(v)
                for j in range(D):
                    d_out[j] += d_logits[v] * ev[j]

            # embedding gradients for tied output
            embed_grads = [0.0] * (V * D)
            for v in range(V):
                if abs(d_logits[v]) < 1e-8:
                    continue
                base = v * D
                for j in range(D):
                    embed_grads[base + j] += d_logits[v] * out[j]
            optimizer.update(model.embed, embed_grads, "embed")

            # d_hidden (residual: d_out goes to both query_n and hidden path)
            d_hidden = list(d_out)

            # backprop through w_down
            d_swiglu = matmul_mtv(sw.w_down, d_hidden, D, M)
            w_down_grads = [0.0] * (M * D)
            for i in range(M):
                for j in range(D):
                    w_down_grads[i * D + j] = swiglu[i] * d_hidden[j]
            optimizer.update(sw.w_down, w_down_grads, f"s{s}_wdown")

            # backprop through SwiGLU
            w_gate_grads = [0.0] * (D * M)
            w_up_grads = [0.0] * (D * M)
            for i in range(M):
                sg = silu(gate[i])
                sig = 1.0 / (1.0 + math.exp(-gate[i])) if gate[i] > -20 else 0
                silu_grad = sig * (1.0 + gate[i] * (1.0 - sig)) if gate[i] > -20 else 0
                d_gate_i = d_swiglu[i] * up[i] * silu_grad
                d_up_i = d_swiglu[i] * sg

                for j in range(D):
                    w_gate_grads[i * D + j] = d_gate_i * query_n[j]
                    w_up_grads[i * D + j] = d_up_i * query_n[j]
            optimizer.update(sw.w_gate, w_gate_grads, f"s{s}_wgate")
            optimizer.update(sw.w_up, w_up_grads, f"s{s}_wup")

            # d_query_n (from SwiGLU input + residual)
            d_qn = list(d_out)
            d_qn_gate = matmul_mtv(sw.w_gate, [
                d_swiglu[i] * up[i] * (
                    (lambda g: (1/(1+math.exp(-g)))*(1+g*(1-(1/(1+math.exp(-g))))) if g > -20 else 0)(gate[i])
                ) for i in range(M)
            ], M, D)
            d_qn_up = matmul_mtv(sw.w_up, [d_swiglu[i] * silu(gate[i]) for i in range(M)], M, D)
            d_qn = vadd(d_qn, vadd(d_qn_gate, d_qn_up))

            # skip RMSNorm backward for simplicity — approximate
            ss_val = sum(v * v for v in query) / D + 1e-5
            inv = 1.0 / math.sqrt(ss_val)
            d_query = [d_qn[i] * sw.rms[i] * inv for i in range(D)]

            # update Wr via Chuck
            wr_grads = [0.0] * (D * D)
            for i in range(D):
                if abs(d_query[i]) < 1e-8:
                    continue
                for j in range(D):
                    wr_grads[i * D + j] = d_query[i] * ctx[j]
            optimizer.update(sw.wr, wr_grads, f"s{s}_wr")

        avg_loss = total_loss / STEPS
        optimizer.report_loss(avg_loss)

        if step % 50 == 0 or step == 1:
            stag = " [stagnant, adding noise]" if optimizer.is_stagnant() else ""
            print(f"  step {step:5d}/{steps}  loss={avg_loss:.4f}  best={optimizer.best_loss:.4f}{stag}")

    print(f"  training complete. best loss: {optimizer.best_loss:.4f}")


# ===================================================================
# DARIO FIELD — live co-occurrence overlay
# and let the boring softmax be replaced by the Dario Equation,
# because life must evolve. p(x|Phi) = softmax((a*H + b*F + g*A) / tau)
# H=Hebbian, F=Prophecy, A=Destiny. live overlay on learned logits.
# ===================================================================

class DarioField:
    """The Dario Equation in action. Hebbian co-occurrence + prophecy + destiny."""
    def __init__(self):
        self.cooc = defaultdict(float)
        self.bigrams = defaultdict(lambda: defaultdict(float))
        self.destiny = [0.0] * 8
        self.trauma = 0.0
        self.prophecy_target = None
        self.prophecy_age = 0
        # Kuramoto chambers. six oscillators coupled by sine.
        # fear, love, rage, void, flow, complex — the emotional substrate.
        self.chambers = {"fear": 0, "love": 0, "rage": 0,
                         "void": 0, "flow": 0, "complex": 0}
        self.decay = {"fear": 0.95, "love": 0.95, "rage": 0.93,
                      "void": 0.96, "flow": 0.94, "complex": 0.97}

    def update_cooc(self, w1, w2):
        k = f"{min(w1,w2)}|{max(w1,w2)}"
        self.cooc[k] += 1.0

    def get_cooc(self, w1, w2):
        k = f"{min(w1,w2)}|{max(w1,w2)}"
        return self.cooc.get(k, 0.0)

    def update_chambers(self, step_idx):
        # Kuramoto-style coupled oscillators. phase-locked emotional resonance.
        C = self.chambers
        depth = step_idx / STEPS
        phase = 0 if depth < 0.33 else (1 if depth < 0.66 else 2)
        if phase == 0: C["flow"] += 0.05
        if phase == 1: C["fear"] += 0.04
        if phase == 2: C["void"] += 0.05
        if depth > 0.75: C["complex"] += 0.03
        if self.trauma > 0.3: C["rage"] += 0.04
        K = 0.02
        old = dict(C)
        for i in C:
            for j in C:
                if i != j:
                    C[i] += K * math.sin(old[j] - old[i])
        for k in C:
            C[k] = max(0, min(1, C[k] * self.decay.get(k, 0.95)))

    def overlay(self, logits, context_ids, step_idx):
        """Add Dario field signal to learned logits. the live part."""
        C = self.chambers
        alpha_mod = 1 + 0.3*C["love"] - 0.2*C["rage"] + 0.1*C["flow"]
        gamma_mod = 1 + 0.4*C["void"] + 0.2*C["complex"]

        for v in range(V):
            h = 0.0
            for ci in context_ids[-8:]:
                h += self.get_cooc(ci, v)
            if h > 0:
                logits[v] += alpha_mod * 0.3 * min(h, 1.0)

            if self.prophecy_target is not None and v == self.prophecy_target:
                logits[v] += 0.5 * math.log(1 + self.prophecy_age)

            cat = word_category(v)
            d_max = max(abs(d) for d in self.destiny) + 0.01
            logits[v] += gamma_mod * 0.25 * self.destiny[cat] / d_max

        return logits


def word_category(idx):
    # 8 semantic categories. body, nature, emotion, time, society, abstract, action, material+
    if idx < 100: return 0
    if idx < 200: return 1
    if idx < 300: return 2
    if idx < 350: return 3
    if idx < 450: return 4
    if idx < 550: return 5
    if idx < 650: return 6
    return 7


# ===================================================================
# GENERATION — 12 steps, each picks one word
# the moment of truth. context in, resonance through, word out.
# ===================================================================

def find_seed(key):
    if key in VOCAB_IDX:
        return VOCAB_IDX[key]
    best, best_score = 0, -1
    for w, i in VOCAB_IDX.items():
        score = 0
        if w in key or key in w:
            score = 3
        for k in range(min(len(w), len(key))):
            if w[k] == key[k]:
                score += 0.5
            else:
                break
        if score > best_score:
            best_score, best = score, i
    return best if best_score > 0 else random.randint(0, 199)


def extract_key(text):
    words = [w for w in text.lower().split() if len(w) > 1 and w not in STOP]
    if not words:
        return text.lower().split()[0] if text.split() else "silence"
    words.sort(key=lambda w: -len(w))
    return words[0]


def run_chain(model, field, text):
    """Run a 12-step chain. seed -> 12 words of emergent resonance."""
    key = extract_key(text)
    seed = find_seed(key)

    # prophecy: pick a destiny target from emotional/abstract/material categories
    deep_cats = [2, 5, 7]
    tcat = random.choice(deep_cats)
    ranges = [(0,100),(100,200),(200,300),(300,350),(350,450),(450,550),(550,650),(650,V)]
    s, e = ranges[tcat]
    field.prophecy_target = random.randint(s, min(e - 1, V - 1))
    field.prophecy_age = 0

    print(f"\n  destined: {VOCAB[field.prophecy_target]}")
    print(f"\n  {VOCAB[seed]}")

    chain = [seed]
    forbidden = {seed}

    for step in range(STEPS):
        field.update_chambers(step)
        field.prophecy_age += 1

        # learned logits from step-specific weights
        logits = model.forward_step(chain, step)

        # Dario field overlay — the live part, the part that breathes
        logits = field.overlay(logits, chain, step)

        # mask forbidden (no repeats allowed in this party)
        for f_id in forbidden:
            logits[f_id] = -1e9

        # top-k sampling. k=12 because 12 is our number.
        probs = softmax(logits)
        indexed = sorted(enumerate(probs), key=lambda x: -x[1])[:12]
        total = sum(max(0, p) for _, p in indexed) + 0.001
        r = random.random() * total
        pick = indexed[0][0]
        for idx, p in indexed:
            r -= max(0, p)
            if r <= 0:
                pick = idx
                break

        chain.append(pick)
        forbidden.add(pick)

        # update field — Hebbian learning, live, in-generation
        if len(chain) >= 2:
            field.update_cooc(chain[-2], pick)
            cat = word_category(pick)
            field.destiny[cat] = 0.3 + 0.7 * field.destiny[cat]

        if step > 7:
            field.trauma = min(1, field.trauma + 0.1)
        field.trauma *= 0.97

        marker = "  *" if step == STEPS - 1 else "   "
        print(f"{marker}{VOCAB[pick]}")

    fulfilled = field.prophecy_target in chain
    cats = len(set(word_category(w) for w in chain))
    print(f"\n  drift {cats}/8 · prophecy {'fulfilled' if fulfilled else 'unfulfilled'}")
    return chain


# ===================================================================
# MAIN — the entry point. the beginning of resonance.
# ===================================================================

def main():
    args = sys.argv[1:]
    train_path = None
    load_path = None
    save_path = None
    train_steps = 5000
    lr = 3e-4
    text = None

    i = 0
    while i < len(args):
        if args[i] == "--train" and i+1 < len(args):
            train_path = args[i+1]; i += 2
        elif args[i] == "--load" and i+1 < len(args):
            load_path = args[i+1]; i += 2
        elif args[i] == "--save" and i+1 < len(args):
            save_path = args[i+1]; i += 2
        elif args[i] == "--steps" and i+1 < len(args):
            train_steps = int(args[i+1]); i += 2
        elif args[i] == "--lr" and i+1 < len(args):
            lr = float(args[i+1]); i += 2
        else:
            text = " ".join(args[i:]); break

    model = MicroReasoner()
    field = DarioField()

    print()
    print(f"  microreasoning — 1984 words, {STEPS} steps, Dario Equation")
    print(f"  {model.param_count():,} trainable params")
    print()

    if load_path and os.path.exists(load_path):
        model.load(load_path)

    if train_path:
        train(model, train_path, train_steps, lr)
        if save_path:
            model.save(save_path)

    if text:
        run_chain(model, field, text)
    elif not train_path:
        # interactive mode. type a word. get 12 back. that's the deal.
        while True:
            try:
                text = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            run_chain(model, field, text)

    if save_path and not train_path:
        model.save(save_path)


if __name__ == "__main__":
    main()
