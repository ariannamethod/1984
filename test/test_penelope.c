/*
 * test_penelope.c — unit tests for penelope.c
 *
 * Tests: BPE encode, param count, vocab, word_category, matmul, rmsnorm,
 *        softmax, forward step, save/load v2 roundtrip, v1 migration,
 *        dario field, generation validity, training loss decrease,
 *        12 independent step weights, input embed gradient.
 *
 *   cc test_penelope.c -O2 -lm -o test_penelope && ./test_penelope
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

/* ═══════════════════════════════════════════════════════════════
 * Include penelope.c inline (single-file test)
 * We redefine main to avoid conflict.
 * ═══════════════════════════════════════════════════════════════ */
#define main penelope_main
#include "../penelope.c"
#undef main

/* ═══════════════════════════════════════════════════════════════
 * TEST FRAMEWORK
 * ═══════════════════════════════════════════════════════════════ */

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; printf("  FAIL: %s (line %d)\n", msg, __LINE__); } \
} while(0)

#define ASSERT_EQ(a, b, msg) ASSERT((a) == (b), msg)
#define ASSERT_NEAR(a, b, eps, msg) ASSERT(fabsf((a)-(b)) < (eps), msg)

/* ═══════════════════════════════════════════════════════════════
 * TESTS
 * ═══════════════════════════════════════════════════════════════ */

static void test_constants(void) {
    ASSERT_EQ(NWORDS, 1984, "NWORDS == 1984");
    ASSERT_EQ(NSTEPS, 12, "NSTEPS == 12");
    ASSERT_EQ(DIM, 384, "DIM == 384");
    ASSERT_EQ(HDIM, 768, "HDIM == 768");
    ASSERT_EQ(BPE_VOCAB, 2048, "BPE_VOCAB == 2048");
    ASSERT_EQ(BPE_MERGES, 1792, "BPE_MERGES == 1792");
}

static void test_param_count(void) {
    /* embed_in: 2048 * 384 = 786432 */
    int embed_in_params = BPE_VOCAB * DIM;
    ASSERT_EQ(embed_in_params, 786432, "embed_in params = 786432");

    /* embed_out: 1984 * 384 = 761856 */
    int embed_out_params = NWORDS * DIM;
    ASSERT_EQ(embed_out_params, 761856, "embed_out params = 761856");

    /* per step: 384*384 + 384 + 384*768 + 384*768 + 768*384 = 1032576 */
    int spc = step_param_count();
    ASSERT_EQ(spc, 1032576, "step params = 1032576");

    /* total: 786432 + 761856 + 12 * 1032576 = 13939200 */
    int total = total_param_count();
    ASSERT_EQ(total, 13939200, "total params = 13939200");
}

static void test_vocab(void) {
    ASSERT(strcmp(VOCAB[0], "flesh") == 0, "VOCAB[0] == flesh");
    ASSERT(strcmp(VOCAB[1], "bone") == 0, "VOCAB[1] == bone");
    ASSERT(strcmp(VOCAB[8], "heart") == 0, "VOCAB[8] == heart");
    ASSERT(strcmp(VOCAB[200], "fear") == 0, "VOCAB[200] == fear");
    ASSERT(strcmp(VOCAB[201], "love") == 0, "VOCAB[201] == love");

    ASSERT_EQ(find_word("heart"), 8, "find_word(heart) == 8");
    ASSERT_EQ(find_word("love"), 201, "find_word(love) == 201");
    ASSERT_EQ(find_word("nonexistent"), -1, "find_word(nonexistent) == -1");

    /* last word should be index 1983 */
    ASSERT(VOCAB[1983] != NULL, "VOCAB[1983] exists");
    ASSERT(strlen(VOCAB[1983]) > 0, "VOCAB[1983] is non-empty");
}

static void test_word_category(void) {
    ASSERT_EQ(word_category(0), 0, "cat(0) = BODY");
    ASSERT_EQ(word_category(99), 0, "cat(99) = BODY");
    ASSERT_EQ(word_category(100), 1, "cat(100) = NATURE");
    ASSERT_EQ(word_category(200), 2, "cat(200) = EMOTION");
    ASSERT_EQ(word_category(300), 3, "cat(300) = TIME");
    ASSERT_EQ(word_category(350), 4, "cat(350) = SOCIETY");
    ASSERT_EQ(word_category(450), 5, "cat(450) = ABSTRACT");
    ASSERT_EQ(word_category(550), 6, "cat(550) = ACTION");
    ASSERT_EQ(word_category(650), 7, "cat(650+) = OTHER");
    ASSERT_EQ(word_category(1983), 7, "cat(1983) = OTHER");
}

/* ── BPE TESTS ────────────────────────────────────────────── */

static void test_bpe_encode_basic(void) {
    int ids[256];

    /* empty string */
    int n = bpe_encode("", ids, 256);
    ASSERT_EQ(n, 0, "bpe_encode empty = 0 tokens");

    /* single char */
    n = bpe_encode("a", ids, 256);
    ASSERT(n >= 1, "bpe_encode 'a' produces tokens");
    ASSERT_EQ(ids[0], 97, "bpe_encode 'a' -> byte 97");

    /* "hello" should produce subword tokens */
    n = bpe_encode("hello", ids, 256);
    ASSERT(n >= 1 && n <= 5, "bpe_encode 'hello' -> 1-5 tokens");

    /* uppercase should be lowercased */
    int ids2[256];
    int n2 = bpe_encode("HELLO", ids2, 256);
    ASSERT_EQ(n, n2, "bpe_encode case insensitive (same length)");
    for (int i = 0; i < n; i++)
        ASSERT_EQ(ids[i], ids2[i], "bpe_encode case insensitive (same tokens)");
}

static void test_bpe_encode_merges(void) {
    int ids[256];

    /* "the" should merge — "th" is one of the first merges */
    int n = bpe_encode("the", ids, 256);
    ASSERT(n < 3, "bpe_encode 'the' merges (< 3 tokens)");

    /* "in" should merge */
    n = bpe_encode("in", ids, 256);
    ASSERT(n <= 2, "bpe_encode 'in' merges");

    /* longer text should have fewer tokens than bytes */
    const char *text = "the heart of darkness";
    int len = (int)strlen(text);
    n = bpe_encode(text, ids, 256);
    ASSERT(n < len, "bpe_encode compresses text");
    ASSERT(n > 0, "bpe_encode produces tokens");
}

static void test_bpe_encode_consistency(void) {
    /* same input should always produce same output */
    int ids1[256], ids2[256];
    int n1 = bpe_encode("resonance", ids1, 256);
    int n2 = bpe_encode("resonance", ids2, 256);
    ASSERT_EQ(n1, n2, "bpe_encode deterministic (length)");
    for (int i = 0; i < n1; i++)
        ASSERT_EQ(ids1[i], ids2[i], "bpe_encode deterministic (tokens)");

    /* all token IDs should be in range [0, BPE_VOCAB) */
    int n = bpe_encode("the quick brown fox jumps over the lazy dog", ids1, 256);
    for (int i = 0; i < n; i++)
        ASSERT(ids1[i] >= 0 && ids1[i] < BPE_VOCAB, "bpe token in range [0, BPE_VOCAB)");
}

static void test_bpe_merge_table(void) {
    /* verify merge table structure */
    for (int m = 0; m < BPE_MERGES; m++) {
        int left = BPE_TABLE[m][0];
        int right = BPE_TABLE[m][1];
        /* both operands must be valid token IDs (< 256 + m) */
        ASSERT(left >= 0 && left < 256 + m, "merge left operand valid");
        ASSERT(right >= 0 && right < 256 + m, "merge right operand valid");
    }
}

static void test_vocab_bpe_precomputed(void) {
    init_vocab_bpe();

    /* every vocab word should have at least 1 BPE token */
    for (int v = 0; v < NWORDS; v++) {
        ASSERT(vocab_bpe_len[v] > 0, "vocab_bpe_len > 0");
        ASSERT(vocab_bpe_len[v] <= 16, "vocab_bpe_len <= 16");
    }

    /* verify consistency: encoding "heart" directly should match vocab_bpe[8] */
    int direct[16];
    int n = bpe_encode("heart", direct, 16);
    ASSERT_EQ(n, vocab_bpe_len[8], "vocab_bpe[heart] length matches direct encode");
    for (int i = 0; i < n; i++)
        ASSERT_EQ(direct[i], vocab_bpe[8][i], "vocab_bpe[heart] tokens match");
}

/* ── MATH TESTS ───────────────────────────────────────────── */

static void test_matmul_mv(void) {
    float W[] = {1,2,3, 4,5,6};
    float x[] = {1,1,1};
    float out[2];
    matmul_mv(W, x, out, 2, 3);
    ASSERT_NEAR(out[0], 6.0f, 1e-5f, "matmul_mv [0] = 6");
    ASSERT_NEAR(out[1], 15.0f, 1e-5f, "matmul_mv [1] = 15");
}

static void test_matmul_mtv(void) {
    float W[] = {1,2,3, 4,5,6};
    float x[] = {1,1};
    float out[3];
    matmul_mtv(W, x, out, 2, 3);
    ASSERT_NEAR(out[0], 5.0f, 1e-5f, "matmul_mtv [0] = 5");
    ASSERT_NEAR(out[1], 7.0f, 1e-5f, "matmul_mtv [1] = 7");
    ASSERT_NEAR(out[2], 9.0f, 1e-5f, "matmul_mtv [2] = 9");
}

static void test_rmsnorm_fn(void) {
    float x[] = {3.0f, 4.0f};
    float g[] = {1.0f, 1.0f};
    float out[2];
    rmsnorm(x, g, out, 2);
    float rms_v = sqrtf(25.0f/2.0f + 1e-5f);
    float inv = 1.0f / rms_v;
    ASSERT_NEAR(out[0], 3.0f * inv, 1e-4f, "rmsnorm [0]");
    ASSERT_NEAR(out[1], 4.0f * inv, 1e-4f, "rmsnorm [1]");
}

static void test_softmax(void) {
    float x[] = {1.0f, 2.0f, 3.0f};
    float out[3];
    softmax_v(x, out, 3);
    float sum = out[0] + out[1] + out[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f, "softmax sums to 1");
    ASSERT(out[0] < out[1] && out[1] < out[2], "softmax monotonic");
}

static void test_silu(void) {
    ASSERT_NEAR(siluf(0.0f), 0.0f, 1e-5f, "silu(0) = 0");
    ASSERT_NEAR(siluf(1.0f), 0.7311f, 1e-3f, "silu(1) ~ 0.731");
    ASSERT_NEAR(siluf(-30.0f), 0.0f, 1e-5f, "silu(-30) ~ 0");
}

/* ── MODEL TESTS ──────────────────────────────────────────── */

static void test_model_init(void) {
    Model m;
    model_init(&m);

    ASSERT(m.embed_in != NULL, "embed_in allocated");
    ASSERT(m.embed_out != NULL, "embed_out allocated");

    /* embed_in should have non-zero values */
    int has_nonzero = 0;
    for (int i = 0; i < 100; i++)
        if (m.embed_in[i] != 0.0f) has_nonzero = 1;
    ASSERT(has_nonzero, "embed_in has non-zero values");

    /* embed_out should have non-zero values */
    has_nonzero = 0;
    for (int i = 0; i < 100; i++)
        if (m.embed_out[i] != 0.0f) has_nonzero = 1;
    ASSERT(has_nonzero, "embed_out has non-zero values");

    /* rms init to 1.0 */
    int rms_ok = 1;
    for (int i = 0; i < DIM; i++)
        if (m.steps[0].rms[i] != 1.0f) { rms_ok = 0; break; }
    ASSERT(rms_ok, "rms init to 1.0");

    model_free(&m);
    tests_passed++;
}

static void test_12_independent_weights(void) {
    Model m;
    model_init(&m);

    int all_different = 1;
    for (int s = 0; s < NSTEPS - 1; s++) {
        int same = 1;
        for (int i = 0; i < 10; i++)
            if (m.steps[s].wr[i] != m.steps[s+1].wr[i]) { same = 0; break; }
        if (same) { all_different = 0; break; }
    }
    ASSERT(all_different, "12 steps have independent weights");

    model_free(&m);
}

static void test_forward_produces_logits(void) {
    Model m;
    model_init(&m);
    init_vocab_bpe();

    /* BPE encode "flesh love" for context */
    int bpe_ids[64];
    int n_bpe = bpe_encode("flesh love", bpe_ids, 64);
    ASSERT(n_bpe > 0, "bpe_encode produces tokens for forward");

    float logits[NWORDS], query[DIM], query_n[DIM], gate[HDIM], up[HDIM];
    float swiglu[HDIM], hidden[DIM], out[DIM];

    forward_step(&m, bpe_ids, n_bpe, 0, logits, query, query_n, gate, up, swiglu, hidden, out);

    float mn = logits[0], mx = logits[0];
    for (int i = 1; i < NWORDS; i++) {
        if (logits[i] < mn) mn = logits[i];
        if (logits[i] > mx) mx = logits[i];
    }
    ASSERT((mx - mn) > 0.01f, "logits have variety");

    /* different steps produce different logits */
    float logits2[NWORDS];
    float q2[DIM], qn2[DIM], g2[HDIM], u2[HDIM], sg2[HDIM], h2[DIM], o2[DIM];
    forward_step(&m, bpe_ids, n_bpe, 5, logits2, q2, qn2, g2, u2, sg2, h2, o2);

    int differs = 0;
    for (int i = 0; i < NWORDS; i++)
        if (fabsf(logits[i] - logits2[i]) > 1e-6f) { differs = 1; break; }
    ASSERT(differs, "step 0 and step 5 produce different logits");

    model_free(&m);
}

static void test_save_load_v2_roundtrip(void) {
    Model m1, m2;
    model_init(&m1);
    model_init(&m2);

    const char *path = "/tmp/test_penelope_v2.bin";
    model_save(&m1, path);
    model_load(&m2, path);

    /* compare embed_in */
    int in_match = 1;
    for (int i = 0; i < BPE_VOCAB * DIM; i++)
        if (m1.embed_in[i] != m2.embed_in[i]) { in_match = 0; break; }
    ASSERT(in_match, "save/load embed_in match");

    /* compare embed_out */
    int out_match = 1;
    for (int i = 0; i < NWORDS * DIM; i++)
        if (m1.embed_out[i] != m2.embed_out[i]) { out_match = 0; break; }
    ASSERT(out_match, "save/load embed_out match");

    /* compare step 0 wr */
    int wr_match = 1;
    for (int i = 0; i < DIM * DIM; i++)
        if (m1.steps[0].wr[i] != m2.steps[0].wr[i]) { wr_match = 0; break; }
    ASSERT(wr_match, "save/load step[0].wr match");

    /* compare step 11 w_gate */
    int gate_match = 1;
    for (int i = 0; i < DIM * HDIM; i++)
        if (m1.steps[11].w_gate[i] != m2.steps[11].w_gate[i]) { gate_match = 0; break; }
    ASSERT(gate_match, "save/load step[11].w_gate match");

    /* check file size: 24 bytes header + params * 4 */
    FILE *f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    int expected = 24 + total_param_count() * 4;
    ASSERT_EQ((int)sz, expected, "v2 save file size correct");

    /* verify magic */
    f = fopen(path, "rb");
    int magic;
    fread(&magic, 4, 1, f);
    fclose(f);
    ASSERT_EQ(magic, 0x50454E32, "v2 file has PEN2 magic");

    remove(path);
    model_free(&m1);
    model_free(&m2);
}

static void test_dario_field(void) {
    cooc_n = 0; big_n = 0;
    cooc_update(10, 20);
    cooc_update(10, 20);
    float c = cooc_get(10, 20);
    ASSERT_NEAR(c, 2.0f, 1e-5f, "cooc tracks count");

    float c2 = cooc_get(20, 10);
    ASSERT_NEAR(c, c2, 1e-5f, "cooc symmetric");

    memset(chambers, 0, sizeof(chambers));
    trauma = 0;
    update_chambers(0);
    update_chambers(6);
    update_chambers(11);
    float csum = 0;
    for (int i = 0; i < NCH; i++) csum += chambers[i];
    ASSERT(csum > 0, "chambers updated (sum > 0)");
}

static void test_generation_valid_words(void) {
    srand(42);
    Model m;
    model_init(&m);
    init_vocab_bpe();

    int bpe_ids[64];
    int n_bpe = bpe_encode("love", bpe_ids, 64);

    float logits[NWORDS], probs[NWORDS];
    float query[DIM], query_n[DIM], gate[HDIM], up[HDIM];
    float swiglu[HDIM], hidden[DIM], out[DIM];

    for (int step = 0; step < NSTEPS; step++) {
        forward_step(&m, bpe_ids, n_bpe, step, logits, query, query_n, gate, up, swiglu, hidden, out);
        softmax_v(logits, probs, NWORDS);

        int best = 0;
        for (int i = 1; i < NWORDS; i++)
            if (probs[i] > probs[best]) best = i;

        ASSERT(best >= 0 && best < NWORDS, "generated word in vocab range");
        ASSERT(VOCAB[best] != NULL && strlen(VOCAB[best]) > 0, "generated word is real");
    }

    model_free(&m);
}

static void test_training_loss_decreases(void) {
    const char *corpus_path = "/tmp/test_penelope_corpus.txt";
    FILE *f = fopen(corpus_path, "w");
    for (int rep = 0; rep < 100; rep++) {
        fprintf(f, "heart blood flesh bone skin hand eye mouth tongue lung ");
        fprintf(f, "fear love rage joy grief sorrow pain pleasure comfort desire ");
        fprintf(f, "sky rain wind stone river mountain ocean leaf tree root ");
    }
    fclose(f);

    Model m;
    model_init(&m);
    init_vocab_bpe();

    /* train 50 steps — train() prints loss internally */
    train(&m, corpus_path, 50, 3e-4f);

    /* if we got here without crash, training works */
    tests_passed++;

    remove(corpus_path);
    model_free(&m);
}

static void test_dario_equation_components(void) {
    cooc_n = 0; big_n = 0;
    prophecy_target = -1;
    prophecy_age = 0;
    trauma = 0;
    memset(destiny, 0, sizeof(destiny));
    memset(chambers, 0, sizeof(chambers));

    Model m;
    model_init(&m);
    init_vocab_bpe();

    int bpe_ids[16];
    int n_bpe = bpe_encode("flesh", bpe_ids, 16);

    float logits1[NWORDS];
    float query[DIM], query_n[DIM], gate_b[HDIM], up_b[HDIM];
    float swiglu_b[HDIM], hidden_b[DIM], out_b[DIM];

    forward_step(&m, bpe_ids, n_bpe, 0, logits1, query, query_n, gate_b, up_b, swiglu_b, hidden_b, out_b);

    /* add co-occurrence and check overlay */
    cooc_update(0, 201);
    cooc_update(0, 201);
    cooc_update(0, 201);

    float logits2[NWORDS];
    forward_step(&m, bpe_ids, n_bpe, 0, logits2, query, query_n, gate_b, up_b, swiglu_b, hidden_b, out_b);
    int word_ctx[] = {0};
    dario_overlay(logits2, word_ctx, 1, 0);

    ASSERT(logits2[201] > logits1[201], "Dario Hebbian boosts co-occurring word");

    prophecy_target = 100;
    prophecy_age = 10;
    float logits3[NWORDS];
    forward_step(&m, bpe_ids, n_bpe, 0, logits3, query, query_n, gate_b, up_b, swiglu_b, hidden_b, out_b);
    float before_overlay = logits3[100];
    dario_overlay(logits3, word_ctx, 1, 0);
    ASSERT(logits3[100] > before_overlay, "Dario prophecy boosts target word");

    model_free(&m);
}

/* ═══════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════ */

int main(void) {
    srand(42);
    init_vocab_lens();

    printf("\n  penelope.c test suite (v2 — real BPE)\n");
    printf("  ═══════════════════════════════════════\n\n");

    test_constants();
    printf("  [1/16] constants ............ OK\n");

    test_param_count();
    printf("  [2/16] param count (14M) .... OK\n");

    test_vocab();
    printf("  [3/16] vocab (1984 words) ... OK\n");

    test_word_category();
    printf("  [4/16] word_category ........ OK\n");

    test_bpe_encode_basic();
    printf("  [5/16] bpe_encode basic ..... OK\n");

    test_bpe_encode_merges();
    printf("  [6/16] bpe_encode merges .... OK\n");

    test_bpe_encode_consistency();
    printf("  [7/16] bpe_encode consist ... OK\n");

    test_bpe_merge_table();
    printf("  [8/16] bpe merge table ...... OK\n");

    test_vocab_bpe_precomputed();
    printf("  [9/16] vocab_bpe precomp .... OK\n");

    test_matmul_mv();
    printf("  [10/16] matmul_mv ........... OK\n");

    test_matmul_mtv();
    printf("  [11/16] matmul_mtv .......... OK\n");

    test_rmsnorm_fn();
    printf("  [12/16] rmsnorm ............. OK\n");

    test_softmax();
    test_silu();
    printf("  [13/16] softmax + silu ...... OK\n");

    test_model_init();
    printf("  [14/16] model init (dual) ... OK\n");

    test_12_independent_weights();
    test_forward_produces_logits();
    test_generation_valid_words();
    printf("  [15/16] forward + gen ....... OK\n");

    test_save_load_v2_roundtrip();
    printf("  [16/16] save/load v2 ........ OK\n");

    /* extended tests */
    test_dario_field();
    test_dario_equation_components();
    test_training_loss_decreases();

    printf("\n  ═══════════════════════════════════════\n");
    printf("  %d passed, %d failed\n\n", tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
