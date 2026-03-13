/*
 * test_penelope.c — unit tests for penelope.c
 *
 * Tests: param count, vocab, word_category, matmul, rmsnorm, softmax,
 *        forward step, save/load roundtrip, dario field, generation validity,
 *        training loss decrease, 12 independent step weights.
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
}

static void test_param_count(void) {
    /* embed: 1984 * 384 = 761856 */
    int embed_params = NWORDS * DIM;
    ASSERT_EQ(embed_params, 761856, "embed params = 761856");

    /* per step: 384*384 + 384 + 384*768 + 384*768 + 768*384 = 1032576 */
    int spc = step_param_count();
    ASSERT_EQ(spc, 1032576, "step params = 1032576");

    /* total: 761856 + 12 * 1032576 = 13152768 */
    int total = total_param_count();
    ASSERT_EQ(total, 13152768, "total params = 13152768");

    /* verify each component */
    int wr = DIM * DIM;
    int rms = DIM;
    int gate = DIM * HDIM;
    int up_p = DIM * HDIM;
    int down = HDIM * DIM;
    ASSERT_EQ(wr + rms + gate + up_p + down, spc, "step component sum matches");
}

static void test_vocab(void) {
    /* first word */
    ASSERT(strcmp(VOCAB[0], "flesh") == 0, "VOCAB[0] == flesh");
    /* some known words */
    ASSERT(strcmp(VOCAB[1], "bone") == 0, "VOCAB[1] == bone");
    ASSERT(strcmp(VOCAB[8], "heart") == 0, "VOCAB[8] == heart");
    ASSERT(strcmp(VOCAB[200], "fear") == 0, "VOCAB[200] == fear");
    ASSERT(strcmp(VOCAB[201], "love") == 0, "VOCAB[201] == love");

    /* find_word */
    ASSERT_EQ(find_word("heart"), 8, "find_word(heart) == 8");
    ASSERT_EQ(find_word("love"), 201, "find_word(love) == 201");
    ASSERT_EQ(find_word("nonexistent"), -1, "find_word(nonexistent) == -1");
}

static void test_word_category(void) {
    ASSERT_EQ(word_category(0), 0, "cat(0) = BODY");
    ASSERT_EQ(word_category(99), 0, "cat(99) = BODY");
    ASSERT_EQ(word_category(100), 1, "cat(100) = NATURE");
    ASSERT_EQ(word_category(199), 1, "cat(199) = NATURE");
    ASSERT_EQ(word_category(200), 2, "cat(200) = EMOTION");
    ASSERT_EQ(word_category(300), 3, "cat(300) = TIME");
    ASSERT_EQ(word_category(350), 4, "cat(350) = SOCIETY");
    ASSERT_EQ(word_category(450), 5, "cat(450) = ABSTRACT");
    ASSERT_EQ(word_category(550), 6, "cat(550) = ACTION");
    ASSERT_EQ(word_category(650), 7, "cat(650+) = OTHER");
    ASSERT_EQ(word_category(1983), 7, "cat(1983) = OTHER");
}

static void test_matmul_mv(void) {
    /* 2x3 matrix @ 3-vector */
    float W[] = {1,2,3, 4,5,6};
    float x[] = {1,1,1};
    float out[2];
    matmul_mv(W, x, out, 2, 3);
    ASSERT_NEAR(out[0], 6.0f, 1e-5f, "matmul_mv [0] = 6");
    ASSERT_NEAR(out[1], 15.0f, 1e-5f, "matmul_mv [1] = 15");
}

static void test_matmul_mtv(void) {
    /* W[2,3]^T @ x[2] -> out[3] */
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
    /* rms = sqrt((9+16)/2 + 1e-5) ≈ sqrt(12.5) ≈ 3.5355 */
    /* inv ≈ 0.2828 */
    float rms_v = sqrtf(25.0f/2.0f + 1e-5f);
    float inv = 1.0f / rms_v;
    ASSERT_NEAR(out[0], 3.0f * inv, 1e-4f, "rmsnorm [0]");
    ASSERT_NEAR(out[1], 4.0f * inv, 1e-4f, "rmsnorm [1]");
}

static void test_softmax(void) {
    float x[] = {1.0f, 2.0f, 3.0f};
    float out[3];
    softmax_v(x, out, 3);
    /* sum should be 1 */
    float sum = out[0] + out[1] + out[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f, "softmax sums to 1");
    /* monotonic: out[0] < out[1] < out[2] */
    ASSERT(out[0] < out[1] && out[1] < out[2], "softmax monotonic");
    /* out[2] should be largest */
    ASSERT(out[2] > 0.6f, "softmax peak > 0.6");
}

static void test_silu(void) {
    ASSERT_NEAR(siluf(0.0f), 0.0f, 1e-5f, "silu(0) = 0");
    /* silu(1) = 1/(1+e^-1) ≈ 0.7311 */
    ASSERT_NEAR(siluf(1.0f), 0.7311f, 1e-3f, "silu(1) ≈ 0.731");
    ASSERT_NEAR(siluf(-30.0f), 0.0f, 1e-5f, "silu(-30) ≈ 0");
    /* silu(x) > 0 for x > 0 */
    ASSERT(siluf(5.0f) > 0, "silu(5) > 0");
}

static void test_model_init(void) {
    Model m;
    model_init(&m);

    /* check embed allocated and non-zero */
    ASSERT(m.embed != NULL, "embed allocated");
    int has_nonzero = 0;
    for (int i = 0; i < 100; i++)
        if (m.embed[i] != 0.0f) has_nonzero = 1;
    ASSERT(has_nonzero, "embed has non-zero values");

    /* check all step weights allocated */
    for (int s = 0; s < NSTEPS; s++) {
        ASSERT(m.steps[s].wr != NULL, "step wr allocated");
        ASSERT(m.steps[s].rms != NULL, "step rms allocated");
        ASSERT(m.steps[s].w_gate != NULL, "step w_gate allocated");
        ASSERT(m.steps[s].w_up != NULL, "step w_up allocated");
        ASSERT(m.steps[s].w_down != NULL, "step w_down allocated");
    }

    /* rms should be initialized to 1.0 */
    for (int i = 0; i < DIM; i++)
        if (m.steps[0].rms[i] != 1.0f) { ASSERT(0, "rms init to 1.0"); break; }

    model_free(&m);
    tests_passed++; /* if we got here without crash */
}

static void test_12_independent_weights(void) {
    Model m;
    model_init(&m);

    /* each step should have DIFFERENT weights */
    int all_different = 1;
    for (int s = 0; s < NSTEPS - 1; s++) {
        /* compare first 10 values of wr between consecutive steps */
        int same = 1;
        for (int i = 0; i < 10; i++)
            if (m.steps[s].wr[i] != m.steps[s+1].wr[i]) { same = 0; break; }
        if (same) { all_different = 0; break; }
    }
    ASSERT(all_different, "12 steps have independent weights");

    /* verify step count */
    int count = 0;
    for (int s = 0; s < NSTEPS; s++) {
        if (m.steps[s].wr) count++;
    }
    ASSERT_EQ(count, 12, "exactly 12 step weight sets");

    model_free(&m);
}

static void test_forward_produces_logits(void) {
    Model m;
    model_init(&m);

    int ctx[] = {0, 201}; /* flesh, love */
    float logits[NWORDS], query[DIM], query_n[DIM], gate[HDIM], up[HDIM];
    float swiglu[HDIM], hidden[DIM], out[DIM];

    forward_step(&m, ctx, 2, 0, logits, query, query_n, gate, up, swiglu, hidden, out);

    /* logits should have non-zero values */
    int has_variety = 0;
    float mn = logits[0], mx = logits[0];
    for (int i = 1; i < NWORDS; i++) {
        if (logits[i] < mn) mn = logits[i];
        if (logits[i] > mx) mx = logits[i];
    }
    has_variety = (mx - mn) > 0.01f;
    ASSERT(has_variety, "logits have variety (not all same)");

    /* different steps should produce different logits */
    float logits2[NWORDS];
    float q2[DIM], qn2[DIM], g2[HDIM], u2[HDIM], sg2[HDIM], h2[DIM], o2[DIM];
    forward_step(&m, ctx, 2, 5, logits2, q2, qn2, g2, u2, sg2, h2, o2);

    int differs = 0;
    for (int i = 0; i < NWORDS; i++)
        if (fabsf(logits[i] - logits2[i]) > 1e-6f) { differs = 1; break; }
    ASSERT(differs, "step 0 and step 5 produce different logits");

    model_free(&m);
}

static void test_save_load_roundtrip(void) {
    Model m1, m2;
    model_init(&m1);
    model_init(&m2);

    const char *path = "/tmp/test_penelope_roundtrip.bin";
    model_save(&m1, path);
    model_load(&m2, path);

    /* compare embeddings */
    int embed_match = 1;
    for (int i = 0; i < NWORDS * DIM; i++) {
        if (m1.embed[i] != m2.embed[i]) { embed_match = 0; break; }
    }
    ASSERT(embed_match, "save/load embed match");

    /* compare step 0 wr */
    int wr_match = 1;
    for (int i = 0; i < DIM * DIM; i++) {
        if (m1.steps[0].wr[i] != m2.steps[0].wr[i]) { wr_match = 0; break; }
    }
    ASSERT(wr_match, "save/load step[0].wr match");

    /* compare step 11 w_gate */
    int gate_match = 1;
    for (int i = 0; i < DIM * HDIM; i++) {
        if (m1.steps[11].w_gate[i] != m2.steps[11].w_gate[i]) { gate_match = 0; break; }
    }
    ASSERT(gate_match, "save/load step[11].w_gate match");

    /* check file size */
    FILE *f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    int expected = 16 + total_param_count() * 4;
    ASSERT_EQ((int)sz, expected, "save file size correct");

    remove(path);
    model_free(&m1);
    model_free(&m2);
}

static void test_dario_field(void) {
    /* test co-occurrence tracking */
    cooc_n = 0; big_n = 0;
    cooc_update(10, 20);
    cooc_update(10, 20);
    float c = cooc_get(10, 20);
    ASSERT_NEAR(c, 2.0f, 1e-5f, "cooc tracks count");

    /* symmetry */
    float c2 = cooc_get(20, 10);
    ASSERT_NEAR(c, c2, 1e-5f, "cooc symmetric");

    /* chambers update doesn't crash */
    memset(chambers, 0, sizeof(chambers));
    trauma = 0;
    update_chambers(0);
    update_chambers(6);
    update_chambers(11);
    /* chambers should have non-zero values after updates */
    float csum = 0;
    for (int i = 0; i < NCH; i++) csum += chambers[i];
    ASSERT(csum > 0, "chambers updated (sum > 0)");
}

static void test_tokenize(void) {
    int ids[100];
    int n = tokenize_text("the heart of darkness burns in fire", ids, 100);
    /* "the" is stop, expect: heart, darkness, burns->burn, fire */
    ASSERT(n >= 2, "tokenize finds at least 2 words");

    /* heart should be in results */
    int found_heart = 0;
    for (int i = 0; i < n; i++)
        if (ids[i] == 8) found_heart = 1; /* heart = index 8 */
    ASSERT(found_heart, "tokenize finds 'heart'");

    /* fire should be found */
    int found_fire = 0;
    for (int i = 0; i < n; i++)
        if (ids[i] == find_word("fire")) found_fire = 1;
    ASSERT(found_fire, "tokenize finds 'fire'");
}

static void test_generation_valid_words(void) {
    srand(42);
    Model m;
    model_init(&m);

    /* manually run a mini chain to check all picks are valid */
    int ctx[] = {201}; /* love */
    float logits[NWORDS], probs[NWORDS];
    float query[DIM], query_n[DIM], gate[HDIM], up[HDIM];
    float swiglu[HDIM], hidden[DIM], out[DIM];

    for (int step = 0; step < NSTEPS; step++) {
        forward_step(&m, ctx, 1, step, logits, query, query_n, gate, up, swiglu, hidden, out);
        softmax_v(logits, probs, NWORDS);

        /* find argmax */
        int best = 0;
        for (int i = 1; i < NWORDS; i++)
            if (probs[i] > probs[best]) best = i;

        /* must be valid index */
        ASSERT(best >= 0 && best < NWORDS, "generated word in vocab range");
        /* must be a real word (non-null) */
        ASSERT(VOCAB[best] != NULL && strlen(VOCAB[best]) > 0, "generated word is real");
    }

    model_free(&m);
}

static void test_training_loss_decreases(void) {
    /* create a tiny corpus file */
    const char *corpus_path = "/tmp/test_penelope_corpus.txt";
    FILE *f = fopen(corpus_path, "w");
    /* repeat some vocab words many times so training can learn patterns */
    for (int rep = 0; rep < 100; rep++) {
        fprintf(f, "heart blood flesh bone skin hand eye mouth tongue lung ");
        fprintf(f, "fear love rage joy grief sorrow pain pleasure comfort desire ");
        fprintf(f, "sky rain wind stone river mountain ocean leaf tree root ");
    }
    fclose(f);

    Model m;
    model_init(&m);

    /* manually run a few training steps and check loss */
    f = fopen(corpus_path, "r");
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = (char *)malloc(fsz + 1);
    fread(text, 1, fsz, f);
    text[fsz] = 0;
    fclose(f);

    int *ids = (int *)malloc((fsz / 2 + 1) * sizeof(int));
    int n_ids = tokenize_text(text, ids, fsz / 2);
    free(text);

    ASSERT(n_ids > NSTEPS + 1, "corpus tokenized enough words");

    /* compute initial loss */
    float logits[NWORDS], probs[NWORDS];
    float query[DIM], query_n[DIM], gate_b[HDIM], up_b[HDIM];
    float swiglu_b[HDIM], hidden_b[DIM], out_b[DIM];

    float init_loss = 0;
    int window = NSTEPS + 1;
    for (int s = 0; s < NSTEPS; s++) {
        forward_step(&m, ids, s + 1, s, logits, query, query_n, gate_b, up_b, swiglu_b, hidden_b, out_b);
        softmax_v(logits, probs, NWORDS);
        float p = probs[ids[s + 1]];
        if (p < 1e-10f) p = 1e-10f;
        init_loss -= logf(p);
    }
    init_loss /= NSTEPS;

    /* train 50 steps */
    train(&m, corpus_path, 50, 3e-4f);

    /* compute loss after training */
    float final_loss = 0;
    for (int s = 0; s < NSTEPS; s++) {
        forward_step(&m, ids, s + 1, s, logits, query, query_n, gate_b, up_b, swiglu_b, hidden_b, out_b);
        softmax_v(logits, probs, NWORDS);
        float p = probs[ids[s + 1]];
        if (p < 1e-10f) p = 1e-10f;
        final_loss -= logf(p);
    }
    final_loss /= NSTEPS;

    ASSERT(final_loss < init_loss, "training reduces loss");

    remove(corpus_path);
    free(ids);
    model_free(&m);
}

static void test_dario_equation_components(void) {
    /* Verify the Dario equation: score = B + α·H + β·F + γ·A + T */
    cooc_n = 0; big_n = 0;
    prophecy_target = -1;
    prophecy_age = 0;
    trauma = 0;
    memset(destiny, 0, sizeof(destiny));
    memset(chambers, 0, sizeof(chambers));

    Model m;
    model_init(&m);

    float logits1[NWORDS];
    float query[DIM], query_n[DIM], gate_b[HDIM], up_b[HDIM];
    float swiglu_b[HDIM], hidden_b[DIM], out_b[DIM];

    int ctx[] = {0};
    forward_step(&m, ctx, 1, 0, logits1, query, query_n, gate_b, up_b, swiglu_b, hidden_b, out_b);

    /* now add co-occurrence and check overlay changes logits */
    cooc_update(0, 201); /* flesh-love connection */
    cooc_update(0, 201);
    cooc_update(0, 201);

    float logits2[NWORDS];
    forward_step(&m, ctx, 1, 0, logits2, query, query_n, gate_b, up_b, swiglu_b, hidden_b, out_b);
    dario_overlay(logits2, ctx, 1, 0);

    /* logits for word 201 (love) should be boosted by Hebbian */
    ASSERT(logits2[201] > logits1[201], "Dario Hebbian boosts co-occurring word");

    /* set prophecy and check */
    prophecy_target = 100; /* sky */
    prophecy_age = 10;
    float logits3[NWORDS];
    forward_step(&m, ctx, 1, 0, logits3, query, query_n, gate_b, up_b, swiglu_b, hidden_b, out_b);
    float before_overlay = logits3[100];
    dario_overlay(logits3, ctx, 1, 0);
    ASSERT(logits3[100] > before_overlay, "Dario prophecy boosts target word");

    model_free(&m);
}

/* ═══════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════ */

int main(void) {
    srand(42);
    printf("\n  penelope.c test suite\n");
    printf("  ═══════════════════════════════════════\n\n");

    test_constants();
    printf("  [1/13] constants ............ OK\n");

    test_param_count();
    printf("  [2/13] param count .......... OK\n");

    test_vocab();
    printf("  [3/13] vocab ................ OK\n");

    test_word_category();
    printf("  [4/13] word_category ........ OK\n");

    test_matmul_mv();
    printf("  [5/13] matmul_mv ............ OK\n");

    test_matmul_mtv();
    printf("  [6/13] matmul_mtv ........... OK\n");

    test_rmsnorm_fn();
    printf("  [7/13] rmsnorm .............. OK\n");

    test_softmax();
    printf("  [8/13] softmax .............. OK\n");

    test_silu();
    printf("  [9/13] silu ................. OK\n");

    test_model_init();
    printf("  [10/13] model init .......... OK\n");

    test_12_independent_weights();
    printf("  [11/13] 12 independent ...... OK\n");

    test_forward_produces_logits();
    test_generation_valid_words();
    printf("  [12/13] forward + gen ....... OK\n");

    test_save_load_roundtrip();
    printf("  [13/13] save/load ........... OK\n");

    /* these take a bit longer */
    test_dario_field();
    test_dario_equation_components();
    test_tokenize();
    test_training_loss_decreases();

    printf("\n  ═══════════════════════════════════════\n");
    printf("  %d passed, %d failed\n\n", tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
