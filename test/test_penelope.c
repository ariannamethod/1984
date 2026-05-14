/*
 * test_penelope.c - v7 unit tests for penelope.c
 *
 * The older test file described the pre-v7 step architecture. These tests
 * assert the current PEN7 body: 8 transformer layers, BPE-in / word-out, and
 * the Dario overlay.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define main penelope_main
#include "../penelope.c"
#undef main

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
    } \
} while (0)

#define CHECK_EQ(a, b, msg) CHECK((a) == (b), msg)
#define CHECK_NEAR(a, b, eps, msg) CHECK(fabsf((a) - (b)) < (eps), msg)

static void init_runtime_tables(void) {
    init_vocab_lens();
    init_vocab_bpe();
    init_bpe_decode();
    init_ext_vocab();
}

static void test_constants(void) {
    CHECK_EQ(NWORDS, 1984, "NWORDS == 1984");
    CHECK_EQ(DIM, 448, "DIM == 448");
    CHECK_EQ(HDIM, 896, "HDIM == 896");
    CHECK_EQ(N_HEADS, 7, "N_HEADS == 7");
    CHECK_EQ(HEAD_DIM, 64, "HEAD_DIM == 64");
    CHECK_EQ(N_LAYERS, 8, "N_LAYERS == 8");
    CHECK_EQ(MAX_SEQ, 256, "MAX_SEQ == 256");
    CHECK_EQ(BPE_VOCAB, 2048, "BPE_VOCAB == 2048");
    CHECK_EQ(BPE_MERGES, 1792, "BPE_MERGES == 1792");
}

static void test_param_count(void) {
    int layer = layer_param_count();
    int expected_layer = DIM + DIM*DIM*5 + 2 + DIM + DIM*HDIM*2 + HDIM*DIM;
    int expected_global = BPE_VOCAB*DIM + MAX_SEQ*DIM + DIM + BPE_VOCAB*DIM;
    CHECK_EQ(layer, expected_layer, "layer_param_count matches v7 formula");
    CHECK_EQ(total_param_count(), expected_global + N_LAYERS * expected_layer,
             "total_param_count matches v7 formula");
    CHECK_EQ(total_param_count(), 19619280, "total params == 19,619,280");
}

static void test_vocab_and_categories(void) {
    CHECK(strcmp(VOCAB[0], "flesh") == 0, "VOCAB[0] == flesh");
    CHECK(strcmp(VOCAB[8], "heart") == 0, "VOCAB[8] == heart");
    CHECK(strcmp(VOCAB[200], "fear") == 0, "VOCAB[200] == fear");
    CHECK(strcmp(VOCAB[201], "love") == 0, "VOCAB[201] == love");
    CHECK(VOCAB[1983] != NULL && strlen(VOCAB[1983]) > 0, "VOCAB[1983] exists");
    CHECK_EQ(find_word("heart"), 8, "find_word heart");
    CHECK_EQ(find_word("love"), 201, "find_word love");
    CHECK_EQ(find_word("nonexistent"), -1, "find_word missing");

    CHECK_EQ(word_category(0), 0, "body category start");
    CHECK_EQ(word_category(99), 0, "body category end");
    CHECK_EQ(word_category(100), 1, "nature category");
    CHECK_EQ(word_category(200), 2, "emotion category");
    CHECK_EQ(word_category(300), 3, "time category");
    CHECK_EQ(word_category(350), 4, "society category");
    CHECK_EQ(word_category(450), 5, "abstract category");
    CHECK_EQ(word_category(550), 6, "action category");
    CHECK_EQ(word_category(650), 7, "macro-other category");
    CHECK_EQ(word_category(1983), 7, "last word category");
}

static void test_bpe(void) {
    int ids[256], ids2[256];
    int n = bpe_encode("", ids, 256);
    CHECK_EQ(n, 0, "empty BPE encodes to zero tokens");

    n = bpe_encode("a", ids, 256);
    CHECK_EQ(n, 1, "single byte BPE length");
    CHECK_EQ(ids[0], 97, "single byte BPE id");

    n = bpe_encode("HELLO", ids, 256);
    int n2 = bpe_encode("hello", ids2, 256);
    CHECK_EQ(n, n2, "BPE lowercases length");
    for (int i = 0; i < n && i < n2; i++) CHECK_EQ(ids[i], ids2[i], "BPE lowercases ids");

    n = bpe_encode("the heart of darkness", ids, 256);
    CHECK(n > 0, "BPE text produces tokens");
    CHECK(n < (int)strlen("the heart of darkness"), "BPE compresses merged text");
    for (int i = 0; i < n; i++) CHECK(ids[i] >= 0 && ids[i] < BPE_VOCAB, "BPE id in range");

    for (int m = 0; m < BPE_MERGES; m++) {
        CHECK(BPE_TABLE[m][0] >= 0 && BPE_TABLE[m][0] < 256 + m, "merge left valid");
        CHECK(BPE_TABLE[m][1] >= 0 && BPE_TABLE[m][1] < 256 + m, "merge right valid");
    }
}

static void test_vocab_bpe_and_extended_vocab(void) {
    for (int v = 0; v < NWORDS; v++) {
        CHECK(vocab_bpe_len[v] > 0, "vocab BPE has tokens");
        CHECK(vocab_bpe_len[v] <= 16, "vocab BPE token cap");
    }

    int direct[16];
    int n = bpe_encode("heart", direct, 16);
    CHECK_EQ(n, vocab_bpe_len[8], "heart BPE length matches precompute");
    for (int i = 0; i < n; i++) CHECK_EQ(direct[i], vocab_bpe[8][i], "heart BPE ids match");

    CHECK(ext_vocab_n >= NWORDS, "extended vocab contains hardcoded words");
    CHECK(ext_vocab_n <= MAX_EXT_VOCAB, "extended vocab under cap");
    CHECK(strcmp(ext_vocab[0].word, VOCAB[0]) == 0, "extended vocab starts with hardcoded vocab");
    CHECK(ext_vocab_find("heart") >= 0, "extended vocab can find hardcoded word");
}

static void test_math_helpers(void) {
    float W[] = {1, 2, 3, 4, 5, 6};
    float x3[] = {1, 1, 1};
    float out2[2];
    matmul_mv(W, x3, out2, 2, 3);
    CHECK_NEAR(out2[0], 6.0f, 1e-5f, "matmul_mv row 0");
    CHECK_NEAR(out2[1], 15.0f, 1e-5f, "matmul_mv row 1");

    float x2[] = {1, 1};
    float out3[3];
    matmul_mtv(W, x2, out3, 2, 3);
    CHECK_NEAR(out3[0], 5.0f, 1e-5f, "matmul_mtv col 0");
    CHECK_NEAR(out3[1], 7.0f, 1e-5f, "matmul_mtv col 1");
    CHECK_NEAR(out3[2], 9.0f, 1e-5f, "matmul_mtv col 2");

    float rx[] = {3, 4};
    float rg[] = {1, 1};
    float rout[2];
    rmsnorm(rx, rg, rout, 2);
    float inv = 1.0f / sqrtf(25.0f / 2.0f + 1e-5f);
    CHECK_NEAR(rout[0], 3.0f * inv, 1e-4f, "rmsnorm first");
    CHECK_NEAR(rout[1], 4.0f * inv, 1e-4f, "rmsnorm second");

    float sm_in[] = {1, 2, 3};
    float sm_out[3];
    softmax_v(sm_in, sm_out, 3);
    CHECK_NEAR(sm_out[0] + sm_out[1] + sm_out[2], 1.0f, 1e-5f, "softmax sums to one");
    CHECK(sm_out[0] < sm_out[1] && sm_out[1] < sm_out[2], "softmax preserves order");
    CHECK_NEAR(siluf(0.0f), 0.0f, 1e-5f, "silu zero");
    CHECK_NEAR(siluf(1.0f), 0.7311f, 1e-3f, "silu one");
}

static void test_model_init_forward(void) {
    Model m;
    model_init(&m);

    CHECK(m.tok_emb != NULL, "tok_emb allocated");
    CHECK(m.pos_emb != NULL, "pos_emb allocated");
    CHECK(m.final_norm != NULL, "final_norm allocated");
    CHECK(m.lm_head != NULL, "lm_head allocated");
    CHECK(m.layers[0].wq != NULL, "layer wq allocated");
    CHECK(m.layers[0].wr != NULL, "layer wr allocated");
    CHECK(m.layers[0].w_gate != NULL, "layer SwiGLU gate allocated");
    CHECK_NEAR(m.layers[0].gate[0], 0.0f, 1e-6f, "RRPRAM/QKV gate 0 init");
    CHECK_NEAR(m.layers[0].gate[1], 0.0f, 1e-6f, "RRPRAM/QKV gate 1 init");
    CHECK_NEAR(m.final_norm[0], 1.0f, 1e-6f, "final norm init");
    CHECK_NEAR(m.layers[0].attn_norm[0], 1.0f, 1e-6f, "attn norm init");
    CHECK_NEAR(m.layers[0].ffn_norm[0], 1.0f, 1e-6f, "ffn norm init");

    int has_nonzero = 0;
    for (int i = 0; i < 128; i++) if (m.tok_emb[i] != 0.0f) has_nonzero = 1;
    CHECK(has_nonzero, "tok_emb initialized nonzero");

    int ids[64];
    int n = bpe_encode("flesh love", ids, 64);
    float logits[BPE_VOCAB];
    forward(&m, ids, n, logits);
    float mn = logits[0], mx = logits[0];
    for (int i = 1; i < BPE_VOCAB; i++) {
        if (logits[i] < mn) mn = logits[i];
        if (logits[i] > mx) mx = logits[i];
    }
    CHECK(mx > mn, "forward produces non-flat BPE logits");

    float word_scores[NWORDS];
    bpe_logits_to_word_scores(logits, word_scores, NWORDS);
    mn = word_scores[0];
    mx = word_scores[0];
    for (int i = 1; i < NWORDS; i++) {
        if (word_scores[i] < mn) mn = word_scores[i];
        if (word_scores[i] > mx) mx = word_scores[i];
    }
    CHECK(mx > mn, "BPE logits convert to non-flat word scores");

    model_free(&m);
}

static void test_save_load_pen7(void) {
    Model m1, m2;
    model_init(&m1);
    model_init(&m2);

    const char *path = "/tmp/test_penelope_pen7.bin";
    model_save(&m1, path);
    CHECK(model_load(&m2, path) == 1, "model_load accepts saved PEN7");

    CHECK_EQ(memcmp(m1.tok_emb, m2.tok_emb, BPE_VOCAB * DIM * sizeof(float)), 0,
             "save/load tok_emb exact");
    CHECK_EQ(memcmp(m1.pos_emb, m2.pos_emb, MAX_SEQ * DIM * sizeof(float)), 0,
             "save/load pos_emb exact");
    CHECK_EQ(memcmp(m1.lm_head, m2.lm_head, BPE_VOCAB * DIM * sizeof(float)), 0,
             "save/load lm_head exact");
    CHECK_EQ(memcmp(m1.layers[0].wr, m2.layers[0].wr, DIM * DIM * sizeof(float)), 0,
             "save/load layer 0 wr exact");
    CHECK_EQ(memcmp(m1.layers[N_LAYERS-1].w_down, m2.layers[N_LAYERS-1].w_down,
                    HDIM * DIM * sizeof(float)), 0,
             "save/load final layer w_down exact");

    FILE *f = fopen(path, "rb");
    CHECK(f != NULL, "saved file opens");
    if (f) {
        int header[8];
        fread(header, sizeof(int), 8, f);
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fclose(f);
        CHECK_EQ(header[0], 0x50454E37, "PEN7 magic");
        CHECK_EQ(header[1], BPE_VOCAB, "PEN7 BPE vocab");
        CHECK_EQ(header[2], NWORDS, "PEN7 word vocab");
        CHECK_EQ(header[3], DIM, "PEN7 dim");
        CHECK_EQ(header[4], HDIM, "PEN7 hdim");
        CHECK_EQ(header[5], N_HEADS, "PEN7 heads");
        CHECK_EQ(header[6], N_LAYERS, "PEN7 layers");
        CHECK_EQ(header[7], MAX_SEQ, "PEN7 max seq");
        CHECK_EQ((int)sz, 32 + total_param_count() * 4, "PEN7 file size");
    }

    remove(path);
    model_free(&m1);
    model_free(&m2);
}

static void test_dario_overlay(void) {
    cooc_n = 0;
    big_n = 0;
    memset(destiny, 0, sizeof(destiny));
    memset(chambers, 0, sizeof(chambers));
    trauma = 0;
    prophecy_target = -1;
    prophecy_age = 0;

    cooc_update(0, 201);
    cooc_update(0, 201);
    CHECK_NEAR(cooc_get(0, 201), 2.0f, 1e-5f, "cooc count");
    CHECK_NEAR(cooc_get(201, 0), 2.0f, 1e-5f, "cooc symmetry");

    float scores[NWORDS];
    memset(scores, 0, sizeof(scores));
    int chain[] = {0};
    dario_overlay(scores, chain, 1, 0);
    CHECK(scores[201] > 0.0f, "Hebbian overlay boosts love after flesh");

    prophecy_target = 100;
    prophecy_age = 8;
    float before = scores[100];
    dario_overlay(scores, chain, 1, 1);
    CHECK(scores[100] > before, "prophecy overlay boosts target");

    update_chambers(0);
    update_chambers(4);
    update_chambers(7);
    float csum = 0;
    for (int i = 0; i < NCH; i++) csum += chambers[i];
    CHECK(csum > 0.0f, "chambers update during overlay");
}

int main(void) {
    srand(42);
    init_runtime_tables();

    printf("\n  penelope.c v7 test suite\n");
    printf("  =======================================\n\n");

    test_constants();
    test_param_count();
    test_vocab_and_categories();
    test_bpe();
    test_vocab_bpe_and_extended_vocab();
    test_math_helpers();
    test_model_init_forward();
    test_save_load_pen7();
    test_dario_overlay();

    printf("\n  =======================================\n");
    printf("  %d passed, %d failed\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
