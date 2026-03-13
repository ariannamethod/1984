#!/usr/bin/env julia
# test_penelope_jl.jl — comprehensive unit tests for penelope.jl
# Tests all functions to verify functional identity with Python/Rust/C versions.

include(joinpath(dirname(@__DIR__), "penelope.jl"))

using Test

passed = 0
failed = 0

macro check(name, expr)
    quote
        try
            result = $(esc(expr))
            if result
                global passed += 1
                println("  OK: ", $(esc(name)))
            else
                global failed += 1
                println("  FAIL: ", $(esc(name)))
            end
        catch e
            global failed += 1
            println("  FAIL: ", $(esc(name)), " — ", e)
        end
    end
end

println()
println("  penelope.jl unit test suite")
println("  ═══════════════════════════════════════")
println()

# ═══════════════════════════════════════════════════════════════
# 1. VOCABULARY & CONSTANTS
# ═══════════════════════════════════════════════════════════════
println("  --- Vocabulary & Constants ---")

@check "V matches VOCAB length" V == length(VOCAB)
@check "V == 1990" V == 1990
@check "STEPS == 12" STEPS == 12
@check "D == 384" D == 384
@check "M == 768" M == 768

@check "VOCAB[1] == flesh (first word)" VOCAB[1] == "flesh"
@check "VOCAB[2] == bone" VOCAB[2] == "bone"
@check "VOCAB[3] == blood" VOCAB[3] == "blood"
@check "VOCAB[end] == investment (last word)" VOCAB[end] == "investment"

@check "VOCAB_IDX has entries" length(VOCAB_IDX) > 0
@check "VOCAB_IDX[flesh] == 0" VOCAB_IDX["flesh"] == 0
@check "VOCAB_IDX[blood] == 2" VOCAB_IDX["blood"] == 2
@check "VOCAB_IDX[sky] == 100" VOCAB_IDX["sky"] == 100
@check "VOCAB_IDX[fear] == 200" VOCAB_IDX["fear"] == 200
@check "VOCAB_IDX[war] == 350" VOCAB_IDX["war"] == 350
@check "VOCAB_IDX[truth] == 398 (first occurrence)" VOCAB_IDX["truth"] == 398
@check "VOCAB_IDX[walk] == 550" VOCAB_IDX["walk"] == 550
@check "VOCAB_IDX[iron] == 650" VOCAB_IDX["iron"] == 650

# Duplicates: VOCAB_IDX should have first occurrence
@check "VOCAB_IDX maps to first occurrence" begin
    # 'sweat' appears at index 18 and 82, first = 18
    VOCAB[19] == "sweat" && VOCAB[83] == "sweat" && VOCAB_IDX["sweat"] == 18
end

@check "STOP contains expected words" all(w -> w in STOP, ["i", "the", "a", "and", "or", "but", "in", "on"])
@check "STOP does not contain vocab words" !("flesh" in STOP) && !("blood" in STOP)

# ═══════════════════════════════════════════════════════════════
# 2. MATH FUNCTIONS
# ═══════════════════════════════════════════════════════════════
println()
println("  --- Math Functions ---")

@check "_zeros returns zeros" begin
    z = _zeros(5)
    length(z) == 5 && all(x -> x == 0.0, z)
end

@check "_dot product" begin
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    abs(_dot(a, b) - 32.0) < 1e-10
end

@check "_dot of orthogonal vectors" begin
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    abs(_dot(a, b)) < 1e-10
end

@check "vadd" begin
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    r = vadd(a, b)
    r == [5.0, 7.0, 9.0]
end

@check "vsub" begin
    a = [4.0, 5.0, 6.0]
    b = [1.0, 2.0, 3.0]
    r = vsub(a, b)
    r == [3.0, 3.0, 3.0]
end

@check "vscale" begin
    a = [1.0, 2.0, 3.0]
    r = vscale(a, 2.0)
    r == [2.0, 4.0, 6.0]
end

@check "vscale by zero" begin
    a = [1.0, 2.0, 3.0]
    r = vscale(a, 0.0)
    all(x -> x == 0.0, r)
end

@check "matmul_mv identity" begin
    # 2x2 identity matrix stored flat: [1,0,0,1]
    W = [1.0, 0.0, 0.0, 1.0]
    x = [3.0, 7.0]
    r = matmul_mv(W, x, 2, 2)
    abs(r[1] - 3.0) < 1e-10 && abs(r[2] - 7.0) < 1e-10
end

@check "matmul_mv general" begin
    # [[1,2],[3,4]] @ [5,6] = [17, 39]
    W = [1.0, 2.0, 3.0, 4.0]
    x = [5.0, 6.0]
    r = matmul_mv(W, x, 2, 2)
    abs(r[1] - 17.0) < 1e-10 && abs(r[2] - 39.0) < 1e-10
end

@check "matmul_mv non-square" begin
    # [[1,2,3],[4,5,6]] @ [1,1,1] = [6, 15]
    W = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    x = [1.0, 1.0, 1.0]
    r = matmul_mv(W, x, 2, 3)
    abs(r[1] - 6.0) < 1e-10 && abs(r[2] - 15.0) < 1e-10
end

@check "matmul_mtv identity" begin
    W = [1.0, 0.0, 0.0, 1.0]
    x = [3.0, 7.0]
    r = matmul_mtv(W, x, 2, 2)
    abs(r[1] - 3.0) < 1e-10 && abs(r[2] - 7.0) < 1e-10
end

@check "matmul_mtv transpose" begin
    # W = [[1,2],[3,4]], W^T = [[1,3],[2,4]]
    # W^T @ [5,6] = [1*5+3*6, 2*5+4*6] = [23, 34]
    W = [1.0, 2.0, 3.0, 4.0]
    x = [5.0, 6.0]
    r = matmul_mtv(W, x, 2, 2)
    abs(r[1] - 23.0) < 1e-10 && abs(r[2] - 34.0) < 1e-10
end

@check "rmsnorm normalizes" begin
    x = [3.0, 4.0]
    g = [1.0, 1.0]
    r = rmsnorm(x, g, 2)
    # rms = sqrt((9+16)/2 + 1e-5) ≈ sqrt(12.50001)
    # inv = 1/sqrt(12.50001) ≈ 0.28284
    length(r) == 2 && abs(r[1] / r[2] - 0.75) < 1e-3
end

@check "rmsnorm with gain" begin
    x = [1.0, 1.0]
    g = [2.0, 3.0]
    r = rmsnorm(x, g, 2)
    # ratio should be g[1]/g[2] = 2/3
    abs(r[1] / r[2] - 2.0/3.0) < 1e-5
end

@check "_silu(0) == 0" abs(_silu(0.0)) < 1e-10
@check "_silu(large positive) ≈ x" abs(_silu(10.0) - 10.0) < 0.01
@check "_silu(large negative) == 0" _silu(-30.0) == 0.0
@check "_silu(1) ≈ 0.731" abs(_silu(1.0) - 1.0 / (1.0 + exp(-1.0))) < 1e-10

@check "_softmax sums to 1" begin
    x = [1.0, 2.0, 3.0]
    p = _softmax(x)
    abs(sum(p) - 1.0) < 1e-10
end

@check "_softmax preserves order" begin
    x = [1.0, 2.0, 3.0]
    p = _softmax(x)
    p[1] < p[2] < p[3]
end

@check "_softmax uniform input" begin
    x = [1.0, 1.0, 1.0]
    p = _softmax(x)
    all(v -> abs(v - 1.0/3.0) < 1e-10, p)
end

@check "_softmax handles large values" begin
    x = [1000.0, 1001.0, 1002.0]
    p = _softmax(x)
    abs(sum(p) - 1.0) < 1e-10
end

@check "_randn returns finite" begin
    vals = [_randn() for _ in 1:100]
    all(isfinite, vals)
end

@check "_randn has roughly zero mean" begin
    vals = [_randn() for _ in 1:10000]
    abs(sum(vals) / length(vals)) < 0.1
end

# ═══════════════════════════════════════════════════════════════
# 3. MODEL CONSTRUCTION
# ═══════════════════════════════════════════════════════════════
println()
println("  --- Model Construction ---")

@check "step_param_count" begin
    expected = D*D + D + D*M + D*M + M*D
    step_param_count() == expected
end

@check "step_param_count value" step_param_count() == 1032576

@check "StepWeights construction" begin
    sw = StepWeights()
    length(sw.wr) == D*D &&
    length(sw.rms) == D &&
    length(sw.w_gate) == D*M &&
    length(sw.w_up) == D*M &&
    length(sw.w_down) == M*D
end

@check "StepWeights rms initialized to 1" begin
    sw = StepWeights()
    all(x -> x == 1.0, sw.rms)
end

@check "Penelope construction" begin
    m = Penelope()
    length(m.embed) == V * D && length(m.steps) == STEPS
end

@check "param_count matches expected" begin
    m = Penelope()
    pc = param_count(m)
    expected = V * D + STEPS * step_param_count()
    pc == expected
end

@check "param_count value" begin
    m = Penelope()
    param_count(m) == 13155072
end

# ═══════════════════════════════════════════════════════════════
# 4. EMBEDDING & CONTEXT
# ═══════════════════════════════════════════════════════════════
println()
println("  --- Embedding & Context ---")

@check "get_embed returns D-dimensional vector" begin
    m = Penelope()
    e = get_embed(m, 0)
    length(e) == D
end

@check "get_embed different indices give different vectors" begin
    m = Penelope()
    e0 = get_embed(m, 0)
    e1 = get_embed(m, 1)
    e0 != e1
end

@check "get_embed returns correct slice" begin
    m = Penelope()
    e = get_embed(m, 5)
    # Should be embed[5*D+1 : 6*D]
    expected = m.embed[5*D+1 : 6*D]
    e == expected
end

@check "pool_context empty" begin
    m = Penelope()
    ctx = pool_context(m, Int[])
    length(ctx) == D && all(x -> x == 0.0, ctx)
end

@check "pool_context single word" begin
    m = Penelope()
    ctx = pool_context(m, [0])
    e = get_embed(m, 0)
    all(i -> abs(ctx[i] - e[i]) < 1e-10, 1:D)
end

@check "pool_context average" begin
    m = Penelope()
    ctx = pool_context(m, [0, 1])
    e0 = get_embed(m, 0)
    e1 = get_embed(m, 1)
    expected = vscale(vadd(e0, e1), 0.5)
    all(i -> abs(ctx[i] - expected[i]) < 1e-10, 1:D)
end

# ═══════════════════════════════════════════════════════════════
# 5. FORWARD PASS
# ═══════════════════════════════════════════════════════════════
println()
println("  --- Forward Pass ---")

@check "forward_step returns V logits" begin
    m = Penelope()
    logits = forward_step(m, [0], 0)
    length(logits) == V
end

@check "forward_step all finite" begin
    m = Penelope()
    logits = forward_step(m, [0, 1, 2], 0)
    all(isfinite, logits)
end

@check "forward_step different steps give different logits" begin
    m = Penelope()
    l0 = forward_step(m, [0], 0)
    l1 = forward_step(m, [0], 1)
    l0 != l1
end

@check "forward_step softmax probabilities sum to 1" begin
    m = Penelope()
    logits = forward_step(m, [0, 100, 200], 5)
    probs = _softmax(logits)
    abs(sum(probs) - 1.0) < 1e-8
end

# ═══════════════════════════════════════════════════════════════
# 6. WORD CATEGORY
# ═══════════════════════════════════════════════════════════════
println()
println("  --- Word Category ---")

@check "word_category BODY" word_category(0) == 0 && word_category(99) == 0
@check "word_category NATURE" word_category(100) == 1 && word_category(199) == 1
@check "word_category EMOTION" word_category(200) == 2 && word_category(299) == 2
@check "word_category TIME" word_category(300) == 3 && word_category(349) == 3
@check "word_category SOCIETY" word_category(350) == 4 && word_category(449) == 4
@check "word_category ABSTRACT" word_category(450) == 5 && word_category(549) == 5
@check "word_category ACTION" word_category(550) == 6 && word_category(649) == 6
@check "word_category MATERIAL+" word_category(650) == 7 && word_category(1983) == 7

# ═══════════════════════════════════════════════════════════════
# 7. TOKENIZER
# ═══════════════════════════════════════════════════════════════
println()
println("  --- Tokenizer ---")

@check "tokenize exact word" begin
    ids = tokenize_text("flesh")
    length(ids) == 1 && ids[1] == 0
end

@check "tokenize multiple words" begin
    ids = tokenize_text("flesh blood bone")
    length(ids) == 3 && ids[1] == 0 && ids[2] == 2 && ids[3] == 1
end

@check "tokenize filters stop words" begin
    ids = tokenize_text("the blood and the bone")
    length(ids) == 2  # only "blood" and "bone"
end

@check "tokenize case insensitive" begin
    ids = tokenize_text("FLESH Blood BONE")
    length(ids) == 3
end

@check "tokenize short words filtered" begin
    ids = tokenize_text("a x y z blood")
    length(ids) == 1 && ids[1] == 2  # only "blood"
end

@check "tokenize prefix matching" begin
    # "darkness" is in vocab, test that "darkn" prefix-matches to "darkness"
    ids = tokenize_text("darknessing")
    # "darknessing" should prefix-match "darkness" (8 common chars)
    length(ids) == 1 && VOCAB[ids[1] + 1] == "darkness"
end

@check "tokenize no match for nonsense" begin
    ids = tokenize_text("zzzzzzz")
    isempty(ids)  # no 3+ prefix match
end

@check "tokenize empty string" begin
    ids = tokenize_text("")
    isempty(ids)
end

# ═══════════════════════════════════════════════════════════════
# 8. DARIO FIELD
# ═══════════════════════════════════════════════════════════════
println()
println("  --- Dario Field ---")

@check "DarioField construction" begin
    f = DarioField()
    length(f.destiny) == 8 &&
    f.trauma == 0.0 &&
    f.prophecy_target === nothing &&
    f.prophecy_age == 0 &&
    length(f.chambers) == 6
end

@check "DarioField chambers initialized to zero" begin
    f = DarioField()
    all(v -> v == 0.0, values(f.chambers))
end

@check "DarioField decay values" begin
    f = DarioField()
    f.decay["fear"] == 0.95 &&
    f.decay["love"] == 0.95 &&
    f.decay["rage"] == 0.93 &&
    f.decay["void"] == 0.96 &&
    f.decay["flow"] == 0.94 &&
    f.decay["complex"] == 0.97
end

@check "update_cooc! and get_cooc" begin
    f = DarioField()
    @assert get_cooc(f, 0, 1) == 0.0
    update_cooc!(f, 0, 1)
    get_cooc(f, 0, 1) == 1.0
end

@check "cooc is symmetric" begin
    f = DarioField()
    update_cooc!(f, 5, 10)
    get_cooc(f, 5, 10) == get_cooc(f, 10, 5)
end

@check "cooc accumulates" begin
    f = DarioField()
    update_cooc!(f, 0, 1)
    update_cooc!(f, 0, 1)
    update_cooc!(f, 0, 1)
    get_cooc(f, 0, 1) == 3.0
end

@check "update_chambers! modifies chambers" begin
    f = DarioField()
    update_chambers!(f, 0)  # phase 0 → flow += 0.05
    f.chambers["flow"] > 0.0
end

@check "update_chambers! phase 0 (early step)" begin
    f = DarioField()
    update_chambers!(f, 0)
    # depth = 0/12 = 0 < 0.33, phase = 0, flow gets boosted
    f.chambers["flow"] > 0.0
end

@check "update_chambers! phase 1 (mid step)" begin
    f = DarioField()
    update_chambers!(f, 5)
    # depth = 5/12 ≈ 0.42, 0.33 ≤ 0.42 < 0.66, phase = 1, fear gets boosted
    f.chambers["fear"] > 0.0
end

@check "update_chambers! phase 2 (late step)" begin
    f = DarioField()
    update_chambers!(f, 10)
    # depth = 10/12 ≈ 0.83 ≥ 0.66, phase = 2, void gets boosted
    f.chambers["void"] > 0.0
end

@check "update_chambers! chambers clamped to [0,1]" begin
    f = DarioField()
    for _ in 1:100
        update_chambers!(f, 11)
    end
    all(v -> 0.0 <= v <= 1.0, values(f.chambers))
end

@check "overlay! modifies logits" begin
    m = Penelope()
    f = DarioField()
    update_cooc!(f, 0, 1)
    f.prophecy_target = 5
    f.prophecy_age = 3
    logits = zeros(Float64, V)
    overlay!(f, logits, [0, 1], 0)
    !all(x -> x == 0.0, logits)
end

@check "overlay! prophecy boost" begin
    f = DarioField()
    f.prophecy_target = 5
    f.prophecy_age = 10
    logits = zeros(Float64, V)
    overlay!(f, logits, [0], 0)
    logits[6] > 0.0  # target index 5, 1-based = 6
end

# ═══════════════════════════════════════════════════════════════
# 9. SEED & KEY EXTRACTION
# ═══════════════════════════════════════════════════════════════
println()
println("  --- Seed & Key Extraction ---")

@check "extract_key filters stop words" begin
    k = extract_key("the darkness of the void")
    k == "darkness"
end

@check "extract_key picks longest" begin
    k = extract_key("rain darkness")
    k == "darkness"
end

@check "extract_key handles all stop words" begin
    k = extract_key("the a an")
    # Fallback: returns first word lowercased
    !isempty(k)
end

@check "extract_key empty fallback" begin
    k = extract_key("")
    k == "silence"
end

@check "find_seed exact match" begin
    idx = find_seed("darkness")
    # darkness is VOCAB_IDX["darkness"]
    idx == VOCAB_IDX["darkness"]
end

@check "find_seed prefix match" begin
    idx = find_seed("fles")
    # "fles" → closest prefix match should be "flesh" (index 0)
    VOCAB[idx + 1] == "flesh"
end

@check "find_seed no match returns valid index" begin
    idx = find_seed("zzzzzzz")
    0 <= idx < V
end

# ═══════════════════════════════════════════════════════════════
# 10. SAVE / LOAD BINARY FORMAT
# ═══════════════════════════════════════════════════════════════
println()
println("  --- Save / Load ---")

const TEST_BIN = "/tmp/test_penelope_jl.bin"

@check "save creates file" begin
    m = Penelope()
    save_model(m, TEST_BIN)
    isfile(TEST_BIN)
end

@check "save file has correct size" begin
    expected = 16 + param_count(Penelope()) * 4
    filesize(TEST_BIN) == expected
end

@check "save file header" begin
    data = read(TEST_BIN)
    v = reinterpret(Int32, data[1:4])[1]
    d = reinterpret(Int32, data[5:8])[1]
    m = reinterpret(Int32, data[9:12])[1]
    s = reinterpret(Int32, data[13:16])[1]
    v == V && d == D && m == M && s == STEPS
end

@check "load restores weights (within Float32 precision)" begin
    m1 = Penelope()
    save_model(m1, TEST_BIN)
    m2 = Penelope()  # fresh random model
    load_model!(m2, TEST_BIN)
    maxdiff = maximum(abs.(m1.embed .- m2.embed))
    maxdiff < 1e-6  # Float32 roundtrip precision
end

@check "load restores step weights" begin
    m1 = Penelope()
    save_model(m1, TEST_BIN)
    m2 = Penelope()
    load_model!(m2, TEST_BIN)
    maxdiff_wr = maximum(abs.(m1.steps[1].wr .- m2.steps[1].wr))
    maxdiff_gate = maximum(abs.(m1.steps[6].w_gate .- m2.steps[6].w_gate))
    maxdiff_down = maximum(abs.(m1.steps[12].w_down .- m2.steps[12].w_down))
    maxdiff_wr < 1e-6 && maxdiff_gate < 1e-6 && maxdiff_down < 1e-6
end

@check "load preserves forward pass" begin
    m1 = Penelope()
    save_model(m1, TEST_BIN)
    m2 = Penelope()
    load_model!(m2, TEST_BIN)
    l1 = forward_step(m1, [0, 100], 0)
    l2 = forward_step(m2, [0, 100], 0)
    maxdiff = maximum(abs.(l1 .- l2))
    maxdiff < 1e-3  # accumulated Float32 imprecision
end

# Cleanup
rm(TEST_BIN, force=true)

# ═══════════════════════════════════════════════════════════════
# 11. GENERATION (run_chain)
# ═══════════════════════════════════════════════════════════════
println()
println("  --- Generation ---")

@check "run_chain produces 13 words (1 seed + 12 steps)" begin
    m = Penelope()
    f = DarioField()
    chain = run_chain(m, f, "darkness")
    length(chain) == 13
end

@check "run_chain all word IDs valid" begin
    m = Penelope()
    f = DarioField()
    chain = run_chain(m, f, "blood rain")
    all(id -> 0 <= id < V, chain)
end

@check "run_chain no duplicate words" begin
    m = Penelope()
    f = DarioField()
    chain = run_chain(m, f, "fire and ice")
    length(Set(chain)) == length(chain)
end

@check "run_chain updates field state" begin
    m = Penelope()
    f = DarioField()
    @assert isempty(f.cooc)
    run_chain(m, f, "ocean depth")
    !isempty(f.cooc)
end

@check "run_chain sets prophecy" begin
    m = Penelope()
    f = DarioField()
    run_chain(m, f, "shadow")
    f.prophecy_target !== nothing
end

# ═══════════════════════════════════════════════════════════════
# 12. CLI / MAIN
# ═══════════════════════════════════════════════════════════════
println()
println("  --- CLI ---")

@check "CLI single chain output" begin
    output = read(`julia $(joinpath(dirname(@__DIR__), "penelope.jl")) "darkness eats"`, String)
    contains(output, "1984 words") &&
    contains(output, "Dario Equation") &&
    contains(output, "destined:") &&
    contains(output, "trainable params")
end

@check "CLI output has drift metric" begin
    output = read(`julia $(joinpath(dirname(@__DIR__), "penelope.jl")) "blood rain"`, String)
    contains(output, "drift") && contains(output, "prophecy")
end

@check "CLI save/load roundtrip" begin
    tmpbin = "/tmp/test_cli_jl.bin"
    # Save
    run(pipeline(`julia $(joinpath(dirname(@__DIR__), "penelope.jl")) --save $tmpbin "test"`, stderr=devnull))
    saved = isfile(tmpbin)
    # Load
    output = read(`julia $(joinpath(dirname(@__DIR__), "penelope.jl")) --load $tmpbin "fire"`, String)
    loaded = contains(output, "loaded")
    rm(tmpbin, force=true)
    saved && loaded
end

# ═══════════════════════════════════════════════════════════════
# 13. FORMAT HELPER
# ═══════════════════════════════════════════════════════════════
println()
println("  --- Formatting ---")

@check "_format_int formats with commas" begin
    _format_int(1000) == "1,000" &&
    _format_int(1000000) == "1,000,000" &&
    _format_int(13155072) == "13,155,072" &&
    _format_int(42) == "42"
end

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
println()
println("  ═══════════════════════════════════════")
println("  $passed passed, $failed failed")
println()

exit(failed)
