#!/bin/bash
# test_penelope.sh — integration tests for all Penelope versions
# Runs: C, Rust, Python (if allowed)

set -e
cd "$(dirname "$0")/.."

PASS=0
FAIL=0
fail() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }
pass() { echo "  OK: $1"; PASS=$((PASS+1)); }

echo ""
echo "  penelope integration test suite"
echo "  ═══════════════════════════════════════"
echo ""

# ─── C VERSION ───
echo "  --- C version ---"

# compile
if cc penelope.c -O2 -lm -o penelope_test_bin 2>/dev/null; then
    pass "C compiles"
else
    fail "C compile"
fi

# runs and produces output
OUTPUT=$(./penelope_test_bin "darkness eats" 2>&1)
if echo "$OUTPUT" | grep -q "1984 words"; then
    pass "C prints header"
else
    fail "C header"
fi

if echo "$OUTPUT" | grep -q "Dario Equation"; then
    pass "C mentions Dario Equation"
else
    fail "C Dario Equation"
fi

if echo "$OUTPUT" | grep -q "destined:"; then
    pass "C shows prophecy target"
else
    fail "C prophecy target"
fi

# count output lines (should have 12 generated words + header)
WORD_LINES=$(echo "$OUTPUT" | grep -cE '^\s+\*?\w' || true)
if [ "$WORD_LINES" -ge 12 ]; then
    pass "C generates 12+ words"
else
    fail "C generates 12+ words (got $WORD_LINES)"
fi

# param count
if echo "$OUTPUT" | grep -q "13152768"; then
    pass "C param count = 13,152,768"
else
    fail "C param count"
fi

rm -f penelope_test_bin

# ─── C UNIT TESTS ───
echo ""
echo "  --- C unit tests ---"
if cc test/test_penelope.c -O2 -lm -o test/test_penelope_bin 2>/dev/null; then
    RESULT=$(./test/test_penelope_bin 2>&1)
    if echo "$RESULT" | grep -q "0 failed"; then
        CPASS=$(echo "$RESULT" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+')
        pass "C unit tests: $CPASS passed, 0 failed"
    else
        fail "C unit tests had failures"
    fi
    rm -f test/test_penelope_bin
else
    fail "C unit tests compile"
fi

# ─── RUST VERSION ───
echo ""
echo "  --- Rust version ---"

if command -v rustc &>/dev/null; then
    if rustc -O penelope.rs -o penelope_rs_test 2>/dev/null; then
        pass "Rust compiles"
    else
        fail "Rust compile"
    fi

    OUTPUT=$(./penelope_rs_test "darkness eats" 2>&1)
    if echo "$OUTPUT" | grep -q "1984 words"; then
        pass "Rust prints header"
    else
        fail "Rust header"
    fi

    if echo "$OUTPUT" | grep -q "Dario Equation"; then
        pass "Rust mentions Dario Equation"
    else
        fail "Rust Dario Equation"
    fi

    if echo "$OUTPUT" | grep -q "destined:"; then
        pass "Rust shows prophecy target"
    else
        fail "Rust prophecy target"
    fi

    WORD_LINES=$(echo "$OUTPUT" | grep -cE '^\s+\*?\w' || true)
    if [ "$WORD_LINES" -ge 12 ]; then
        pass "Rust generates 12+ words"
    else
        fail "Rust generates 12+ words (got $WORD_LINES)"
    fi

    rm -f penelope_rs_test
else
    echo "  SKIP: rustc not found"
fi

# ─── BINARY COMPATIBILITY ───
echo ""
echo "  --- Binary format ---"

# C: save, verify file exists
cc penelope.c -O2 -lm -o penelope_test_bin 2>/dev/null
# use perl timeout on macOS, timeout on linux
if command -v timeout &>/dev/null; then
    echo "" | timeout 5 ./penelope_test_bin --save /tmp/penelope_compat_test.bin 2>&1 >/dev/null || true
else
    echo "" | perl -e 'alarm 5; exec @ARGV' ./penelope_test_bin --save /tmp/penelope_compat_test.bin 2>&1 >/dev/null || true
fi

if [ -f /tmp/penelope_compat_test.bin ]; then
    SZ=$(wc -c < /tmp/penelope_compat_test.bin)
    EXPECTED=$((16 + 13152768 * 4))
    if [ "$SZ" -eq "$EXPECTED" ]; then
        pass "C save file size = $EXPECTED bytes"
    else
        fail "C save file size: got $SZ, expected $EXPECTED"
    fi
    rm -f /tmp/penelope_compat_test.bin
else
    fail "C save file not created"
fi

rm -f penelope_test_bin

# ─── AML (AriannaMethod Language) ───
echo ""
echo "  --- AML (AriannaMethod Language) ---"

if cc ariannamethod/ariannamethod.c -o amlc_test_bin 2>/dev/null; then
    pass "amlc compiler builds"
    if ./amlc_test_bin penelope.aml -o penelope_aml_test_bin 2>/dev/null; then
        pass "penelope.aml compiles"
        # run and check output
        AML_OUT=$(timeout 30 ./penelope_aml_test_bin "darkness eats" 2>&1 || true)
        if echo "$AML_OUT" | grep -q "1984 words"; then
            pass "AML: prints '1984 words'"
        else
            fail "AML: missing '1984 words'"
        fi
        if echo "$AML_OUT" | grep -q "13152768"; then
            pass "AML: param count = 13,152,768"
        else
            fail "AML: param count mismatch"
        fi
        if echo "$AML_OUT" | grep -q "destined:"; then
            pass "AML: shows prophecy target"
        else
            fail "AML: missing prophecy target"
        fi
        rm -f penelope_aml_test_bin
    else
        fail "penelope.aml compile"
    fi
    rm -f amlc_test_bin
else
    echo "  SKIP: cc not found"
fi

echo ""
echo "  ═══════════════════════════════════════"
echo "  $PASS passed, $FAIL failed"
echo ""

exit $FAIL
