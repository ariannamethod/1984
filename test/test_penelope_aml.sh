#!/bin/bash
# test_penelope_aml.sh — tests for penelope.aml (AriannaMethod Language version)
#
# Tests: compiler build, AML compilation, output format, param count,
#        word generation, prophecy target, save file size, --emit-c
#
# Usage: bash test/test_penelope_aml.sh

set -e
cd "$(dirname "$0")/.."

PASS=0
FAIL=0
fail() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }
pass() { echo "  OK: $1"; PASS=$((PASS+1)); }

echo ""
echo "  penelope.aml test suite"
echo "  ═══════════════════════════════════════"
echo ""

# ─── BUILD COMPILER ───
echo "  --- Build amlc compiler ---"

AMLC=/tmp/test_amlc_$$
PENELOPE_AML=/tmp/test_penelope_aml_$$

if cc ariannamethod/ariannamethod.c -o "$AMLC" 2>/dev/null; then
    pass "amlc compiles"
else
    fail "amlc compile"
    echo "  ABORT: cannot continue without compiler"
    exit 1
fi

# ─── COMPILE PENELOPE.AML ───
echo ""
echo "  --- Compile penelope.aml ---"

AMLC_OUT=$("$AMLC" penelope.aml -o "$PENELOPE_AML" 2>&1)
if [ -f "$PENELOPE_AML" ]; then
    pass "penelope.aml compiles to binary"
else
    fail "penelope.aml compile"
    echo "  ABORT: cannot continue without binary"
    echo "  amlc output: $AMLC_OUT"
    rm -f "$AMLC"
    exit 1
fi

# ─── BLOOD BLOCK COUNT ───
if echo "$AMLC_OUT" | grep -q "11 BLOOD block"; then
    pass "amlc found 11 BLOOD blocks"
else
    fail "BLOOD block count (expected 11)"
fi

if echo "$AMLC_OUT" | grep -q "BLOOD MAIN present"; then
    pass "BLOOD MAIN present"
else
    fail "BLOOD MAIN not found"
fi

# ─── OUTPUT FORMAT ───
echo ""
echo "  --- Output format ---"

OUTPUT=$("$PENELOPE_AML" "darkness eats" 2>&1)

if echo "$OUTPUT" | grep -q "1984 words"; then
    pass "prints '1984 words'"
else
    fail "missing '1984 words'"
fi

if echo "$OUTPUT" | grep -q "Dario Equation"; then
    pass "mentions Dario Equation"
else
    fail "missing 'Dario Equation'"
fi

if echo "$OUTPUT" | grep -q "Arianna Method"; then
    pass "credits Arianna Method"
else
    fail "missing 'Arianna Method'"
fi

if echo "$OUTPUT" | grep -q "destined:"; then
    pass "shows prophecy target"
else
    fail "missing prophecy target"
fi

# ─── PARAM COUNT ───
if echo "$OUTPUT" | grep -q "13152768"; then
    pass "param count = 13,152,768"
else
    fail "param count mismatch"
fi

# ─── WORD GENERATION ───
WORD_LINES=$(echo "$OUTPUT" | grep -cE '^\s+\*?\w' || true)
if [ "$WORD_LINES" -ge 12 ]; then
    pass "generates 12+ words (got $WORD_LINES)"
else
    fail "generates 12+ words (got $WORD_LINES)"
fi

# ─── DRIFT AND PROPHECY ───
if echo "$OUTPUT" | grep -q "drift.*prophecy"; then
    pass "shows drift/prophecy summary"
else
    fail "missing drift/prophecy summary"
fi

# ─── SAVE FILE SIZE ───
echo ""
echo "  --- Binary format ---"

SAVE_PATH="/tmp/test_aml_save_$$.bin"
echo "" | timeout 10 "$PENELOPE_AML" --save "$SAVE_PATH" 2>&1 >/dev/null || true

if [ -f "$SAVE_PATH" ]; then
    SZ=$(wc -c < "$SAVE_PATH")
    EXPECTED=$((16 + 13152768 * 4))
    if [ "$SZ" -eq "$EXPECTED" ]; then
        pass "save file size = $EXPECTED bytes"
    else
        fail "save file size: got $SZ, expected $EXPECTED"
    fi
    rm -f "$SAVE_PATH"
else
    fail "save file not created"
fi

# ─── EMIT-C ───
echo ""
echo "  --- Transpiler ---"

EMIT_OUT=$("$AMLC" penelope.aml --emit-c 2>/dev/null)
EMIT_LINES=$(echo "$EMIT_OUT" | wc -l)
if [ "$EMIT_LINES" -ge 1000 ]; then
    pass "--emit-c generates $EMIT_LINES lines of C"
else
    fail "--emit-c generates too few lines ($EMIT_LINES)"
fi

if echo "$EMIT_OUT" | grep -q "BLOOD COMPILE: penelope_vocab"; then
    pass "--emit-c includes vocab block marker"
else
    fail "--emit-c missing vocab block marker"
fi

if echo "$EMIT_OUT" | grep -q "int main"; then
    pass "--emit-c includes main()"
else
    fail "--emit-c missing main()"
fi

# ─── CLEANUP ───
rm -f "$AMLC" "$PENELOPE_AML"

echo ""
echo "  ═══════════════════════════════════════"
echo "  $PASS passed, $FAIL failed"
echo ""

exit $FAIL
