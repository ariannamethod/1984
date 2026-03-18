# Penelope Health Report — YYYY-MM-DD

## Weight Loading
- **Loaded from:** (path to .bin file, or "none — weightless mode")
- **Format:** PEN7 / weightless
- **Status:** OK / MISMATCH / NOT FOUND

## Input
<!-- The haiku received from Haiku organism -->
```
(paste haiku here)
```

## Seed
- **Key extracted:** (longest non-stop word from input)
- **Seed word:** (vocab word matched from key)
- **Seed index:** (index in 1984-word vocabulary)

## Output (12 words)
<!-- The 12 associative words generated -->
```
1.  word1
2.  word2
3.  word3
4.  word4
5.  word5
6.  word6
7.  word7
8.  word8
9.  word9
10. word10
11. word11
12. word12
```

## Chain Analysis
- **Categories visited:** (list which of the 29 semantic categories appear)
- **Macro-category diversity (drift):** X / 8
- **Repeated categories:** (any category appearing more than twice)
- **Extended vocab words used:** (any words from BPE-extended set, not hardcoded 1984)

## BPE Context
- **Final BPE buffer length:** X tokens
- **Extended vocab size:** X words (1984 hardcoded + N from BPE)

## Prophecy
- **Target word:** (the prophecy target set at start)
- **Target category:** (EMOTION / ABSTRACT / MATERIAL+)
- **Fulfilled:** YES / NO
- **Prophecy age at end:** X (incremented each step)

## Dario Field State
- **Chambers:** fear=X.XX love=X.XX rage=X.XX void=X.XX flow=X.XX complex=X.XX
- **Trauma:** X.XX
- **Destiny vector:** [X.XX, X.XX, X.XX, X.XX, X.XX, X.XX, X.XX, X.XX]

## Status
- [ ] HEALTHY — 12 words, diverse categories, prophecy tracked
- [ ] DEGRADED — fewer words, low diversity, or anomalies
- [ ] FAILED — no output, crash, or weight loading error

## Notes
<!-- Any observations, anomalies, patterns -->
