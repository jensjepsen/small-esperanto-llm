#!/bin/bash
# Bulk-generate procedural word problems + parallel rewrap.
# Phase 1: 8 procedural types, 10k each = 80k base problems (~seconds, free).
# Phase 2: 1× rewrap each via Gemini Flash Lite with 20 parallel workers.
# Phase 3: concatenate everything into one shuffled SFT JSONL.
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

OUT_DIR=data/word_problems
mkdir -p "$OUT_DIR"

# tuneable
PER_TYPE_N=${PER_TYPE_N:-10000}
VARIANTS=${VARIANTS:-1}
WORKERS=${WORKERS:-20}
BATCH=${BATCH:-5}

TYPES="ratio percent inverse-rate consecutive coin age mixture distance"

echo "================================================================"
echo "Phase 1: procedural generation (8 types × ${PER_TYPE_N} each)"
echo "================================================================"
for t in $TYPES; do
    out="${OUT_DIR}/${t}_proc.jsonl"
    uv run python scripts/word_problems_procedural.py \
        --type "$t" --n "$PER_TYPE_N" --out "$out" 2>&1 | grep -v warning
done

echo
echo "================================================================"
echo "Phase 2: parallel Gemini rewrap (${VARIANTS} variants each, ${WORKERS} workers)"
echo "================================================================"
GOOGLE_API_KEY="${GOOGLE_API_KEY:-$(tr -d '\n\r' < ~/gem)}"
export GOOGLE_API_KEY
for t in $TYPES; do
    in="${OUT_DIR}/${t}_proc.jsonl"
    out="${OUT_DIR}/${t}_rewrap.jsonl"
    log="${OUT_DIR}/${t}_rewrap.log"
    echo "→ $t: $(wc -l < "$in") originals → $out"
    uv run --extra gemini python scripts/word_problems_rewrap.py \
        --input "$in" --output "$out" \
        --variants "$VARIANTS" \
        --batch-size "$BATCH" \
        --workers "$WORKERS" \
        --keep-original \
        --report-every 30 2>&1 | tee "$log" | grep -E "^launching|^  \[|^done"
done

echo
echo "================================================================"
echo "Phase 3: concatenate"
echo "================================================================"
combined="${OUT_DIR}/all_word_problems.jsonl"
> "$combined"
for t in $TYPES; do
    cat "${OUT_DIR}/${t}_rewrap.jsonl" >> "$combined"
done
total=$(wc -l < "$combined")
echo "wrote $total rows → $combined"
