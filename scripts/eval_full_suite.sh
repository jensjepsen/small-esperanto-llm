#!/usr/bin/env bash
# Full eval suite — the battery reported on the danish-lm-400m-sft-v31 cards,
# run through the same per-task scripts those cards were produced with.
#
# Usage:  bash scripts/eval_full_suite.sh <ckpt> <outdir> [batch_size]
#
# An earlier version of this file routed the MC/generation battery through
# eval_downstream_once.py (the training callback) because eval_arc_da,
# eval_piqa_da and eval_gpqa_da took no --batch-size and ran one item at a
# time. That was the wrong trade: it made the suite fast but produced numbers
# from a different harness than the cards, so v31-vs-v33 deltas of a couple of
# points could not be attributed to the model. Those three scripts now batch
# (see batched_eval.py, equivalence asserted by verify_batched_eval.py), so the
# suite uses the card's harness throughout.
#
# `uv run --no-sync`: WORKLOAD=sft pins torch<2.9 for the prebuilt FA2 wheel,
# while the `all` extra still declares vllm>=0.17 (torch>=2.10), so a plain
# `uv run` re-resolves and exits unsatisfiable. Override with PY=... to point
# at an interpreter directly.
set -u
CKPT="${1:?ckpt path required}"
OUT="${2:?output dir required}"
BS="${3:-128}"
mkdir -p "$OUT"
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/tmp/hf-cache}"
PY="${PY:-uv run --no-sync python}"

echo "=== Full eval suite ==="
echo "  ckpt:  $CKPT"
echo "  out:   $OUT"
echo "  batch: $BS"
echo

run() {           # run <name> <grep-pattern> -- <cmd...>
  local name="$1" pat="$2"; shift 3
  echo ">> $name"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "   FAILED (exit $rc) — tail of $OUT/$name.log:"
    tail -5 "$OUT/$name.log" | sed 's/^/     /'
  else
    grep -aE "$pat" "$OUT/$name.log" | tail -4 | sed 's/^/   /'
  fi
  echo
}

# ── 1. generation-based ────────────────────────────────────────────────────
run gsm8k    "=== gsm8k" -- \
  $PY -u scripts/eval_gsm8k_da_gen.py "$CKPT" --batch-size "$BS" \
      --out "$OUT/gsm8k.jsonl"

run citgen   "=== citizen" -- \
  $PY -u scripts/eval_cit_da_gen.py "$CKPT" --batch-size "$BS"

run sciq_gen "pass@1|open-Q" -- \
  $PY -u scripts/eval_sciq_da_openq.py --ckpt "$CKPT" --batch-size "$BS"

run textman  "ChrF" -- \
  $PY -u scripts/eval_textman_da.py --ckpt "$CKPT" --batch-size "$BS" \
      --subtype both

# ── 2. chat-format multiple choice ─────────────────────────────────────────
run citmc         "=== cit-mc" -- \
  $PY -u scripts/eval_cit_da_mc.py "$CKPT" --batch-size "$BS"

run sciq_mc       "=== sciq-mc" -- \
  $PY -u scripts/eval_sciq_da_mc.py "$CKPT" --batch-size "$BS"

run arc_easy      "=== arc" -- \
  $PY -u scripts/eval_arc_da.py --ckpt "$CKPT" --config arc_easy \
      --mode chat-mc --batch-size "$BS"

run arc_challenge "=== arc" -- \
  $PY -u scripts/eval_arc_da.py --ckpt "$CKPT" --config arc_challenge \
      --mode chat-mc --batch-size "$BS"

run openbookqa    "=== openbookqa" -- \
  $PY -u scripts/eval_arc_da.py --ckpt "$CKPT" \
      --dataset jensjepsen/danish-openbookqa --config main \
      --mode chat-mc --batch-size "$BS"

# --mode chat-mc, pinned: eval_piqa_da defaults to `raw` (continuation
# log-prob), which scores ~7pp higher than the letter-generation mode the
# cards report PIQA under. Leaving it default silently produced a v33 number
# that could not be compared to v31's 53.00.
run piqa          "=== piqa" -- \
  $PY -u scripts/eval_piqa_da.py --ckpt "$CKPT" --batch-size "$BS" \
      --mode chat-mc

run gpqa          "=== gpqa" -- \
  $PY -u scripts/eval_gpqa_da.py --ckpt "$CKPT" \
      --data jensjepsen/danish-gpqa-diamond-v1 --batch-size "$BS"

# ── 3. length-normalized log-prob MC ───────────────────────────────────────
run mc_logprob "sciq|citmc|arc" -- \
  $PY -u scripts/probe_mc_logprob.py --ckpt "$CKPT"

# ── 4. instruction following + ICL ─────────────────────────────────────────
run ifeval_da "prompt-strict|prompt-loose|inst-strict|inst-loose" -- \
  $PY -u scripts/eval_ifeval_da.py "$CKPT" --batch-size "$BS" \
      --max-new-tokens 512 --dump-jsonl "$OUT/ifeval_da.jsonl"

run icl "exact|key-set" -- \
  $PY -u scripts/eval_icl_schema_format.py --ckpt "$CKPT" --batch-size "$BS"

echo "=== Suite done — logs in $OUT ==="
