#!/usr/bin/env bash
# Full eval suite — the battery reported on the danish-lm-400m-sft-v31 cards.
#
# Usage:  bash scripts/eval_full_suite.sh <ckpt> <outdir> [batch_size]
#
# Four groups, matching the card's sections:
#   1. chat-format generation  gsm8k, sciq, citgen, citmc, arc_easy,
#                              arc_challenge, piqa, gpqa, textman x2, icl
#   2. ifeval-da               prompt/inst x strict/loose (541 rows)
#   3. length-normalized logp  sciq, citmc, arc_easy, arc_challenge
#   4. openbookqa              chat-MC (not in the callback registry)
#
# Group 1 goes through eval_downstream_once.py rather than the per-task
# scripts: those all reimplement generation and several (eval_piqa_da,
# eval_gpqa_da, eval_arc_da) take no --batch-size at all, so they run one item
# at a time and leave the GPU idle. The callback batches every task at $BS and
# is the same code the training loop uses, so these numbers are comparable to
# the in-training curve as well as across tasks.
#
# The earlier scratchpad suites called `uv run python` and a pod-local
# /root/eval_downstream_mc.py. Neither survives: `uv run` re-resolves and dies
# where WORKLOAD=sft pinned torch<2.9, and the pod-local helper vanished with
# the pod it was copied to. Hence `uv run --no-sync` and repo scripts only.
set -u
CKPT="${1:?ckpt path required}"
OUT="${2:?output dir required}"
BS="${3:-256}"
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

echo ">> [1/4] chat-format generation battery (batched at $BS)"
$PY -u scripts/eval_downstream_once.py --ckpt "$CKPT" --n 0 \
  --batch-size "$BS" \
  --evals gsm8k sciq citgen citmc arc_easy arc_challenge piqa gpqa \
          textman_summary textman_rewrite icl \
  2>&1 | tee "$OUT/gen_battery.log"
echo

echo ">> [2/4] ifeval-da (541 rows, 4 metrics)"
$PY -u scripts/eval_ifeval_da.py "$CKPT" --batch-size "$BS" \
  --max-new-tokens 512 --dump-jsonl "$OUT/ifeval_da.jsonl" \
  > "$OUT/ifeval_da.log" 2>&1
grep -aE "prompt-strict|prompt-loose|inst-strict|inst-loose" "$OUT/ifeval_da.log" | tail -4
echo

echo ">> [3/4] length-normalized log-prob MC"
$PY -u scripts/probe_mc_logprob.py --ckpt "$CKPT" \
  > "$OUT/mc_logprob.log" 2>&1
grep -aE "sciq|citmc|arc" "$OUT/mc_logprob.log" | tail -6
echo

echo ">> [4/4] openbookqa (chat-MC, 500 rows)"
$PY -u scripts/eval_arc_da.py --ckpt "$CKPT" \
  --dataset jensjepsen/danish-openbookqa --config main \
  --mode chat-mc > "$OUT/openbookqa.log" 2>&1
grep -aE "acc|FINAL|===" "$OUT/openbookqa.log" | tail -3
echo

echo "=== Suite done — logs in $OUT ==="
