#!/bin/bash
# SFT launch — matches v16_sft_v10_wp4 baseline (Jul 7) with two
# new-this-session additions: metamath-gsm ASCII math and SQuAD-EO
# reading comp. No caps — every source used at full size.
#
# Usage:
#   bash scripts/launch_sft_v10.sh <checkpoint-dir> [output-dir]
#
# On the current box (1080 Ti Pascal) the batch/accum config below
# assumes packed sequences at max_length=512 with fp16. On H100/5090
# override --batch-size and --gradient-accumulation via env or edit here.
set -euo pipefail

CHECKPOINT=${1:?usage: launch_sft_v10.sh <checkpoint> [output]}
OUTPUT=${2:-runs/sft/v10_sft_v10_mix}

SFT_SOURCES=(
  # v16_sft_v10_wp4 baseline (13 sources) —
  jensjepsen/esperanto-orca-math
  jensjepsen/esperanto-alpaca-distill
  jensjepsen/esperanto-alpaca-cleaned
  jensjepsen/esperanto-sft-dolly
  jensjepsen/esperanto-gsm8k
  jensjepsen/esperanto-algebra-sft-v5-mixed
  jensjepsen/esperanto-arith-chain-sft-v1
  jensjepsen/esperanto-word-problems-v4:sft
  jensjepsen/esperanto-sciq-sft
  jensjepsen/esperanto-balanced-copa-sft
  jensjepsen/esperanto-piqa-sft
  jensjepsen/esperanto-mmlu-sft
  jensjepsen/esperanto-triviaqa-sft
  # NEW this session —
  jensjepsen/esperanto-metamath-gsm:sft-ascii
  jensjepsen/esperanto-squad:sft
  # Creative writing / descriptive continuations — fills a genre gap that
  # every other source (all Q/A, math, MC) misses.
  jensjepsen/esperanto-sft-creative
)

echo "=== SFT mix (${#SFT_SOURCES[@]} sources, full-size) ==="
printf '  %s\n' "${SFT_SOURCES[@]}"

HF_HOME=${HF_HOME:-/mnt/data2/hf_cache} \
uv run python -u scripts/train_sft_packed.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT" \
  --sft-data "${SFT_SOURCES[@]}" \
  --epochs 3 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --learning-rate 5e-5 \
  --max-length 512 \
  --lr-scheduler cosine_with_min_lr \
  --warmup-steps 200 \
  --save-fraction-of-epoch 0.5 \
  --save-total-limit 3 \
  --wandb-run-name "v10_sft_v10_mix" \
  --wandb-tags sft v10 mix16 ascii-math
