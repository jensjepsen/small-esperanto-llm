#!/usr/bin/env bash
# Danish SFT v5 (mix9-if2) — v4 with IF dataset upgraded v1 → v2.
#
# Delta from v4 mix9:
#   SWAP  danish-instruction-following-v1 (65k, 17 constraints)
#      →  danish-instruction-following-v2:sft:train (64,732 rows, 29 constraints,
#                                                   ~150 render variants,
#                                                   deep surface variety,
#                                                   IFEval-parity taxonomy)
#
# Rationale: v4's IF training scored 73-80% on in-distribution but only 38%
# on OOD phrasings and 0/7 on IFEval-shape prompts because IF-v1 used one
# canonical phrasing per constraint. IF-v2 rewrites the same constraints
# with 4-7 alternative phrasings each + quote-style + casing + punctuation
# variation + uniform 1-5 combo-size distribution. Model should now learn
# constraint semantics rather than specific character sequences.
#
# Same 8 non-IF datasets as v4. Same optimizer/schedule.
# Total ~1,088k rows across 9 sources.
# Output written to overlay (/root), not /workspace — pod-quota isolation.

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ckpt310k \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v5_mix9if2 \
  --no-morpheme-preprocess \
  --sft-data \
    jensjepsen/danish-word-problems-v2:sft \
    jensjepsen/danish-metamath-gsm:sft \
    jensjepsen/danish-algebra-sft-v5-mixed \
    jensjepsen/danish-arith-chain-sft-v1 \
    jensjepsen/danish-wiki-grounded-sft-v3:sft \
    jensjepsen/danish-text-to-question-v2:sft \
    jensjepsen/danish-sciq:sft:train \
    jensjepsen/danish-gsm8k:sft:train \
    jensjepsen/danish-instruction-following-v2:sft:train \
  --epochs 2 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --learning-rate 5e-5 \
  --max-length 512 \
  --lr-scheduler cosine_with_min_lr \
  --warmup-steps 200 \
  --save-fraction-of-epoch 0.25 \
  --save-total-limit 3 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v5_sft_mix9if2 \
  --wandb-tags sft da v5 mix9 no-morpheme if-v2 no-alpaca no-dolly
