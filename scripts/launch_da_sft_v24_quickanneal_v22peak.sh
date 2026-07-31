#!/usr/bin/env bash
# Danish SFT v24 — quick (0.2ep) linear anneal from v22 training PEAK
# ckpt-35304 (agg 0.254, epoch 3.0 final ckpt of v22).
#
# v23 tried anneal from v22-avg-top3 (already-smoothed) and got only
# marginal movement (peak 0.261 vs base 0.264). Hypothesis: averaging
# 3 ckpts pre-anneal put the model in a flat loss region with no room
# to polish. Try annealing from the SINGLE peak ckpt — more room, more
# gradient signal, closer to what v18/v21 did with v16-peak.
#
# Same recipe as v18/v21/v23: 0.2ep, LR 1e-5 linear→0, warmup 0.
#
# Runtime ~40 min on 5090.

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

CKPT=${CKPT:-/root/runs/sft/da_v22_mix13_mc_letters_3e/best/step-35304-agg-0.254}

uv run python -u scripts/train_sft_packed.py \
  --checkpoint "$CKPT" \
  --tokenizer "$CKPT" \
  --output-dir /root/runs/sft/da_v24_quickanneal_from_v22_peak \
  --no-morpheme-preprocess \
  --sft-data \
    jensjepsen/danish-metamath-gsm:sft \
    jensjepsen/danish-algebra-sft-v5-mixed \
    jensjepsen/danish-arith-chain-sft-v1 \
    jensjepsen/danish-wiki-grounded-sft-v3:sft \
    jensjepsen/danish-text-to-question-v2:sft \
    jensjepsen/danish-sciq:sft:train \
    jensjepsen/danish-gsm8k:sft:train \
    jensjepsen/danish-instruction-following-v3:sft:train \
    jensjepsen/danish-wiki-closedqa-v1:sft \
    jensjepsen/danish-word-problems-v2 \
    jensjepsen/danish-wiki-closedqa-stem-v1:sft \
    jensjepsen/danish-wiki-broadqa-stem-v1:sft \
    jensjepsen/danish-wiki-mc-letters-v1 \
  --epochs 0.2 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --learning-rate 1e-5 \
  --max-length 512 \
  --lr-scheduler linear \
  --warmup-steps 0 \
  --save-steps 400 \
  --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen citmc \
  --downstream-batch-size 32 \
  --top-k-downstream 3 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v24_quickanneal_from_v22_peak \
  --wandb-tags sft da v24 anneal from-v22-peak-35304 quick-0.2ep lr-1e-5 linear-decay full-set top3 mc-letters
