#!/usr/bin/env bash
# Danish SFT v23 — quick (0.2ep) linear anneal from v22-avg-top3.
#
# Basis: v22 top-3 averaged (gsm 18.45, sciq 12.30 / sciq-mc 59.6,
# cit-gen 28.6 / cit-mc 46.1, IF 56.2) — the best v22 model.
#
# Same recipe as v18/v21 (which took v16-peak from agg 0.183 → 0.195):
#   0.2ep, LR 1e-5 linear→0, warmup 0, save every 400 for 5 evals,
#   top-3 preservation, full-set downstream (all 4: gsm/sciq/citgen/citmc).
#
# Uses same v22 mix (v16 mix + wiki-mc-letters) so the anneal continues
# teaching label emission rather than eroding it.
#
# No optim state carried over — v22 avg is a static averaged model,
# not a training checkpoint (fresh-optim anneal, same as v18/v21).
#
# Runtime ~40 min on 5090.

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

CKPT=${CKPT:-/root/v22_avg_top3}

uv run python -u scripts/train_sft_packed.py \
  --checkpoint "$CKPT" \
  --tokenizer "$CKPT" \
  --output-dir /root/runs/sft/da_v23_quickanneal_from_v22_avg \
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
  --wandb-run-name da_v23_quickanneal_from_v22_avg \
  --wandb-tags sft da v23 anneal from-v22-avg-top3 quick-0.2ep lr-1e-5 linear-decay full-set top3 mc-letters
