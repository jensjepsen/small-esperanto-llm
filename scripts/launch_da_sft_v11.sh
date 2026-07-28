#!/usr/bin/env bash
# Danish SFT v11 (mix11-if-v3-wpv2-stem) — v10 + STEM wiki-closedqa added.
#
# Deltas from v10:
#   ADD  jensjepsen/danish-wiki-closedqa-stem-v1  (78,318 rows)
#     Dense factual Q/A from 4,986 STEM-focused Danish Wikipedia articles
#     (Fysik, Kemi, Biologi, Astronomi, Geologi, Matematik at depth-2 with
#     asteroid + stub filtering, ≥1200 char text length). 16 Q/A per article
#     via gemma-3-12b on OpenRouter, concise-answer prompt + JSON salvage.
#     Cost: $0.83. See [[reference-danish-wiki-closedqa-stem]].
#
# Motivation:
#   v10's SciQ open-Q = 9.9% is capped by ~zero STEM exposure in the mix.
#   Existing danish-wiki-closedqa-v1 (from salience-filter) has only ~74
#   STEM articles (0.7% of its 11k). This dataset adds ~5k STEM articles
#   with proper Danish scientific vocabulary (kapillarrør, binær fission,
#   isoleret system etc. — exactly the terms v10 missed).
#
# Kept: (v10's 10 sources) metamath-gsm, algebra-v5, arith-chain-v1,
# wiki-grounded-v3, text-to-question-v2, sciq, gsm8k, instruction-following-v3,
# wiki-closedqa-v1, word-problems-v2 — plus wiki-closedqa-stem-v1 = 11 sources.
#
# Same optimizer/schedule as v5–v10 (2 epochs, batch 32, lr 5e-5,
# cosine_with_min_lr, 200 warmup).
# Total ~1.35M rows (v10 was ~1.27M).

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ckpt310k \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v11_mix11_ifv3_wpv2_stem \
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
  --wandb-run-name da_v11_sft_mix11_ifv3_wpv2_stem \
  --wandb-tags sft da v11 mix11 no-morpheme if-v3 wp-v2 wiki-closedqa stem
