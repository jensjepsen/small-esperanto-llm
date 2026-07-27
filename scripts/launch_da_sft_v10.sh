#!/usr/bin/env bash
# Danish SFT v10 (mix10-if-v3-wp-v2) — v9 with wp-reworded → wp-v2 swap.
#
# Deltas from v9:
#   SWAP jensjepsen/danish-word-problems-reworded-v1  (207k)
#     →  jensjepsen/danish-word-problems-v2            (240k)
#
# Motivation:
#   v9 evals showed SciQ open-Q regressed to 8.6% (recall used to be higher on
#   earlier DA SFTs). Hypothesis: wp-reworded's ~10-15% hidden semantic drift
#   ([[reference-danish-wp-reworded]]) may leak noise into factoid extraction,
#   and its natural-language wrappers dilute the procedural math structure
#   that wp-v2 provides cleanly. wp-v2 = 16 compositional recipes with
#   verified math chains and idiom bank ([[reference-danish-word-problems]]).
#
#   The size delta is small (+33k), so total mix stays comparable to v9
#   (~1.26M rows vs v9's ~1.23M). Everything else identical → clean A/B
#   between wp-reworded (v9) and wp-v2 (v10).
#
# Kept: (v9's 9 sources) metamath-gsm, algebra-v5, arith-chain-v1,
# wiki-grounded-v3, text-to-question-v2, sciq, gsm8k, instruction-following-v3,
# wiki-closedqa-v1 — plus wp-v2 = 10 total sources.
#
# Same optimizer/schedule as v5–v9.

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ckpt310k \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v10_mix10_ifv3_wpv2 \
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
  --wandb-run-name da_v10_sft_mix10_ifv3_wpv2 \
  --wandb-tags sft da v10 mix10 no-morpheme if-v3 wp-v2 wiki-closedqa
