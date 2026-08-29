#!/usr/bin/env bash
# Danish SFT v3 (mix10) — v2 + dolly + alpaca restored.
#
# Delta from v2 mix8: add jensjepsen/danish-dolly-15k + jensjepsen/danish-alpaca-cleaned
# for general-instruction breadth. v2 showed 3-5pp regression on Danish
# civics/knowledge evals from dropping these; adding back accepts some
# US-context bleed for broader instruction coverage.
#
# Total ~1,033k rows (v2 was ~967k).

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ckpt310k \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /workspace/runs/sft/da_v3_mix10 \
  --no-morpheme-preprocess \
  --sft-data \
    jensjepsen/danish-word-problems-v2:sft \
    jensjepsen/danish-metamath-gsm:sft \
    jensjepsen/danish-algebra-sft-v5-mixed \
    jensjepsen/danish-arith-chain-sft-v1 \
    jensjepsen/danish-wiki-grounded-sft-v2:sft \
    jensjepsen/danish-text-to-question:sft \
    jensjepsen/danish-sciq:sft:train \
    jensjepsen/danish-gsm8k:sft:train \
    jensjepsen/danish-dolly-15k:sft \
    jensjepsen/danish-alpaca-cleaned:sft \
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
  --wandb-run-name da_v3_sft_mix10 \
  --wandb-tags sft da v3 mix10 no-morpheme dolly alpaca metamath
