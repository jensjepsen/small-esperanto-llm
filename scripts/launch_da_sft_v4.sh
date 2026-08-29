#!/usr/bin/env bash
# Danish SFT v4 (mix9) — v3 minus alpaca+dolly, plus IF-v1, upgraded wiki+T2Q.
#
# Delta from v3 mix10:
#   DROP  jensjepsen/danish-dolly-15k        (US-context bleed, no cit-MC lift)
#   DROP  jensjepsen/danish-alpaca-cleaned   (same reason)
#   UP    danish-wiki-grounded-sft-v2 → v3   (50k → 95k, 8 categories)
#   UP    danish-text-to-question → v2       (23k → 34k, rebuilt from wiki-v3)
#   ADD   danish-instruction-following-v1    (65k IF rows, 17 constraints)
#
# Rationale: IF-v1 covers general-instruction breadth in DA-native prompts
# with programmatically-verified constraint compliance, replacing the
# US-translated Alpaca/Dolly generic instructions.
#
# Total ~1,087k rows across 9 sources.
# Output written to overlay (/root), not /workspace — pod-quota isolation.

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ckpt310k \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v4_mix9 \
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
    jensjepsen/danish-instruction-following-v1:sft \
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
  --wandb-run-name da_v4_sft_mix9 \
  --wandb-tags sft da v4 mix9 no-morpheme if-v1 no-alpaca no-dolly
