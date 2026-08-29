#!/usr/bin/env bash
# Danish SFT v7 (mix8-no-wp) — v5/v6 with BOTH danish-word-problems-* dropped.
#
# Delta from v5 mix9-if2 / v6 mix9-wp-reworded:
#   DROP  jensjepsen/danish-word-problems-v2 (240k procedural math stories)
#   DROP  jensjepsen/danish-word-problems-reworded-v1 (207k Gemma-reworded)
#
# Rationale: v5→v6 swap of wp-v2 → wp-reworded produced NO GSM8K lift
# (both hovered 18-20% pass@1 across all checkpoints). Motivates ablating
# both to confirm whether the procedural-wp branch contributes anything.
#
# If v7 GSM8K stays near 20%, procedural word-problems weren't buying us
# math capability — recipe execution ability comes from algebra/arith-chain/
# metamath instead, and the stories were dead weight.
# If v7 drops meaningfully, the stories WERE contributing — retry v6-style
# improvement (LLM-judged reword, or GSM8K-shape wrapper).
#
# Kept: metamath-gsm, algebra-v5, arith-chain-v1, wiki-grounded-v3,
# text-to-question-v2, sciq, gsm8k, instruction-following-v2 (8 sources,
# ~852k rows).
# Same optimizer/schedule as v5/v6.
# Output written to overlay (/root), not /workspace — pod-quota isolation.

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ckpt310k \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v7_mix8nowp \
  --no-morpheme-preprocess \
  --sft-data \
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
  --wandb-run-name da_v7_sft_mix8nowp \
  --wandb-tags sft da v7 mix8 no-morpheme if-v2 no-wp no-alpaca no-dolly
