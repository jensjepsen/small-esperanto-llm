#!/usr/bin/env bash
# SFT cold-start (2k examples) on quantity-reasoning, then GRPO from the
# resulting checkpoint. Runs sequentially; GRPO only starts if SFT succeeds.
set -euo pipefail

SFT_OUT=runs/large/checkpoint-44000-quant-sft
GRPO_OUT=runs/large/checkpoint-44000-grpo-quant

uv run python scripts/train_sft.py \
    --checkpoint runs/large/checkpoint-44000 \
    --sft-data jensjepsen/esperanto-sft-quantity-reasoning \
    --output-dir "$SFT_OUT" \
    --epochs 1 --batch-size 8 --gradient-accumulation 4 \
    --save-steps 100 \
    --wandb-tags quant sft cold-start

uv run python scripts/train_grpo.py \
    --checkpoint "$SFT_OUT/final" \
    --dataset jensjepsen/esperanto-sft-quantity-reasoning \
    --output-dir "$GRPO_OUT" \
    --prompt-style chat --presence-target math-answer \
    --num-generations 8 --batch-size 8 --grad-accum 4 \
    --max-completion-len 250 \
    --epochs 1 --save-steps 10 --save-total-limit 3 --logging-steps 1 \
    --wandb-tags quant math warm-start
