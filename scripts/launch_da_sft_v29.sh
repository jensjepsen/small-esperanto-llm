#!/usr/bin/env bash
# Danish SFT v29 — v28 mix + task-expansion (RC / reason / textman).
#
# Adds 3 new gemini-2.5-generated datasets sourced from Danish Wikipedia
# (21k articles each, 5-6 subtypes per dataset, ~336k SFT rows):
#   - danish-rc-v1       — reading comprehension (multi_fact, numeric,
#                           attribution, ordering, causal_inference)
#   - danish-reason-v1   — reasoning (causal_chain, argumentation, multi_step,
#                           ranking, analogy, fact_check)
#   - danish-textman-v1  — text manipulation (summary, rewrite, style_transfer,
#                           extraction, elaborate, genre_transform)
#
# vs v28 the only change is the +3 datasets. Base + pipeline identical.
# Purpose: measure the impact of teaching text→text task diversity that v28's
# mix lacked (nothing there taught summarization, rewrite, extraction,
# multi-step reasoning, etc.).
#
# Expected step count: v28 was ~35k steps/3e; adding ~336k rows ≈ +22% more
# steps → ~43k steps/3e. On H100 bs=128 ≈ 4.5h wall.
set -euo pipefail
cd /root/espllm

export PATH="$HOME/.local/bin:$PATH"

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext2048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /workspace/runs/sft/da_v29_mix16_taskexp_ropext_3e \
  --no-morpheme-preprocess \
  --sft-data \
    jensjepsen/danish-metamath-gsm:sft \
    jensjepsen/danish-algebra-sft-v5-mixed \
    jensjepsen/danish-arith-chain-sft-v1 \
    jensjepsen/danish-wiki-grounded-sft-v3:sft \
    jensjepsen/danish-text-to-question-v2:sft \
    jensjepsen/danish-sciq:sft:train \
    jensjepsen/danish-gsm8k:sft:train \
    jensjepsen/danish-instruction-following-v4:sft:train \
    jensjepsen/danish-wiki-closedqa-v1:sft \
    jensjepsen/danish-word-problems-v2 \
    jensjepsen/danish-wiki-closedqa-stem-v1:sft \
    jensjepsen/danish-wiki-broadqa-stem-v1:sft \
    jensjepsen/danish-wiki-mc-letters-v1 \
    jensjepsen/danish-rc-v1 \
    jensjepsen/danish-reason-v1 \
    jensjepsen/danish-textman-v1 \
  --epochs 3 --batch-size 128 --gradient-accumulation 1 \
  --optim adamw_bnb_8bit \
  --learning-rate 3e-5 --lr-scheduler constant_with_warmup --warmup-steps 500 \
  --save-fraction-of-epoch 0.25 --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen citmc --downstream-batch-size 32 --top-k-downstream 7 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v29_sft_mix16_taskexp_ropext_3e \
  --wandb-tags sft da v29 mix16 ropext2048 if-v4 mc-letters task-expansion rc reason textman constant-lr epochs-3 flatten-packing \
  --no-torch-compile
