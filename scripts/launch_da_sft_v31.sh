#!/usr/bin/env bash
# Danish SFT v31 — v29 mix + newly translated ARC-DA + OpenBookQA-DA on
# ropext8048 base.
#
# vs v29:
#   - Base swapped ropext2048-v1 → ropext8048-v1 (avg-4 SWA of steps 600/800/1000/1200)
#   - +danish-arc:sft:train        — 6740 rows (ARC-Easy + ARC-Challenge, 2 styles each)
#   - +danish-openbookqa:sft:train — 9914 rows (OBQA main, 2 styles each)
#
# Batching: DataCollatorWithFlattening packs at collate time, so trainer
# iterates raw ROWS not packs — same step count as v29 requires same eff_bs.
#   --batch-size 32 --gradient-accumulation 4 → eff_bs=128 (matches v29)
# At bs=16 seq=8048 the H100 was at 55% util / 44% mem — bs=32 fills it.
#
# Expected ~43-45k steps × 3 epochs on H100 80GB.
set -euo pipefail
cd /root/espllm

export PATH="$HOME/.local/bin:$PATH"

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext8048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /workspace/runs/sft/da_v31_mix18_ropext8048_3e \
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
    jensjepsen/danish-arc:sft:train \
    jensjepsen/danish-openbookqa:sft:train \
  --epochs 3 --batch-size 32 --gradient-accumulation 4 \
  --optim adamw_bnb_8bit \
  --learning-rate 3e-5 --lr-scheduler constant_with_warmup --warmup-steps 500 \
  --save-fraction-of-epoch 0.25 --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen citmc --downstream-batch-size 32 --top-k-downstream 7 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v31_sft_mix18_ropext8048_3e \
  --wandb-tags sft da v31 mix18 ropext8048 if-v4 mc-letters task-expansion rc reason textman arc obqa constant-lr epochs-3 flatten-packing \
  --no-torch-compile
