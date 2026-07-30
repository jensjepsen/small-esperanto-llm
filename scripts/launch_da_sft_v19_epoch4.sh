#!/usr/bin/env bash
# Danish SFT v19 — 4th "epoch" from v18/final on same constant LR (3e-5).
#
# Story so far:
#   v16 = 3ep constant LR 3e-5, base=danish-lm-400m-base-ckpt310k (peak
#         agg 0.183 at ckpt-29436, wandered off peak; final 0.177)
#   v18 = 0.2ep linear anneal 1e-5→0 from v16-29436 (beat v12: agg 0.195)
#   v19 = 1 more epoch of constant 3e-5 from v18/final. Isolates whether
#         the model has more headroom on plain constant training after
#         the anneal reset the loss surface, or whether v18's ~epoch-2.75
#         anneal was the true peak.
#
# Continues v16's wandb run (id=1l1ak676) with step offset 34263 so the
# chart is one contiguous constant-LR line past the v18 anneal detour.
#
# Runtime ~3.5h on 5090 (10.7k steps + 4 downstream evals).

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

CKPT=${CKPT:-/root/runs/sft/da_v18_anneal_from_v16_peak/final}

uv run python -u scripts/train_sft_packed.py \
  --checkpoint "$CKPT" \
  --tokenizer "$CKPT" \
  --output-dir /root/runs/sft/da_v19_epoch4_constlr \
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
  --epochs 1 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --learning-rate 3e-5 \
  --max-length 512 \
  --lr-scheduler constant_with_warmup \
  --warmup-steps 0 \
  --save-fraction-of-epoch 0.25 \
  --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen \
  --downstream-batch-size 32 \
  --top-k-downstream 3 \
  --wandb-project danish-lm-sft \
  --wandb-run-id 1l1ak676 \
  --wandb-step-offset 34263 \
  --wandb-tags sft da v19 epoch4-constlr from-v18-final continues-v16 lr-3e-5 full-set top3
