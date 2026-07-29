#!/usr/bin/env bash
# Danish SFT v15 — anneal from v14 epoch-3.0 peak checkpoint.
#
# Motivation: v14 (5-epoch probe) peaked around epoch 3.0 (GSM 24.5) then
# degraded (GSM crashed to 16.5 at 3.25, drifted 17-20 through epoch 4.5).
# Hypothesis: at that point of training, LR was still too high for stable
# late-stage refinement. Instead of continuing to overtrain, resume from
# the peak and do a short, aggressive LR-decay anneal to consolidate.
#
# Recipe:
#   - Resume from /root/v14_backup/checkpoint-32112-epoch3 (GSM peak)
#   - Small anneal budget: 0.2 epochs (~2,140 steps ~= 12 min compute)
#   - Peak LR 1e-5 (5× lower than v14's 5e-5)
#   - Linear decay to 0 with zero warmup
#   - Same 12-source mix
#   - Frequent downstream evals (every 500 steps = ~5 evals total)
#     to see convergence trajectory in real time
#
# Runtime ~15-25 min total (compute + eval overhead).

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

CKPT=${CKPT:-/root/v14_backup/checkpoint-32112-epoch3}

uv run python -u scripts/train_sft_packed.py \
  --checkpoint "$CKPT" \
  --tokenizer "$CKPT" \
  --output-dir /root/runs/sft/da_v15_anneal_from_v14_e3 \
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
  --epochs 0.2 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --learning-rate 1e-5 \
  --max-length 512 \
  --lr-scheduler linear \
  --warmup-steps 0 \
  --save-steps 400 \
  --save-total-limit 6 \
  --downstream-evals gsm8k sciq citgen \
  --downstream-n 200 \
  --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v15_anneal_from_v14_e3 \
  --wandb-tags sft da v15 anneal from-v14-e3 short lr-1e-5 linear-decay
