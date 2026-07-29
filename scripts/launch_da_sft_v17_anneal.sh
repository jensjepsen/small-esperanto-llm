#!/usr/bin/env bash
# Danish SFT v17 — anneal from v16 final ckpt (step-32112, epoch 3.0).
#
# v16 (3ep constant 3e-5) trajectory: agg peaked at ckpt-29436 (0.183, ep2.75),
# final ckpt-32112 fell back to 0.177 — first sign of over-fit tail. Try
# short linear-decay anneal at 1e-5 (3× lower than v16's constant LR) to see
# if reduced-LR polishing on the last ckpt beats the mid-run peak.
#
# Recipe (mirrors v15's anneal-from-v14-e3 pattern):
#   - Resume from best/step-32112-agg-0.177 (has optimizer+scheduler state)
#   - 0.2 epochs (~2140 steps ≈ 12 min compute + 5 downstream evals)
#   - Peak LR 1e-5, linear decay to 0, zero warmup
#   - Full-set downstream + top-3 preservation
#   - Save every 400 steps (~5 anneal checkpoints)
#
# Runtime ~15-25 min total.

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

CKPT=${CKPT:-/root/runs/sft/da_v16_mix12_3e_constlr/best/step-32112-agg-0.177}

uv run python -u scripts/train_sft_packed.py \
  --checkpoint "$CKPT" \
  --tokenizer "$CKPT" \
  --output-dir /root/runs/sft/da_v17_anneal_from_v16_e3 \
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
  --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen \
  --downstream-batch-size 32 \
  --top-k-downstream 3 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v17_anneal_from_v16_e3 \
  --wandb-tags sft da v17 anneal from-v16-e3 short lr-1e-5 linear-decay full-set top3
