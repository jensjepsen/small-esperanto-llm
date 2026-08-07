#!/usr/bin/env bash
# Resume v28 from checkpoint-35292 (end of ep 3) for 2 additional epochs.
# Continues into the SAME output_dir + wandb run (id=s09cb79r) so trainer_state,
# optimizer.pt, scheduler.pt, and rng_state.pth are all restored — no lr warm
# start, no dataloader reshuffle drift, no wandb split.
#
# --epochs 5 → HF Trainer recomputes max_steps ≈ 58820, continues from step
# 35292. Constant_with_warmup scheduler is already past warmup (500 steps),
# stays at 3e-5 for remaining ~23528 steps (≈ 1h 15min on H100).
set -euo pipefail
cd /root/espllm

export PATH="$HOME/.local/bin:$PATH"

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext2048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v28_mix13_v25data_ropext_3e \
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
  --epochs 5 --batch-size 128 --gradient-accumulation 1 \
  --learning-rate 3e-5 --lr-scheduler constant_with_warmup --warmup-steps 500 \
  --save-fraction-of-epoch 0.25 --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen citmc --downstream-batch-size 32 --top-k-downstream 3 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v28_sft_mix13_v25data_ropext_3e \
  --wandb-run-id s09cb79r \
  --wandb-tags sft da v28 mix13 v25data ropext2048 if-v4 mc-letters constant-lr epochs-5 flatten-packing pipeline-ablation resume \
  --no-torch-compile \
  --resume
