#!/usr/bin/env bash
# v30 CONTINUATION — resume from checkpoint-49938 (3.0 ep) → 4.0 ep.
#
# Adds one more epoch on top of the v30 mix19 run to see if downstream
# keeps climbing past 2.25 (v30 peak, agg 0.328). Constant LR (3e-5)
# already baked past warmup, so continuation is seamless.
#
# MUST continue same wandb run (41u9zx6o) — via --wandb-run-id.
# HF Trainer resume picks up checkpoint-49938 automatically from output-dir.
#
# ~1.4h wall on H100 for the extra 16646 steps.
set -euo pipefail
cd /root/espllm

export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

export HF_HOME=/root/hf-cache
export HF_DATASETS_CACHE=/root/hf-cache/datasets

OUT=/workspace/runs/sft/da_v30_mix19_stemreason_ropext_3e
# prep_cache symlink already in place, tokenised splits reused verbatim.

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext2048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir "$OUT" \
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
    jensjepsen/danish-sci-reasoning-v1 \
    jensjepsen/danish-sci-factcheck-v1 \
    jensjepsen/danish-sci-taskgen-v1 \
  --epochs 4 --batch-size 128 --gradient-accumulation 1 \
  --optim adamw_bnb_8bit \
  --learning-rate 3e-5 --lr-scheduler constant_with_warmup --warmup-steps 500 \
  --save-fraction-of-epoch 0.25 --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen citmc piqa arc gpqa textman_summary textman_rewrite \
  --downstream-batch-size 256 --top-k-downstream 7 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v30_sft_mix19_stemreason_ropext_3e \
  --wandb-run-id 41u9zx6o \
  --wandb-tags sft da v30 mix19 ropext2048 if-v4 mc-letters task-expansion stem-reasoning stem-factcheck stem-taskgen constant-lr epochs-4 flatten-packing continuation \
  --no-torch-compile \
  --resume
