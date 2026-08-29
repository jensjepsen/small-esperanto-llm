#!/usr/bin/env bash
# Danish SFT v2 (mix8) launch script.
#
# Diff from v1_mix7:
#   - drop dolly (American-context bleed)
#   - drop alpaca-cleaned (same)
#   - replace word-problems-v1 → v2 (compose + reverse + idiom bank)
#   - add danish-metamath-gsm (234,992 math word problems)
#   - add danish-wiki-grounded-sft-v2 (49,869 Danish-anchored SFT)
#   - add danish-text-to-question (23,280 T→Q, addresses generation gap)
#   - epochs 3 → 2 (v1 overfit at ep 2.25)
#   - --wandb-project set explicitly (env var was overridden by argparse
#     default in v1, landed in wrong wandb project)
#
# Mix (~967k rows, ~90% math):
#   word-problems-v2 (240k) + metamath-gsm (235k) +
#   algebra-v5 (300k) + arith-chain (100k) = 875k math
#   wiki-grounded-v2 (50k) + text-to-question (23k) +
#   sciq:train (12k) + gsm8k:train (7k) = 92k language/task

set -euo pipefail

cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ckpt310k \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /workspace/runs/sft/da_v2_mix8 \
  --no-morpheme-preprocess \
  --sft-data \
    jensjepsen/danish-word-problems-v2:sft \
    jensjepsen/danish-metamath-gsm:sft \
    jensjepsen/danish-algebra-sft-v5-mixed \
    jensjepsen/danish-arith-chain-sft-v1 \
    jensjepsen/danish-wiki-grounded-sft-v2:sft \
    jensjepsen/danish-text-to-question:sft \
    jensjepsen/danish-sciq:sft:train \
    jensjepsen/danish-gsm8k:sft:train \
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
  --wandb-run-name da_v2_sft_mix8 \
  --wandb-tags sft da v2 mix8 no-morpheme no-alpaca no-dolly metamath-added
