#!/usr/bin/env bash
# Danish SFT v14 (mix12-if-v3-wpv2-stem-broad, 5 epochs) — schedule probe.
#
# Same 12-source mix as v12/v13. Goal: gauge where the returns stop by
# running longer and monitoring downstream evals during training (via the
# new DownstreamEvalCallback — eval_downstream_{gsm8k,sciq,citgen} metrics
# every quarter-epoch). Ships /final = last ckpt (load_best_model_at_end
# default flipped OFF post-v12).
#
# Rationale for 5 epochs: v12 (3ep) beat v11 (2ep) despite an eval-loss
# spike after epoch 2. Downstream metrics kept improving through epoch 3.
# 5 epochs tells us if the trend continues, plateaus, or reverses. With
# in-training downstream evals we'll SEE the trajectory instead of guessing.
#
# Cosine LR still — could switch to constant/WSD later if we see gains.
# Total ~1.37M rows × 5 epochs = ~6.8M example passes. Runtime ~5h on 5090.

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ckpt310k \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v14_mix12_5e \
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
  --epochs 5 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --learning-rate 5e-5 \
  --max-length 512 \
  --lr-scheduler cosine_with_min_lr \
  --warmup-steps 200 \
  --save-fraction-of-epoch 0.25 \
  --save-total-limit 4 \
  --downstream-evals gsm8k sciq citgen \
  --downstream-n 200 \
  --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v14_sft_mix12_5e \
  --wandb-tags sft da v14 mix12 no-morpheme if-v3 wp-v2 wiki-closedqa stem stem-broad epochs-5 downstream-eval schedule-probe
