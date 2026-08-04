#!/usr/bin/env bash
# Danish SFT v25 — v22 mix + IF-v3 → IF-v4 swap.
#
# Motivation: v22-avg-top3 hit ifeval-da prompt-strict 19.5% because
# IF-v3 had 40 constraints whose checkers subtly diverged from google's
# IFEval verifiers (highlight sections, repeat_prompt, two_responses,
# and 3 missing families: json/postscript/constrained_response). IF-v4
# (jensjepsen/danish-instruction-following-v4) adds 6 google-aligned
# constraints trained via gemini flash lite at 80% pass rate. Swapping
# should transfer ifeval-da scores meaningfully — targeted at the
# detectable_format (9.5%) and combination (1.6%) family gaps.
#
# Recipe otherwise identical to v22 (best DA baseline on MC formats):
#   3 epochs constant_with_warmup LR 3e-5, warmup 200, packed 512
#   Downstream: gsm / sciq / citgen / citmc every 0.25 epoch, full-set
#   Top-3 preservation for later avg-top3 (matches v16-avg / v22-avg win)
#
# Runtime ~5h on 5090 (v22 was 4h 15m + 1h eval overhead).

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ckpt310k \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v25_mix13_if_v4_3e \
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
  --epochs 3 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --learning-rate 3e-5 \
  --max-length 512 \
  --lr-scheduler constant_with_warmup \
  --warmup-steps 200 \
  --save-fraction-of-epoch 0.25 \
  --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen citmc \
  --downstream-batch-size 32 \
  --top-k-downstream 3 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v25_sft_mix13_if_v4_3e \
  --wandb-tags sft da v25 mix13 if-v4 mc-letters constant-lr epochs-3 downstream-eval full-set top3
