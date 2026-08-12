#!/usr/bin/env bash
# Danish SFT v32 — v31 recipe with cosine anneal instead of constant, 3 epochs.
#
# Changes vs v31:
#   --lr-scheduler constant_with_warmup → cosine_with_min_lr
#     Peak LR 3e-5 (same), warmup 500 (same), decays cosine to 3e-6 (10% of peak)
#   --epochs 4 → 3  (v31 ep 3.0 was the peak; ep 4 overfit)
#
# v31 (constant LR, 4ep) results, for reference:
#   ep 3.0 peak: agg 0.258 (best single ckpt)
#   avg-top3:    agg 0.272 (best model)
#   avg-top7:    agg 0.268
# v31 saw eval_loss uptick from step 46293 onward (mid-ep4 overfit under
# constant LR). Cosine annealing across the 3 epochs should let the model
# consolidate late without needing epoch 4.
#
# Everything else identical: same base, same 18-dataset mix, same batching,
# same eval cadence, same top-7 preservation.
set -euo pipefail
cd /root/espllm

export PATH="$HOME/.local/bin:$PATH"

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext8048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /workspace/runs/sft/da_v32_mix18_ropext8048_cosine_4e \
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
  --learning-rate 3e-5 --lr-scheduler cosine_with_min_lr --warmup-steps 500 \
  --save-fraction-of-epoch 0.25 --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen citmc arc_easy arc_challenge \
  --downstream-batch-size 32 --top-k-downstream 7 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v32_sft_mix18_ropext8048_cosine_4e \
  --wandb-tags sft da v32 mix18 ropext8048 if-v4 mc-letters task-expansion rc reason textman arc obqa cosine_with_min_lr epochs-3 flatten-packing \
  --no-torch-compile
