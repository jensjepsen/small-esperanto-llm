#!/usr/bin/env bash
# Danish ICL v1 — continued SFT on schema-induction rows only.
#
# Data: jensjepsen/danish-icl-json-v1:sft:train (8,469 rows). Each row packs
# 1-5 worked examples sharing a JSON schema into ONE user turn with no
# instruction, so the schema must be induced from the examples; ~half the rows
# replace field names with meaning-free symbols (symbol tuning).
#
# ICL-ONLY, no ballast from the v31 mix. That risks pulling format behaviour
# toward "always emit JSON", so downstream gsm8k/citgen run every eval step
# at n=200 purely as a collapse detector, and checkpoints are saved 3x per
# epoch so an earlier one can be picked offline.
#
# vs the v31 recipe (which produced the base):
#   - lr 3e-5 -> 1e-5      continuing from a converged average, not fresh SFT
#   - warmup 500 -> 50     the whole run is ~795 steps
#   - eff_bs 128 -> 32     8.5k rows is 40x smaller than the v31 mix; a
#                          smaller batch buys steps (265/epoch vs 66)
#   - seq 8048 -> 3072     longest row is 2,874 tokens
#   - save_total_limit 2 -> 10, selection by offline exact-match not SWA
#     (avg-top3 costs ~2pp on format-heavy metrics; schema induction is
#     scored by exact match, so best-single is the right target)
#
# TOKENIZER: the v31 CHECKPOINT's tokenizer, not jensjepsen/danish-tokenizer.
# The checkpoint carries 16007 tokens with the chat tokens at 16000-16002
# matching the model's vocab_size; the bare danish-tokenizer is 16000 and maps
# <|user|>/<|assistant|>/<|end|> to unk.
set -euo pipefail
cd /root/espllm

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/tmp/hf-cache
export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

CKPT=jensjepsen/danish-lm-400m-sft-v31-avg-top3

uv run python -u scripts/train_sft_packed.py \
  --checkpoint "$CKPT" \
  --tokenizer "$CKPT" \
  --output-dir /root/runs/da_icl_v1 \
  --no-morpheme-preprocess \
  --sft-data jensjepsen/danish-icl-json-v1:sft:train \
  --epochs 3 --batch-size 8 --gradient-accumulation 4 \
  --optim adamw_bnb_8bit \
  --learning-rate 1e-5 --lr-scheduler constant_with_warmup --warmup-steps 50 \
  --max-length 3072 \
  --flatten-packing \
  --save-fraction-of-epoch 0.34 --save-total-limit 10 \
  --downstream-evals gsm8k citgen --downstream-n 200 --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_icl_v1_only_bs8ga4_lr1e-5 \
  --wandb-tags sft da icl symbol-tuning json-schema icl-only constant-lr epochs-3 flatten-packing \
  --no-torch-compile
