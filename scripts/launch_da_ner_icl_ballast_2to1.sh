#!/usr/bin/env bash
# Danish NER + ICL + 2:1 proportional v31 ballast.
#
# 109,999 ballast rows against 55k structured -- double the 1:1 run, same
# proportional composition, same lr 1e-5. Ballast VOLUME is the only variable
# left untested, and the two levers tried before it both failed:
#
#   composition   equal-size lost on every metric at epoch 1 (gsm8k 11.5 vs
#                 17.0, icl 21.0 vs 36.5). Giving gsm8k 14x more rows made
#                 gsm8k WORSE, because it cost 13k rows of procedural math.
#   step size     lr 5e-6 lost on every metric at epoch 1 (gsm8k 14.0 vs 17.0,
#                 icl 20.0 vs 36.5) -- retention did not improve and
#                 acquisition slowed.
#
# Both failing the same way says the erosion tracks exposure to the
# structured-output distribution rather than how fast or how it is mixed, so
# the remaining lever is how much of the original distribution is replayed
# alongside it. 1:1 recovered roughly a third of the gsm8k loss (26.0 base ->
# 17.0 at epoch 1); this asks whether 2:1 recovers proportionally more or
# whether it plateaus too.
#
# Structured data is unchanged at 55k, so NER/ICL cannot be confounded. Total
# ~165k rows, ~5,150 steps/epoch, so this is roughly 45 minutes rather than 25.
# Ballast has been close to free on the structured tasks so far (NER exact 55.7
# vs 54.5, span-wrap faithful 65% vs 62% between no-ballast and 1:1), which is
# what makes more of it cheap to try.
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
  --output-dir /root/runs/da_ner_icl_ballast_2to1 \
  --no-morpheme-preprocess \
  --sft-data \
    jensjepsen/danish-ner-sft-v1:sft:train \
    jensjepsen/danish-icl-schema-format-v3:sft:train \
    jensjepsen/danish-algebra-sft-v5-mixed:default:train:17191 \
    jensjepsen/danish-word-problems-v2:default:train:13753 \
    jensjepsen/danish-metamath-gsm:sft:train:13466 \
    jensjepsen/danish-wiki-mc-letters-v1:default:train:12076 \
    jensjepsen/danish-wiki-closedqa-v1:sft:train:7588 \
    jensjepsen/danish-reason-v1:default:train:6849 \
    jensjepsen/danish-instruction-following-v4:sft:train:6842 \
    jensjepsen/danish-textman-v1:default:train:6800 \
    jensjepsen/danish-arith-chain-sft-v1:default:train:5730 \
    jensjepsen/danish-wiki-grounded-sft-v3:sft:train:5461 \
    jensjepsen/danish-rc-v1:default:train:4573 \
    jensjepsen/danish-wiki-closedqa-stem-v1:sft:train:4488 \
    jensjepsen/danish-text-to-question-v2:sft:train:1969 \
    jensjepsen/danish-wiki-broadqa-stem-v1:sft:train:1162 \
    jensjepsen/danish-sciq:sft:train:669 \
    jensjepsen/danish-openbookqa:sft:train:568 \
    jensjepsen/danish-gsm8k:sft:train:428 \
    jensjepsen/danish-arc:sft:train:386 \
  --epochs 3 --batch-size 32 --gradient-accumulation 1 \
  --optim adamw_bnb_8bit \
  --learning-rate 1e-5 --lr-scheduler constant_with_warmup --warmup-steps 50 \
  --max-length 3072 \
  --flatten-packing \
  --save-fraction-of-epoch 0.5 --eval-fraction-of-epoch 0.5 \
  --save-total-limit 8 \
  --downstream-evals gsm8k citgen sciq ifeval icl --downstream-n 0 --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_ner_icl_ballast_2to1_bs32ga1_lr1e-5 \
  --wandb-tags sft da ner icl span-wrap ballast proportional-replay ratio-2to1 constant-lr epochs-3 h100 \
  --no-torch-compile
