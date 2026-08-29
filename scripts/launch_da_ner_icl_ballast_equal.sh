#!/usr/bin/env bash
# Danish NER + ICL + EQUAL-SIZE v31 ballast.
#
# A/B against da_ner_icl_ballast_prop: identical structured data (55k), same
# total ballast (55,008 vs 54,999), same hyperparameters. Only the ballast
# COMPOSITION differs -- 3,056 rows from each of the 18 sources instead of
# 2.9% of each.
#
# What that changes, and why it is the right next test. Proportional gave the
# big procedural sets most of the budget and the small benchmark-aligned ones
# almost none:
#     danish-gsm8k       214 -> 3,056   (14x)
#     danish-arc         193 -> 3,056   (16x)
#     danish-openbookqa  284 -> 3,056
#     danish-sciq        335 -> 3,056   (9x)
#     danish-algebra   8,596 -> 3,056   (down)
#     word-problems    6,877 -> 3,056   (down)
#     metamath-gsm     6,733 -> 3,056   (down)
#     mc-letters       6,038 -> 3,056   (down)
# So this separates two explanations for the proportional run's partial gsm8k
# recovery (26.0 base -> 17.0 at ep1 -> 11.0 at ep3): general math VOLUME from
# the large procedural sets, versus gsm8k-SHAPED rows specifically. If equal
# beats proportional on gsm8k/sciq/arc, shape matters; if it loses, volume did
# the work and hand-weighting is the wrong lever.
#
# Every source has more rows than the cap, so no source is truncated by
# availability and the comparison is clean.
#
# NOTE: the ifeval detector now scores IFEval-DA inst-strict, comparable to the
# v31 card (35.2) and to a same-session base reading of 37.8%. The proportional
# run used the older mean-constraint-fraction metric, so its ifeval series is
# NOT comparable to this one -- the other four detectors are.
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
  --output-dir /root/runs/da_ner_icl_ballast_equal \
  --no-morpheme-preprocess \
  --sft-data \
    jensjepsen/danish-ner-sft-v1:sft:train \
    jensjepsen/danish-icl-schema-format-v3:sft:train \
    jensjepsen/danish-metamath-gsm:sft:train:3056 \
    jensjepsen/danish-algebra-sft-v5-mixed:default:train:3056 \
    jensjepsen/danish-arith-chain-sft-v1:default:train:3056 \
    jensjepsen/danish-wiki-grounded-sft-v3:sft:train:3056 \
    jensjepsen/danish-text-to-question-v2:sft:train:3056 \
    jensjepsen/danish-sciq:sft:train:3056 \
    jensjepsen/danish-gsm8k:sft:train:3056 \
    jensjepsen/danish-instruction-following-v4:sft:train:3056 \
    jensjepsen/danish-wiki-closedqa-v1:sft:train:3056 \
    jensjepsen/danish-word-problems-v2:default:train:3056 \
    jensjepsen/danish-wiki-closedqa-stem-v1:sft:train:3056 \
    jensjepsen/danish-wiki-broadqa-stem-v1:sft:train:3056 \
    jensjepsen/danish-wiki-mc-letters-v1:default:train:3056 \
    jensjepsen/danish-rc-v1:default:train:3056 \
    jensjepsen/danish-reason-v1:default:train:3056 \
    jensjepsen/danish-textman-v1:default:train:3056 \
    jensjepsen/danish-arc:sft:train:3056 \
    jensjepsen/danish-openbookqa:sft:train:3056 \
  --epochs 3 --batch-size 32 --gradient-accumulation 1 \
  --optim adamw_bnb_8bit \
  --learning-rate 1e-5 --lr-scheduler constant_with_warmup --warmup-steps 50 \
  --max-length 3072 \
  --flatten-packing \
  --save-fraction-of-epoch 0.5 --eval-fraction-of-epoch 0.5 \
  --save-total-limit 8 \
  --downstream-evals gsm8k citgen sciq ifeval icl --downstream-n 200 --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_ner_icl_ballast_equal_bs32ga1_lr1e-5 \
  --wandb-tags sft da ner icl span-wrap ballast equal-replay constant-lr epochs-3 h100 \
  --no-torch-compile
