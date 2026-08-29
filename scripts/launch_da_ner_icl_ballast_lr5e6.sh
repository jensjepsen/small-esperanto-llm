#!/usr/bin/env bash
# Danish NER + ICL + proportional v31 ballast at HALF the learning rate.
#
# lr 1e-5 -> 5e-6. Learning rate is the other lever on forgetting, and the
# evidence so far says composition is not the one worth pulling:
#
#   proportional ballast, lr 1e-5   gsm8k 17.0 -> 12.5 -> 11.0  (base 26.0)
#   equal-size ballast,   lr 1e-5   gsm8k 11.5 at epoch 1, icl 21.0 vs 36.5
#
# Equal-size lost on every comparable metric, so re-weighting toward the
# benchmark-aligned sets is the wrong knob -- general procedural volume was
# doing the work. What proportional showed instead was a trajectory: roughly
# base-level gsm8k after one epoch, eroding steadily after. That is the shape
# of a step size too large to sit still, not of a mix that lacks something.
#
# Ballast composition is IDENTICAL to the proportional run, so lr is the only
# variable. Eval every half epoch, since the interesting movement happened
# between the proportional run's whole-epoch points.
#
# Expect slower acquisition of the structured tasks too; if NER/ICL land well
# short at 3 epochs, the follow-up is more epochs at this lr rather than a
# return to 1e-5.
#
# vs da_ner_icl_v1 (same structured data, no ballast), which reached 55.7%
# exact / 68.2 entity-F1 on NER eval and 65% span-wrap faithfulness, but cost
# gsm8k 17->8.0% and citgen 28.5->21.5% at n=200:
#   + ~55,000 ballast rows sampled from the SAME 18-dataset mix v31 was
#     trained on, so this is literal replay of the distribution that produced
#     the abilities being lost, not a guess at what matters.
#   Caps are strictly PROPORTIONAL to each source's size (sum 54,999 of the
#     mix's 1,919,572 rows, ~2.9% of each). Simple first: no re-weighting
#     toward whatever is dropping hardest.
#
# A consequence of proportional worth watching: gsm8k contributes only 214
# rows (0.4%) and sciq 335, because they are small datasets -- while gsm8k is
# the metric falling furthest. If the drop only partly recovers, that is the
# first place to look.
#
# Structured data is held CONSTANT and ballast is added on top rather than
# displacing it, so a change in NER/ICL scores cannot be confounded with
# having removed the rows that taught them. Total ~110k rows, ~3,440
# steps/epoch at eff_bs 32.
#
# Detectors now include ifeval and icl alongside gsm8k/citgen/sciq: two
# metrics could not show whether ballast helps across the board, and IF was
# drifting unmeasured despite being the ability most similar to structured
# output.
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
  --output-dir /root/runs/da_ner_icl_ballast_lr5e6 \
  --no-morpheme-preprocess \
  --sft-data \
    jensjepsen/danish-ner-sft-v1:sft:train \
    jensjepsen/danish-icl-schema-format-v3:sft:train \
    jensjepsen/danish-algebra-sft-v5-mixed:default:train:8596 \
    jensjepsen/danish-word-problems-v2:default:train:6877 \
    jensjepsen/danish-metamath-gsm:sft:train:6733 \
    jensjepsen/danish-wiki-mc-letters-v1:default:train:6038 \
    jensjepsen/danish-wiki-closedqa-v1:sft:train:3794 \
    jensjepsen/danish-reason-v1:default:train:3424 \
    jensjepsen/danish-instruction-following-v4:sft:train:3421 \
    jensjepsen/danish-textman-v1:default:train:3400 \
    jensjepsen/danish-arith-chain-sft-v1:default:train:2865 \
    jensjepsen/danish-wiki-grounded-sft-v3:sft:train:2730 \
    jensjepsen/danish-rc-v1:default:train:2286 \
    jensjepsen/danish-wiki-closedqa-stem-v1:sft:train:2244 \
    jensjepsen/danish-text-to-question-v2:sft:train:984 \
    jensjepsen/danish-wiki-broadqa-stem-v1:sft:train:581 \
    jensjepsen/danish-sciq:sft:train:335 \
    jensjepsen/danish-openbookqa:sft:train:284 \
    jensjepsen/danish-gsm8k:sft:train:214 \
    jensjepsen/danish-arc:sft:train:193 \
  --epochs 3 --batch-size 32 --gradient-accumulation 1 \
  --optim adamw_bnb_8bit \
  --learning-rate 5e-6 --lr-scheduler constant_with_warmup --warmup-steps 50 \
  --max-length 3072 \
  --flatten-packing \
  --save-fraction-of-epoch 0.5 --eval-fraction-of-epoch 0.5 \
  --save-total-limit 8 \
  --downstream-evals gsm8k citgen sciq ifeval icl --downstream-n 200 --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_ner_icl_ballast_prop_bs32ga1_lr5e-6 \
  --wandb-tags sft da ner icl span-wrap ballast proportional-replay constant-lr lr5e-6 epochs-3 h100 \
  --no-torch-compile
