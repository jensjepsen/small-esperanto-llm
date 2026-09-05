#!/usr/bin/env bash
# Danish SFT v38 — v37's mix with ONE change:
#
#   jensjepsen/danish-tool-dialogues-v3  ->  -v4
#
# WHAT v4 ADDS: a `returns` block on every tool spec.
#
# Specs described their INPUTS and said nothing about their OUTPUT -- 0 of
# 86,304 carried a returns key. That is survivable for an English-reading
# model, which can guess `cups_left` holds remaining cups. This model is
# Danish-only: `cups_left` tokenises to ['c','ups','_','le','ft'], fragments
# with no meaning in its space, one of which ('og') is Danish for "and". So an
# unfamiliar result was unreadable and the model could only fall back on
# tool->result pairings memorised in training.
#
# Measured exactly that way on v37: grounding near-gold on SEEN tools
# (tool_answer 77-80% against an 83.1% ceiling) but guesswork on novel ones --
# probed on an unseen get_coffee_status it answered with `last_service` instead
# of `cups_left`, and on calculate_dog_years never reported dog_years at all.
#
# Input parameters already carried a Danish description each, which is why
# argument binding works (argF1 74.7%). v4 gives results the same channel:
# 1,724 (tool, field) descriptions, capped at 6 per spec, row-present first.
#
# v4 also fixes untranslated nested parameter descriptions -- 89.3% of them
# read "The length of the shape" inside an otherwise Danish spec, across 7.6%
# of rows, because segments() never recursed into a parameter that is itself
# an object. Present since v1.
#
# READ tool_answer FIRST. It is the metric v4 targets, and the one with a
# published ceiling: gold replies score 83.1%, a reply lifted from another row
# 3.4%, fluent Danish with no facts 0.0%. v37 ended around 77-80%.
# right-tool should be read against ~24.7% chance, NOT against v36's 84-93%,
# which were measured when the answer sat in catalogue position 0.
#
# v37 for comparison (nine-eval aggregate, same eval set):
#   step-26439 agg 0.422 | tool_seen argF1 74.7 | tool_unseen 57.1
#   tool_answer 77.4 | right-tool 88.2 / 73.6
#
# FLASH-ATTENTION IS REQUIRED (flatten-packing refuses without it):
#     WORKLOAD=sft bash scripts/setup_vastai.sh large
set -euo pipefail
cd /root/espllm

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/tmp/hf-cache
export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8
export ESPLLM_LIGER=0          # mutually exclusive with torch.compile
# Score against the corpus this run TRAINS on. Pointing the tool eval at an
# older corpus is what made v36 read 9pp worse than v35 purely for being
# self-consistent -- the yardstick was the defect.
export ESPLLM_TOOL_EVAL_REPO=jensjepsen/danish-tool-dialogues-v4

# --no-sync: WORKLOAD=sft pins torch<2.9 so a prebuilt FA2 wheel matches, but
# the `all` extra still declares vllm>=0.17 (torch>=2.10), so a plain `uv run`
# re-resolves and dies before training starts.
uv run --no-sync python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext8048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/da_sft_v38_full \
  --no-morpheme-preprocess \
  --attn-impl flash_attention_2 \
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
    jensjepsen/danish-textman-v2 \
    jensjepsen/danish-arc:sft:train \
    jensjepsen/danish-openbookqa:sft:train \
    jensjepsen/danish-ner-sft-v1:sft:train \
    jensjepsen/danish-icl-schema-format-v3:sft:train \
    jensjepsen/danish-extraction-v1:sft:train \
    jensjepsen/danish-tool-dialogues-v4:sft:train \
  --source-cap danish-extraction-v1=60000 \
  --epochs 3 --batch-size 128 --gradient-accumulation 1 \
  --optim adamw_bnb_8bit \
  --learning-rate 3e-5 --lr-scheduler constant_with_warmup --warmup-steps 500 \
  --max-length 8048 \
  --flatten-packing \
  --torch-compile \
  --save-fraction-of-epoch 0.25 --eval-fraction-of-epoch 0.25 \
  --save-total-limit 3 --top-k-downstream 3 \
  --downstream-evals gsm8k citgen sciq ifeval icl extraction tool_seen tool_unseen \
                     tool_answer \
  --downstream-n 0 --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_sft_v38_full_mix22_tooldialogues_v4 \
  --wandb-tags sft da v38 full-resft mix22 tool-dialogues-v4 returns-schema \
               multiturn textman-v2 extraction adam8bit fa2 torch-compile \
               no-liger epochs-3 h100 \
  "$@"
