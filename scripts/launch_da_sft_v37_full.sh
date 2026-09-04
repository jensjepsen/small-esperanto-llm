#!/usr/bin/env bash
# Danish SFT v37 — v36's mix with ONE change:
#
#   jensjepsen/danish-tool-dialogues-v2  ->  -v3
#
# WHAT v3 FIXES: tool SELECTION was neither taught nor measured.
#
# The source corpus lists the tool that gets CALLED first in 98.4% of
# multi-tool rows, and 69.7% of rows offer only one tool at all. So "call the
# first catalogued tool" scored 99.2% right-tool on the eval -- beating the
# trained model's 84-93%. The ordering leaked the answer and there was almost
# never a choice to make.
#
# The model learned exactly that. Given a hand-built catalogue of four novel
# tools it called the FIRST one for every question -- dice-rolling for a query
# about coffee, dice-rolling for dog years -- slot-filling whatever numbers
# appeared in the prompt. Correct behaviour, given the corpus.
#
# It also explains the tool_unseen > tool_seen inversion seen across five
# measurements: unseen simply has more single-tool catalogues (74.0% vs
# 71.4%). It was never evidence of generalisation.
#
# v3 shuffles each catalogue and pads it to 6 tools with distractors drawn
# only from non-held-out tools. Measured on the published corpus:
#     mean catalogue     1.29 -> 6.00
#     gold listed first 99.5% -> 16.8%   (chance 16.7%)
# Render-time only: same translations, no API cost.
#
# READ right-tool AGAINST 16.8%, NOT AGAINST v36's 84-93%. Those numbers were
# measured on a task where the answer was in position 0; these are measured on
# a real 1-in-6 choice, so a DROP is expected and is not a regression. argF1
# is likewise now conditioned on a genuine selection.
#
# NINE EVALS, including tool_answer (grounded 80.6% at v36 step-22662, against
# a gold ceiling of 83.1%). Aggregates are means over nine evals and are not
# comparable with v35's eight-eval values.
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
export ESPLLM_TOOL_EVAL_REPO=jensjepsen/danish-tool-dialogues-v3

# --no-sync: WORKLOAD=sft pins torch<2.9 so a prebuilt FA2 wheel matches, but
# the `all` extra still declares vllm>=0.17 (torch>=2.10), so a plain `uv run`
# re-resolves and dies before training starts.
uv run --no-sync python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext8048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/da_sft_v37_full \
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
    jensjepsen/danish-tool-dialogues-v3:sft:train \
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
  --wandb-run-name da_sft_v37_full_mix22_tooldialogues_v3 \
  --wandb-tags sft da v37 full-resft mix22 tool-dialogues-v3 catalogue-shuffle distractors \
               multiturn textman-v2 extraction adam8bit fa2 torch-compile \
               no-liger epochs-3 h100 \
  "$@"
