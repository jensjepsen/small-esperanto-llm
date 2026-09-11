#!/usr/bin/env bash
# Danish SFT tool-only, SYMBOLIZED DATA ONLY -- 5090.
#
# Arm B of a corpus A/B against launch_da_sft_toolmix.sh. Same base, same
# hyperparameters, same evals; the only variable is the training data.
#
# THE QUESTION. Over the full eval_unseen_tools split of proc-v1 at
# da_sft_toolmix step-1816, 7.1% of calls invented a key name -- `vessel_id_filter`
# for `vessel_identifier`, `sky_map` for `data_upload_destination`. The model
# has a parameter vocabulary from training and reaches for it instead of
# reading the schema in front of it. Symbolized rows replace every parameter
# and return-field name with a per-row random symbol, so there is nothing to
# recall and the Danish description is the only route from question to slot.
# Training on ONLY those asks whether that route can be made the default.
#
# THE DATA. jensjepsen/danish-tool-dialogues-symonly-v1 (private), 30,166 rows:
# 17,057 v9 twins + 13,109 proc twins. Those are exactly the twins that sit in
# each corpus's published TRAIN split -- not every twin in the local files,
# which also cover eval rows and would leak the eval sets into training.
# Verified: 7,586 distinct argument keys across a 4,000-row sample, none longer
# than 6 characters or containing an underscore.
#
# NO ABSTENTION CONFIG, unlike toolmix. Those rows are not symbolized, so
# including them would break the one-variable claim. The cost is that
# tool_refusal has no training signal here and should sit near its 50% floor
# all run -- that is expected, not a regression, and it is why tool_refusal
# must NOT be read as an A/B result.
#
# EVALS ARE UNCHANGED, deliberately -- all 11, non-symbolized included. The
# GAP between tool_unseen and tool_unseen_sym is the measurement. Scoring only
# the symbolized halves would say whether the model got better at symbols
# while hiding whether it got worse at everything else.
#
# STEPS. 30,166 rows at effective batch 64 is 471 steps/epoch, so 8 epochs is
# 3,772 -- about half the toolmix run. Do not read absolute numbers against
# toolmix: different data, different step count. Read the sym/non-sym gap.
#
# START THE CHECKPOINT WATCHER FIRST:
#     HF_TOKEN=$(cat /root/hf_token) BEST_SUBDIR=best nohup \
#       bash scripts/watch_push_best_ckpts.sh /root/runs/da_sft_symonly \
#         jensjepsen/danish-lm-400m-sft-symonly-mid 180 &
set -euo pipefail
cd "${ESPLLM_ROOT:-/root/espllm}"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=${HF_HOME:-/tmp/hf-cache}
export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8
export ESPLLM_LIGER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TOOL_DATA=${TOOL_REPO:-jensjepsen/danish-tool-dialogues-v9}
PROC_DATA=${PROC_REPO:-jensjepsen/danish-tool-dialogues-proc-v1}
SYM_DATA=${SYM_REPO:-jensjepsen/danish-tool-dialogues-symonly-v1}
# Eval repos stay on the UNsymbolized corpora: the evals read their own
# *_sym splits from these, so both halves are scored without the training
# data having to supply either.
export ESPLLM_TOOL_EVAL_REPO=$TOOL_DATA
export ESPLLM_TOOL_EVAL_REPO_B=$PROC_DATA

uv run --no-sync python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext8048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir "${OUT_DIR:-/root/runs/da_sft_symonly}" \
  --no-morpheme-preprocess \
  --attn-impl flash_attention_2 \
  --sft-data "${SYM_DATA}:sft:train" \
  --epochs 8 --batch-size 8 --gradient-accumulation 8 \
  --optim adamw_bnb_8bit \
  --learning-rate 3e-5 --lr-scheduler constant_with_warmup --warmup-steps 300 \
  --max-length 3584 \
  --flatten-packing \
  --torch-compile \
  --save-fraction-of-epoch 0.5 --eval-fraction-of-epoch 0.5 \
  --save-total-limit 3 --top-k-downstream 3 \
  --downstream-evals tool_seen tool_unseen tool_seen_sym tool_unseen_sym \
                     tool_answer tool_refusal \
                     tool_seen_b tool_unseen_b tool_seen_sym_b \
                     tool_unseen_sym_b tool_answer_b \
  --downstream-n 150 --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name "${RUN_NAME:-da_sft_symonly_v1}" \
  --wandb-tags sft da toolonly symonly corpus-ab adam8bit fa2 torch-compile \
               no-liger epochs-8 5090 \
  "$@"
