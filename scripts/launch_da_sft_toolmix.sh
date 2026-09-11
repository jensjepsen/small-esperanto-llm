#!/usr/bin/env bash
# Danish SFT tool-ONLY, PROCEDURAL + TOOLMIND mix -- 5090.
#
# Same shape as launch_da_sft_toolonly.sh (see that file for why tool-only,
# why from base, why 8 epochs). Two things differ, both forced by the data.
#
# THE MIX. 13,109 procedural rows + ~35,070 ToolMind-derived rows. The two are
# built by opposite methods and fail differently: ToolMind is translated from
# Glaive, so its defects are translation defects, and the procedural corpus is
# code-generated structure with an LLM writing only prose, so its defects are
# schema defects. Training on both is the point -- neither alone covers the
# other's failure surface.
#
# MAX-LENGTH 3584, NOT 2816. Measured over the procedural corpus with the ckpt
# tokenizer: mean 1,950, median 1,919, max 3,453 -- against ToolMind's mean
# ~1,300 / max 2,639. At 2816, 9.7% of procedural rows truncate, and they
# truncate at the END, which is the assistant's answer turn: the trained
# target. 3584 clears the longest row. The memory cost is near zero because it
# only binds on that 9.7%.
#
# BATCH 16 x GA 4 -- effective 64, unchanged, so step counts stay comparable.
# The mixed corpus averages ~1,476 tokens/row against v9's ~1,300, so
# activations run ~14% above the toolonly calibration: ~22GB of 32GB rather
# than ~20GB. Still inside the card. Do NOT raise the batch to "use" the rest;
# hold TOKENS constant, not samples.
#
# EVAL REPO IS v9, NOT THE PROCEDURAL REPO. The six tool evals include
# tool_seen_sym / tool_unseen_sym, and symbolized twins exist only for v9 --
# symbolize_twins.py has not been run over the procedural build, so its
# eval_seen_sym / eval_unseen_sym splits are empty. Pointing the eval repo at
# the procedural corpus would silently score two of six evals on nothing.
# Scoring on v9 is honest but PARTIAL: it measures half the training mix, and
# the procedural corpus's own held-out splits (578 seen / 807 unseen) are not
# scored by this run. Read the numbers as "v9 performance after training on
# both", not as a corpus A/B.
#
# The procedural repo is PRIVATE, so the pod needs a token that can read it:
#   scp ~/.hf_token root@POD:/root/hf_token   then   export HF_TOKEN=$(cat ...)
#
# START THE CHECKPOINT WATCHER FIRST, not after:
#     HF_TOKEN=$(cat /root/hf_token) BEST_SUBDIR=best nohup \
#       bash scripts/watch_push_best_ckpts.sh /root/runs/da_sft_toolmix \
#         jensjepsen/danish-lm-400m-sft-toolmix-mid 180 &
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
export ESPLLM_TOOL_EVAL_REPO=$TOOL_DATA

uv run --no-sync python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext8048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir "${OUT_DIR:-/root/runs/da_sft_toolmix}" \
  --no-morpheme-preprocess \
  --attn-impl flash_attention_2 \
  --sft-data "${TOOL_DATA}:sft:train" "${TOOL_DATA}:abstention:train" \
             "${PROC_DATA}:sft:train" \
  --epochs 8 --batch-size 16 --gradient-accumulation 4 \
  --optim adamw_bnb_8bit \
  --learning-rate 3e-5 --lr-scheduler constant_with_warmup --warmup-steps 300 \
  --max-length 3584 \
  --flatten-packing \
  --torch-compile \
  --save-fraction-of-epoch 1.0 --eval-fraction-of-epoch 1.0 \
  --save-total-limit 3 --top-k-downstream 3 \
  --downstream-evals tool_seen tool_unseen tool_seen_sym tool_unseen_sym \
                     tool_answer tool_refusal \
  --downstream-n 250 --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name "${RUN_NAME:-da_sft_toolmix_proc1_v9}" \
  --wandb-tags sft da toolonly toolmix procedural adam8bit fa2 torch-compile \
               no-liger epochs-8 5090 \
  "$@"
