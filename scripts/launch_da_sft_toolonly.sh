#!/usr/bin/env bash
# Danish SFT tool-ONLY -- a 5090-sized rig for corpus A/Bs.
#
# NOT a publishable run, and not comparable to tooldev or v41. This exists to
# answer questions about the TOOL CORPUS cheaply: symbolized ratio, distractor
# density, description-dependence. Everything else is stripped.
#
# WHY. tooldev already cut the mix from 2.09M rows to 725k and still spends
# 95% of its compute on non-tool data to answer tool questions:
#
#   config                       rows     steps   tool%   wall
#   v41 as run              2,085,687    48,883   0.86%   6.7h  H100
#   tooldev                   725,264    33,999   4.71%   4.6h  H100
#   THIS                       35,070     4,384    100%   ~1.5h 5090
#
# 20x fewer rows than tooldev and it fits on a consumer card.
#
# MAX-LENGTH 2816, NOT 8048. Measured over the v9 tool corpus: median 1,226
# tokens, p99 2,238, max 2,639. Nothing is truncated and the activation
# ceiling drops ~3x, which is what makes 32GB workable.
#
# BATCH 16 x GA 4 -- effective 64.
#
# With --flatten-packing, --batch-size counts SAMPLES, not tokens, and the
# tooldev run measured ~0.98GB of activations per sample above ~4GB of weights
# and 8-bit optimizer state. 16 samples is ~20GB of 32GB, leaving headroom;
# batch 32 would be ~35GB and would not fit. Hold TOKENS constant, not samples
# -- the same trap that OOMed tooldev at batch 128.
#
# EVAL ONCE PER EPOCH, AT n=250 -- NOT every half-epoch on the full sets.
#
# The arithmetic bites in the other direction here. 547 steps/epoch means a
# half-epoch cadence is 16 eval blocks, and a full-set block on six tool evals
# takes ~7 minutes -- 1.9h of eval against 1.5h of training. Once per epoch at
# n=250 is 8 blocks of ~2.5min: ~20 minutes, which is a sensible fraction of
# the run.
#
# The cost is noise. 250 of ~700 rows per split will move a point or two
# between blocks, so read the TREND across epochs, not any single block, and
# confirm a winner at full n before believing a small gap.
#
# NOTE the eval-steps clamp: eval_steps = max(save_steps, ...), so lowering
# --eval-fraction-of-epoch alone does nothing. Both fractions move together.
#
# 8 EPOCHS, NOT 3. 35k rows at effective batch 64 is 548 steps/epoch, and 3
# epochs would be 1,644 steps -- too few for the warmup (500) to even finish
# mattering. 8 epochs gives 4,384.
#
# FROM THE BASE MODEL, NOT AN SFT CHECKPOINT.
#
# Continued training from, say, v9 step-33996 would be cheaper still and would
# preserve general Danish -- but every corpus A/B would then be measured on a
# model that has ALREADY seen distractors and symbolized twins, which is the
# variable under test. Starting from base keeps each arm independent. The cost
# is that non-tool metrics are meaningless here: this model will be bad at
# gsm8k and ifeval by construction, and they are not scored.
#
# WHAT THIS CANNOT TELL YOU. Absolute numbers are not comparable to tooldev --
# different data, different steps, different starting behaviour. Only compare
# tool-only runs to each other, and only on the tool metrics.
#
# BLACKWELL / sm_120 -- VERIFY BEFORE COMMITTING A RUN:
#   * torch must be the cu128 build; a cu121 wheel crashes on sm_120, and a
#     `git reset --hard` on a pod reverts the pyproject edit that pins it.
#   * --flatten-packing REFUSES without flash-attention. Prebuilt FA2 wheels
#     for sm_120 are thin on the ground. If FA2 is unavailable the fallback is
#     --no-flatten-packing, which reintroduces cross-sample attention
#     (measured 8.36 leakage vs FA2's 0.0) and changes what the run means.
#     Check `import flash_attn` BEFORE launching, not after.
#   * bf16 is native on Blackwell, unlike the 1080 Ti.
#
# START THE CHECKPOINT WATCHER FIRST, not after:
#     HF_TOKEN=$(cat /root/hf_token) BEST_SUBDIR=best nohup \
#       bash scripts/watch_push_best_ckpts.sh /root/runs/da_sft_toolonly \
#         jensjepsen/danish-lm-400m-sft-toolonly-mid 180 &
set -euo pipefail
cd "${ESPLLM_ROOT:-/root/espllm}"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=${HF_HOME:-/tmp/hf-cache}
export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8
export ESPLLM_LIGER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The eval repo MUST match the training corpus. Pointing it at an older
# version is what made v36 read 9pp worse than v35 purely for being
# self-inconsistent.
TOOL_DATA=${TOOL_REPO:-jensjepsen/danish-tool-dialogues-v9}
export ESPLLM_TOOL_EVAL_REPO=$TOOL_DATA

uv run --no-sync python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext8048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir "${OUT_DIR:-/root/runs/da_sft_toolonly}" \
  --no-morpheme-preprocess \
  --attn-impl flash_attention_2 \
  --sft-data "${TOOL_DATA}:sft:train" "${TOOL_DATA}:abstention:train" \
  --epochs 8 --batch-size 16 --gradient-accumulation 4 \
  --optim adamw_bnb_8bit \
  --learning-rate 3e-5 --lr-scheduler constant_with_warmup --warmup-steps 300 \
  --max-length 2816 \
  --flatten-packing \
  --torch-compile \
  --save-fraction-of-epoch 1.0 --eval-fraction-of-epoch 1.0 \
  --save-total-limit 3 --top-k-downstream 3 \
  --downstream-evals tool_seen tool_unseen tool_seen_sym tool_unseen_sym \
                     tool_answer tool_refusal \
  --downstream-n 250 --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name "${RUN_NAME:-da_sft_toolonly_v9}" \
  --wandb-tags sft da toolonly corpus-ab adam8bit fa2 torch-compile \
               no-liger epochs-8 5090 \
  "$@"
