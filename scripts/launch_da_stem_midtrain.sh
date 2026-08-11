#!/usr/bin/env bash
# Danish 400M STEM mid-train: continued pretrain on translated vital-5 STEM.
#
# Loads jensjepsen/danish-lm-400m-base-ropext2048-v1 and does 5 epochs of
# next-token pretrain on jensjepsen/danish-vital-stem-da-tokenized-v1
# (10k articles, 44.6M DA tokens from vital-5 STEM EN→DA via flash-lite).
#
# Live-tracked via MCLogprobCallback:
#   eval/sciq_mc_logprob  — length-normed log P scoring on 200 SciQ items
#   eval/citmc_logprob    — length-normed log P scoring on 300 citmc items
# Base sciq_mc_logprob: ~42.5%. Target: any measurable lift.
#
# Runtime: ~25-40 min on 1×H100 80GB. Cost: ~$1-2 at RunPod rates.
#
# Launch:
#   OUTPUT_DIR=/workspace/runs/da_stem_midtrain_v1 bash scripts/launch_da_stem_midtrain.sh
set -euo pipefail
cd "${ESPLLM_ROOT:-/root/espllm}"

export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=danish-lm-pretrain
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run torchrun --nproc_per_node=1 -m esperanto_lm.train \
  --config da_400m_stem_midtrain \
  --from-pretrained jensjepsen/danish-lm-400m-base-ropext2048-v1 \
  --mc-logprob-eval \
  --output-dir "${OUTPUT_DIR:-/workspace/runs/da_stem_midtrain_v1}" \
  --tokenizer jensjepsen/danish-tokenizer \
  --pretokenized-dataset \
    jensjepsen/danish-vital-stem-da-tokenized-v1 \
  --no-use-benchmarks
