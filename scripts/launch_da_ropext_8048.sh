#!/usr/bin/env bash
# Danish RoPE-extension continued pretrain: 2048 → 8048 (4× on top of ropext2048).
#
# Loads jensjepsen/danish-lm-400m-base-ropext2048-v1, bumps rope_theta
# 500000 → 2000000 (matching 4× extension), then continues pretraining on
# Danish long-doc data.
#
# Also attaches MCLogprobCallback (sciq/citmc/arc) for continuous MC signal
# during the extension — mostly as sanity that MC quality doesn't collapse
# from the rope adaptation.
#
# Runtime: ~30-45min on 1×H100 80GB.
set -euo pipefail
cd "${ESPLLM_ROOT:-/root/espllm}"

export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=danish-lm-pretrain
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run torchrun --nproc_per_node=1 -m esperanto_lm.train \
  --config da_400m_ropext_8048 \
  --from-pretrained jensjepsen/danish-lm-400m-base-ckpt310k \
  --rope-extend-theta 3000000 \
  --min-doc-tokens 8048 \
  --long-short-eval \
  --long-short-eval-docs 128 \
  --long-short-eval-short-len 2048 \
  --long-short-eval-batch-size 4 \
  --mc-logprob-eval \
  --mc-logprob-n-sciq 1000 \
  --mc-logprob-n-citmc 720 \
  --mc-logprob-n-arc 1167 \
  --output-dir "${OUTPUT_DIR:-/workspace/runs/da_ropext_8048_v1}" \
  --tokenizer jensjepsen/danish-tokenizer \
  --pretokenized-dataset \
    jensjepsen/danish-pretokenized-16k \
  --no-use-benchmarks
