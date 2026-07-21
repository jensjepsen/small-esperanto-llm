#!/usr/bin/env bash
# Danish 400M pretrain driver: launches torchrun with per-rank NUMA binding.
# Assumes 2-GPU-per-socket layout (RunPod H100 SXM5 default):
#   GPU 0,1 -> NUMA 0, GPU 2,3 -> NUMA 1
# Verify with `nvidia-smi topo -m` before running.

set -euo pipefail

# Activate uv-managed venv so `python` in the NUMA worker resolves correctly
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

: "${WANDB_PROJECT:=danish-lm-pretrain}"
: "${WANDB_NAME:=v1_400m_$(date +%Y%m%d_%H%M)}"
export WANDB_PROJECT WANDB_NAME

# HF cache on the fast overlay (workspace has a 30GB quota — don't use)
: "${HF_HOME:=/tmp/hf-cache}"
: "${HF_DATASETS_CACHE:=$HF_HOME/datasets}"
: "${HF_HUB_ENABLE_HF_TRANSFER:=1}"
export HF_HOME HF_DATASETS_CACHE HF_HUB_ENABLE_HF_TRANSFER

# HF token from /tmp/hf-cache/token (or already-exported HF_TOKEN)
if [[ -z "${HF_TOKEN:-}" && -f "$HF_HOME/token" ]]; then
  HF_TOKEN="$(cat "$HF_HOME/token")"
  export HF_TOKEN
fi

exec torchrun --nproc_per_node=4 --nnodes=1 \
  scripts/numa_worker.sh \
  --config h100_400m_4gpu_da \
  --tokenizer jensjepsen/danish-tokenizer \
  --pretokenized-dataset \
    jensjepsen/danish-pretokenized-16k \
    jensjepsen/danish-pretokenized-16k-supp \
    jensjepsen/danish-math-pretokenized-16k \
  --no-use-benchmarks \
  --output-dir /tmp/runs/v1_danish_400m \
  "$@"
