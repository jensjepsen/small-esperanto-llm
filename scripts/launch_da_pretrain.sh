#!/usr/bin/env bash
# Danish 400M pretrain driver: launches torchrun with per-rank NUMA binding.
# Assumes 2-GPU-per-socket layout (RunPod H100 SXM5 default):
#   GPU 0,1 -> NUMA 0, GPU 2,3 -> NUMA 1
# Verify with `nvidia-smi topo -m` before running.

set -euo pipefail

: "${WANDB_PROJECT:=danish-lm-pretrain}"
: "${WANDB_NAME:=v1_400m_$(date +%Y%m%d_%H%M)}"
export WANDB_PROJECT WANDB_NAME

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
