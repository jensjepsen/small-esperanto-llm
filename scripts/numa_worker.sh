#!/usr/bin/env bash
# Per-rank NUMA binding worker invoked by torchrun.
# Reads LOCAL_RANK, binds to the NUMA node hosting that GPU, then execs Python.
set -euo pipefail

# Edit this table if `nvidia-smi topo -m` shows a different layout
GPU_TO_NUMA=(0 0 1 1)

rank="${LOCAL_RANK:-0}"
node="${GPU_TO_NUMA[$rank]}"

echo "[rank $rank] binding to NUMA node $node"
exec numactl --cpunodebind="$node" --membind="$node" \
  python -m esperanto_lm.train "$@"
