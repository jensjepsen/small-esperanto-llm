#!/usr/bin/env bash
# Per-rank NUMA binding worker invoked by torchrun.
# Reads LOCAL_RANK, binds to the NUMA node hosting that GPU, then execs Python.
set -euo pipefail

# Edit this table if `nvidia-smi topo -m` shows a different layout
GPU_TO_NUMA=(0 0 1 1)

rank="${LOCAL_RANK:-0}"
node="${GPU_TO_NUMA[$rank]}"

echo "[rank $rank] binding to NUMA node $node (CPU only; memory follows via kernel local-alloc)"
# --membind requires CAP_SYS_NICE which RunPod containers don't grant.
# CPU affinity alone gets ~95% of the benefit — kernel allocates memory
# on the NUMA node of the CPU touching it (local-alloc default).
exec numactl --cpunodebind="$node" \
  python -m esperanto_lm.train "$@"
