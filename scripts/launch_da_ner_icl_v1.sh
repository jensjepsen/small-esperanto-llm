#!/usr/bin/env bash
# Danish NER + ICL schema-format — continued SFT on v31.
#
# Two sources, both ours, both structured-output induction:
#   jensjepsen/danish-ner-sft-v1:sft:train             24,000 rows
#   jensjepsen/danish-icl-schema-format-v3:sft:train   33,933 rows
#
# They share a prompt shape and an output-format vocabulary but differ in the
# task: the ICL set induces a JSON-ish schema over synthetic passages, the NER
# set extracts entities from real Danish text and adds the SPAN-WRAP formats
# (reproduce the passage with entities tagged in place). Span-wrap has scored
# 0% on every checkpoint measured so far, and the NER data is the first that
# trains it directly.
#
# vs v3 (da_icl_v3_paired_bs16ga2_lr1e-5), which reached 41.5% exact on unseen
# schemas and 78.5% on a near-neighbour unseen format:
#   - +24k NER rows -> 57,933 total, 1,810 steps/epoch at eff_bs 32
#   - bs 32 x ga 1 rather than 16 x 2: same eff_bs, so the optimizer
#     trajectory and step count are unchanged, but the H100 has the headroom
#     and it halves the number of accumulation boundaries
#
# STILL NO GENERAL BALLAST. Both sources are structured-output tasks, so the
# v1-style format lock-in risk has not gone away -- it has only broadened from
# "always JSON" to "always structured". gsm8k/citgen at n=200 run as detectors
# but did NOT catch the lock-in last time (they ask plain questions); judge it
# with scripts/probe_da_lines.py / probe_da_spans.py afterwards.
#
# H100 NUMA: this box reports 2 nodes with the GPU on node 0. Unpinned runs on
# RunPod H100s have cost ~3x before, so the trainer is bound to node 0.
set -euo pipefail
cd /root/espllm

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/tmp/hf-cache
export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

CKPT=jensjepsen/danish-lm-400m-sft-v31-avg-top3
# Pick a pinning that this container actually permits. --membind needs
# set_mempolicy, which RunPod blocks ("Operation not permitted"), and under
# `set -e` that killed the run before a single step. Probe each candidate with
# `true` and take the first that works; no pinning is a valid outcome.
GPU_NODE=-1
NODE_FILE=$(ls -d /sys/bus/pci/devices/*/numa_node 2>/dev/null | head -1)
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr "A-Z" "a-z" | sed "s/^0000//")
[ -e "/sys/bus/pci/devices/0000$BUS/numa_node" ] &&   GPU_NODE=$(cat "/sys/bus/pci/devices/0000$BUS/numa_node")
PIN=""
if command -v numactl >/dev/null && [ "$GPU_NODE" -ge 0 ]; then
  CPUS=$(cat /sys/devices/system/node/node$GPU_NODE/cpulist 2>/dev/null || echo "")
  for cand in "numactl --cpunodebind=$GPU_NODE --membind=$GPU_NODE" \
              "numactl --cpunodebind=$GPU_NODE" \
              ${CPUS:+"taskset -c $CPUS"}; do
    if $cand true 2>/dev/null; then PIN="$cand"; break; fi
  done
fi
echo "pinning: ${PIN:-none}  (gpu numa node $GPU_NODE)"

$PIN uv run python -u scripts/train_sft_packed.py \
  --checkpoint "$CKPT" \
  --tokenizer "$CKPT" \
  --output-dir /root/runs/da_ner_icl_v1 \
  --no-morpheme-preprocess \
  --sft-data \
    jensjepsen/danish-ner-sft-v1:sft:train \
    jensjepsen/danish-icl-schema-format-v3:sft:train \
  --epochs 3 --batch-size 32 --gradient-accumulation 1 \
  --optim adamw_bnb_8bit \
  --learning-rate 1e-5 --lr-scheduler constant_with_warmup --warmup-steps 50 \
  --max-length 3072 \
  --flatten-packing \
  --save-fraction-of-epoch 0.5 --eval-fraction-of-epoch 1.0 \
  --save-total-limit 8 \
  --downstream-evals gsm8k citgen --downstream-n 200 --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_ner_icl_v1_bs32ga1_lr1e-5 \
  --wandb-tags sft da ner icl span-wrap multi-format symbol-tuning constant-lr epochs-3 flatten-packing h100 \
  --no-torch-compile
