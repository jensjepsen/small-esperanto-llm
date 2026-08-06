#!/usr/bin/env bash
# Danish RoPE-extension continued pretrain: 512 → 2048 context.
#
# Loads jensjepsen/danish-lm-400m-base-ckpt310k, bumps rope_theta
# 10000 → 500000 and max_position_embeddings 512 → 2048, then continues
# pretraining on Danish long-doc data for ~500M tokens to teach the
# attention weights to use positions past 512.
#
# Inline eval via LongShortPerplexityCallback every 250 steps:
#   - eval/short_nll  (positions [0, 512)) — regression watchdog
#   - eval/long_nll   (positions [512, 2048)) — extension progress
#   - eval/long_short_ratio — target ≤ 1.15 by end of run
#
# Baseline (NTK-only, no continued pretrain):
#   short_ppl 13.4, long_ppl 972, ratio 2.65
# Target end-of-run:
#   short_ppl ≤ 15 (≤5% regression), long_ppl ≤ 20, ratio ≤ 1.15
#
# Data: reuses the existing 512-chunked danish-pretokenized-16k dataset.
# The streaming `_chunk_stream` function concatenates the 512 chunks then
# re-chunks at max_length=2048 (from the YAML), so no new data prep
# needed — 4× 512-chunks are packed into each 2048 sequence. Note: this
# is fineweb2-dominated stitched-web-page data. Good enough for RoPE
# frequency adaptation; if long-range reasoning quality on downstream
# tasks disappoints, revisit with a long-doc-filtered pretokenized set.
#
# Runtime: ~1.5-2h on 1×H100 80GB (bs=32, grad_ckpt off, torch_compile on).
# For 5090 or other smaller cards, drop batch and add grad_accum + turn
# grad_checkpointing on — expect 8-12h wall.
#
# Launch:
#   OUTPUT_DIR=/workspace/runs/da_ropext_2048_v1 bash scripts/launch_da_ropext_2048.sh

set -euo pipefail
cd "${ESPLLM_ROOT:-/root/espllm}"

export WANDB_PROJECT=danish-lm-ropext
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

torchrun --nproc_per_node=1 -m esperanto_lm.train \
  --config da_400m_ropext_2048 \
  --from-pretrained jensjepsen/danish-lm-400m-base-ckpt310k \
  --rope-extend-theta 500000 \
  --min-doc-tokens 2048 \
  --long-short-eval \
  --long-short-eval-docs 32 \
  --long-short-eval-short-len 512 \
  --long-short-eval-batch-size 16 \
  --output-dir "${OUTPUT_DIR:-/workspace/runs/da_ropext_2048_v1}" \
  --tokenizer jensjepsen/danish-tokenizer \
  --pretokenized-dataset \
    jensjepsen/danish-pretokenized-16k \
  --no-use-benchmarks
