#!/usr/bin/env bash
# Danish SFT v27 — v26 mix on ropext2048 base, with flatten-packing.
#
# One change vs v26:
#   FLATTEN PACKING (default in train_sft_packed.py after commit 0a4a8ec).
#   Uses transformers.DataCollatorWithFlattening → FA2 varlen kernel.
#   Effects:
#     - No cross-sample attention contamination (block-diagonal via cu_seqlens)
#     - Per-sample RoPE (each conversation's tokens start at position 0)
#     - No padding waste (concatenates rows exactly to their lengths)
#     - ~14× more optimizer updates per epoch on the same data
#       (v26: 3502 steps/epoch → v27 with 128-row effective batch: ~12.7k
#       steps/epoch. Matches v25's step density that worked well.)
#
# Retuned for flatten:
#   --batch-size 16          rows/micro-batch (NOT packed 2048-seqs like v26)
#   --gradient-accumulation 8    → 128 rows / opt step ≈ 18k tokens/step
#                                  (comparable to v25's 16k tok/step regime;
#                                   v26 was 65k tok/step, too large-batch)
#   --warmup-steps 500       ~2% of first epoch (v26's 200 was <1% now)
#   Everything else identical to v26.
#
# H100 variant: bump per_device to 64 (--batch-size 64 --gradient-accumulation 2)
# to keep the same 128 rows/opt step but with more parallelism per micro-batch.
#
# Runtime estimate:
#   5090 (bs=16 ga=8): ~4-5h per epoch (varlen speedup partly offsets more
#     opt steps). Total ~12-15h for 3 epochs. Similar to v26 despite more
#     updates because per-step is faster.
#   H100 (bs=64 ga=2): ~1.5-2h per epoch → ~5-6h total.
#
# Data: identical 14 sources to v26 (13 v25 sources + tool-calls-v1).
# Base: ropext2048-v1 (autodetect picks max_length=2048).

set -euo pipefail
cd /root/espllm

# uv install target from setup_vastai.sh — nohup shells don't inherit it.
export PATH="$HOME/.local/bin:$PATH"

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext2048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v27_mix14_flatten_3e \
  --no-morpheme-preprocess \
  --sft-data \
    jensjepsen/danish-metamath-gsm:sft \
    jensjepsen/danish-algebra-sft-v5-mixed \
    jensjepsen/danish-arith-chain-sft-v1 \
    jensjepsen/danish-wiki-grounded-sft-v3:sft \
    jensjepsen/danish-text-to-question-v2:sft \
    jensjepsen/danish-sciq:sft:train \
    jensjepsen/danish-gsm8k:sft:train \
    jensjepsen/danish-instruction-following-v4:sft:train \
    jensjepsen/danish-wiki-closedqa-v1:sft \
    jensjepsen/danish-word-problems-v2 \
    jensjepsen/danish-wiki-closedqa-stem-v1:sft \
    jensjepsen/danish-wiki-broadqa-stem-v1:sft \
    jensjepsen/danish-wiki-mc-letters-v1 \
    jensjepsen/danish-tool-calls-v1:sft:train \
  --epochs 3 \
  --batch-size 16 \
  --gradient-accumulation 8 \
  --learning-rate 3e-5 \
  --lr-scheduler constant_with_warmup \
  --warmup-steps 500 \
  --save-fraction-of-epoch 0.25 \
  --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen citmc \
  --downstream-batch-size 32 \
  --top-k-downstream 3 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v27_sft_mix14_flatten_3e \
  --wandb-tags sft da v27 mix14 ropext2048 if-v4 tool-calls mc-letters constant-lr epochs-3 flatten-packing torch-compile downstream-eval full-set top3
