#!/usr/bin/env bash
# Danish SFT v28 — v25 mix on ropext2048 base + all v27 training improvements.
#
# Isolates the training-pipeline impact vs v25 by using v25's EXACT data
# (no tool-calls) but under v27's improved pipeline:
#   - Base: jensjepsen/danish-lm-400m-base-ropext2048-v1 (was ckpt-310k for v25)
#   - Flatten-packing via DataCollatorWithFlattening + FA2 varlen
#     (v25 used custom pre-packing with cross-sample attention contamination)
#   - Per-sample RoPE positions (v25 had position drift across packed samples)
#   - Effective batch = 128 rows / opt-step ≈ 18k tokens/step (matches v25)
#   - warmup 500 (matches v27's step-density-adjusted value)
#   - torch_compile off (Liger fused_linear_cross_entropy bombs dynamo)
#   - flash-attn 2.8.3 wheel (trivial on H100 via setup_vastai.sh)
#
# vs v27 the only change is: DROP jensjepsen/danish-tool-calls-v1:sft:train.
# So this is v27's recipe with v25's data. Direct answer to "is v27's
# downstream regression caused by tool-calls in the mix, or by base/pipeline?"
#
# Data (13 sources, matching v25):
#   1. metamath-gsm            — 235k DA GSM word problems
#   2. algebra-sft-v5-mixed    — 300k procedural solve chains
#   3. arith-chain-sft-v1      — 100k arith simplification
#   4. wiki-grounded-sft-v3    — 95k grounded Q/A
#   5. text-to-question-v2     — 34k text→question
#   6. sciq:train              — 13.7k science MC
#   7. gsm8k:train             — 8.8k DA GSM8K
#   8. instruction-following-v4:train — 119k IF w/ 46 constraints
#   9. wiki-closedqa-v1        — 132k factual Q/A
#  10. word-problems-v2        — 240k compositional WPs
#  11. wiki-closedqa-stem-v1   — 78k STEM Q/A
#  12. wiki-broadqa-stem-v1    — broad-Q STEM
#  13. wiki-mc-letters-v1      — 210k MC letter/digit label emission
#
# Runtime estimate on 1×H100 80GB (bs=64 ga=2 = 128 rows/opt-step):
#   ~1.5-2h for 3 epochs (~12k steps/epoch × 3 = 36k opt steps).
#
# For 5090 32GB: drop --batch-size 64 → 16, raise --gradient-accumulation
# 2 → 8 to keep effective batch. Expect ~3.5-4h wall.

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
  --output-dir /root/runs/sft/da_v28_mix13_v25data_ropext_3e \
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
  --epochs 3 --batch-size 128 --gradient-accumulation 1 \
  --learning-rate 3e-5 --lr-scheduler constant_with_warmup --warmup-steps 500 \
  --save-fraction-of-epoch 0.25 --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen citmc --downstream-batch-size 32 --top-k-downstream 3 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v28_sft_mix13_v25data_ropext_3e \
  --wandb-tags sft da v28 mix13 v25data ropext2048 if-v4 mc-letters constant-lr epochs-3 flatten-packing pipeline-ablation \
  --no-torch-compile
