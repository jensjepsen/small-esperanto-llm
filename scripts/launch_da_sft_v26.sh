#!/usr/bin/env bash
# Danish SFT v26 — v25 mix + tool-calls, on the RoPE-extended 2048 base.
#
# Two changes vs v25:
#   1) BASE: jensjepsen/danish-lm-400m-base-ropext2048-v1 (was ckpt310k).
#      Same weights initialised from ckpt310k then continued-pretrained for
#      164M tokens with rope_theta 10k→500k and max_position_embeddings
#      512→2048. Held-out short_ppl 13.72→8.72 (-36%), long_ppl 748→17.42
#      (-98%), cit-MC preserved at 57.64% (base 57.78%). Enables the whole
#      tool-call dataset to fit without truncation (p95 ≈ 1900 tokens).
#   2) DATA: adds jensjepsen/danish-tool-calls-v1:sft:train (37,032 rows,
#      "separated" 5-msg format: user → assistant(reasoning) → tool_call
#      → tool_result → assistant(followup)). Teaches:
#        - Function selection under 1-5 tool catalog
#        - Argument grounding: literal, inferred (7pm→"19:00", "fire"→4), enums
#        - Refuse when catalog doesn't match; clarify when required arg missing
#        - Multi-round tool_result consumption + Danish followup
#
# --max-length is intentionally OMITTED: train_sft_packed.py autodetects
# it from model.config.max_position_embeddings (commit 8f622ce). So this
# script trains at 2048 automatically on the ropext base; if someone
# points it at the old ckpt310k it'd train at 512 automatically instead.
#
# Recipe otherwise IDENTICAL to v25:
#   3 epochs constant_with_warmup LR 3e-5, warmup 200
#   Downstream: gsm / sciq / citgen / citmc every 0.25 epoch, full-set
#   Top-3 preservation for later avg-top3
#
# Runtime estimate:
#   1×5090 24GB (bs=8 × ga=4): ~4-6h per epoch → ~12-18h for 3 epochs.
#   1×H100 80GB: bump to bs=32 ga=1 (same effective batch), ~2-4h/epoch → ~6-12h.
# 5090 config below. bs=8 fits 24GB at seq=2048 with room to spare;
# effective batch stays at 32 to match v25 optimizer dynamics.
#
# Data sources (14 total — 13 from v25 + 1 new):
#   1. metamath-gsm            — 235k DA GSM WPs
#   2. algebra-sft-v5-mixed    — 300k procedural solve chains
#   3. arith-chain-sft-v1      — 100k arithmetic simplification
#   4. wiki-grounded-sft-v3    — 95k grounded Q/A
#   5. text-to-question-v2     — 34k text→question
#   6. sciq:train              — 13.7k science MC
#   7. gsm8k:train             — 8.8k DA GSM8K
#   8. instruction-following-v4:train — 119k IF w/ 46 constraints incl 6 google-aligned
#   9. wiki-closedqa-v1        — 132k factual Q/A
#  10. word-problems-v2        — 240k compositional WPs
#  11. wiki-closedqa-stem-v1   — 78k STEM Q/A
#  12. wiki-broadqa-stem-v1    — broad-Q STEM
#  13. wiki-mc-letters-v1      — 210k MC to teach letter/digit label emission
#  14. tool-calls-v1:sft:train — 37k Danish tool-call convos (NEW)

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext2048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v26_mix14_ropext2048_3e \
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
  --batch-size 8 \
  --gradient-accumulation 4 \
  --learning-rate 3e-5 \
  --lr-scheduler constant_with_warmup \
  --warmup-steps 200 \
  --save-fraction-of-epoch 0.25 \
  --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen citmc \
  --downstream-batch-size 32 \
  --top-k-downstream 3 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v26_sft_mix14_ropext2048_3e \
  --wandb-tags sft da v26 mix14 ropext2048 if-v4 tool-calls mc-letters constant-lr epochs-3 downstream-eval full-set top3
