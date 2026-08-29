#!/usr/bin/env bash
# Danish SFT v6 (mix9-wp-reworded) — v5 with wp-v2 SWAPPED for wp-reworded-v1.
#
# Delta from v5 mix9-if2:
#   SWAP  jensjepsen/danish-word-problems-v2:sft (240,000 rows)
#      →  jensjepsen/danish-word-problems-reworded-v1:sft (207,312 rows)
#
# Same recipe answers, but questions rewritten from wp-v2's template surface
# ("Vi går baglæns", "(75% af den største andel)", etc.) into natural
# GSM8K-style Danish. Isolates the surface-form effect: any v6 vs v5 delta
# is purely attributable to the reword, not to added training volume.
#
# Rationale: v5 GSM8K probe showed the model can execute wp-v2 recipes when
# fed the recipe surface form, but fails to recognise the recipe in natural
# GSM8K-style prose ("En butik sælger sko til 800 kr med 20% rabat, hvad var
# original pris?"). Swap teaches only the natural → recipe mapping.
#
# CAVEAT: reworded dataset has ~10-15% subtle semantic drift the regex filter
# didn't catch (fraction vs ratio conflation, hver↔tilsammen). Judge cleanup
# deferred to v7 if v6 shows GSM8K lift worth improving on.
#
# All other 8 sources kept identical to v5. Same optimizer/schedule.
# Total ~1,056k rows across 9 sources (was ~1,088k in v5).
# Output written to overlay (/root), not /workspace — pod-quota isolation.

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ckpt310k \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v6_mix9wpreword \
  --no-morpheme-preprocess \
  --sft-data \
    jensjepsen/danish-word-problems-reworded-v1:sft \
    jensjepsen/danish-metamath-gsm:sft \
    jensjepsen/danish-algebra-sft-v5-mixed \
    jensjepsen/danish-arith-chain-sft-v1 \
    jensjepsen/danish-wiki-grounded-sft-v3:sft \
    jensjepsen/danish-text-to-question-v2:sft \
    jensjepsen/danish-sciq:sft:train \
    jensjepsen/danish-gsm8k:sft:train \
    jensjepsen/danish-instruction-following-v2:sft:train \
  --epochs 2 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --learning-rate 5e-5 \
  --max-length 512 \
  --lr-scheduler cosine_with_min_lr \
  --warmup-steps 200 \
  --save-fraction-of-epoch 0.25 \
  --save-total-limit 3 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v6_sft_mix9wpreword \
  --wandb-tags sft da v6 mix9 no-morpheme if-v2 wp-reworded-swap no-alpaca no-dolly
