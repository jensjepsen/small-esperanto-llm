#!/usr/bin/env bash
# Danish SFT v8 (mix9-nowp-wikiclosedqa) — v7 base + wiki-closedqa-v1 added.
#
# Delta from v7 mix8-nowp:
#   ADD  jensjepsen/danish-wiki-closedqa-v1:sft (132,419 rows)
#        — dense factual Q/A generated from 11,345 salience-filtered Danish
#        Wikipedia articles via Gemma-3-12b. ~11.7 Q/A per article, focused
#        on countries, capitals, historical figures, science, biography,
#        geography, culture. See [[reference-danish-wiki-closedqa]].
#
# Rationale: v6 baseline probes on this dataset's questions showed 1/15
# correct — model confidently confabulates on general-knowledge factuals
# ("Peter Pan skabt af Frank Miller", "efter Tiberius regerede Augustus",
# "Beyond the Realms of Death skrevet af Mick Jagger og Keith Richards").
# Cit-MC has been capped at ~55% and cit-gen at ~15% across v4→v7 —
# knowledge-bound, not capability-bound. This dataset targets the ceiling
# with dense factual grounding across ~11k mainstream topics.
#
# Kept: (v7's 8 sources) metamath-gsm, algebra-v5, arith-chain-v1,
# wiki-grounded-v3, text-to-question-v2, sciq, gsm8k, instruction-following-v2.
# Still dropped: both danish-word-problems-* (v7 ablation showed no math impact).
#
# Same optimizer/schedule as v5/v6/v7.
# Total ~984k rows across 9 sources.
# Output written to overlay (/root), not /workspace — pod-quota isolation.

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ckpt310k \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v8_mix9nowp_wikiclosedqa \
  --no-morpheme-preprocess \
  --sft-data \
    jensjepsen/danish-metamath-gsm:sft \
    jensjepsen/danish-algebra-sft-v5-mixed \
    jensjepsen/danish-arith-chain-sft-v1 \
    jensjepsen/danish-wiki-grounded-sft-v3:sft \
    jensjepsen/danish-text-to-question-v2:sft \
    jensjepsen/danish-sciq:sft:train \
    jensjepsen/danish-gsm8k:sft:train \
    jensjepsen/danish-instruction-following-v2:sft:train \
    jensjepsen/danish-wiki-closedqa-v1:sft \
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
  --wandb-run-name da_v8_sft_mix9nowp_wikiclosedqa \
  --wandb-tags sft da v8 mix9 no-morpheme if-v2 no-wp wiki-closedqa
