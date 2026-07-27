#!/usr/bin/env bash
# Danish SFT v9 (mix11-if-v3-wp-reworded-rephrase) — v8 base with IF-v3
# swap, wp-reworded-v1 re-added, and a rephrase-instruction dataset added.
#
# Deltas from v8:
#   SWAP jensjepsen/danish-instruction-following-v2:sft:train
#     →  jensjepsen/danish-instruction-following-v3:sft:train
#     v3 = 109,485 train + 1,000 eval, 40 constraints (v2 had 29),
#     LLM-paraphrase cache (719 pairs × 30 phrasings) applied at prob=0.7
#     during generation. Motivation: v6 IF probe measured 27pp gap between
#     IN-DIST (82%) and OOD phrasings (55%); v3's paraphrase diversity
#     targets that gap directly. See [[reference-danish-if]].
#
#   ADD  jensjepsen/danish-word-problems-reworded-v1
#     207k natural-language rewrites of danish-word-problems-v2 questions
#     via Gemma (~10-15% hidden drift, un-judged). Dropped from v7/v8
#     during ablation ([[project-synth-wp-effect]] showed dropping BOTH
#     wp datasets cost IF held-out -4.5pp with no GSM8K change). Adding
#     reworded-v1 alone (not wp-v2) gives natural-language surface variety
#     without the templated question shapes that made wp-v2 identifiable.
#
#   ADD  jensjepsen/danish-rephrase-wp-v1
#     ~392k rephrase-instruction rows built from wp-reworded-v1's
#     (q_orig, q_new) pairs. Both directions (orig→new AND new→orig) with
#     ~80 varied instruction templates (imperative/polite/question/roleplay).
#     Fills the "rephrase this sentence" capability gap probed on v6
#     (which currently returns verbatim copies on rephrase asks). Math-WP
#     domain only for now — general-purpose rephrase deferred pending v9
#     eval.
#
# Kept: (v8's 8 sources) metamath-gsm, algebra-v5, arith-chain-v1,
# wiki-grounded-v3, text-to-question-v2, sciq, gsm8k, wiki-closedqa-v1
# — plus the three changes above = 11 total sources.
#
# Same optimizer/schedule as v5/v6/v7/v8.
# Total ~1.5M rows across 11 sources (v8 was ~984k).
# Output written to overlay (/root), not /workspace — pod-quota isolation.

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ckpt310k \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v9_mix11_ifv3_wpreworded_rephrase \
  --no-morpheme-preprocess \
  --sft-data \
    jensjepsen/danish-metamath-gsm:sft \
    jensjepsen/danish-algebra-sft-v5-mixed \
    jensjepsen/danish-arith-chain-sft-v1 \
    jensjepsen/danish-wiki-grounded-sft-v3:sft \
    jensjepsen/danish-text-to-question-v2:sft \
    jensjepsen/danish-sciq:sft:train \
    jensjepsen/danish-gsm8k:sft:train \
    jensjepsen/danish-instruction-following-v3:sft:train \
    jensjepsen/danish-wiki-closedqa-v1:sft \
    jensjepsen/danish-word-problems-reworded-v1 \
    jensjepsen/danish-rephrase-wp-v1:sft \
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
  --wandb-run-name da_v9_sft_mix11_ifv3_wpreworded_rephrase \
  --wandb-tags sft da v9 mix11 no-morpheme if-v3 wp-reworded rephrase-wp wiki-closedqa
