#!/usr/bin/env bash
# Danish SFT v12 (mix12-if-v3-wpv2-stem-broad) — v11 + broad-Q STEM added.
#
# Deltas from v11:
#   ADD  jensjepsen/danish-wiki-broadqa-stem-v1  (~19-20k kept Q/A rows)
#     Broad, explanatory Q/A (150-400 word answers) from 6,187 STEM+medicine
#     +env articles (Fysik/Kemi/Biologi/Astronomi/Geologi/Matematik +
#     Lægevidenskab/Sygdomme/Miljø at depth-2, asteroid + stub filtered).
#     4 rich Q/A per article via gemma-3-12b, self-reference filter (both Q
#     and A). Cost ~$1.90.
#
# Motivation:
#   v11's tight STEM dataset ([[reference-danish-wiki-closedqa-stem]]) gave
#   +0.9pp on SciQ open-Q. Analysis showed per-answer gradient (~15 tokens)
#   was too thin for real knowledge instillation at 400M scale — model
#   learned lookup patterns, not concept webs. Broad Q/A gives 400-800
#   tokens of gradient per fact, teaching the model to compose facts in
#   context. Also expands article corpus by +2k medicine/env articles that
#   were missed in v11's Fysik/Kemi/etc.-only tree.
#
# Kept: (v11's 11 sources) — plus wiki-broadqa-stem-v1 = 12 sources.
#
# DEPARTURE from v5–v11: 3 epochs instead of 2.
# Rationale: eval loss ~0.49 at end of 2 epochs suggests the 400M model
# is not yet saturated. Extra epoch increases each sample's exposure from
# 2× → 3×, giving broad-Q knowledge more chance to embed. Cost: +1h runtime.
# Total ~1.37M rows × 3 = ~4.1M example passes (v11 was ~2.7M).

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ckpt310k \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v12_mix12_ifv3_wpv2_stem_broad \
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
    jensjepsen/danish-word-problems-v2 \
    jensjepsen/danish-wiki-closedqa-stem-v1:sft \
    jensjepsen/danish-wiki-broadqa-stem-v1:sft \
  --epochs 3 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --learning-rate 5e-5 \
  --max-length 512 \
  --lr-scheduler cosine_with_min_lr \
  --warmup-steps 200 \
  --save-fraction-of-epoch 0.25 \
  --save-total-limit 3 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v12_sft_mix12_ifv3_wpv2_stem_broad \
  --wandb-tags sft da v12 mix12 no-morpheme if-v3 wp-v2 wiki-closedqa stem stem-broad epochs-3
