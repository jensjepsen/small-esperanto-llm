#!/usr/bin/env bash
# Danish SFT v13 (mix12-if-v3-wpv2-stem-broad, 2 epochs) — clean A/B vs v11.
#
# Same 12-source mix as v12 (v11 + wiki-broadqa-stem-v1), same optimizer,
# same schedule — but 2 epochs instead of v12's 3.
#
# Why: v12 (3 epochs) showed textbook overfitting after epoch 2.0 — eval
# loss spiked 0.562 → 0.613 at 2.25 and never recovered. HF Trainer's
# load_best_model_at_end picked ckpt-21408 (epoch 2.0) as /final, which was
# math-strong but knowledge-weak. The last checkpoint (ckpt-32121) recovered
# on downstream. See [[project-v12-best-ckpt-selection]].
#
# v13 = trim to 2 epochs so eval-loss curve stays monotonic, and /final =
# last ckpt = clean comparison to v11. Isolates the effect of adding
# broadqa-stem-v1 without the extra-epoch confound.
#
# Kept: (v11's 11 sources) — plus wiki-broadqa-stem-v1 = 12 sources.
# Total ~1.37M rows × 2 epochs = ~2.7M example passes (matches v11).

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ckpt310k \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v13_mix12_ifv3_wpv2_stem_broad_2e \
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
  --wandb-run-name da_v13_sft_mix12_ifv3_wpv2_stem_broad_2e \
  --wandb-tags sft da v13 mix12 no-morpheme if-v3 wp-v2 wiki-closedqa stem stem-broad epochs-2 ab-vs-v11
