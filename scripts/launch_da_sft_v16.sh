#!/usr/bin/env bash
# Danish SFT v16 (mix12, 3 epochs, constant LR + warmup) — schedule probe.
#
# Companion to v14 (5-epoch cosine) & v15 (anneal). Isolates the schedule
# variable: constant LR (no decay) for the whole run so we can see whether
# v14's late-stage GSM drift was caused by too-hot cosine tail or by
# overtraining itself. Fixed LR + warmup = simpler baseline.
#
# LR chosen 3e-5 (~60% of v14's 5e-5 cosine peak) so total "gradient budget"
# per pass is comparable to v14 at eq epoch count (cosine averages ~60% peak).
# Warmup 200 steps matches v14.
#
# Full-set downstream evals every 0.25 epoch (~8-12 min overhead each);
# top-3 by mean-downstream preserved into best/ so save_total_limit=2
# rolls the recent pool while the actually-good ckpts persist.
#
# Runtime ~4h train + ~1.5h downstream eval overhead on 5090.

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ckpt310k \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/sft/da_v16_mix12_3e_constlr \
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
  --learning-rate 3e-5 \
  --max-length 512 \
  --lr-scheduler constant_with_warmup \
  --warmup-steps 200 \
  --save-fraction-of-epoch 0.25 \
  --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen \
  --downstream-batch-size 32 \
  --top-k-downstream 3 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v16_sft_mix12_3e_constlr \
  --wandb-tags sft da v16 mix12 no-morpheme if-v3 wp-v2 wiki-closedqa stem stem-broad epochs-3 downstream-eval full-set top3 constant-lr warmup-200 lr-3e-5 schedule-probe
