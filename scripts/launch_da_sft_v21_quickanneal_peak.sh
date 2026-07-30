#!/usr/bin/env bash
# Danish SFT v21 — quick (0.2ep) linear anneal from v16 constant-LR PEAK
# ckpt-29436 (agg 0.183, epoch 2.75). Reproduces v18's recipe on the new
# pod (v18's ckpts were lost when the old pod died before we could push
# optim/scheduler state to HF).
#
# Reference:
#   v18 = same recipe, same base ckpt → peaked agg 0.195 at step-1600
#         (beat v12's 0.192 baseline)
#   v20 = 0.5ep anneal from v16-FINAL (32112) → agg 0.185 (marginal
#         improvement over v17's 0.184 from same base)
#   v21 = re-run v18 for a reproducible artifact + top-3 preserved to
#         disk that we can push to HF properly (with optim this time)
#
# Runtime ~40 min (12 min compute + 25 min eval overhead).

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

CKPT=${CKPT:-/root/hf_v16_29436/step-29436-agg-0.183}

if [[ ! -f "$CKPT/model.safetensors" ]]; then
  echo "==> Fetching v16-29436 from HF..."
  uv run --no-project --with huggingface_hub \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('jensjepsen/danish-lm-400m-sft-v16', allow_patterns=['step-29436-agg-0.183/*'], local_dir='/root/hf_v16_29436')"
fi

uv run python -u scripts/train_sft_packed.py \
  --checkpoint "$CKPT" \
  --tokenizer "$CKPT" \
  --output-dir /root/runs/sft/da_v21_quickanneal_from_v16_peak \
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
  --epochs 0.2 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --learning-rate 1e-5 \
  --max-length 512 \
  --lr-scheduler linear \
  --warmup-steps 0 \
  --save-steps 400 \
  --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen \
  --downstream-batch-size 32 \
  --top-k-downstream 3 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v21_quickanneal_from_v16_peak \
  --wandb-tags sft da v21 anneal from-v16-peak-29436 quick-0.2ep lr-1e-5 linear-decay full-set top3 reproduces-v18
