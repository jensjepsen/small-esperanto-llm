#!/usr/bin/env bash
# Danish SFT v18 — anneal from v16 constant-LR PEAK (step-29436, epoch 2.75).
#
# v17 tried anneal from v16's FINAL ckpt (32112, agg 0.177) and reached
# agg 0.184 at step-800 — matched v16's own peak but didn't beat v12
# (agg 0.192). Hypothesis: starting from the constant-LR peak (29436,
# agg 0.183) gives the anneal a stronger base than the drifted-off-peak
# final ckpt, and closes more of the v12 gap.
#
# Caveat: v16 optimizer state was NOT preserved on HF (see feedback memo
# hf-push-include-optim). So this anneal runs with FRESH Adam moments —
# same shape as v17. First-50-step momentum warmup will happen.
#
# Recipe (mirrors v17):
#   - Fetch step-29436-agg-0.183 subfolder from jensjepsen/danish-lm-400m-sft-v16
#   - 0.2 epochs (~2140 steps ≈ 12 min compute + 5 evals @ ~5 min each)
#   - Peak LR 1e-5, linear decay to 0, zero warmup
#   - Full-set downstream, top-3 preservation, save every 400 steps
#
# Runtime ~40 min total (25 min compute + 25 min eval; some overlap).

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

CKPT_DIR=${CKPT_DIR:-/root/hf_v16_29436/step-29436-agg-0.183}

if [[ ! -f "$CKPT_DIR/model.safetensors" ]]; then
  echo "==> Fetching ckpt from HF..."
  uv run --no-project --with huggingface_hub \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('jensjepsen/danish-lm-400m-sft-v16', allow_patterns=['step-29436-agg-0.183/*'], local_dir='/root/hf_v16_29436')"
fi

uv run python -u scripts/train_sft_packed.py \
  --checkpoint "$CKPT_DIR" \
  --tokenizer "$CKPT_DIR" \
  --output-dir /root/runs/sft/da_v18_anneal_from_v16_peak \
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
  --wandb-run-name da_v18_anneal_from_v16_peak \
  --wandb-tags sft da v18 anneal from-v16-peak-29436 short lr-1e-5 linear-decay full-set top3 fresh-optim
