#!/usr/bin/env bash
# Danish SFT v20 — LONGER anneal from v16 epoch-3 ckpt (32112).
#
# v17 = 0.2ep linear 1e-5→0 from v16-32112, peaked at agg 0.184.
# v19 = 1ep constant 3e-5 from v16-32112, peaked at 0.182 at cumul-3.25
#       then monotonic decline — constant LR doesn't extract more.
#
# v20 tests: does 2.5× v17's anneal budget (0.5 epochs vs 0.2) squeeze
# more out of the same v16-32112 base? Same peak LR (1e-5), same linear-
# to-0 shape, just longer. Fine-grained saves every 1070 steps (10 % of
# anneal) — 5 evals, same COUNT as v17 for a matched-cadence read.
#
# Runtime ~1h 30m (30 min train + 28 min eval overhead).

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

CKPT=${CKPT:-/root/hf_v16_32112/step-32112-agg-0.177}

if [[ ! -f "$CKPT/model.safetensors" ]]; then
  echo "==> Fetching v16-32112 from HF..."
  uv run --no-project --with huggingface_hub \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('jensjepsen/danish-lm-400m-sft-v16', allow_patterns=['step-32112-agg-0.177/*'], local_dir='/root/hf_v16_32112')"
fi

uv run python -u scripts/train_sft_packed.py \
  --checkpoint "$CKPT" \
  --tokenizer "$CKPT" \
  --output-dir /root/runs/sft/da_v20_anneal_from_v16_e3_long \
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
  --epochs 0.5 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --learning-rate 1e-5 \
  --max-length 512 \
  --lr-scheduler linear \
  --warmup-steps 0 \
  --save-steps 1070 \
  --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen \
  --downstream-batch-size 32 \
  --top-k-downstream 3 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v20_anneal_from_v16_e3_long \
  --wandb-tags sft da v20 anneal from-v16-e3 long-0.5ep lr-1e-5 linear-decay full-set top3
