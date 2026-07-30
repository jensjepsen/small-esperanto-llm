#!/usr/bin/env bash
# Danish SFT v19 — 4th epoch of v16, from v16-final (ckpt-32112),
# same constant LR 3e-5. NOT from v18 (that was a separate anneal branch).
#
# Story so far:
#   v16 = 3ep constant LR 3e-5, base=danish-lm-400m-base-ckpt310k
#         (peak agg 0.183 at ckpt-29436, wandered off peak; final 0.177)
#   v18 = separate branch: 0.2ep linear anneal from v16-29436 → agg 0.195
#   v19 = MAIN v16 line continued — another epoch of plain constant 3e-5
#         from v16's actual final ckpt-32112. Tests whether v16's tail-of-
#         epoch-3 drift reverses or continues under more of the same LR.
#
# Continues v16's wandb run (id=1l1ak676) with step offset 32121 so the
# chart is one contiguous constant-LR line — as if v16 had trained 4
# epochs from the start. (v18 is a sibling branch, not on this chart.)
#
# Ckpt fetched from HF subfolder (pod is fresh). Runtime ~3.5h on 5090.

set -euo pipefail
cd /root/espllm

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

CKPT=${CKPT:-/root/hf_v16_32112/step-32112-agg-0.177}

if [[ ! -f "$CKPT/model.safetensors" ]]; then
  echo "==> Fetching v16-32112 ckpt from HF..."
  uv run --no-project --with huggingface_hub \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('jensjepsen/danish-lm-400m-sft-v16', allow_patterns=['step-32112-agg-0.177/*'], local_dir='/root/hf_v16_32112')"
fi

uv run python -u scripts/train_sft_packed.py \
  --checkpoint "$CKPT" \
  --tokenizer "$CKPT" \
  --output-dir /root/runs/sft/da_v19_epoch4_constlr \
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
  --epochs 1 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --learning-rate 3e-5 \
  --max-length 512 \
  --lr-scheduler constant_with_warmup \
  --warmup-steps 0 \
  --save-fraction-of-epoch 0.25 \
  --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen \
  --downstream-batch-size 32 \
  --top-k-downstream 3 \
  --wandb-project danish-lm-sft \
  --wandb-run-id 1l1ak676 \
  --wandb-step-offset 32121 \
  --wandb-tags sft da v19 epoch4-constlr from-v16-final continues-v16 lr-3e-5 full-set top3
