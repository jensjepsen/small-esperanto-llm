#!/usr/bin/env bash
# Danish ICL v2 — continued SFT on multi-format schema-induction rows.
#
# Data: jensjepsen/danish-icl-schema-format-v2:sft:train (33,937 rows, 4x v1). Two
# induction axes now vary per row: the schema (113 field-sets in train) and
# the output format (6 renderers in train; kv_eq and tagged are held out for
# the eval_format / eval_both splits).
#
# ICL-ONLY again, no ballast — the lock-in risk is accepted. v1 showed the
# cost precisely: training on JSON alone produced a model that answers a
# "type: enhed" request with malformed JSON. The bet here is that varying the
# format across rows is itself the fix, since a single-format model can no
# longer fit the training distribution.
#
# vs v1 (da_icl_v1_only_bs8ga4_lr1e-5), which reached 48.3% on unseen schemas
# from a 0.3% base and was still climbing at 3 epochs:
#   - 4x the rows -> 1061 steps/epoch at eff_bs 32 (v1 had 252)
#   - eval every epoch rather than 3x per epoch; the run is long enough now
#     that per-epoch is the useful granularity
# Everything else held: lr 1e-5 constant / 50 warmup, eff_bs 32, seq 3072,
# adamw_bnb_8bit, flatten-packing, v31 CHECKPOINT tokenizer (16007 tokens,
# chat tokens at 16000-16002; jensjepsen/danish-tokenizer is 16000 and maps
# them to unk).
#
# gsm8k/citgen at n=200 stay as collapse detectors, but note they did NOT
# catch v1's format lock-in — citgen asks plain questions, so it cannot see a
# model that has become rigid about structured-output requests. Judge that
# with scripts/probe_da_lines.py / probe_da_spans.py after the run.
#
# ~3,180 steps; v1 ran 756 steps in 681s on one RTX 5090, so expect ~50-60 min
# plus eval overhead.
set -euo pipefail
cd /root/espllm

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/tmp/hf-cache
export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

CKPT=jensjepsen/danish-lm-400m-sft-v31-avg-top3

uv run python -u scripts/train_sft_packed.py \
  --checkpoint "$CKPT" \
  --tokenizer "$CKPT" \
  --output-dir /root/runs/da_icl_v2 \
  --no-morpheme-preprocess \
  --sft-data jensjepsen/danish-icl-schema-format-v2:sft:train \
  --epochs 3 --batch-size 8 --gradient-accumulation 4 \
  --optim adamw_bnb_8bit \
  --learning-rate 1e-5 --lr-scheduler constant_with_warmup --warmup-steps 50 \
  --max-length 3072 \
  --flatten-packing \
  --save-fraction-of-epoch 0.5 --eval-fraction-of-epoch 1.0 \
  --save-total-limit 8 \
  --downstream-evals gsm8k citgen --downstream-n 0 --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_icl_v2_multifmt_bs8ga4_lr1e-5 \
  --wandb-tags sft da icl multi-format symbol-tuning icl-only constant-lr epochs-3 flatten-packing \
  --no-torch-compile
