#!/usr/bin/env bash
# Danish ICL v3 — continued SFT on multi-format schema-induction rows.
#
# Data: jensjepsen/danish-icl-json-v3:sft:train (33,937 rows, 4x v1). Two
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
# vs v2 (da_icl_v2_multifmt_bs8ga4_lr1e-5):
#   - train now includes the PAIRED format `tagged`; bracket_pair, brace_pair
#     and kv_eq are held out. v2 showed transfer inside the line family
#     (kv_eq 65.1% from a 0.0% base) and none across to paired delimiters
#     (tagged 0.5%), with a structural signature: 100% of tagged predictions
#     opened a tag, 62% carried every correct value, 2% balanced their tags.
#     This run asks whether training on ONE paired format makes the family
#     reachable -- and by extension whether `spans` is reachable by SFT.
#   - batch-size 8->16, grad-accum 4->2. eff_bs stays 32, so the optimizer
#     trajectory and step count are unchanged; v2 ran at 49% VRAM and 56-93%
#     GPU util, so this just halves the number of forward passes.
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
  --output-dir /root/runs/da_icl_v3 \
  --no-morpheme-preprocess \
  --sft-data jensjepsen/danish-icl-json-v3:sft:train \
  --epochs 3 --batch-size 16 --gradient-accumulation 2 \
  --optim adamw_bnb_8bit \
  --learning-rate 1e-5 --lr-scheduler constant_with_warmup --warmup-steps 50 \
  --max-length 3072 \
  --flatten-packing \
  --save-fraction-of-epoch 0.5 --eval-fraction-of-epoch 1.0 \
  --save-total-limit 8 \
  --downstream-evals gsm8k citgen --downstream-n 200 --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_icl_v3_paired_bs16ga2_lr1e-5 \
  --wandb-tags sft da icl multi-format paired-formats symbol-tuning icl-only constant-lr epochs-3 flatten-packing \
  --no-torch-compile
