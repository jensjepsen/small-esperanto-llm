#!/usr/bin/env bash
# Full v5 eval suite on the final checkpoint of da_v5_mix9if2.
# Runs each eval sequentially, tees to /root/eval_v5/<name>.log,
# and appends a one-line SUMMARY at /root/eval_v5/summary.log.

set -uo pipefail
cd /root/espllm

CKPT=/root/runs/sft/da_v5_mix9if2/final
LOG_DIR=/root/eval_v5
SUM=$LOG_DIR/summary.log
mkdir -p $LOG_DIR
: > $SUM
echo "=== v5 eval suite starting @ $(date -u +%FT%TZ) ===" | tee -a $SUM
echo "CKPT=$CKPT" | tee -a $SUM

# 1) IF-v2 eval (200 rows, batch=8) — quickest
echo "--- [1/6] IF-v2 eval (n=200) ---" | tee -a $SUM
uv run python -u scripts/probe_if_v5.py --checkpoint $CKPT --tokenizer $CKPT \
    --n 200 --batch-size 8 --temp 0.0 2>&1 | tee $LOG_DIR/if.log
grep -aE 'all-constraints-pass' $LOG_DIR/if.log | tee -a $SUM

# 2) DA MC logprob (ARC-da + HellaSwag-da + Citizen)
echo "--- [2/6] DA MC (logprob) ---" | tee -a $SUM
uv run python -u scripts/eval_da_mc.py --ckpt $CKPT --tokenizer $CKPT \
    --step 20644 --csv $LOG_DIR/da_mc.csv --hs-samples 500 2>&1 | tee $LOG_DIR/da_mc.log
tail -30 $LOG_DIR/da_mc.log | grep -aE 'acc|ARC|Hella|Cit|=' | head -20 | tee -a $SUM

# 3) GSM8K (200 rows)
echo "--- [3/6] GSM8K (n=200 greedy) ---" | tee -a $SUM
uv run python -u scripts/eval_gsm8k_da_gen.py $CKPT \
    --out $LOG_DIR/gsm8k.jsonl --n 200 --dtype bf16 2>&1 | tee $LOG_DIR/gsm8k.log
grep -aE 'gsm8k\[da\]' $LOG_DIR/gsm8k.log | tee -a $SUM

# 4) SciQ MC letter
echo "--- [4/6] SciQ MC-letter (n=500) ---" | tee -a $SUM
uv run python -u scripts/eval_sciq_da_gen.py --ckpt $CKPT \
    --dump $LOG_DIR/sciq_mc.jsonl 2>&1 | tee $LOG_DIR/sciq_mc.log
tail -20 $LOG_DIR/sciq_mc.log | grep -aE 'acc|=|SciQ' | head -6 | tee -a $SUM

# 5) SciQ open-ended
echo "--- [5/6] SciQ open-ended ---" | tee -a $SUM
uv run python -u scripts/eval_sciq_da_openq.py --ckpt $CKPT \
    --dump $LOG_DIR/sciq_open.jsonl 2>&1 | tee $LOG_DIR/sciq_open.log
tail -20 $LOG_DIR/sciq_open.log | grep -aE 'acc|=|SciQ' | head -6 | tee -a $SUM

# 6) Citizen tests generative
echo "--- [6/6] Citizen tests gen ---" | tee -a $SUM
uv run python -u scripts/eval_cit_da_gen.py $CKPT --n 200 2>&1 \
    | tee $LOG_DIR/cit_gen.log
tail -20 $LOG_DIR/cit_gen.log | grep -aE 'acc|=|Cit|citizen' | head -6 | tee -a $SUM

echo "=== v5 eval suite done @ $(date -u +%FT%TZ) ===" | tee -a $SUM
