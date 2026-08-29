#!/bin/bash
# Resume GRPO on v10-mix16 SFT model against GSM8K-EO from
# jensjepsen/eo-grpo-v10-gsm8k-ckpt400 (step 400, T=0.7, n=12,
# math-only reward, skip-zero-adv).
#
# Progression so far (greedy pass@1 on 1279-row GSM8K-EO test):
#   SFT ckpt-45344     : 20.72% (baseline)
#   GRPO ckpt-100      : untested
#   GRPO ckpt-200      : 21.97%  (+1.25 pp)
#   GRPO ckpt-300      : 22.20%  (+1.48 pp)
#   GRPO ckpt-400      : 23.30%  (+2.58 pp)   ← resume point
#
# pass@3 at ckpt-300 dropped to 33.15% (SFT was 34.64%) — RL is
# sharpening greedy at the cost of diversity, expected from math-only.
#
# Usage on fresh pod (after setup_vastai.sh + wandb + hf token):
#   bash scripts/launch_grpo_v10_gsm8k_resume.sh /root/runs/v10_grpo_gsm8k
#
# The output-dir needs to contain a `checkpoint-N` subdir. If you pass the
# HF repo `jensjepsen/eo-grpo-v10-gsm8k-ckpt400` it will be pulled first.
set -euo pipefail

OUTPUT=${1:-/root/runs/v10_grpo_gsm8k}
CKPT_REPO=jensjepsen/eo-grpo-v10-gsm8k-ckpt400

# If no local checkpoint present, download the 400-step one so --resume
# has something to pick up.
if ! ls "$OUTPUT"/checkpoint-* &>/dev/null; then
    echo "[bootstrap] fetching $CKPT_REPO → $OUTPUT/checkpoint-400"
    mkdir -p "$OUTPUT/checkpoint-400"
    HF_HOME=${HF_HOME:-/root/hf-cache} uv run --no-project python -c "
from huggingface_hub import snapshot_download
snapshot_download('$CKPT_REPO', local_dir='$OUTPUT/checkpoint-400')
"
fi

HF_HOME=${HF_HOME:-/root/hf-cache} \
uv run python -u scripts/train_grpo.py \
  --checkpoint jensjepsen/eo-sft-v10-mix16-ckpt45344 \
  --dataset jensjepsen/esperanto-gsm8k \
  --output-dir "$OUTPUT" \
  --prompt-style chat \
  --math-only \
  --skip-zero-adv \
  --num-generations 12 \
  --batch-size 12 \
  --grad-accum 4 \
  --max-completion-len 400 \
  --temperature 0.7 \
  --epochs 1 \
  --save-steps 100 \
  --logging-steps 5 \
  --resume \
  --wandb-run-name v10_grpo_gsm8k_resume \
  --wandb-tags grpo v10 gsm8k math-only skip-zero-adv resume
