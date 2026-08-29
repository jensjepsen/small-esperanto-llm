#!/usr/bin/env bash
# GRPO 5-way: the usual 3-way interleave (IF-combined + gsm8k + json) plus NER
# and ICL schema/format, ~0.2 each.
#
# From v33-avg-top3, which is the first SFT to carry NER and schema/format
# ability at all (NER exact 0.2 -> 58.7 vs v31, span-wrap faithful 0 -> 73).
# GRPO on those two tasks only makes sense on a policy that can already sample
# correct answers sometimes -- on v31 both reward channels would be flat zero
# and contribute nothing but noise.
#
# Precision: no GRPO_FP16_EVERYWHERE. Default is bf16 autocast over fp32
# master weights, which is what the low-beta work depends on -- bf16 masters
# round away the small updates (KL doubled in the A/B). L40S is Ada, so bf16 is
# native and there is no reason to reach for the FP16-everywhere recipe here.
#
# NER comes from danish-ner-sft-v1, not dane_plus: three prompt modes and
# fourteen formats scored by reward_structured, which is where the headroom is
# (instruction mode 48.4 vs demonstrations 60.4; span-wrap 7.5 on an unseen
# delimiter). Both NER and ICL rows were in v33's SFT mix, so watch
# rewards/*/fzs --- if a channel saturates it stops producing advantage and
# should be swapped for a harder split.
#
# max-prompt-length 2048: ICL prompts carry their demonstrations and run to
# 2154 tokens (mean 395). The 768 the 3-way recipe used would silently truncate
# the demonstrations off a large share of ICL rows, making them unanswerable
# and their reward channel pure noise.
set -uo pipefail
cd /root/espllm
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/tmp/hf-cache
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export WANDB_PROJECT=danish-lm-grpo
unset WANDB_RUN_ID WANDB_RESUME

OUTPUT_DIR=${OUTPUT_DIR:-/root/runs/grpo/mixed5_v1}
BASE_REPO=${BASE_REPO:-jensjepsen/danish-lm-400m-sft-v33-avg-top3}
CKPT_LOCAL="$OUTPUT_DIR/base_ckpt"
EXTRA=${EXTRA:-}
PY="${PY:-uv run --no-sync python}"

if [ ! -f "$CKPT_LOCAL/model.safetensors" ]; then
  echo "== downloading base ckpt ($BASE_REPO) =="
  mkdir -p "$CKPT_LOCAL"
  uv run --no-sync huggingface-cli download "$BASE_REPO" --local-dir "$CKPT_LOCAL"
fi

$PY -u scripts/train_grpo_verifier.py \
  --task mixed \
  --combined-source jensjepsen/danish-if-grpo-combined-v1 \
  --json-source jensjepsen/danish-json-grpo-v1 \
  --icl-source jensjepsen/danish-icl-schema-format-v3 \
  --ner-source jensjepsen/danish-ner-sft-v1 \
  --gsm-frac 0.2 --json-frac 0.2 --ner-frac 0.2 --icl-frac 0.2 \
  --checkpoint "$CKPT_LOCAL" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 3 --batch-size 8 --grad-accum 4 --num-generations 32 \
  --max-prompt-length 2048 --max-completion-length 384 \
  --learning-rate 1e-6 --beta 0.004 --warmup-steps 10 --logging-steps 5 \
  --use-vllm-server --vllm-mode colocate --vllm-gpu-memory-utilization 0.30 \
  --save-steps 125 --eval-steps 0 \
  --greedy-eval-steps 125 --greedy-eval-max-rows 200 \
  --skip-zero-adv --best-k 3 \
  --wandb-run-name grpo_mixed5_v1_ner_icl_from_v33 \
  $EXTRA
