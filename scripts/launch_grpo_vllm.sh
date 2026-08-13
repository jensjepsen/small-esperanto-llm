#!/usr/bin/env bash
# Launch GRPO with vLLM rollout server on a 2-GPU box (e.g. 2× 5090).
#
# GPU 0 → training (fwd+bwd, optimizer)
# GPU 1 → vLLM server (all rollouts, weight-sync from trainer)
#
# TRL's `trl vllm-serve` starts a vLLM server that implements the weight-sync
# protocol the GRPOTrainer expects. Regular `vllm serve` won't work.
#
# Usage:
#   TASK=gsm8k CKPT=jensjepsen/danish-lm-400m-sft-v31-avg-top3 \
#     bash scripts/launch_grpo_vllm.sh
#
# Env knobs:
#   TASK           gsm8k | ifeval           (required)
#   CKPT           HF repo or local path    (required)
#   OUTPUT_DIR     run output directory     (default /workspace/runs/grpo/<task>)
#   RUN_NAME       wandb run name           (default grpo_<task>_vllm)
#   EPOCHS         (default 1)
#   BATCH_SIZE     per-device prompt batch  (default 16)
#   GRAD_ACCUM     (default 2)
#   NUM_GEN        rollouts per prompt      (default 4)
#   LR             (default 1e-6)
#   MAX_PROMPT_LEN (default 512)
#   MAX_COMP_LEN   (default 256)
#   EVAL_STEPS     sampled eval every N     (default 50)
#   GREEDY_STEPS   greedy eval every N      (default 50)
#   VLLM_HOST      (default 127.0.0.1)
#   VLLM_PORT      (default 8000)
#   VLLM_GPU_MEM   fraction for vLLM        (default 0.9)
#   MAX_ROWS       cap train rows           (default 0 = all)
set -euo pipefail
cd "${ESPLLM_ROOT:-/root/espllm}"

: "${TASK:?TASK required (gsm8k|ifeval)}"
: "${CKPT:?CKPT required}"

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/runs/grpo/${TASK}_vllm}"
RUN_NAME="${RUN_NAME:-grpo_${TASK}_vllm}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NUM_GEN="${NUM_GEN:-4}"
LR="${LR:-1e-6}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_COMP_LEN="${MAX_COMP_LEN:-256}"
EVAL_STEPS="${EVAL_STEPS:-50}"
GREEDY_STEPS="${GREEDY_STEPS:-50}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.9}"
MAX_ROWS="${MAX_ROWS:-0}"

export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT="${WANDB_PROJECT:-danish-lm-grpo}"
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')

# One-time: install vllm if missing
if ! uv run python -c "import vllm" 2>/dev/null; then
  echo "== installing vllm =="
  uv pip install vllm
fi

VLLM_LOG=/workspace/vllm_server.log
TRAIN_LOG=/workspace/grpo_${TASK}_vllm.log

echo "== launching vLLM server on GPU 1 =="
CUDA_VISIBLE_DEVICES=1 nohup uv run trl vllm-serve \
  --model "$CKPT" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --gpu_memory_utilization "$VLLM_GPU_MEM" \
  --dtype bfloat16 \
  --max_model_len $((MAX_PROMPT_LEN + MAX_COMP_LEN)) \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "vLLM PID=$VLLM_PID  log=$VLLM_LOG"

# Wait for vLLM server to be ready (probes /health)
echo "== waiting for vLLM server =="
for i in $(seq 1 90); do
  if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "vLLM ready after ${i}s"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM died before ready. Check $VLLM_LOG"; exit 1
  fi
  sleep 2
done
if ! curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
  echo "vLLM never became ready. Check $VLLM_LOG"; exit 1
fi

echo "== launching GRPO trainer on GPU 0 =="
GREEDY_ARGS=""
if [ "$GREEDY_STEPS" -gt 0 ]; then
  GREEDY_ARGS="--greedy-eval-steps $GREEDY_STEPS --greedy-eval-max-rows 200"
fi
MAX_ROWS_ARGS=""
if [ "$MAX_ROWS" -gt 0 ]; then
  MAX_ROWS_ARGS="--max-rows $MAX_ROWS"
fi

CUDA_VISIBLE_DEVICES=0 nohup uv run python -u scripts/train_grpo_verifier.py \
  --task "$TASK" \
  --checkpoint "$CKPT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --num-generations "$NUM_GEN" \
  --max-prompt-length "$MAX_PROMPT_LEN" \
  --max-completion-length "$MAX_COMP_LEN" \
  --learning-rate "$LR" \
  --warmup-steps 10 \
  --logging-steps 5 \
  --save-steps 100 \
  --eval-steps "$EVAL_STEPS" \
  --eval-max-rows 200 \
  $GREEDY_ARGS \
  $MAX_ROWS_ARGS \
  --skip-zero-adv \
  --use-vllm-server \
  --vllm-host "$VLLM_HOST" \
  --vllm-port "$VLLM_PORT" \
  --wandb-run-name "$RUN_NAME" \
  > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "trainer PID=$TRAIN_PID  log=$TRAIN_LOG"

echo "== both processes launched =="
echo "  tail -f $VLLM_LOG   # vLLM server"
echo "  tail -f $TRAIN_LOG  # GRPO trainer"
echo "  kill $TRAIN_PID $VLLM_PID  # stop both"
