#!/usr/bin/env bash
# Launch GRPO with vLLM rollout on a SINGLE GPU (H100 80GB default, but
# also works on a 5090 32GB with tighter defaults — see 5090 SMOKE below).
#
# Both processes on cuda:0:
#   vLLM   → ~40% of VRAM (weights + big KV cache)
#   Trainer→ ~50% of VRAM (weights + AdamW optim + grads + activations)
#   Head-room ~10% for peaks
#
# 5090 SMOKE (32GB):
#   VLLM_GPU_MEM=0.25 BATCH_SIZE=8 NUM_GEN=4 MAX_COMP_LEN=192 MAX_ROWS=200 \
#     TASK=gsm8k CKPT=... bash scripts/launch_grpo_vllm_h100.sh
#   → vLLM ~8GB, trainer ~15GB, ~9GB slack. Enough to smoke-test the plumbing.
#
# For a 400M model on H100 this leaves plenty of margin. Bump VLLM_GPU_MEM
# if you don't see OOM after a few steps.
#
# Usage:
#   TASK=gsm8k CKPT=jensjepsen/danish-lm-400m-sft-v31-avg-top3 \
#     bash scripts/launch_grpo_vllm_h100.sh
set -euo pipefail
cd "${ESPLLM_ROOT:-/root/espllm}"

: "${TASK:?TASK required (gsm8k|ifeval|json|mixed)}"
: "${CKPT:?CKPT required}"

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/runs/grpo/${TASK}_vllm_h100}"
RUN_NAME="${RUN_NAME:-grpo_${TASK}_vllm_h100}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NUM_GEN="${NUM_GEN:-8}"
LR="${LR:-1e-6}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_COMP_LEN="${MAX_COMP_LEN:-256}"
EVAL_STEPS="${EVAL_STEPS:-50}"
GREEDY_STEPS="${GREEDY_STEPS:-50}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
# 0.4 = ~32 GB on H100-80. Trainer footprint (400M model, bs=16, seq=768,
# AdamW-8bit optim) peaks around 25-30 GB, leaving ~15-20 GB headroom.
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.4}"
# 'colocate' (default, TRL 0.18+) runs vLLM inside trainer process — one CUDA
# context, no separate `trl vllm-serve` needed. 'server' keeps the old 2-GPU
# behavior for backwards compat (still requires launching trl vllm-serve).
VLLM_MODE="${VLLM_MODE:-colocate}"
MAX_ROWS="${MAX_ROWS:-0}"

export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT="${WANDB_PROJECT:-danish-lm-grpo}"
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')

if ! uv run python -c "import vllm" 2>/dev/null; then
  echo "== installing vllm via `uv sync --extra vllm` (respects pyproject pins) =="
  uv sync --extra train --extra vllm
fi

VLLM_LOG=/workspace/vllm_server_h100.log
TRAIN_LOG=/workspace/grpo_${TASK}_vllm_h100.log

VLLM_PID=""
if [ "$VLLM_MODE" = "server" ]; then
  echo "== launching vLLM server on cuda:0 (server mode) =="
  CUDA_VISIBLE_DEVICES=0 nohup uv run trl vllm-serve \
    --model "$CKPT" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --gpu_memory_utilization "$VLLM_GPU_MEM" \
    --dtype bfloat16 \
    --max_model_len $((MAX_PROMPT_LEN + MAX_COMP_LEN)) \
    --enable_prefix_caching \
    > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  echo "vLLM PID=$VLLM_PID  log=$VLLM_LOG"

  echo "== waiting for vLLM server =="
  for i in $(seq 1 120); do
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
else
  echo "== colocate mode: vLLM will run inside trainer process =="
fi

echo "== launching GRPO trainer on cuda:0 (vllm-mode=${VLLM_MODE}) =="
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
  --vllm-mode "$VLLM_MODE" \
  --vllm-host "$VLLM_HOST" \
  --vllm-port "$VLLM_PORT" \
  --vllm-gpu-memory-utilization "$VLLM_GPU_MEM" \
  --wandb-run-name "$RUN_NAME" \
  > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "trainer PID=$TRAIN_PID  log=$TRAIN_LOG"

echo "== trainer launched on cuda:0 (vllm=${VLLM_MODE}) =="
[ -n "$VLLM_PID" ] && echo "  tail -f $VLLM_LOG   # vLLM server"
echo "  tail -f $TRAIN_LOG  # GRPO trainer"
echo "  nvidia-smi          # confirm process(es) visible on cuda:0"
echo "  kill $TRAIN_PID${VLLM_PID:+ $VLLM_PID}  # stop"
