#!/usr/bin/env bash
# Launch v12 400M pretraining on 2× H100 80GB via DDP (torchrun).
#
# Expects:
#  - cwd inside the small-esperanto-llm repo
#  - /root/.wandb_key and /root/hf_token populated
#  - /workspace/{runs,hf-cache} writable (overlay mounted)
#  - .venv with all deps + bitsandbytes (setup_vastai.sh installs the latter)
#
# Effective batch (verified): per_device(64) × accum(2) × GPUs(2) = 256
# — matches h100_400m.yaml single-GPU effective batch exactly.
#
# Run name and tags are set so wandb groups it with the rest of the
# v12 series. Output dir is fixed; bump if you do multiple runs.

set -euo pipefail

cd "$(dirname "$0")/../.."

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/runs/v12_h100_400m}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"

# Verify 2 GPUs visible
N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$N_GPU" -lt 2 ]; then
    echo "ERROR: need 2 GPUs, found $N_GPU"
    exit 1
fi
echo "Found $N_GPU GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$(dirname "$OUTPUT_DIR")" /workspace/hf-cache

# Standard env: wandb + HF + caches all on overlay/workspace
export WANDB_API_KEY=$(cat /root/.wandb_key)
export HF_TOKEN=$(cat /root/hf_token)
export HF_HOME=/workspace/hf-cache
export WANDB_RUN_GROUP=v12_400m_2gpu
export WANDB_TAGS="v12,h100,2gpu,fresh,400m,liger_flce,paged_adamw_8bit,tied_emb,torch_compile"

# torchrun handles process spawning + sets RANK/WORLD_SIZE/LOCAL_RANK
# env vars that HF Trainer picks up automatically.
nohup .venv/bin/torchrun \
    --standalone \
    --nproc_per_node=2 \
    -m esperanto_lm.train \
    --config h100_400m_2gpu \
    --output-dir "$OUTPUT_DIR" \
    --tokenizer tokenizer_morpheme \
  > "$LOG_FILE" 2>&1 &

PID=$!
disown
echo "launched pid=$PID  log=$LOG_FILE  output=$OUTPUT_DIR"
echo "Monitor via:  tail -f $LOG_FILE"
