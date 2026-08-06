#!/bin/bash
# Setup script for cloud GPU instances (vast.ai, RunPod, etc.)
# Usage: bash scripts/setup_vastai.sh [small|medium|large]
set -e

CONFIG=${1:-medium}
echo "=== Esperanto LM setup ==="
echo "Config: $CONFIG"

# Use local SSD for caches (overlay / is small, use /tmp on NVMe)
export UV_CACHE_DIR=/tmp/uv-cache
export UV_PYTHON_INSTALL_DIR=/tmp/uv-python
export HF_HOME=/tmp/hf-cache
export HF_DATASETS_CACHE=/tmp/hf-cache/datasets

# Install system dependencies
apt-get update && apt-get install -y zstd

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Remove any existing torch index/source config
sed -i '/\[\[tool\.uv\.index\]\]/,/^$/d' pyproject.toml
sed -i '/\[tool\.uv\.sources\]/,/^$/d' pyproject.toml

# Detect CUDA driver version + GPU compute capability and pick torch backend
# accordingly. Blackwell consumer (RTX 5090, compute cap 12.0) needs cu128+
# wheels (introduced in torch 2.6); older arches (cap < 10) are fine on cu126.
# Without this distinction, a cu126 wheel on Blackwell crashes at first kernel
# launch with "CUDA error: no kernel image is available".
CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
GPU_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
GPU_CAP_MAJOR=$(echo "$GPU_CAP" | cut -d. -f1)
echo "Detected CUDA driver: ${CUDA_VERSION:-none}  GPU compute cap: ${GPU_CAP:-none}"

if [ "$CUDA_MAJOR" = "13" ]; then
    export UV_TORCH_BACKEND=cu128
    echo "Using UV_TORCH_BACKEND=cu128 (CUDA 13 driver)"
elif [ "$CUDA_MAJOR" = "12" ] && [ "${GPU_CAP_MAJOR:-0}" -ge 10 ] 2>/dev/null; then
    # Blackwell or newer needs cu128 wheels. Pin torch to 2.8.* — that's the
    # newest series with prebuilt flash-attn wheels for cu128+py311. Torch
    # 2.9/2.10 have no flash-attn wheel (would source-compile ~2h+); Blackwell
    # sm_120 support requires flash-attn ≥ 2.8.3 which is cu128-only.
    export UV_TORCH_BACKEND=cu128
    sed -i 's/torch>=2.3.0,<2.11/torch==2.8.*/' pyproject.toml
    echo "Using UV_TORCH_BACKEND=cu128 with torch==2.8.* (Blackwell+ GPU, flash-attn wheel compat)"
elif [ "$CUDA_MAJOR" = "12" ]; then
    # Pre-Blackwell on CUDA 12.x driver — cu126 is fine and keeps the older torch range.
    export UV_TORCH_BACKEND=cu126
    sed -i 's/torch>=2.3.0/torch>=2.3.0,<2.11/' pyproject.toml
    echo "Using UV_TORCH_BACKEND=cu126 with torch<2.11"
else
    export UV_TORCH_BACKEND=auto
    echo "Using UV_TORCH_BACKEND=auto"
fi

# Delete lockfile to resolve fresh for this platform
rm -f uv.lock

# Pin python and sync deps (with `train` extra for liger-kernel — required for
# the LM and SFT training scripts, which import it via try/except but get the
# 30-40% throughput boost when present)
uv python pin 3.11
uv sync --extra train

# Flash-attn: install via prebuilt wheel (--no-build-isolation reuses the
# venv's torch instead of resolving a fresh one in an isolation env, which
# would trigger a ~2h source compile). Wheel picks correct combo from the
# pinned torch version. Needed for varlen packing in SFT
# (DataCollatorWithFlattening) and for the fastest attention path in
# pretrain. Only installed when the pinned torch has wheels; skip on the
# `auto` backend where wheels may not exist.
if [ "$UV_TORCH_BACKEND" = "cu128" ] || [ "$UV_TORCH_BACKEND" = "cu126" ]; then
    echo "=== Installing flash-attn ==="
    uv pip install setuptools wheel packaging
    uv pip install flash-attn --no-build-isolation
    uv run python -c "import flash_attn; print(f'flash-attn OK ({flash_attn.__version__})')"
fi

# Verify Liger kernel actually applies. Install can succeed while
# `apply_liger_kernel_to_llama` raises at runtime (transformers version
# mismatch). train.py wraps the import in try/except, so a silent failure
# costs ~30-40% wall + ~40% VRAM without a peep. Catch it here at setup
# time. `set -e` aborts the script if this fails so the next launch can't
# quietly run without Liger.
echo "=== Verifying Liger kernel ==="
uv run python -c "
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama(rope=True, rms_norm=True, swiglu=True,
                             cross_entropy=True, fused_linear_cross_entropy=False)
print('Liger kernel OK')
"

# Verify bitsandbytes loads (needed for paged_adamw_8bit on bigger
# models). Optional at runtime — HF Trainer only imports it when
# optim=paged_adamw_8bit — but the install can fail silently in
# CUDA-mismatch situations, so we catch it here.
echo "=== Verifying bitsandbytes ==="
uv run python -c "
import bitsandbytes as bnb
print(f'bitsandbytes OK ({bnb.__version__})')
"

# Download tokenizer from HF Hub
echo "=== Downloading tokenizer from HF Hub ==="
uv run python scripts/download_from_hub.py --tokenizer

# All data (Wikipedia, HPLT, Gutenberg, factoids, sentences) is loaded
# automatically from HF Hub during training when local files are missing.

# Print GPU info
echo "=== GPU Info ==="
nvidia-smi
uv run python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'bf16 supported: {torch.cuda.is_bf16_supported()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "=== Ready! ==="
echo "Pretrain:  uv run train --config $CONFIG --output-dir /workspace/runs/$CONFIG --min-article-length 500"
echo "SFT:       uv run python scripts/train_sft.py --checkpoint /workspace/runs/$CONFIG/checkpoint-XXXXX"
