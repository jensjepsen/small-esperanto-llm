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

# TRL 1.10 supports vllm 0.17-0.26. vllm>=0.17 requires torch>=2.10. No
# flash-attn prebuilt wheel exists for torch 2.10 (would source-compile ~2h+),
# so we fall back to SDPA. This matches the pre-kill Aug 22 stack exactly and
# is the only way to stay inside TRL 1.10's officially-supported vllm range.
# Set SKIP_FLASH_ATTN=1 to skip the flash-attn install below.
export SKIP_FLASH_ATTN=${SKIP_FLASH_ATTN:-1}

if [ "$CUDA_MAJOR" = "13" ]; then
    # CUDA 13 driver → cu128 wheels. Don't pin torch; let vllm 0.17+ pull torch 2.10.
    export UV_TORCH_BACKEND=cu128
    echo "Using UV_TORCH_BACKEND=cu128 (CUDA 13 driver, torch resolves to 2.10 via vllm 0.17+, SDPA attention)"
elif [ "$CUDA_MAJOR" = "12" ] && [ "${GPU_CAP_MAJOR:-0}" -ge 10 ] 2>/dev/null; then
    # Blackwell (5090) or newer — cu128 wheels. Don't pin torch; vllm 0.17+ pulls torch 2.10.
    # This matches the pre-kill Aug 22 stack (torch 2.10 + vllm 0.19.1 + SDPA).
    export UV_TORCH_BACKEND=cu128
    echo "Using UV_TORCH_BACKEND=cu128 (Blackwell+ GPU, torch resolves to 2.10 via vllm 0.17+, SDPA attention)"
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

# Flash-attn: install prebuilt wheel from GitHub releases. `pip install
# flash-attn` from PyPI is SOURCE-ONLY (2-3h compile). The GitHub release
# assets have per-(torch, cuda, cxx11abi, cpython) binary wheels — download
# the matching one by URL. Works identically on 5090 (Blackwell) and H100
# (Hopper) as long as UV_TORCH_BACKEND=cu128 (both need cu128 wheels).
#
# Wheel name pattern:
#   flash_attn-{VER}+cu12torch{TORCH_MM}cxx11abi{ABI}-cp{PY}-cp{PY}-linux_x86_64.whl
# where TORCH_MM is e.g. "2.8" (major.minor) and ABI is TRUE/FALSE matching
# `torch.compiled_with_cxx11_abi()`.
if [ "$SKIP_FLASH_ATTN" = "1" ]; then
    echo "=== Skipping flash-attn install (SKIP_FLASH_ATTN=1; torch 2.10+ has no wheel, would source-compile ~2h+; SDPA is fine for GRPO on 400M) ==="
elif [ "$UV_TORCH_BACKEND" = "cu128" ] || [ "$UV_TORCH_BACKEND" = "cu126" ]; then
    echo "=== Installing flash-attn ==="
    uv pip install setuptools wheel packaging
    # Query the pinned torch version + ABI + python version to build the
    # correct wheel URL. This avoids any source compile path.
    TORCH_MM=$(uv run python -c "import torch, re; print(re.match(r'(\d+\.\d+)', torch.__version__).group(1))")
    TORCH_ABI=$(uv run python -c "import torch; print('TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')")
    PY_CP=$(uv run python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    FA_VER=${FA_VER:-2.8.3}
    WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/flash_attn-${FA_VER}+cu12torch${TORCH_MM}cxx11abi${TORCH_ABI}-${PY_CP}-${PY_CP}-linux_x86_64.whl"
    echo "  torch=${TORCH_MM}  abi=${TORCH_ABI}  py=${PY_CP}  fa=${FA_VER}"
    echo "  wheel: ${WHEEL}"
    uv pip install "${WHEEL}"
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
