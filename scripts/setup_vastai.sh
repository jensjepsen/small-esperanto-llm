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

# Detect CUDA version and add matching PyTorch index
CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
echo "Detected CUDA: ${CUDA_VERSION:-none}"

if [ -n "$CUDA_VERSION" ]; then
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
    CU_TAG="cu${CUDA_MAJOR}${CUDA_MINOR}"

    TORCH_INDEX="https://download.pytorch.org/whl/${CU_TAG}"
    echo "Using PyTorch stable for ${CU_TAG}"

    cat >> pyproject.toml << EOF

[[tool.uv.index]]
url = "${TORCH_INDEX}"
name = "pytorch-${CU_TAG}"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-${CU_TAG}" }
EOF
fi

# Delete lockfile to resolve fresh for this platform
rm -f uv.lock

# Pin python and sync deps
uv python pin 3.11
uv sync

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
    print(f'bf16 supported: {torch.cuda.is_bf16_supported()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "=== Ready! ==="
echo "Pretrain:  uv run train --config $CONFIG --output-dir /workspace/runs/$CONFIG --min-article-length 500"
echo "SFT:       uv run python scripts/train_sft.py --checkpoint /workspace/runs/$CONFIG/checkpoint-XXXXX"
