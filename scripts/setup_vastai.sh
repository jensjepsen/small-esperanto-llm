#!/bin/bash
# Setup script for cloud GPU instances (vast.ai, RunPod, etc.)
# Usage: bash scripts/setup_vastai.sh [small|medium|large]
set -e

CONFIG=${1:-medium}
echo "=== Esperanto LM setup ==="
echo "Config: $CONFIG"

# Use local SSD for uv caches (network volumes are slow)
export UV_CACHE_DIR=/tmp/uv-cache
export UV_PYTHON_INSTALL_DIR=/tmp/uv-python

# Install system dependencies
apt-get update && apt-get install -y zstd

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Remove local cu121 torch pin — cloud instances use default PyPI torch
sed -i '/\[\[tool\.uv\.index\]\]/,/^$/d' pyproject.toml
sed -i '/\[tool\.uv\.sources\]/,/^$/d' pyproject.toml

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
echo "Pretrain:  uv run train --config $CONFIG --output-dir runs/$CONFIG --min-article-length 500"
echo "SFT:       uv run python scripts/train_sft.py --checkpoint runs/$CONFIG/checkpoint-XXXXX"
