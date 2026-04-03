#!/bin/bash
# Setup script for vast.ai instances
# Usage: bash scripts/setup_vastai.sh [small|medium]
set -e

CONFIG=${1:-small}
echo "=== Esperanto LM setup for vast.ai ==="
echo "Config: $CONFIG"

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Remove local cu121 torch pin — cloud instances use default PyPI torch
sed -i '/\[\[tool\.uv\.index\]\]/,/^$/d' pyproject.toml
sed -i '/\[tool\.uv\.sources\]/,/^$/d' pyproject.toml

# Pin python and sync deps
uv python pin 3.11
uv sync

# Download data
echo "=== Downloading data ==="
uv run download-data
uv run download-hplt --min-score 7
uv run download-gutenberg

# Train tokenizer
echo "=== Training tokenizer ==="
uv run train-tokenizer

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
echo "=== Ready! Run with: ==="
echo "uv run train --config $CONFIG --output-dir runs/$CONFIG --use-hplt --use-gutenberg --min-article-length 500"
