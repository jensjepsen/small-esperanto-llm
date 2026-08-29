#!/bin/bash
# Minimal pod setup for MT translation only (no LM training deps).
# Skips Liger/bitsandbytes checks and LM tokenizer download.
set -e

export UV_CACHE_DIR=/tmp/uv-cache
export UV_PYTHON_INSTALL_DIR=/tmp/uv-python
export HF_HOME=/workspace/hf-cache
export HF_DATASETS_CACHE=/workspace/hf-cache/datasets
mkdir -p "$HF_HOME"

# Persist for interactive shells later
cat > /root/.bashrc.mt <<'EOF'
export UV_CACHE_DIR=/tmp/uv-cache
export HF_HOME=/workspace/hf-cache
export HF_DATASETS_CACHE=/workspace/hf-cache/datasets
export PATH="$HOME/.local/bin:$PATH"
EOF
grep -q bashrc.mt /root/.bashrc || echo 'source /root/.bashrc.mt' >> /root/.bashrc

apt-get update -qq && apt-get install -y -qq tmux

curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -q
export PATH="$HOME/.local/bin:$PATH"

# Strip torch index/source blocks that pin cu121 (unusable on Blackwell)
sed -i '/\[\[tool\.uv\.index\]\]/,/^$/d' pyproject.toml
sed -i '/\[tool\.uv\.sources\]/,/^$/d' pyproject.toml

# Detect Blackwell → cu128 with torch>=2.6
GPU_CAP_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${GPU_CAP_MAJOR:-0}" -ge 10 ] 2>/dev/null; then
    export UV_TORCH_BACKEND=cu128
    sed -i 's/torch>=2.3.0/torch>=2.6.0,<2.11/' pyproject.toml
    echo "[setup] Blackwell+ detected — cu128 torch>=2.6"
else
    export UV_TORCH_BACKEND=cu126
    sed -i 's/torch>=2.3.0/torch>=2.3.0,<2.11/' pyproject.toml
    echo "[setup] pre-Blackwell — cu126"
fi

rm -f uv.lock
uv python pin 3.11
# --no-install-project because we only need deps; the project source lives
# on the local box and we run scripts directly by path.
uv sync --no-install-project

echo "[setup] verifying torch + cuda…"
uv run python -c "
import torch
print(f'torch {torch.__version__}  cuda ok: {torch.cuda.is_available()}')
print(f'device: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print(f'bf16 supported: {torch.cuda.is_bf16_supported()}')
"

echo ""
echo "[setup] done."
echo "Next steps:"
echo "  huggingface-cli login  # for jensjepsen/eo-mt-v11 (private)"
echo "  tmux new -s mt"
echo "  cd /workspace/espllm && uv run python scripts/translate_metamath_pod.py \\"
echo "    --output /workspace/metamath_eo.jsonl --batch-size 192 --chunk-rows 768"
