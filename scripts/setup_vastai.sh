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

# Two mutually exclusive stacks. Pick with WORKLOAD=grpo|sft.
#
#   grpo (default)  vllm>=0.17 for rollouts -> torch>=2.10 -> NO flash-attn
#                   wheel exists for py3.11+torch2.10+cu12 (torch 2.10 wheels
#                   are cp312+cu13 only), so attention falls back to SDPA.
#                   Fine here: GRPO does not pack sequences.
#
#   sft             torch pinned <2.9 so the cp311+cu12 FA2 wheel matches, and
#                   flash-attn installed. REQUIRED for SFT: train_sft_packed.py
#                   uses DataCollatorWithFlattening, and under SDPA the packed
#                   samples attend across their boundaries. Measured on a
#                   400M ckpt: swapping the first packed sample moved the
#                   second sample's logits by 7.8 under SDPA and by 0.0 under
#                   FA2. That is the exact contamination the flattening
#                   collator was adopted to remove.
#
# Do not set SKIP_FLASH_ATTN=1 for an SFT box. The old default was 1 with the
# note "SDPA is fine for GRPO on 400M" -- true for GRPO, and silently wrong
# once the same image was used for packed SFT.
export WORKLOAD=${WORKLOAD:-grpo}
if [ "$WORKLOAD" = "sft" ]; then
    export SKIP_VLLM=1
    export SKIP_FLASH_ATTN=${SKIP_FLASH_ATTN:-0}
    # vllm lives in the `all` extra, so the resolver rejects torch<2.9 even
    # with --extra train. Pin torch explicitly after the sync instead.
    export PIN_TORCH=${PIN_TORCH:-2.8.0}
else
    export SKIP_FLASH_ATTN=${SKIP_FLASH_ATTN:-1}
fi
echo "=== WORKLOAD=$WORKLOAD  SKIP_VLLM=${SKIP_VLLM:-0}  SKIP_FLASH_ATTN=$SKIP_FLASH_ATTN ==="

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
# 30-40% throughput boost when present).
#
# vllm MUST be part of this same resolution, not installed afterwards. The
# `vllm` extra pins vllm>=0.17,<=0.26 precisely because newer vllm (0.27+)
# requires torch 2.9-2.13 on CUDA 13, which conflicts with the cu128 stack
# Blackwell needs. A later `uv pip install vllm` resolves vllm ALONE, ignores
# the extras pin, and silently upgrades torch to 2.13+cu130 — which leaves
# torch.cuda.is_available() False against a 12.8 driver and strands the
# nvidia-* runtime libs (libcusparseLt.so.0 goes missing). Resolving both
# extras together is what makes uv honour the pin and keep torch on cu128.
#
# Set SKIP_VLLM=1 for boxes that only run SFT/pretrain and don't need rollout
# acceleration.
uv python pin 3.11
if [ "${SKIP_VLLM:-0}" = "1" ]; then
    echo "=== Syncing deps (train extra only; SKIP_VLLM=1) ==="
    uv sync --extra train
else
    echo "=== Syncing deps (train + vllm extras, resolved together) ==="
    uv sync --extra train --extra vllm
fi

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
# For an SFT box, force torch onto a version that HAS a prebuilt FA2 wheel.
# `uv sync` cannot do this: the `all` extra carries vllm>=0.17 which requires
# torch>=2.10, so the resolver calls torch<2.9 unsatisfiable no matter which
# extras are selected. Installing torch directly sidesteps that; torchvision
# and torchaudio are not used by the SFT path.
if [ -n "${PIN_TORCH:-}" ]; then
    CUR=$(uv run python -c "import torch;print(torch.__version__.split('+')[0])" 2>/dev/null || echo none)
    if [ "$CUR" != "$PIN_TORCH" ]; then
        echo "=== Pinning torch $CUR -> $PIN_TORCH (so a prebuilt FA2 wheel exists) ==="
        uv pip install --index-url "https://download.pytorch.org/whl/${UV_TORCH_BACKEND}" "torch==${PIN_TORCH}"
    fi
fi

if [ "$SKIP_FLASH_ATTN" = "1" ]; then
    echo "=== Skipping flash-attn install (SKIP_FLASH_ATTN=1). OK for GRPO. NOT OK for packed SFT: SDPA leaks attention across packed samples. Use WORKLOAD=sft. ==="
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
    # --no-deps: flash-attn declares torch and would otherwise re-resolve it.
    # einops is its only other runtime import, so install that explicitly.
    uv pip install --no-deps "${WHEEL}"
    uv pip install --no-deps einops
    uv run python -c "import flash_attn; print(f'flash-attn OK ({flash_attn.__version__})')"
    # Prove ISOLATION, not just import. A packed batch must not let one sample
    # attend to another; that is the whole reason FA2 is required here.
    uv run python - <<'FAVERIFY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithFlattening
C = "jensjepsen/danish-lm-400m-sft-v31-avg-top3"
tok = AutoTokenizer.from_pretrained(C)
coll = DataCollatorWithFlattening()
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
a, b = enc("Danmark er et land i Norden."), enc("Kvantemekanik beskriver partikler.")
t = enc("Hovedstaden i Danmark hedder")
m = AutoModelForCausalLM.from_pretrained(
    C, attn_implementation="flash_attention_2", dtype=torch.bfloat16).cuda().eval()
def lg(p):
    f = [{"input_ids": p, "labels": p}, {"input_ids": t, "labels": t}]
    batch = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in coll(f).items()}
    with torch.no_grad():
        return m(**{k: v for k, v in batch.items() if k != "labels"}).logits[0, -1].float()
d = (lg(a) - lg(b)).abs().max().item()
assert d < 1e-3, f"packed samples still leak under FA2 (max logit delta {d})"
print(f"packed-sample isolation verified (max logit delta {d})")
FAVERIFY
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
