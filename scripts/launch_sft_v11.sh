#!/bin/bash
# SFT launch — v11 = v10 mix16 + 3 datasets that were previously built but
# excluded from v10. Motivation: v10 lost factoid recall relative to base
# pretrain (verbatim probe: nationality 71→41, birthplace 29→3, father 20→0;
# interrogative Q/A: nationality 60→48, birthplace 8→2). The training data
# to close that gap already exists on HF; we just weren't training on it.
#
# Additions:
#   esperanto-sft-factoid       131k   multi-turn wikidata Q/A over the
#                                       same properties the probe measured
#   esperanto-sft-wikidata-icl   15k   ICL list-completion over the same slice
#   esperanto-sft-atomic-qa      30k   ATOMIC-style commonsense Q/A
#
# Usage:
#   bash scripts/launch_sft_v11.sh <checkpoint-dir> [output-dir]
set -euo pipefail

CHECKPOINT=${1:?usage: launch_sft_v11.sh <checkpoint> [output]}
OUTPUT=${2:-runs/sft/v11_sft_v11_mix}

SFT_SOURCES=(
  # v10 baseline (16 sources) —
  jensjepsen/esperanto-orca-math
  jensjepsen/esperanto-alpaca-distill
  jensjepsen/esperanto-alpaca-cleaned
  jensjepsen/esperanto-sft-dolly
  jensjepsen/esperanto-gsm8k
  jensjepsen/esperanto-algebra-sft-v5-mixed
  jensjepsen/esperanto-arith-chain-sft-v1
  jensjepsen/esperanto-word-problems-v4:sft
  jensjepsen/esperanto-sciq-sft
  jensjepsen/esperanto-balanced-copa-sft
  jensjepsen/esperanto-piqa-sft
  jensjepsen/esperanto-mmlu-sft
  jensjepsen/esperanto-triviaqa-sft
  jensjepsen/esperanto-metamath-gsm:sft-ascii
  jensjepsen/esperanto-squad:sft
  jensjepsen/esperanto-sft-creative
  # NEW in v11 — factoid recall repair —
  jensjepsen/esperanto-sft-factoid
  jensjepsen/esperanto-sft-wikidata-icl
  jensjepsen/esperanto-sft-atomic-qa
  # NEW in v11 — NLI / commonsense reasoning with rationales —
  #   ECQA:  CommonsenseQA + human rationale (5-choice)
  #   e-CARE: causal reasoning + conceptual explanation (2-choice)
  #   e-SNLI: NLI with human-written rationale (3-way, 100k cap)
  jensjepsen/esperanto-ecqa:sft
  jensjepsen/esperanto-ecare:sft
  jensjepsen/esperanto-esnli:sft
)

echo "=== SFT mix (${#SFT_SOURCES[@]} sources, full-size) ==="
printf '  %s\n' "${SFT_SOURCES[@]}"

HF_HOME=${HF_HOME:-/mnt/data2/hf_cache} \
uv run python -u scripts/train_sft_packed.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT" \
  --sft-data "${SFT_SOURCES[@]}" \
  --epochs 3 \
  --batch-size 32 \
  --gradient-accumulation 1 \
  --learning-rate 5e-5 \
  --max-length 512 \
  --lr-scheduler cosine_with_min_lr \
  --warmup-steps 200 \
  --save-fraction-of-epoch 0.5 \
  --save-total-limit 3 \
  --wandb-run-name "v11_sft_v11_mix" \
  --wandb-tags sft v11 mix19 factoid-recall-repair
