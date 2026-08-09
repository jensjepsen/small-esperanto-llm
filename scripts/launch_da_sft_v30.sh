#!/usr/bin/env bash
# Danish SFT v30 — v29 mix + STEM sci-reasoning / factcheck / taskgen.
#
# Adds 3 new STEM-derived SFT sources from da_wiki STEM articles
# (4986 curated pages via category walk from Fysik/Kemi/Biologi/Matematik/
# Astronomi/Geologi roots):
#
#   - danish-sci-reasoning-v1  — worked-calc / mechanism / counterfactual
#                                (open+closed, ~88k rows)
#   - danish-sci-factcheck-v1  — balanced SAND/FALSK verification
#                                (open+closed, ~87k rows)
#   - danish-sci-taskgen-v1    — reverse-direction: model generates tasks
#                                from an article (~186k rows, includes
#                                single/multi/qonly/balanced-batch variants)
#
# vs v29 the only change is the +3 STEM datasets. Base + pipeline identical.
# Purpose: give v29 explicit reasoning practice on science content — worked
# math, mechanism chains, counterfactuals, structured fact verification —
# where v29 was near-random on ARC-DA and GPQA-diamond.
#
# Expected step count: v29 ~42k steps/3e; adding ~360k rows ≈ +20% more
# steps → ~50k steps/3e. On H100 bs=128 ≈ 3.5h wall.
set -euo pipefail
cd /root/espllm

export PATH="$HOME/.local/bin:$PATH"

export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8

uv run python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext2048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /workspace/runs/sft/da_v30_mix19_stemreason_ropext_3e \
  --no-morpheme-preprocess \
  --sft-data \
    jensjepsen/danish-metamath-gsm:sft \
    jensjepsen/danish-algebra-sft-v5-mixed \
    jensjepsen/danish-arith-chain-sft-v1 \
    jensjepsen/danish-wiki-grounded-sft-v3:sft \
    jensjepsen/danish-text-to-question-v2:sft \
    jensjepsen/danish-sciq:sft:train \
    jensjepsen/danish-gsm8k:sft:train \
    jensjepsen/danish-instruction-following-v4:sft:train \
    jensjepsen/danish-wiki-closedqa-v1:sft \
    jensjepsen/danish-word-problems-v2 \
    jensjepsen/danish-wiki-closedqa-stem-v1:sft \
    jensjepsen/danish-wiki-broadqa-stem-v1:sft \
    jensjepsen/danish-wiki-mc-letters-v1 \
    jensjepsen/danish-rc-v1 \
    jensjepsen/danish-reason-v1 \
    jensjepsen/danish-textman-v1 \
    jensjepsen/danish-sci-reasoning-v1 \
    jensjepsen/danish-sci-factcheck-v1 \
    jensjepsen/danish-sci-taskgen-v1 \
  --epochs 3 --batch-size 128 --gradient-accumulation 1 \
  --optim adamw_bnb_8bit \
  --learning-rate 3e-5 --lr-scheduler constant_with_warmup --warmup-steps 500 \
  --save-fraction-of-epoch 0.25 --save-total-limit 2 \
  --downstream-evals gsm8k sciq citgen citmc piqa arc gpqa textman_summary textman_rewrite \
  --downstream-batch-size 128 --top-k-downstream 7 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_v30_sft_mix19_stemreason_ropext_3e \
  --wandb-tags sft da v30 mix19 ropext2048 if-v4 mc-letters task-expansion stem-reasoning stem-factcheck stem-taskgen constant-lr epochs-3 flatten-packing \
  --no-torch-compile
