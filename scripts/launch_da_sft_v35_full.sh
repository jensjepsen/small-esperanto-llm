#!/usr/bin/env bash
# Danish SFT v35 — v34's mix with two changes, both attributable.
#
#   1. + jensjepsen/danish-tool-dialogues-v1  (17,138 rows, ~33 MB)
#   2.   danish-textman-v1 -> v2              (the textman_extraction subtype
#                                              removed; 99,610 rows, ~374 MB)
#
# Everything else is held at v34's values -- bs 128 x ga 1, lr 3e-5
# constant_with_warmup/500, 3 epochs, max_len 8048, FA2 + flatten packing,
# compile on / Liger off, adamw_bnb_8bit, extraction capped at 60k.
#
# WHY THESE TWO. textman_extraction used ONE fixed schema across 20,018 rows
# and 26% of its `numbers` were absent from the passage, so it taught a
# hardcoded key set while danish-extraction-v1 teaches "read the keys you were
# given" -- v34 trained both, at a 1:3.6 byte ratio, which is the confound
# recorded in project_v34_sft. v2 is a FILTER of v1, so the five surviving
# subtypes are byte-identical and this comparison isolates the removal.
#
# The tool dialogues are the first MULTI-TURN data in the mix: every one of the
# other 21 sources is a single user turn followed by a single assistant turn
# (median 4 turns here, tool results fed back). That also makes the multi-turn
# label-masking fix load-bearing for the first time -- before it, later user
# turns were training targets, i.e. the model was taught to write the user's
# messages. See _build_label_masker.
#
# TWO NEW EVALS, deliberately separate:
#   tool_seen    unseen CONVERSATIONS over tools that appear in training
#   tool_unseen  conversations that CALL a tool held out entirely (56/56
#                verified absent from train)
# Only the second measures generalisation. Both score graded pair-F1 on
# arguments with a wrong tool scoring zero, and log call_rate / tool_acc as
# sub-metrics so a zero can be attributed rather than guessed at.
#
# FLASH-ATTENTION IS REQUIRED (flatten-packing refuses without it):
#     WORKLOAD=sft bash scripts/setup_vastai.sh large
set -euo pipefail
cd /root/espllm

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/tmp/hf-cache
export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8
export ESPLLM_LIGER=0          # mutually exclusive with torch.compile

# --no-sync: WORKLOAD=sft pins torch<2.9 so a prebuilt FA2 wheel matches, but
# the `all` extra still declares vllm>=0.17 (torch>=2.10), so a plain `uv run`
# re-resolves and dies before training starts.
uv run --no-sync python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext8048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/da_sft_v35_full \
  --no-morpheme-preprocess \
  --attn-impl flash_attention_2 \
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
    jensjepsen/danish-textman-v2 \
    jensjepsen/danish-arc:sft:train \
    jensjepsen/danish-openbookqa:sft:train \
    jensjepsen/danish-ner-sft-v1:sft:train \
    jensjepsen/danish-icl-schema-format-v3:sft:train \
    jensjepsen/danish-extraction-v1:sft:train \
    jensjepsen/danish-tool-dialogues-v1:sft:train \
  --source-cap danish-extraction-v1=60000 \
  --epochs 3 --batch-size 128 --gradient-accumulation 1 \
  --optim adamw_bnb_8bit \
  --learning-rate 3e-5 --lr-scheduler constant_with_warmup --warmup-steps 500 \
  --max-length 8048 \
  --flatten-packing \
  --torch-compile \
  --save-fraction-of-epoch 0.25 --eval-fraction-of-epoch 0.25 \
  --save-total-limit 3 --top-k-downstream 3 \
  --downstream-evals gsm8k citgen sciq ifeval icl extraction tool_seen tool_unseen \
  --downstream-n 0 --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_sft_v35_full_mix22_tooldialogues_textmanv2 \
  --wandb-tags sft da v35 full-resft mix22 tool-dialogues multiturn textman-v2 \
               extraction adam8bit fa2 torch-compile no-liger epochs-3 h100 \
  "$@"
