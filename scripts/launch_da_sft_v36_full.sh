#!/usr/bin/env bash
# Danish SFT v36 — v35's mix with ONE change:
#
#   jensjepsen/danish-tool-dialogues-v1  ->  -v2
#
# Everything else is held at v35's values -- bs 128 x ga 1, lr 3e-5
# constant_with_warmup/500, 3 epochs, max_len 8048, FA2 + flatten packing,
# compile on / Liger off, adamw_bnb_8bit, extraction capped at 60k, the same
# eight downstream evals. v35 finished at agg 0.408 (step 45312, the LAST
# checkpoint, still climbing), so this is a clean single-variable A/B.
#
# WHAT v2 FIXES. v1 translated argument values PER ROW, so the same string
# acquired two Danish forms in different rows: calculate_area.shape came back
# "rektangel" 506 times and "rectangle" 69, get_news.category split
# sports/sport 202/85. 66 slots held both languages, 3,231 values, minority
# share 40.8%. For 2,804 of those the spec declares no enum, so nothing in the
# prompt tells the model which form that row wants -- it learns the majority
# and is scored wrong whenever gold holds the minority. Roughly 4.6% of string
# arguments were unreachable by construction.
#
# v2 translates every distinct value ONCE, keyed by (tool, arg_key, value),
# and hands those canonical terms to the per-row pass as a glossary so prose
# and payload agree. Measured on smoke: unquoted English terms in assistant
# prose 26% -> 7%, and spec/call agreement holds by construction rather than
# by a post-hoc repair.
#
# v2 also pins values for 363 rows (2.12%) whose tool answer depends on the
# literal characters -- check_palindrome, check_word_count, check_spelling,
# generate_anagram, reverse_string. v1 translated `racecar` to `racerbil` and
# then asserted it was a palindrome, against a tool result computed on the
# English string. Those rows taught a false fact and passed every gate.
#
# READ tool_seen / tool_unseen AGAINST v35:
#   tool_seen    emitted 93.1%  right-tool 93.1%  argF1 81.2%
#   tool_unseen  emitted 93.6%  right-tool 93.6%  argF1 82.5%
# argF1 is the metric v2 should move; emission and tool choice are already
# near ceiling and have no headroom worth reading. NOTE the eval sets are
# regenerated from v2, so a v35-vs-v36 comparison on the tool evals is NOT
# strictly like-for-like -- the enum coin-flip is removed from the TEST as
# well as the train. Expect a jump partly for that reason; the non-tool evals
# are the unconfounded ones.
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
  --output-dir /root/runs/da_sft_v36_full \
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
    jensjepsen/danish-tool-dialogues-v2:sft:train \
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
  --wandb-run-name da_sft_v36_full_mix22_tooldialogues_v2 \
  --wandb-tags sft da v36 full-resft mix22 tool-dialogues-v2 value-lexicon \
               multiturn textman-v2 extraction adam8bit fa2 torch-compile \
               no-liger epochs-3 h100 \
  "$@"
