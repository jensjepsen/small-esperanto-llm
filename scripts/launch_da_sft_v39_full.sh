#!/usr/bin/env bash
# Danish SFT v39 — v38's mix with ONE change:
#
#   jensjepsen/danish-tool-dialogues-v4  ->  -v5
#
# WHAT v5 CHANGES: the reasoning is gone, and the calls are answered.
#
#   reasoning tokens              0        v4: 78.7% of the tool corpus
#   grounded answer turns    25,389        v4:  8,284
#   rows ending in an answer   99.9%       v4: rows ended at the call
#   tools documenting returns    94%       v4: 47%
#
# WHY. In v4 both the reasoning that precedes a call and the answer that
# follows a result live in the <|assistant|> slot, at 22.7:1 by token. After a
# <|tool_result|> the model chose between two learned registers on those odds
# and picked the narrating one. Probed on four unseen tools it emitted 7 of 8
# calls correctly and grounded 0 of 8 answers -- handed {"dog_years": 28} it
# replied "4 * 12 = 48", and twice re-opened <|tool_call|> in the answer slot.
#
# Underneath the ratio was a coverage hole. glaive splits its tool vocabulary
# almost exactly in half: 445 of 890 tools never get a result ANYWHERE in the
# raw data (68.0% of raw calls are unanswered). get_lyrics is called 100 times
# and never once returns anything, so for half the catalogue the answer turn
# did not exist at any ratio. v5 proposes a returns contract for those tools
# and generates the missing result+answer against it.
#
# WHAT IS SYNTHETIC HERE. Those 443 contracts are INVENTED -- nothing in the
# source says what those tools return -- and ~16k of the 25,389 answer turns
# are generated against them, gated for grounding, invented numbers, meta-talk
# and field shape. ToolMind's APIGen-MT and tau-train files carry REAL results
# for 85% and 80% of their calls; taking those beats inventing, and is the
# obvious next move if v5 reads well enough to be worth doing properly.
#
# NATURAL WEIGHT, DELIBERATELY. v5 carries ~30% of v4's trained tokens, and
# this run does NOT compensate. So a v39-vs-v38 delta mixes "reasoning removed"
# with "70% less tool data" and cannot separate them. That is the chosen
# trade: cheapest first look. If tool metrics move, the follow-up is a run with
# the tool rows upweighted to hold trained-token share constant -- only that
# one isolates the reasoning change.
#
# READ IN THIS ORDER:
#   tool_answer          the metric v5 targets. Ceiling 83.1% (gold replies);
#                        a reply lifted from another row scores 3.4%, fluent
#                        Danish with no facts 0.0%. v38 ran 74-80%.
#   tool_seen/unseen     argF1. The RISK metric. Reasoning was the immediate
#                        context that built each call, and the last clean
#                        measurement had it HELPING -- free 4/4 vs forced 3/4
#                        on unseen tools. A drop here means reasoning earned
#                        its tokens and belongs back behind its own marker
#                        rather than deleted.
#   right-tool           against ~24.7% chance, NOT against v36's 84-93%,
#                        which were measured when the answer sat in position 0.
#
# Everything else in the mix is unchanged from v38, so non-tool evals should
# hold. If sciq/ifeval/icl/extraction move much, suspect the shorter tool rows
# changing how the packer fills sequences, not the datasets.
#
# v38 for comparison (nine-eval aggregate, same eval set):
#   step-37770 agg 0.437 | tool_seen argF1 76.8 | tool_unseen 65.2
#   tool_answer 80.3 | right-tool 92.2 / 83.6
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
# Score against the corpus this run TRAINS on. Pointing the tool eval at an
# older corpus is what made v36 read 9pp worse than v35 purely for being
# self-consistent -- the yardstick was the defect. This matters more than
# usual here: v4's eval prompts stop before a reasoning turn this model is no
# longer trained to produce.
export ESPLLM_TOOL_EVAL_REPO=jensjepsen/danish-tool-dialogues-v5

# --no-sync: WORKLOAD=sft pins torch<2.9 so a prebuilt FA2 wheel matches, but
# the `all` extra still declares vllm>=0.17 (torch>=2.10), so a plain `uv run`
# re-resolves and dies before training starts.
uv run --no-sync python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext8048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/da_sft_v39_full \
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
    jensjepsen/danish-tool-dialogues-v5:sft:train \
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
                     tool_answer \
  --downstream-n 0 --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_sft_v39_full_mix22_tooldialogues_v5_noreason \
  --wandb-tags sft da v39 full-resft mix22 tool-dialogues-v5 no-reasoning \
               answered-calls proposed-returns natural-weight \
               multiturn textman-v2 extraction adam8bit fa2 torch-compile \
               no-liger epochs-3 h100 \
  "$@"
