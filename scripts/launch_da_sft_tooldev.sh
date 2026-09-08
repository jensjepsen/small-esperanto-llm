#!/usr/bin/env bash
# Danish SFT tool-dev — a fast A/B loop for tool-corpus work.
#
# NOT a publishable run. This is a screening config: cheap enough to iterate
# on corpus changes, dense enough in tool data that the signal arrives early.
# Rank ideas here, then confirm the winner on the full mix before believing a
# number.
#
# WHY. v41 took 6h54m to test a change touching 1.7% of the data. The tool
# corpus is 0.84% of a 2.04M-row mix, and its metrics only became legible
# around step 18k -- 40% of the way in. That is a poor loop for iterating on
# tool data.
#
#   config                            rows    steps  tool%   wall
#   v41 as run                   2,085,687   48,883  0.86%   6.7h
#   math@10k only                1,268,528   29,731  2.81%   4.6h
#   THIS (math@10k, other@60k)     764,032   17,907  4.67%   ~2.2h
#
# Capping maths alone is NOT enough -- it is 43% of the mix but the remaining
# 57% is still 1.2M rows. The 3x win needs the other large sources capped too.
#
# --source-cap takes a RANDOM sample, not a prefix: the trainer does
# `ds.shuffle(seed=42).select(range(cap))`, so a cap is a subsample rather than
# a filter, and it is deterministic across runs. It also fails loudly on a key
# that matches no --sft-data entry, so a typo cannot silently leave a source
# uncapped.
#
# SECOND PURPOSE: does the maths block earn its 43%?
#
# The July v7 ablation dropped BOTH word-problem sets (447k rows) and measured
# GSM8K 19.5 -> 20.2 (unchanged) with cit-gen unchanged -- but IFEval fell
# 41.0 -> 36.5, -4.5pp. The recorded hypothesis was that word problems
# incidentally reinforce format-compliance patterns (#### markers, list
# structure, "svaret er:" cadence) that also serve instruction-following.
#
# That ablation went to ZERO. Capping keeps the patterns and cuts the
# repetition, so if the IF benefit is exposure rather than volume, a cap should
# preserve it. The same note records that metamath, algebra and arith-chain --
# 635k rows -- have never been ablated individually at all.
#
# So gsm8k and ifeval are the ONLY non-tool evals kept: they are precisely the
# two the prior ablation implicates, and they are cheap (1,317 and 541 rows).
# citgen, sciq, icl and extraction are dropped -- icl at 1,000 rows and
# extraction at 200 long multi-shot prompts are most of the sweep cost.
#
# BATCH 32 x GA 4 -- effective batch 128, the same as v41.
#
# With --flatten-packing, --batch-size counts SAMPLES, not tokens:
# DataCollatorWithFlattening concatenates variable-length conversations with no
# padding, so tokens-per-step is set by the length distribution of the mix.
# Capping maths removed almost all the CHEAP rows -- algebra averages 51
# tokens, arith-chain 43, metamath 275 -- leaving the sources that now dominate
# at ~1,000+ (extraction 1,372, tool-dialogues 1,134, rc-v1 1,012). Mean
# tokens-per-sample rose roughly an order of magnitude, and 128 samples at
# 8048 max_length OOMed at step 17 of 16,995 on an 80GB H100.
#
# This is the trap in any config that reshapes the mix: the same batch size is
# a different amount of memory. Hold TOKENS constant, not samples.
#
# 16 was an over-correction. Measured at batch 16: 19.7 GB of 80 used, 86% GPU
# util, 1.3 it/s, ETA 3h39m -- so ~0.98 GB of activations per sample once the
# ~4 GB of weights and optimizer state are subtracted. That puts batch 32 at
# ~35 GB and batch 128 at ~125 GB, which is why 128 OOMed. 32 halves the
# per-step launch overhead of 16 while still using under half the card.
#
# GA 2, NOT 4 -- effective batch 64, and the point is OPTIMIZER STEPS.
#
# At ga 4 this config runs 16,995 steps against v41's 45,342: 2.7x fewer
# updates, because it is a 2.7x smaller corpus at the same effective batch.
# Halving ga doubles the steps to ~34,000 for almost nothing -- total
# forward/backward work is identical, only the optimizer updates double, and
# for 400M params under 8-bit Adam that is a few percent of wall clock.
#
# This also fixes a confound in the maths-cap control below. That probe matched
# TRAINING FRACTION (17% of each run), which meant v41 @7556 had 2.7x more
# optimizer steps as well as 22x more maths rows -- so the gsm8k gap it
# measured is not attributable to the cap alone. The nearest matched-STEP pair,
# v41 @3778 (8.9%) vs tooldev @2832 (2.7%), still favours v41 but by much less.
#
# UNTESTED: --learning-rate 3e-5 was set for effective batch 128. At 64 it does
# more per sample than it was tuned for. Linear scaling says halve it, but that
# cancels the extra updates, so it is left alone deliberately -- if the next
# run is unstable early, that is the first thing to change.
#
# READ FIRST, in this order:
#
#   tool_unseen_sym   symbolized twin of eval_unseen. `tool_unseen` withholds
#   vs tool_unseen    the tool NAME but keeps the parameter vocabulary, so it
#                     scores name substitution. Measured on v41 step-30224,
#                     same five rows, only the key strings changed: argF1
#                     1.000 real vs 0.600 symbolized. THE GAP IS THE
#                     DESCRIPTION-READING SIGNAL and closing it is the point
#                     of the twins.
#
#   tool_*_marker_rate   minus call_rate = the malformed-call rate. v40 could
#                        not tell silence from broken JSON; these can.
#
#   tool_refusal no_action  answerable prompts the model neither calls for nor
#                           declines. The deflection metric.
#
#   gsm8k / ifeval    the maths-cap test. ANSWERED, and the answer is NO for
#                     gsm8k: 10k is past the cliff. Controlled probe, same 30
#                     rows, k=12, matched at 17% of each run --
#
#                       tooldev @2832 (maths@10k)  greedy  0.0%  per-sample 1.4%
#                       v41     @7556 (full maths) greedy 23.3%  per-sample 14.2%
#
#                     "17% is early" is ruled out by the matched fraction. So
#                     DO NOT promote this cap to the main mix. 60k each is the
#                     untested middle: still a third off the corpus, 6x more
#                     maths than this.
#
#                     IFEval DID survive: 36.8 / 35.0 here against v41's
#                     36.8 / 36.9 at matched steps -- so the IF benefit the
#                     July word-problem ablation found is about exposure, not
#                     volume.
#
#                     For SCREENING this is fine: gsm8k is irrelevant when the
#                     question is about tool data.
#
# NOT COMPARABLE TO v41 OR v40. Different mix, different density, fewer steps.
# Compare tool-dev runs to each other only. A change that helps at 4.67% tool
# density may wash out at 0.84% -- and today's result was that a 0.03% slice
# moved behaviour, so density effects here are demonstrably not negligible.
#
# CORPUS: expects a v8-style repo carrying symbolized twins, i.e. the
# eval_seen_sym / eval_unseen_sym splits produced by scripts/symbolize_twins.py
# and routed by push_tool_dialogues_hf.py. Point ESPLLM_TOOL_EVAL_REPO at the
# same repo the run trains on -- pointing it at an older corpus is what made
# v36 read 9pp worse than v35 purely for being self-consistent.
#
# FLASH-ATTENTION IS REQUIRED (flatten-packing refuses without it):
#     WORKLOAD=sft bash scripts/setup_vastai.sh large
# START THE CHECKPOINT WATCHER BEFORE TRAINING, not after:
#     HF_TOKEN=$(cat /root/hf_token) BEST_SUBDIR=best nohup \
#       bash scripts/watch_push_best_ckpts.sh /root/runs/da_sft_tooldev \
#         jensjepsen/danish-lm-400m-sft-tooldev-mid 180 &
set -euo pipefail
cd /root/espllm

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/tmp/hf-cache
export WANDB_PROJECT=danish-lm-sft
export WANDB_API_KEY=$(grep -m1 password ~/.netrc | awk '{print $2}')
export ESPLLM_NUM_PROC=8
export ESPLLM_LIGER=0          # mutually exclusive with torch.compile
# 8.85 GiB sat reserved-but-unallocated when this OOMed; the allocator's own
# advice for that pattern.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ESPLLM_TOOL_EVAL_REPO=${TOOL_REPO:-jensjepsen/danish-tool-dialogues-v8}

TOOL_DATA=${TOOL_REPO:-jensjepsen/danish-tool-dialogues-v8}

uv run --no-sync python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext8048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/da_sft_tooldev \
  --no-morpheme-preprocess \
  --attn-impl flash_attention_2 \
  --sft-data \
    jensjepsen/danish-metamath-gsm:sft \
    jensjepsen/danish-algebra-sft-v5-mixed \
    jensjepsen/danish-arith-chain-sft-v1 \
    jensjepsen/danish-word-problems-v2 \
    jensjepsen/danish-wiki-grounded-sft-v3:sft \
    jensjepsen/danish-text-to-question-v2:sft \
    jensjepsen/danish-sciq:sft:train \
    jensjepsen/danish-gsm8k:sft:train \
    jensjepsen/danish-instruction-following-v4:sft:train \
    jensjepsen/danish-wiki-closedqa-v1:sft \
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
    "${TOOL_DATA}:sft:train" \
    "${TOOL_DATA}:abstention:train" \
  --source-cap \
    danish-metamath-gsm=10000 \
    danish-algebra-sft-v5-mixed=10000 \
    danish-arith-chain-sft-v1=10000 \
    danish-word-problems-v2=10000 \
    danish-wiki-mc-letters-v1=60000 \
    danish-wiki-closedqa-v1=60000 \
    danish-reason-v1=60000 \
    danish-instruction-following-v4=60000 \
    danish-textman-v2=60000 \
    danish-wiki-grounded-sft-v3=60000 \
    danish-rc-v1=60000 \
    danish-wiki-closedqa-stem-v1=60000 \
    danish-extraction-v1=60000 \
  --epochs 3 --batch-size 32 --gradient-accumulation 2 \
  --optim adamw_bnb_8bit \
  --learning-rate 3e-5 --lr-scheduler constant_with_warmup --warmup-steps 500 \
  --max-length 8048 \
  --flatten-packing \
  --torch-compile \
  --save-fraction-of-epoch 0.5 --eval-fraction-of-epoch 0.5 \
  --save-total-limit 3 --top-k-downstream 3 \
  --downstream-evals gsm8k ifeval tool_seen tool_unseen tool_seen_sym \
                     tool_unseen_sym tool_answer tool_refusal \
  --downstream-n 0 --downstream-batch-size 64 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_sft_tooldev_math10k_other60k_twins \
  --wandb-tags sft da tooldev screening math-capped tool-dense twins \
               symbolized adam8bit fa2 torch-compile no-liger epochs-3 h100 \
  "$@"
