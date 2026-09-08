#!/usr/bin/env bash
# Danish SFT v41 — v40's mix with the tool corpus repaired.
#
#   jensjepsen/danish-tool-dialogues-v6  ->  -v7   (sft + abstention)
#   ESPLLM_TOOL_EVAL_REPO                ->  -v7
#
# ONE THING CHANGES: the tool corpus. Everything else is v40's mix, batch,
# schedule and eval config, so a difference in the tool metrics is attributable.
#
# WHAT v7 REMOVES. v6 carried a contiguous batch of upstream rows -- 19258..20016
# of glaive-function-calling-v2-query.jsonl, which survived translation in order
# and landed at sft rows 17004..17666 -- where the assistant DECLINES a task its
# catalogue covers, a canned user turn says more functions are available, and it
# then calls correctly. v6 row 17283 refuses check_flight_status and invents
# referral URLs before complying. That batch supplied 98.2% of the rows with no
# tool call at all and 92.6% of the rows shipping a reasoning trace as the
# answer:
#
#                             v6      v7
#   deflect-then-call        593       0
#   never calls a tool       448       0
#   reasoning as the answer  394       0
#   identifier in Danish   ~1100       0   (translator bug, see below)
#   <think> block shipped    522       0
#
# THE QUESTION THIS RUN ASKS. v40's `emitted-a-call` fell for three consecutive
# checkpoints -- 98.8 -> 95.1 -> 93.4 on seen, 92.8 -> 84.8 on unseen -- while
# `wrongly-refused` stayed at exactly 0.0%, so it was NOT over-refusal. The
# probe showed what it was instead: "Selvfølgelig, det kan jeg godt hjælpe med.
# Lad mig kaste 3 terninger for dig." and then nothing. A promise to act with no
# action, reproduced in near-identical words at step-22674 and step-26453, with
# the forced prefill calling correctly both times.
#
# That is the text signature of the 593 rows v7 drops. If removing them keeps
# emission up, the hypothesis survives; if emission decays the same way, the
# cause is elsewhere and late-training drift is the better explanation. It is
# correlational either way -- those rows are ~0.05% of a ~2M-conversation mix --
# so treat a null result as informative rather than as vindication of v6.
#
# READ THESE THREE FIRST, all new in this run:
#
#   tool_seen_marker_rate    the <|tool_call|> marker before any parsing.
#   tool_seen_call_rate      the call that PARSES. Their DIFFERENCE is the
#                            malformed-call rate. v40 did both failures and the
#                            old single number could not tell them apart: at
#                            step-22674 it emitted `{"name":…,"arguments":{…},
#                            "time":…}` with `time` outside `arguments`, and at
#                            26453 it emitted no marker at all.
#
#   tool_refusal_no_action   answerable prompts where the model neither called
#                            nor declined. THE metric for the question above.
#                            Invisible before: it earns no credit on the
#                            positives and costs nothing on the negatives, which
#                            is how wrongly-refused held at 0.0% through an 8pp
#                            fall in emission.
#
#   tool_refusal_absent_field  the tool fits and must be called, but the result
#                            lacks the field asked for -- call, read, decline on
#                            that field. 128 rows were TRAINED on this in v40
#                            and never scored; probes read 0/3 across four
#                            checkpoints. A local smoke on v40 step-26453 put it
#                            at 15% against no-tool's 50%, so expect it to be
#                            the weak half.
#
# ABSTENTION STAYS AT NATURAL WEIGHT (673 rows), deliberately. v40 answered
# whether it is learnable: correct-refusal went 10.7 -> 29.0 with wrongly-refused
# at 0.0 throughout, against v39 -- which had zero abstention rows -- scoring
# 0/7 on the same probe. Upweighting is now an optimisation, and doing it here
# would confound the emission question this run exists to answer. If v41
# confirms the deflection hypothesis, v42 upweights, and the target is the 128
# absent-field rows specifically rather than all 673.
#
# NOT COMPARABLE TO v40:
#   tool_refusal   three buckets instead of two, pooled differently, and the
#                  positive half's prompt format was wrong in v40 (hand-built
#                  `{USER}{q}{END}{ASST}`, an <|end|> after a user turn, which
#                  occurs nowhere in training). A local re-score of v40's own
#                  step-26453 under the corrected format read no-tool 50% where
#                  the run had logged 28.2% -- three confounds in that gap
#                  (format, corpus, sample), none isolated, but do not read the
#                  two scales as one.
#   tool_seen /    v7's eval splits were rebuilt, so the row sets differ.
#   tool_unseen /  Trends WITHIN this run are the readable thing.
#   tool_answer
#
# Non-tool evals are unchanged from v40 and should hold. v40 for reference,
# stopped at 63% (step ~28,400 of 45,351):
#   peak agg 0.479 @ step-18895 | tool_seen 88.9 | tool_unseen 78.0
#   tool_answer 91.3 @ 26453 | tool_refusal 64.5 @ 18895 (old scale)
# agg never beat 18895 across the next three evals, over the same span emission
# was falling.
#
# EVAL COST IS UP ~6%: tool_refusal is 478 prompts (150 no-tool + 128
# absent-field + 200 answerable) against v40's 300, because absent-field is only
# 128 rows and a pooled cap left it ~28 items -- unreadable, and it is the half
# that fails. Batch 64 is unchanged and was proven on an 80GB H100 in v40 with
# no OOM across seven evals.
#
# FLASH-ATTENTION IS REQUIRED (flatten-packing refuses without it):
#     WORKLOAD=sft bash scripts/setup_vastai.sh large
# START THE CHECKPOINT WATCHER BEFORE TRAINING, not after:
#     HF_TOKEN=$(cat /root/hf_token) BEST_SUBDIR=best nohup \
#       bash scripts/watch_push_best_ckpts.sh /root/runs/da_sft_v41_full \
#         jensjepsen/danish-lm-400m-sft-v41-mid 180 &
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
# self-consistent -- the yardstick was the defect.
export ESPLLM_TOOL_EVAL_REPO=jensjepsen/danish-tool-dialogues-v7

# --no-sync: WORKLOAD=sft pins torch<2.9 so a prebuilt FA2 wheel matches, but
# the `all` extra still declares vllm>=0.17 (torch>=2.10), so a plain `uv run`
# re-resolves and dies before training starts.
uv run --no-sync python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext8048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/da_sft_v41_full \
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
    jensjepsen/danish-tool-dialogues-v7:sft:train \
    jensjepsen/danish-tool-dialogues-v7:abstention:train \
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
                     tool_answer tool_refusal \
  --downstream-n 0 --downstream-batch-size 64 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_sft_v41_full_mix23_tooldialogues_v7 \
  --wandb-tags sft da v41 full-resft mix23 tool-dialogues-v7 pedagogy-filtered \
               no-deflection-rows identifier-clean abstention natural-weight \
               marker-rate no-action absent-field multiturn textman-v2 \
               extraction adam8bit fa2 torch-compile no-liger epochs-3 h100 \
  "$@"
