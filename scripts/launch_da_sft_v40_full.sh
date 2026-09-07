#!/usr/bin/env bash
# Danish SFT v40 — v39's mix with the tool corpus rebuilt, plus abstention.
#
#   jensjepsen/danish-tool-dialogues-v5  ->  -v6
#   + jensjepsen/danish-tool-dialogues-v6:abstention:train   (694 rows)
#   + tool_refusal as a tenth downstream eval
#
# WHAT v6 CHANGES. v5's tool contracts were built by keying the returns map on
# the tool NAME. A name is not a function in this corpus: 379 of 875 names carry
# more than one parameter schema and search_movies carries 63, because glaive
# invented every dialogue independently. So the return fields of unrelated
# functions were unioned -- search_quotes acquired an AAPL stock ticker -- and
# only 65.9% of each declared contract was actually present in the payload it
# described. v6 keys on (name, signature) and keeps fields present in >=90% of
# that signature's real payloads:
#
#   contract fidelity F1     90.6%   (v5 75.3)   precision 92.7 (65.9)
#   specs with a contract    98.2%   (v5 47%)
#
# The same shape of bug was in the answers. The v5 gate asked only "does the
# reply cite SOME payload value", which makes reciting the whole payload
# optimal -- 43% of v5's answers cite every field. The tool_answer eval scored
# it the same way (recall over all payload values) and therefore ranked a
# padded model answer above the reference reply, 47.8% to 26.1%. v6's generator
# declares which fields the question asks for, the gate rejects answers that
# miss them or drag in others, and the eval is now F1 against the fields gold
# cites:
#
#   cited values per answer    2.2   (v5 3.7), 96% of them relevant
#
# READ tool_refusal FIRST, then the calling metrics.
#
#   tool_refusal   NEW and two-sided. 694 abstention rows teach the model to
#                  decline when no tool can serve the request; nothing else in
#                  the suite would notice that spreading onto answerable
#                  questions -- tool_seen would only show emitted-a-call
#                  falling, without saying whether the model refused, rambled,
#                  or emitted malformed JSON. Reports refused-when-it-should
#                  and wrongly-refused separately; the headline is their
#                  balanced accuracy. A model trained without abstention data
#                  should read ~50% (never refuses: 0% correct, 0% false).
#                  Watch wrongly-refused above all: over-refusal is a worse
#                  failure than the one abstention fixes.
#
#   tool_seen /    argF1. The RISK metric for this run. v6 has 21% FEWER answer
#   tool_unseen    turns than v5 (12,550 vs 15,141) because the precision gate
#                  rejects more, and 694 rows teach declining. Both push toward
#                  a model that says less. If these fall, over-restriction is
#                  the first hypothesis and tool_refusal is what tests it.
#                  v39 best: 90.4 seen / 80.2 unseen (avg-top3 91.0 / 80.6).
#
#   tool_answer    NOT COMPARABLE TO v39. The metric changed from recall over
#                  all payload values to F1 against the fields gold cites, and
#                  the eval split was rebuilt with corrected contracts. v39's
#                  71.7 belongs to the old yardstick. Under the new one a model
#                  that answers only what was asked scores HIGHER, and a model
#                  that recites the payload scores lower -- the reverse of
#                  before. Gold scores 100% by construction; a reply lifted
#                  from another row still scores ~0.
#
# ABSTENTION IS AT NATURAL WEIGHT: 694 rows in a ~2M-conversation mix, i.e.
# almost nothing. This run asks whether the behaviour is learnable at all, not
# whether it is learned well. If tool_refusal moves off ~50% without hurting
# tool_seen, the follow-up is to upweight it.
#
# Everything else in the mix is unchanged from v39, so non-tool evals should
# hold. v6 is a similar token size to v5, so flat weighting is right here --
# unlike the v4 -> v5 step, where the corpus lost 70% of its tokens.
#
# v39 for comparison (final checkpoint, step-45324):
#   agg 0.445 | tool_seen 90.4 | tool_unseen 80.2 | tool_answer 71.7*
#   avg-top3: tool_seen 91.0 | tool_unseen 80.6 | tool_answer 72.3*
#   *old metric, old split
#
# EVAL BATCH 96, not 32. The Pro 6000 has 96GB against the H100's 80, and the
# eval is generation-bound. Worth knowing when reading the tables: batch size
# shifts generation slightly through padding and attention-mask numerics, so
# v40's downstream numbers are not exactly comparable to v39's at batch 32.
# The tool metrics are the ones this run is about, and they change yardstick
# anyway (v6 split, F1 rather than recall), so nothing is lost that was not
# already lost -- but the non-tool evals gain a small asterisk.
#
# FLASH-ATTENTION IS REQUIRED (flatten-packing refuses without it):
#     WORKLOAD=sft bash scripts/setup_vastai.sh large
# START THE CHECKPOINT WATCHER BEFORE TRAINING, not after:
#     HF_TOKEN=$(cat /root/hf_token) BEST_SUBDIR=best nohup \
#       bash scripts/watch_push_best_ckpts.sh /root/runs/da_sft_v40_full \
#         jensjepsen/danish-lm-400m-sft-v40-mid 180 &
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
# self-consistent -- the yardstick was the defect. tool_refusal also reads its
# positive half from this repo's `abstention` config.
export ESPLLM_TOOL_EVAL_REPO=jensjepsen/danish-tool-dialogues-v6

# --no-sync: WORKLOAD=sft pins torch<2.9 so a prebuilt FA2 wheel matches, but
# the `all` extra still declares vllm>=0.17 (torch>=2.10), so a plain `uv run`
# re-resolves and dies before training starts.
uv run --no-sync python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext8048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/da_sft_v40_full \
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
    jensjepsen/danish-tool-dialogues-v6:sft:train \
    jensjepsen/danish-tool-dialogues-v6:abstention:train \
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
  --downstream-n 0 --downstream-batch-size 96 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_sft_v40_full_mix23_tooldialogues_v6_abstention \
  --wandb-tags sft da v40 full-resft mix23 tool-dialogues-v6 signature-contracts \
               precision-gate abstention tool-refusal no-reasoning \
               multiturn textman-v2 extraction adam8bit fa2 torch-compile \
               no-liger epochs-3 h100 \
  "$@"
