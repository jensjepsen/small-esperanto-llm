#!/usr/bin/env bash
# Danish SFT v33 — full re-SFT: the v31 18-dataset mix PLUS the new
# structured-output data, from the pretrained base.
#
# This is a fresh SFT in the v31 lineage, not continued SFT on v31. Base is
# the same ropext8048 pretrain checkpoint v31 started from, and the LR/warmup
# match v31 (3e-5, 500) rather than the 1e-5 used for continued runs.
#
# Data: 1,919,572 rows of the v31 mix, uncapped, + 24,000 NER + 33,933 ICL.
# The continued-SFT experiments showed replay VOLUME is what protects the
# general abilities (composition and LR both made things worse), and a full
# re-SFT is the limit of that: the structured data is 3% of the corpus rather
# than 50%, so there is nothing to forget.
#
# Throughput config, measured on this H100 at bs128/FA2 (see
# scripts/bench_sft_throughput.py):
#     liger on,  compile off   110,574 tok/s   28% MFU
#     liger off, compile on    127,277 tok/s   32% MFU   <- chosen
#     neither                   98,275 tok/s   25% MFU
#     both                     CRASH (inductor AssertionError)
# Liger and torch.compile cannot both be on. Taking compile costs Liger's
# fused LM head, so optimizer state will not transfer to/from a Liger run --
# acceptable here because this is a fresh SFT, not a resume.
#
# FLASH-ATTENTION IS REQUIRED. train_sft_packed.py now refuses --flatten-packing
# without it: under SDPA the flattening collator emits no cu_seq_lens and
# packed samples attend across their boundaries (measured 7.8 logit shift vs
# 0.0 under FA2). Provision the box with:
#     WORKLOAD=sft bash scripts/setup_vastai.sh large
#
# ~1.98M rows x 3 epochs at eff_bs 128 = ~46,400 steps, ~4.5 h plus eval.
# Eval and save every 0.25 epoch = 12 points; top-3 by downstream aggregate.
# --downstream-n 0 = FULL test sets. A 200-row subsample carries ~3.5pp of
# sampling noise per eval, which is the same order as the gaps the top-k
# ranking is built from -- so checkpoint selection ends up partly noise-driven
# even though the step-rotated seed removes the v15 fixed-subset bias. Full
# sets cost ~4-5x the eval-generation time; pay it, the ranking is the point.
# are preserved in best/ so save_total_limit rotation cannot evict them.
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
# re-resolves and dies with "requirements are unsatisfiable" before training
# starts. --no-sync runs the already-built venv untouched.
uv run --no-sync python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext8048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/da_sft_v33_full \
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
    jensjepsen/danish-textman-v1 \
    jensjepsen/danish-arc:sft:train \
    jensjepsen/danish-openbookqa:sft:train \
    jensjepsen/danish-ner-sft-v1:sft:train \
    jensjepsen/danish-icl-schema-format-v3:sft:train \
  --epochs 3 --batch-size 128 --gradient-accumulation 1 \
  --optim adamw_torch_fused \
  --learning-rate 3e-5 --lr-scheduler constant_with_warmup --warmup-steps 500 \
  --max-length 8048 \
  --flatten-packing \
  --torch-compile \
  --save-fraction-of-epoch 0.25 --eval-fraction-of-epoch 0.25 \
  --save-total-limit 3 --top-k-downstream 3 \
  --downstream-evals gsm8k citgen sciq ifeval icl --downstream-n 0 --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_sft_v33_full_mix20_bs128_fa2_compile \
  --wandb-tags sft da v33 full-resft mix20 ner icl span-wrap fa2 torch-compile no-liger epochs-3 h100
