#!/usr/bin/env bash
# Danish SFT v34 — v33's mix plus danish-extraction-v1, capped.
#
# ONE new source and ONE optimizer change against v33. Everything else is held
# at v33's values (bs 128 x ga 1, lr 3e-5 constant_with_warmup/500, 3 epochs,
# max_len 8048, FA2 + flatten packing, compile on / Liger off) so the delta is
# attributable. v33 ran 44,031 steps in 5h01m at train_loss 0.598.
#
# WHY THE CAP. Mix share is set by BYTES, not rows. danish-extraction-v1 is
# 4.2 KB/row -- its ICL prompts embed 1-5 demo passages -- against the mix's
# 0.9 KB average. Uncapped, its 121,197 rows are 507 MB = 21.7% of a v34 mix,
# which would make an unmeasured new capability the second-largest source in
# the corpus, above reason-v1. 60k rows lands it near 12% and keeps the run
# close to v33's wall-clock. Precedent for caution: the v30 sci datasets went
# -0.73pp mean(4) at a far smaller weight (project_v30_sci_datasets_net_loss).
#
# :sft:train IS LOAD-BEARING. The repo's `default` config would also pull
# eval_schema/eval_passage, and training on those turns the new downstream
# metric into a memorisation readout rather than a transfer one.
#
# THE NEW EVAL. --downstream-evals gains `extraction`: exact match on
# eval_schema, capped at 1000 rows (PER_EVAL_CAP). Because extraction's schema
# is proposed per passage, schema and passage are ~1:1 and the two hash
# partitions coincide -- only 6 of eval_schema's 1,801 passages also occur in
# train -- so the metric is unseen text AND unseen field set. v33 scores 0.0%
# on it, as expected for a model that never saw the data; the parser is not
# the reason (gold round-trips 6,226/6,226 = 100%).
#
# 8-BIT ADAM. adamw_bnb_8bit quantizes optimizer STATE only; master weights
# stay fp32 with bf16 autocast, so the constraint in
# feedback_no_bf16_master_weights is untouched. Frees ~3 GB of the ~4.8 GB
# Adam state for a 400M model. Note this makes optimizer state incompatible
# with a v33 fp32-Adam checkpoint -- fine, this is a fresh SFT, not a resume.
#
# FLASH-ATTENTION IS REQUIRED. train_sft_packed.py refuses --flatten-packing
# without it: under SDPA the flattening collator emits no cu_seq_lens and
# packed samples attend across their boundaries. Provision with:
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
# re-resolves and dies with "requirements are unsatisfiable" before training
# starts. --no-sync runs the already-built venv untouched.
uv run --no-sync python -u scripts/train_sft_packed.py \
  --checkpoint jensjepsen/danish-lm-400m-base-ropext8048-v1 \
  --tokenizer jensjepsen/danish-tokenizer \
  --output-dir /root/runs/da_sft_v34_full \
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
    jensjepsen/danish-extraction-v1:sft:train \
  --source-cap danish-extraction-v1=60000 \
  --epochs 3 --batch-size 128 --gradient-accumulation 1 \
  --optim adamw_bnb_8bit \
  --learning-rate 3e-5 --lr-scheduler constant_with_warmup --warmup-steps 500 \
  --max-length 8048 \
  --flatten-packing \
  --torch-compile \
  --save-fraction-of-epoch 0.25 --eval-fraction-of-epoch 0.25 \
  --save-total-limit 3 --top-k-downstream 3 \
  --downstream-evals gsm8k citgen sciq ifeval icl extraction \
  --downstream-n 0 --downstream-batch-size 32 \
  --wandb-project danish-lm-sft \
  --wandb-run-name da_sft_v34_full_mix21_extraction60k_adam8bit_bs128 \
  --wandb-tags sft da v34 full-resft mix21 extraction adam8bit ner icl \
               fa2 torch-compile no-liger epochs-3 h100 \
  "$@"
