"""Push v33 checkpoints to the Hub: avg-top3 at the root, the three singles
as subfolders.

Optimizer/scheduler/RNG state ships with the single checkpoints so training can
be resumed from them; only the averaged model has no optimizer to carry.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO = "jensjepsen/danish-lm-400m-sft-v33-avg-top3"
RUN = "https://wandb.ai/jepsen/danish-lm-sft/runs/w06z2zvb"

SINGLES = {
    "step-25683": "step-25683-agg-0.300",
    "step-29352": "step-29352-agg-0.294",
    "step-40359": "step-40359-agg-0.294",
}

DATASETS = [
    "danish-metamath-gsm", "danish-algebra-sft-v5-mixed",
    "danish-arith-chain-sft-v1", "danish-wiki-grounded-sft-v3",
    "danish-text-to-question-v2", "danish-sciq", "danish-gsm8k",
    "danish-instruction-following-v4", "danish-wiki-closedqa-v1",
    "danish-word-problems-v2", "danish-wiki-closedqa-stem-v1",
    "danish-wiki-broadqa-stem-v1", "danish-wiki-mc-letters-v1",
    "danish-rc-v1", "danish-reason-v1", "danish-textman-v1",
    "danish-arc", "danish-openbookqa", "danish-ner-sft-v1",
    "danish-icl-schema-format-v3",
]

CARD = f"""---
language:
- da
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: jensjepsen/danish-lm-400m-base-ropext8048-v1
tags:
- danish
- sft
- instruction-following
- in-context-learning
- ner
---

# danish-lm-400m-sft-v33-avg-top3

SFT from `jensjepsen/danish-lm-400m-base-ropext8048-v1`, 3 epochs over
1,878,629 rows from 20 datasets. Root of this repo is the equal-weight average
of the three top checkpoints; each is also included as a subfolder with
optimizer state.

| revision | steps | epoch |
|---|---|---|
| root (`avg-top3`) | mean of the three below | — |
| `step-25683` | 25,683 | 1.75 |
| `step-29352` | 29,352 | 2.00 |
| `step-40359` | 40,359 | 2.75 |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("{REPO}")
t = AutoTokenizer.from_pretrained("{REPO}")
# single checkpoint:
m = AutoModelForCausalLM.from_pretrained("{REPO}", subfolder="step-25683")
```

## Evaluation

Full test splits, chat-format, greedy. Run with `scripts/eval_full_suite.sh`.

| metric | avg-top3 |
|---|---|
| GSM8K (0-shot pass@1) | 18.38 |
| CitGen | 28.80 |
| SciQ (open-Q pass@1) | 14.90 |
| TextMan summary (chrF++) | 44.07 |
| TextMan rewrite (chrF++) | 48.45 |
| CitMC | 46.70 |
| SciQ-MC | 58.90 |
| ARC-Easy | 41.12 |
| ARC-Challenge | 27.30 |
| OpenBookQA | 35.40 |
| PIQA (`raw`, length-norm logp) | 60.00 |
| PIQA (`chat-mc`, letter generation) | 49.00 |
| GPQA-Diamond | 26.77 |
| SciQ (length-norm logp) | 42.80 |
| CitMC (length-norm logp) | 56.39 |
| ARC (length-norm logp) | 24.25 |
| IFEval-DA prompt-strict | 24.7 |
| IFEval-DA prompt-loose | 25.2 |
| IFEval-DA inst-strict | 38.9 |
| IFEval-DA inst-loose | 40.0 |

Both PIQA modes are listed because they differ by 11 points and the
conventions disagree: `raw` is the standard PIQA protocol (score each solution
as a continuation, pick the higher length-normalized log-prob), `chat-mc` asks
for a letter. Neither reproduces the 53.00 on the v31 cards, so the v31 figure
came from a third harness and is not comparable to either.

### ICL schema/format induction

`jensjepsen/danish-icl-schema-format-v3`, exact match on the parsed answer.

| split | held out | exact | key-set |
|---|---|---|---|
| `eval_schema` | schema | 56.7 | 93.0 |
| `eval_format` | format | 85.3 | 96.4 |
| `eval_both` | schema + format | 47.8 | 86.8 |

`eval_format` overlaps `danish-ner-sft-v1`, which trains `brace_pair`
(2,276 rows) and the `[x]...[/x]` delimiter pair via `spans_bracket`. Of its
three held-out formats only `kv_eq` (82.7) is unseen in this mix.

### NER

`jensjepsen/danish-ner-sft-v1`, exact match on the parsed answer and micro-F1
over (type, span) pairs. Both splits are built from DANSK test text. FAITHFUL
is span-wrap only: the answer both strips back to the source passage and
carries at least one well-formed tag pair.

**`eval`** — formats seen in training, n=1500: **exact 58.7, entity-F1 70.0**,
faithful 73% (strip 86, tagged 87, bare 12)

| format | exact | F1 | | format | exact | F1 |
|---|---|---|---|---|---|---|
| `brace_pair` | 69.9 | 77.6 | | `tagged` | 63.4 | 72.2 |
| `tsv` | 66.7 | 74.5 | | `kv_bracket` | 58.9 | 71.7 |
| `numbered` | 64.2 | 72.3 | | `kv_colon` | 58.2 | 73.5 |
| `json` | 63.5 | 73.4 | | `kv_arrow` | 55.1 | 66.4 |
| `spans_angle` | 52.3 | 62.5 | | `spans_bracket` | 50.0 | 65.3 |
| `spans_paren` | 42.6 | 57.3 | | | | |

**`eval_format`** — formats held out of both `danish-ner-sft-v1` and
`danish-icl-schema-format-v3` training splits, n=1500: **exact 40.1,
entity-F1 56.8**

| format | exact | F1 | faithful |
|---|---|---|---|
| `bracket_pair` | 60.5 | 71.3 | — |
| `kv_eq` | 53.1 | 66.4 | — |
| `spans_brace` | 7.5 | 17.2 | 6% (strip 83, tagged 15, bare 70) |

Key-value formats transfer to unseen delimiters; span-wrap does not. Three
span-wrap delimiters are trained (`spans_angle`, `spans_bracket`,
`spans_paren`) and the fourth returns the passage untagged in 70% of rows.

| prompt mode | `eval` exact | `eval_format` exact |
|---|---|---|
| `icl` | 60.4 | 42.9 |
| `both` | 63.1 | 38.0 |
| `instruction` | 48.4 | 36.0 |

### Checkpoint comparison

Full test splits, `scripts/eval_downstream_once.py`, ICL capped at 1000 rows.

| model | GSM8K | CitGen | SciQ | IFEval inst-strict | ICL | mean |
|---|---|---|---|---|---|---|
| `step-25683` | 17.2 | 25.4 | 12.3 | 37.3 | 51.9 | 28.82 |
| `step-29352` | 15.3 | 26.8 | 13.0 | 36.6 | 52.8 | 28.90 |
| `step-40359` | 14.8 | 26.8 | 14.4 | 37.6 | 51.0 | 28.92 |
| **avg-top3** | **18.5** | **29.7** | **15.5** | 36.8 | **55.3** | **31.16** |

## Training

| | |
|---|---|
| base | `jensjepsen/danish-lm-400m-base-ropext8048-v1` |
| tokenizer | `jensjepsen/danish-tokenizer` |
| rows | 1,878,629 train / 98,876 eval |
| epochs | 3 (44,031 steps) |
| batch | 128 x grad-accum 1, max_length 8048 |
| optimizer | `adamw_torch_fused`, fp32 master weights, bf16 autocast |
| LR | 3e-5, `constant_with_warmup`, 500 warmup |
| attention | FlashAttention-2, `DataCollatorWithFlattening` |
| kernels | `torch.compile`, Liger off |
| hardware | 1x H100, 5h01m |
| final train loss | 0.598 |

wandb: [`da_sft_v33_full_mix20_bs128_fa2_compile`]({RUN})

## Datasets

{chr(10).join(f"- `jensjepsen/{d}`" for d in DATASETS)}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-dir", type=Path,
                    default=Path("/root/runs/da_sft_v33_full/best"))
    ap.add_argument("--avg-dir", type=Path,
                    default=Path("/root/runs/v33_avg_top3"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--card-only", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)

    if not args.card_only:
        print(f"uploading avg-top3 -> {args.repo} (root)", flush=True)
        api.upload_folder(folder_path=str(args.avg_dir), repo_id=args.repo,
                          commit_message="avg of top-3 checkpoints")
        for sub, d in SINGLES.items():
            src = args.best_dir / d
            assert src.exists(), f"missing {src}"
            print(f"uploading {d} -> {sub}/", flush=True)
            api.upload_folder(folder_path=str(src), repo_id=args.repo,
                              path_in_repo=sub,
                              commit_message=f"{sub} (with optimizer state)")

    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=args.repo, commit_message="model card")
    print(f"-> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
