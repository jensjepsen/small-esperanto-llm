"""Push an ICL SFT checkpoint to the Hub, optimizer state included.

Weights-only pushes make a run unresumable, which has cost real training time
before. The whole checkpoint directory goes up: model.safetensors,
optimizer.pt, scheduler.pt, rng_state.pth, trainer_state.json, tokenizer.

Usage:
  python scripts/push_icl_model_hf.py --ckpt /root/runs/da_icl_v3/checkpoint-3024
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO = "jensjepsen/danish-lm-400m-icl-v3"

CARD = """---
license: apache-2.0
language:
- da
base_model: jensjepsen/danish-lm-400m-sft-v31-avg-top3
tags:
- danish
- in-context-learning
- symbol-tuning
- sft
pipeline_tag: text-generation
---

# danish-lm-400m-icl-v3

400M Danish model tuned for **in-context schema and format induction**: given
a few worked examples in a single user turn and no instruction, infer both the
output schema and the output format from the examples and apply them to a new
passage.

## Recipe

- **Base**: [`jensjepsen/danish-lm-400m-sft-v31-avg-top3`](https://huggingface.co/jensjepsen/danish-lm-400m-sft-v31-avg-top3)
- **Data**: [`jensjepsen/danish-icl-json-v3`](https://huggingface.co/datasets/jensjepsen/danish-icl-json-v3) `sft:train`, 33,933 rows, **ICL-only, no ballast**
- **3 epochs**, 3,024 steps, eff_bs 32 (16 x 2), lr 1e-5 constant + 50 warmup,
  `adamw_bnb_8bit`, seq 3072, FA2 varlen (`DataCollatorWithFlattening`)
- 1,208s on one RTX 5090. Tokenizer is the base checkpoint's (16007 tokens,
  chat tokens at 16000-16002).

## Evals

Exact match on the parsed object, greedy, n=400 per split. Base = the v31
checkpoint this was tuned from, measured in the same session.

| split | base | this model |
|---|---|---|
| val (seen schemas + formats, held-out passages) | 0.2% | **77.8%** |
| eval_schema (unseen schemas) | 0.0% | **41.5%** |
| eval_format (unseen formats) | 0.0% | 25.8% |
| eval_both | 0.0% | 15.5% |

Trained formats on unseen schemas are even: numbered 51.7, tsv 43.8, tagged
43.6, kv_colon 41.0, json 38.7, kv_bracket 36.5, kv_arrow 35.5.

**Meaning-free keys are about as easy as real Danish field names** (val 76.7
vs 78.7; eval_schema 40.1 vs 42.9), which is the evidence the mapping is read
from the demonstrations rather than from field-name semantics — the base
scored 0.6% on symbol rows.

## Limits

**Format transfer is delimiter substitution, not structural generalisation.**
On unseen formats, `kv_eq` (`k=v`, one delimiter away from trained formats)
reaches **78.5%**, while `bracket_pair` (`[k]v[/k]`) scores **0.0%** and
`brace_pair` (`{k}v{/k}`) **0.7%** — despite `bracket_pair` being structurally
identical to the trained `tagged` (`<k>v</k>`). The model copies the opening
delimiter ~100% of the time and gets every value right ~60% of the time, but
produces the matched close only ~37%, falling back to the trained `</tag>`
form. **To get a format into the repertoire, train on it.**

**Out-of-distribution tasks work, limited by task ability not format.** On
dane_plus NER (never trained on; 3-shot, no instruction) it parses 100% in
`tagged` and `kv_colon` against the base's 52%/24%, reaching 26-31 entity F1
where the base gets 0-2.4. Residual errors are entity *typing*, not format.

**Trained ICL-only.** The predecessor trained on a single format became rigid
about structured output; multi-format training was the fix. General
benchmarks were roughly flat but drifted down over the run (gsm8k 17.0 to
12.5 at n=200, citgen 28.5 to 24.0) — mix in ballast if that matters for
your use.

## Prompt shape

Exemplars go **inside one user turn**. This model family breaks on multi-turn
few-shot (the v31 card records GSM8K 18.7 to 2.1%).

```
<|user|>Eksempler:

Tekst:
<passage>
Svar: <rendered answer>

Tekst:
<passage>
Svar: <rendered answer>

Tekst:
<target passage>
Svar:<|end|><|assistant|>
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    d = Path(args.ckpt)
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    need = {"model.safetensors", "optimizer.pt", "scheduler.pt",
            "rng_state.pth", "config.json", "tokenizer.json"}
    missing = need - set(files)
    assert not missing, f"checkpoint missing {sorted(missing)}"
    total = sum(p.stat().st_size for p in d.iterdir() if p.is_file())
    print(f"{d}: {len(files)} files, {total/1e9:.2f} GB")
    for f in files:
        print(f"   {f}")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True,
                    private=args.private)
    api.upload_folder(folder_path=str(d), repo_id=args.repo,
                      repo_type="model",
                      commit_message=f"ICL v3 {d.name} (incl. optimizer state)")
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="model",
                    commit_message="model card")
    print(f"-> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
