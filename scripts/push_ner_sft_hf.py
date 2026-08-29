"""Push the DANSK-derived NER SFT rows to the Hub.

Two configs:
  default  messages + meta (mode, format, symbols, shots, types, domain, ...)
  sft      messages only, ready for train_sft_packed.py

push_to_hub writes a `configs:` block into README.md naming the parquet path
for every split; uploading a card afterwards replaces the whole file, so that
block is reproduced here or the non-default config and any unlisted split
silently vanish from load_dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi

REPO = "jensjepsen/danish-ner-sft-v1"
SPLITS = ["train", "val", "eval", "eval_format"]


def _cfg(cfg: str, base: str) -> str:
    lines = [f"- config_name: {cfg}", "  data_files:"]
    for sp in SPLITS:
        lines += [f"  - split: {sp}", f"    path: {base}/{sp}-*"]
    return "\n".join(lines)


CARD = f"""---
language:
- da
license: apache-2.0
task_categories:
- token-classification
- text-generation
tags:
- danish
- ner
- in-context-learning
- symbol-tuning
configs:
{_cfg("default", "data")}
{_cfg("sft", "sft")}
---

# danish-ner-sft-v1

Danish named-entity SFT rows derived from [`chcaa/dansk-ner`](https://huggingface.co/datasets/chcaa/dansk-ner)
(DANSK), which annotates Danish Gigaword text with the 18 OntoNotes entity
types using the same scheme in every split. Each row asks for a random subset
of entity types (1-6, including types absent from the passage so the empty
marker is exercised) and renders the answer in one of fourteen output formats.

Ten are **key-value**, listing the extracted entities: JSON, `key: value`,
`key=value`, `[key] value`, `value -> key`, numbered, TSV, `<key>value</key>`,
`[key]value[/key]` and `{{key}}value{{/key}}`. Four are **span-wrap**, reproducing the passage with the
entities tagged in place — `<person>Anna</person> bor i <sted>Aarhus</sted>.` —
varying the delimiter pair across `<k>…</k>`, `[k]…[/k]`, `(k)…(/k)` and
`{{k}}…{{/k}}`. Span-wrap answers are verified to strip back to the source
passage exactly, so a faithful answer cannot hallucinate. In roughly 40% of
rows the type names are replaced by meaning-free symbols
(`alfa`/`kat_a`/`f1`/`foo`), applied consistently within a row.

Three prompt modes: **icl** (demonstrations only, no instruction — the
requested type set and the output format must both be induced), **instruction**
(no demonstrations; the instruction names the types and spells out the format),
and **both**.

Splits vary one axis at a time, and every eval split is built from DANSK's
**test** text so it is unseen as well as held out. `val` comes from DANSK dev
and is intended for model selection. All entity spans are character slices of
their own passage, so every answer is verbatim by construction.

| split | source | formats | types |
|---|---|---|---|
| `train` | DANSK train | 11 seen | all 18 |
| `val` | DANSK dev | seen | all 18 |
| `eval` | DANSK test | seen | all 18 |
| `eval_format` | DANSK test | **unseen** (`kv_eq`, `bracket_pair`, `spans_brace`) | all 18 |

`spans_brace` is held out while three sibling span-wrap formats are trained,
so `eval_format` asks whether span-wrap transfers *within its own family*.
Entity types are not held out: DANSK ships dev and test, so unseen text is the
generalisation axis its owners intended.

DANSK's test split is never trained or selected on. EuroEval's `dansk-mini`
benchmark samples each of its splits from the corresponding source split, so
training on DANSK train is ordinary train/test protocol.

Rows with a URL, under 5 tokens, or majority non-alphabetic characters are
dropped — there is no context to extract from. Filtering by domain was
considered and rejected: `Legal` reads as clause fragments but is well
annotated, and `Web` is genuinely mixed rather than uniformly unusable.

Built by `scripts/gen_ner_sft.py`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("scratch/ner_sft"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = {}
    for sp in SPLITS:
        p = args.data_dir / f"{sp}.jsonl"
        assert p.exists(), f"missing {p}"
        data[sp] = [json.loads(l) for l in p.open() if l.strip()]

    def types(sp):
        return {t for r in data[sp] for t in r["meta"]["types"].split("|")}

    def fmts(sp):
        return {r["meta"]["format"] for r in data[sp]}

    # assert the disjointness the split claims, before uploading: an eval
    # split that silently overlaps reports generalisation that never happened
    assert not (fmts("eval_format") & fmts("train")), "eval_format shares a format"

    # and the property that makes every eval honest: unseen text
    BL = re.compile(r"Tekst:\n(.*?)\nSvar:", re.S)

    def passages(sp):
        return {x.strip() for r in data[sp]
                for x in BL.findall(r["messages"][0]["content"])}

    tr = passages("train")
    for sp in SPLITS[1:]:
        ov = len(passages(sp) & tr)
        assert ov == 0, f"{sp} shares {ov} passages with train"

    key = lambda R: {hashlib.md5((x["messages"][0]["content"]
                                  + x["messages"][1]["content"]).encode()
                                 ).hexdigest() for x in R}
    ktr = key(data["train"])
    for sp in SPLITS[1:]:
        assert not (key(data[sp]) & ktr), f"{sp} duplicates a train row"

    for sp in SPLITS:
        m = Counter(r["meta"]["mode"] for r in data[sp])
        print(f"  {sp:<12} n={len(data[sp]):<6} types={len(types(sp)):<3} "
              f"formats={len(fmts(sp))}  modes={dict(m)}")

    if args.dry_run:
        print("\ndry run: disjointness + unseen-text assertions passed")
        return

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    for cfg, fn in (("default", lambda R: [{"messages": r["messages"],
                                            **r["meta"]} for r in R]),
                    ("sft", lambda R: [{"messages": r["messages"]} for r in R])):
        for sp in SPLITS:
            Dataset.from_list(fn(data[sp])).push_to_hub(
                args.repo, config_name=cfg, split=sp,
                commit_message=f"{cfg}/{sp} ({len(data[sp])} rows)")
            print(f"  pushed {cfg}/{sp} ({len(data[sp])} rows)", flush=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset",
                    commit_message="dataset card")
    print(f"-> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
