"""Push danish-extraction-v1 to the Hub.

Three configs:
  default  messages + meta
  sft      messages only
  raw      the extractions the rows are rendered from

`raw` ships because it is the expensive half: every gate, format, symbol
scheme, prompt mode, shot count, task type and split ratio is a render-time
decision that can be redone from it for free. Without it, changing any of
those means re-running the extraction.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi

REPO = "jensjepsen/danish-extraction-v1"
SPLITS = ["train", "eval_schema", "eval_passage", "eval_both"]


def _cfg(cfg, base, splits):
    lines = [f"- config_name: {cfg}", "  data_files:"]
    for sp in splits:
        lines += [f"  - split: {sp}", f"    path: {base}/{sp}-*"]
    return "\n".join(lines)


def card(counts, registers, meta):
    rows = "\n".join(f"| `{s}` | {counts[s]:,} |" for s in SPLITS)
    regs = "\n".join(f"| {k} | {v:,} |" for k, v in registers.most_common())
    return f"""---
language:
- da
license: apache-2.0
task_categories:
- text-generation
tags:
- danish
- information-extraction
- in-context-learning
configs:
{_cfg("default", "data", SPLITS)}
{_cfg("sft", "sft", SPLITS)}
- config_name: raw
  data_files:
  - split: train
    path: raw/train-*
---

# danish-extraction-v1

Danish information-extraction rows over real prose, where the schema is
proposed per passage rather than fixed. Built from
[`danish-foundation-models/danish-dynaword`](https://huggingface.co/datasets/danish-foundation-models/danish-dynaword)
by `scripts/gen_extraction_da.py`.

Each source passage got its own field set: an LLM proposed 3-6 fields for that
text without seeing any values, then filled them in a separate turn. Roughly a
quarter of proposed fields come back empty, which are genuine abstention
targets rather than annotation gaps.

| split | rows |
|---|---|
{rows}

## Registers

| register | passages |
|---|---|
{regs}

## Tasks

**extract** — given a passage and field names, produce each field's value.
Prompts carrying an instruction also state the required output format, rendered
from the format itself (`Svar i formatet: felt\tværdi`). An earlier release
named no format in instruction-only prompts, which left one format out of ten
unspecified and the row unanswerable.

**fill** — placement. The passage has gaps, the removed spans are listed
**in shuffled order**, and the model reconstructs the text. Fully determined by
the prompt: the information is all present and the task is to work out where
each piece belongs. Scored on whether the spans land in the right sequence and
the surrounding prose is reproduced, so neither echoing the gapped text nor
echoing the value list earns credit.

An earlier release masked spans and asked the model to recall them, under a
rule that a span occur exactly once so no copy stayed visible. Those two
requirements conflict: "occurs exactly once" means the text holds no evidence
for what was removed, and the masked spans are extraction values — the tokens
least predictable from context. Probed, the model returned the right marker
set, the right count and a clean parse, and scored 0 on every row, supplying a
different plausible item from the same list.

## What varies

| axis | values |
|---|---|
| field subset | 1-4 present fields, plus 0-2 absent ones |
| key naming | real names, or symbols (`alfa`, `kat_a`, `f1`, `foo`) |
| output format | all 10, in train and eval |
| prompt mode | `icl` (demonstrations only) / `instruction` / `both` |
| shots | 1-5, resampled per row |
| instruction | 34 hand-written, across 9 shapes |
| scaffold labels | 20 across 6 slots |
| blank markers | 30, indexed and bare |

Demonstrations carry their own text *and* their own field list, so they teach
the task and the output format rather than a particular schema.

## Gates

Every value is a verbatim span of its passage (up to whitespace), contains no
newline, matches its declared type, and is deduplicated. Every rendered answer
round-trips through the format parser.

## Splits

`eval_schema` and `eval_passage` are 5% hash partitions on schema and on
passage; `eval_both` is their intersection.

## Configs

`default` (messages + meta), `sft` (messages only), and `raw` — the
extractions the rows are rendered from, so formats, modes, gates and splits can
be re-derived without re-extracting.

Generated with `--registers --uniform`, `--rows-per-passage 4`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rendered", type=Path,
                    default=Path("scratch/extraction_rendered"))
    ap.add_argument("--raw", type=Path,
                    default=Path("scratch/extraction_full/raw.jsonl"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = {}
    for sp in SPLITS:
        p = args.rendered / f"{sp}.jsonl"
        assert p.exists(), f"missing {p}"
        data[sp] = [json.loads(l) for l in p.open() if l.strip()]
    raw = [json.loads(l) for l in args.raw.open() if l.strip()]
    counts = {sp: len(v) for sp, v in data.items()}
    registers = Counter(r["meta"].get("register") for r in raw)

    print(f"rendered: {sum(counts.values()):,} rows  {counts}")
    print(f"raw: {len(raw):,} passages  {dict(registers)}")
    tasks = Counter(x["meta"]["task"] for v in data.values() for x in v)
    print(f"tasks: {dict(tasks)}")
    if args.dry_run:
        print("\ndry run")
        return

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    for cfg, fn in (("default", lambda R: R),
                    ("sft", lambda R: [{"messages": x["messages"]} for x in R])):
        for sp in SPLITS:
            Dataset.from_list(fn(data[sp])).push_to_hub(
                args.repo, config_name=cfg, split=sp,
                commit_message=f"{cfg}/{sp} ({len(data[sp])} rows)")
            print(f"  pushed {cfg}/{sp} ({len(data[sp])})", flush=True)
    Dataset.from_list(raw).push_to_hub(
        args.repo, config_name="raw", split="train",
        commit_message=f"raw extractions ({len(raw)} passages)")
    print(f"  pushed raw ({len(raw)})", flush=True)

    api.upload_file(path_or_fileobj=card(counts, registers, None).encode(),
                    path_in_repo="README.md", repo_id=args.repo,
                    repo_type="dataset", commit_message="dataset card")
    print(f"-> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
