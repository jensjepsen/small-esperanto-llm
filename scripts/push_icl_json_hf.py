"""Push the ICL rows to the Hub.

Two configs, matching the convention used by danish-instruction-following-v4:
  default  messages + meta (schema, format, symbols, shots, task_type, domain)
  sft      messages only, ready for train_sft_packed.py

v2's splits are a factorial over the two induction axes. v1 had only the
schema axis, so nothing in it could detect that a model trained on a single
output format answers every OTHER format in that format -- which is what
measurably happened.

  train        seen schemas, seen formats
  val          seen schemas, seen formats, passages reserved before train
  eval_schema  UNSEEN schemas, seen formats
  eval_format  seen schemas, UNSEEN formats
  eval_both    unseen schemas AND unseen formats

push_to_hub writes a `configs:` block into README.md naming the parquet path
for every split; uploading a card afterwards replaces the whole file, so that
block is reproduced here. Omitting it silently drops the non-default config
and any split not listed from load_dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi

REPO = "jensjepsen/danish-icl-json-v2"
SPLITS = ["train", "val", "eval_schema", "eval_format", "eval_both"]


def _cfg_block(cfg: str, base: str) -> str:
    lines = [f"- config_name: {cfg}", "  data_files:"]
    for sp in SPLITS:
        lines += [f"  - split: {sp}", f"    path: {base}/{sp}-*"]
    return "\n".join(lines)


CARD = f"""---
language:
- da
license: apache-2.0
task_categories:
- text-generation
tags:
- in-context-learning
- symbol-tuning
- danish
configs:
{_cfg_block("default", "data")}
{_cfg_block("sft", "sft")}
---

# danish-icl-json-v2

In-context-learning rows derived from `jensjepsen/danish-json-grpo-v1`. Each row
packs 1-5 worked examples into a single user turn, followed by a held-out
passage; the assistant turn is the answer for that passage. No instruction is
included, so both the schema and the output format have to be inferred from the
examples. Two axes vary per row and are held constant within a row: the schema
(134 field-sets) and the output format (8 renderers — JSON, `key: value`,
`key=value`, `[key] value`, `value -> key`, numbered, TSV, and
`<key>value</key>`). In roughly half the rows the field names are replaced by
meaning-free symbols (`alfa`/`kat_a`/`f1`/`foo`), applied consistently within a
row. Splits partition those axes rather than rows: `eval_schema` uses schemas
absent from training, `eval_format` uses formats absent from training,
`eval_both` uses neither, and `val` shares both axes with training but is built
from passages reserved before any training row was generated. All values are
rendered from the source dataset's `gold_values` rather than generated, and rows
are filtered so that every value is recoverable from its own passage and every
key, boolean value, empty marker and notation appearing in the answer is
demonstrated by at least one example. Built by `scripts/gen_icl_json.py`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("scratch/icl_v2"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = {}
    for sp in SPLITS:
        p = args.data_dir / f"{sp}.jsonl"
        assert p.exists(), f"missing {p}"
        data[sp] = [json.loads(l) for l in p.open() if l.strip()]

    def field(sp, k):
        return {r["meta"][k] for r in data[sp]}

    # The disjointness the splits claim is asserted before upload, not
    # assumed: a silently-overlapping eval split reports generalisation that
    # never happened.
    for sp, axis in (("eval_schema", "schema"), ("eval_format", "format"),
                     ("eval_both", "schema"), ("eval_both", "format")):
        got = len(field(sp, axis) & field("train", axis))
        assert got == 0, f"{sp} shares {got} {axis}s with train"
    key = lambda R: {hashlib.md5((x["messages"][0]["content"]
                                  + x["messages"][1]["content"]).encode()
                                 ).hexdigest() for x in R}
    ktr = key(data["train"])
    for sp in SPLITS[1:]:
        assert not (key(data[sp]) & ktr), f"{sp} duplicates a train row"

    for sp in SPLITS:
        print(f"  {sp:<12} n={len(data[sp]):<7} "
              f"schemas={len(field(sp, 'schema')):<4} "
              f"formats={sorted(field(sp, 'format'))}")

    def flat(rows):
        return [{"messages": r["messages"], **r["meta"]} for r in rows]

    def sft(rows):
        return [{"messages": r["messages"]} for r in rows]

    if args.dry_run:
        print("\ndry run: disjointness assertions passed")
        return

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    for cfg, fn in (("default", flat), ("sft", sft)):
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
