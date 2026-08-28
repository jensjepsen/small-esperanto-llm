"""Push the JSON in-context-learning rows to the Hub.

Two configs, matching the convention used by danish-instruction-following-v4:
  default  messages + meta (schema, symbols, shots, task_type, domain)
  sft      messages only, ready for train_sft.py

Splits are train / eval, where eval holds SCHEMAS absent from train -- a row
split would measure memorisation of the 134 schemas in the source rather than
generalisation to an unseen one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi

REPO = "jensjepsen/danish-icl-json-v1"

# push_to_hub writes a `configs:` block into README.md naming the parquet
# paths for each config. Uploading a card afterwards replaces the whole file,
# so that block has to be reproduced here or the non-default config silently
# disappears from the Hub viewer and from load_dataset.
CARD = """---
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
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: eval
    path: data/eval-*
  - split: val
    path: data/val-*
- config_name: sft
  data_files:
  - split: train
    path: sft/train-*
  - split: eval
    path: sft/eval-*
  - split: val
    path: sft/val-*
---

# danish-icl-json-v1

In-context-learning rows derived from `jensjepsen/danish-json-grpo-v1`. Each
row packs 1-5 worked examples sharing a JSON schema into a single user turn,
followed by a held-out passage; the assistant turn is the answer for that
passage. No instruction is included, so the schema and the output format have
to be inferred from the examples. In roughly half the rows the field names are
replaced by meaning-free symbols (`alfa`/`beta`/..., `kat_a`/..., `f1`/...,
`foo`/`bar`/...), applied consistently within a row. The `eval` split uses
schemas that do not appear in `train`; the `val` split reuses train schemas but
is built from source passages that occur in neither other split. All values are
rendered from the source
dataset's `gold_values` rather than generated, and rows are filtered so that
every value is recoverable from its own passage and every key, boolean value
and notation appearing in the answer is demonstrated by at least one example.
Built by `scripts/gen_icl_json.py`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("scratch/icl_json_10k"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--val", type=Path, default=None,
                    help="val.jsonl -- rows whose source passages appear in "
                         "neither train nor eval")
    ap.add_argument("--only", default=None,
                    help="push just this split (train|eval|val)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    train = [json.loads(l) for l in (args.data_dir / "train.jsonl").open()]
    ev = [json.loads(l) for l in
          (args.data_dir / "eval_heldout_schema.jsonl").open()]
    vp = args.val or (args.data_dir / "val.jsonl")
    val = [json.loads(l) for l in vp.open()] if vp.exists() else []
    assert train and ev, "empty split"
    sch_t = {r["meta"]["schema"] for r in train}
    sch_e = {r["meta"]["schema"] for r in ev}
    assert not (sch_t & sch_e), f"schema leak: {sorted(sch_t & sch_e)[:3]}"
    print(f"train={len(train):,} ({len(sch_t)} schemas)  "
          f"eval={len(ev):,} ({len(sch_e)} schemas)  overlap=0")
    if val:
        sch_v = {r["meta"]["schema"] for r in val}
        assert not (sch_v & sch_e), "val leaks an eval schema"
        import hashlib as _h
        key = lambda R: {_h.md5((x["messages"][0]["content"]
                                 + x["messages"][1]["content"]).encode()).hexdigest()
                         for x in R}
        assert not (key(val) & (key(train) | key(ev))), "val duplicates a row"
        print(f"val={len(val):,} ({len(sch_v)} schemas, "
              f"{len(sch_v & sch_t)} shared with train, 0 with eval)")

    def flat(rows):
        return [{"messages": r["messages"], **r["meta"]} for r in rows]

    def sft(rows):
        return [{"messages": r["messages"]} for r in rows]

    if args.dry_run:
        print(json.dumps(flat(train)[0], ensure_ascii=False)[:400])
        return

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    splits = [("train", train), ("eval", ev)] + ([("val", val)] if val else [])
    if args.only:
        splits = [x for x in splits if x[0] == args.only]
    for cfg, fn in (("default", flat), ("sft", sft)):
        for split, rows in splits:
            Dataset.from_list(fn(rows)).push_to_hub(
                args.repo, config_name=cfg, split=split,
                commit_message=f"{cfg}/{split} ({len(rows)} rows)")
            print(f"  pushed {cfg}/{split} ({len(rows)} rows)", flush=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset",
                    commit_message="dataset card")
    print(f"-> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
