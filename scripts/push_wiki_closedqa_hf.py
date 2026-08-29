"""Push /mnt/data2/wiki_closedqa_v4/rows.jsonl → HuggingFace.

  jensjepsen/danish-wiki-closedqa-v1
    - default: {orig_pageid, orig_title, tier, q, a}
    - sft    : {messages: [user q, assistant a]} — drop-in for train_sft_packed.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/mnt/data2/wiki_closedqa_v4/rows.jsonl")
    ap.add_argument("--repo",  default="jensjepsen/danish-wiki-closedqa-v1")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    tok_path = Path.home() / ".cache/huggingface/token"
    token = tok_path.read_text().strip() if tok_path.exists() else None

    rows = []
    with open(args.jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r.get("q") and r.get("a"):
                rows.append(r)
    print(f"loaded {len(rows):,} rows from {args.jsonl}", flush=True)

    default_ds = Dataset.from_list(rows)
    sft_ds = Dataset.from_list([
        {"messages": [
            {"role": "user",      "content": r["q"]},
            {"role": "assistant", "content": r["a"]},
        ]}
        for r in rows
    ])

    print(f"\npushing default → {args.repo}", flush=True)
    default_ds.push_to_hub(args.repo, config_name="default", token=token,
                           private=args.private)

    print(f"\npushing sft → {args.repo}", flush=True)
    sft_ds.push_to_hub(args.repo, config_name="sft", token=token,
                       private=args.private)

    print(f"\ndone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
