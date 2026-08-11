"""Push Danish-translated OpenBookQA to HF Hub.

Single config "main", preserving official train/validation/test splits.
Rows carry both DA and EN fields.

Row schema:
    id             str
    question       str   — DA
    choices        list[dict{label, text}]  — DA
    answerKey      str
    en_question    str
    en_choices     list[dict{label, text}]

Usage:
    uv run python scripts/push_openbookqa_da_hf.py \\
        --input data/openbookqa_da.jsonl \\
        --repo jensjepsen/danish-openbookqa
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict


def load(path: Path) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("reject_reason"):
                continue
            if not r.get("da_question") or not r.get("da_choices"):
                continue
            row = {
                "id": r["id"],
                "question": r["da_question"],
                "choices": r["da_choices"],
                "answerKey": r["answerKey"],
                "en_question": r["en_question"],
                "en_choices": r["en_choices"],
            }
            buckets[r["split"]].append(row)
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/openbookqa_da.jsonl"))
    ap.add_argument("--repo", default="jensjepsen/danish-openbookqa")
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HF_HUB_TOKEN")
    if not token:
        tp = Path.home() / ".cache/huggingface/token"
        if tp.exists():
            token = tp.read_text().strip()
    if not token:
        print("no HF token", file=sys.stderr)
        sys.exit(2)

    buckets = load(args.input)
    for split, rows in sorted(buckets.items()):
        print(f"  {split:<12} {len(rows):>5}")

    dd = DatasetDict({
        split: Dataset.from_list(buckets[split])
        for split in ("train", "validation", "test")
        if split in buckets
    })
    print(f"\npushing to {args.repo}…", flush=True)
    dd.push_to_hub(args.repo, config_name="main", token=token)

    print(f"\ndone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
