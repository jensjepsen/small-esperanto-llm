"""Push Danish-translated AI2 ARC (Challenge + Easy) to HF Hub.

Flat schema, one config per subset (arc_challenge, arc_easy), preserving
official train/validation/test splits. Rows carry both DA and EN fields.

Row schema:
    id             str   — original ARC row id (e.g. Mercury_SC_401336)
    question       str   — DA
    choices        list[dict{label, text}]  — DA
    answerKey      str   — original ARC gold letter (A/B/C/D or 1/2/3/4)
    en_question    str
    en_choices     list[dict{label, text}]

Usage:
    uv run python scripts/push_arc_da_hf.py \\
        --input data/arc_da.jsonl \\
        --repo jensjepsen/danish-arc
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict


def load(path: Path) -> dict[tuple[str, str], list[dict]]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
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
            buckets[(r["source"], r["split"])].append(row)
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/arc_da.jsonl"))
    ap.add_argument("--repo", default="jensjepsen/danish-arc")
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
    for (src, split), rows in sorted(buckets.items()):
        print(f"  {src:<15} {split:<12} {len(rows):>5}")

    for cfg in ("arc_challenge", "arc_easy"):
        dd = DatasetDict({
            split: Dataset.from_list(buckets[(cfg, split)])
            for split in ("train", "validation", "test")
            if (cfg, split) in buckets
        })
        print(f"\npushing {cfg} to {args.repo}…", flush=True)
        dd.push_to_hub(args.repo, config_name=cfg, token=token)

    print(f"\ndone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
