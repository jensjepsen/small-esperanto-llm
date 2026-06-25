"""Push procedural algebra pretrain JSONL to HF Hub.

Reads {text: ...} JSONL and pushes as a `train` split. Drops metadata
fields if present (pretrain loaders consume `text` only).
"""
import argparse
import json
import os
from pathlib import Path

from datasets import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--repo", required=True,
                    help="HF repo id, e.g. jensjepsen/esperanto-algebra-pretrain")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--no-private", action="store_false", dest="private")
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN") or (
        Path.home() / ".cache/huggingface/token").read_text().strip()

    rows = []
    with args.input.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append({"text": r["text"]})
    print(f"loaded {len(rows):,} rows from {args.input}", flush=True)

    ds = Dataset.from_list(rows)
    print(f"pushing to {args.repo} (private={args.private})…", flush=True)
    ds.push_to_hub(args.repo, token=token, private=args.private)
    print(f"done → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
