"""Push Danish text-to-question SFT dataset to HF Hub."""
import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("/mnt/data2/da_tq/tq.jsonl"))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = args.token or os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        p = Path.home() / ".cache/huggingface/token"
        if p.exists(): token = p.read_text().strip()
    if not token:
        print("no HF token", file=sys.stderr); sys.exit(2)

    rows = [json.loads(l) for l in args.input.open() if l.strip()]
    print(f"loaded {len(rows):,} rows", flush=True)

    Dataset.from_list(rows).shuffle(seed=42).push_to_hub(
        args.repo, config_name="sft", token=token, private=args.private)
    print(f"done → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
