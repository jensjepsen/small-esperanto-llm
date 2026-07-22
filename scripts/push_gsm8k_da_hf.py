"""Push Danish GSM8K to HF Hub.

Two configs, each with train + test splits:
  - default: full parallel schema (EN + DA per row)
  - sft:     DA messages format for direct SFT consumption
"""
import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict


def load_rows(path: Path):
    return [r for r in map(json.loads, path.open()) if r.get("da") is not None]


def to_default(rows):
    return [{
        "id": r["id"],
        "split": r["split"],
        "idx": r["idx"],
        "en_question": r["en"]["question"],
        "en_answer":   r["en"]["answer"],
        "da_question": r["da"]["question"],
        "da_answer":   r["da"]["answer"],
    } for r in rows]


def to_sft(rows):
    return [{
        "messages": [
            {"role": "user",      "content": r["da"]["question"]},
            {"role": "assistant", "content": r["da"]["answer"]},
        ],
    } for r in rows]


def split_dd(rows, transform):
    train = [r for r in rows if r["split"] == "train"]
    test  = [r for r in rows if r["split"] == "test"]
    return DatasetDict({
        "train": Dataset.from_list(transform(train)),
        "test":  Dataset.from_list(transform(test)),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-configs", nargs="*", default=[])
    args = ap.parse_args()

    token = args.token or os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        p = Path.home() / ".cache/huggingface/token"
        if p.exists(): token = p.read_text().strip()
    if not token:
        print("No HF token found.", file=sys.stderr); sys.exit(2)

    print(f"loading {args.input}…", flush=True)
    rows = load_rows(args.input)
    train_n = sum(1 for r in rows if r["split"] == "train")
    test_n  = sum(1 for r in rows if r["split"] == "test")
    print(f"  {len(rows):,} rows (train={train_n}, test={test_n})", flush=True)

    if "default" not in args.skip_configs:
        print("pushing default config…", flush=True)
        split_dd(rows, to_default).push_to_hub(
            args.repo, config_name="default", token=token, private=args.private)

    if "sft" not in args.skip_configs:
        print("pushing sft config…", flush=True)
        split_dd(rows, to_sft).push_to_hub(
            args.repo, config_name="sft", token=token, private=args.private)

    print(f"\ndone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
