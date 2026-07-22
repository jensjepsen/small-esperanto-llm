"""Push Danish alpaca-cleaned to HF Hub.

Two configs:
  - default: full parallel schema (both EN source + DA translation per row)
  - sft:     Danish-only messages format for direct SFT consumption
"""
import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset


def load_rows(path: Path):
    return [r for r in map(json.loads, path.open()) if r.get("da") is not None]


def to_default(rows):
    return [{
        "id": r["id"],
        "en_instruction": r["en"]["instruction"],
        "en_input":       r["en"]["input"],
        "en_output":      r["en"]["output"],
        "da_instruction": r["da"]["instruction"],
        "da_input":       r["da"]["input"],
        "da_output":      r["da"]["output"],
    } for r in rows]


def to_sft(rows):
    """user = instruction (+ input if present), assistant = output."""
    out = []
    for r in rows:
        da = r["da"]
        user = da["instruction"]
        if da["input"].strip():
            user = f"{da['instruction']}\n\n{da['input']}"
        out.append({
            "messages": [
                {"role": "user",      "content": user},
                {"role": "assistant", "content": da["output"]},
            ],
        })
    return out


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
        if p.exists():
            token = p.read_text().strip()
    if not token:
        print("No HF token found.", file=sys.stderr); sys.exit(2)

    print(f"loading {args.input}…", flush=True)
    rows = load_rows(args.input)
    print(f"  {len(rows):,} rows", flush=True)

    if "default" not in args.skip_configs:
        print("pushing default config…", flush=True)
        Dataset.from_list(to_default(rows)).shuffle(seed=args.seed).push_to_hub(
            args.repo, config_name="default", token=token, private=args.private)

    if "sft" not in args.skip_configs:
        print("pushing sft config…", flush=True)
        Dataset.from_list(to_sft(rows)).shuffle(seed=args.seed).push_to_hub(
            args.repo, config_name="sft", token=token, private=args.private)

    print(f"\ndone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
