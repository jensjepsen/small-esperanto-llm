"""Push the Danish wp_compose v2 dataset to HF Hub.

Two configs (matches jensjepsen/esperanto-word-problems-v4 layout):
  - default: full schema {question, answer, chain_lines, final,
                          recipe, n_steps, direction}
  - sft:     messages format for direct SFT consumption

Usage:
  HF_HUB_TOKEN=$(cat ~/.cache/huggingface/token) \\
  uv run python scripts/push_wp_compose_da_hf.py \\
    --input /mnt/data2/word_problems_da_v2/all.jsonl \\
    --repo jensjepsen/danish-word-problems-v2
"""
import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.open():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def to_messages(rows: list[dict]) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["answer"]},
            ],
            "final": r["final"],
            "recipe": r["recipe"],
            "n_steps": r["n_steps"],
            "direction": r.get("direction", "forward"),
        }
        for r in rows
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--token", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    token = args.token or os.getenv("HF_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        token_path = Path.home() / ".cache/huggingface/token"
        if token_path.exists():
            token = token_path.read_text().strip()
    if not token:
        print("No HF token found.", file=sys.stderr)
        sys.exit(2)

    print(f"loading {args.input}…", flush=True)
    rows = load_rows(args.input)
    print(f"  {len(rows):,} rows", flush=True)

    print("building datasets…", flush=True)
    full_ds = Dataset.from_list(rows).shuffle(seed=args.seed)
    sft_ds  = Dataset.from_list(to_messages(rows)).shuffle(seed=args.seed)

    print(f"pushing to {args.repo} (default / sft)…", flush=True)
    full_ds.push_to_hub(args.repo, config_name="default",
                        token=token, private=args.private)
    sft_ds.push_to_hub(args.repo, config_name="sft",
                       token=token, private=args.private)
    print("done")


if __name__ == "__main__":
    main()
