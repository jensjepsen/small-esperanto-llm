"""Push the word-problems dataset to HF Hub.

Two configs:
  - default: full schema (type, question_eo, chain_eo, answer, strategy, params,
             rewrap_idx, is_original)
  - sft: messages format for direct SFT consumption

Usage:
  HF_HUB_TOKEN=$(cat ~/.cache/huggingface/token) \\
  uv run python scripts/push_word_problems_hf.py \\
    --input data/word_problems/all_word_problems.jsonl \\
    --repo jensjepsen/esperanto-word-problems
"""
import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict


def load_full(path: Path):
    rows = []
    for line in path.open():
        try:
            r = json.loads(line)
            # normalize params dict to JSON string (HF schema can't handle heterogeneous dicts)
            if isinstance(r.get("params"), dict):
                r["params"] = json.dumps(r["params"], ensure_ascii=False)
            rows.append(r)
        except json.JSONDecodeError:
            continue
    return rows


def to_messages(rows):
    """Convert to SFT messages format."""
    out = []
    for r in rows:
        out.append({
            "messages": [
                {"role": "user", "content": r["question_eo"]},
                {"role": "assistant", "content": r["chain_eo"]},
            ],
            "type": r.get("type", ""),
            "answer": r.get("answer"),
            "strategy": r.get("strategy", ""),
            "is_original": bool(r.get("is_original", True)),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--repo", required=True, help="e.g. jensjepsen/esperanto-word-problems")
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
        print("No HF token found. Pass --token or set HF_HUB_TOKEN.", file=sys.stderr)
        sys.exit(2)

    print(f"loading {args.input}…", flush=True)
    rows = load_full(args.input)
    print(f"  {len(rows):,} rows", flush=True)

    # full schema config
    full_ds = Dataset.from_list(rows).shuffle(seed=args.seed)
    # SFT messages config
    sft_ds = Dataset.from_list(to_messages(rows)).shuffle(seed=args.seed)

    print(f"pushing default config to {args.repo}…", flush=True)
    full_ds.push_to_hub(args.repo, config_name="default", token=token,
                        private=args.private)

    print(f"pushing sft config to {args.repo}…", flush=True)
    sft_ds.push_to_hub(args.repo, config_name="sft", token=token,
                       private=args.private)

    print(f"\ndone → https://huggingface.co/datasets/{args.repo}")
    print(f"  default: {len(full_ds):,} rows (full schema)")
    print(f"  sft:     {len(sft_ds):,} rows (messages format)")


if __name__ == "__main__":
    main()
