"""Push /mnt/data2/wp_reword_v1.jsonl → HuggingFace as two configs:

  jensjepsen/danish-word-problems-reworded-v1
    - default: {orig_idx, q_orig, q_new, a, status, attempts}
    - sft    : {messages: [{role, content}]} — drop-in for train_sft_packed.py

Filter: only status ∈ {ok, ok_retry} — rejects and api_fails excluded.

Usage:
    uv run python scripts/push_wp_reword_hf.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict


def load_rows(jsonl_path: Path) -> list[dict]:
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") not in ("ok", "ok_retry"):
                continue
            if not r.get("q_new"):
                continue
            rows.append({
                "orig_idx": r["orig_idx"],
                "q_orig":   r["q_orig"],
                "q_new":    r["q_new"],
                "a":        r["a"],
                "status":   r["status"],
                "attempts": r.get("attempts", 1),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/mnt/data2/wp_reword_v1.jsonl")
    ap.add_argument("--repo",  default="jensjepsen/danish-word-problems-reworded-v1")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    tok_path = Path.home() / ".cache/huggingface/token"
    token = tok_path.read_text().strip() if tok_path.exists() else None

    rows = load_rows(Path(args.jsonl))
    print(f"loaded {len(rows):,} clean-status rows from {args.jsonl}", flush=True)

    default_ds = Dataset.from_list(rows)
    sft_ds = Dataset.from_list([
        {"messages": [
            {"role": "user",      "content": r["q_new"]},
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
