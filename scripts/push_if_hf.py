"""Push /mnt/data2/if/danish_if_v2.jsonl → HuggingFace as two configs
with train + eval splits:

  jensjepsen/danish-instruction-following-v2
    - default: {train, eval} — full rows (messages + source + constraints + params + attempts)
    - sft    : {train, eval} — messages only (drop-in for train_sft_packed.py)

Usage:
    /home/jepsen/src/espllm/.venv/bin/python scripts/push_if_hf.py \\
      --jsonl /mnt/data2/if/danish_if_v2.jsonl \\
      --repo jensjepsen/danish-instruction-following-v2 \\
      --eval-n 1000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


def load_rows(jsonl_path: Path) -> list[dict]:
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            r.pop("_task_hash", None)
            r["params"] = json.dumps(r["params"], ensure_ascii=False)
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/mnt/data2/if/danish_if_v2.jsonl")
    ap.add_argument("--repo", default="jensjepsen/danish-instruction-following-v2")
    ap.add_argument("--eval-n", type=int, default=1000,
                    help="Rows to hold out as eval split (shuffled first).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    tok_path = Path.home() / ".cache/huggingface/token"
    token = tok_path.read_text().strip() if tok_path.exists() else None

    rows = load_rows(Path(args.jsonl))
    print(f"loaded {len(rows):,} rows from {args.jsonl}", flush=True)

    # Deterministic shuffle-then-split
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    eval_rows = rows[: args.eval_n]
    train_rows = rows[args.eval_n :]
    print(f"split: train={len(train_rows):,}  eval={len(eval_rows):,}")

    # default config: full rows, train + eval
    default_dd = DatasetDict({
        "train": Dataset.from_list(train_rows),
        "eval":  Dataset.from_list(eval_rows),
    })

    # sft config: messages only
    sft_dd = DatasetDict({
        "train": Dataset.from_list([{"messages": r["messages"]} for r in train_rows]),
        "eval":  Dataset.from_list([{"messages": r["messages"]} for r in eval_rows]),
    })

    print(f"\npushing default → {args.repo}", flush=True)
    default_dd.push_to_hub(args.repo, config_name="default", token=token,
                            private=args.private)

    print(f"\npushing sft → {args.repo}", flush=True)
    sft_dd.push_to_hub(args.repo, config_name="sft", token=token,
                       private=args.private)

    print(f"\ndone → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
