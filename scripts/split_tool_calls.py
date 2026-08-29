"""Split a raw tool-call JSONL into train + eval splits, stratified by
difficulty bucket. Deterministic (seed-controlled) so re-running produces
the same split.

Multi_chain rows are ALWAYS held in the train split (too few — 23 total in
v1 — to sacrifice any). All other difficulties are sampled proportionally
to hit --eval-size total.

Usage:
    python scripts/split_tool_calls.py \\
        --in  data/tool_calls/v1.jsonl \\
        --out-train data/tool_calls/v1_train.jsonl \\
        --out-eval  data/tool_calls/v1_eval.jsonl \\
        --eval-size 500 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


ALWAYS_TRAIN_ONLY = {"multi_chain"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-eval", required=True)
    ap.add_argument("--eval-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(line) for line in Path(args.inp).read_text().splitlines()
            if line.strip()]
    print(f"loaded {len(rows)} rows from {args.inp}", flush=True)

    # Bucket by difficulty.
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_diff[r.get("difficulty", "unknown")].append(r)

    print("input distribution:")
    for d, group in sorted(by_diff.items(), key=lambda kv: -len(kv[1])):
        print(f"  {d:12s}  {len(group):6d}")

    # Compute per-bucket eval counts proportional to eligible buckets.
    eligible = {d: g for d, g in by_diff.items() if d not in ALWAYS_TRAIN_ONLY}
    eligible_total = sum(len(g) for g in eligible.values())
    per_bucket_eval: dict[str, int] = {}
    for d, g in eligible.items():
        # Proportional; round to nearest, cap at group size.
        n = round(args.eval_size * len(g) / eligible_total)
        per_bucket_eval[d] = min(n, len(g))

    # Sample eval + emit train remainder.
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    for d, g in by_diff.items():
        if d in ALWAYS_TRAIN_ONLY or d not in per_bucket_eval:
            train_rows.extend(g)
            continue
        shuffled = list(g)
        rng.shuffle(shuffled)
        n_eval = per_bucket_eval[d]
        eval_rows.extend(shuffled[:n_eval])
        train_rows.extend(shuffled[n_eval:])

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_train, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out_eval, "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(train_rows)} train → {args.out_train}")
    print(f"wrote {len(eval_rows)} eval  → {args.out_eval}")
    print("\neval distribution:")
    eval_dist: dict[str, int] = defaultdict(int)
    for r in eval_rows:
        eval_dist[r["difficulty"]] += 1
    for d, n in sorted(eval_dist.items(), key=lambda kv: -kv[1]):
        print(f"  {d:12s}  {n}")


if __name__ == "__main__":
    main()
