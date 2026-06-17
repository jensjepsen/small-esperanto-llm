"""Reservoir-sample N pairs from the filtered CCMatrix JSONL."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("mt/data/parallel/ccmatrix_filtered.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    reservoir: list[str] = []
    total = 0
    with args.inp.open() as f:
        for line in f:
            total += 1
            if len(reservoir) < args.n:
                reservoir.append(line)
            else:
                j = rng.randrange(total)
                if j < args.n:
                    reservoir[j] = line
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for line in reservoir:
            f.write(line)
    print(f"Reservoir-sampled {len(reservoir)} from {total} -> {args.out}")


if __name__ == "__main__":
    main()
