"""Bulk generator: runs each Danish wp_compose_da recipe (forward + reverse)
plus mixture as a peer recipe, dedups by (question_prefix, answer), writes
one JSONL file.

Usage:
    uv run python scripts/gen_wp_compose_da_bulk.py \\
        --per-recipe 10000 \\
        --out /mnt/data2/word_problems_da_v2/all.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wp_compose_da import RECIPES, REVERSABLE_RECIPES
from wp_mixture_da import make_row as make_mixture_row


def _sample_forward(recipe_name: str, rng: random.Random) -> dict | None:
    recipe = RECIPES[recipe_name]
    n_steps = rng.choices([2, 3, 4, 5], weights=[3, 3, 2, 1])[0]
    try:
        p = recipe(rng, n_steps=n_steps)
    except RuntimeError:
        return None
    p["recipe"] = recipe_name
    p["n_steps"] = n_steps
    p["direction"] = "forward"
    return p


def _sample_reverse(recipe_name: str, rng: random.Random) -> dict | None:
    recipe = RECIPES[recipe_name]
    n_steps = rng.choices([2, 3, 4, 5], weights=[3, 3, 2, 1])[0]
    try:
        p = recipe(rng, n_steps=n_steps, reverse=True)
    except RuntimeError:
        return None
    p.setdefault("recipe", f"{recipe_name}_reverse")
    p.setdefault("n_steps", n_steps)
    p.setdefault("direction", "reverse")
    return p


def _emit_batch(sampler, rng, target: int, seen: set,
                 max_attempts_per_row: int = 500) -> list[dict]:
    """Sample from `sampler(rng) -> dict | None` until `target` unique rows.

    Dedup by (question[:60], answer[:80]) — mirrors EO convention.
    """
    out: list[dict] = []
    consecutive_misses = 0
    while len(out) < target:
        row = sampler(rng)
        if row is None:
            consecutive_misses += 1
            if consecutive_misses > max_attempts_per_row:
                break
            continue
        key = (row["question"][:60], row["answer"][:80])
        if key in seen:
            consecutive_misses += 1
            if consecutive_misses > max_attempts_per_row:
                break
            continue
        seen.add(key)
        consecutive_misses = 0
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-recipe", type=int, default=10000,
                    help="target rows per recipe direction (forward + reverse "
                         "counted separately; mixture treated as a peer recipe)")
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/data2/word_problems_da_v2/all.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Uniform recipe list: forward compose recipes, reverse compose recipes,
    # and mixture — all peer entries called via the same batching loop.
    jobs: list[tuple[str, callable, int]] = []
    for recipe_name in sorted(RECIPES):
        seed_off = hash(recipe_name) % 100000
        jobs.append((
            recipe_name,
            (lambda name: lambda r: _sample_forward(name, r))(recipe_name),
            args.seed + seed_off,
        ))
        if recipe_name in REVERSABLE_RECIPES:
            jobs.append((
                f"{recipe_name}_reverse",
                (lambda name: lambda r: _sample_reverse(name, r))(recipe_name),
                args.seed + 500000 + seed_off,
            ))
    jobs.append(("mixture", lambda r: make_mixture_row(r), args.seed + 999999))

    per_recipe_counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    all_rows: list[dict] = []

    t0 = time.time()
    for name, sampler, seed in jobs:
        rng = random.Random(seed)
        rows = _emit_batch(sampler, rng, args.per_recipe, seen)
        per_recipe_counts[name] = len(rows)
        all_rows.extend(rows)
        print(f"[{time.time()-t0:6.1f}s] {name:30s} {len(rows):6d} / {args.per_recipe}",
              flush=True)

    # Shuffle before writing so recipes are interleaved.
    shuf = random.Random(args.seed)
    shuf.shuffle(all_rows)

    print(f"writing {len(all_rows):,} rows → {args.out}", flush=True)
    with args.out.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Final report
    print()
    print("=" * 60)
    print(f"TOTAL: {len(all_rows):,} rows in {time.time()-t0:.1f}s")
    print("=" * 60)
    for r in sorted(per_recipe_counts):
        print(f"  {r:30s} {per_recipe_counts[r]:6d}")
    print(f"  {'(dedup pool size)':30s} {len(seen):6d}")


if __name__ == "__main__":
    main()
