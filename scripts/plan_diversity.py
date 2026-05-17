"""Plan-diversity audit over bench samples.

Reads the JSONL sample file produced by `bench_samplers.py --samples-out`
and reports:

  - count of fired plans
  - unique full plans (verb + role bindings)
  - duplicates at full-plan level
  - unique verb skeletons (verb sequence only)
  - top-N most-repeated skeletons

Skeleton repeats are the more interesting number: full plans tend to
be unique simply because role bindings vary across scenes, but the
verb sequences cluster around the planner's canonical "shortest plan
for this drive shape" template. High skeleton variety means the
sampler is exploring goal shapes; high repeat count for any one
skeleton means a drive shape is funneling everything through the
same chain.

Usage:
    python3 scripts/plan_diversity.py [samples_file]

Default file: runs/bench_samples.jsonl
"""
import collections
import json
import sys


def _to_hashable(v):
    if isinstance(v, list):
        return tuple(v)
    return v


def _canon(plan):
    return tuple(
        (step["verb"],
         tuple(sorted((k, _to_hashable(v))
                      for k, v in step["roles"].items())))
        for step in plan)


def _skeleton(plan):
    return tuple(step["verb"] for step in plan)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/bench_samples.jsonl"
    plans = []
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            if s.get("kind") == "fired":
                plans.append(s.get("plan", []))

    full = collections.Counter(_canon(p) for p in plans)
    skel = collections.Counter(_skeleton(p) for p in plans)

    print(f"fired plans:          {len(plans)}")
    print(f"unique full plans:    {len(full)}")
    print(f"duplicate full plans: "
          f"{sum(1 for v in full.values() if v > 1)}")
    print(f"unique skeletons:     {len(skel)}")
    print(f"plans using a repeated skeleton: "
          f"{sum(n for n in skel.values() if n > 1)}")
    print()
    print("top 10 skeletons (n × verb1 → verb2 → ...):")
    for s, n in skel.most_common(10):
        if not s:
            continue
        print(f"  {n:3}×  {' → '.join(s)}")


if __name__ == "__main__":
    main()
