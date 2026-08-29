"""Generate simple arithmetic Q/A rows.

Output JSONL with `{text: "EXPR = RESULT"}` format — mirrors algebra
pretrain shape so it can be easily converted to SFT messages
(first line → user prompt, second line → assistant). Each row is a
single line, so the SFT conversion is trivial: prompt = `EXPR`,
completion = `RESULT`.

Distribution (default ratios):
  - add 1-digit / 2-digit / 3-digit
  - sub 1-digit / 2-digit / 3-digit (always non-negative? configurable)
  - mul 1-digit / 1×2-digit / 2-digit
  - div with whole-number result (1-digit divisor, 2-digit divisor)
  - negative-result variants for add/sub
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path


def _gen(rng: random.Random) -> dict:
    """Pick a random arithmetic problem with a whole or signed-int result."""
    kind = rng.choices(
        ["add1", "add2", "add3", "sub1", "sub2", "sub3",
         "mul1", "mul12", "mul2", "div1", "div2", "neg_add", "neg_sub"],
        weights=[3, 4, 3, 3, 4, 3, 3, 4, 3, 3, 3, 2, 2],
    )[0]
    if kind == "add1":
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        expr, res = f"{a} + {b}", a + b
    elif kind == "add2":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        expr, res = f"{a} + {b}", a + b
    elif kind == "add3":
        a, b = rng.randint(100, 999), rng.randint(10, 999)
        expr, res = f"{a} + {b}", a + b
    elif kind == "sub1":
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        a, b = max(a, b), min(a, b)  # keep result non-negative
        expr, res = f"{a} - {b}", a - b
    elif kind == "sub2":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        a, b = max(a, b), min(a, b)
        expr, res = f"{a} - {b}", a - b
    elif kind == "sub3":
        a, b = rng.randint(100, 999), rng.randint(10, 999)
        a, b = max(a, b), min(a, b)
        expr, res = f"{a} - {b}", a - b
    elif kind == "mul1":
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        expr, res = f"{a} * {b}", a * b
    elif kind == "mul12":
        a, b = rng.randint(2, 9), rng.randint(10, 99)
        expr, res = f"{a} * {b}", a * b
    elif kind == "mul2":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        expr, res = f"{a} * {b}", a * b
    elif kind == "div1":
        # 1-digit divisor with whole-number quotient
        d = rng.randint(2, 9)
        q = rng.randint(2, 12)
        n = d * q
        expr, res = f"{n} / {d}", q
    elif kind == "div2":
        # 2-digit divisor with whole-number quotient
        d = rng.randint(11, 25)
        q = rng.randint(2, 12)
        n = d * q
        expr, res = f"{n} / {d}", q
    elif kind == "neg_add":
        # Add producing a negative result, e.g. -5 + 3 = -2
        a, b = rng.randint(1, 30), rng.randint(1, 30)
        a, b = max(a, b), min(a, b)
        expr, res = f"-{a} + {b}", -a + b
    elif kind == "neg_sub":
        # a - b with b > a so result is negative
        a, b = rng.randint(1, 30), rng.randint(1, 30)
        a, b = min(a, b), max(a, b)
        expr, res = f"{a} - {b}", a - b
    else:
        raise ValueError(kind)
    return {"text": f"{expr} = {res}", "_kind": kind}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kinds = Counter()
    seen = set()
    n = 0
    dups = 0
    with args.out.open("w") as f:
        while n < args.n:
            r = _gen(rng)
            text = r["text"]
            # Dedup — finite single-digit ops would otherwise repeat heavily
            if text in seen:
                dups += 1
                if dups > args.n * 5:
                    # Hit saturation in single-digit space; accept duplicates
                    break
                continue
            seen.add(text)
            kinds[r["_kind"]] += 1
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1
    print(f"DONE  emitted={n:,}  dups_skipped={dups:,}  kinds={dict(kinds)}")


if __name__ == "__main__":
    main()
