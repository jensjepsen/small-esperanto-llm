"""Generate arithmetic-simplification chains that mirror the algebra format:
first line is the FULL expression to compute, subsequent lines progressively
simplify by reducing the leftmost evaluable sub-expression, final line is
the single-value answer.

Example (mirrors `8x - 72 = 16  →  ...  →  x = 11` shape):

    (42 / 3 + 20) * 2
    (14 + 20) * 2
    34 * 2
    68

Each line is the SAME expression after one reduction step. The model sees
the problem at the top and learns to simplify it step by step, exactly the
same shape as the algebra chain.

Construction: pick a starting integer and a sequence of (op, k) pairs.
Build the parenthesized expression incrementally so left-to-right
evaluation always matches the op order. Each reduction step substitutes
the leftmost sub-expression with its value, one at a time.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# wordify_text is language-specific; picked in main() based on --lang.
wordify_text = None  # type: ignore


def _safe(v: int, max_abs: int = 100_000) -> bool:
    return -max_abs <= v <= max_abs


def _build_chain(rng: random.Random, depth_choices=None) -> dict | None:
    """Pick start + ops; return list of (value, op_str, k, new_value).

    op_str is the literal op character. new_value is the running value
    after applying that op to value. The list represents the forward
    construction; rendering walks it both ways:
      - to build the initial expression (innermost first)
      - to produce simplification steps (leftmost reduction)
    """
    depth_choices = depth_choices or [1, 2, 2, 3, 3, 4]
    depth = rng.choice(depth_choices)
    v = rng.randint(2, 50)
    start = v
    ops = []  # list of (op_char, k, new_value)
    for _ in range(depth):
        kind = rng.choices(
            ["add", "sub", "mul", "div"],
            weights=[3, 3, 2, 2],
        )[0]
        if kind == "add":
            k = rng.randint(1, 99)
            new_v = v + k
            if not _safe(new_v): continue
            ops.append(("+", k, new_v))
        elif kind == "sub":
            k = rng.randint(1, 99)
            new_v = v - k
            if not _safe(new_v): continue
            ops.append(("-", k, new_v))
        elif kind == "mul":
            k = rng.randint(2, 15)
            new_v = v * k
            if not _safe(new_v): continue
            ops.append(("*", k, new_v))
        elif kind == "div":
            if v == 0: continue
            divisors = [d for d in range(2, 16) if d <= abs(v) and v % d == 0]
            if not divisors: continue
            k = rng.choice(divisors)
            new_v = v // k
            ops.append(("/", k, new_v))
        v = new_v
    if not ops:
        return None
    return {"start": start, "ops": ops}


def _render(start: int, ops: list) -> str:
    """Build the simplification chain.

    Initial expression is built by progressive left-association:
      after op0:  start op0 k0
      after op1:  (start op0 k0) op1 k1
      after op2:  ((start op0 k0) op1 k1) op2 k2
    so leftmost sub-expression always evaluates to the running value
    after that op, regardless of operator precedence.

    Reduction steps then replace the leftmost parenthesised sub-expression
    with its value, one per line, until the expression collapses to a single
    integer.
    """
    # Build the initial expression as a left-folded paren tree.
    # We represent each level as a string; outermost = full expression.
    levels = [str(start)]  # levels[i] = the leftmost sub-expr after i ops
    for op, k, _ in ops:
        levels.append(f"({levels[-1]}) {op} {k}" if len(ops) > 1 else f"{levels[-1]} {op} {k}")
    # Outermost expression
    if len(ops) == 1:
        # single op: "5 + 3" — no need for paren wrapping
        # levels[1] = "start op k" so just use it
        expr = levels[-1]
    else:
        # For >1 op the inner sub-exprs need parens for left-to-right
        # eval, but the OUTERMOST level shouldn't be wrapped.
        # Rebuild with paren-on-inner-only:
        expr = str(start)
        for i, (op, k, _) in enumerate(ops):
            if i == 0:
                expr = f"{expr} {op} {k}"
            else:
                expr = f"({expr}) {op} {k}"

    lines = [expr]
    # Now reduce step by step. Each step replaces the leftmost sub-expr
    # `(...)` (or the bare `start op k` if there's no paren yet) with its value.
    # ops[0..i-1] have been applied; substitute step shows result of ops[i].
    # We rebuild the expression after each step.
    for i in range(len(ops)):
        # value after applying ops[0..i] is ops[i][2]
        v_so_far = ops[i][2]
        # remaining ops to be applied: ops[i+1:]
        remaining = ops[i+1:]
        if not remaining:
            new_expr = str(v_so_far)
        elif len(remaining) == 1:
            op, k, _ = remaining[0]
            new_expr = f"{v_so_far} {op} {k}"
        else:
            new_expr = str(v_so_far)
            for j, (op, k, _) in enumerate(remaining):
                if j == 0:
                    new_expr = f"{new_expr} {op} {k}"
                else:
                    new_expr = f"({new_expr}) {op} {k}"
        lines.append(new_expr)
    return "\n".join(lines)


def gen_one(rng: random.Random) -> dict | None:
    c = _build_chain(rng)
    if c is None: return None
    text = _render(c["start"], c["ops"])
    # Final value after all ops is what the chain reduces to.
    final = c["ops"][-1][2] if c["ops"] else c["start"]
    return {"text": text, "answer": str(final), "_n_ops": len(c["ops"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--word-frac", type=float, default=0.15,
                    help="Per-number probability of replacing a digit "
                         "with its language word form. 0 = pure digits.")
    ap.add_argument("--lang", choices=["eo", "da"], default="eo",
                    help="Language for wordify_text number rendering.")
    args = ap.parse_args()

    global wordify_text
    if args.lang == "eo":
        from esperanto_lm.eo_numbers import wordify_text as _wt
    else:
        from esperanto_lm.da_numbers import wordify_text as _wt
    wordify_text = _wt

    rng = random.Random(args.seed)
    seen = set()
    n = dups = rej = 0
    depth_dist = Counter()
    with args.out.open("w") as f:
        while n < args.n:
            r = gen_one(rng)
            if r is None:
                rej += 1
                continue
            if args.word_frac > 0:
                r["text"] = wordify_text(r["text"], rng, p_word=args.word_frac)
            # Append canonical `#### N` answer marker AFTER wordify so the
            # marker stays as digits even with word_frac > 0.
            r["text"] = r["text"] + "\n#### " + r["answer"]
            text = r["text"]
            if text in seen:
                dups += 1
                if dups > args.n * 5: break
                continue
            seen.add(text)
            depth_dist[r["_n_ops"]] += 1
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1
    print(f"DONE  emitted={n:,}  dups_skipped={dups:,}  rej={rej}")
    print(f"  depth distribution (n_ops -> count): {dict(depth_dist)}")


if __name__ == "__main__":
    main()
