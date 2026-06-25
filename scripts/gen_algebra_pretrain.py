"""Procedural sympy-based algebra equations + symbolic solution chains
for pretrain mixing.

Math is correct by construction (sympy does the algebra). No prose.
Chain renders vary by surface form only (vertical, arrow, with-computation,
compact) — not by natural-language phrasing.

Generation strategy:
  - Sample a target solution
  - Compose 1-4 reversible ops on both sides (add/sub/mul/div/negate/
    add-var-term/square). Each `mul k` randomly chooses parenthesized or
    expanded form.
  - Verify final by substituting the sampled solution.
  - Render in one of several styles.

Output JSONL: one record per equation as {"text": <chain>}.

Usage:
    uv run python scripts/gen_algebra_pretrain.py \\
        --n 100000 --out data/algebra_pretrain_100k.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import sympy as sp
from sympy import Eq, Symbol, Integer, Rational, sqrt, simplify, expand, factor, srepr

VARS = ["x", "y", "z", "a", "b", "n", "k", "p", "m", "t"]

# Each "frame" records what op was applied and how to render its undo step.
# eq_after is the sympy Eq after applying the op.
# undo_render(eq_before_undo) -> str describing the inverse op (e.g. "÷ 5")
class Op:
    __slots__ = ("kind", "k", "render_paren", "eq_after")

    def __init__(self, kind, k, render_paren, eq_after):
        self.kind = kind            # 'add','sub','mul','div','neg','addvar','sqr'
        self.k = k                  # operand (Integer/Rational), or var-coef for addvar
        self.render_paren = render_paren  # True = keep parens for mul; False = expand
        self.eq_after = eq_after


def _is_safe_coef(c):
    """Avoid trivial / degenerate ops."""
    if c == 0 or c == 1:
        return False
    return True


def _maybe_sym_int(rng: random.Random, lo: int, hi: int) -> Integer:
    n = 0
    while n == 0:
        n = rng.randint(lo, hi)
    return Integer(n)


def _apply_op(eq: Eq, var: Symbol, rng: random.Random, depth: int) -> Op:
    """Pick a random reversible op + apply to eq. Returns Op record."""
    # Weight ops by depth. Square is rare. add-var-term only once (rough).
    choices = [
        ("add", 5),
        ("sub", 5),
        ("mul", 4),
        ("div", 3),
        ("neg", 1),
        ("addvar", 2 if depth == 0 else 0),  # only at outermost (gives both-sides)
        ("sqr", 1 if depth == 0 else 0),     # only at outermost
    ]
    kinds, weights = zip(*[c for c in choices if c[1] > 0])
    kind = rng.choices(kinds, weights=weights, k=1)[0]

    if kind == "add":
        k = _maybe_sym_int(rng, -15, 15)
        return Op("add", k, False, Eq(eq.lhs + k, eq.rhs + k))
    if kind == "sub":
        k = _maybe_sym_int(rng, 1, 15)  # positive — rendered as "- k"
        return Op("sub", k, False, Eq(eq.lhs - k, eq.rhs - k))
    if kind == "mul":
        k = _maybe_sym_int(rng, 2, 12)
        # 60% expand at shallow depth, decreasing to keep parens at deep
        keep_paren = rng.random() < min(0.7, 0.2 + 0.15 * depth)
        if keep_paren:
            new_lhs = sp.Mul(k, eq.lhs, evaluate=False)
        else:
            new_lhs = expand(k * eq.lhs)
        return Op("mul", k, keep_paren, Eq(new_lhs, eq.rhs * k))
    if kind == "div":
        k = _maybe_sym_int(rng, 2, 8)
        return Op("div", k, False, Eq(eq.lhs / k, eq.rhs / k))
    if kind == "neg":
        return Op("neg", None, False, Eq(-eq.lhs, -eq.rhs))
    if kind == "addvar":
        # add k*var to both sides → both-sides form
        coef = _maybe_sym_int(rng, 1, 6)
        term = coef * var
        return Op("addvar", coef, False, Eq(eq.lhs + term, eq.rhs + term))
    if kind == "sqr":
        # only safe when current RHS is positive (so x = ±√rhs is ok and
        # we declare the positive root). Require eq is currently 'var = pos'.
        # Caller should guard; here we just apply.
        return Op("sqr", None, False, Eq(eq.lhs ** 2, eq.rhs ** 2))
    raise ValueError(kind)


_SUPER = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _polish(s: str) -> str:
    """Common cosmetic cleanups applied to any rendered sub-expression."""
    # x**2 → x²  (unicode superscript)
    def _sup(m): return m.group(1) + m.group(2).translate(_SUPER)
    s = re.sub(r"([a-z\)])\*\*(\d+)", _sup, s)
    # 5*x → 5x
    s = re.sub(r"(\d)\*([a-z])", r"\1\2", s)
    # x*5 → 5x
    s = re.sub(r"([a-z])\*(\d)", r"\2\1", s)
    # 3*(...) → 3(...)
    s = re.sub(r"(\d)\*\(", r"\1(", s)
    # x*(...) → x(...)
    s = re.sub(r"([a-z])\*\(", r"\1(", s)
    # )*(  → )(
    s = re.sub(r"\)\*\(", r")(", s)
    # )*x  → )x
    s = re.sub(r"\)\*([a-z])", r")\1", s)
    # a*b (variable juxtaposition for two-var products) — run twice to
    # catch chains like a*b*c → abc
    for _ in range(2):
        s = re.sub(r"([a-z])\*([a-z])", r"\1\2", s)
    # "+ -k" → "- k"
    s = re.sub(r"\+\s*-\s*", "- ", s)
    # "- -k" → "+ k"  (e.g. "65 - -3" from add op with negative k)
    s = re.sub(r"-\s+-\s*", "+ ", s)
    return s


def _render_eq(eq: Eq) -> str:
    """Plain text rendering of an Eq — clean coefficient/var notation."""
    return _polish(str(eq.lhs) + " = " + str(eq.rhs))


def _render_op_inverse(op: Op) -> str:
    """Short label like 'subtrahu 5' / '÷ 3' / 'aldonu 2x'."""
    if op.kind == "add":
        return f"- {op.k}" if op.k > 0 else f"+ {-op.k}"
    if op.kind == "sub":
        return f"+ {op.k}"
    if op.kind == "mul":
        return f"÷ {op.k}"
    if op.kind == "div":
        return f"× {op.k}"
    if op.kind == "neg":
        return "× (-1)"
    if op.kind == "addvar":
        return f"- {op.k}{ _render_op_inverse_var(op) }"
    if op.kind == "sqr":
        return "√"
    return ""


def _render_op_inverse_var(op: Op) -> str:
    # placeholder kept for symmetry; only addvar uses
    return ""  # caller appends var letter directly


def _peel_chain(ops: list[Op], eq_final: Eq, var: Symbol, sampled_answer) -> list[Eq]:
    """Reverse-apply ops to produce intermediate eqs from puzzle → solution."""
    eqs = [eq_final]
    cur = eq_final
    for op in reversed(ops):
        if op.kind == "add":
            cur = Eq(cur.lhs - op.k, cur.rhs - op.k)
        elif op.kind == "sub":
            cur = Eq(cur.lhs + op.k, cur.rhs + op.k)
        elif op.kind == "mul":
            cur = Eq(cur.lhs / op.k, cur.rhs / op.k)
            # If we kept parens, dividing by `k` should collapse the wrapper —
            # use sympy to simplify the lhs back to its pre-mul form
            cur = Eq(sp.simplify(cur.lhs), sp.simplify(cur.rhs))
        elif op.kind == "div":
            cur = Eq(cur.lhs * op.k, cur.rhs * op.k)
            cur = Eq(sp.simplify(cur.lhs), sp.simplify(cur.rhs))
        elif op.kind == "neg":
            cur = Eq(-cur.lhs, -cur.rhs)
        elif op.kind == "addvar":
            cur = Eq(cur.lhs - op.k * var, cur.rhs - op.k * var)
            cur = Eq(sp.simplify(cur.lhs), sp.simplify(cur.rhs))
        elif op.kind == "sqr":
            # we declare the positive root
            cur = Eq(sp.simplify(sp.sqrt(cur.lhs)), sp.simplify(sp.sqrt(cur.rhs)))
        eqs.append(cur)
    # Final eq should be `var = sampled_answer` (possibly with sign flip for neg/sqr)
    return eqs


# ── Renderers — produce final chain text from list of eqs ────────────────

def render_vertical(eqs: list[Eq]) -> str:
    """One eq per line, blank line at end."""
    return "\n".join(_render_eq(e) for e in eqs)


def render_arrow(eqs: list[Eq]) -> str:
    """eq1 → eq2 → … → eqN, single line."""
    return " → ".join(_render_eq(e) for e in eqs)


def render_with_calc(eqs: list[Eq], ops: list[Op]) -> str:
    """Show the intermediate computation: `5x = 18 - 3 = 15` style.

    Limited to numeric simplification on the rhs where one op is undone:
    we show `rhs_before` op_inverse_literal `rhs_after`.
    """
    lines = [_render_eq(eqs[0])]
    rev_ops = list(reversed(ops))
    for i, (prev, cur, op) in enumerate(zip(eqs[:-1], eqs[1:], rev_ops)):
        # Compose calc-visible line for the RHS
        if op.kind == "add":
            calc = f"{prev.rhs} - {op.k}"
        elif op.kind == "sub":
            calc = f"{prev.rhs} + {op.k}"
        elif op.kind == "mul":
            calc = f"{prev.rhs} / {op.k}"
        elif op.kind == "div":
            calc = f"{prev.rhs} * {op.k}"
        elif op.kind == "neg":
            calc = f"-({prev.rhs})"
        elif op.kind == "sqr":
            calc = f"√{prev.rhs}"
        else:
            calc = None
        lhs = _polish(str(cur.lhs))
        if calc:
            lines.append(_polish(f"{lhs} = {calc} = {cur.rhs}"))
        else:
            lines.append(_render_eq(cur))
    return "\n".join(lines)


def render_compact(eqs: list[Eq], var: Symbol) -> str:
    """Just the puzzle and the final answer: `eq; var = answer`."""
    return f"{_render_eq(eqs[0])}; {var} = {eqs[-1].rhs}"


# ── Top-level generator ──────────────────────────────────────────────────

def _safe_eq_repr(eq: Eq) -> bool:
    """Reject pathological equations: too-long, degenerate (0=0), etc."""
    s = _render_eq(eq)
    if len(s) > 120: return False
    if "=" not in s: return False
    if "zoo" in s or "nan" in s or "I" in s: return False
    return True


def gen_one(rng: random.Random) -> dict | None:
    var_name = rng.choice(VARS)
    var = Symbol(var_name)

    # Sample answer: integer or simple fraction
    if rng.random() < 0.85:
        answer = Integer(rng.choice([n for n in range(-15, 16) if n != 0]))
    else:
        num = rng.choice([n for n in range(-12, 13) if n != 0])
        den = rng.choice([d for d in range(2, 11)])
        answer = Rational(num, den)

    eq = Eq(var, answer)
    ops: list[Op] = []
    depth = rng.choice([1, 2, 2, 3, 3, 4])  # bias toward 2-3 ops
    for d in range(depth):
        try:
            op = _apply_op(eq, var, rng, depth=d)
        except Exception:
            return None
        eq = op.eq_after
        ops.append(op)
        # Reject if intermediate gets out of hand
        try:
            if not _safe_eq_repr(eq):
                return None
        except Exception:
            return None

    if not _safe_eq_repr(eq):
        return None

    # Build the chain by reversing
    try:
        chain_eqs = _peel_chain(ops, eq, var, answer)
    except Exception:
        return None

    # Verification: substitute answer into the puzzle, simplify, must give True or 0
    try:
        check = sp.simplify(eq.lhs.subs(var, answer) - eq.rhs.subs(var, answer))
        if check != 0:
            return None
    except Exception:
        return None

    # Final eq should match `var = answer` (allowing sign for neg/sqr)
    final = chain_eqs[-1]
    if not (final.lhs == var or final.lhs == -var):
        # Try once more flipping if lhs is -var
        if final.lhs == -var:
            chain_eqs[-1] = Eq(var, -final.rhs)
        else:
            return None

    # Render style — bias against compact since it doesn't teach chains
    style = rng.choices(
        ["vertical", "arrow", "with_calc", "compact"],
        weights=[35, 30, 27, 8], k=1)[0]
    if style == "vertical":
        text = render_vertical(chain_eqs)
    elif style == "arrow":
        text = render_arrow(chain_eqs)
    elif style == "with_calc":
        text = render_with_calc(chain_eqs, ops)
    else:
        text = render_compact(chain_eqs, var)

    return {"text": text, "_kind": "solve", "_style": style, "_n_ops": len(ops),
            "_answer": str(answer), "_has_var_var": any(o.kind == "addvar" for o in ops)}


# ── Polynomial identities (expand / factor / collect) ────────────────────

def _two_distinct_vars(rng: random.Random) -> tuple[Symbol, Symbol]:
    a, b = rng.sample(VARS, 2)
    return Symbol(a), Symbol(b)


def _render_stages(stages: list[sp.Expr | str], style: str, intro: str = "") -> str:
    """Render a list of stages (sympy exprs or already-strings) as a chain.

    style:
      - 'eq':        first = last (single-line, no intermediate)
      - 'multiline': line 1, then `= line N` for each subsequent
      - 'arrow':     line1 = line2 = ... (all on one line)
    intro: optional prefix like 'Expand: ' (prepended to first stage)
    """
    rendered = [_polish(str(s)) if not isinstance(s, str) else _polish(s) for s in stages]
    if style == "eq":
        text = f"{rendered[0]} = {rendered[-1]}"
    elif style == "multiline":
        lines = [rendered[0]] + [f"= {s}" for s in rendered[1:]]
        text = "\n".join(lines)
    else:  # arrow / inline
        text = " = ".join(rendered)
    return (intro + text) if intro else text


def gen_expand_identity(rng: random.Random) -> dict | None:
    """Expansion identity: factored form → (optional FOIL) → simplified."""
    shape = rng.choice([
        "distrib_single", "binomial_xx", "binomial_diff_vars",
        "scalar_times_poly", "square", "diff_squares",
    ])

    if shape == "distrib_single":
        # a(b + c) → a·b + a·c  (one meaningful intermediate)
        a, b = _two_distinct_vars(rng)
        c = _maybe_sym_int(rng, -10, 10)
        lhs = sp.Mul(a, (b + c), evaluate=False)
        # intermediate: explicit a*b + a*c (will polish to ab + ac)
        mid = sp.Add(sp.Mul(a, b, evaluate=False),
                     sp.Mul(a, c, evaluate=False), evaluate=False)
        final = expand(lhs)
        stages = [lhs, mid, final] if str(mid) != str(final) else [lhs, final]
    elif shape == "binomial_xx":
        # (x+a)(x+b) → x·x + b·x + a·x + a·b → x² + (a+b)x + a·b
        x = Symbol(rng.choice(VARS))
        c1, c2 = _maybe_sym_int(rng, -10, 10), _maybe_sym_int(rng, -10, 10)
        lhs = sp.Mul((x + c1), (x + c2), evaluate=False)
        foil = sp.Add(sp.Mul(x, x, evaluate=False),
                      sp.Mul(c2, x, evaluate=False),
                      sp.Mul(c1, x, evaluate=False),
                      sp.Mul(c1, c2, evaluate=False), evaluate=False)
        final = expand(lhs)
        stages = [lhs, foil, final]
    elif shape == "binomial_diff_vars":
        # (x+a)(y+b) → x·y + b·x + a·y + a·b (no further collection)
        x, y = _two_distinct_vars(rng)
        c1, c2 = _maybe_sym_int(rng, -10, 10), _maybe_sym_int(rng, -10, 10)
        lhs = sp.Mul((x + c1), (y + c2), evaluate=False)
        foil = sp.Add(sp.Mul(x, y, evaluate=False),
                      sp.Mul(c2, x, evaluate=False),
                      sp.Mul(c1, y, evaluate=False),
                      sp.Mul(c1, c2, evaluate=False), evaluate=False)
        final = expand(lhs)
        stages = [lhs, foil, final] if str(foil) != str(final) else [lhs, final]
    elif shape == "scalar_times_poly":
        # k·(x+a)·(y+b) → distribute k, then FOIL → final
        x, y = _two_distinct_vars(rng)
        k = _maybe_sym_int(rng, 2, 8)
        c1, c2 = _maybe_sym_int(rng, -8, 8), _maybe_sym_int(rng, -8, 8)
        lhs = sp.Mul(k, (x + c1), (y + c2), evaluate=False)
        # intermediate: expanded inner product, k still outside
        inner = expand((x + c1) * (y + c2))
        mid = sp.Mul(k, inner, evaluate=False)
        final = expand(lhs)
        stages = [lhs, mid, final]
    elif shape == "square":
        # (x+a)² → (x+a)(x+a) → x² + 2ax + a²
        x = Symbol(rng.choice(VARS))
        c = _maybe_sym_int(rng, -8, 8)
        lhs = sp.Pow((x + c), 2, evaluate=False)
        mid1 = sp.Mul((x + c), (x + c), evaluate=False)
        final = expand(lhs)
        stages = [lhs, mid1, final]
    else:  # diff_squares: (x+a)(x-a) → x·x - ax + ax - a² → x² - a²
        x = Symbol(rng.choice(VARS))
        c = _maybe_sym_int(rng, 2, 10)
        lhs = sp.Mul((x + c), (x - c), evaluate=False)
        foil = sp.Add(sp.Mul(x, x, evaluate=False),
                      sp.Mul(-c, x, evaluate=False),
                      sp.Mul(c, x, evaluate=False),
                      sp.Mul(-c, c, evaluate=False), evaluate=False)
        final = expand(lhs)
        stages = [lhs, foil, final]

    # Verify
    if expand(lhs - stages[-1]) != 0:
        return None

    style = rng.choices(["eq", "multiline", "arrow"], weights=[15, 60, 25], k=1)[0]
    intro = ""
    if rng.random() < 0.2:
        intro = "Expand: "
    text = _render_stages(stages, style, intro)
    return {"text": text, "_kind": "expand", "_style": style, "_shape": shape}


def gen_factor_identity(rng: random.Random) -> dict | None:
    """Factoring identity: expanded → (common-factor noted) → factored form."""
    shape = rng.choice([
        "distrib_single", "binomial_diff_squares", "binomial_xx", "scalar_factor",
    ])

    if shape == "distrib_single":
        # ab + ac → a·b + a·c → a(b + c)
        a, b = _two_distinct_vars(rng)
        c = _maybe_sym_int(rng, -10, 10)
        factored = sp.Mul(a, (b + c), evaluate=False)
        expanded = expand(factored)
        mid = sp.Add(sp.Mul(a, b, evaluate=False),
                     sp.Mul(a, c, evaluate=False), evaluate=False)
        stages = [expanded, mid, factored] if str(mid) != str(expanded) else [expanded, factored]
    elif shape == "binomial_diff_squares":
        # x² - a² → x² - (a)² → (x+a)(x-a)
        x = Symbol(rng.choice(VARS))
        c = _maybe_sym_int(rng, 2, 10)
        factored = sp.Mul((x + c), (x - c), evaluate=False)
        expanded = expand(factored)
        # intermediate: rewrite the constant as a square
        mid_str = _polish(str(expanded)).replace(f"- {c*c}", f"- {c}²")
        stages = [expanded, mid_str, factored]
    elif shape == "binomial_xx":
        # x² + (a+b)x + ab → (x+a)(x+b)
        # Useful intermediate: note that a+b and a*b match the middle/last terms
        x = Symbol(rng.choice(VARS))
        c1, c2 = _maybe_sym_int(rng, -10, 10), _maybe_sym_int(rng, -10, 10)
        factored = sp.Mul((x + c1), (x + c2), evaluate=False)
        expanded = expand(factored)
        # Skip intermediate for this shape — sympy doesn't give a clean
        # "find roots" step text. Just expand → factor.
        stages = [expanded, factored]
    else:  # scalar_factor
        # kx + ky → k·x + k·y → k(x + y)
        x, y = _two_distinct_vars(rng)
        k = _maybe_sym_int(rng, 2, 8)
        factored = sp.Mul(k, (x + y), evaluate=False)
        expanded = expand(factored)
        mid = sp.Add(sp.Mul(k, x, evaluate=False),
                     sp.Mul(k, y, evaluate=False), evaluate=False)
        stages = [expanded, mid, factored] if str(mid) != str(expanded) else [expanded, factored]

    if expand(factored - expand(expanded)) != 0:
        return None

    style = rng.choices(["eq", "multiline", "arrow"], weights=[15, 60, 25], k=1)[0]
    intro = ""
    if rng.random() < 0.2:
        intro = "Factor: "
    text = _render_stages(stages, style, intro)
    return {"text": text, "_kind": "factor", "_style": style, "_shape": shape}


def gen_collect_terms(rng: random.Random) -> dict | None:
    """Like-terms collection: 2x + 3x + 5 = (2+3)x + 5 = 5x + 5."""
    x = Symbol(rng.choice(VARS))
    var_coefs = []
    consts = []
    n_terms = rng.randint(2, 4)
    for _ in range(n_terms):
        c = _maybe_sym_int(rng, -8, 8)
        var_coefs.append(int(c))
    n_const = rng.randint(1, 3)
    for _ in range(n_const):
        k = _maybe_sym_int(rng, -10, 10)
        consts.append(int(k))

    # original-order presentation
    parts = [c * x for c in var_coefs] + consts
    rng.shuffle(parts)
    lhs = sum(parts[1:], parts[0])

    coef_sum = sum(var_coefs)
    const_sum = sum(consts)

    # intermediate: pull the var coefficients together
    coef_expr = " + ".join(str(c) for c in var_coefs)
    coef_expr_polished = re.sub(r"\+\s*-\s*", "- ", coef_expr)
    const_expr = " + ".join(str(c) for c in consts) if consts else ""
    const_expr_polished = re.sub(r"\+\s*-\s*", "- ", const_expr)
    if const_expr_polished:
        mid_str = f"({coef_expr_polished}){x} + ({const_expr_polished})"
    else:
        mid_str = f"({coef_expr_polished}){x}"
    final_expr = coef_sum * x + const_sum
    if expand(lhs - final_expr) != 0:
        return None

    style = rng.choices(["eq", "multiline", "arrow"], weights=[15, 60, 25], k=1)[0]
    text = _render_stages([lhs, mid_str, final_expr], style)
    return {"text": text, "_kind": "collect", "_style": style, "_shape": "collect"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solve-frac", type=float, default=0.65,
                    help="fraction of records that are solve-equation chains "
                         "(rest split among expand/factor/collect)")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    kind_count = Counter()
    style_count = Counter()
    shape_count = Counter()
    n_emitted = 0
    n_rejected = 0
    with args.out.open("w") as f:
        while n_emitted < args.n:
            r = rng.random()
            if r < args.solve_frac:
                rec = gen_one(rng)
            elif r < args.solve_frac + (1 - args.solve_frac) * 0.5:
                rec = gen_expand_identity(rng)
            elif r < args.solve_frac + (1 - args.solve_frac) * 0.8:
                rec = gen_factor_identity(rng)
            else:
                rec = gen_collect_terms(rng)
            if rec is None:
                n_rejected += 1
                continue
            kind_count[rec["_kind"]] += 1
            style_count[rec["_style"]] += 1
            if "_shape" in rec:
                shape_count[f"{rec['_kind']}:{rec['_shape']}"] += 1
            f.write(json.dumps({"text": rec["text"]}, ensure_ascii=False) + "\n")
            n_emitted += 1
    print(f"\nwrote {n_emitted:,} → {args.out}  ({n_rejected:,} rejected)")
    print("\nby kind:")
    for k, v in sorted(kind_count.items()):
        print(f"  {k:14s} {v:>7,}")
    print("\nby style:")
    for k, v in sorted(style_count.items()):
        print(f"  {k:14s} {v:>7,}")
    print("\nby shape (identity kinds):")
    for k, v in sorted(shape_count.items()):
        print(f"  {k:35s} {v:>7,}")


if __name__ == "__main__":
    main()
