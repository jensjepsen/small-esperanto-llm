"""Procedural algebra dataset generator (v2) — sympy-native rendering.

Goals over v1 (scripts/gen_algebra_pretrain.py):
- All math rendered through sympy's StrPrinter (subclassed), no string-regex
  rendering of expressions. Eliminates the `xx`/`aa`/`1m0`/`3*3` bugs.
- All "intermediate teaching" steps are built with sympy `evaluate=False`
  so the un-collected form is preserved; `expand()/collect()/factor()` gives
  the canonical RHS.
- Reject pass for: tautologies (`X = X`), no-progression chains
  (consecutive identical lines), and single-line rows that should have
  intermediate steps.
- `gen_one` solve chains kept (already correct) — only renderer swapped.

Run:
    python scripts/gen_algebra_v2.py --n 5000 --out /tmp/algebra_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path as _P

# Script runs under PyPy with --no-project, so the espllm package isn't
# installed in the env. Add src/ to sys.path so we can import its modules.
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
# wordify_text is language-specific; picked in main() based on --lang.
# Default: EO. Set by main() so gen_one/_worker see the right function.
wordify_text = None  # type: ignore
from pathlib import Path

import sympy as sp
from sympy import Add, Eq, Integer, Mul, Pow, Rational, Symbol, expand, factor


# ── Rendering — sympy StrPrinter subclass + light cosmetics ──────────────

_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_VARS = ["a", "b", "k", "m", "n", "p", "t", "x", "y", "z"]


class AlgebraPrinter(sp.printing.str.StrPrinter):
    """Render sympy expressions in algebra-textbook style.

    Rules:
      x**2          → x²
      (x+1)**2      → (x+1)²
      x*x           → x²        (collapses Mul(x,x,evaluate=False))
      3*x           → 3x        (single-letter var)
      x*y           → xy        (two single-letter vars)
      3*(x+1)       → 3(x+1)
      (x+1)*(x-1)   → (x+1)(x-1)
    """

    def _print_Pow(self, expr):
        base, exp = expr.base, expr.exp
        if isinstance(exp, Integer) and exp >= 0:
            sup = str(exp).translate(_SUP)
            if base.is_Symbol or (base.is_Atom and len(str(base)) == 1):
                return f"{self._print(base)}{sup}"
            return f"({self._print(base)}){sup}"
        return super()._print_Pow(expr)


def _polish(s: str) -> str:
    """Post-printer cosmetics: drop `*` where unambiguous in display."""
    # x**N → x^N collapsed for any base — handled by printer for Pow already,
    # but keep belt-and-suspenders for raw strings emitted by other paths.
    # 3*x → 3x  (digit then single lowercase letter, not followed by another letter)
    s = re.sub(r"(\d)\*([a-z])(?![a-z])", r"\1\2", s)
    # x*5 → 5x  (letter then digit; normalize)
    s = re.sub(r"\b([a-z])\*(\d)", r"\2\1", s)
    # 3*(...) → 3(...)
    s = re.sub(r"(\d)\*\(", r"\1(", s)
    # )*(  → )(
    s = re.sub(r"\)\*\(", r")(", s)
    # )*x → )x
    s = re.sub(r"\)\*([a-z])", r")\1", s)
    # x*( → x(
    s = re.sub(r"\b([a-z])\*\(", r"\1(", s)
    # Repeated single-letter mul: x*x → x² and x*y → xy
    # Run twice to fold chains like x*y*z → xyz
    for _ in range(2):
        s = re.sub(r"\b([a-z])\*\1\b",
                   lambda m: m.group(1) + "²", s)
        s = re.sub(r"\b([a-z])\*([a-z])\b", r"\1\2", s)
    # Drop leading "1" coefficient: "1x" → "x", "- 1x" → "- x", etc.
    # Only when "1" is preceded by start-of-string, space, "(", or "+/-".
    s = re.sub(r"(?<![0-9])1([a-z])", r"\1", s)
    # "+ -k" → "- k"
    s = re.sub(r"\+\s*-\s*", "- ", s)
    # "- -k" → "+ k"
    s = re.sub(r"-\s+-\s*", "+ ", s)
    return s


def render(expr) -> str:
    """Sympy expression → display string."""
    return _polish(AlgebraPrinter().doprint(expr))


def render_eq(eq: Eq) -> str:
    return f"{render(eq.lhs)} = {render(eq.rhs)}"


# ── solve-equation chains (gen_one) — unchanged math, new renderer ────────


@dataclass
class Op:
    kind: str
    k: object
    eq_after: Eq


PRESETS = {
    "easy": dict(
        depths=[1],
        op_weights={"add": 5, "sub": 5, "mul": 3, "div": 2, "neg": 0, "addvar": 0},
        add_k=(-9, 9), sub_k=(1, 9), mul_k=(2, 6), div_k=(2, 5),
        addvar_coef=(1, 3),
        ans_int_range=(-9, 9), ans_rational_frac=0.0,
        keep_paren_base=0.0, keep_paren_per_depth=0.0,
    ),
    "medium": dict(
        depths=[1, 2, 2, 3],
        op_weights={"add": 5, "sub": 5, "mul": 4, "div": 3, "neg": 1, "addvar": 1},
        add_k=(-12, 12), sub_k=(1, 12), mul_k=(2, 10), div_k=(2, 8),
        addvar_coef=(1, 4),
        ans_int_range=(-12, 12), ans_rational_frac=0.10,
        keep_paren_base=0.3, keep_paren_per_depth=0.10,
    ),
    "hard": dict(
        depths=[1, 2, 2, 3, 3, 4],
        op_weights={"add": 5, "sub": 5, "mul": 4, "div": 3, "neg": 1, "addvar": 2},
        add_k=(-15, 15), sub_k=(1, 15), mul_k=(2, 12), div_k=(2, 8),
        addvar_coef=(1, 6),
        ans_int_range=(-15, 15), ans_rational_frac=0.15,
        keep_paren_base=0.4, keep_paren_per_depth=0.15,
    ),
}


def _apply_op(eq: Eq, var: Symbol, rng: random.Random, depth: int, preset: dict) -> Op:
    choices = list(preset["op_weights"].items())
    # addvar only at outermost depth (so it produces both-sides form once)
    if depth > 0:
        choices = [(k, w) for k, w in choices if k != "addvar"]
    choices = [(k, w) for k, w in choices if w > 0]
    kinds, weights = zip(*choices)
    kind = rng.choices(kinds, weights=weights, k=1)[0]
    if kind == "add":
        k = _nonzero(rng, *preset["add_k"])
        return Op("add", k, Eq(eq.lhs + k, eq.rhs + k))
    if kind == "sub":
        k = _nonzero(rng, *preset["sub_k"])
        return Op("sub", k, Eq(eq.lhs - k, eq.rhs - k))
    if kind == "mul":
        k = _nonzero(rng, *preset["mul_k"])
        keep_paren = rng.random() < preset["keep_paren_base"] + preset["keep_paren_per_depth"] * depth
        if keep_paren:
            new_lhs = Mul(k, eq.lhs, evaluate=False)
            # Symbol-bearing RHS must stay factored to match LHS — otherwise
            # the factored LHS expands to a different polynomial than the
            # auto-distributed RHS (different polynomials sharing one root).
            # Constant RHS evaluates normally — no mismatch risk.
            if eq.rhs.free_symbols:
                new_rhs = Mul(k, eq.rhs, evaluate=False)
            else:
                new_rhs = eq.rhs * k
        else:
            new_lhs = expand(k * eq.lhs)
            new_rhs = expand(k * eq.rhs)
        return Op("mul", k, Eq(new_lhs, new_rhs))
    if kind == "div":
        k = _nonzero(rng, *preset["div_k"])
        return Op("div", k, Eq(eq.lhs / k, eq.rhs / k))
    if kind == "neg":
        return Op("neg", None, Eq(-eq.lhs, -eq.rhs))
    if kind == "addvar":
        coef = _nonzero(rng, *preset["addvar_coef"])
        term = coef * var
        return Op("addvar", coef, Eq(eq.lhs + term, eq.rhs + term))
    raise ValueError(kind)


def _nonzero(rng: random.Random, lo: int, hi: int) -> Integer:
    n = 0
    while n == 0:
        n = rng.randint(lo, hi)
    return Integer(n)


def _collapse_ops(ops: list[Op]) -> list[Op]:
    """Fold consecutive commutative ops into single ops.

    add+add/sub  →  add or sub (net signed sum)
    sub+add/sub  →  add or sub (net signed sum)
    mul+mul/div  →  mul or div (net factor, if integer-clean)
    div+mul/div  →  mul or div
    Cancellations (net = 0 for add/sub, net = 1 for mul/div) drop both ops.

    Other categories (neg, addvar, mixing with add/mul) pass through.
    Updates each kept op's `eq_after` to remain consistent (since the
    composition is the same regardless of how we factor it).
    """
    out: list[Op] = []
    for op in ops:
        if not out:
            out.append(op); continue
        last = out[-1]
        # add/sub category
        if last.kind in ("add", "sub") and op.kind in ("add", "sub"):
            net = (last.k if last.kind == "add" else -last.k) + \
                  (op.k if op.kind == "add" else -op.k)
            if net == 0:
                out.pop()                                    # cancel
            elif net > 0:
                out[-1] = Op("add", Integer(net), op.eq_after)
            else:
                out[-1] = Op("sub", Integer(-net), op.eq_after)
            continue
        # mul/div category
        if last.kind in ("mul", "div") and op.kind in ("mul", "div"):
            l_factor = last.k if last.kind == "mul" else Rational(1, last.k)
            o_factor = op.k if op.kind == "mul" else Rational(1, op.k)
            net = sp.Rational(l_factor * o_factor)
            if net == 1:
                out.pop()                                    # cancel
            elif net.q == 1:                                 # integer
                out[-1] = Op("mul", Integer(net.p), op.eq_after)
            elif net.p == 1 and net.q > 1:                   # 1/k
                out[-1] = Op("div", Integer(net.q), op.eq_after)
            else:
                # awkward like 2/3 — keep both for now (rare)
                out.append(op)
            continue
        out.append(op)
    return out


def _peel(ops: list[Op], puzzle: Eq, var: Symbol) -> tuple[list[Eq], list[Op]]:
    """Return (chain_eqs, ops_used) — ops_used is the collapsed list aligned
    1:1 with chain_eqs[1:] (each op produces the next eq via its inverse).
    """
    ops = _collapse_ops(ops)
    eqs = [puzzle]
    cur = puzzle
    for op in reversed(ops):
        if op.kind == "add":   cur = Eq(cur.lhs - op.k, cur.rhs - op.k)
        elif op.kind == "sub": cur = Eq(cur.lhs + op.k, cur.rhs + op.k)
        elif op.kind == "mul": cur = Eq(sp.simplify(cur.lhs / op.k), sp.simplify(cur.rhs / op.k))
        elif op.kind == "div": cur = Eq(sp.simplify(cur.lhs * op.k), sp.simplify(cur.rhs * op.k))
        elif op.kind == "neg": cur = Eq(-cur.lhs, -cur.rhs)
        elif op.kind == "addvar":
            cur = Eq(sp.simplify(cur.lhs - op.k * var), sp.simplify(cur.rhs - op.k * var))
        eqs.append(cur)
    return eqs, ops


def gen_one(rng: random.Random, preset: dict | str = "hard",
            verify_text: bool = True) -> dict | None:
    if isinstance(preset, str):
        preset = PRESETS[preset]
    var = Symbol(rng.choice(_VARS))
    lo, hi = preset["ans_int_range"]
    if rng.random() < (1 - preset["ans_rational_frac"]):
        answer = Integer(rng.choice([n for n in range(lo, hi + 1) if n != 0]))
    else:
        answer = Rational(rng.choice([n for n in range(-12, 13) if n != 0]),
                          rng.choice(range(2, 11)))
    eq = Eq(var, answer)
    depth = rng.choice(preset["depths"])
    ops: list[Op] = []
    for d in range(depth):
        try:
            op = _apply_op(eq, var, rng, d, preset)
        except Exception:
            return None
        eq = op.eq_after
        ops.append(op)
        if not _safe(eq):
            return None
    try:
        chain, ops = _peel(ops, eq, var)
    except Exception:
        return None
    # Reject if collapse folded every op to identity — chain would render as
    # just the seed `var = answer`, which teaches nothing.
    if not ops:
        return None
    # Level 1 verify: every chain Eq must hold under the answer.
    try:
        for ce in chain:
            if sp.simplify(ce.lhs.subs(var, answer) - ce.rhs.subs(var, answer)) != 0:
                return None
    except Exception:
        return None
    final = chain[-1]
    if final.lhs == -var:
        chain[-1] = Eq(var, -final.rhs)
    elif final.lhs != var:
        return None
    text = _render_didactic(chain, ops, var)
    # Level 2 verify: round-trip every rendered line through sympy parser
    # and check each parsed Eq holds under the answer.
    if verify_text and not _verify_rendered_text(text, var, answer):
        return None
    # `answer` is appended as `#### N` in _worker AFTER wordify, so the
    # marker line always stays as digits (matches gsm8k / word-problems).
    return {"text": text, "answer": render(chain[-1].rhs),
            "_kind": "solve", "_style": "didactic", "_n_ops": len(ops)}


# Sympy text parser for round-trip verification.
_PARSE_TRANSFORMS = None
def _verify_rendered_text(text: str, var: Symbol, answer) -> bool:
    """Parse each `=`-separated line back to a sympy Eq and verify it
    holds when `var` is substituted with `answer`. Catches rendering
    bugs (e.g., wrong multiplier, dropped paren).
    """
    global _PARSE_TRANSFORMS
    if _PARSE_TRANSFORMS is None:
        from sympy.parsing.sympy_parser import (
            standard_transformations, implicit_multiplication_application,
            convert_xor,
        )
        _PARSE_TRANSFORMS = (standard_transformations +
                             (implicit_multiplication_application, convert_xor))
    from sympy.parsing.sympy_parser import parse_expr
    # Undo cosmetic substitutions so the parser can read it
    _UNSUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    def _unprep(s):
        # x² → x**2; superscripts → ^ then convert_xor handles them
        out = []
        i = 0
        while i < len(s):
            c = s[i]
            if c in "⁰¹²³⁴⁵⁶⁷⁸⁹":
                # collect a run of superscripts
                run = ""
                while i < len(s) and s[i] in "⁰¹²³⁴⁵⁶⁷⁸⁹":
                    run += s[i]
                    i += 1
                out.append(f"**{run.translate(_UNSUP)}")
            else:
                out.append(c)
                i += 1
        return "".join(out)

    local = {str(var): var}
    for line in text.split("\n"):
        if "=" not in line:
            return False
        # split on first `=` only — middle `=` of with_calc would also work
        # but didactic style only ever has one `=` per line
        lhs_s, rhs_s = line.split("=", 1)
        try:
            lhs = parse_expr(_unprep(lhs_s.strip()), local_dict=local,
                             transformations=_PARSE_TRANSFORMS)
            rhs = parse_expr(_unprep(rhs_s.strip()), local_dict=local,
                             transformations=_PARSE_TRANSFORMS)
        except Exception:
            return False
        try:
            diff = sp.simplify(lhs.subs(var, answer) - rhs.subs(var, answer))
            if diff != 0:
                return False
        except Exception:
            return False
    return True


def _paren_if_add(expr) -> str:
    """Render with surrounding parens iff top-level is an Add (to prevent
    precedence ambiguity when concatenating `... * k` or `... / k`)."""
    s = render(expr)
    if expr.is_Add:
        return f"({s})"
    return s


def _render_didactic(chain: list[Eq], ops: list[Op], var: Symbol) -> str:
    """Each peel step gets an explicit `apply same op to both sides` line
    followed by the simplified result.

        10a = 70
        10a / 10 = 70 / 10
        a = 7
    """
    lines = [render_eq(chain[0])]
    rev = list(reversed(ops))
    for prev, cur, op in zip(chain[:-1], chain[1:], rev):
        # For mul/div ops we need parens around Add-typed sides to avoid
        # `a/3 - 4 * 3` being read as `a/3 - (4 * 3)`.
        if op.kind in ("mul", "div"):
            l, r = _paren_if_add(prev.lhs), _paren_if_add(prev.rhs)
        else:
            l, r = render(prev.lhs), render(prev.rhs)
        if op.kind == "add":    op_l, op_r = f"{l} - {op.k}", f"{r} - {op.k}"
        elif op.kind == "sub":  op_l, op_r = f"{l} + {op.k}", f"{r} + {op.k}"
        elif op.kind == "mul":  op_l, op_r = f"{l} / {op.k}", f"{r} / {op.k}"
        elif op.kind == "div":  op_l, op_r = f"{l} * {op.k}", f"{r} * {op.k}"
        elif op.kind == "neg":  op_l, op_r = f"-({l})",       f"-({r})"
        elif op.kind == "addvar":
            term = render(op.k * var)
            op_l, op_r = f"{l} - {term}", f"{r} - {term}"
        else:
            op_l = op_r = None
        if op_l is not None:
            lines.append(_polish(f"{op_l} = {op_r}"))
        lines.append(render_eq(cur))
    return "\n".join(lines)


def _render_with_calc(eqs, ops):
    """Show intermediate computation: `5x = 18 - 3 = 15`."""
    lines = [render_eq(eqs[0])]
    rev = list(reversed(ops))
    for prev, cur, op in zip(eqs[:-1], eqs[1:], rev):
        if op.kind == "add":   calc = f"{render(prev.rhs)} - {op.k}"
        elif op.kind == "sub": calc = f"{render(prev.rhs)} + {op.k}"
        elif op.kind == "mul": calc = f"{render(prev.rhs)} / {op.k}"
        elif op.kind == "div": calc = f"{render(prev.rhs)} * {op.k}"
        elif op.kind == "neg": calc = f"-({render(prev.rhs)})"
        else:                  calc = None
        if calc:
            # Run the computed-text through _polish to fix "- -k" → "+ k" etc.
            lines.append(_polish(f"{render(cur.lhs)} = {calc} = {render(cur.rhs)}"))
        else:
            lines.append(render_eq(cur))
    return "\n".join(lines)


def _safe(eq: Eq) -> bool:
    s = render_eq(eq)
    if len(s) > 120 or "zoo" in s or "nan" in s or "I" in s:
        return False
    return True


# ── expand-identity — sympy-native, no string concat ─────────────────────

def gen_expand_identity(rng: random.Random) -> dict | None:
    """Generate an expansion problem with a true intermediate FOIL step.

    Shapes:
      distrib_single:  a(x + b)        → ax + ab
      binomial:        (x + a)(x + b)  → x² + ax + bx + ab → x² + (a+b)x + ab
      square:          (x + a)²        → x² + 2ax + a²
      diff_squares:    (x + a)(x - a)  → x² - ax + ax - a² → x² - a²
      scalar_times:    c(x + a)(x + b) → c(x² + ax + bx + ab) → cx² + …
    """
    shape = rng.choice(["distrib", "binomial", "square", "diff_squares", "scalar"])
    var_name = rng.choice(_VARS)
    x = Symbol(var_name)

    if shape == "distrib":
        # a(x + b)
        a = _nonzero(rng, 2, 10)
        b = _nonzero(rng, -10, 10)
        lhs = Mul(a, Add(x, b, evaluate=False), evaluate=False)
        # mid: ax + ab (constant product pre-evaluated)
        mid = Add(Mul(a, x, evaluate=False), Integer(a * b), evaluate=False)
        end = expand(a * (x + b))

    elif shape == "binomial":
        a = _nonzero(rng, -10, 10)
        b = _nonzero(rng, -10, 10)
        if a + b == 0:  # avoid trivial diff-squares
            return None
        lhs = Mul(Add(x, a, evaluate=False), Add(x, b, evaluate=False), evaluate=False)
        # mid: x² + ax + bx + ab  (ab pre-computed per textbook convention)
        mid = Add(Pow(x, 2),
                  Mul(a, x, evaluate=False),
                  Mul(b, x, evaluate=False),
                  Integer(a * b),
                  evaluate=False)
        end = expand((x + a) * (x + b))

    elif shape == "square":
        a = _nonzero(rng, -10, 10)
        lhs = Pow(Add(x, a, evaluate=False), 2)
        # mid: (x+a)(x+a) — shows the FOIL setup
        mid_paren = Mul(Add(x, a, evaluate=False), Add(x, a, evaluate=False), evaluate=False)
        # then expand inline
        end = expand((x + a) ** 2)
        return {
            "text": "\n".join([render(lhs), render(mid_paren), render(end)]),
            "_kind": "expand", "_style": "square", "_shape": shape,
        }

    elif shape == "diff_squares":
        a = _nonzero(rng, 2, 10)
        lhs = Mul(Add(x, a, evaluate=False), Add(x, -a, evaluate=False), evaluate=False)
        # mid: x² + ax - ax - a² (constant term pre-computed)
        mid = Add(Pow(x, 2),
                  Mul(a, x, evaluate=False),
                  Mul(-a, x, evaluate=False),
                  Integer(-(a * a)),
                  evaluate=False)
        end = expand((x + a) * (x - a))

    elif shape == "scalar":
        c = _nonzero(rng, 2, 7)
        a = _nonzero(rng, -8, 8)
        b = _nonzero(rng, -8, 8)
        if a + b == 0:
            return None
        inner = expand((x + a) * (x + b))
        lhs = Mul(c, Add(x + a, evaluate=False), Add(x + b, evaluate=False), evaluate=False)
        mid = Mul(c, inner, evaluate=False)
        end = expand(c * (x + a) * (x + b))

    # validate: all three lines distinct and meaningful
    a_s, b_s, c_s = render(lhs), render(mid), render(end)
    if a_s == b_s or b_s == c_s or a_s == c_s:
        return None  # degenerate (e.g. distrib where mid == end)
    style = rng.choice(["vertical", "arrow", "intro"])
    if style == "vertical":
        text = "\n".join([a_s, "= " + b_s, "= " + c_s])
    elif style == "arrow":
        text = f"{a_s} = {b_s} = {c_s}"
    else:
        text = f"Expand: {a_s}\n= {b_s}\n= {c_s}"
    return {"text": text, "_kind": "expand", "_style": style, "_shape": shape}


# ── factor-identity — reverse of expand ──────────────────────────────────

def gen_factor_identity(rng: random.Random) -> dict | None:
    """Generate a factoring problem: expanded polynomial → factored form."""
    shape = rng.choice(["common", "binomial", "square", "diff_squares"])
    var_name = rng.choice(_VARS)
    x = Symbol(var_name)
    if shape == "common":
        # ax + ab → a(x + b)
        a = _nonzero(rng, 2, 10)
        b = _nonzero(rng, -10, 10)
        expanded = Add(Mul(a, x, evaluate=False), Integer(a * b), evaluate=False)
        factored = factor(a * x + a * b)
    elif shape == "binomial":
        # x² + (a+b)x + ab → (x+a)(x+b)
        a = _nonzero(rng, -8, 8)
        b = _nonzero(rng, -8, 8)
        if a + b == 0 or a == b:
            return None
        expanded = expand((x + a) * (x + b))
        factored = factor(expanded)
    elif shape == "square":
        a = _nonzero(rng, -8, 8)
        expanded = expand((x + a) ** 2)
        factored = factor(expanded)
    elif shape == "diff_squares":
        a = _nonzero(rng, 2, 10)
        expanded = expand((x + a) * (x - a))
        factored = factor(expanded)
    a_s, b_s = render(expanded), render(factored)
    if a_s == b_s:
        return None
    style = rng.choice(["arrow", "intro"])
    if style == "arrow":
        text = f"{a_s} = {b_s}"
    else:
        text = f"Factor: {a_s} = {b_s}"
    return {"text": text, "_kind": "factor", "_style": style, "_shape": shape}


# ── collect-like-terms — actually show unsimplified middle ───────────────

def gen_collect_terms(rng: random.Random) -> dict | None:
    """Build an expression with redundant `kx + jx + …` terms, show
    collection: `kx + jx + l = (k+j)x + l`.
    """
    var_name = rng.choice(_VARS)
    x = Symbol(var_name)
    n_xterms = rng.randint(3, 5)
    n_consts = rng.randint(1, 3)
    # Coef==±1 should render as just ±x (no leading 1)
    x_coefs = [rng.choice([-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8]) for _ in range(n_xterms)]
    consts  = [rng.choice([-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9]) for _ in range(n_consts)]
    sum_x = sum(x_coefs); sum_c = sum(consts)
    if sum_x == 0 and sum_c == 0:
        return None
    # Build un-collected: render the SAME shuffled order in both `raw` and
    # the explicit-grouping middle (so the chain reads consistently).
    items = [("x", c) for c in x_coefs] + [("c", k) for k in consts]
    rng.shuffle(items)
    raw_terms = [(Mul(c, x, evaluate=False) if k == "x" else Integer(c))
                 for k, c in items]
    raw = Add(*raw_terms, evaluate=False)
    # Build the grouped middle from the SAME shuffle so signs/order match.
    grouped_xs = [c for k, c in items if k == "x"]
    grouped_cs = [c for k, c in items if k == "c"]
    if not grouped_xs:
        return None  # no x-terms after dedup; skip
    def _group(coefs):
        return " + ".join(str(c) for c in coefs).replace("+ -", "- ")
    if grouped_cs:
        mid_text = f"({_group(grouped_xs)}){var_name} + ({_group(grouped_cs)})"
    else:
        mid_text = f"({_group(grouped_xs)}){var_name}"
    collected = sum_x * x + sum_c
    a_s, c_s = render(raw), render(collected)
    if a_s == c_s:
        return None
    style = rng.choice(["vertical", "arrow"])
    sep = " = " if style == "arrow" else "\n= "
    text = f"{a_s}{sep}{mid_text}{sep}{c_s}"
    return {"text": text, "_kind": "collect", "_style": style}


# ── driver ────────────────────────────────────────────────────────────────

def gen(rng: random.Random, solve_frac=0.65, difficulty: str = "mixed"):
    """`difficulty` ∈ {easy, medium, hard, mixed}.

    `mixed` picks one of easy/medium/hard uniformly per record — yields a
    curriculum-blended dataset useful for SFT signal across difficulties.
    """
    if difficulty == "mixed":
        preset_name = rng.choice(["easy", "medium", "hard"])
    else:
        preset_name = difficulty
    r = rng.random()
    if r < solve_frac:
        return gen_one(rng, preset=preset_name)
    elif r < solve_frac + (1 - solve_frac) * 0.5:
        return gen_expand_identity(rng)
    elif r < solve_frac + (1 - solve_frac) * 0.8:
        return gen_factor_identity(rng)
    else:
        return gen_collect_terms(rng)


def _worker(args_tuple) -> tuple[list[str], int, Counter]:
    """Generate `n` records starting from `seed`. Returns (texts, n_rejected, kinds)."""
    worker_id, n, seed, solve_frac, difficulty, word_frac = args_tuple
    rng = random.Random(seed)
    texts = []
    n_rej = 0
    kinds = Counter()
    while len(texts) < n:
        r = gen(rng, solve_frac=solve_frac, difficulty=difficulty)
        if r is None:
            n_rej += 1
            continue
        kinds[r["_kind"]] += 1
        text = r["text"]
        if word_frac > 0:
            text = wordify_text(text, rng, p_word=word_frac)
        # Append canonical `#### N` answer marker AFTER wordify, so the
        # marker line always stays as digits even when word_frac > 0.
        # Only solve chains have a single numeric answer; expand/factor/
        # collect produce identities and skip the marker.
        if r.get("answer") is not None:
            text = text + "\n#### " + r["answer"]
        texts.append(text)
    return texts, n_rej, kinds


def main():
    import os, time, multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solve-frac", type=float, default=0.65)
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"],
                    default="mixed",
                    help="Preset for op selection / coef ranges / depth. "
                         "`mixed` blends easy/medium/hard uniformly per record.")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto (cpu-2, capped 64); 1 = sequential")
    ap.add_argument("--word-frac", type=float, default=0.15,
                    help="Per-number probability of replacing a digit "
                         "(integer or X/Y fraction) with its language "
                         "word form. Default 0.15 produces intra-chain "
                         "variation without overwhelming the digit form; "
                         "0 = pure digits.")
    ap.add_argument("--lang", choices=["eo", "da"], default="eo",
                    help="Language for wordify_text number rendering.")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    global wordify_text
    if args.lang == "eo":
        from esperanto_lm.eo_numbers import wordify_text as _wt
    else:
        from esperanto_lm.da_numbers import wordify_text as _wt
    wordify_text = _wt

    if args.workers == 0:
        n_workers = min(64, max(1, (os.cpu_count() or 4) - 2))
    else:
        n_workers = max(1, args.workers)

    # Cap per-task chunk at 100k rows so parent gets frequent progress
    # updates + streaming file writes. For n >> 4*100k, this creates many
    # more tasks than workers; pool.imap load-balances them naturally.
    max_chunk = 100_000
    chunk = min(max_chunk, (args.n + n_workers - 1) // n_workers)
    tasks = []
    remaining = args.n
    w = 0
    while remaining > 0:
        take = min(chunk, remaining)
        tasks.append((w, take, args.seed * 10_000 + w, args.solve_frac, args.difficulty, args.word_frac))
        remaining -= take
        w += 1
    print(f"workers: {n_workers}  chunk: {chunk:,}  tasks: {len(tasks)}  target: {args.n:,}  difficulty: {args.difficulty}  word_frac: {args.word_frac}", flush=True)

    kinds = Counter()
    n_emitted = 0
    n_rej = 0
    t0 = time.time()

    with args.out.open("w") as f:
        if n_workers == 1:
            texts, rej, kc = _worker(tasks[0])
            for t in texts:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            n_emitted += len(texts); n_rej += rej; kinds.update(kc)
        else:
            with mp.Pool(n_workers) as pool:
                for texts, rej, kc in pool.imap_unordered(_worker, tasks):
                    for t in texts:
                        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                    n_emitted += len(texts); n_rej += rej; kinds.update(kc)
                    dt = time.time() - t0
                    rate = n_emitted / dt if dt > 0 else 0
                    print(f"  [{n_emitted:,}/{args.n:,}] {rate:.0f} rec/s  rej={n_rej:,}", flush=True)

    print(f"\nDONE  emitted={n_emitted:,}  rejected={n_rej:,}  kinds={dict(kinds)}  wall={(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
