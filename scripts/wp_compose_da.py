"""Compositional word-problem generator (Danish port of wp_compose.py).

Danish port of the Esperanto wp_compose.py:
  * All EN-mirror code removed — Danish-only output.
  * Noun morphology adapted for Danish gender (en/et) and definite suffixes.
  * Danish has NO accusative case (unlike EO) — render_qty is simpler.
  * Prose templates are translated to idiomatic Danish (~10-15 variants per
    Op role). Recipe LOGIC (how numbers get chained) is identical to EO.

Output schema (unchanged from EO):
    {question, answer, chain_lines, final, recipe, n_steps, direction}

Usage:
    uv run python scripts/wp_compose_da.py
    uv run python scripts/wp_compose_da.py --count 200 > sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sympy as sp


# ─── Vocab ──────────────────────────────────────────────────────────────────

NAMES = [
    "Anna", "Peter", "Marie", "Anders", "Sofie", "Lars", "Emma", "Mikkel",
    "Camilla", "Rasmus", "Trine", "Jonas", "Louise", "Frederik", "Signe",
    "Simon", "Ida", "Christian", "Katrine", "Magnus", "Ellen", "Oskar",
    "Karla", "Villads", "Nynne", "Aksel", "Astrid", "Emil", "Malou", "Bertil",
    "Merete", "Ove", "Sanne", "Yrsa", "Bjørn", "Freja", "Ivan", "Julie",
    "Kirsten", "Torben",
]

# Names ending in s/x/z take an apostrophe instead of extra 's' for possessives.
_S_ENDING = set(n for n in NAMES if n[-1].lower() in "sxz")


def poss(name: str) -> str:
    """Danish possessive: 'Peters' / 'Jens''."""
    return f"{name}'" if name in _S_ENDING else f"{name}s"


# Scenario framings sampled at question-open time. Not every recipe uses them,
# but recipes that opt in inject one of these leading phrases.
SCENARIO_FRAMES = [
    "I 2015,",  "I løbet af skoleåret,",  "En dag,",  "Efter skole,",
    "På markedet,",  "I sommers,",  "Før festen,",  "I morges,",
    "Sidste uge,",  "Til festen,",  "Efter måltidet,",  "I butikken,",
]

# Danish noun tuple: (indef_sg, gender, def_sg, indef_pl, def_pl)
# Gender is "en" or "et" — used as the article for `n == 1`.
Noun = tuple[str, str, str, str, str]

CHILDLIKE_NOUNS: list[Noun] = [
    ("barn",       "et", "barnet",       "børn",        "børnene"),
    ("dreng",      "en", "drengen",      "drenge",      "drengene"),
    ("pige",       "en", "pigen",        "piger",       "pigerne"),
    ("elev",       "en", "eleven",       "elever",      "eleverne"),
    ("kunde",      "en", "kunden",       "kunder",      "kunderne"),
    ("gæst",       "en", "gæsten",       "gæster",      "gæsterne"),
    ("besøgende",  "en", "besøgende",    "besøgende",   "besøgende"),
]

GROUPING_NOUNS: list[Noun] = [
    ("gruppe",     "en", "gruppen",      "grupper",     "grupperne"),
    ("hold",       "et", "holdet",       "hold",        "holdene"),
    ("klasse",     "en", "klassen",      "klasser",     "klasserne"),
    ("bord",       "et", "bordet",       "borde",       "bordene"),
    ("bus",        "en", "bussen",       "busser",      "busserne"),
    ("værelse",    "et", "værelset",     "værelser",    "værelserne"),
    ("kasse",      "en", "kassen",       "kasser",      "kasserne"),
]

OBJECT_NOUNS: list[Noun] = [
    ("bog",        "en", "bogen",        "bøger",       "bøgerne"),
    ("blyant",     "en", "blyanten",     "blyanter",    "blyanterne"),
    ("æble",       "et", "æblet",        "æbler",       "æblerne"),
    ("legetøj",    "et", "legetøjet",    "legetøj",     "legetøjene"),
    ("blomst",     "en", "blomsten",     "blomster",    "blomsterne"),
    ("bolle",      "en", "bollen",       "boller",      "bollerne"),
    ("billet",     "en", "billetten",    "billetter",   "billetterne"),
    ("hæfte",      "et", "hæftet",       "hæfter",      "hæfterne"),
    ("pen",        "en", "pennen",       "penne",       "pennene"),
    ("småkage",    "en", "småkagen",     "småkager",    "småkagerne"),
    ("kugle",      "en", "kuglen",       "kugler",      "kuglerne"),
    ("æg",         "et", "ægget",        "æg",          "æggene"),
    ("stjerne",    "en", "stjernen",     "stjerner",    "stjernerne"),
    ("chokolade",  "en", "chokoladen",   "chokolader",  "chokoladerne"),
    ("kort",       "et", "kortet",       "kort",        "kortene"),
    ("konvolut",   "en", "konvolutten",  "konvolutter", "konvolutterne"),
]

# Currency
KRONE: Noun = ("krone", "en", "kronen", "kroner", "kronerne")

SHOP_ITEMS: list[Noun] = [
    ("cykel",      "en", "cyklen",       "cykler",      "cyklerne"),
    ("skjorte",    "en", "skjorten",     "skjorter",    "skjorterne"),
    ("bog",        "en", "bogen",        "bøger",       "bøgerne"),
    ("computer",   "en", "computeren",   "computere",   "computerne"),
]

COUNT_ITEMS: list[Noun] = [
    ("elev",       "en", "eleven",       "elever",      "eleverne"),
    ("studerende", "en", "studerende",   "studerende",  "studerende"),
    ("barn",       "et", "barnet",       "børn",        "børnene"),
]

# Test subjects — used by average recipe
SUBJECT_NOUNS: list[Noun] = [
    ("matematik",  "en", "matematikken", "matematikker", "matematikkerne"),
    ("historie",   "en", "historien",    "historier",   "historierne"),
    ("kemi",       "en", "kemien",       "kemier",      "kemierne"),
    ("biologi",    "en", "biologien",    "biologier",   "biologierne"),
]

# Human-readable subject name for use inline (`i matematik` etc.)
SUBJECT_INLINE = {
    "matematik": "matematik",
    "historie":  "historie",
    "kemi":      "kemi",
    "biologi":   "biologi",
}


# ─── Ctx ────────────────────────────────────────────────────────────────────

@dataclass
class Var:
    value: float
    noun: Noun | None = None


@dataclass
class Ctx:
    rng: random.Random
    vars: dict[str, Var] = field(default_factory=dict)
    chain: list[str] = field(default_factory=list)
    prose: list[str] = field(default_factory=list)
    applied_ops: list["Op"] = field(default_factory=list)
    protagonist: str = ""

    @classmethod
    def new(cls, rng: random.Random) -> "Ctx":
        c = cls(rng=rng)
        c.protagonist = rng.choice(NAMES)
        return c

    def bind(self, name: str, value: float, noun: Noun | None = None) -> Var:
        v = Var(value=value, noun=noun)
        self.vars[name] = v
        return v

    def get(self, name: str) -> Var:
        return self.vars[name]

    def n(self, name: str) -> float:
        return self.vars[name].value

    def render(self, question: str, final_var: str) -> dict:
        v = self.vars[final_var].value
        final_str = str(int(v)) if v == int(v) else str(v)
        answer = " ".join(self.prose) + f" #### {final_str}"
        return {
            "question": question,
            "answer": answer.strip(),
            "chain_lines": self.chain,
            "final": final_str,
        }

    def render_reverse(
        self,
        forward_prose: str,
        forward_final_var: str,
        ask_var: str,
        closer: str,
        recipe_name: str = "reverse",
    ) -> dict:
        """Reverse-frame the forward problem — see wp_compose.py docstring."""
        answer_val = self.vars[ask_var].value
        ans_str = str(int(answer_val)) if answer_val == int(answer_val) else str(answer_val)

        path_var: str = ask_var
        path_sides: list[str | None] = []
        for op in self.applied_ops:
            if op.kind == "frac":
                if op.lhs == path_var:
                    path_sides.append("lhs")
                    path_var = op.out
                else:
                    path_sides.append(None)
            else:
                if op.lhs == path_var:
                    path_sides.append("lhs")
                    path_var = op.out
                elif op.rhs == path_var:
                    path_sides.append("rhs")
                    path_var = op.out
                else:
                    path_sides.append(None)

        assert path_var == forward_final_var, (
            f"ask_var {ask_var} does not chain into {forward_final_var}: "
            f"trace ended at {path_var}"
        )

        reverse_chain: list[str] = []
        reverse_prose: list[str] = []
        known_val: float = self.vars[forward_final_var].value

        for op, side in zip(reversed(self.applied_ops), reversed(path_sides)):
            if side is None:
                continue
            known_side = "rhs" if side == "lhs" else "lhs"
            if op.kind == "frac":
                unknown = op.reverse_step(known_val, known_side, None)
                reverse_chain.append(
                    op.reverse_chain_line(known_val, known_side, None, unknown))
                variants = REVERSE_STEP_PROSE[op.kind]
                idx = self.rng.randrange(len(variants))
                fmt_kwargs = dict(a=fmt_num(known_val), c=fmt_num(unknown),
                                  n=op.num, d=op.denom)
                reverse_prose.append(variants[idx].format(**fmt_kwargs))
            else:
                other_side_var = op.rhs if known_side == "rhs" else op.lhs
                other_side_val = self.vars[other_side_var].value
                unknown = op.reverse_step(known_val, known_side, other_side_val)
                reverse_chain.append(
                    op.reverse_chain_line(known_val, known_side, other_side_val, unknown))
                variants = REVERSE_STEP_PROSE[op.kind]
                idx = self.rng.randrange(len(variants))
                fmt_kwargs = dict(a=fmt_num(known_val),
                                  b=fmt_num(other_side_val),
                                  c=fmt_num(unknown))
                reverse_prose.append(variants[idx].format(**fmt_kwargs))
            known_val = unknown

        assert abs(known_val - answer_val) < 1e-9, (
            f"reverse chain didn't recover: known={known_val} answer={answer_val}"
        )

        question = f"{forward_prose} {closer}"
        answer = " ".join(reverse_prose) + f" #### {ans_str}"
        return {
            "question": question,
            "answer": answer.strip(),
            "chain_lines": reverse_chain,
            "final": ans_str,
        }


# Reverse-step prose per Op kind. Placeholders:
#   {a} = known output value entering this reverse step
#   {b} = other-side value (side input)
#   {c} = computed unknown
#   For frac: also {n}=numerator, {d}=denominator
REVERSE_STEP_PROSE: dict[str, list[str]] = {
    "mul": [
        "Vi går baglæns gennem multiplikationen: {a} / {b} = {c}.",
        "Vi dividerer for at gå tilbage: {a} / {b} = {c}.",
        "Det oprindelige tal er {a} / {b} = {c}.",
        "Altså var den oprindelige mængde {a} / {b} = {c}.",
        "Ved at dividere baglæns: {a} / {b} = {c}.",
        "Vi gør multiplikationen om: {a} / {b} = {c}.",
        "Vend multiplikationen om ved division: {a} / {b} = {c}.",
        "Den foregående værdi var {a} / {b} = {c}.",
        "Baglæns ved division: {a} / {b} = {c}.",
        "Altså var input til multiplikationen {c}, da {a} / {b} = {c}.",
    ],
    "add": [
        "Vi trækker det tillagte fra: {a} - {b} = {c}.",
        "Den oprindelige værdi var {a} - {b} = {c}.",
        "Altså før tillægget var det {a} - {b} = {c}.",
        "Baglæns gennem addition: {a} - {b} = {c}.",
        "Vi gør additionen om: {a} - {b} = {c}.",
        "Ved at trække det tillagte fra: {a} - {b} = {c}.",
        "Vend additionen om: {a} - {b} = {c}.",
        "Den foregående værdi var {a} - {b} = {c}.",
        "Baglæns ved subtraktion: {a} - {b} = {c}.",
    ],
    "sub": [
        "Vi lægger til for at gå tilbage: {a} + {b} = {c}.",
        "Altså før subtraktionen var det {a} + {b} = {c}.",
        "Den oprindelige værdi var {a} + {b} = {c}.",
        "Baglæns gennem subtraktion: {a} + {b} = {c}.",
        "Vi gør subtraktionen om: {a} + {b} = {c}.",
        "Læg til for at vende om: {a} + {b} = {c}.",
        "Genskab det fjernede: {a} + {b} = {c}.",
        "Den foregående værdi var {a} + {b} = {c}.",
        "Baglæns ved addition: {a} + {b} = {c}.",
    ],
    "div": [
        "Vi ganger for at gå tilbage: {a} * {b} = {c}.",
        "Det oprindelige tal var {a} * {b} = {c}.",
        "Altså før divisionen var det {a} * {b} = {c}.",
        "Baglæns gennem divisionen: {a} * {b} = {c}.",
        "Vi gør divisionen om: {a} * {b} = {c}.",
        "Gang for at vende om: {a} * {b} = {c}.",
        "Den foregående værdi var {a} * {b} = {c}.",
        "Baglæns ved multiplikation: {a} * {b} = {c}.",
        "Altså var input til divisionen {a} * {b} = {c}.",
    ],
    "frac": [
        "Vi går baglæns gennem brøken: {a} * {d} / {n} = {c}.",
        "Den oprindelige basis var {a} * {d} / {n} = {c}.",
        "Altså var den oprindelige basis {a} * {d} / {n} = {c}.",
        "Vi gør brøken om: {a} * {d} / {n} = {c}.",
        "Vend {n}/{d} om: at gange med {d}/{n} giver {c}.",
        "Baglæns gennem brøken: {a} * {d} / {n} = {c}.",
        "Den foregående basis var {a} * {d} / {n} = {c}.",
        "Ved at vende {n}/{d} om: {a} * {d} / {n} = {c}.",
    ],
    "pct": [
        "Vi går baglæns gennem procentregningen: {a} * 100 / {b} = {c}.",
        "Den oprindelige basis var {a} * 100 / {b} = {c}.",
        "Altså var den oprindelige basis {a} * 100 / {b} = {c}.",
        "Vi gør procentregningen om: {a} * 100 / {b} = {c}.",
        "Vend {b}% om: {a} * 100 / {b} = {c}.",
        "Baglæns gennem procenten: {a} * 100 / {b} = {c}.",
        "Den foregående basis var {a} * 100 / {b} = {c}.",
        "Division med {b}% giver {a} * 100 / {b} = {c}.",
    ],
}


# ─── Morphology helpers ─────────────────────────────────────────────────────

def maybe_frame(rng: random.Random, p: float = 0.35) -> str:
    """Optionally prepend a scenario frame. Returned as leading str or empty."""
    if rng.random() < p:
        return rng.choice(SCENARIO_FRAMES) + " "
    return ""


def hver_of(noun: Noun) -> str:
    """Gender-agreeing 'each' — 'hvert' for et, 'hver' for en."""
    return "hvert" if noun[1] == "et" else "hver"


def render_qty(n: int, noun: Noun) -> str:
    """Danish quantity phrasing with gender agreement for `n == 1`.

    render_qty(1, ("bog","en","bogen","bøger","bøgerne"))  → "en bog"
    render_qty(1, ("barn","et","barnet","børn","børnene"))  → "et barn"
    render_qty(5, ("bog", ...))                             → "5 bøger"
    """
    indef_sg, gender, _def_sg, indef_pl, _def_pl = noun
    if n == 1:
        return f"{gender} {indef_sg}"
    return f"{n} {indef_pl}"


def fmt_num(x: float) -> str:
    if x == int(x):
        return str(int(x))
    return f"{x:g}"


# ─── Ops ────────────────────────────────────────────────────────────────────

class Op:
    """Base class for math ops. See wp_compose.py for full docs."""
    kind: str = ""

    def __init__(self, lhs: str, rhs: str, out: str):
        self.lhs = lhs
        self.rhs = rhs
        self.out = out

    def apply(self, ctx: Ctx) -> None:
        raise NotImplementedError

    def _chain_line(self, ctx: Ctx, a: float, sym: str, b: float, c: float) -> None:
        ctx.chain.append(f"{fmt_num(a)} {sym} {fmt_num(b)} = {fmt_num(c)}")

    @staticmethod
    def _pick_emit(ctx: Ctx, lib: dict, role: str, **fmt_kwargs) -> None:
        variants = lib[role]
        idx = ctx.rng.randrange(len(variants))
        ctx.prose.append(variants[idx].format(**fmt_kwargs))


class Mul(Op):
    kind = "mul"
    def apply(self, ctx: Ctx) -> None:
        a, b = ctx.n(self.lhs), ctx.n(self.rhs)
        c = a * b
        ctx.bind(self.out, c, noun=ctx.get(self.lhs).noun)
        self._chain_line(ctx, a, "*", b, c)
        role = "first" if not ctx.prose else "chained"
        self._pick_emit(ctx, _MUL_PROSE, role,
                        a=fmt_num(a), b=fmt_num(b), c=fmt_num(c))
        ctx.applied_ops.append(self)

    def reverse_step(self, out_val, known_side, known_val):
        assert known_val != 0
        return out_val / known_val

    def reverse_chain_line(self, out_val, known_side, known_val, unknown):
        return f"{fmt_num(out_val)} / {fmt_num(known_val)} = {fmt_num(unknown)}"


class Add(Op):
    kind = "add"
    def apply(self, ctx: Ctx) -> None:
        a, b = ctx.n(self.lhs), ctx.n(self.rhs)
        c = a + b
        ctx.bind(self.out, c, noun=ctx.get(self.lhs).noun)
        self._chain_line(ctx, a, "+", b, c)
        role = "first" if not ctx.prose else "chained"
        self._pick_emit(ctx, _ADD_PROSE, role,
                        a=fmt_num(a), b=fmt_num(b), c=fmt_num(c))
        ctx.applied_ops.append(self)

    def reverse_step(self, out_val, known_side, known_val):
        return out_val - known_val

    def reverse_chain_line(self, out_val, known_side, known_val, unknown):
        return f"{fmt_num(out_val)} - {fmt_num(known_val)} = {fmt_num(unknown)}"


class Sub(Op):
    kind = "sub"
    def apply(self, ctx: Ctx) -> None:
        a, b = ctx.n(self.lhs), ctx.n(self.rhs)
        c = a - b
        ctx.bind(self.out, c, noun=ctx.get(self.lhs).noun)
        self._chain_line(ctx, a, "-", b, c)
        role = "first" if not ctx.prose else "chained"
        self._pick_emit(ctx, _SUB_PROSE, role,
                        a=fmt_num(a), b=fmt_num(b), c=fmt_num(c))
        ctx.applied_ops.append(self)

    def reverse_step(self, out_val, known_side, known_val):
        return known_val - out_val if known_side == "lhs" else out_val + known_val

    def reverse_chain_line(self, out_val, known_side, known_val, unknown):
        if known_side == "lhs":
            return f"{fmt_num(known_val)} - {fmt_num(out_val)} = {fmt_num(unknown)}"
        return f"{fmt_num(out_val)} + {fmt_num(known_val)} = {fmt_num(unknown)}"


class Div(Op):
    kind = "div"
    def apply(self, ctx: Ctx) -> None:
        a, b = ctx.n(self.lhs), ctx.n(self.rhs)
        assert a % b == 0, f"Div {a}/{b} not integer — caller must resample."
        c = a / b
        ctx.bind(self.out, c, noun=ctx.get(self.lhs).noun)
        self._chain_line(ctx, a, "/", b, c)
        role = "first" if not ctx.prose else "chained"
        self._pick_emit(ctx, _DIV_PROSE, role,
                        a=fmt_num(a), b=fmt_num(b), c=fmt_num(c))
        ctx.applied_ops.append(self)

    def reverse_step(self, out_val, known_side, known_val):
        return known_val / out_val if known_side == "lhs" else out_val * known_val

    def reverse_chain_line(self, out_val, known_side, known_val, unknown):
        if known_side == "lhs":
            return f"{fmt_num(known_val)} / {fmt_num(out_val)} = {fmt_num(unknown)}"
        return f"{fmt_num(out_val)} * {fmt_num(known_val)} = {fmt_num(unknown)}"


class Frac(Op):
    """(num/denom) * base → result. Rejects non-integer results at apply-time."""
    kind = "frac"
    def __init__(self, base_var: str, num: int, denom: int, out: str):
        super().__init__(base_var, "", out)
        self.num = num
        self.denom = denom

    def apply(self, ctx: Ctx) -> None:
        base = ctx.n(self.lhs)
        assert (base * self.num) % self.denom == 0, \
            f"Frac {self.num}/{self.denom} of {base} not integer"
        result = base * self.num // self.denom
        ctx.bind(self.out, result, noun=ctx.get(self.lhs).noun)
        ctx.chain.append(f"{self.num}/{self.denom} * {fmt_num(base)} = {fmt_num(result)}")
        role = "first" if not ctx.prose else "chained"
        self._pick_emit(ctx, _FRAC_PROSE, role,
                        n=self.num, d=self.denom, b=fmt_num(base), r=fmt_num(result))
        ctx.applied_ops.append(self)

    def reverse_step(self, out_val, known_side, known_val=None):
        assert self.num != 0
        return out_val * self.denom / self.num

    def reverse_chain_line(self, out_val, known_side, known_val, unknown):
        return f"{fmt_num(out_val)} * {self.denom} / {self.num} = {fmt_num(unknown)}"


class Avg(Op):
    """Average of a list of variables. Emits chain: "sum / N = avg"."""
    kind = "avg"
    def __init__(self, value_vars: list[str], out: str):
        super().__init__("", "", out)
        self.value_vars = value_vars

    def apply(self, ctx: Ctx) -> None:
        vals = [ctx.n(v) for v in self.value_vars]
        total = sum(vals)
        n = len(vals)
        assert total % n == 0, f"Avg of {vals} not integer"
        avg = total // n
        ctx.bind(self.out, avg,
                 noun=ctx.get(self.value_vars[0]).noun if self.value_vars else None)
        sum_str = " + ".join(fmt_num(v) for v in vals)
        ctx.chain.append(f"{sum_str} = {fmt_num(total)}")
        ctx.chain.append(f"{fmt_num(total)} / {n} = {fmt_num(avg)}")
        role = "first" if not ctx.prose else "chained"
        self._pick_emit(ctx, _AVG_PROSE, role,
                        vals=sum_str, t=fmt_num(total), n=n, a=fmt_num(avg))
        ctx.applied_ops.append(self)


class Pct(Op):
    """X% of base → amount, in one of 3 styles: direct | decimal | multiplier."""
    kind = "pct"
    def __init__(self, pct_var: str, base_var: str, out: str, style: str = "direct"):
        super().__init__(pct_var, base_var, out)
        self.style = style

    def apply(self, ctx: Ctx) -> None:
        pct = int(ctx.n(self.lhs))
        base = ctx.n(self.rhs)
        assert (base * pct) % 100 == 0, f"pct: {pct}% of {base} not integer"
        amount = base * pct // 100
        ctx.bind(self.out, amount, noun=ctx.get(self.rhs).noun)

        if self.style == "decimal":
            dec = pct / 100
            dec_str = fmt_num(dec)
            ctx.chain.append(f"{pct} / 100 = {dec_str}")
            ctx.chain.append(f"{dec_str} * {fmt_num(base)} = {fmt_num(amount)}")
            role = "first" if not ctx.prose else "chained"
            self._pick_emit(ctx, _PCT_DECIMAL_PROSE, role,
                            p=pct, d=dec_str, b=fmt_num(base), a=fmt_num(amount))
        else:
            ctx.chain.append(f"{pct} / 100 * {fmt_num(base)} = {fmt_num(amount)}")
            role = "first" if not ctx.prose else "chained"
            self._pick_emit(ctx, _PCT_DIRECT_PROSE, role,
                            p=pct, b=fmt_num(base), a=fmt_num(amount))
        ctx.applied_ops.append(self)

    def reverse_step(self, out_val, known_side, known_val):
        assert known_val != 0
        return out_val * 100 / known_val

    def reverse_chain_line(self, out_val, known_side, known_val, unknown):
        return (f"{fmt_num(out_val)} * 100 / {fmt_num(known_val)} "
                f"= {fmt_num(unknown)}")


class LinearSolve(Op):
    """Solve `sum_coef * x + const = target` for x. See wp_compose.py docs."""
    kind = "linsolve"
    def __init__(self, sum_coef_var: str, const_var: str, target_var: str,
                 out: str, var_name: str = "x", lhs_shape: str | None = None):
        super().__init__(sum_coef_var, const_var, out)
        self.target_var = target_var
        self.var_name = var_name
        self.lhs_shape = lhs_shape

    def apply(self, ctx: Ctx) -> None:
        c = int(ctx.n(self.lhs))
        b = int(ctx.n(self.rhs))
        t = int(ctx.n(self.target_var))
        assert c != 0, "LinearSolve: coef cannot be zero"
        num = t - b
        assert num % c == 0, f"LinearSolve: ({t}-{b})/{c} not integer"

        sv = sp.Symbol(self.var_name)
        eq = sp.Eq(c * sv + b, t)
        sol = sp.solve(eq, sv)
        assert len(sol) == 1 and int(sol[0]) == num // c, \
            f"LinearSolve: sympy disagrees ({sol} vs {num // c})"
        x = num // c
        ctx.bind(self.out, x)

        v = self.var_name

        def _combined() -> str:
            if b == 0:
                return f"{c}{v}"
            return f"{c}{v} + {b}" if b > 0 else f"{c}{v} - {-b}"

        if self.lhs_shape is not None:
            ctx.chain.append(f"{self.lhs_shape} = {t}")
        ctx.chain.append(f"{_combined()} = {t}")
        if b != 0:
            op_str = "-" if b > 0 else "+"
            ctx.chain.append(f"{c}{v} = {t} {op_str} {abs(b)}")
            ctx.chain.append(f"{c}{v} = {num}")
        ctx.chain.append(f"{c}{v} / {c} = {num} / {c}")
        ctx.chain.append(f"{v} = {x}")

        role = "first" if not ctx.prose else "chained"
        eq_str = f"{c}{v} + {b}" if b > 0 else f"{c}{v} - {-b}" if b < 0 else f"{c}{v}"
        self._pick_emit(ctx, _LINSOLVE_PROSE, role,
                        v=v, eq=eq_str, t=t, c=c, num=num, x=x)
        ctx.applied_ops.append(self)


# ─── Prose libraries ────────────────────────────────────────────────────────

_MUL_PROSE = {
    "first": [
        "Der er {a} * {b} = {c} i alt.",
        "{a} gange {b} er lig {c}.",
        "Det samlede antal er {a} * {b} = {c}.",
        "Vi multiplicerer: {a} * {b} = {c}.",
        "{a} pr. {b} giver {a} * {b} = {c}.",
        "Udregning: {a} * {b} = {c}.",
        "Først beregner vi produktet: {a} * {b} = {c}.",
        "Produktet af {a} og {b} er {c}.",
        "Når vi ganger {a} med {b}, får vi {c}.",
        "Vi starter med at gange: {a} * {b} = {c}.",
        "Den indledende udregning: {a} * {b} = {c}.",
        "Vi finder produktet af {a} og {b}: {a} * {b} = {c}.",
        "Ganske enkelt: {a} * {b} = {c}.",
        "Bemærk, at {a} * {b} = {c}.",
        "Resultatet af multiplikationen: {a} * {b} = {c}.",
    ],
    "chained": [
        "Derefter: {a} * {b} = {c}.",
        "Nu ganger vi med {b}: {a} * {b} = {c}.",
        "{a} * {b} = {c}.",
        "Det giver {a} * {b} = {c}.",
        "Ved multiplikation: {a} * {b} = {c}.",
        "Så: {a} * {b} = {c}.",
        "Derfor {a} * {b} = {c}.",
        "Herefter {a} * {b} = {c}.",
        "Nu beregner vi: {a} * {b} = {c}.",
        "Vi ganger med {b}: {a} * {b} = {c}.",
        "Resultat: {a} * {b} = {c}.",
        "Altså {a} * {b} = {c}.",
        "Vi ganger og får {c}.",
    ],
}

_ADD_PROSE = {
    "first": [
        "Vi lægger sammen: {a} + {b} = {c}.",
        "I alt bliver det {a} + {b} = {c}.",
        "Tilsammen: {a} + {b} = {c}.",
        "{a} plus {b} giver {c}.",
        "Summen er {a} + {b} = {c}.",
        "Ved summering: {a} + {b} = {c}.",
        "Addition: {a} + {b} = {c}.",
        "Den samlede værdi er {a} + {b} = {c}.",
        "Udregning af summen: {a} + {b} = {c}.",
        "Lad os summere: {a} + {b} = {c}.",
        "Summen af {a} og {b} er {c}.",
        "Ved at lægge {a} og {b} sammen får vi {c}.",
    ],
    "chained": [
        "Læg {b} mere til: {a} + {b} = {c}.",
        "Nu er der {a} + {b} = {c}.",
        "Tilsammen: {a} + {b} = {c}.",
        "Derfor: {a} + {b} = {c}.",
        "{a} + {b} = {c}.",
        "Derefter summeres: {a} + {b} = {c}.",
        "Så {a} + {b} = {c}.",
        "Det fører til {a} + {b} = {c}.",
        "Nu lægges {b} til: {a} + {b} = {c}.",
        "Den nye sum: {a} + {b} = {c}.",
        "Efter addition: {a} + {b} = {c}.",
        "Samlet: {a} + {b} = {c}.",
        "Så det samlede resultat er {c}.",
    ],
}

_SUB_PROSE = {
    "first": [
        "Vi trækker fra: {a} - {b} = {c}.",
        "Der er {a} - {b} = {c} tilbage.",
        "{a} minus {b} er lig {c}.",
        "Trækker {b} fra {a}: {a} - {b} = {c}.",
        "Forskellen er {a} - {b} = {c}.",
        "Efter subtraktion: {a} - {b} = {c}.",
        "Udregning af forskellen: {a} - {b} = {c}.",
        "Vi fjerner {b}: {a} - {b} = {c}.",
        "Resten er lig {a} - {b} = {c}.",
        "Reduktion: {a} - {b} = {c}.",
    ],
    "chained": [
        "Efter at {b} er væk, er der {a} - {b} = {c} tilbage.",
        "Tilbage: {a} - {b} = {c}.",
        "Ved at trække {b} fra: {a} - {b} = {c}.",
        "{a} - {b} = {c}.",
        "Så der er {a} - {b} = {c} tilbage.",
        "Nu {a} - {b} = {c}.",
        "Så forskellen: {a} - {b} = {c}.",
        "Ved at reducere med {b}: {a} - {b} = {c}.",
        "Efter at have fjernet {b}: {a} - {b} = {c}.",
        "Den resterende værdi er {c}.",
        "Så forskellen er: {a} - {b} = {c}.",
        "Dette efterlader {a} - {b} = {c}.",
    ],
}

_DIV_PROSE = {
    "first": [
        "Vi dividerer: {a} / {b} = {c}.",
        "{a} / {b} = {c}.",
        "Hver del får {a} / {b} = {c}.",
        "Division: {a} / {b} = {c}.",
        "Kvotienten er {a} / {b} = {c}.",
        "Ved at dividere {a} med {b} får vi {c}.",
        "Fordel {a} mellem {b}: {a} / {b} = {c}.",
        "Hver får {a} / {b} = {c}.",
        "Ligeligt fordelt: {a} / {b} = {c}.",
        "Udregning af division: {a} / {b} = {c}.",
        "{a} delt med {b} giver {c}.",
    ],
    "chained": [
        "Ved fordeling mellem {b}: {a} / {b} = {c}.",
        "Hver gruppe får {a} / {b} = {c}.",
        "{a} / {b} = {c}.",
        "Så hver får {a} / {b} = {c}.",
        "Herefter {a} / {b} = {c}.",
        "Derefter dividerer vi: {a} / {b} = {c}.",
        "Ved at fordele {a} over {b}: {a} / {b} = {c}.",
        "Hver del: {a} / {b} = {c}.",
        "Nu dividerer vi med {b}: {a} / {b} = {c}.",
        "Så {a} / {b} = {c}.",
        "Kvotienten er {a} / {b} = {c}.",
    ],
}

_FRAC_PROSE = {
    "first": [
        "Vi finder {n}/{d} af {b}: {n}/{d} * {b} = {r}.",
        "{n}/{d} af {b} er {r}.",
        "Beregn {n}/{d} af {b}: {n}/{d} * {b} = {r}.",
        "{n}/{d}-delen af {b} er lig {r}.",
        "Tag {n}/{d} af {b}: {n}/{d} * {b} = {r}.",
        "Først finder vi {n}/{d}: {n}/{d} * {b} = {r}.",
        "Brøken {n}/{d} af {b} giver {r}.",
        "Vi ganger {b} med {n}/{d}: {n} * {b} / {d} = {r}.",
        "Anvender {n}/{d} på {b}: {r}.",
        "{n}/{d} af {b}: {r}.",
    ],
    "chained": [
        "Derefter {n}/{d} af {b}: {n}/{d} * {b} = {r}.",
        "{n}/{d} af {b} = {r}.",
        "Nu finder vi {n}/{d}: {n}/{d} * {b} = {r}.",
        "Herefter giver brøken {n}/{d} af {b} {r}.",
        "Ud fra det er {n}/{d} af {b} = {r}.",
        "Så {n}/{d} af {b} er lig {r}.",
        "Anvender {n}/{d}: {n}/{d} * {b} = {r}.",
        "Og {n}/{d} af {b} = {r}.",
        "Så {n}/{d} af {b} er {r}.",
    ],
}

_AVG_PROSE = {
    "first": [
        "Sum: {vals} = {t}. Middelværdi: {t} / {n} = {a}.",
        "Læg dem alle sammen: {vals} = {t}. Middel: {t} / {n} = {a}.",
        "Middelværdien er ({vals}) / {n} = {a}.",
        "Sum: {vals} = {t}. Middel: {t} / {n} = {a}.",
        "Summen er {vals} = {t}. Divideret med {n}: {t} / {n} = {a}.",
        "Beregn summen: {vals} = {t}. Så {t} / {n} = {a}.",
        "Først summen ({vals}) = {t}, derefter divideres med {n}: {a}.",
        "Det aritmetiske gennemsnit: ({vals}) / {n} = {a}.",
        "Find middelværdien: ({vals}) / {n} = {a}.",
    ],
    "chained": [
        "Nu middelværdien: ({vals}) / {n} = {a}.",
        "Summen er {t}, så middel = {t} / {n} = {a}.",
        "Derefter beregnes middel: {t} / {n} = {a}.",
        "Summen {t} divideret med {n} giver {a}.",
        "Så middelværdien er {t} / {n} = {a}.",
        "Og gennemsnittet er {t} / {n} = {a}.",
        "Så middel er ({vals}) / {n} = {a}.",
        "Middelværdi = {t} / {n} = {a}.",
    ],
}

_PCT_DIRECT_PROSE = {
    "first": [
        "Vi beregner {p}% af {b}: {p}/100 * {b} = {a}.",
        "{p}% af {b} er {p}/100 * {b} = {a}.",
        "Find {p}%: {p}/100 * {b} = {a}.",
        "Procenten {p}% af {b} giver {p}/100 * {b} = {a}.",
        "Anvender {p}% på {b}: {p}/100 * {b} = {a}.",
        "Vi ganger {b} med {p}/100: {a}.",
        "{p}% af {b} er lig {p} * {b} / 100 = {a}.",
        "Beregning af {p}%: {p}/100 * {b} = {a}.",
        "Først finder vi {p}% af {b}: {a}.",
        "{p}-procentværdien af {b} er {a}.",
    ],
    "chained": [
        "Derefter finder vi {p}% af {b}: {p}/100 * {b} = {a}.",
        "{p}% af {b} = {p}/100 * {b} = {a}.",
        "Nu anvender vi {p}% på {b}: {a}.",
        "Så {p}% af {b} er lig {a}.",
        "Og {p}% af {b} giver {a}.",
        "Således {p}/100 * {b} = {a}.",
        "Efter at have anvendt {p}%: {p}/100 * {b} = {a}.",
        "Så {p}% af {b} = {a}.",
    ],
}

_PCT_DECIMAL_PROSE = {
    "first": [
        "Først omregner vi {p}% til decimaltal: {p}/100 = {d}. Nu {d} * {b} = {a}.",
        "{p}% = {d}, altså {d} * {b} = {a}.",
        "Omregn {p}% til decimaltal: {p}/100 = {d}. Så {d} * {b} = {a}.",
        "Decimaltallet for {p}% er {d}, så {d} * {b} = {a}.",
        "Vi skriver {p}% som {d}: {d} * {b} = {a}.",
        "På decimalform: {p}/100 = {d}. Ganges: {d} * {b} = {a}.",
        "Betragt {p}% som {d}, så {d} * {b} = {a}.",
        "Procentmultiplikatoren er {d}: {d} * {b} = {a}.",
    ],
    "chained": [
        "Omregn {p}% til decimaltal: {p}/100 = {d}. Så {d} * {b} = {a}.",
        "Nu {p}% = {d}, altså {d} * {b} = {a}.",
        "Derefter omregnes {p}% til {d} og ganges: {d} * {b} = {a}.",
        "Brug decimaltallet {d} = {p}/100: {d} * {b} = {a}.",
        "Så {p}/100 = {d}, altså {d} * {b} = {a}.",
        "På decimalform: {d} * {b} = {a}.",
        "Således {d} * {b} = {a}.",
    ],
}

_LINSOLVE_PROSE = {
    "first": [
        "Lad {v} være den ukendte. Ligningen: {eq} = {t}. Vi løser for {v}: {v} = {x}.",
        "Lad {v} være den ukendte. Så {eq} = {t}. Løsning: {v} = {x}.",
        "Vi kalder det ukendte {v}. Ligningen bliver {eq} = {t}. Ved isolation: {v} = {x}.",
        "Definér {v} som den ukendte. Ligningen {eq} = {t} giver {v} = {x}.",
        "Opskriv ligningen: {eq} = {t}. Løser vi for {v}, får vi {v} = {x}.",
        "Hvis {v} er den ukendte, så er {eq} = {t}, altså {v} = {x}.",
        "Brug {v} for det ukendte. Ligning: {eq} = {t}. Løsning: {v} = {x}.",
        "Vi opstiller ligningen {eq} = {t}, hvor {v} er ukendt. Resultat: {v} = {x}.",
        "Ved at definere {v} som den ukendte: {eq} = {t}, altså {v} = {x}.",
        "Markér den ukendte som {v}. Ligningen {eq} = {t} løst giver {v} = {x}.",
    ],
    "chained": [
        "Nu opstiller vi ligningen: {eq} = {t}. Løsning: {v} = {x}.",
        "Den næste ligning er {eq} = {t}, altså {v} = {x}.",
        "Derefter bygges ligningen {eq} = {t}. Løsning: {v} = {x}.",
        "Herefter giver ligningen {eq} = {t} {v} = {x}.",
        "Nu {eq} = {t}, altså {v} = {x}.",
        "Ud fra dette er {eq} = {t}, og {v} = {x}.",
        "Og ligningen {eq} = {t} løst: {v} = {x}.",
    ],
}


# ═══ RECIPES ════════════════════════════════════════════════════════════════

# ─── Recipe 1: ratio_parts ──────────────────────────────────────────────────

def ratio_parts_recipe(rng: random.Random, n_steps: int = 2,
                        reverse: bool = False) -> dict:
    """N groups × K per group → total. Optionally: minus absent, then / packs."""
    for _try in range(100):
        ctx = Ctx.new(rng)
        child = rng.choice(CHILDLIKE_NOUNS)
        group = rng.choice(GROUPING_NOUNS)

        n_groups = rng.randint(3, 12)
        per_group = rng.randint(4, 15)
        ctx.bind("groups", n_groups, noun=group)
        ctx.bind("per_group", per_group, noun=child)

        if not reverse:
            frame = maybe_frame(rng)
            openers = [
                f"{frame}der er {render_qty(n_groups, group)} med {render_qty(per_group, child)} i hver.",
                f"{frame}i {render_qty(n_groups, group)} indeholder hver {render_qty(per_group, child)}.",
                f"{frame}{ctx.protagonist} ser {render_qty(n_groups, group)}, hver med {render_qty(per_group, child)}.",
                f"{frame}hver af de {n_groups} {group[3]} har {render_qty(per_group, child)}.",
                f"{frame}der blev fordelt {render_qty(per_group, child)} i hver af {render_qty(n_groups, group)}.",
                f"{frame}{render_qty(n_groups, group)} er fulde af {render_qty(per_group, child)} hver.",
            ]
            opener = rng.choice(openers)
            if not frame:
                opener = opener[0].upper() + opener[1:]
            q = [opener]
        else:
            frame = maybe_frame(rng)
            r_openers = [
                f"{frame}der er {render_qty(n_groups, group)}, som hver indeholder det samme ukendte antal {child[3]}.",
                f"{frame}i {render_qty(n_groups, group)} har hver det samme antal {child[3]}.",
                f"{frame}{ctx.protagonist} har {render_qty(n_groups, group)}, hver med det samme, men ukendte antal {child[3]}.",
                f"{frame}der blev fordelt lige mange {child[3]} i hver af {render_qty(n_groups, group)}.",
            ]
            opener = rng.choice(r_openers)
            if not frame:
                opener = opener[0].upper() + opener[1:]
            q = [opener]

        Mul("groups", "per_group", "total").apply(ctx)
        final_var = "total"

        if n_steps >= 3:
            absent = rng.randint(1, min(6, int(ctx.n("total")) // 2))
            ctx.bind("absent", absent, noun=child)
            verb = "er" if absent != 1 else "er"
            q.append(f"{render_qty(absent, child)} {verb} fraværende.")
            Sub("total", "absent", "present").apply(ctx)
            final_var = "present"

        if n_steps >= 4:
            cur = int(ctx.n(final_var))
            divisors = [k for k in (2, 3, 4, 5, 6) if cur % k == 0 and k <= cur]
            if not divisors:
                continue
            n_packs = rng.choice(divisors)
            ctx.bind("packs", n_packs)
            q.append(f"De deler sig i {n_packs} lige store grupper.")
            Div(final_var, "packs", "per_pack").apply(ctx)
            final_var = "per_pack"

        if not reverse:
            closers = [
                f"Hvor mange {child[3]} er der i det endelige resultat?",
                f"Hvad er det endelige antal {child[3]}?",
                f"Beregn det endelige antal {child[3]}.",
                f"Find ud af, hvor mange {child[3]} der er tilbage til sidst.",
                f"Hvor mange {child[3]} er der til sidst?",
                f"Bestem den endelige mængde {child[3]}.",
            ]
            q.append(rng.choice(closers))
            return ctx.render(" ".join(q), final_var)

        # Reverse: state final value, ask for per_group
        final_val = int(ctx.n(final_var))
        state_finals = [
            f"Til sidst er der i alt {render_qty(final_val, child)}.",
            f"Den endelige mængde er {render_qty(final_val, child)}.",
            f"Efter alt er resultatet {render_qty(final_val, child)}.",
        ]
        q.append(rng.choice(state_finals))
        closers = [
            f"Hvor mange {child[3]} er der i {hver_of(group)} {group[0]}?",
            f"Find antallet af {child[3]} i {hver_of(group)} {group[0]}.",
            f"Beregn hvor mange {child[3]} {hver_of(group)} {group[0]} indeholdt.",
        ]
        result = ctx.render_reverse(
            forward_prose=" ".join(q),
            forward_final_var=final_var,
            ask_var="per_group",
            closer=rng.choice(closers),
        )
        result["recipe"] = "ratio_parts_reverse"
        result["n_steps"] = n_steps
        result["direction"] = "reverse"
        return result

    raise RuntimeError("ratio_parts_recipe: couldn't sample divisible params in 100 tries")


# ─── Recipe 2: percent ──────────────────────────────────────────────────────

def percent_recipe(rng: random.Random, n_steps: int = 2, op: str | None = None,
                    reverse: bool = False) -> dict:
    """Percent problems in 3 notation styles × 5 scenarios."""
    ops = ["discount", "markup", "tax", "of-amount", "saving"]
    if op is None:
        op = rng.choice(["of-amount", "saving"]) if reverse else rng.choice(ops)
    style = rng.choice(["direct", "decimal", "multiplier"])
    if op in ("of-amount", "saving"):
        style = rng.choice(["direct", "decimal"])

    if op in ("discount", "markup", "tax", "saving"):
        scenario_kind = "shop"
    elif op == "of-amount":
        scenario_kind = rng.choice(["shop", "count"])

    for _try in range(100):
        pct = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80])
        g = math.gcd(pct, 100)
        step = 100 // g
        base = step * rng.randint(2, 40)

        pct2 = None
        if n_steps >= 3 and op == "discount":
            amount = base * pct // 100
            res_val = base - amount
            for _p2 in [5, 10, 15, 20, 25]:
                if (res_val * _p2) % 100 == 0:
                    pct2 = _p2
                    break
            if pct2 is None:
                continue

        ctx = Ctx.new(rng)
        ctx.bind("pct", pct)
        ctx.bind("base", base, noun=None)
        p = ctx.protagonist

        if scenario_kind == "shop":
            item = rng.choice(SHOP_ITEMS)
            base_qty = render_qty(base, KRONE)     # "200 kroner"
            item_indef = render_qty(1, item)       # "en cykel"
            item_def = item[2]                     # "cyklen"

            frame = maybe_frame(rng)
            if op == "discount":
                openers = [
                    f"{frame}{p} køber {item_indef}, som koster {base_qty}. Butikken giver {pct}% rabat. Hvor meget betaler {p}?",
                    f"{frame}den oprindelige pris på {item_indef} er {base_qty}. Med {pct}% rabat, hvad er den nye pris?",
                    f"{frame}{item_indef} koster {base_qty}. Med en rabat på {pct}%, hvor meget skal {p} betale?",
                    f"{frame}{p} ser {item_indef} til {base_qty} med {pct}% rabat. Beregn den endelige pris.",
                ]
            elif op == "markup":
                openers = [
                    f"{frame}prisen på {item_indef} var {base_qty}, men steg med {pct}%. Hvad er den nye pris?",
                    f"{frame}{p} har {item_indef}, som koster {base_qty}. Prisen stiger med {pct}%. Hvor meget koster {item_def} nu?",
                    f"{frame}{item_indef} kostede {base_qty}. Efter en stigning på {pct}%, hvad er den nye pris?",
                ]
            elif op == "tax":
                openers = [
                    f"{frame}{p} køber {item_indef} for {base_qty}. Momsen er {pct}%. Hvor meget betaler {p} i alt?",
                    f"{frame}{item_indef} koster {base_qty}. Med {pct}% moms, hvad er totalen?",
                    f"{frame}prisen på {item_indef} er {base_qty}, og der lægges {pct}% moms til. Hvad er den endelige pris?",
                ]
            elif op == "of-amount":
                if not reverse:
                    openers = [
                        f"{frame}{p} beregnede {pct}% af {base_qty}. Hvad er resultatet?",
                        f"{frame}find {pct}% af {base_qty}.",
                        f"{frame}hvor meget er {pct}% af {base_qty}?",
                    ]
                else:
                    openers = [
                        f"{frame}{p} beregnede {pct}% af et ukendt beløb i {KRONE[3]}.",
                        f"{frame}{pct}% af et ukendt beløb i {KRONE[3]} er lig følgende beløb.",
                        f"{frame}{p} ved, at {pct}% af sit budget i {KRONE[3]} er lig et bestemt beløb.",
                    ]
            elif op == "saving":
                if not reverse:
                    openers = [
                        f"{frame}{p} købte {item_indef}, som kostede {base_qty}, med {pct}% rabat. Hvor mange {KRONE[3]} sparede {p}?",
                        f"{frame}{item_indef} kostede {base_qty} med en rabat på {pct}%. Hvor meget sparede {p}?",
                        f"{frame}{p} fik {pct}% rabat på {item_indef} til {base_qty}. Beregn besparelsen.",
                    ]
                else:
                    openers = [
                        f"{frame}{p} købte {item_indef} med {pct}% rabat. Den oprindelige pris er endnu ukendt.",
                        f"{frame}{p} fik {pct}% rabat på {item_indef}. Den oprindelige pris er ukendt.",
                    ]
            q = rng.choice(openers)
            if not frame:
                q = q[0].upper() + q[1:]
        else:  # count
            item = rng.choice(COUNT_ITEMS)
            if not reverse:
                q = (f"I en klasse er der {render_qty(base, item)}. {pct}% af dem "
                     f"bruger briller. Hvor mange {item[3]} bruger briller?")
            else:
                q = (f"I en klasse er der et ukendt antal {item[3]}. "
                     f"{pct}% af dem bruger briller.")

        # Compute the answer
        if style == "multiplier" and op in ("discount", "markup", "tax"):
            factor = (100 - pct if op == "discount" else 100 + pct)
            assert (base * factor) % 100 == 0
            res = base * factor // 100
            mult = factor / 100
            mult_str = fmt_num(mult)
            sign = "-" if op == "discount" else "+"
            ctx.chain.append(f"1 {sign} {pct}/100 = {mult_str}")
            ctx.chain.append(f"{fmt_num(base)} * {mult_str} = {fmt_num(res)}")
            ctx.prose.append(f"Multiplikatoren er 1 {sign} {pct}/100 = {mult_str}.")
            ctx.prose.append(f"Resultat = {fmt_num(base)} * {mult_str} = {fmt_num(res)}.")
            ctx.bind("res", res)
            return ctx.render(q, "res")

        Pct("pct", "base", "amount", style=style).apply(ctx)
        if op == "discount":
            Sub("base", "amount", "res").apply(ctx)
            final = "res"
        elif op in ("markup", "tax"):
            Add("base", "amount", "res").apply(ctx)
            final = "res"
        else:
            final = "amount"

        if n_steps >= 3 and op == "discount" and pct2 is not None and final == "res":
            ctx.bind("pct2", pct2)
            if not reverse:
                q += f" Læg nu {pct2}% moms oveni den nye pris. Hvad er den endelige pris?"
            Pct("pct2", "res", "tax_amt", style="direct").apply(ctx)
            Add("res", "tax_amt", "final_price").apply(ctx)
            final = "final_price"

        if not reverse:
            return ctx.render(q, final)

        if op not in ("of-amount", "saving"):
            raise RuntimeError(f"percent_recipe: reverse not supported for op={op}")

        final_val = int(ctx.n(final))
        if scenario_kind == "count":
            states = [
                f" {final_val} {item[3]} bruger briller.",
                f" Der er {render_qty(final_val, item)}, som bruger briller.",
            ]
            closers = [
                f"Hvor mange {item[3]} er der i klassen?",
                f"Find det samlede antal {item[3]}.",
                f"Beregn antallet af {item[3]} i klassen.",
            ]
        elif op == "of-amount":
            states = [
                f" Resultatet er {render_qty(final_val, KRONE)}.",
                f" Det er lig {render_qty(final_val, KRONE)}.",
                f" Beløbet er {render_qty(final_val, KRONE)}.",
            ]
            closers = [
                f"Hvad var det oprindelige beløb i {KRONE[3]}?",
                f"Beregn det oprindelige beløb.",
                f"Find den oprindelige værdi.",
            ]
        else:  # saving
            states = [
                f" {p} sparede {render_qty(final_val, KRONE)}.",
                f" Det sparede beløb er {render_qty(final_val, KRONE)}.",
            ]
            closers = [
                f"Hvad var den oprindelige pris på {item_def}?",
                f"Find den oprindelige pris.",
                f"Beregn den oprindelige pris på {item_def}.",
            ]
        q += rng.choice(states)
        result = ctx.render_reverse(
            forward_prose=q,
            forward_final_var=final,
            ask_var="base",
            closer=rng.choice(closers),
        )
        result["recipe"] = "percent_reverse"
        result["n_steps"] = n_steps
        result["direction"] = "reverse"
        return result

    raise RuntimeError("percent_recipe: couldn't sample divisible params")


# ─── Recipe 3: average ──────────────────────────────────────────────────────

def average_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """Average of test scores (n_steps=2) or fraction of average (n_steps=3)."""
    for _try in range(100):
        subject = rng.choice(SUBJECT_NOUNS)
        subject_inline = SUBJECT_INLINE[subject[0]]
        n_scores = rng.choice([3, 4, 5])
        scores = [rng.randint(60, 100) for _ in range(n_scores)]
        if sum(scores) % n_scores != 0:
            continue

        ctx = Ctx.new(rng)
        p = ctx.protagonist
        for i, s in enumerate(scores):
            ctx.bind(f"s{i}", s, noun=None)
        scores_str = ", ".join(str(s) for s in scores[:-1]) + f" og {scores[-1]}"
        frame = maybe_frame(rng)
        openers = [
            f"{frame}{p} fik {n_scores} karakterer i {subject_inline}: {scores_str}. Hvad er gennemsnittet?",
            f"{frame}{poss(p)} karakterer i {subject_inline} er: {scores_str}. Find gennemsnittet.",
            f"{frame}efter {n_scores} prøver i {subject_inline} fik {p}: {scores_str}. Beregn gennemsnitskarakteren.",
            f"{frame}{p} skrev {n_scores} prøver i {subject_inline} og fik {scores_str}. Hvad er middelresultatet?",
        ]
        q = rng.choice(openers)
        if not frame:
            q = q[0].upper() + q[1:]

        Avg([f"s{i}" for i in range(n_scores)], "avg").apply(ctx)
        final = "avg"

        if n_steps >= 3:
            avail_pcts = [pp for pp in [10, 20, 25, 50, 75] if int(ctx.n("avg")) * pp % 100 == 0]
            if not avail_pcts:
                continue
            pct = rng.choice(avail_pcts)
            ctx.bind("pct", pct)
            q += f" Bagefter, hvor meget er {pct}% af gennemsnitskarakteren?"
            Pct("pct", "avg", "result", style=rng.choice(["direct", "decimal"])).apply(ctx)
            final = "result"

        return ctx.render(q, final)

    raise RuntimeError("average_recipe: couldn't sample divisible params")


# ─── Recipe 4: fraction_cascade ─────────────────────────────────────────────

def fraction_cascade_recipe(rng: random.Random, n_steps: int = 2,
                             reverse: bool = False) -> dict:
    """Fraction-of-fraction."""
    fractions = [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 5), (2, 5), (3, 5), (4, 5),
                 (1, 6), (5, 6), (1, 7), (2, 7), (3, 7), (5, 7), (1, 8), (3, 8), (5, 8), (7, 8),
                 (1, 9), (2, 9), (4, 9), (5, 9), (7, 9), (1, 10), (3, 10), (7, 10), (9, 10)]
    # (sub_pop, verb_phrase, obj_phrase)
    stories = [
        ("piger",     "har en rød trøje på", "en rød trøje"),
        ("drenge",    "har en cykel",        "en cykel"),
        ("elever",    "bor i byen",          "i byen"),
        ("elever",    "kan svømme",          "svømning"),
        ("børn",      "spiller fodbold",     "fodbold"),
        ("kunder",    "kommer tilbage dagen efter", "dagen efter"),
    ]

    for _try in range(100):
        f1 = rng.choice(fractions)
        f2 = rng.choice(fractions)
        base_noun = rng.choice(COUNT_ITEMS)
        story = rng.choice(stories)
        sub_pop, verb_phrase, obj_phrase = story

        base = f1[1] * rng.randint(2, 40)
        step1_result = base * f1[0] // f1[1]
        if n_steps >= 3 and (step1_result * f2[0]) % f2[1] != 0:
            continue

        ctx = Ctx.new(rng)
        ctx.bind("base", base, noun=base_noun)
        p = ctx.protagonist
        frame = maybe_frame(rng)

        if not reverse:
            openers = [
                f"{frame}i en gruppe er der {render_qty(base, base_noun)}. {f1[0]}/{f1[1]} af dem er {sub_pop}.",
                f"{frame}ud af {render_qty(base, base_noun)} er {f1[0]}/{f1[1]} {sub_pop}.",
                f"{frame}{p} talte {render_qty(base, base_noun)}; {f1[0]}/{f1[1]} af dem er {sub_pop}.",
            ]
        else:
            openers = [
                f"{frame}i en gruppe er der et ukendt antal {base_noun[3]}. {f1[0]}/{f1[1]} af dem er {sub_pop}.",
                f"{frame}ud af gruppen af {base_noun[3]} er {f1[0]}/{f1[1]} {sub_pop}.",
                f"{frame}{p} talte en gruppe af {base_noun[3]}; {f1[0]}/{f1[1]} af dem er {sub_pop}.",
            ]
        opener = rng.choice(openers)
        if not frame:
            opener = opener[0].upper() + opener[1:]

        Frac("base", f1[0], f1[1], "girls").apply(ctx)
        final = "girls"

        if n_steps >= 3:
            opener += f" Ud af de {sub_pop} {verb_phrase} {f2[0]}/{f2[1]}."
            Frac("girls", f2[0], f2[1], "red").apply(ctx)
            final = "red"
            if not reverse:
                closers = [
                    f" Hvor mange {base_noun[3]} {verb_phrase}?",
                    f" Beregn antallet, som {verb_phrase}.",
                    f" Find ud af, hvor mange der {verb_phrase}.",
                ]
                opener += rng.choice(closers)
        else:
            if not reverse:
                closers = [
                    f" Hvor mange er {sub_pop}?",
                    f" Find antallet af {sub_pop}.",
                    f" Hvor mange {base_noun[3]} er {sub_pop}?",
                ]
                opener += rng.choice(closers)

        if not reverse:
            return ctx.render(opener, final)

        final_val = int(ctx.n(final))
        which = sub_pop if n_steps == 2 else f"{sub_pop}, som {verb_phrase}"
        state_finals = [
            f" Der er {final_val} {which}.",
            f" Antallet af {which} er {final_val}.",
            f" Til sidst er {final_val} af dem {which}.",
        ]
        opener += rng.choice(state_finals)
        closers = [
            f"Hvor mange {base_noun[3]} er der i alt?",
            f"Find det samlede antal {base_noun[3]}.",
            f"Beregn antallet af {base_noun[3]} i gruppen.",
        ]
        result = ctx.render_reverse(
            forward_prose=opener,
            forward_final_var=final,
            ask_var="base",
            closer=rng.choice(closers),
        )
        result["recipe"] = "fraction_cascade_reverse"
        result["n_steps"] = n_steps
        result["direction"] = "reverse"
        return result

    raise RuntimeError("fraction_cascade_recipe: couldn't sample")


# ─── Recipe 5: ratio_diff ───────────────────────────────────────────────────

def ratio_diff_recipe(rng: random.Random, n_steps: int = 3,
                       reverse: bool = False) -> dict:
    """N:M ratio splits `total` between two people; compute each part; report difference."""
    names = rng.sample(NAMES, 2)
    obj = rng.choice(OBJECT_NOUNS)

    for _try in range(100):
        a, b = rng.choice([(2, 3), (3, 5), (1, 4), (2, 5), (3, 4), (1, 3), (4, 5), (3, 7)])
        unit = rng.randint(3, 30)
        total = (a + b) * unit
        if total > 300:
            continue

        ctx = Ctx.new(rng)
        ctx.bind("total", total, noun=obj)
        ctx.bind("parts", a + b, noun=obj)
        ctx.bind("a", a * unit, noun=obj)
        ctx.bind("b", b * unit, noun=obj)

        if not reverse:
            openers = [
                f"{names[0]} og {names[1]} deler {render_qty(total, obj)} i forholdet {a}:{b}. Hvad er forskellen mellem deres andele?",
                f"I forholdet {a}:{b} deler {names[0]} og {names[1]} {render_qty(total, obj)}. Hvor mange flere har den ene end den anden?",
                f"{names[0]} får {a} dele, {names[1]} får {b} dele af i alt {render_qty(total, obj)}. Find forskellen.",
            ]
            q = rng.choice(openers)

            Div("total", "parts", "unit").apply(ctx)
            ctx.bind("ra", a)
            ctx.bind("rb", b)
            Mul("ra", "unit", "part_a").apply(ctx)
            Mul("rb", "unit", "part_b").apply(ctx)
            Sub("part_b", "part_a", "diff").apply(ctx)
            final = "diff"

            if n_steps >= 5:
                diff_val = int(ctx.n("diff"))
                portions = [k for k in (2, 3, 4, 5) if diff_val % k == 0]
                if not portions:
                    continue
                k = rng.choice(portions)
                ctx.bind("k", k)
                q += f" Hvis vi deler forskellen ligeligt mellem {k} personer, hvor meget får hver?"
                Div("diff", "k", "per_person").apply(ctx)
                final = "per_person"

            return ctx.render(q, final)

        # Reverse path (branching chain — manual prose)
        larger_r, smaller_r = (b, a) if b > a else (a, b)
        diff = (larger_r - smaller_r) * unit
        if diff <= 0:
            continue
        openers = [
            f"{names[0]} og {names[1]} deler et ukendt antal i forholdet {a}:{b}. Forskellen mellem deres andele er {render_qty(diff, obj)}.",
            f"I forholdet {a}:{b} deler {names[0]} og {names[1]} nogle {obj[3]}. Den ene har {render_qty(diff, obj)} mere end den anden.",
            f"{names[0]} får {a} dele, {names[1]} får {b} dele af et ukendt total. Forskellen er {render_qty(diff, obj)}.",
        ]
        q = rng.choice(openers)
        closers = [
            f" Beregn det samlede antal.",
            f" Find ud af, hvor mange {obj[3]} der er i alt.",
            f" Hvor mange {obj[3]} er der i alt?",
        ]
        # Manual chain — branching, so we hand-write both prose + chain.
        step = larger_r - smaller_r
        parts = a + b
        ctx.chain.append(f"{larger_r} - {smaller_r} = {step}")
        ctx.chain.append(f"{diff} / {step} = {unit}")
        ctx.chain.append(f"{a} + {b} = {parts}")
        ctx.chain.append(f"{unit} * {parts} = {total}")
        ctx.prose.append(f"Forskellen mellem forholdstallene er {larger_r} - {smaller_r} = {step}.")
        ctx.prose.append(f"Så værdien af én del er {diff} / {step} = {unit}.")
        ctx.prose.append(f"Summen af forholdstallene er {a} + {b} = {parts}.")
        ctx.prose.append(f"Så totalen er {unit} * {parts} = {total}.")
        ctx.bind("total_rev", total, noun=obj)
        result = ctx.render(q + rng.choice(closers), "total_rev")
        result["recipe"] = "ratio_diff_reverse"
        result["n_steps"] = n_steps
        result["direction"] = "reverse"
        return result

    raise RuntimeError("ratio_diff_recipe: couldn't sample")


# ─── Recipe 6: consec_avg ───────────────────────────────────────────────────

def consec_avg_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """N consecutive integers summing to S. Find the middle (= avg) or smallest/largest."""
    count = rng.choice([3, 5, 7, 9])
    ask_choices = ["smallest", "largest"] if n_steps >= 3 else ["smallest", "largest", "middle"]
    ask = rng.choice(ask_choices)
    start = rng.randint(2, 200)
    values = [start + i for i in range(count)]
    total = sum(values)
    ctx = Ctx.new(rng)
    for i, v in enumerate(values):
        ctx.bind(f"v{i}", v)
    ctx.bind("total", total)
    ctx.bind("count", count)

    what = {"smallest": "mindste", "largest": "største", "middle": "mellemste"}[ask]
    openers = [
        f"Summen af {count} på hinanden følgende heltal er {total}. Hvad er det {what}?",
        f"{count} på hinanden følgende heltal summerer til {total}. Find det {what}.",
        f"Hvis {count} på hinanden følgende heltal har en sum på {total}, hvad er så det {what}?",
        f"Der er {count} på hinanden følgende heltal, hvis sum er lig {total}. Beregn det {what}.",
    ]
    q = rng.choice(openers)

    ctx.chain.append(f"{total} / {count} = {values[count // 2]}")
    ctx.prose.append(f"Gennemsnit = sum / antal: {total} / {count} = {values[count // 2]}.")
    ctx.bind("avg", values[count // 2])

    if ask == "middle":
        final_var = "avg"
    elif ask == "smallest":
        offset = count // 2
        ctx.chain.append(f"{values[count // 2]} - {offset} = {values[0]}")
        ctx.prose.append(f"Det mindste = mellemste - {offset}: {values[count // 2]} - {offset} = {values[0]}.")
        ctx.bind("smallest", values[0])
        final_var = "smallest"
    else:
        offset = count // 2
        ctx.chain.append(f"{values[count // 2]} + {offset} = {values[-1]}")
        ctx.prose.append(f"Det største = mellemste + {offset}: {values[count // 2]} + {offset} = {values[-1]}.")
        ctx.bind("largest", values[-1])
        final_var = "largest"

    if n_steps >= 3 and final_var != "avg":
        k = rng.randint(2, 5)
        ctx.bind("k", k)
        q += f" Hvad er {k} gange den værdi?"
        Mul(final_var, "k", "scaled").apply(ctx)
        final_var = "scaled"

    return ctx.render(q, final_var)


# ─── Recipe 7: inverse_rate ─────────────────────────────────────────────────

_WORKERS: list[Noun] = [
    ("arbejder",  "en", "arbejderen",  "arbejdere",  "arbejderne"),
    ("pumpe",     "en", "pumpen",      "pumper",     "pumperne"),
    ("maskine",   "en", "maskinen",    "maskiner",   "maskinerne"),
    ("høstmaskine","en", "høstmaskinen","høstmaskiner","høstmaskinerne"),
    ("kok",       "en", "kokken",      "kokke",      "kokkene"),
]
_TIME_UNITS: list[Noun] = [
    ("time",      "en", "timen",       "timer",      "timerne"),
    ("minut",     "et", "minuttet",    "minutter",   "minutterne"),
    ("dag",       "en", "dagen",       "dage",       "dagene"),
]

# (worker_noun, verb_infinitive, task_indef, time_noun)
INV_SCENARIOS: list[tuple[Noun, str, str, Noun]] = [
    (_WORKERS[0], "male",    "en væg",    _TIME_UNITS[0]),
    (_WORKERS[1], "fylde",   "en pool",   _TIME_UNITS[1]),
    (_WORKERS[2], "trykke",  "en bog",    _TIME_UNITS[0]),
    (_WORKERS[3], "høste",   "en mark",   _TIME_UNITS[2]),
    (_WORKERS[4], "tilberede","et måltid",_TIME_UNITS[0]),
]


def inverse_rate_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """W1 workers × T1 time = const. Find T2 for W2 workers (or W2 for T2)."""
    for _try in range(100):
        w1 = rng.randint(2, 30)
        t1 = rng.randint(2, 60)
        const = w1 * t1
        divs = [d for d in range(1, const + 1) if const % d == 0 and d != w1 and 1 <= d <= 60]
        if not divs:
            continue
        w2 = rng.choice(divs)
        t2 = const // w2

        worker, verb, task, tunit = rng.choice(INV_SCENARIOS)
        ask = rng.choice(["find-time", "find-workers"])
        ctx = Ctx.new(rng)
        ctx.bind("w1", w1)
        ctx.bind("t1", t1)

        t1_ren = render_qty(t1, tunit)
        t2_ren = render_qty(t2, tunit)
        w1_ren = render_qty(w1, worker)
        w2_ren = render_qty(w2, worker)
        tunit_pl = tunit[3]
        workers_pl = worker[3]

        frame = maybe_frame(rng)
        if ask == "find-time":
            ctx.bind("w2", w2)
            openers = [
                f"{frame}{w1_ren} bruger {t1_ren} på at {verb} {task}. Hvor mange {tunit_pl} bruger {w2_ren}?",
                f"{frame}hvis {w1_ren} færdiggør {task} på {t1_ren}, hvor mange {tunit_pl} bruger {w2_ren} så?",
                f"{frame}{w1_ren} bruger {t1_ren} på opgaven. Hvor længe for {w2_ren}?",
            ]
            Mul("w1", "t1", "const").apply(ctx)
            Div("const", "w2", "t2").apply(ctx)
            final = "t2"
        else:
            ctx.bind("t2", t2)
            openers = [
                f"{frame}{w1_ren} bruger {t1_ren} på at {verb} {task}. Hvor mange {workers_pl} skal der til for at gøre det på {t2_ren}?",
                f"{frame}{w1_ren} færdiggør {task} på {t1_ren}. Hvor mange {workers_pl} skal der til for at gøre det samme på {t2_ren}?",
                f"{frame}opgaven at {verb} {task} tager {t1_ren} med {w1_ren}. Hvor mange {workers_pl} skal der til, for at det kun tager {t2_ren}?",
            ]
            Mul("w1", "t1", "const").apply(ctx)
            Div("const", "t2", "w2").apply(ctx)
            final = "w2"
        q = rng.choice(openers)
        if not frame:
            q = q[0].upper() + q[1:]

        if n_steps >= 3 and ask == "find-time":
            w3_candidates = [d for d in range(1, const + 1)
                             if const % d == 0 and d != w1 and d != w2 and 1 <= d <= 40]
            if not w3_candidates:
                continue
            w3 = rng.choice(w3_candidates)
            ctx.bind("w3", w3)
            q += f" Og hvor mange {tunit_pl} bruger {render_qty(w3, worker)}?"
            Div("const", "w3", "t3").apply(ctx)
            final = "t3"

        return ctx.render(q, final)

    raise RuntimeError("inverse_rate_recipe: couldn't sample")


# ─── Recipe 8: ratio_fraction ───────────────────────────────────────────────

def ratio_fraction_recipe(rng: random.Random, n_steps: int = 2,
                           reverse: bool = False) -> dict:
    """Ratio via fraction-of-total: r_i/(r_1+r_2) * total = part_i."""
    names = rng.sample(NAMES, 2)
    obj = rng.choice(OBJECT_NOUNS)

    for _try in range(100):
        a, b = rng.choice([(2, 3), (3, 5), (1, 4), (3, 4), (1, 3), (4, 5), (3, 7), (2, 5)])
        unit = rng.randint(3, 30)
        total = (a + b) * unit
        if total > 300:
            continue

        ask = rng.choice(["larger", "smaller", "direct"])
        target_r = max(a, b) if ask == "larger" else min(a, b) if ask == "smaller" else a
        target_name = names[0] if target_r == a else names[1]

        ctx = Ctx.new(rng)
        ctx.bind("ra", a)
        ctx.bind("rb", b)
        ctx.bind("total", total, noun=obj)

        which = {"larger": "største andel", "smaller": "mindste andel",
                 "direct": f"{poss(target_name)} andel"}[ask]
        if not reverse:
            openers = [
                f"{names[0]} og {names[1]} deler {render_qty(total, obj)} i forholdet {a}:{b}. Hvad er den {which}?",
                f"I forholdet {a}:{b} deler {names[0]} og {names[1]} {render_qty(total, obj)}. Find den {which}.",
                f"Deres samlede antal er {render_qty(total, obj)} i forholdet {a}:{b}. Hvad er den {which}?",
            ]
        else:
            openers = [
                f"{names[0]} og {names[1]} deler et ukendt antal {obj[3]} i forholdet {a}:{b}.",
                f"I forholdet {a}:{b} deler {names[0]} og {names[1]} nogle {obj[3]}.",
                f"Deres samlede antal er endnu ukendt, men forholdet er {a}:{b}.",
            ]
        q = rng.choice(openers)

        Add("ra", "rb", "r_sum").apply(ctx)
        Frac("total", target_r, a + b, "part").apply(ctx)
        final = "part"

        if n_steps >= 3:
            part_val = int(ctx.n("part"))
            pcts = [p for p in [10, 20, 25, 40, 50, 75] if (part_val * p) % 100 == 0]
            if not pcts:
                continue
            pct = rng.choice(pcts)
            ctx.bind("pct", pct)
            if not reverse:
                q += f" Derefter foræres {pct}% af den andel bort. Hvor meget foræres bort?"
            Pct("pct", "part", "gift", style="direct").apply(ctx)
            final = "gift"

        if not reverse:
            return ctx.render(q, final)

        final_val = int(ctx.n(final))
        if final == "gift":
            states = [
                f" Det bortforærede beløb er {render_qty(final_val, obj)} ({pct}% af den {which}).",
                f" Ud af den {which} er {pct}% givet bort, hvilket er {render_qty(final_val, obj)}.",
            ]
        else:
            states = [
                f" Den {which} er {render_qty(final_val, obj)}.",
                f" {target_name} modtager {render_qty(final_val, obj)}.",
            ]
        q += rng.choice(states)
        closers = [
            f"Hvor mange {obj[3]} er der i alt?",
            f"Find det samlede antal {obj[3]}.",
            f"Beregn den samlede mængde {obj[3]}.",
        ]
        result = ctx.render_reverse(
            forward_prose=q,
            forward_final_var=final,
            ask_var="total",
            closer=rng.choice(closers),
        )
        result["recipe"] = "ratio_fraction_reverse"
        result["n_steps"] = n_steps
        result["direction"] = "reverse"
        return result

    raise RuntimeError("ratio_fraction_recipe: couldn't sample")


# ─── Recipe 9: distance_direct ──────────────────────────────────────────────

_VEHICLES: list[Noun] = [
    ("bil",         "en", "bilen",         "biler",         "bilerne"),
    ("cykel",       "en", "cyklen",        "cykler",        "cyklerne"),
    ("lastbil",     "en", "lastbilen",     "lastbiler",     "lastbilerne"),
    ("motorcykel",  "en", "motorcyklen",   "motorcykler",   "motorcyklerne"),
]


def distance_direct_recipe(rng: random.Random, n_steps: int = 1) -> dict:
    """D = R * T with a single ask: find D, R, or T."""
    for _try in range(100):
        r = rng.choice([40, 50, 60, 70, 75, 80, 90, 100, 120])
        t = rng.randint(2, 8)
        d = r * t
        ask = "d" if n_steps >= 2 else rng.choice(["d", "r", "t"])

        vehicle = rng.choice(_VEHICLES)
        name = rng.choice(NAMES)
        ctx = Ctx.new(rng)
        ctx.protagonist = name

        frame = maybe_frame(rng)
        vehicle_indef = render_qty(1, vehicle)  # "en bil"
        vehicle_def = vehicle[2]                # "bilen"
        if ask == "d":
            ctx.bind("r", r); ctx.bind("t", t)
            openers = [
                f"{frame}{name} kører {vehicle_indef} med {r} km/t i {t} timer. Hvor mange kilometer tilbagelægger {name}?",
                f"{frame}på {poss(name)} {vehicle[0]} kører {name} med {r} km/t i {t} timer. Hvad er den tilbagelagte distance?",
                f"{frame}{name} kører sin {vehicle[0]} med {r} km/t i {t} timer. Find distancen.",
            ]
            Mul("r", "t", "d").apply(ctx)
            final = "d"
        elif ask == "r":
            ctx.bind("d", d); ctx.bind("t", t)
            openers = [
                f"{frame}{name} kører {d} km på sin {vehicle[0]} på {t} timer. Hvad er farten?",
                f"{frame}efter {t} timers kørsel har {name} tilbagelagt {d} km på sin {vehicle[0]}. Beregn farten.",
                f"{frame}{poss(name)} {vehicle[0]} tilbagelægger {d} km på {t} timer. Hvad er farten?",
            ]
            Div("d", "t", "r").apply(ctx)
            final = "r"
        else:
            if d % r != 0:
                continue
            ctx.bind("d", d); ctx.bind("r", r)
            openers = [
                f"{frame}{name} kører sin {vehicle[0]} med {r} km/t. Hvor mange timer skal der til for at tilbagelægge {d} km?",
                f"{frame}på sin {vehicle[0]} kører {name} med {r} km/t. Hvor lang tid tager det at køre {d} km?",
                f"{frame}{name} skal køre {d} km med {r} km/t på sin {vehicle[0]}. Hvor mange timer varer det?",
            ]
            Div("d", "r", "t").apply(ctx)
            final = "t"
        q = rng.choice(openers)
        if not frame:
            q = q[0].upper() + q[1:]

        if n_steps >= 2 and ask == "d":
            r2_candidates = [r2 for r2 in [40, 50, 60, 75, 80, 100, 120] if d % r2 == 0 and r2 != r]
            if not r2_candidates:
                continue
            r2 = rng.choice(r2_candidates)
            ctx.bind("r2", r2)
            q += f" Hvis {name} vender tilbage med {r2} km/t, hvor mange timer varer returturen?"
            Div("d", "r2", "t2").apply(ctx)
            final = "t2"

            if n_steps >= 3:
                ctx.bind("t_orig", t)
                q += f" Og hvor mange timer varer hele turen tur/retur?"
                Add("t_orig", "t2", "t_total").apply(ctx)
                final = "t_total"

        return ctx.render(q, final)

    raise RuntimeError("distance_direct_recipe: couldn't sample")


# ─── Recipe 10: coin_assume ─────────────────────────────────────────────────

# denominations: (small_val, big_val, currency_adj_stem, item_noun)
# In Danish "5-krone-mønt" / "50-øre-mønt" — we use the stem to form
# "N-{stem}-mønter".
_MØNT: Noun = ("mønt", "en", "mønten", "mønter", "mønterne")
_SEDDEL: Noun = ("seddel", "en", "sedlen", "sedler", "sedlerne")

# (small_val, big_val, unit_singular, unit_plural, item_noun)
_COIN_DENOMS: list[tuple[int, int, str, str, Noun]] = [
    (1,  5,  "øre",   "øre",    _MØNT),
    (5,  25, "øre",   "øre",    _MØNT),
    (10, 50, "øre",   "øre",    _MØNT),
    (5,  20, "krone", "kroner", _SEDDEL),
]


def coin_assume_recipe(rng: random.Random, n_steps: int = 3) -> dict:
    """Coin problem via "assume all small" reasoning — no algebra."""
    for _try in range(100):
        small_val, big_val, unit_sg, unit_pl, item = rng.choice(_COIN_DENOMS)
        total_count = rng.randint(5, 30)
        count_big = rng.randint(1, total_count - 1)
        count_small = total_count - count_big
        total_value = count_small * small_val + count_big * big_val

        name = rng.choice(NAMES)
        ask = rng.choice(["find-big", "find-small"])
        ctx = Ctx.new(rng)
        ctx.protagonist = name
        ctx.bind("small_val", small_val)
        ctx.bind("big_val", big_val)
        ctx.bind("total_count", total_count)
        ctx.bind("total_value", total_value)

        target_val = big_val if ask == "find-big" else small_val
        # In Danish: "5-krone-mønter" / "50-øre-mønter"
        target_desc = f"{target_val}-{unit_sg}-{item[3]}"
        small_desc = f"{small_val} {unit_sg if small_val == 1 else unit_pl}"
        big_desc = f"{big_val} {unit_sg if big_val == 1 else unit_pl}"
        total_val_desc = f"{total_value} {unit_sg if total_value == 1 else unit_pl}"
        q = (f"{name} har {render_qty(total_count, item)} med en samlet værdi på "
             f"{total_val_desc}. Nogle er værd {small_desc}, andre {big_desc}. "
             f"Hvor mange {target_desc} er der?")

        Mul("small_val", "total_count", "assumed_total").apply(ctx)
        Sub("total_value", "assumed_total", "extra").apply(ctx)
        ctx.bind("step_up", big_val - small_val)
        ctx.chain.append(f"{big_val} - {small_val} = {big_val - small_val}")
        ctx.prose.append(
            f"Hver {big_val}-{unit_sg}-{item[0]} bidrager med {big_val} - {small_val} = "
            f"{big_val - small_val} mere end en {small_val}-{unit_sg}-{item[0]}."
        )
        Div("extra", "step_up", "count_big").apply(ctx)

        if ask == "find-big":
            final = "count_big"
        else:
            Sub("total_count", "count_big", "count_small").apply(ctx)
            final = "count_small"

        if n_steps >= 5:
            target_denom_var = "big_val" if ask == "find-big" else "small_val"
            q += f" Hvad er den samlede værdi af disse {target_desc}?"
            Mul(target_denom_var, final, "target_value").apply(ctx)
            final = "target_value"

        return ctx.render(q, final)

    raise RuntimeError("coin_assume_recipe: couldn't sample")


# ─── Recipe 11: distance_catchup ────────────────────────────────────────────

def distance_catchup_recipe(rng: random.Random, n_steps: int = 2,
                             reverse: bool = False) -> dict:
    """A leaves at ra km/h; h hours later B leaves at rb (rb > ra) and catches up."""
    for _try in range(100):
        ra = rng.choice([40, 50, 60, 70, 75, 80])
        rb = ra + rng.choice([20, 30, 40, 50])
        h = rng.randint(2, 6)
        num = ra * h
        den = rb - ra
        if num % den != 0:
            continue
        t = num // den
        if not (1 <= t <= 12):
            continue

        names = rng.sample(NAMES, 2)
        vehicle = rng.choice(_VEHICLES)
        ctx = Ctx.new(rng)
        ctx.protagonist = names[1]
        ctx.bind("ra", ra); ctx.bind("h", h); ctx.bind("rb", rb)

        if not reverse:
            q = (f"{names[0]} kører af sted på sin {vehicle[0]} med {ra} km/t. "
                 f"Efter {h} timer kører {names[1]} fra samme sted "
                 f"i samme retning med {rb} km/t. "
                 f"Om hvor mange timer indhenter {names[1]} {names[0]}?")
        else:
            q = (f"{names[0]} kører af sted på sin {vehicle[0]} med {ra} km/t. "
                 f"Efter et ukendt antal timer kører {names[1]} fra samme sted "
                 f"i samme retning med {rb} km/t.")

        Mul("ra", "h", "head_start").apply(ctx)
        Sub("rb", "ra", "gap").apply(ctx)
        Div("head_start", "gap", "t").apply(ctx)
        final = "t"

        if n_steps >= 4:
            if not reverse:
                q += f" Hvor mange km havde {names[0]} kørt, da {names[1]} indhentede?"
            Add("h", "t", "a_total_time").apply(ctx)
            Mul("ra", "a_total_time", "a_dist").apply(ctx)
            final = "a_dist"

        if not reverse:
            return ctx.render(q, final)

        if n_steps >= 4:
            raise RuntimeError("distance_catchup: reverse not supported for n_steps=4")

        final_val = int(ctx.n(final))
        q += f" {names[1]} indhenter {names[0]} efter {final_val} timer."
        closer = f" Beregn, efter hvor mange timer {names[1]} kørte efter {names[0]}."
        result = ctx.render_reverse(
            forward_prose=q,
            forward_final_var=final,
            ask_var="h",
            closer=closer,
        )
        result["recipe"] = "distance_catchup_reverse"
        result["n_steps"] = n_steps
        result["direction"] = "reverse"
        return result

    raise RuntimeError("distance_catchup_recipe: couldn't sample")


# ─── Recipe 12: distance_meeting ────────────────────────────────────────────

def distance_meeting_recipe(rng: random.Random, n_steps: int = 2,
                             reverse: bool = False) -> dict:
    """Two vehicles start at opposite ends of distance D, moving toward each other."""
    for _try in range(100):
        r1 = rng.choice([30, 40, 50, 60, 70])
        r2 = rng.choice([40, 50, 60, 80, 90, 120])
        if r1 == r2:
            continue
        t = rng.randint(2, 6)
        d = (r1 + r2) * t

        names = rng.sample(NAMES, 2)
        ctx = Ctx.new(rng)
        ctx.protagonist = names[0]
        ctx.bind("r1", r1); ctx.bind("r2", r2); ctx.bind("d", d)

        if not reverse:
            q = (f"{names[0]} og {names[1]} kører af sted samtidig fra to byer "
                 f"med en afstand på {d} km og kører mod hinanden. "
                 f"{names[0]} kører med {r1} km/t, {names[1]} med {r2} km/t. "
                 f"Om hvor mange timer mødes de?")
        else:
            q = (f"{names[0]} og {names[1]} kører af sted samtidig fra to byer med "
                 f"ukendt afstand imellem, og kører mod hinanden. "
                 f"{names[0]} kører med {r1} km/t, {names[1]} med {r2} km/t.")

        Add("r1", "r2", "r_sum").apply(ctx)
        Div("d", "r_sum", "t").apply(ctx)
        final = "t"

        if n_steps >= 3:
            if not reverse:
                q += f" Hvor mange km har {names[0]} kørt, når de mødes?"
            Mul("r1", "t", "d1").apply(ctx)
            final = "d1"

        if not reverse:
            return ctx.render(q, final)

        final_val = int(ctx.n(final))
        if n_steps >= 3:
            q += f" {names[0]} har kørt {final_val} km, indtil de mødtes."
        else:
            q += f" De mødes efter {final_val} timer."
        # Prepend the forward-derived sum of speeds so the reverse chain's
        # {b}={r_sum} operand is grounded, not a bare constant.
        preamble = (f" Den samlede fart mod hinanden er "
                    f"{r1} + {r2} = {r1 + r2} km/t.")
        closer = preamble + " Beregn den oprindelige afstand mellem de to byer."
        result = ctx.render_reverse(
            forward_prose=q,
            forward_final_var=final,
            ask_var="d",
            closer=closer,
        )
        result["recipe"] = "distance_meeting_reverse"
        result["n_steps"] = n_steps
        result["direction"] = "reverse"
        return result

    raise RuntimeError("distance_meeting_recipe: couldn't sample")


# ─── Recipe 13: distance_avg (harmonic mean) ────────────────────────────────

def distance_avg_recipe(rng: random.Random, n_steps: int = 3) -> dict:
    """Round-trip: out at rout, back at rback, same distance."""
    for _try in range(100):
        rout = rng.choice([30, 40, 45, 50, 60, 70, 75, 80, 90, 100, 120, 150])
        rback = rng.choice([20, 30, 40, 45, 50, 60, 75, 80, 90, 100])
        if rout == rback:
            continue
        num = 2 * rout * rback
        den = rout + rback
        if num % den != 0:
            continue
        avg = num // den

        name = rng.choice(NAMES)
        vehicle = rng.choice(_VEHICLES)
        ctx = Ctx.new(rng)
        ctx.protagonist = name
        ctx.bind("two", 2)
        ctx.bind("rout", rout); ctx.bind("rback", rback)

        frame = maybe_frame(rng)
        openers = [
            f"{frame}{name} kører på sin {vehicle[0]} fra by A til by B med {rout} km/t og tilbage med {rback} km/t. Hvad er gennemsnitsfarten for hele turen?",
            f"{frame}på sin {vehicle[0]} kører {name} med {rout} km/t frem og {rback} km/t tilbage. Beregn gennemsnitsfarten for turen tur/retur.",
            f"{frame}{name} tager turen tur/retur: først med {rout} km/t, derefter tilbage med {rback} km/t. Hvad er gennemsnitsfarten?",
            f"{frame}{poss(name)} {vehicle[0]} kører med {rout} km/t og tilbage med {rback} km/t. Find gennemsnitsfarten.",
        ]
        q = rng.choice(openers)
        if not frame:
            q = q[0].upper() + q[1:]

        Mul("two", "rout", "two_rout").apply(ctx)
        Mul("two_rout", "rback", "numer").apply(ctx)
        Add("rout", "rback", "denom").apply(ctx)
        Div("numer", "denom", "avg").apply(ctx)
        final = "avg"

        if n_steps >= 5:
            candidates = [d for d in (60, 90, 120, 180, 240, 300)
                          if d % rout == 0 and d % rback == 0]
            if not candidates:
                continue
            d = rng.choice(candidates)
            ctx.bind("d", d)
            q += f" Hvis afstanden fra A til B er {d} km, hvor mange timer varer hele turen tur/retur?"
            Div("d", "rout", "t_out").apply(ctx)
            Div("d", "rback", "t_back").apply(ctx)
            Add("t_out", "t_back", "t_total").apply(ctx)
            final = "t_total"

        return ctx.render(q, final)

    raise RuntimeError("distance_avg_recipe: couldn't sample")


# ─── Recipe 14: age_simple (algebraic) ──────────────────────────────────────

_AGE_RELATIONS = [
    ("mor", "søn"), ("far", "datter"),
    ("onkel", "nevø"), ("bror", "søster"),
]


def age_simple_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """Age problem: old is r times younger; sum is known.
    Solve x + r*x = sum → (r+1)*x = sum → x = sum/(r+1).
    """
    for _try in range(100):
        ratio = rng.choice([2, 3, 4, 5])
        young = rng.randint(4, 20)
        old = ratio * young
        sum_now = young + old

        old_role, young_role = rng.choice(_AGE_RELATIONS)
        names = rng.sample(NAMES, 2)
        ask = rng.choice(["young", "old"])

        ctx = Ctx.new(rng)
        ctx.bind("sum_coef", ratio + 1)
        ctx.bind("zero", 0)
        ctx.bind("sum_now", sum_now)
        ctx.bind("ratio", ratio)

        mul_word = {2: "dobbelt så", 3: "tre gange så", 4: "fire gange så", 5: "fem gange så"}[ratio]
        target = "yngste" if ask == "young" else "ældste"
        openers = [
            f"{names[0]} er {mul_word} gammel som {names[1]}. "
            f"Tilsammen er de {sum_now} år. Hvor gammel er den {target}?",
            f"{poss(names[0])} alder er {ratio} gange {poss(names[1])}. "
            f"Summen af deres aldre er {sum_now}. Find den {target}s alder.",
            f"{names[1]} er x år gammel. {names[0]} er {ratio}x. "
            f"Tilsammen er de {sum_now} år gamle. Hvor gammel er den {target}?",
        ]
        q = rng.choice(openers)

        LinearSolve("sum_coef", "zero", "sum_now", "young",
                    var_name="x", lhs_shape=f"x + {ratio}*x").apply(ctx)
        if ask == "young":
            return ctx.render(q, "young")
        Mul("ratio", "young", "old").apply(ctx)
        return ctx.render(q, "old")

    raise RuntimeError("age_simple_recipe: couldn't sample")


# ─── Recipe 15: consec_first_as_x (algebraic) ───────────────────────────────

def consec_first_as_x_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """N consecutive integers summing to S. Setup: x + (x+1) + ... = S."""
    count = rng.choice([3, 4, 5, 6, 7, 8])
    ask_choices = ["smallest", "largest"] if count % 2 == 0 else ["smallest", "largest", "middle"]
    ask = rng.choice(ask_choices)
    start = rng.randint(2, 120)
    values = [start + i for i in range(count)]
    total = sum(values)
    const = sum(range(count))

    ctx = Ctx.new(rng)
    ctx.bind("n", count)
    ctx.bind("const", const)
    ctx.bind("total", total)

    lhs_terms = " + ".join(["x"] + [f"(x + {i})" for i in range(1, count)])
    what = {"smallest": "mindste", "largest": "største", "middle": "mellemste"}[ask]
    frame = maybe_frame(rng)
    openers = [
        f"{frame}summen af {count} på hinanden følgende heltal er {total}. Hvad er det {what}?",
        f"{frame}{count} på hinanden følgende heltal summerer til {total}. Find det {what}.",
        f"{frame}hvis {count} på hinanden følgende heltal har en sum på {total}, hvad er så det {what}?",
        f"{frame}der er {count} på hinanden følgende heltal, hvis sum er lig {total}. Beregn det {what}.",
        f"{frame}find det {what} af {count} på hinanden følgende heltal, hvis sum er {total}.",
        f"{frame}når {count} på hinanden følgende heltal summeres til {total}, hvad er så det {what}?",
        f"{frame}hvilket af {count} på hinanden følgende heltal er det {what}, hvis deres sum er {total}?",
    ]
    q = rng.choice(openers)
    if not frame:
        q = q[0].upper() + q[1:]

    LinearSolve("n", "const", "total", "x",
                var_name="x", lhs_shape=lhs_terms).apply(ctx)

    if ask == "smallest":
        return ctx.render(q, "x")
    offset = (count - 1) if ask == "largest" else (count // 2)
    ctx.bind("offset", offset)
    Add("x", "offset", "result").apply(ctx)
    return ctx.render(q, "result")


# ─── Recipe 16: ratio_algebra ───────────────────────────────────────────────

def ratio_algebra_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """Ratio via algebra: r1*x + r2*x = total → (r1+r2)*x = total → x = total/(r1+r2)."""
    names = rng.sample(NAMES, 2)
    obj = rng.choice(OBJECT_NOUNS)

    for _try in range(100):
        a, b = rng.choice([(2, 3), (3, 5), (1, 4), (3, 4), (1, 3), (4, 5), (3, 7), (2, 5)])
        unit = rng.randint(3, 30)
        total = (a + b) * unit
        if total > 300:
            continue

        ctx = Ctx.new(rng)
        ctx.bind("r_sum", a + b)
        ctx.bind("zero", 0)
        ctx.bind("total", total, noun=obj)
        ctx.bind("ra", a)
        ctx.bind("rb", b)

        ask = rng.choice(["direct-a", "direct-b", "larger"])
        which = {"direct-a": f"{poss(names[0])} andel",
                 "direct-b": f"{poss(names[1])} andel",
                 "larger": "største andel"}[ask]
        openers = [
            f"{names[0]} og {names[1]} deler {render_qty(total, obj)} i forholdet {a}:{b}. Hvad er {which}?",
            f"I forholdet {a}:{b} deler {names[0]} og {names[1]} {render_qty(total, obj)}. Find {which}.",
            f"Det samlede antal er {render_qty(total, obj)}, delt i forholdet {a}:{b} mellem {names[0]} og {names[1]}. Hvad er {which}?",
        ]
        q = rng.choice(openers)

        LinearSolve("r_sum", "zero", "total", "x",
                    var_name="x", lhs_shape=f"{a}*x + {b}*x").apply(ctx)

        target_coef = "ra" if ask == "direct-a" else ("rb" if ask == "direct-b" else ("rb" if b >= a else "ra"))
        Mul(target_coef, "x", "part").apply(ctx)
        return ctx.render(q, "part")

    raise RuntimeError("ratio_algebra_recipe: couldn't sample")


RECIPES = {
    "ratio_parts": ratio_parts_recipe,
    "ratio_diff": ratio_diff_recipe,
    "ratio_fraction": ratio_fraction_recipe,
    "percent": percent_recipe,
    "average": average_recipe,
    "fraction_cascade": fraction_cascade_recipe,
    "consec_avg": consec_avg_recipe,
    "inverse_rate": inverse_rate_recipe,
    "distance_direct": distance_direct_recipe,
    "distance_catchup": distance_catchup_recipe,
    "distance_meeting": distance_meeting_recipe,
    "distance_avg": distance_avg_recipe,
    "coin_assume": coin_assume_recipe,
    "age_simple": age_simple_recipe,
    "consec_first_as_x": consec_first_as_x_recipe,
    "ratio_algebra": ratio_algebra_recipe,
}


REVERSABLE_RECIPES = {
    "ratio_parts",
    "fraction_cascade",
    "ratio_fraction",
    "percent",
    "ratio_diff",
    "distance_meeting",
    "distance_catchup",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recipe", choices=list(RECIPES.keys()), default=None)
    ap.add_argument("--reverse-frac", type=float, default=0.0,
                    help="Fraction (0-1) of generated rows using reverse mode.")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.count > 0:
        for _ in range(args.count):
            recipe_name = args.recipe or rng.choice(list(RECIPES.keys()))
            recipe = RECIPES[recipe_name]
            n_steps = rng.choices([2, 3, 4, 5], weights=[3, 3, 2, 1])[0]
            do_reverse = (
                args.reverse_frac > 0
                and recipe_name in REVERSABLE_RECIPES
                and rng.random() < args.reverse_frac
            )
            try:
                if do_reverse:
                    p = recipe(rng, n_steps=n_steps, reverse=True)
                    p.setdefault("recipe", recipe_name)
                    p.setdefault("n_steps", n_steps)
                    p.setdefault("direction", "reverse")
                else:
                    p = recipe(rng, n_steps=n_steps)
                    p["recipe"] = recipe_name
                    p["n_steps"] = n_steps
                    p["direction"] = "forward"
                print(json.dumps(p, ensure_ascii=False))
            except RuntimeError:
                continue
        return

    # sample mode
    for name, fn in RECIPES.items():
        print("=" * 70)
        print(name)
        print("=" * 70)
        for _ in range(2):
            try:
                p = fn(rng)
                print(f"Q: {p['question']}")
                print(f"A: {p['answer']}")
                print(f"   chain: {p['chain_lines']}\n")
            except RuntimeError as e:
                print(f"   ({e})\n")


if __name__ == "__main__":
    main()
