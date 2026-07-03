"""Compositional word-problem generator POC (v2).

v1 → v2 fixes:
  1. Integer divisibility enforced via reject-and-resample.
  2. Grammatical agreement: `render_qty(1, noun)` → "1 studento";
     `render_qty(5, noun)` → "5 studentojn" (with case).
  3. Richer prose libraries — 6-8 variants per op, split by role
     (first_step vs chained_step).

New migration: `Percent` (3 strategies: direct, decimal, multiplier)
with 5 scenarios (discount, markup, tax, of-amount, saving).

Usage:
    uv run python scripts/wp_compose.py
    uv run python scripts/wp_compose.py --count 200 > sample.jsonl
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

from esperanto_lm.morphology import decompose  # noqa: F401
# Reuse gen_algebra_v2's sympy-native rendering for the algebraic Op.
import sympy as sp
from gen_algebra_v2 import render as _alg_render, render_eq as _alg_render_eq


# ─── Vocab ──────────────────────────────────────────────────────────────────

NAMES = ["Ana", "Petro", "Marko", "Sara", "Elena", "Luka", "Nia", "Ivo",
         "Karlo", "Mira", "Julia", "Roberto", "Klara", "Vasilis", "Maria",
         "Aleksandro", "Olga", "Erik", "Zofia", "Kristina", "Dario", "Amalia",
         "Nikolao", "Katarina", "Filip", "Anastasia", "Tomas", "Rebeka",
         "Georgi", "Silvia", "Adam", "Helena", "Miloŝ", "Beatrice",
         "Aleksio", "Dagmara", "Bruno", "Ines", "Valentin", "Renata"]

# Scenario framings sampled at question-open time. Not every recipe uses them,
# but recipes that opt in inject one of these leading phrases.
SCENARIO_FRAMES = [
    "En 2015,",  "Dum la lernojaro,",  "Iun tagon,",  "Post la lernejo,",
    "En la merkato,",  "Dum somero,",  "Antaŭ la festo,",  "Hodiaŭ matene,",
    "Pasintan semajnon,",  "En festo,",  "Post la manĝo,",  "En la vendejo,",
]

# For recipes that render entities from any pool, an extended object pool
# broadens the surface. Each is (nom_sg, nom_pl, acc_sg, acc_pl).

# each tuple is (nom_sg, nom_pl, acc_sg, acc_pl)
CHILDLIKE_NOUNS = [
    ("infano",  "infanoj",  "infanon",  "infanojn"),
    ("knabo",   "knaboj",   "knabon",   "knabojn"),
    ("knabino", "knabinoj", "knabinon", "knabinojn"),
    ("studento","studentoj","studenton","studentojn"),
    ("lernanto","lernantoj","lernanton","lernantojn"),
    ("kliento", "klientoj", "klienton", "klientojn"),
    ("gasto",   "gastoj",   "gaston",   "gastojn"),
    ("vizitanto","vizitantoj","vizitanton","vizitantojn"),
]
GROUPING_NOUNS = [
    ("grupo",   "grupoj",   "grupon",   "grupojn"),
    ("teamo",   "teamoj",   "teamon",   "teamojn"),
    ("klaso",   "klasoj",   "klason",   "klasojn"),
    ("tablo",   "tabloj",   "tablon",   "tablojn"),
    ("aŭtobuso","aŭtobusoj","aŭtobuson","aŭtobusojn"),
    ("ĉambro",  "ĉambroj",  "ĉambron",  "ĉambrojn"),
    ("kesto",   "kestoj",   "keston",   "kestojn"),
    ("vagono",  "vagonoj",  "vagonon",  "vagonojn"),
]
OBJECT_NOUNS = [
    ("libro",  "libroj",  "libron",  "librojn"),
    ("krajono","krajonoj","krajonon","krajonojn"),
    ("pomo",   "pomoj",   "pomon",   "pomojn"),
    ("ludilo", "ludiloj", "ludilon", "ludilojn"),
    ("floro",  "floroj",  "floron",  "florojn"),
    ("bulko",  "bulkoj",  "bulkon",  "bulkojn"),
    ("bileto", "biletoj", "bileton", "biletojn"),
    ("kajero", "kajeroj", "kajeron", "kajerojn"),
    ("plumo",  "plumoj",  "plumon",  "plumojn"),
    ("kekso",  "keksoj",  "kekson",  "keksojn"),
    ("marmoro","marmoroj","marmoron","marmorojn"),
    ("ovo",    "ovoj",    "ovon",    "ovojn"),
    ("stelo",  "steloj",  "stelon",  "stelojn"),
    ("ĉokolado","ĉokoladoj","ĉokoladon","ĉokoladojn"),
    ("kartodo","kartodoj","kartodon","kartodojn"),
    ("koverto","kovertoj","koverton","kovertojn"),
]

Noun = tuple[str, str, str, str]  # (nom_sg, nom_pl, acc_sg, acc_pl)

# Currency: full Noun tuple so we can render with case agreement
EUR: Noun = ("eŭro", "eŭroj", "eŭron", "eŭrojn")

# Shop items — Noun tuples so we can render "biciklon" (acc_sg) or "biciklo" (nom_sg)
SHOP_ITEMS: list[Noun] = [
    ("biciklo",   "bicikloj",   "biciklon",   "biciklojn"),
    ("ĉemizo",    "ĉemizoj",    "ĉemizon",    "ĉemizojn"),
    ("libro",     "libroj",     "libron",     "librojn"),
    ("komputilo", "komputiloj", "komputilon", "komputilojn"),
]
COUNT_ITEMS: list[Noun] = [
    ("studento", "studentoj", "studenton", "studentojn"),
    ("lernanto", "lernantoj", "lernanton", "lernantojn"),
    ("infano",   "infanoj",   "infanon",   "infanojn"),
]
# Test-score / grade scenarios — used by average recipe
SUBJECT_NOUNS: list[Noun] = [
    ("matematiko", "matematikoj", "matematikon", "matematikojn"),
    ("historio",   "historioj",   "historion",   "historiojn"),
    ("kemio",      "kemioj",      "kemion",      "kemiojn"),
    ("biologio",   "biologioj",   "biologion",   "biologiojn"),
]


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
        # Prose paragraphs joined; end-marker last.
        answer = " ".join(self.prose) + f" #### {final_str}"
        return {
            "question": question,
            "answer": answer.strip(),
            "chain_lines": self.chain,
            "final": final_str,
        }


# ─── Morphology helpers ─────────────────────────────────────────────────────

def maybe_frame(rng: random.Random, p: float = 0.35) -> str:
    """Optionally prepend a scenario frame to a question. p=0.35 means 35%
    of samples get "En 2015, ..." / "Iun tagon, ..." prefixes.

    Returned as a leading string or empty. Use like:
        q = maybe_frame(rng) + main_question
    """
    if rng.random() < p:
        return rng.choice(SCENARIO_FRAMES) + " "
    return ""


def render_qty(n: int, noun: Noun, case: str = "nom") -> str:
    """`5 studentoj` / `1 studento` — proper singular/plural agreement.

    case: 'nom' or 'acc' — picks the right column of the noun tuple.
    """
    if case == "acc":
        singular, plural = noun[2], noun[3]
    else:
        singular, plural = noun[0], noun[1]
    return f"{n} {singular if n == 1 else plural}"


# alias used in question templates — always accusative for direct objects of
# transitive verbs like `havas`, `aĉetas`, `kostas`, `donas`, `manĝas`
def qty_acc(n: int, noun: Noun) -> str:
    return render_qty(n, noun, case="acc")


def fmt_num(x: float) -> str:
    if x == int(x):
        return str(int(x))
    # trim trailing zeros for decimals (e.g. 0.20 → 0.2)
    return f"{x:g}"


# ─── Ops ────────────────────────────────────────────────────────────────────

class Op:
    """Base class for math ops. Subclasses set kind + provide apply().

    Ops render themselves in TWO roles:
      - `first_step`  : the op that opens the reasoning chain
      - `chained`     : an op that consumes an already-derived intermediate

    Prose libraries live per-role so a single problem doesn't repeat the
    same opener twice.
    """
    kind: str = ""

    def __init__(self, lhs: str, rhs: str, out: str):
        self.lhs = lhs
        self.rhs = rhs
        self.out = out

    def apply(self, ctx: Ctx) -> None:
        raise NotImplementedError

    def _chain_line(self, ctx: Ctx, a: float, sym: str, b: float, c: float) -> None:
        ctx.chain.append(f"{fmt_num(a)} {sym} {fmt_num(b)} = {fmt_num(c)}")


class Mul(Op):
    kind = "mul"
    def apply(self, ctx: Ctx) -> None:
        a, b = ctx.n(self.lhs), ctx.n(self.rhs)
        c = a * b
        ctx.bind(self.out, c, noun=ctx.get(self.lhs).noun)
        self._chain_line(ctx, a, "*", b, c)
        role = "first" if not ctx.prose else "chained"
        variants = _MUL_PROSE[role]
        ctx.prose.append(ctx.rng.choice(variants).format(a=fmt_num(a), b=fmt_num(b), c=fmt_num(c)))


class Add(Op):
    kind = "add"
    def apply(self, ctx: Ctx) -> None:
        a, b = ctx.n(self.lhs), ctx.n(self.rhs)
        c = a + b
        ctx.bind(self.out, c, noun=ctx.get(self.lhs).noun)
        self._chain_line(ctx, a, "+", b, c)
        role = "first" if not ctx.prose else "chained"
        ctx.prose.append(ctx.rng.choice(_ADD_PROSE[role]).format(a=fmt_num(a), b=fmt_num(b), c=fmt_num(c)))


class Sub(Op):
    kind = "sub"
    def apply(self, ctx: Ctx) -> None:
        a, b = ctx.n(self.lhs), ctx.n(self.rhs)
        c = a - b
        ctx.bind(self.out, c, noun=ctx.get(self.lhs).noun)
        self._chain_line(ctx, a, "-", b, c)
        role = "first" if not ctx.prose else "chained"
        ctx.prose.append(ctx.rng.choice(_SUB_PROSE[role]).format(a=fmt_num(a), b=fmt_num(b), c=fmt_num(c)))


class Div(Op):
    kind = "div"
    def apply(self, ctx: Ctx) -> None:
        a, b = ctx.n(self.lhs), ctx.n(self.rhs)
        # Framework contract: caller must ensure divisibility. We assert.
        assert a % b == 0, f"Div {a}/{b} not integer — caller must resample."
        c = a / b
        ctx.bind(self.out, c, noun=ctx.get(self.lhs).noun)
        self._chain_line(ctx, a, "/", b, c)
        role = "first" if not ctx.prose else "chained"
        ctx.prose.append(ctx.rng.choice(_DIV_PROSE[role]).format(a=fmt_num(a), b=fmt_num(b), c=fmt_num(c)))


class Frac(Op):
    """(num/denom) * base → result.  Rejects non-integer results at apply-time.
    lhs = base_var; rhs unused (num, denom are constructor args); out = result.
    """
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
        template = ctx.rng.choice(_FRAC_PROSE[role])
        ctx.prose.append(template.format(n=self.num, d=self.denom, b=fmt_num(base), r=fmt_num(result)))


class Avg(Op):
    """Average of a list of variables. inputs: `values_key` in ctx.vars_lists.
    Constructor: Avg([var_name1, var_name2, ...], out).
    Emits chain: "sum / N = avg".
    """
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
        template = ctx.rng.choice(_AVG_PROSE[role])
        ctx.prose.append(template.format(vals=sum_str, t=fmt_num(total), n=n, a=fmt_num(avg)))


class Pct(Op):
    """X% of base → amount.  Written in one of 3 notation styles.
    self.lhs = pct_var; self.rhs = base_var; self.out = amount_var
    style: 'direct' | 'decimal' | 'multiplier'
    (multiplier only makes sense in composite ops — see PctChange.)
    """
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
            # emit intermediate: pct/100 = decimal
            dec = pct / 100
            dec_str = fmt_num(dec)
            ctx.chain.append(f"{pct} / 100 = {dec_str}")
            ctx.chain.append(f"{dec_str} * {fmt_num(base)} = {fmt_num(amount)}")
            role = "first" if not ctx.prose else "chained"
            template = ctx.rng.choice(_PCT_DECIMAL_PROSE[role])
            ctx.prose.append(template.format(p=pct, d=dec_str, b=fmt_num(base), a=fmt_num(amount)))
        else:
            # direct: pct/100 * base = amount
            ctx.chain.append(f"{pct} / 100 * {fmt_num(base)} = {fmt_num(amount)}")
            role = "first" if not ctx.prose else "chained"
            template = ctx.rng.choice(_PCT_DIRECT_PROSE[role])
            ctx.prose.append(template.format(p=pct, b=fmt_num(base), a=fmt_num(amount)))


class LinearSolve(Op):
    """Solve `sum_coef * x + const = target` for x. Renders in the didactic
    style used by gen_algebra_v2 — same "same op both sides" intermediates
    the SFT model already sees.

    - lhs         : ctx var holding sum_coef
    - rhs         : ctx var holding const
    - target_var  : ctx var holding target (equation RHS)
    - out         : ctx var name for the solved value
    - var_name    : the symbolic variable to use in rendered lines (default 'x')
    - lhs_shape   : optional str shown *before* combining (e.g. 'x + 3*x')
                    for word-problems where the equation naturally has terms
                    to combine — one extra prose line, then the didactic block.
    """
    kind = "linsolve"
    def __init__(self, sum_coef_var: str, const_var: str, target_var: str,
                 out: str, var_name: str = "x", lhs_shape: str | None = None):
        super().__init__(sum_coef_var, const_var, out)
        self.target_var = target_var
        self.var_name = var_name
        self.lhs_shape = lhs_shape

    def apply(self, ctx: Ctx) -> None:
        c = int(ctx.n(self.lhs))       # combined coefficient
        b = int(ctx.n(self.rhs))       # constant term
        t = int(ctx.n(self.target_var))
        assert c != 0, "LinearSolve: coef cannot be zero"
        num = t - b
        assert num % c == 0, f"LinearSolve: ({t}-{b})/{c} not integer"

        # sympy sanity check — matches gen_algebra_v2 pattern
        sv = sp.Symbol(self.var_name)
        eq = sp.Eq(c * sv + b, t)
        sol = sp.solve(eq, sv)
        assert len(sol) == 1 and int(sol[0]) == num // c, \
            f"LinearSolve: sympy disagrees ({sol} vs {num // c})"
        x = num // c
        ctx.bind(self.out, x)

        v = self.var_name
        # Format `c*v +/- b` cleanly (drop `+ 0`)
        def _combined() -> str:
            if b == 0:
                return f"{c}{v}"
            return f"{c}{v} + {b}" if b > 0 else f"{c}{v} - {-b}"

        # If caller provided a shape hint (e.g. "x + 3*x"), show the raw setup first.
        if self.lhs_shape is not None:
            ctx.chain.append(f"{self.lhs_shape} = {t}")
        # Combined-terms line — always shown
        ctx.chain.append(f"{_combined()} = {t}")
        # Isolate the coef*v term (subtract the constant) if b ≠ 0
        if b != 0:
            op_str = "-" if b > 0 else "+"
            ctx.chain.append(f"{c}{v} = {t} {op_str} {abs(b)}")
            ctx.chain.append(f"{c}{v} = {num}")
        # divide-both-sides didactic line + final
        ctx.chain.append(f"{c}{v} / {c} = {num} / {c}")
        ctx.chain.append(f"{v} = {x}")

        # single prose block — narrative frame around the algebra
        role = "first" if not ctx.prose else "chained"
        template = ctx.rng.choice(_LINSOLVE_PROSE[role])
        ctx.prose.append(template.format(
            v=v,
            eq=f"{c}{v} + {b}" if b > 0 else f"{c}{v} - {-b}" if b < 0 else f"{c}{v}",
            t=t, c=c, num=num, x=x,
        ))


# ─── Prose libraries ────────────────────────────────────────────────────────

_MUL_PROSE = {
    "first": [
        "Estas {a} * {b} = {c} entute.",
        "{a} fojoj {b} egalas {c}.",
        "La totala nombro estas {a} * {b} = {c}.",
        "Ni multiplikas: {a} * {b} = {c}.",
        "{a} po {b} donas {a} * {b} = {c}.",
        "Kalkulo: {a} * {b} = {c}.",
        "Unue kalkulu la produkton: {a} * {b} = {c}.",
        "La produkto de {a} kaj {b} estas {c}.",
        "Multiplikante {a} per {b}, ni ricevas {c}.",
        "Ni komencu multiplikante: {a} * {b} = {c}.",
        "La komenca kalkulo: {a} * {b} = {c}.",
        "Trovi la produkton de {a} kaj {b}: {a} * {b} = {c}.",
        "Simple: {a} * {b} = {c}.",
        "Ni notu ke {a} * {b} = {c}.",
        "Rezulto de multipliko: {a} * {b} = {c}.",
    ],
    "chained": [
        "Poste, {a} * {b} = {c}.",
        "Nun ni multiplikas per {b}: {a} * {b} = {c}.",
        "{a} * {b} = {c}.",
        "Tio donas {a} * {b} = {c}.",
        "Multobligante: {a} * {b} = {c}.",
        "Sekve, {a} * {b} = {c}.",
        "Kaj do {a} * {b} = {c}.",
        "Tial {a} * {b} = {c}.",
        "Nun kalkulu: {a} * {b} = {c}.",
        "Multiplikante per {b}: {a} * {b} = {c}.",
        "Rezulto: {a} * {b} = {c}.",
        "Do {a} * {b} = {c}.",
        "Ni multiplikas kaj ricevas {c}.",
    ],
}
_ADD_PROSE = {
    "first": [
        "Ni aldonas: {a} + {b} = {c}.",
        "Entute estas {a} + {b} = {c}.",
        "Kune: {a} + {b} = {c}.",
        "{a} plus {b} donas {c}.",
        "La sumo estas {a} + {b} = {c}.",
        "Sumigante, {a} + {b} = {c}.",
        "Adicio: {a} + {b} = {c}.",
        "La kombinita valoro estas {a} + {b} = {c}.",
        "Kalkulo de sumo: {a} + {b} = {c}.",
        "Ni sumigu: {a} + {b} = {c}.",
        "La totalo el {a} kaj {b} estas {c}.",
        "Aldonante {a} kaj {b}, ni ricevas {c}.",
    ],
    "chained": [
        "Aldonu {b} pliajn: {a} + {b} = {c}.",
        "Nun estas {a} + {b} = {c}.",
        "Kune, {a} + {b} = {c}.",
        "Tial: {a} + {b} = {c}.",
        "{a} + {b} = {c}.",
        "Poste sumigu: {a} + {b} = {c}.",
        "Sekve {a} + {b} = {c}.",
        "Tio kondukas al {a} + {b} = {c}.",
        "Nun aldonu {b}: {a} + {b} = {c}.",
        "La nova sumo: {a} + {b} = {c}.",
        "Post aldono: {a} + {b} = {c}.",
        "Kombine, {a} + {b} = {c}.",
        "Tial la kombina rezulto estas {c}.",
    ],
}
_SUB_PROSE = {
    "first": [
        "Ni subtrahas: {a} - {b} = {c}.",
        "Restas {a} - {b} = {c}.",
        "{a} minus {b} egalas {c}.",
        "Subtrahante {b} de {a}: {a} - {b} = {c}.",
        "La diferenco estas {a} - {b} = {c}.",
        "Post subtraho: {a} - {b} = {c}.",
        "Kalkulo de diferenco: {a} - {b} = {c}.",
        "Ni forigos {b}: {a} - {b} = {c}.",
        "La resto egalas {a} - {b} = {c}.",
        "Redukto: {a} - {b} = {c}.",
    ],
    "chained": [
        "Post kiam {b} foriras, restas {a} - {b} = {c}.",
        "Restas {a} - {b} = {c}.",
        "Subtrahante {b}: {a} - {b} = {c}.",
        "{a} - {b} = {c}.",
        "Tial restas {a} - {b} = {c}.",
        "Nun {a} - {b} = {c}.",
        "Do la diferenco: {a} - {b} = {c}.",
        "Reduktante per {b}: {a} - {b} = {c}.",
        "Post forigo de {b}: {a} - {b} = {c}.",
        "La restanta valoro estas {c}.",
        "Sekve la diferenco: {a} - {b} = {c}.",
        "Ĉi tio lasas {a} - {b} = {c}.",
    ],
}
_DIV_PROSE = {
    "first": [
        "Ni dividas: {a} / {b} = {c}.",
        "{a} / {b} = {c}.",
        "Ĉiu parto havas {a} / {b} = {c}.",
        "Divizio: {a} / {b} = {c}.",
        "La kvociento estas {a} / {b} = {c}.",
        "Dividante {a} per {b}, ni ricevas {c}.",
        "Distribuu {a} inter {b}: {a} / {b} = {c}.",
        "Ĉiu ricevas {a} / {b} = {c}.",
        "Egalpartige: {a} / {b} = {c}.",
        "Kalkulo de divizio: {a} / {b} = {c}.",
    ],
    "chained": [
        "Dividante inter {b}: {a} / {b} = {c}.",
        "Ĉiu grupo havas {a} / {b} = {c}.",
        "{a} / {b} = {c}.",
        "Tial ĉiu ricevas {a} / {b} = {c}.",
        "Sekve {a} / {b} = {c}.",
        "Poste dividu: {a} / {b} = {c}.",
        "Distribuante {a} tra {b}: {a} / {b} = {c}.",
        "Ĉiu parto: {a} / {b} = {c}.",
        "Nun dividante per {b}: {a} / {b} = {c}.",
        "Do {a} / {b} = {c}.",
        "La kvociento estas {a} / {b} = {c}.",
    ],
}
_LINSOLVE_PROSE = {
    "first": [
        "Estu {v} la nekonato. La ekvacio: {eq} = {t}. Ni solvas por {v}: {v} = {x}.",
        "Estu {v} la nekonato. Do {eq} = {t}. Solvante, {v} = {x}.",
        "Ni indiku la nekonaton per {v}. La ekvacio iĝas {eq} = {t}. Post izolado: {v} = {x}.",
    ],
    "chained": [
        "Nun ni starigas ekvacion: {eq} = {t}. Solvo: {v} = {x}.",
        "La sekva ekvacio estas {eq} = {t}, do {v} = {x}.",
    ],
}
_FRAC_PROSE = {
    "first": [
        "Ni trovas {n}/{d} el {b}: {n}/{d} * {b} = {r}.",
        "{n}/{d} de {b} estas {r}.",
        "Kalkulu {n}/{d} de {b}: {n}/{d} * {b} = {r}.",
    ],
    "chained": [
        "Poste {n}/{d} el {b}: {n}/{d} * {b} = {r}.",
        "{n}/{d} el {b} = {r}.",
        "Nun trovi {n}/{d}: {n}/{d} * {b} = {r}.",
    ],
}
_AVG_PROSE = {
    "first": [
        "Sumo: {vals} = {t}. Meza valoro: {t} / {n} = {a}.",
        "Aldonu ĉiujn: {vals} = {t}. Meza: {t} / {n} = {a}.",
        "Meza valoro estas ({vals}) / {n} = {a}.",
    ],
    "chained": [
        "Nun la meza: ({vals}) / {n} = {a}.",
        "Sumo estas {t}, do meza = {t} / {n} = {a}.",
    ],
}
_PCT_DIRECT_PROSE = {
    "first": [
        "Ni kalkulas {p}% el {b}: {p}/100 * {b} = {a}.",
        "{p}% de {b} estas {p}/100 * {b} = {a}.",
        "Trovi {p}%: {p}/100 * {b} = {a}.",
    ],
    "chained": [
        "Poste ni trovas {p}% de {b}: {p}/100 * {b} = {a}.",
        "{p}% el {b} = {p}/100 * {b} = {a}.",
    ],
}
_PCT_DECIMAL_PROSE = {
    "first": [
        "Unue transformu {p}% al decimalo: {p}/100 = {d}. Nun {d} * {b} = {a}.",
        "{p}% = {d}, tial {d} * {b} = {a}.",
    ],
    "chained": [
        "Konvertu {p}% al decimalo: {p}/100 = {d}. Tiam {d} * {b} = {a}.",
        "Nun {p}% = {d}, do {d} * {b} = {a}.",
    ],
}


# ─── Recipe 1: ratio_parts ──────────────────────────────────────────────────

def ratio_parts_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """N groups × K per group → total. Optionally: minus absent, then / packs."""
    # Resample outer parameters until (n_steps=4 case) integer division works.
    for _try in range(100):
        ctx = Ctx.new(rng)
        child = rng.choice(CHILDLIKE_NOUNS)
        group = rng.choice(GROUPING_NOUNS)

        n_groups = rng.randint(3, 12)
        per_group = rng.randint(4, 15)
        ctx.bind("groups", n_groups, noun=group)
        ctx.bind("per_group", per_group, noun=child)

        # 6 sentence-structure variants + optional scenario frame
        frame = maybe_frame(rng)
        openers = [
            f"{frame}estas {render_qty(n_groups, group)} kun {render_qty(per_group, child)} en ĉiu.",
            f"{frame}en {render_qty(n_groups, group)}, ĉiu enhavas {render_qty(per_group, child)}.",
            f"{frame}{ctx.protagonist} vidas {qty_acc(n_groups, group)}, ĉiun kun {render_qty(per_group, child)}.",
            f"{frame}ĉiu el la {render_qty(n_groups, group)} havas {qty_acc(per_group, child)}.",
            f"{frame}oni disdonis {render_qty(per_group, child)} en ĉiun de {render_qty(n_groups, group)}.",
            f"{frame}{render_qty(n_groups, group)} estas plenaj de {render_qty(per_group, child)} ĉiu.",
        ]
        # capitalize first char if frame was empty
        opener = rng.choice(openers)
        if not frame:
            opener = opener[0].upper() + opener[1:]
        q = [opener]
        Mul("groups", "per_group", "total").apply(ctx)
        final_var = "total"

        if n_steps >= 3:
            absent = rng.randint(1, min(6, int(ctx.n("total")) // 2))
            ctx.bind("absent", absent, noun=child)
            q.append(f"{render_qty(absent, child, case='nom')} forestas.")
            Sub("total", "absent", "present").apply(ctx)
            final_var = "present"

        if n_steps >= 4:
            cur = int(ctx.n(final_var))
            divisors = [k for k in (2, 3, 4, 5, 6) if cur % k == 0 and k <= cur]
            if not divisors:
                continue  # resample outer params
            n_packs = rng.choice(divisors)
            ctx.bind("packs", n_packs)
            q.append(f"Ili dividas sin en {n_packs} egalajn grupojn.")
            Div(final_var, "packs", "per_pack").apply(ctx)
            final_var = "per_pack"

        # 6 closing-question variants
        closers = [
            f"Kiom da {child[1]} estas en la fina rezulto?",
            f"Kiu estas la fina nombro de {child[1]}?",
            f"Kalkulu la finan nombron de {child[1]}.",
            f"Trovu kiom da {child[1]} restas fine.",
            f"Kiom da {child[1]} estas fine?",
            f"Determinu la finan kvanton de {child[1]}.",
        ]
        q.append(rng.choice(closers))
        return ctx.render(" ".join(q), final_var)

    raise RuntimeError("ratio_parts_recipe: couldn't sample divisible params in 100 tries")


# ─── Recipe 2: percent ──────────────────────────────────────────────────────

def percent_recipe(rng: random.Random, n_steps: int = 2, op: str | None = None) -> dict:
    """Percent problems in 3 notation styles × 5 scenarios.

    n_steps=2: single percentage (of-amount, saving)  or  base ± pct%  (discount, markup, tax)
    n_steps=3: stacked — e.g. discount then tax
    """
    ops = ["discount", "markup", "tax", "of-amount", "saving"]
    if op is None:
        op = rng.choice(ops)
    style = rng.choice(["direct", "decimal", "multiplier"])
    # multiplier only makes sense with a ± change; if of-amount/saving, fall through
    if op in ("of-amount", "saving"):
        style = rng.choice(["direct", "decimal"])

    # Some ops only fit certain scenarios — filter up front instead of retrying.
    if op in ("discount", "markup", "tax", "saving"):
        scenario_kind = "shop"
    elif op == "of-amount":
        scenario_kind = rng.choice(["shop", "count"])

    for _try in range(100):
        pct = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80])
        g = math.gcd(pct, 100)
        step = 100 // g
        base = step * rng.randint(2, 40)

        # For n_steps=3 stacking: need pct2 that divides the discounted result
        pct2 = None
        if n_steps >= 3 and op == "discount":
            amount = base * pct // 100
            res_val = base - amount
            for _p2 in [5, 10, 15, 20, 25]:
                if (res_val * _p2) % 100 == 0:
                    pct2 = _p2
                    break
            if pct2 is None:
                continue  # resample outer params

        ctx = Ctx.new(rng)
        ctx.bind("pct", pct)
        ctx.bind("base", base, noun=None)
        p = ctx.protagonist

        if scenario_kind == "shop":
            item: Noun = rng.choice(SHOP_ITEMS)
            base_qty_acc = qty_acc(base, EUR)     # "200 eŭrojn"
            base_qty_nom = render_qty(base, EUR)  # "200 eŭroj"
            item_acc = item[2]                    # "biciklon"
            item_nom = item[0]                    # "biciklo"

            frame = maybe_frame(rng)
            if op == "discount":
                q = frame + rng.choice([
                    f"{p} aĉetas {item_acc} kiu kostas {base_qty_acc}. "
                    f"La vendejo donas rabaton de {pct}%. Kiom {p} pagas?",
                    f"la origina prezo de {item_nom} estas {base_qty_nom}. "
                    f"Kun {pct}% rabato, kiu estas la nova prezo?",
                    f"{item_nom} kostas {base_qty_acc}. "
                    f"Kun rabato de {pct}%, kiom {p} devas pagi?",
                    f"{p} vidas {item_acc} je {base_qty_acc}, kun {pct}% rabato. "
                    f"Kalkulu la finan prezon.",
                ])
            elif op == "markup":
                q = frame + rng.choice([
                    f"la prezo de {item_nom} estis {base_qty_nom} sed pliiĝis je {pct}%. "
                    f"Kiu estas la nova prezo?",
                    f"{p} havas {item_acc} kiu kostas {base_qty_acc}. "
                    f"La prezo pliiĝas je {pct}%. Kiom kostas nun?",
                    f"{item_nom} kostis {base_qty_acc}. "
                    f"Post pliiĝo de {pct}%, kiu estas la nova prezo?",
                ])
            elif op == "tax":
                q = frame + rng.choice([
                    f"{p} aĉetas {item_acc} por {base_qty_nom}. "
                    f"La imposto estas {pct}%. Kiom entute {p} pagas?",
                    f"{item_nom} kostas {base_qty_acc}. Kun {pct}% imposto, "
                    f"kiu estas la totalo?",
                    f"la prezo de {item_nom} estas {base_qty_nom}, "
                    f"kaj oni aldonas {pct}% imposton. Kiom estas la finkosto?",
                ])
            elif op == "of-amount":
                q = frame + rng.choice([
                    f"{p} kalkulis {pct}% de {base_qty_nom}. Kiu estas la rezulto?",
                    f"trovu {pct}% el {base_qty_nom}.",
                    f"kiom estas {pct}% de {base_qty_nom}?",
                ])
            elif op == "saving":
                q = frame + rng.choice([
                    f"{p} aĉetis {item_acc} kiu kostis {base_qty_acc} "
                    f"kun {pct}% rabato. Kiom da {EUR[1]} {p} ŝparis?",
                    f"{item_nom} kostis {base_qty_acc}, kun rabato de {pct}%. "
                    f"Kiom {p} ŝparis?",
                    f"{p} akiris {pct}% rabaton sur {item_acc} de {base_qty_acc}. "
                    f"Kalkulu la ŝparon.",
                ])
            # Capitalize first char if no frame
            if not frame:
                q = q[0].upper() + q[1:]
        else:  # count
            item = rng.choice(COUNT_ITEMS)
            noun_acc_pl = item[3]   # "studentojn"
            noun_nom_pl = item[1]   # "studentoj"
            q = (f"En klaso estas {render_qty(base, item)}. {pct}% el ili "
                 f"portas okulvitrojn. Kiom {noun_acc_pl} portas okulvitrojn?")

        # Compute the answer using appropriate ops
        if style == "multiplier" and op in ("discount", "markup", "tax"):
            # 1 ± pct/100 = mult; base * mult = res
            factor = (100 - pct if op == "discount" else 100 + pct)
            assert (base * factor) % 100 == 0
            res = base * factor // 100
            mult = factor / 100
            mult_str = fmt_num(mult)
            sign = "-" if op == "discount" else "+"
            ctx.chain.append(f"1 {sign} {pct}/100 = {mult_str}")
            ctx.chain.append(f"{fmt_num(base)} * {mult_str} = {fmt_num(res)}")
            ctx.prose.append(f"La multobligilo estas 1 {sign} {pct}/100 = {mult_str}.")
            ctx.prose.append(f"Rezulto = {fmt_num(base)} * {mult_str} = {fmt_num(res)}.")
            ctx.bind("res", res)
            return ctx.render(q, "res")

        # direct / decimal path — always compute the % first
        Pct("pct", "base", "amount", style=style).apply(ctx)
        if op == "discount":
            Sub("base", "amount", "res").apply(ctx)
            final = "res"
        elif op in ("markup", "tax"):
            Add("base", "amount", "res").apply(ctx)
            final = "res"
        else:  # of-amount, saving
            final = "amount"

        # n_steps=3: stack a second percentage on top (discount then tax)
        if n_steps >= 3 and op == "discount" and pct2 is not None and final == "res":
            ctx.bind("pct2", pct2)
            q += f" Nun aldonu {pct2}% imposton sur la nova prezo. Kiu estas la fina prezo?"
            Pct("pct2", "res", "tax_amt", style="direct").apply(ctx)
            Add("res", "tax_amt", "final_price").apply(ctx)
            final = "final_price"

        return ctx.render(q, final)

    raise RuntimeError("percent_recipe: couldn't sample divisible params")


# ─── CLI ────────────────────────────────────────────────────────────────────

# ─── Recipe 3: average ──────────────────────────────────────────────────────

def average_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """Average of test scores (n_steps=2) or fraction of average (n_steps=3).

    n_steps=2: given N scores → find average
    n_steps=3: given N scores → find average → find X% of average
    """
    for _try in range(100):
        subject = rng.choice(SUBJECT_NOUNS)
        n_scores = rng.choice([3, 4, 5])
        # sample scores whose sum is divisible by n
        scores = [rng.randint(60, 100) for _ in range(n_scores)]
        if sum(scores) % n_scores != 0:
            continue

        ctx = Ctx.new(rng)
        p = ctx.protagonist
        # bind each score
        for i, s in enumerate(scores):
            ctx.bind(f"s{i}", s, noun=None)
        # build question
        scores_str = ", ".join(str(s) for s in scores[:-1]) + f" kaj {scores[-1]}"
        frame = maybe_frame(rng)
        q = frame + rng.choice([
            f"{p} ricevis {n_scores} poentarojn en {subject[2]}: {scores_str}. "
            f"Kiu estas la meza poentaro?",
            f"la poentaroj de {p} en {subject[2]} estas: {scores_str}. "
            f"Trovu la mezan.",
            f"post {n_scores} testoj en {subject[2]}, {p} ricevis: {scores_str}. "
            f"Kalkulu la mezan poentaron.",
            f"{p} skribis {n_scores} testojn en {subject[2]} kaj ricevis {scores_str}. "
            f"Kiu estas la meza rezulto?",
        ])
        if not frame:
            q = q[0].upper() + q[1:]

        Avg([f"s{i}" for i in range(n_scores)], "avg").apply(ctx)
        final = "avg"

        if n_steps >= 3:
            # find some percentage of the average — need divisibility
            avail_pcts = [p for p in [10, 20, 25, 50, 75] if int(ctx.n("avg")) * p % 100 == 0]
            if not avail_pcts:
                continue
            pct = rng.choice(avail_pcts)
            ctx.bind("pct", pct)
            q += f" Poste, kiom estas {pct}% el la meza poentaro?"
            Pct("pct", "avg", "result", style=rng.choice(["direct", "decimal"])).apply(ctx)
            final = "result"

        return ctx.render(q, final)

    raise RuntimeError("average_recipe: couldn't sample divisible params")


# ─── Recipe 4: fraction_cascade ─────────────────────────────────────────────

def fraction_cascade_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """Fraction-of-fraction. n_steps=2: single fraction. n_steps=3: fraction of fraction."""
    fractions = [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 5), (2, 5), (3, 5), (4, 5)]

    for _try in range(100):
        f1 = rng.choice(fractions)
        f2 = rng.choice(fractions)
        base_noun = rng.choice(COUNT_ITEMS)

        # base must be divisible by f1's denom, and (base*f1[0]/f1[1]) by f2's denom
        base = f1[1] * rng.randint(2, 20)
        step1_result = base * f1[0] // f1[1]
        if n_steps >= 3 and (step1_result * f2[0]) % f2[1] != 0:
            continue

        ctx = Ctx.new(rng)
        ctx.bind("base", base, noun=base_noun)
        p = ctx.protagonist

        q = f"En grupo estas {render_qty(base, base_noun)}. {f1[0]}/{f1[1]} el ili estas knabinoj."
        Frac("base", f1[0], f1[1], "girls").apply(ctx)
        final = "girls"

        if n_steps >= 3:
            q += f" El la knabinoj, {f2[0]}/{f2[1]} portas ruĝan ĉemizon."
            Frac("girls", f2[0], f2[1], "red").apply(ctx)
            final = "red"
            q += f" Kiom {base_noun[3]} portas ruĝan ĉemizon?"
        else:
            q += f" Kiom estas knabinoj?"

        return ctx.render(q, final)

    raise RuntimeError("fraction_cascade_recipe: couldn't sample")


# ─── Recipe 5: ratio_diff ───────────────────────────────────────────────────

def ratio_diff_recipe(rng: random.Random, n_steps: int = 3) -> dict:
    """N:M ratio splits `total` between two people; compute each part; report difference.
    Uses only Div + Mul + Sub — no algebra.
    """
    names = rng.sample(NAMES, 2)
    obj = rng.choice(OBJECT_NOUNS)

    for _try in range(100):
        # ratio (a, b), pick unit so total is realistic
        a, b = rng.choice([(2, 3), (3, 5), (1, 4), (2, 5), (3, 4), (1, 3), (4, 5), (3, 7)])
        unit = rng.randint(3, 30)
        total = (a + b) * unit
        if total > 300:
            continue

        ctx = Ctx.new(rng)
        ctx.bind("total", total, noun=obj)
        ctx.bind("parts", a + b, noun=obj)
        ctx.bind("a", a * unit, noun=obj)  # used as `larger` value later, but computed by chain
        ctx.bind("b", b * unit, noun=obj)

        q = rng.choice([
            f"{names[0]} kaj {names[1]} dividas {qty_acc(total, obj)} "
            f"laŭ la rilatumo {a}:{b}. Kiu estas la diferenco inter iliaj partoj?",
            f"En rilatumo {a}:{b}, {names[0]} kaj {names[1]} dividas "
            f"{qty_acc(total, obj)}. Kiom pli havas unu ol la alia?",
            f"{names[0]} ricevas {a} partojn, {names[1]} ricevas {b} partojn, "
            f"el entute {render_qty(total, obj)}. Trovu la diferencon.",
        ])

        # Steps: total / (a+b) = unit; a*unit; b*unit; diff
        # But we need the chain to *derive* unit; use Div op.
        Div("total", "parts", "unit").apply(ctx)      # step 1: total/(a+b) = unit
        ctx.bind("ra", a)
        ctx.bind("rb", b)
        Mul("ra", "unit", "part_a").apply(ctx)         # step 2: a * unit
        Mul("rb", "unit", "part_b").apply(ctx)         # step 3: b * unit
        Sub("part_b", "part_a", "diff").apply(ctx)     # step 4
        final = "diff"

        # n_steps=5: split the difference into K equal portions
        if n_steps >= 5:
            diff_val = int(ctx.n("diff"))
            portions = [k for k in (2, 3, 4, 5) if diff_val % k == 0]
            if not portions:
                continue
            k = rng.choice(portions)
            ctx.bind("k", k)
            q += f" Se ni disdividas la diferencon egalpartige inter {k} personoj, kiom ricevas ĉiu?"
            Div("diff", "k", "per_person").apply(ctx)
            final = "per_person"

        return ctx.render(q, final)

    raise RuntimeError("ratio_diff_recipe: couldn't sample")


# ─── Recipe 6: consec_avg ───────────────────────────────────────────────────

def consec_avg_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """N consecutive integers summing to S. Find the middle (= avg) or smallest/largest.

    Only works for odd N (so the middle is an integer) and step=1.
    """
    count = rng.choice([3, 5])   # odd
    # When extending, avoid "middle" so we have a value to scale
    ask_choices = ["smallest", "largest"] if n_steps >= 3 else ["smallest", "largest", "middle"]
    ask = rng.choice(ask_choices)
    start = rng.randint(2, 40)
    values = [start + i for i in range(count)]
    total = sum(values)
    ctx = Ctx.new(rng)
    for i, v in enumerate(values):
        ctx.bind(f"v{i}", v)
    ctx.bind("total", total)
    ctx.bind("count", count)

    what = {"smallest": "plej malgranda", "largest": "plej granda", "middle": "meza"}[ask]
    q = rng.choice([
        f"La sumo de {count} sinsekvaj entjeroj estas {total}. Kiu estas la {what}?",
        f"{count} sinsekvaj entjeroj sumigas al {total}. Trovu la {what}n.",
        f"Se {count} sinsekvaj entjeroj havas sumon de {total}, kiu estas la {what}?",
        f"Estas {count} sinsekvaj entjeroj kies sumo egalas {total}. Kalkulu la {what}n.",
    ])

    # Step 1: divide sum by count → get the average = middle value
    ctx.chain.append(f"{total} / {count} = {values[count // 2]}")
    ctx.prose.append(f"Meznombro = sumo / kalkulo: {total} / {count} = {values[count // 2]}.")
    ctx.bind("avg", values[count // 2])

    if ask == "middle":
        final_var = "avg"
    elif ask == "smallest":
        offset = count // 2   # avg - offset = smallest
        ctx.chain.append(f"{values[count // 2]} - {offset} = {values[0]}")
        ctx.prose.append(f"Plej malgranda = meza - {offset}: {values[count // 2]} - {offset} = {values[0]}.")
        ctx.bind("smallest", values[0])
        final_var = "smallest"
    else:  # largest
        offset = count // 2
        ctx.chain.append(f"{values[count // 2]} + {offset} = {values[-1]}")
        ctx.prose.append(f"Plej granda = meza + {offset}: {values[count // 2]} + {offset} = {values[-1]}.")
        ctx.bind("largest", values[-1])
        final_var = "largest"

    # n_steps=3: multiply the result by a factor (age analogue: "in K years")
    if n_steps >= 3 and final_var != "avg":
        k = rng.randint(2, 5)
        ctx.bind("k", k)
        q += f" Kio estas {k} foje tiu valoro?"
        Mul(final_var, "k", "scaled").apply(ctx)
        final_var = "scaled"

    return ctx.render(q, final_var)


# ─── Recipe 7: inverse_rate ─────────────────────────────────────────────────

# Worker Noun tuples: (nom_sg, nom_pl, acc_sg, acc_pl)
_WORKERS: list[Noun] = [
    ("laboristo",  "laboristoj",  "laboriston",  "laboristojn"),
    ("pumpilo",    "pumpiloj",    "pumpilon",    "pumpilojn"),
    ("maŝino",     "maŝinoj",     "maŝinon",     "maŝinojn"),
    ("rikoltisto", "rikoltistoj", "rikoltiston", "rikoltistojn"),
    ("kuiristo",   "kuiristoj",   "kuiriston",   "kuiristojn"),
]
# Time-unit Noun tuples
_TIME_UNITS: list[Noun] = [
    ("horo",   "horoj",   "horon",   "horojn"),
    ("minuto", "minutoj", "minuton", "minutojn"),
    ("tago",   "tagoj",   "tagon",   "tagojn"),
]

# (worker_noun, verb_infinitive, task_acc, time_noun)
INV_SCENARIOS: list[tuple[Noun, str, str, Noun]] = [
    (_WORKERS[0], "farbi",   "muron",     _TIME_UNITS[0]),
    (_WORKERS[1], "plenigi", "naĝejon",   _TIME_UNITS[1]),
    (_WORKERS[2], "presi",   "libron",    _TIME_UNITS[0]),
    (_WORKERS[3], "rikolti", "kampon",    _TIME_UNITS[2]),
    (_WORKERS[4], "prepari", "manĝon",    _TIME_UNITS[0]),
]


def inverse_rate_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """W1 workers × T1 time = const. Find T2 for W2 workers (or W2 for T2)."""
    for _try in range(100):
        w1 = rng.randint(2, 12)
        t1 = rng.randint(2, 24)
        const = w1 * t1
        divs = [d for d in range(1, const + 1) if const % d == 0 and d != w1 and 1 <= d <= 40]
        if not divs:
            continue
        w2 = rng.choice(divs)
        t2 = const // w2

        worker, verb, task, tunit = rng.choice(INV_SCENARIOS)
        ask = rng.choice(["find-time", "find-workers"])
        ctx = Ctx.new(rng)
        ctx.bind("w1", w1)
        ctx.bind("t1", t1)

        # Time renderings: nom (for "kiom da horoj") and acc (for "bezonas 19 horojn")
        t1_acc = qty_acc(t1, tunit)                # "19 horojn"
        t2_acc = qty_acc(t2, tunit)                # "3 horojn"
        w1_nom = render_qty(w1, worker)            # "4 maŝinoj" or "1 maŝino"
        w2_nom = render_qty(w2, worker)            # "1 maŝino" etc.
        tunit_pl = tunit[1]                        # "horoj"
        workers_pl = worker[1]                     # "maŝinoj"

        if ask == "find-time":
            ctx.bind("w2", w2)
            q = (f"{w1_nom} bezonas {t1_acc} por {verb} {task}. "
                 f"Kiom da {tunit_pl} bezonatas por {w2_nom}?")
            Mul("w1", "t1", "const").apply(ctx)
            Div("const", "w2", "t2").apply(ctx)
            final = "t2"
        else:
            ctx.bind("t2", t2)
            q = (f"{w1_nom} bezonas {t1_acc} por {verb} {task}. "
                 f"Kiom da {workers_pl} bezonatas por fini en {t2_acc}?")
            Mul("w1", "t1", "const").apply(ctx)
            Div("const", "t2", "w2").apply(ctx)
            final = "w2"

        # n_steps=3: compare against a third team size
        if n_steps >= 3 and ask == "find-time":
            w3_candidates = [d for d in range(1, const + 1)
                             if const % d == 0 and d != w1 and d != w2 and 1 <= d <= 40]
            if not w3_candidates:
                continue
            w3 = rng.choice(w3_candidates)
            ctx.bind("w3", w3)
            q += f" Kaj kiom da {tunit_pl} bezonatas por {render_qty(w3, worker)}?"
            Div("const", "w3", "t3").apply(ctx)
            final = "t3"

        return ctx.render(q, final)

    raise RuntimeError("inverse_rate_recipe: couldn't sample")


# ─── Recipe 8: ratio_fraction ───────────────────────────────────────────────

def ratio_fraction_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """Ratio via fraction-of-total: r_i/(r_1+r_2) * total = part_i.
    Direct + larger + smaller asks. Uses Add + Frac.
    """
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

        which = {"larger": "pli granda parto", "smaller": "pli malgranda parto",
                 "direct": f"parto de {target_name}"}[ask]
        q = rng.choice([
            f"{names[0]} kaj {names[1]} dividas {qty_acc(total, obj)} laŭ la "
            f"rilatumo {a}:{b}. Kiu estas la {which}?",
            f"En rilatumo {a}:{b}, {names[0]} kaj {names[1]} dividas "
            f"{qty_acc(total, obj)}. Trovu la {which}n.",
            f"Ilia dividita nombro estas {render_qty(total, obj)}, "
            f"en rilatumo {a}:{b}. Kiu estas la {which}?",
        ])

        # step 1: sum the ratio parts
        Add("ra", "rb", "r_sum").apply(ctx)
        # step 2: apply Frac(target_r, r_sum) to total
        Frac("total", target_r, a + b, "part").apply(ctx)
        final = "part"

        # n_steps=3: apply a percentage to the part
        if n_steps >= 3:
            part_val = int(ctx.n("part"))
            pcts = [p for p in [10, 20, 25, 40, 50, 75] if (part_val * p) % 100 == 0]
            if not pcts:
                continue
            pct = rng.choice(pcts)
            ctx.bind("pct", pct)
            q += f" Poste, {pct}% de tiu parto estas donacita. Kiom estas donacita?"
            Pct("pct", "part", "gift", style="direct").apply(ctx)
            final = "gift"

        return ctx.render(q, final)

    raise RuntimeError("ratio_fraction_recipe: couldn't sample")


# ─── Recipe 9: distance_direct ──────────────────────────────────────────────

_VEHICLES: list[Noun] = [
    ("aŭto",     "aŭtoj",     "aŭton",     "aŭtojn"),
    ("biciklo",  "bicikloj",  "biciklon",  "biciklojn"),
    ("kamiono",  "kamionoj",  "kamionon",  "kamionojn"),
    ("motorciklo","motorcikloj","motorciklon","motorciklojn"),
]

def distance_direct_recipe(rng: random.Random, n_steps: int = 1) -> dict:
    """D = R * T with a single ask: find D, R, or T.

    n_steps=1 (default): pure D=R*T
    n_steps=2: also add a return-trip using the derived quantity
    """
    for _try in range(100):
        r = rng.choice([40, 50, 60, 70, 75, 80, 90, 100, 120])
        t = rng.randint(2, 8)
        d = r * t
        # n_steps>=2 only makes sense after "find distance" — pin ask to 'd'
        ask = "d" if n_steps >= 2 else rng.choice(["d", "r", "t"])

        vehicle = rng.choice(_VEHICLES)
        name = rng.choice(NAMES)
        ctx = Ctx.new(rng)
        ctx.protagonist = name

        frame = maybe_frame(rng)
        if ask == "d":
            ctx.bind("r", r); ctx.bind("t", t)
            q = frame + rng.choice([
                f"{name} veturas per sia {vehicle[0]} je {r} km/h dum {t} horoj. "
                f"Kiom da kilometroj {name} kovras?",
                f"per sia {vehicle[0]}, {name} moviĝas je {r} km/h "
                f"dum {t} horoj. Kiu estas la kovrita distanco?",
                f"{name} rajdas sian {vehicle[2]} je {r} km/h dum {t} horoj. "
                f"Trovu la distancon.",
            ])
            Mul("r", "t", "d").apply(ctx)
            final = "d"
        elif ask == "r":
            ctx.bind("d", d); ctx.bind("t", t)
            q = frame + rng.choice([
                f"{name} veturas {d} km per sia {vehicle[0]} en {t} horoj. "
                f"Kiu estas la rapideco?",
                f"post {t} horoj de veturado, {name} kovris {d} km per sia {vehicle[0]}. "
                f"Kalkulu la rapidecon.",
                f"la {vehicle[0]} de {name} kovras {d} km en {t} horoj. "
                f"Kiu estas la rapideco?",
            ])
            Div("d", "t", "r").apply(ctx)
            final = "r"
        else:  # ask == "t"
            if d % r != 0:
                continue
            ctx.bind("d", d); ctx.bind("r", r)
            q = frame + rng.choice([
                f"{name} veturas per sia {vehicle[0]} je {r} km/h. Kiom da horoj "
                f"bezonas por kovri {d} km?",
                f"per sia {vehicle[0]}, {name} moviĝas je {r} km/h. "
                f"Kiom da tempo bezonas por {d} km?",
                f"{name} bezonas veturi {d} km je {r} km/h per sia {vehicle[0]}. "
                f"Kiom da horoj tio daŭros?",
            ])
            Div("d", "r", "t").apply(ctx)
            final = "t"
        if not frame:
            q = q[0].upper() + q[1:]

        if n_steps >= 2 and ask == "d":
            # add return trip at different speed for n_steps=2
            r2_candidates = [r2 for r2 in [40, 50, 60, 75, 80, 100, 120] if d % r2 == 0 and r2 != r]
            if not r2_candidates:
                continue  # resample outer params
            r2 = rng.choice(r2_candidates)
            ctx.bind("r2", r2)
            q += f" Se {name} revenas je {r2} km/h, kiom da horoj daŭros la reveno?"
            Div("d", "r2", "t2").apply(ctx)
            final = "t2"

            # n_steps=3: total round-trip time (out + back)
            if n_steps >= 3:
                ctx.bind("t_orig", t)
                q += f" Kaj kiom da horoj daŭras la tuta rondiro?"
                Add("t_orig", "t2", "t_total").apply(ctx)
                final = "t_total"

        return ctx.render(q, final)

    raise RuntimeError("distance_direct_recipe: couldn't sample")


# ─── Recipe 10: coin_assume ─────────────────────────────────────────────────

_PENCO: Noun = ("penco", "pencoj", "pencon", "pencojn")
_CENDO: Noun = ("cendo", "cendoj", "cendon", "cendojn")
_MONERO: Noun = ("monero", "moneroj", "moneron", "monerojn")
_BILETO: Noun = ("bileto", "biletoj", "bileton", "biletojn")

# denominations: (small_val, big_val, currency_noun, item_noun)
_COIN_DENOMS: list[tuple[int, int, Noun, Noun]] = [
    (1, 5,   _PENCO, _MONERO),
    (5, 25,  _PENCO, _MONERO),
    (10, 50, _CENDO, _MONERO),
    (5, 20,  EUR,    _BILETO),
]


def coin_assume_recipe(rng: random.Random, n_steps: int = 3) -> dict:
    """Coin problem via "assume all small" reasoning — no algebra.

    Given total_count items each worth small OR big; total value V.
    Reasoning: if all were small, total would be S*N. Actual is V.
    Difference V - S*N must come from bigs: each big adds (big-small) more.
    So count_big = (V - S*N) / (big - small).
    Uses Mul + Sub + Div.
    """
    for _try in range(100):
        small_val, big_val, currency, item = rng.choice(_COIN_DENOMS)
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

        # Grammar: `havas Y` → acc; `valoras Y` → acc; `kun totala valoro Y` → nom
        cur_sg = currency[0]  # for adjective-forming "penco-a" etc.
        target_val = big_val if ask == "find-big" else small_val
        q = (f"{name} havas {qty_acc(total_count, item)} kun totala valoro "
             f"{render_qty(total_value, currency)}. Iuj valoras "
             f"{qty_acc(small_val, currency)}, aliaj {qty_acc(big_val, currency)}. "
             f"Kiom da {target_val}-{cur_sg}-aj {item[1]}?")

        # step 1: assume all small: small_val * total_count
        Mul("small_val", "total_count", "assumed_total").apply(ctx)
        # step 2: extra value = total_value - assumed_total
        Sub("total_value", "assumed_total", "extra").apply(ctx)
        # step 3: each big adds (big - small)
        ctx.bind("step_up", big_val - small_val)
        ctx.chain.append(f"{big_val} - {small_val} = {big_val - small_val}")
        ctx.prose.append(
            f"Ĉiu {big_val}-{cur_sg}-a aldonas {big_val} - {small_val} = "
            f"{big_val - small_val} pli ol {small_val}-{cur_sg}-a."
        )
        # step 4: count_big = extra / step_up
        Div("extra", "step_up", "count_big").apply(ctx)

        if ask == "find-big":
            final = "count_big"
        else:
            # find-small: subtract from total
            Sub("total_count", "count_big", "count_small").apply(ctx)
            final = "count_small"

        # n_steps=5: also compute total value of the target denomination
        if n_steps >= 5:
            target_denom_var = "big_val" if ask == "find-big" else "small_val"
            q += f" Kiu estas la totala valoro de tiuj {target_val}-{cur_sg}-aj?"
            Mul(target_denom_var, final, "target_value").apply(ctx)
            final = "target_value"

        return ctx.render(q, final)

    raise RuntimeError("coin_assume_recipe: couldn't sample")


# ─── Recipe 11: distance_catchup ────────────────────────────────────────────

def distance_catchup_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """A leaves at ra km/h; h hours later B leaves at rb (rb > ra) and catches up.
    t = ra*h / (rb-ra).  Uses Mul + Sub + Div.
    """
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

        q = (f"{names[0]} ekveturas per sia {vehicle[0]} je {ra} km/h. "
             f"Post {h} horoj, {names[1]} ekiras de la sama loko "
             f"en la sama direkto je {rb} km/h. "
             f"Post kiom da horoj {names[1]} atingos {names[0]}?")

        # step 1: head start distance = ra * h
        Mul("ra", "h", "head_start").apply(ctx)
        # step 2: speed diff = rb - ra
        Sub("rb", "ra", "gap").apply(ctx)
        # step 3: time to catch up = head_start / gap
        Div("head_start", "gap", "t").apply(ctx)
        final = "t"

        # n_steps=4: also compute how far A had gone by catch-up
        if n_steps >= 4:
            q += f" Kiom da km {names[0]} veturis kiam {names[1]} atingis?"
            # A's total time = h + t; A's distance = ra * (h + t)
            Add("h", "t", "a_total_time").apply(ctx)
            Mul("ra", "a_total_time", "a_dist").apply(ctx)
            final = "a_dist"

        return ctx.render(q, final)

    raise RuntimeError("distance_catchup_recipe: couldn't sample")


# ─── Recipe 12: distance_meeting ────────────────────────────────────────────

def distance_meeting_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """Two vehicles start at opposite ends of distance D, moving toward each other.
    t = D / (r1 + r2).  Uses Add + Div.
    """
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

        q = (f"{names[0]} kaj {names[1]} ekiras samtempe de du urboj "
             f"distancaj je {d} km, veturante unu al la alia. "
             f"{names[0]} veturas je {r1} km/h, {names[1]} je {r2} km/h. "
             f"Post kiom da horoj ili renkontiĝos?")

        Add("r1", "r2", "r_sum").apply(ctx)
        Div("d", "r_sum", "t").apply(ctx)
        final = "t"

        # n_steps=3: also ask how far each traveled — Mul each speed by t
        if n_steps >= 3:
            q += f" Kiom da km {names[0]} veturis kiam ili renkontiĝas?"
            Mul("r1", "t", "d1").apply(ctx)
            final = "d1"

        return ctx.render(q, final)

    raise RuntimeError("distance_meeting_recipe: couldn't sample")


# ─── Recipe 13: distance_avg (harmonic mean) ────────────────────────────────

def distance_avg_recipe(rng: random.Random, n_steps: int = 3) -> dict:
    """Round-trip: out at rout, back at rback, same distance.
    Avg speed = 2*rout*rback / (rout + rback).  Uses Mul + Add + Div.
    """
    for _try in range(100):
        rout = rng.choice([40, 50, 60, 75, 80, 90, 120])
        rback = rng.choice([30, 40, 50, 60, 75, 80])
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

        q = (f"{name} veturas per sia {vehicle[0]} de urbo A al urbo B je "
             f"{rout} km/h, kaj revenas je {rback} km/h. "
             f"Kiu estas la meza rapideco por la tuta rondiro?")

        # step 1: 2 * rout = 2rout
        Mul("two", "rout", "two_rout").apply(ctx)
        # step 2: two_rout * rback = numer
        Mul("two_rout", "rback", "numer").apply(ctx)
        # step 3: rout + rback = denom
        Add("rout", "rback", "denom").apply(ctx)
        # step 4: numer / denom = avg
        Div("numer", "denom", "avg").apply(ctx)
        final = "avg"

        # n_steps=5: given a specified round-trip distance, compute total time
        if n_steps >= 5:
            candidates = [d for d in (60, 90, 120, 180, 240, 300)
                          if d % rout == 0 and d % rback == 0]
            if not candidates:
                continue
            d = rng.choice(candidates)
            ctx.bind("d", d)
            q += f" Se la distanco A-al-B estas {d} km, kiom da horoj daŭras la tuta rondiro?"
            Div("d", "rout", "t_out").apply(ctx)
            Div("d", "rback", "t_back").apply(ctx)
            Add("t_out", "t_back", "t_total").apply(ctx)
            final = "t_total"

        return ctx.render(q, final)

    raise RuntimeError("distance_avg_recipe: couldn't sample")


# ─── Recipe 14: age_simple (algebraic) ──────────────────────────────────────

_AGE_RELATIONS = [
    ("patrino", "filo"), ("patro", "filino"),
    ("onklo", "nevo"), ("frato", "fratino"),
]

def age_simple_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """Age problem: old is r times younger; sum is known.
    Solve x + r*x = sum → (r+1)*x = sum → x = sum/(r+1).
    Uses LinearSolve (algebraic) then optional Mul for `ask=old`.
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

        mul_word = {2: "dufoje", 3: "trifoje", 4: "kvarfoje", 5: "kvinfoje"}[ratio]
        target = "juna" if ask == "young" else "olda"
        q = rng.choice([
            f"{names[0]} estas {mul_word} pli aĝa ol {names[1]}. "
            f"Kune ili havas {sum_now} jarojn. Kiom aĝa estas la {target} persono?",
            f"La aĝo de {names[0]} estas {ratio} fojoj tiu de {names[1]}. "
            f"La sumo de iliaj aĝoj estas {sum_now}. Trovu la aĝon de la {target}.",
            f"{names[1]} estas x jarojn aĝa. {names[0]} estas {ratio}x. "
            f"Kune ili estas {sum_now} jarojn aĝaj. Kiom aĝa estas la {target}?",
        ])

        # Solve x + r*x = sum → (r+1)*x = sum, with combining prose
        LinearSolve("sum_coef", "zero", "sum_now", "young",
                    var_name="x", lhs_shape=f"x + {ratio}*x").apply(ctx)
        if ask == "young":
            return ctx.render(q, "young")
        Mul("ratio", "young", "old").apply(ctx)
        return ctx.render(q, "old")

    raise RuntimeError("age_simple_recipe: couldn't sample")


# ─── Recipe 15: consec_first_as_x (algebraic) ───────────────────────────────

def consec_first_as_x_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """N consecutive integers summing to S. Setup: x + (x+1) + (x+2) + ... = S.
    Combine: N*x + Σoffsets = S. Solve via LinearSolve. Uses Mul for follow-ups.
    """
    count = rng.choice([3, 4, 5])
    ask_choices = ["smallest", "largest"] if count % 2 == 0 else ["smallest", "largest", "middle"]
    ask = rng.choice(ask_choices)
    start = rng.randint(2, 40)
    values = [start + i for i in range(count)]
    total = sum(values)
    const = sum(range(count))  # 0+1+2+...+(N-1)

    ctx = Ctx.new(rng)
    ctx.bind("n", count)
    ctx.bind("const", const)
    ctx.bind("total", total)

    lhs_terms = " + ".join(["x"] + [f"(x + {i})" for i in range(1, count)])
    what = {"smallest": "plej malgranda", "largest": "plej granda", "middle": "meza"}[ask]
    q = rng.choice([
        f"La sumo de {count} sinsekvaj entjeroj estas {total}. Kiu estas la {what}?",
        f"{count} sinsekvaj entjeroj sumigas al {total}. Trovu la {what}n.",
        f"Se {count} sinsekvaj entjeroj havas sumon de {total}, kiu estas la {what}?",
        f"Estas {count} sinsekvaj entjeroj kies sumo egalas {total}. Kalkulu la {what}n.",
    ])

    LinearSolve("n", "const", "total", "x",
                var_name="x", lhs_shape=lhs_terms).apply(ctx)

    if ask == "smallest":
        return ctx.render(q, "x")
    # x + (count-1) for largest, x + (count//2) for middle
    offset = (count - 1) if ask == "largest" else (count // 2)
    ctx.bind("offset", offset)
    Add("x", "offset", "result").apply(ctx)
    return ctx.render(q, "result")


# ─── Recipe 16: ratio_algebra ───────────────────────────────────────────────

def ratio_algebra_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """Ratio via algebra: r1*x + r2*x = total → (r1+r2)*x = total → x = total/(r1+r2).
    Then r_i * x = part_i. Uses LinearSolve + Mul.
    """
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
        which = {"direct-a": f"parto de {names[0]}",
                 "direct-b": f"parto de {names[1]}",
                 "larger": "pli granda parto"}[ask]
        q = rng.choice([
            f"{names[0]} kaj {names[1]} dividas {qty_acc(total, obj)} laŭ la "
            f"rilatumo {a}:{b}. Kiu estas la {which}?",
            f"En rilatumo {a}:{b}, {names[0]} kaj {names[1]} dividas "
            f"{qty_acc(total, obj)}. Trovu la {which}n.",
            f"La totala nombro estas {render_qty(total, obj)}, dividita en la "
            f"rilatumo {a}:{b} inter {names[0]} kaj {names[1]}. Kiu estas la {which}?",
        ])

        # x + 3x = total → 4x = total → x = total/4
        LinearSolve("r_sum", "zero", "total", "x",
                    var_name="x", lhs_shape=f"{a}*x + {b}*x").apply(ctx)

        # multiply by the target ratio to get the part
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recipe", choices=list(RECIPES.keys()), default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.count > 0:
        for _ in range(args.count):
            recipe_name = args.recipe or rng.choice(list(RECIPES.keys()))
            recipe = RECIPES[recipe_name]
            # 2/3 base + occasional 4/5 extensions — bias toward 2/3 so overall
            # distribution matches natural GSM8K density (mostly 2-3-step chains).
            n_steps = rng.choices([2, 3, 4, 5], weights=[3, 3, 2, 1])[0]
            try:
                p = recipe(rng, n_steps=n_steps)
                p["recipe"] = recipe_name
                p["n_steps"] = n_steps
                print(json.dumps(p, ensure_ascii=False))
            except RuntimeError:
                continue
        return

    # sample mode: print a couple of each
    print("=" * 70)
    print("ratio_parts_recipe")
    print("=" * 70)
    for n in [2, 3, 4]:
        print(f"\n--- n_steps={n} ---")
        for _ in range(2):
            p = ratio_parts_recipe(rng, n_steps=n)
            print(f"Q: {p['question']}")
            print(f"A: {p['answer']}")
            print(f"   chain: {p['chain_lines']}\n")

    print("=" * 70)
    print("percent_recipe")
    print("=" * 70)
    for op in ["discount", "markup", "tax", "of-amount", "saving"]:
        print(f"\n--- op={op} ---")
        for _ in range(2):
            try:
                p = percent_recipe(rng, op=op)
                print(f"Q: {p['question']}")
                print(f"A: {p['answer']}")
                print(f"   chain: {p['chain_lines']}\n")
            except RuntimeError as e:
                print(f"   ({e})\n")

    print("=" * 70)
    print("average_recipe")
    print("=" * 70)
    for n in [2, 3]:
        print(f"\n--- n_steps={n} ---")
        for _ in range(2):
            try:
                p = average_recipe(rng, n_steps=n)
                print(f"Q: {p['question']}")
                print(f"A: {p['answer']}")
                print(f"   chain: {p['chain_lines']}\n")
            except RuntimeError as e:
                print(f"   ({e})\n")

    print("=" * 70)
    print("fraction_cascade_recipe")
    print("=" * 70)
    for n in [2, 3]:
        print(f"\n--- n_steps={n} ---")
        for _ in range(2):
            try:
                p = fraction_cascade_recipe(rng, n_steps=n)
                print(f"Q: {p['question']}")
                print(f"A: {p['answer']}")
                print(f"   chain: {p['chain_lines']}\n")
            except RuntimeError as e:
                print(f"   ({e})\n")

    for label, fn in [("ratio_diff_recipe", ratio_diff_recipe),
                      ("ratio_fraction_recipe", ratio_fraction_recipe),
                      ("consec_avg_recipe", consec_avg_recipe),
                      ("inverse_rate_recipe", inverse_rate_recipe),
                      ("distance_direct_recipe", distance_direct_recipe),
                      ("distance_catchup_recipe", distance_catchup_recipe),
                      ("distance_meeting_recipe", distance_meeting_recipe),
                      ("distance_avg_recipe", distance_avg_recipe),
                      ("coin_assume_recipe", coin_assume_recipe)]:
        print("=" * 70)
        print(label)
        print("=" * 70)
        for _ in range(3):
            try:
                p = fn(rng)
                print(f"Q: {p['question']}")
                print(f"A: {p['answer']}")
                print(f"   chain: {p['chain_lines']}\n")
            except RuntimeError as e:
                print(f"   ({e})\n")


if __name__ == "__main__":
    main()
