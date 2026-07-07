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
# EN mirror — same index order as SCENARIO_FRAMES so parallel emission works.
SCENARIO_FRAMES_EN = [
    "In 2015,",  "During the school year,",  "One day,",  "After school,",
    "At the market,",  "During summer,",  "Before the party,",  "This morning,",
    "Last week,",  "At the party,",  "After the meal,",  "At the shop,",
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
# EN parallel nouns — same index as the EO list.  (singular, plural).
# When a recipe needs EN alongside EO, pick the SAME index into both lists.
CHILDLIKE_NOUNS_EN = [
    ("child",    "children"),
    ("boy",      "boys"),
    ("girl",     "girls"),
    ("student",  "students"),
    ("pupil",    "pupils"),
    ("customer", "customers"),
    ("guest",    "guests"),
    ("visitor",  "visitors"),
]
GROUPING_NOUNS_EN = [
    ("group",       "groups"),
    ("team",        "teams"),
    ("class",       "classes"),
    ("table",       "tables"),
    ("bus",         "buses"),
    ("room",        "rooms"),
    ("box",         "boxes"),
    ("wagon",       "wagons"),
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
    # English parallel prose. Populated by Ops whose prose-lib has an EN
    # mirror (see `_MUL_PROSE_EN` etc.). If ALL Ops on a chain have EN,
    # `len(prose_en) == len(prose)` and render() can emit question_en/answer_en.
    # Partial EN emission is intentional — a recipe that mixes EN-mirrored
    # Ops with EN-less ones simply won't get the EN pair, silently.
    prose_en: list[str] = field(default_factory=list)
    protagonist_en: str = ""
    # Ops appended in application order — walked in reverse by render_reverse.
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

    def render(self, question: str, final_var: str,
                question_en: str | None = None) -> dict:
        v = self.vars[final_var].value
        final_str = str(int(v)) if v == int(v) else str(v)
        # Prose paragraphs joined; end-marker last.
        answer = " ".join(self.prose) + f" #### {final_str}"
        result = {
            "question": question,
            "answer": answer.strip(),
            "chain_lines": self.chain,
            "final": final_str,
        }
        # Emit English parallel iff (a) caller supplied question_en, and (b)
        # every applied Op has an EN mirror → prose_en length matches prose.
        if question_en is not None and len(self.prose_en) == len(self.prose):
            answer_en = " ".join(self.prose_en) + f" #### {final_str}"
            result["question_en"] = question_en
            result["answer_en"] = answer_en.strip()
        return result

    def render_reverse(
        self,
        forward_prose: str,
        forward_final_var: str,
        ask_var: str,
        closer: str,
        recipe_name: str = "reverse",
        forward_prose_en: str | None = None,
        closer_en: str | None = None,
    ) -> dict:
        """Reverse-frame the forward problem.

        Given a forward narrative + chain that computed forward_final_var from
        the initial vars, produce a NEW question that STATES the value of
        forward_final_var and ASKS for ask_var (an early input), plus a
        reasoning chain that walks the applied_ops backwards.

        The forward narrative is rebuilt by the CALLER via `forward_prose` —
        which should NOT state the value of ask_var (that's what we're
        hiding) and SHOULD state the value of forward_final_var (that's
        what's given). Sample recipes handle this by using an "unknown
        quantity" template for the ask_var position.

        Only invertible ops are supported (no Avg / LinearSolve).
        """
        answer_val = self.vars[ask_var].value
        ans_str = str(int(answer_val)) if answer_val == int(answer_val) else str(answer_val)

        # Trace forward through applied_ops to find which side (lhs/rhs) of
        # each Op is on the ask_var → forward_final path. Starts at ask_var,
        # ends at forward_final_var. If an op has ask_var (or a downstream
        # descendant) on its lhs, we'll invert on the rhs (rhs is known);
        # if on its rhs, we'll invert on the lhs (lhs is known).
        path_var: str = ask_var
        path_sides: list[str] = []  # per op in application order, "lhs" or "rhs"
        for op in self.applied_ops:
            if op.kind == "frac":
                # Frac's lhs is the base; num/denom are literals not ctx vars.
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
        reverse_prose_en: list[str] = []
        # If any op-kind on the path lacks an EN reverse-step lib, we drop
        # EN emission entirely (like Ctx.render's contract).
        en_ok = True
        known_val: float = self.vars[forward_final_var].value

        for op, side in zip(reversed(self.applied_ops), reversed(path_sides)):
            if side is None:
                # This op isn't on the ask_var → final path; skip it.
                continue
            # `side` is which SIDE carries the ask_var chain. So the OTHER
            # side is the "known constant" we use to invert.
            known_side = "rhs" if side == "lhs" else "lhs"
            if op.kind == "frac":
                unknown = op.reverse_step(known_val, known_side, None)
                reverse_chain.append(
                    op.reverse_chain_line(known_val, known_side, None, unknown))
                # Same-index EO/EN pick — preserves parallel alignment.
                variants_eo = REVERSE_STEP_PROSE[op.kind]
                idx = self.rng.randrange(len(variants_eo))
                fmt_kwargs = dict(a=fmt_num(known_val), c=fmt_num(unknown),
                                   n=op.num, d=op.denom)
                reverse_prose.append(variants_eo[idx].format(**fmt_kwargs))
                variants_en = REVERSE_STEP_PROSE_EN.get(op.kind, [])
                if idx < len(variants_en):
                    reverse_prose_en.append(variants_en[idx].format(**fmt_kwargs))
                else:
                    en_ok = False
            else:
                other_side_var = op.rhs if known_side == "rhs" else op.lhs
                other_side_val = self.vars[other_side_var].value
                unknown = op.reverse_step(known_val, known_side, other_side_val)
                reverse_chain.append(
                    op.reverse_chain_line(known_val, known_side, other_side_val, unknown))
                variants_eo = REVERSE_STEP_PROSE[op.kind]
                idx = self.rng.randrange(len(variants_eo))
                fmt_kwargs = dict(a=fmt_num(known_val),
                                   b=fmt_num(other_side_val),
                                   c=fmt_num(unknown))
                reverse_prose.append(variants_eo[idx].format(**fmt_kwargs))
                variants_en = REVERSE_STEP_PROSE_EN.get(op.kind, [])
                if idx < len(variants_en):
                    reverse_prose_en.append(variants_en[idx].format(**fmt_kwargs))
                else:
                    en_ok = False
            known_val = unknown

        # Sanity check: reverse walk lands on the ask_var value
        assert abs(known_val - answer_val) < 1e-9, (
            f"reverse chain didn't recover: known={known_val} answer={answer_val}"
        )

        question = f"{forward_prose} {closer}"
        answer = " ".join(reverse_prose) + f" #### {ans_str}"
        result = {
            "question": question,
            "answer": answer.strip(),
            "chain_lines": reverse_chain,
            "final": ans_str,
        }
        # Emit parallel EN iff caller supplied both forward_prose_en +
        # closer_en, AND every reverse-step had an EN mirror.
        if forward_prose_en is not None and closer_en is not None and en_ok:
            answer_en = " ".join(reverse_prose_en) + f" #### {ans_str}"
            result["question_en"] = f"{forward_prose_en} {closer_en}"
            result["answer_en"] = answer_en.strip()
        return result


# Reverse-step prose per Op kind. Placeholders:
#   {a} = known output value entering this reverse step
#   {b} = other-side value (side input)
#   {c} = computed unknown
#   For frac: also {n}=numerator, {d}=denominator
REVERSE_STEP_PROSE: dict[str, list[str]] = {
    "mul": [
        "Ni retropasi la multiplikon: {a} / {b} = {c}.",
        "Ni dividas por retropasi: {a} / {b} = {c}.",
        "La origina nombro estas {a} / {b} = {c}.",
        "Do la origina kvanto estis {a} / {b} = {c}.",
        "Dividante por retropasi: {a} / {b} = {c}.",
        "Malfaru la multiplikon: {a} / {b} = {c}.",
        "Inversigi la multiplikon per divido: {a} / {b} = {c}.",
        "La antaŭa valoro estis {a} / {b} = {c}.",
        "Retroiri per divido: {a} / {b} = {c}.",
        "Do la enigo de la multipliko estis {c}, ĉar {a} / {b} = {c}.",
    ],
    "add": [
        "Ni subtrahas la aldonon: {a} - {b} = {c}.",
        "La origina valoro estis {a} - {b} = {c}.",
        "Do antaŭ la aldono estis {a} - {b} = {c}.",
        "Retropasi la aldonon: {a} - {b} = {c}.",
        "Malfaru la aldonon: {a} - {b} = {c}.",
        "Subtrahante la aldonon: {a} - {b} = {c}.",
        "Inversigi la aldonon: {a} - {b} = {c}.",
        "La antaŭa valoro estis {a} - {b} = {c}.",
        "Retroiri per subtraho: {a} - {b} = {c}.",
    ],
    "sub": [
        "Ni aldonas por retropasi: {a} + {b} = {c}.",
        "Do antaŭ la subtraho estis {a} + {b} = {c}.",
        "La origina valoro estis {a} + {b} = {c}.",
        "Retropasi la subtrahon: {a} + {b} = {c}.",
        "Malfaru la subtrahon: {a} + {b} = {c}.",
        "Aldonu por inversigi: {a} + {b} = {c}.",
        "Redoni la forigitan: {a} + {b} = {c}.",
        "La antaŭa valoro estis {a} + {b} = {c}.",
        "Retroiri per aldono: {a} + {b} = {c}.",
    ],
    "div": [
        "Ni multiplikas por retropasi: {a} * {b} = {c}.",
        "La origina nombro estis {a} * {b} = {c}.",
        "Do antaŭ la divido estis {a} * {b} = {c}.",
        "Retropasi la dividon: {a} * {b} = {c}.",
        "Malfaru la dividon: {a} * {b} = {c}.",
        "Multipliku por inversigi: {a} * {b} = {c}.",
        "La antaŭa valoro estis {a} * {b} = {c}.",
        "Retroiri per multipliko: {a} * {b} = {c}.",
        "Do la enigo de la divido estis {a} * {b} = {c}.",
    ],
    "frac": [
        "Ni retropasi la frakcion: {a} * {d} / {n} = {c}.",
        "La origina bazo estis {a} * {d} / {n} = {c}.",
        "Do la origina bazo estis {a} * {d} / {n} = {c}.",
        "Malfaru la frakcion: {a} * {d} / {n} = {c}.",
        "Inversigi {n}/{d}: multipliki per {d}/{n} donas {c}.",
        "Retroiri tra la frakcio: {a} * {d} / {n} = {c}.",
        "La antaŭa bazo estis {a} * {d} / {n} = {c}.",
        "Inversigante {n}/{d}: {a} * {d} / {n} = {c}.",
    ],
    "pct": [
        "Ni retropasi la procenton: {a} * 100 / {b} = {c}.",
        "La origina bazo estis {a} * 100 / {b} = {c}.",
        "Do la origina bazo estis {a} * 100 / {b} = {c}.",
        "Malfaru la procenton: {a} * 100 / {b} = {c}.",
        "Inversigi {b}%: {a} * 100 / {b} = {c}.",
        "Retroiri tra la procento: {a} * 100 / {b} = {c}.",
        "La antaŭa bazo estis {a} * 100 / {b} = {c}.",
        "Divizio per {b}% donas {a} * 100 / {b} = {c}.",
    ],
}

# EN mirror — same indexing as REVERSE_STEP_PROSE so render_reverse can
# emit parallel EN chains alongside the EO ones.
REVERSE_STEP_PROSE_EN: dict[str, list[str]] = {
    "mul": [
        "We reverse the multiplication: {a} / {b} = {c}.",
        "We divide to reverse: {a} / {b} = {c}.",
        "The original number is {a} / {b} = {c}.",
        "So the original quantity was {a} / {b} = {c}.",
        "Dividing to reverse: {a} / {b} = {c}.",
        "Undo the multiplication: {a} / {b} = {c}.",
        "Invert the multiplication by division: {a} / {b} = {c}.",
        "The previous value was {a} / {b} = {c}.",
        "Backwards by division: {a} / {b} = {c}.",
        "So the input to the multiplication was {c}, since {a} / {b} = {c}.",
    ],
    "add": [
        "We subtract the addition: {a} - {b} = {c}.",
        "The original value was {a} - {b} = {c}.",
        "So before the addition it was {a} - {b} = {c}.",
        "Reverse the addition: {a} - {b} = {c}.",
        "Undo the addition: {a} - {b} = {c}.",
        "Subtracting the addition: {a} - {b} = {c}.",
        "Invert the addition: {a} - {b} = {c}.",
        "The previous value was {a} - {b} = {c}.",
        "Backwards by subtraction: {a} - {b} = {c}.",
    ],
    "sub": [
        "We add to reverse: {a} + {b} = {c}.",
        "So before the subtraction it was {a} + {b} = {c}.",
        "The original value was {a} + {b} = {c}.",
        "Reverse the subtraction: {a} + {b} = {c}.",
        "Undo the subtraction: {a} + {b} = {c}.",
        "Add to invert: {a} + {b} = {c}.",
        "Restore what was removed: {a} + {b} = {c}.",
        "The previous value was {a} + {b} = {c}.",
        "Backwards by addition: {a} + {b} = {c}.",
    ],
    "div": [
        "We multiply to reverse: {a} * {b} = {c}.",
        "The original number was {a} * {b} = {c}.",
        "So before the division it was {a} * {b} = {c}.",
        "Reverse the division: {a} * {b} = {c}.",
        "Undo the division: {a} * {b} = {c}.",
        "Multiply to invert: {a} * {b} = {c}.",
        "The previous value was {a} * {b} = {c}.",
        "Backwards by multiplication: {a} * {b} = {c}.",
        "So the input to the division was {a} * {b} = {c}.",
    ],
    "frac": [
        "We reverse the fraction: {a} * {d} / {n} = {c}.",
        "The original base was {a} * {d} / {n} = {c}.",
        "So the original base was {a} * {d} / {n} = {c}.",
        "Undo the fraction: {a} * {d} / {n} = {c}.",
        "Invert {n}/{d}: multiplying by {d}/{n} gives {c}.",
        "Backwards through the fraction: {a} * {d} / {n} = {c}.",
        "The previous base was {a} * {d} / {n} = {c}.",
        "Inverting {n}/{d}: {a} * {d} / {n} = {c}.",
    ],
    "pct": [
        "We reverse the percentage: {a} * 100 / {b} = {c}.",
        "The original base was {a} * 100 / {b} = {c}.",
        "So the original base was {a} * 100 / {b} = {c}.",
        "Undo the percentage: {a} * 100 / {b} = {c}.",
        "Invert {b}%: {a} * 100 / {b} = {c}.",
        "Backwards through the percentage: {a} * 100 / {b} = {c}.",
        "The previous base was {a} * 100 / {b} = {c}.",
        "Division by {b}% gives {a} * 100 / {b} = {c}.",
    ],
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


def maybe_frame_bi(rng: random.Random, p: float = 0.35) -> tuple[str, str]:
    """Bilingual variant — returns (eo_frame, en_frame) tuple, indexed
    identically into SCENARIO_FRAMES / SCENARIO_FRAMES_EN. Both empty when
    no frame is picked.
    """
    if rng.random() < p:
        idx = rng.randrange(len(SCENARIO_FRAMES))
        return SCENARIO_FRAMES[idx] + " ", SCENARIO_FRAMES_EN[idx] + " "
    return "", ""


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


def render_qty_en(n: int, noun_en: tuple[str, str]) -> str:
    """English quantity phrasing — singular/plural agreement.

    noun_en: (singular, plural). Simpler than EO — no accusative case.
        render_qty_en(1, ("boy", "boys"))  → "1 boy"
        render_qty_en(5, ("boy", "boys"))  → "5 boys"
    """
    return f"{n} {noun_en[0] if n == 1 else noun_en[1]}"


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

    @staticmethod
    def _pick_emit(ctx: Ctx, lib_eo: dict, role: str, lib_en: dict | None,
                    **fmt_kwargs) -> None:
        """Sample an index once from the EO library, emit both EO and EN
        (if EN lib provided) using the SAME index.

        This is how we keep parallel EO/EN alignment: index-per-call, not
        independent random picks. If EN lib exists but has fewer templates
        at that index, we skip EN emission for this call (partial coverage
        is fine — recipe checks len(prose_en) == len(prose) at render).

        Use like:
            self._pick_emit(ctx, _MUL_PROSE, role, _MUL_PROSE_EN,
                             a=fmt_num(a), b=fmt_num(b), c=fmt_num(c))
        """
        variants_eo = lib_eo[role]
        idx = ctx.rng.randrange(len(variants_eo))
        ctx.prose.append(variants_eo[idx].format(**fmt_kwargs))
        if lib_en is not None:
            variants_en = lib_en.get(role, [])
            if idx < len(variants_en):
                ctx.prose_en.append(variants_en[idx].format(**fmt_kwargs))


class Mul(Op):
    kind = "mul"
    def apply(self, ctx: Ctx) -> None:
        a, b = ctx.n(self.lhs), ctx.n(self.rhs)
        c = a * b
        ctx.bind(self.out, c, noun=ctx.get(self.lhs).noun)
        self._chain_line(ctx, a, "*", b, c)
        role = "first" if not ctx.prose else "chained"
        self._pick_emit(ctx, _MUL_PROSE, role, _MUL_PROSE_EN,
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
        self._pick_emit(ctx, _ADD_PROSE, role, _ADD_PROSE_EN,
                         a=fmt_num(a), b=fmt_num(b), c=fmt_num(c))
        ctx.applied_ops.append(self)

    def reverse_step(self, out_val, known_side, known_val):
        # out = lhs + rhs → unknown = out - known
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
        self._pick_emit(ctx, _SUB_PROSE, role, _SUB_PROSE_EN,
                         a=fmt_num(a), b=fmt_num(b), c=fmt_num(c))
        ctx.applied_ops.append(self)

    def reverse_step(self, out_val, known_side, known_val):
        # out = lhs - rhs. If lhs known → rhs = lhs - out; if rhs known → lhs = out + rhs.
        return known_val - out_val if known_side == "lhs" else out_val + known_val

    def reverse_chain_line(self, out_val, known_side, known_val, unknown):
        if known_side == "lhs":
            return f"{fmt_num(known_val)} - {fmt_num(out_val)} = {fmt_num(unknown)}"
        return f"{fmt_num(out_val)} + {fmt_num(known_val)} = {fmt_num(unknown)}"


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
        self._pick_emit(ctx, _DIV_PROSE, role, _DIV_PROSE_EN,
                         a=fmt_num(a), b=fmt_num(b), c=fmt_num(c))
        ctx.applied_ops.append(self)

    def reverse_step(self, out_val, known_side, known_val):
        # out = lhs / rhs. If lhs known → rhs = lhs / out; if rhs known → lhs = out * rhs.
        return known_val / out_val if known_side == "lhs" else out_val * known_val

    def reverse_chain_line(self, out_val, known_side, known_val, unknown):
        if known_side == "lhs":
            return f"{fmt_num(known_val)} / {fmt_num(out_val)} = {fmt_num(unknown)}"
        return f"{fmt_num(out_val)} * {fmt_num(known_val)} = {fmt_num(unknown)}"


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
        self._pick_emit(ctx, _FRAC_PROSE, role, _FRAC_PROSE_EN,
                         n=self.num, d=self.denom, b=fmt_num(base), r=fmt_num(result))
        ctx.applied_ops.append(self)

    def reverse_step(self, out_val, known_side, known_val=None):
        # We only invert on `base`: out = base * num/denom → base = out * denom / num
        assert self.num != 0
        return out_val * self.denom / self.num

    def reverse_chain_line(self, out_val, known_side, known_val, unknown):
        return f"{fmt_num(out_val)} * {self.denom} / {self.num} = {fmt_num(unknown)}"


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
        self._pick_emit(ctx, _AVG_PROSE, role, _AVG_PROSE_EN,
                         vals=sum_str, t=fmt_num(total), n=n, a=fmt_num(avg))
        ctx.applied_ops.append(self)
    # NOTE: Avg has no clean single-var inverse → skip in reverse mode.


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
            self._pick_emit(ctx, _PCT_DECIMAL_PROSE, role, _PCT_DECIMAL_PROSE_EN,
                             p=pct, d=dec_str, b=fmt_num(base), a=fmt_num(amount))
        else:
            # direct: pct/100 * base = amount
            ctx.chain.append(f"{pct} / 100 * {fmt_num(base)} = {fmt_num(amount)}")
            role = "first" if not ctx.prose else "chained"
            self._pick_emit(ctx, _PCT_DIRECT_PROSE, role, _PCT_DIRECT_PROSE_EN,
                             p=pct, b=fmt_num(base), a=fmt_num(amount))
        ctx.applied_ops.append(self)

    def reverse_step(self, out_val, known_side, known_val):
        # out = base * pct/100. We invert on `base` (rhs) given pct known (lhs).
        # → base = out * 100 / pct
        assert known_val != 0
        return out_val * 100 / known_val

    def reverse_chain_line(self, out_val, known_side, known_val, unknown):
        return (f"{fmt_num(out_val)} * 100 / {fmt_num(known_val)} "
                f"= {fmt_num(unknown)}")


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
        eq_str = f"{c}{v} + {b}" if b > 0 else f"{c}{v} - {-b}" if b < 0 else f"{c}{v}"
        self._pick_emit(ctx, _LINSOLVE_PROSE, role, _LINSOLVE_PROSE_EN,
                         v=v, eq=eq_str, t=t, c=c, num=num, x=x)
        ctx.applied_ops.append(self)


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
# ─── EN mirrors ─────────────────────────────────────────────────────────────
# Parallel English templates. Same key set + same placeholder names as the EO
# libs above. Indexed 1:1 with EO so Op._pick_emit() picks the same slot in both.
# Recipes that stay EO-only don't need to touch these; recipes with EN openers
# populate the parallel Ctx.prose_en, which triggers question_en/answer_en at
# render time. See design note above Op._pick_emit.

_MUL_PROSE_EN = {
    "first": [
        "There are {a} * {b} = {c} in total.",
        "{a} times {b} equals {c}.",
        "The total number is {a} * {b} = {c}.",
        "We multiply: {a} * {b} = {c}.",
        "{a} groups of {b} give {a} * {b} = {c}.",
        "Calculation: {a} * {b} = {c}.",
        "First compute the product: {a} * {b} = {c}.",
        "The product of {a} and {b} is {c}.",
        "Multiplying {a} by {b}, we get {c}.",
        "Start by multiplying: {a} * {b} = {c}.",
        "Initial calculation: {a} * {b} = {c}.",
        "Finding the product of {a} and {b}: {a} * {b} = {c}.",
        "Simply: {a} * {b} = {c}.",
        "Note that {a} * {b} = {c}.",
        "Multiplication result: {a} * {b} = {c}.",
    ],
    "chained": [
        "Then, {a} * {b} = {c}.",
        "Now we multiply by {b}: {a} * {b} = {c}.",
        "{a} * {b} = {c}.",
        "That gives {a} * {b} = {c}.",
        "Multiplying: {a} * {b} = {c}.",
        "Next, {a} * {b} = {c}.",
        "So {a} * {b} = {c}.",
        "Therefore {a} * {b} = {c}.",
        "Now compute: {a} * {b} = {c}.",
        "Multiplying by {b}: {a} * {b} = {c}.",
        "Result: {a} * {b} = {c}.",
        "Hence {a} * {b} = {c}.",
        "We multiply and get {c}.",
    ],
}

_ADD_PROSE_EN = {
    "first": [
        "We add: {a} + {b} = {c}.",
        "Altogether there are {a} + {b} = {c}.",
        "Together: {a} + {b} = {c}.",
        "{a} plus {b} gives {c}.",
        "The sum is {a} + {b} = {c}.",
        "Summing up, {a} + {b} = {c}.",
        "Addition: {a} + {b} = {c}.",
        "The combined value is {a} + {b} = {c}.",
        "Sum calculation: {a} + {b} = {c}.",
        "Let's sum: {a} + {b} = {c}.",
        "The total from {a} and {b} is {c}.",
        "Adding {a} and {b}, we get {c}.",
    ],
    "chained": [
        "Add {b} more: {a} + {b} = {c}.",
        "Now there are {a} + {b} = {c}.",
        "Together, {a} + {b} = {c}.",
        "Therefore: {a} + {b} = {c}.",
        "{a} + {b} = {c}.",
        "Then sum: {a} + {b} = {c}.",
        "Next {a} + {b} = {c}.",
        "That leads to {a} + {b} = {c}.",
        "Now add {b}: {a} + {b} = {c}.",
        "The new sum: {a} + {b} = {c}.",
        "After adding: {a} + {b} = {c}.",
        "Combined, {a} + {b} = {c}.",
        "So the combined result is {c}.",
    ],
}

_SUB_PROSE_EN = {
    "first": [
        "We subtract: {a} - {b} = {c}.",
        "Remaining: {a} - {b} = {c}.",
        "{a} minus {b} equals {c}.",
        "Subtracting {b} from {a}: {a} - {b} = {c}.",
        "The difference is {a} - {b} = {c}.",
        "After subtraction: {a} - {b} = {c}.",
        "Difference calculation: {a} - {b} = {c}.",
        "We remove {b}: {a} - {b} = {c}.",
        "The remainder equals {a} - {b} = {c}.",
        "Reduction: {a} - {b} = {c}.",
    ],
    "chained": [
        "After {b} leaves, {a} - {b} = {c} remain.",
        "Remaining: {a} - {b} = {c}.",
        "Subtracting {b}: {a} - {b} = {c}.",
        "{a} - {b} = {c}.",
        "So {a} - {b} = {c} remain.",
        "Now {a} - {b} = {c}.",
        "So the difference: {a} - {b} = {c}.",
        "Reducing by {b}: {a} - {b} = {c}.",
        "After removing {b}: {a} - {b} = {c}.",
        "The remaining value is {c}.",
        "Then the difference: {a} - {b} = {c}.",
        "This leaves {a} - {b} = {c}.",
    ],
}

_DIV_PROSE_EN = {
    "first": [
        "We divide: {a} / {b} = {c}.",
        "{a} / {b} = {c}.",
        "Each part gets {a} / {b} = {c}.",
        "Division: {a} / {b} = {c}.",
        "The quotient is {a} / {b} = {c}.",
        "Dividing {a} by {b}, we get {c}.",
        "Distribute {a} among {b}: {a} / {b} = {c}.",
        "Each receives {a} / {b} = {c}.",
        "Equally split: {a} / {b} = {c}.",
        "Division calculation: {a} / {b} = {c}.",
    ],
    "chained": [
        "Dividing among {b}: {a} / {b} = {c}.",
        "Each group gets {a} / {b} = {c}.",
        "{a} / {b} = {c}.",
        "So each receives {a} / {b} = {c}.",
        "Next {a} / {b} = {c}.",
        "Then divide: {a} / {b} = {c}.",
        "Distributing {a} across {b}: {a} / {b} = {c}.",
        "Each part: {a} / {b} = {c}.",
        "Now dividing by {b}: {a} / {b} = {c}.",
        "So {a} / {b} = {c}.",
        "The quotient is {a} / {b} = {c}.",
    ],
}

_FRAC_PROSE_EN = {
    "first": [
        "We find {n}/{d} of {b}: {n}/{d} * {b} = {r}.",
        "{n}/{d} of {b} is {r}.",
        "Calculate {n}/{d} of {b}: {n}/{d} * {b} = {r}.",
        "The {n}/{d} part of {b} equals {r}.",
        "Take {n}/{d} of {b}: {n}/{d} * {b} = {r}.",
        "First find {n}/{d}: {n}/{d} * {b} = {r}.",
        "The fraction {n}/{d} of {b} gives {r}.",
        "Multiplying {b} by {n}/{d}: {n} * {b} / {d} = {r}.",
        "Applying {n}/{d} to {b}: {r}.",
        "{n}/{d} of {b}: {r}.",
    ],
    "chained": [
        "Then {n}/{d} of {b}: {n}/{d} * {b} = {r}.",
        "{n}/{d} of {b} = {r}.",
        "Now find {n}/{d}: {n}/{d} * {b} = {r}.",
        "Next the fraction {n}/{d} of {b} gives {r}.",
        "From that, {n}/{d} of {b} = {r}.",
        "So {n}/{d} of {b} equals {r}.",
        "Applying {n}/{d}: {n}/{d} * {b} = {r}.",
        "And {n}/{d} of {b} = {r}.",
        "So {n}/{d} of {b} is {r}.",
    ],
}

_AVG_PROSE_EN = {
    "first": [
        "Sum: {vals} = {t}. Mean value: {t} / {n} = {a}.",
        "Add them all: {vals} = {t}. Mean: {t} / {n} = {a}.",
        "The mean value is ({vals}) / {n} = {a}.",
        "Sum: {vals} = {t}. Mean: {t} / {n} = {a}.",
        "The sum is {vals} = {t}. Dividing by {n}: {t} / {n} = {a}.",
        "Compute the sum: {vals} = {t}. Then {t} / {n} = {a}.",
        "First sum ({vals}) = {t}, then divide by {n}: {a}.",
        "The arithmetic mean: ({vals}) / {n} = {a}.",
        "Find the mean value: ({vals}) / {n} = {a}.",
    ],
    "chained": [
        "Now the mean: ({vals}) / {n} = {a}.",
        "The sum is {t}, so the mean = {t} / {n} = {a}.",
        "Then compute the mean: {t} / {n} = {a}.",
        "The sum {t} divided by {n} gives {a}.",
        "So the mean value is {t} / {n} = {a}.",
        "And the average is {t} / {n} = {a}.",
        "So the mean is ({vals}) / {n} = {a}.",
        "Mean value = {t} / {n} = {a}.",
    ],
}

_PCT_DIRECT_PROSE_EN = {
    "first": [
        "We compute {p}% of {b}: {p}/100 * {b} = {a}.",
        "{p}% of {b} is {p}/100 * {b} = {a}.",
        "Find {p}%: {p}/100 * {b} = {a}.",
        "The percentage {p}% of {b} gives {p}/100 * {b} = {a}.",
        "Applying {p}% to {b}: {p}/100 * {b} = {a}.",
        "Multiplying {b} by {p}/100: {a}.",
        "{p}% of {b} equals {p} * {b} / 100 = {a}.",
        "Calculation of {p}%: {p}/100 * {b} = {a}.",
        "First find {p}% of {b}: {a}.",
        "The {p}-percent value of {b} is {a}.",
    ],
    "chained": [
        "Then we find {p}% of {b}: {p}/100 * {b} = {a}.",
        "{p}% of {b} = {p}/100 * {b} = {a}.",
        "Now applying {p}% to {b}: {a}.",
        "So {p}% of {b} equals {a}.",
        "And {p}% of {b} gives {a}.",
        "Thus {p}/100 * {b} = {a}.",
        "After applying {p}%: {p}/100 * {b} = {a}.",
        "So {p}% of {b} = {a}.",
    ],
}

_PCT_DECIMAL_PROSE_EN = {
    "first": [
        "First convert {p}% to a decimal: {p}/100 = {d}. Now {d} * {b} = {a}.",
        "{p}% = {d}, so {d} * {b} = {a}.",
        "Convert {p}% to a decimal: {p}/100 = {d}. Then {d} * {b} = {a}.",
        "The decimal for {p}% is {d}, so {d} * {b} = {a}.",
        "Write {p}% as {d}: {d} * {b} = {a}.",
        "In decimal form: {p}/100 = {d}. Multiplying: {d} * {b} = {a}.",
        "Treat {p}% as {d}, so {d} * {b} = {a}.",
        "The percentage multiplier is {d}: {d} * {b} = {a}.",
    ],
    "chained": [
        "Convert {p}% to a decimal: {p}/100 = {d}. Then {d} * {b} = {a}.",
        "Now {p}% = {d}, so {d} * {b} = {a}.",
        "Then convert {p}% to {d} and multiply: {d} * {b} = {a}.",
        "Use the decimal {d} = {p}/100: {d} * {b} = {a}.",
        "So {p}/100 = {d}, so {d} * {b} = {a}.",
        "In decimal form: {d} * {b} = {a}.",
        "Thus {d} * {b} = {a}.",
    ],
}

_LINSOLVE_PROSE_EN = {
    "first": [
        "Let {v} be the unknown. The equation: {eq} = {t}. Solving for {v}: {v} = {x}.",
        "Let {v} be the unknown. So {eq} = {t}. Solving, {v} = {x}.",
        "Denote the unknown by {v}. The equation becomes {eq} = {t}. Isolating: {v} = {x}.",
        "Define {v} as the unknown. The equation {eq} = {t} gives {v} = {x}.",
        "Set up the equation: {eq} = {t}. Solving for {v}, we get {v} = {x}.",
        "If {v} is the unknown, then {eq} = {t}, so {v} = {x}.",
        "Use {v} for the unknown. Equation: {eq} = {t}. Solution: {v} = {x}.",
        "Set up the equation {eq} = {t} where {v} is unknown. Result: {v} = {x}.",
        "Defining {v} as the unknown: {eq} = {t}, so {v} = {x}.",
        "Mark the unknown as {v}. The equation {eq} = {t} solved gives {v} = {x}.",
    ],
    "chained": [
        "Now set up the equation: {eq} = {t}. Solution: {v} = {x}.",
        "The next equation is {eq} = {t}, so {v} = {x}.",
        "Then build the equation {eq} = {t}. Solving: {v} = {x}.",
        "Next, the equation {eq} = {t} gives {v} = {x}.",
        "Now {eq} = {t}, so {v} = {x}.",
        "From that, {eq} = {t}, and {v} = {x}.",
        "And the equation {eq} = {t} solved: {v} = {x}.",
    ],
}


_LINSOLVE_PROSE = {
    "first": [
        "Estu {v} la nekonato. La ekvacio: {eq} = {t}. Ni solvas por {v}: {v} = {x}.",
        "Estu {v} la nekonato. Do {eq} = {t}. Solvante, {v} = {x}.",
        "Ni indiku la nekonaton per {v}. La ekvacio iĝas {eq} = {t}. Post izolado: {v} = {x}.",
        "Difinu {v} kiel la nekonato. La ekvacio {eq} = {t} donas {v} = {x}.",
        "Ni skribu la ekvacion: {eq} = {t}. Solvante por {v}, ni ricevas {v} = {x}.",
        "Se {v} estas la nekonato, tiam {eq} = {t}, do {v} = {x}.",
        "Uzu {v} por la nekonato. Ekvacio: {eq} = {t}. Solvo: {v} = {x}.",
        "Ni starigu la ekvacion {eq} = {t} kie {v} estas nekonato. Rezulto: {v} = {x}.",
        "Definante {v} kiel la nekonato: {eq} = {t}, do {v} = {x}.",
        "Marku la nekonaton per {v}. La ekvacio {eq} = {t} solvita donas {v} = {x}.",
    ],
    "chained": [
        "Nun ni starigas ekvacion: {eq} = {t}. Solvo: {v} = {x}.",
        "La sekva ekvacio estas {eq} = {t}, do {v} = {x}.",
        "Poste konstruu ekvacion {eq} = {t}. Solvante: {v} = {x}.",
        "Sekve la ekvacio {eq} = {t} donas {v} = {x}.",
        "Nun {eq} = {t}, do {v} = {x}.",
        "El tio, {eq} = {t}, kaj {v} = {x}.",
        "Kaj la ekvacio {eq} = {t} solvita: {v} = {x}.",
    ],
}
_FRAC_PROSE = {
    "first": [
        "Ni trovas {n}/{d} el {b}: {n}/{d} * {b} = {r}.",
        "{n}/{d} de {b} estas {r}.",
        "Kalkulu {n}/{d} de {b}: {n}/{d} * {b} = {r}.",
        "La {n}/{d} parto de {b} egalas {r}.",
        "Prenu {n}/{d} el {b}: {n}/{d} * {b} = {r}.",
        "Unue trovi {n}/{d}: {n}/{d} * {b} = {r}.",
        "La frakcio {n}/{d} de {b} donas {r}.",
        "Multiplikante {b} per {n}/{d}: {n} * {b} / {d} = {r}.",
        "Aplikante {n}/{d} al {b}: {r}.",
        "{n}/{d} el {b}: {r}.",
    ],
    "chained": [
        "Poste {n}/{d} el {b}: {n}/{d} * {b} = {r}.",
        "{n}/{d} el {b} = {r}.",
        "Nun trovi {n}/{d}: {n}/{d} * {b} = {r}.",
        "Sekve la frakcio {n}/{d} de {b} donas {r}.",
        "El tio, {n}/{d} el {b} = {r}.",
        "Tial {n}/{d} de {b} egalas {r}.",
        "Aplikante {n}/{d}: {n}/{d} * {b} = {r}.",
        "Kaj {n}/{d} el {b} = {r}.",
        "Do {n}/{d} el {b} estas {r}.",
    ],
}
_AVG_PROSE = {
    "first": [
        "Sumo: {vals} = {t}. Meza valoro: {t} / {n} = {a}.",
        "Aldonu ĉiujn: {vals} = {t}. Meza: {t} / {n} = {a}.",
        "Meza valoro estas ({vals}) / {n} = {a}.",
        "Sumigi: {vals} = {t}. Meza: {t} / {n} = {a}.",
        "La sumo estas {vals} = {t}. Dividante per {n}: {t} / {n} = {a}.",
        "Kalkulu sumon: {vals} = {t}. Tiam {t} / {n} = {a}.",
        "Unue sumigu ({vals}) = {t}, poste dividu per {n}: {a}.",
        "La aritmetika mezumo: ({vals}) / {n} = {a}.",
        "Trovi la mezan valoron: ({vals}) / {n} = {a}.",
    ],
    "chained": [
        "Nun la meza: ({vals}) / {n} = {a}.",
        "Sumo estas {t}, do meza = {t} / {n} = {a}.",
        "Poste kalkulu la mezumon: {t} / {n} = {a}.",
        "La sumo {t} dividita per {n} donas {a}.",
        "Do la meza valoro estas {t} / {n} = {a}.",
        "Kaj la mezumo estas {t} / {n} = {a}.",
        "Sekve la meza estas ({vals}) / {n} = {a}.",
        "Meza valoro = {t} / {n} = {a}.",
    ],
}
_PCT_DIRECT_PROSE = {
    "first": [
        "Ni kalkulas {p}% el {b}: {p}/100 * {b} = {a}.",
        "{p}% de {b} estas {p}/100 * {b} = {a}.",
        "Trovi {p}%: {p}/100 * {b} = {a}.",
        "La procento {p}% de {b} donas {p}/100 * {b} = {a}.",
        "Aplikante {p}% al {b}: {p}/100 * {b} = {a}.",
        "Multiplikante {b} per {p}/100: {a}.",
        "{p}% el {b} egalas {p} * {b} / 100 = {a}.",
        "Kalkulo de {p}%: {p}/100 * {b} = {a}.",
        "Unue trovi {p}% el {b}: {a}.",
        "La {p}-procentaĵo de {b} estas {a}.",
    ],
    "chained": [
        "Poste ni trovas {p}% de {b}: {p}/100 * {b} = {a}.",
        "{p}% el {b} = {p}/100 * {b} = {a}.",
        "Nun aplikante {p}% al {b}: {a}.",
        "Sekve {p}% de {b} egalas {a}.",
        "Kaj {p}% el {b} donas {a}.",
        "Tial {p}/100 * {b} = {a}.",
        "Post apliki {p}%: {p}/100 * {b} = {a}.",
        "Do {p}% de {b} = {a}.",
    ],
}
_PCT_DECIMAL_PROSE = {
    "first": [
        "Unue transformu {p}% al decimalo: {p}/100 = {d}. Nun {d} * {b} = {a}.",
        "{p}% = {d}, tial {d} * {b} = {a}.",
        "Konvertu {p}% al decimalo: {p}/100 = {d}. Tiam {d} * {b} = {a}.",
        "La decimalo por {p}% estas {d}, do {d} * {b} = {a}.",
        "Ni skribu {p}% kiel {d}: {d} * {b} = {a}.",
        "En decimala formo: {p}/100 = {d}. Multiplikante: {d} * {b} = {a}.",
        "Traktu {p}% kiel {d}, do {d} * {b} = {a}.",
        "La procenta multobligilo estas {d}: {d} * {b} = {a}.",
    ],
    "chained": [
        "Konvertu {p}% al decimalo: {p}/100 = {d}. Tiam {d} * {b} = {a}.",
        "Nun {p}% = {d}, do {d} * {b} = {a}.",
        "Poste transformu {p}% al {d} kaj multipliku: {d} * {b} = {a}.",
        "Uzu la decimalon {d} = {p}/100: {d} * {b} = {a}.",
        "Sekve {p}/100 = {d}, do {d} * {b} = {a}.",
        "En decimala formo: {d} * {b} = {a}.",
        "Tial {d} * {b} = {a}.",
    ],
}


# ─── Recipe 1: ratio_parts ──────────────────────────────────────────────────

def ratio_parts_recipe(rng: random.Random, n_steps: int = 2,
                        reverse: bool = False) -> dict:
    """N groups × K per group → total. Optionally: minus absent, then / packs.

    When reverse=True, the total (or final) is stated as GIVEN and the
    per-group value is asked as the UNKNOWN. Chain walks the applied Ops
    backwards via ctx.render_reverse.
    """
    # Resample outer parameters until (n_steps=4 case) integer division works.
    for _try in range(100):
        ctx = Ctx.new(rng)
        # Pick nouns by INDEX so we can look up parallel EN nouns from the
        # mirror lists.
        child_idx = rng.randrange(len(CHILDLIKE_NOUNS))
        group_idx = rng.randrange(len(GROUPING_NOUNS))
        child = CHILDLIKE_NOUNS[child_idx]
        group = GROUPING_NOUNS[group_idx]
        child_en = CHILDLIKE_NOUNS_EN[child_idx]
        group_en = GROUPING_NOUNS_EN[group_idx]

        n_groups = rng.randint(3, 12)
        per_group = rng.randint(4, 15)
        ctx.bind("groups", n_groups, noun=group)
        ctx.bind("per_group", per_group, noun=child)

        if not reverse:
            frame, frame_en = maybe_frame_bi(rng)
            # Parallel opener libs — indexed identically. EO first, EN second.
            openers_eo = [
                f"{frame}estas {render_qty(n_groups, group)} kun {render_qty(per_group, child)} en ĉiu.",
                f"{frame}en {render_qty(n_groups, group)}, ĉiu enhavas {render_qty(per_group, child)}.",
                f"{frame}{ctx.protagonist} vidas {qty_acc(n_groups, group)}, ĉiun kun {render_qty(per_group, child)}.",
                f"{frame}ĉiu el la {render_qty(n_groups, group)} havas {qty_acc(per_group, child)}.",
                f"{frame}oni disdonis {render_qty(per_group, child)} en ĉiun de {render_qty(n_groups, group)}.",
                f"{frame}{render_qty(n_groups, group)} estas plenaj de {render_qty(per_group, child)} ĉiu.",
            ]
            openers_en = [
                f"{frame_en}there are {render_qty_en(n_groups, group_en)} with {render_qty_en(per_group, child_en)} in each.",
                f"{frame_en}in {render_qty_en(n_groups, group_en)}, each contains {render_qty_en(per_group, child_en)}.",
                f"{frame_en}{ctx.protagonist} sees {render_qty_en(n_groups, group_en)}, each with {render_qty_en(per_group, child_en)}.",
                f"{frame_en}each of the {render_qty_en(n_groups, group_en)} has {render_qty_en(per_group, child_en)}.",
                f"{frame_en}{render_qty_en(per_group, child_en)} were placed into each of {render_qty_en(n_groups, group_en)}.",
                f"{frame_en}{render_qty_en(n_groups, group_en)} are full of {render_qty_en(per_group, child_en)} each.",
            ]
            idx = rng.randrange(len(openers_eo))
            opener = openers_eo[idx]
            opener_en = openers_en[idx]
            if not frame:
                opener = opener[0].upper() + opener[1:]
                opener_en = opener_en[0].upper() + opener_en[1:]
            q = [opener]
            q_en = [opener_en]
        else:
            # Reverse: STATE groups, but HIDE per_group (mark unknown).
            frame, frame_en = maybe_frame_bi(rng)
            r_openers_eo = [
                f"{frame}estas {render_qty(n_groups, group)}, ĉiu enhavanta la saman nekonatan nombron de {child[1]}.",
                f"{frame}en {render_qty(n_groups, group)}, ĉiu havas la saman nombron de {child[1]}.",
                f"{frame}{ctx.protagonist} havas {qty_acc(n_groups, group)}, ĉiu kun sama sed nekonata nombro de {child[1]}.",
                f"{frame}oni disdonis egalajn kvantojn de {child[1]} en ĉiun de {render_qty(n_groups, group)}.",
            ]
            r_openers_en = [
                f"{frame_en}there are {render_qty_en(n_groups, group_en)}, each containing the same unknown number of {child_en[1]}.",
                f"{frame_en}in {render_qty_en(n_groups, group_en)}, each has the same number of {child_en[1]}.",
                f"{frame_en}{ctx.protagonist} has {render_qty_en(n_groups, group_en)}, each with the same but unknown number of {child_en[1]}.",
                f"{frame_en}equal quantities of {child_en[1]} were placed into each of {render_qty_en(n_groups, group_en)}.",
            ]
            r_idx = rng.randrange(len(r_openers_eo))
            opener = r_openers_eo[r_idx]
            opener_en = r_openers_en[r_idx]
            if not frame:
                opener = opener[0].upper() + opener[1:]
                opener_en = opener_en[0].upper() + opener_en[1:]
            q = [opener]
            q_en = [opener_en]

        Mul("groups", "per_group", "total").apply(ctx)
        final_var = "total"

        if n_steps >= 3:
            absent = rng.randint(1, min(6, int(ctx.n("total")) // 2))
            ctx.bind("absent", absent, noun=child)
            q.append(f"{render_qty(absent, child, case='nom')} forestas.")
            verb = "is" if absent == 1 else "are"
            q_en.append(f"{render_qty_en(absent, child_en)} {verb} absent.")
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
            q_en.append(f"They divide into {n_packs} equal groups.")
            Div(final_var, "packs", "per_pack").apply(ctx)
            final_var = "per_pack"

        if not reverse:
            closers_eo = [
                f"Kiom da {child[1]} estas en la fina rezulto?",
                f"Kiu estas la fina nombro de {child[1]}?",
                f"Kalkulu la finan nombron de {child[1]}.",
                f"Trovu kiom da {child[1]} restas fine.",
                f"Kiom da {child[1]} estas fine?",
                f"Determinu la finan kvanton de {child[1]}.",
            ]
            closers_en = [
                f"How many {child_en[1]} are in the final result?",
                f"What is the final number of {child_en[1]}?",
                f"Calculate the final number of {child_en[1]}.",
                f"Find how many {child_en[1]} are left at the end.",
                f"How many {child_en[1]} are there at the end?",
                f"Determine the final quantity of {child_en[1]}.",
            ]
            c_idx = rng.randrange(len(closers_eo))
            q.append(closers_eo[c_idx])
            q_en.append(closers_en[c_idx])
            return ctx.render(" ".join(q), final_var, question_en=" ".join(q_en))

        # Reverse: state the final value and ask for per_group
        final_val = int(ctx.n(final_var))
        state_finals_eo = [
            f"Fine, entute estas {render_qty(final_val, child, case='nom')}.",
            f"La fina kvanto estas {render_qty(final_val, child, case='nom')}.",
            f"Post ĉio, la rezulto estas {render_qty(final_val, child, case='nom')}.",
        ]
        state_finals_en = [
            f"In the end, there are a total of {render_qty_en(final_val, child_en)}.",
            f"The final quantity is {render_qty_en(final_val, child_en)}.",
            f"After everything, the result is {render_qty_en(final_val, child_en)}.",
        ]
        sf_idx = rng.randrange(len(state_finals_eo))
        q.append(state_finals_eo[sf_idx])
        q_en.append(state_finals_en[sf_idx])
        closers_eo = [
            f"Kiom da {child[1]} estas en ĉiu {group[0]}?",
            f"Trovu la nombron de {child[1]} en ĉiu {group[0]}.",
            f"Kalkulu kiom da {child[1]} enhavis ĉiu {group[0]}.",
        ]
        closers_en = [
            f"How many {child_en[1]} are in each {group_en[0]}?",
            f"Find the number of {child_en[1]} in each {group_en[0]}.",
            f"Calculate how many {child_en[1]} each {group_en[0]} contained.",
        ]
        c_idx = rng.randrange(len(closers_eo))
        result = ctx.render_reverse(
            forward_prose=" ".join(q),
            forward_final_var=final_var,
            ask_var="per_group",
            closer=closers_eo[c_idx],
            forward_prose_en=" ".join(q_en),
            closer_en=closers_en[c_idx],
        )
        result["recipe"] = "ratio_parts_reverse"
        result["n_steps"] = n_steps
        result["direction"] = "reverse"
        return result

    raise RuntimeError("ratio_parts_recipe: couldn't sample divisible params in 100 tries")


# ─── Recipe 2: percent ──────────────────────────────────────────────────────

def percent_recipe(rng: random.Random, n_steps: int = 2, op: str | None = None,
                    reverse: bool = False) -> dict:
    """Percent problems in 3 notation styles × 5 scenarios.

    n_steps=2: single percentage (of-amount, saving)  or  base ± pct%  (discount, markup, tax)
    n_steps=3: stacked — e.g. discount then tax

    reverse=True: only 'of-amount' and 'saving' are supported (single Pct
    linear chain). The rest use base twice → branching, deferred.
    """
    ops = ["discount", "markup", "tax", "of-amount", "saving"]
    if op is None:
        # In reverse mode, restrict to op types with a linear-chain inverse.
        op = rng.choice(["of-amount", "saving"]) if reverse else rng.choice(ops)
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
                if not reverse:
                    q = frame + rng.choice([
                        f"{p} kalkulis {pct}% de {base_qty_nom}. Kiu estas la rezulto?",
                        f"trovu {pct}% el {base_qty_nom}.",
                        f"kiom estas {pct}% de {base_qty_nom}?",
                    ])
                else:
                    q = frame + rng.choice([
                        f"{p} kalkulis {pct}% de nekonata sumo en {EUR[1]}.",
                        f"{pct}% el iu sumo en {EUR[1]} estas jena kvanto.",
                        f"{p} scias, ke {pct}% de sia buĝeto en {EUR[1]} egalas iun kvanton.",
                    ])
            elif op == "saving":
                if not reverse:
                    q = frame + rng.choice([
                        f"{p} aĉetis {item_acc} kiu kostis {base_qty_acc} "
                        f"kun {pct}% rabato. Kiom da {EUR[1]} {p} ŝparis?",
                        f"{item_nom} kostis {base_qty_acc}, kun rabato de {pct}%. "
                        f"Kiom {p} ŝparis?",
                        f"{p} akiris {pct}% rabaton sur {item_acc} de {base_qty_acc}. "
                        f"Kalkulu la ŝparon.",
                    ])
                else:
                    q = frame + rng.choice([
                        f"{p} aĉetis {item_acc} kun {pct}% rabato. "
                        f"La origina prezo estas ankoraŭ nekonata.",
                        f"{p} akiris {pct}% rabaton sur {item_acc}. "
                        f"La origina prezo estas nekonata.",
                    ])
            # Capitalize first char if no frame
            if not frame:
                q = q[0].upper() + q[1:]
        else:  # count
            item = rng.choice(COUNT_ITEMS)
            noun_acc_pl = item[3]   # "studentojn"
            noun_nom_pl = item[1]   # "studentoj"
            if not reverse:
                q = (f"En klaso estas {render_qty(base, item)}. {pct}% el ili "
                     f"portas okulvitrojn. Kiom {noun_acc_pl} portas okulvitrojn?")
            else:
                q = (f"En klaso estas nekonata nombro de {noun_nom_pl}. "
                     f"{pct}% el ili portas okulvitrojn.")

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
            if not reverse:
                q += f" Nun aldonu {pct2}% imposton sur la nova prezo. Kiu estas la fina prezo?"
            Pct("pct2", "res", "tax_amt", style="direct").apply(ctx)
            Add("res", "tax_amt", "final_price").apply(ctx)
            final = "final_price"

        if not reverse:
            return ctx.render(q, final)

        # Reverse mode: only supported for the linear-Pct paths.
        # (discount/markup/tax use base twice — deferred.)
        if op not in ("of-amount", "saving"):
            raise RuntimeError(f"percent_recipe: reverse not supported for op={op}")

        final_val = int(ctx.n(final))
        # In of-amount / saving the final is the amount (result of Pct).
        # We ask for base. Units differ by scenario:
        #   shop:  amount + base both in EUR (or SHOP_ITEMS count in some paths)
        #   count: amount + base both a count of `item`
        if scenario_kind == "count":
            state = rng.choice([
                f" {final_val} {item[1]} portas okulvitrojn.",
                f" Estas {render_qty(final_val, item)} kun okulvitroj.",
            ])
            closer = rng.choice([
                f"Kiom da {item[1]} estas en la klaso?",
                f"Trovu la totalan nombron de {item[1]}.",
                f"Kalkulu la nombron de {item[1]} en la klaso.",
            ])
        elif op == "of-amount":
            state = rng.choice([
                f" La rezulto estas {render_qty(final_val, EUR)}.",
                f" Tio egalas {render_qty(final_val, EUR)}.",
                f" Ĝi egalas {render_qty(final_val, EUR)}.",
            ])
            closer = rng.choice([
                f"Kiu estis la origina sumo en {EUR[1]}?",
                f"Kalkulu la originan sumon.",
                f"Trovu la originalan valoron.",
            ])
        else:  # saving
            state = rng.choice([
                f" {p} ŝparis {render_qty(final_val, EUR)}.",
                f" La ŝparita sumo estas {render_qty(final_val, EUR)}.",
            ])
            closer = rng.choice([
                f"Kiu estis la origina prezo de {item_nom}?",
                f"Trovu la originan prezon.",
                f"Kalkulu la originan prezon de {item_nom}.",
            ])
        q += state
        result = ctx.render_reverse(
            forward_prose=q,
            forward_final_var=final,
            ask_var="base",
            closer=closer,
        )
        result["recipe"] = "percent_reverse"
        result["n_steps"] = n_steps
        result["direction"] = "reverse"
        return result

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

def fraction_cascade_recipe(rng: random.Random, n_steps: int = 2,
                             reverse: bool = False) -> dict:
    """Fraction-of-fraction. n_steps=2: single fraction. n_steps=3: fraction of fraction.

    reverse=True: state the FINAL fraction-count as given; ask for the base.
    """
    fractions = [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 5), (2, 5), (3, 5), (4, 5),
                 (1, 6), (5, 6), (1, 7), (2, 7), (3, 7), (5, 7), (1, 8), (3, 8), (5, 8), (7, 8),
                 (1, 9), (2, 9), (4, 9), (5, 9), (7, 9), (1, 10), (3, 10), (7, 10), (9, 10)]
    # Sub-populations we cascade through (girls/red/etc.) — vary the story
    stories = [
        ("knabinoj", "portas ruĝan ĉemizon", "ruĝan ĉemizon"),
        ("knaboj", "havas biciklon", "biciklon"),
        ("studentoj", "loĝas en la urbo", "en la urbo"),
        ("lernantoj", "sciipovas naĝi", "naĝon"),
        ("infanoj", "ludas piedpilkon", "piedpilkon"),
        ("klientoj", "revenas la sekvan tagon", "la sekvan tagon"),
    ]

    for _try in range(100):
        f1 = rng.choice(fractions)
        f2 = rng.choice(fractions)
        base_noun = rng.choice(COUNT_ITEMS)
        sub_pop, verb_phrase, obj_phrase = rng.choice(stories)

        base = f1[1] * rng.randint(2, 40)
        step1_result = base * f1[0] // f1[1]
        if n_steps >= 3 and (step1_result * f2[0]) % f2[1] != 0:
            continue

        ctx = Ctx.new(rng)
        ctx.bind("base", base, noun=base_noun)
        p = ctx.protagonist
        frame = maybe_frame(rng)

        if not reverse:
            opener = rng.choice([
                f"{frame}en grupo estas {render_qty(base, base_noun)}. {f1[0]}/{f1[1]} el ili estas {sub_pop}.",
                f"{frame}el {render_qty(base, base_noun)}, {f1[0]}/{f1[1]} estas {sub_pop}.",
                f"{frame}{p} kalkulis {render_qty(base, base_noun)}; {f1[0]}/{f1[1]} el ili estas {sub_pop}.",
            ])
        else:
            opener = rng.choice([
                f"{frame}en grupo estas nekonata nombro de {base_noun[1]}. {f1[0]}/{f1[1]} el ili estas {sub_pop}.",
                f"{frame}el la grupo de {base_noun[1]}, {f1[0]}/{f1[1]} estas {sub_pop}.",
                f"{frame}{p} kalkulis grupon de {base_noun[1]}; {f1[0]}/{f1[1]} el ili estas {sub_pop}.",
            ])
        if not frame:
            opener = opener[0].upper() + opener[1:]

        Frac("base", f1[0], f1[1], "girls").apply(ctx)
        final = "girls"

        if n_steps >= 3:
            opener += f" El la {sub_pop}, {f2[0]}/{f2[1]} {verb_phrase}."
            Frac("girls", f2[0], f2[1], "red").apply(ctx)
            final = "red"
            if not reverse:
                opener += rng.choice([
                    f" Kiom {base_noun[3]} {verb_phrase}?",
                    f" Kalkulu la nombron kiuj {verb_phrase}.",
                    f" Trovu kiom {verb_phrase}.",
                ])
        else:
            if not reverse:
                opener += rng.choice([
                    f" Kiom estas {sub_pop}?",
                    f" Trovu la nombron de {sub_pop}.",
                    f" Kiom {base_noun[3]} estas {sub_pop}?",
                ])

        if not reverse:
            return ctx.render(opener, final)

        # Reverse: state final count, ask for base.
        final_val = int(ctx.n(final))
        which = sub_pop if n_steps == 2 else f"{sub_pop} kiuj {verb_phrase}"
        state_final = rng.choice([
            f" Estas {final_val} {which}.",
            f" La nombro de {which} estas {final_val}.",
            f" Fine, {final_val} el ili estas {which}.",
        ])
        opener += state_final
        closer = rng.choice([
            f"Kiom da {base_noun[1]} estas entute?",
            f"Trovu la totalan nombron de {base_noun[1]}.",
            f"Kalkulu la nombron de {base_noun[1]} en la grupo.",
        ])
        result = ctx.render_reverse(
            forward_prose=opener,
            forward_final_var=final,
            ask_var="base",
            closer=closer,
        )
        result["recipe"] = "fraction_cascade_reverse"
        result["n_steps"] = n_steps
        result["direction"] = "reverse"
        return result

    raise RuntimeError("fraction_cascade_recipe: couldn't sample")


# ─── Recipe 5: ratio_diff ───────────────────────────────────────────────────

def ratio_diff_recipe(rng: random.Random, n_steps: int = 3,
                        reverse: bool = False) -> dict:
    """N:M ratio splits `total` between two people; compute each part; report difference.
    Uses only Div + Mul + Sub — no algebra.

    reverse=True: state the diff, ask for total.  Uses recipe-local closed-form
    inverse (the forward chain uses `unit` twice — branching — so we bypass
    render_reverse).
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

        if not reverse:
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

        # ── Reverse path ──
        # diff = (b - a) * unit;  total = (a + b) * unit
        # → total = diff * (a + b) / (b - a)
        # Ensure divisibility (should be, since (a+b)*(b-a) = b^2-a^2 divides).
        # Absolute diff so numbers stay positive regardless of a vs b order.
        larger_r, smaller_r = (b, a) if b > a else (a, b)
        diff = (larger_r - smaller_r) * unit
        if diff <= 0:
            continue  # degenerate (a == b); resample
        # Build reverse question
        q = rng.choice([
            f"{names[0]} kaj {names[1]} dividas nekonatan sumon "
            f"laŭ la rilatumo {a}:{b}. La diferenco inter iliaj partoj estas "
            f"{render_qty(diff, obj)}.",
            f"En rilatumo {a}:{b}, {names[0]} kaj {names[1]} dividas iom da {obj[1]}. "
            f"Unu havas {render_qty(diff, obj)} pli ol la alia.",
            f"{names[0]} ricevas {a} partojn, {names[1]} ricevas {b} partojn "
            f"el nekonata totalo. La diferenco estas {render_qty(diff, obj)}.",
        ])
        closer = rng.choice([
            f" Kalkulu la totalan sumon.",
            f" Trovu kiom da {obj[1]} estas entute.",
            f" Kiom da {obj[1]} estas entute?",
        ])
        # Manual chain (recipe-local closed-form inverse):
        #   1) b - a = step   (difference of ratio parts)
        #   2) diff / step = unit
        #   3) a + b = parts
        #   4) unit * parts = total
        step = larger_r - smaller_r
        parts = a + b
        ctx.chain.append(f"{larger_r} - {smaller_r} = {step}")
        ctx.chain.append(f"{diff} / {step} = {unit}")
        ctx.chain.append(f"{a} + {b} = {parts}")
        ctx.chain.append(f"{unit} * {parts} = {total}")
        ctx.prose.append(
            f"La diferenco de la rilatumaj partoj estas {larger_r} - {smaller_r} = {step}."
        )
        ctx.prose.append(
            f"Do la valoro de unu parto estas {diff} / {step} = {unit}."
        )
        ctx.prose.append(
            f"La sumo de rilatumaj partoj estas {a} + {b} = {parts}."
        )
        ctx.prose.append(
            f"Do la totalo estas {unit} * {parts} = {total}."
        )
        ctx.bind("total_rev", total, noun=obj)
        result = ctx.render(q + closer, "total_rev")
        result["recipe"] = "ratio_diff_reverse"
        result["n_steps"] = n_steps
        result["direction"] = "reverse"
        return result

    raise RuntimeError("ratio_diff_recipe: couldn't sample")


# ─── Recipe 6: consec_avg ───────────────────────────────────────────────────

def consec_avg_recipe(rng: random.Random, n_steps: int = 2) -> dict:
    """N consecutive integers summing to S. Find the middle (= avg) or smallest/largest.

    Only works for odd N (so the middle is an integer) and step=1.
    """
    count = rng.choice([3, 5, 7, 9])   # odd
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

        # Time renderings: nom (for "kiom da horoj") and acc (for "bezonas 19 horojn")
        t1_acc = qty_acc(t1, tunit)                # "19 horojn"
        t2_acc = qty_acc(t2, tunit)                # "3 horojn"
        w1_nom = render_qty(w1, worker)            # "4 maŝinoj" or "1 maŝino"
        w2_nom = render_qty(w2, worker)            # "1 maŝino" etc.
        tunit_pl = tunit[1]                        # "horoj"
        workers_pl = worker[1]                     # "maŝinoj"

        frame = maybe_frame(rng)
        if ask == "find-time":
            ctx.bind("w2", w2)
            q = frame + rng.choice([
                f"{w1_nom} bezonas {t1_acc} por {verb} {task}. "
                f"Kiom da {tunit_pl} bezonatas por {w2_nom}?",
                f"se {w1_nom} finas {task} en {t1_acc}, kiom da {tunit_pl} "
                f"bezonatas por {w2_nom}?",
                f"{w1_nom} bezonas {t1_acc} por la tasko. Kiom por {w2_nom}?",
            ])
            Mul("w1", "t1", "const").apply(ctx)
            Div("const", "w2", "t2").apply(ctx)
            final = "t2"
        else:
            ctx.bind("t2", t2)
            q = frame + rng.choice([
                f"{w1_nom} bezonas {t1_acc} por {verb} {task}. "
                f"Kiom da {workers_pl} bezonatas por fini en {t2_acc}?",
                f"{w1_nom} finas {task} en {t1_acc}. "
                f"Kiom da {workers_pl} necesatas por fini samon en {t2_acc}?",
                f"la tasko {verb} {task} daŭras {t1_acc} kun {w1_nom}. "
                f"Kiom da {workers_pl} necesas por daŭri nur {t2_acc}?",
            ])
            Mul("w1", "t1", "const").apply(ctx)
            Div("const", "t2", "w2").apply(ctx)
            final = "w2"
        if not frame:
            q = q[0].upper() + q[1:]

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

def ratio_fraction_recipe(rng: random.Random, n_steps: int = 2,
                            reverse: bool = False) -> dict:
    """Ratio via fraction-of-total: r_i/(r_1+r_2) * total = part_i.
    Direct + larger + smaller asks. Uses Add + Frac.

    reverse=True: state the final part (or gift) value as given, ask for total.
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
        if not reverse:
            q = rng.choice([
                f"{names[0]} kaj {names[1]} dividas {qty_acc(total, obj)} laŭ la "
                f"rilatumo {a}:{b}. Kiu estas la {which}?",
                f"En rilatumo {a}:{b}, {names[0]} kaj {names[1]} dividas "
                f"{qty_acc(total, obj)}. Trovu la {which}n.",
                f"Ilia dividita nombro estas {render_qty(total, obj)}, "
                f"en rilatumo {a}:{b}. Kiu estas la {which}?",
            ])
        else:
            q = rng.choice([
                f"{names[0]} kaj {names[1]} dividas nekonatan nombron de {obj[1]} "
                f"laŭ la rilatumo {a}:{b}.",
                f"En rilatumo {a}:{b}, {names[0]} kaj {names[1]} dividas iom "
                f"da {obj[1]}.",
                f"Ilia dividita nombro estas ankoraŭ nekonata, "
                f"sed la rilatumo estas {a}:{b}.",
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
            if not reverse:
                q += f" Poste, {pct}% de tiu parto estas donacita. Kiom estas donacita?"
            Pct("pct", "part", "gift", style="direct").apply(ctx)
            final = "gift"

        if not reverse:
            return ctx.render(q, final)

        # Reverse: state the final (part or gift) value; ask for total.
        final_val = int(ctx.n(final))
        role_desc = "donacita" if final == "gift" else which
        if final == "gift":
            state = rng.choice([
                f" La donacita kvanto estas {render_qty(final_val, obj)} ({pct}% de la {which}).",
                f" El la {which}, {pct}% donacita estas {render_qty(final_val, obj)}.",
            ])
        else:
            state = rng.choice([
                f" La {which} estas {render_qty(final_val, obj)}.",
                f" {target_name} ricevas {qty_acc(final_val, obj)}.",
            ])
        q += state
        closer = rng.choice([
            f"Kiom da {obj[1]} estas entute?",
            f"Trovu la totalan nombron de {obj[1]}.",
            f"Kalkulu la totalan kvanton de {obj[1]}.",
        ])
        result = ctx.render_reverse(
            forward_prose=q,
            forward_final_var=final,
            ask_var="total",
            closer=closer,
        )
        result["recipe"] = "ratio_fraction_reverse"
        result["n_steps"] = n_steps
        result["direction"] = "reverse"
        return result

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

def distance_catchup_recipe(rng: random.Random, n_steps: int = 2,
                              reverse: bool = False) -> dict:
    """A leaves at ra km/h; h hours later B leaves at rb (rb > ra) and catches up.
    t = ra*h / (rb-ra).  Uses Mul + Sub + Div.

    reverse=True (n_steps=2 only): state t, ask for h (head-start hours).
    Path h → head_start → t is linear; ra reuse in the Sub doesn't
    interfere because the Sub isn't on the h path.
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

        if not reverse:
            q = (f"{names[0]} ekveturas per sia {vehicle[0]} je {ra} km/h. "
                 f"Post {h} horoj, {names[1]} ekiras de la sama loko "
                 f"en la sama direkto je {rb} km/h. "
                 f"Post kiom da horoj {names[1]} atingos {names[0]}?")
        else:
            q = (f"{names[0]} ekveturas per sia {vehicle[0]} je {ra} km/h. "
                 f"Post nekonata nombro de horoj, {names[1]} ekiras de la sama "
                 f"loko en la sama direkto je {rb} km/h.")

        # step 1: head start distance = ra * h
        Mul("ra", "h", "head_start").apply(ctx)
        # step 2: speed diff = rb - ra
        Sub("rb", "ra", "gap").apply(ctx)
        # step 3: time to catch up = head_start / gap
        Div("head_start", "gap", "t").apply(ctx)
        final = "t"

        # n_steps=4: also compute how far A had gone by catch-up
        if n_steps >= 4:
            if not reverse:
                q += f" Kiom da km {names[0]} veturis kiam {names[1]} atingis?"
            # A's total time = h + t; A's distance = ra * (h + t)
            Add("h", "t", "a_total_time").apply(ctx)
            Mul("ra", "a_total_time", "a_dist").apply(ctx)
            final = "a_dist"

        if not reverse:
            return ctx.render(q, final)

        # Reverse: given t (n_steps=2) or a_dist (n_steps=4), find h.
        # We only support n_steps=2 for reverse (n_steps=4 has an Add step
        # that reuses h — becomes a branching chain).
        if n_steps >= 4:
            raise RuntimeError("distance_catchup: reverse not supported for n_steps=4")

        final_val = int(ctx.n(final))
        q += f" {names[1]} atingas {names[0]} post {final_val} horoj."
        closer = f" Kalkulu, post kiom da horoj {names[1]} ekiris post {names[0]}."
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
    """Two vehicles start at opposite ends of distance D, moving toward each other.
    t = D / (r1 + r2).  Uses Add + Div.

    reverse=True (n_steps=2 only): state t, ask for D. Uses render_reverse
    (chain path: d → t is linear, r_sum is a side input).
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

        if not reverse:
            q = (f"{names[0]} kaj {names[1]} ekiras samtempe de du urboj "
                 f"distancaj je {d} km, veturante unu al la alia. "
                 f"{names[0]} veturas je {r1} km/h, {names[1]} je {r2} km/h. "
                 f"Post kiom da horoj ili renkontiĝos?")
        else:
            q = (f"{names[0]} kaj {names[1]} ekiras samtempe de du urboj kun "
                 f"nekonata distanco inter ili, veturante unu al la alia. "
                 f"{names[0]} veturas je {r1} km/h, {names[1]} je {r2} km/h.")

        Add("r1", "r2", "r_sum").apply(ctx)
        Div("d", "r_sum", "t").apply(ctx)
        final = "t"

        # n_steps=3: also ask how far each traveled — Mul each speed by t
        if n_steps >= 3:
            if not reverse:
                q += f" Kiom da km {names[0]} veturis kiam ili renkontiĝas?"
            Mul("r1", "t", "d1").apply(ctx)
            final = "d1"

        if not reverse:
            return ctx.render(q, final)

        # Reverse: state t (or d1 for n_steps=3) as given, ask for d
        final_val = int(ctx.n(final))
        if n_steps >= 3:
            q += f" {names[0]} veturis {final_val} km ĝis ili renkontiĝis."
        else:
            q += f" Ili renkontiĝas post {final_val} horoj."
        closer = " Kalkulu la originalan distancon inter la du urboj."
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
    """Round-trip: out at rout, back at rback, same distance.
    Avg speed = 2*rout*rback / (rout + rback).  Uses Mul + Add + Div.
    """
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
        q = frame + rng.choice([
            f"{name} veturas per sia {vehicle[0]} de urbo A al urbo B je "
            f"{rout} km/h, kaj revenas je {rback} km/h. "
            f"Kiu estas la meza rapideco por la tuta rondiro?",
            f"per sia {vehicle[0]}, {name} iras je {rout} km/h kaj revenas je {rback} km/h. "
            f"Kalkulu la mezan rapidecon de la rondiro.",
            f"{name} rondiras: unue je {rout} km/h, poste reveno je {rback} km/h. "
            f"Kiu estas la meza rapideco?",
            f"la {vehicle[0]} de {name} iras je {rout} km/h kaj revenas je {rback} km/h. "
            f"Trovu la mezan rapidecon.",
        ])
        if not frame:
            q = q[0].upper() + q[1:]

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
    count = rng.choice([3, 4, 5, 6, 7, 8])
    ask_choices = ["smallest", "largest"] if count % 2 == 0 else ["smallest", "largest", "middle"]
    ask = rng.choice(ask_choices)
    start = rng.randint(2, 120)
    values = [start + i for i in range(count)]
    total = sum(values)
    const = sum(range(count))  # 0+1+2+...+(N-1)

    ctx = Ctx.new(rng)
    ctx.bind("n", count)
    ctx.bind("const", const)
    ctx.bind("total", total)

    lhs_terms = " + ".join(["x"] + [f"(x + {i})" for i in range(1, count)])
    what = {"smallest": "plej malgranda", "largest": "plej granda", "middle": "meza"}[ask]
    frame = maybe_frame(rng)
    q = frame + rng.choice([
        f"la sumo de {count} sinsekvaj entjeroj estas {total}. Kiu estas la {what}?",
        f"{count} sinsekvaj entjeroj sumigas al {total}. Trovu la {what}n.",
        f"se {count} sinsekvaj entjeroj havas sumon de {total}, kiu estas la {what}?",
        f"estas {count} sinsekvaj entjeroj kies sumo egalas {total}. Kalkulu la {what}n.",
        f"trovu la {what}n el {count} sinsekvaj entjeroj kies sumo estas {total}.",
        f"kiam {count} sinsekvaj entjeroj sumiĝas al {total}, kiu estas la {what}?",
        f"kiu el {count} sinsekvaj entjeroj estas la {what}, se ilia sumo estas {total}?",
    ])
    if not frame:
        q = q[0].upper() + q[1:]

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
                    help="Fraction (0-1) of generated rows that use reverse-"
                         "direction mode. Only applies to recipes that support "
                         "reverse (ratio_parts, fraction_cascade, ratio_fraction).")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.count > 0:
        for _ in range(args.count):
            recipe_name = args.recipe or rng.choice(list(RECIPES.keys()))
            recipe = RECIPES[recipe_name]
            # 2/3 base + occasional 4/5 extensions — bias toward 2/3 so overall
            # distribution matches natural GSM8K density (mostly 2-3-step chains).
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
