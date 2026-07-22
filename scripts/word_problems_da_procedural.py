"""Danish procedural word problems (all 8 EO types).

Math by construction; language by annotated lexicon (no morphology rules).
Output schema matches word_problems_procedural.py so downstream code is
interchangeable.

Design decisions:
  * Nouns carry their full paradigm inline — Danish's irregular gender +
    definite formation makes rule-based morphology brittle. Lookup wins.
  * Commodities for percent problems ship as (indef, def_sg) pairs so we
    don't need to derive "cyklen" from "cykel" (syncope) programmatically.
  * Adding a type = one sampler function + prose templates, ~80-100 lines.
  * Prose templates are chosen at random per row → surface-form diversity.
  * Dedup via (question-prefix, answer) key so we don't emit exact repeats.

Usage:
    uv run python scripts/word_problems_da_procedural.py \\
        --type ratio --n 200 --out data/word_problems_da/ratio.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

# ── Danish lexicon ────────────────────────────────────────────────────────

# Countable nouns for ratio problems.
# (indef_sg, gender, def_sg, indef_pl, def_pl)
NOUNS = [
    ("æble",       "et", "æblet",       "æbler",       "æblerne"),
    ("banan",      "en", "bananen",     "bananer",     "bananerne"),
    ("pære",       "en", "pæren",       "pærer",       "pærerne"),
    ("bog",        "en", "bogen",       "bøger",       "bøgerne"),
    ("kage",       "en", "kagen",       "kager",       "kagerne"),
    ("blomst",     "en", "blomsten",    "blomster",    "blomsterne"),
    ("bold",       "en", "bolden",      "bolde",       "boldene"),
    ("kort",       "et", "kortet",      "kort",        "kortene"),
    ("mønt",       "en", "mønten",      "mønter",      "mønterne"),
    ("frimærke",   "et", "frimærket",   "frimærker",   "frimærkerne"),
    ("stol",       "en", "stolen",      "stole",       "stolene"),
    ("bord",       "et", "bordet",      "borde",       "bordene"),
    ("plante",     "en", "planten",     "planter",     "planterne"),
    ("bolsje",     "et", "bolsjet",     "bolsjer",     "bolsjerne"),
    ("nød",        "en", "nødden",      "nødder",      "nødderne"),
    ("agurk",      "en", "agurken",     "agurker",     "agurkerne"),
    ("tomat",      "en", "tomaten",     "tomater",     "tomaterne"),
    ("kylling",    "en", "kyllingen",   "kyllinger",   "kyllingerne"),
    ("perle",      "en", "perlen",      "perler",      "perlerne"),
    ("terning",    "en", "terningen",   "terninger",   "terningerne"),
    # Expanded pool (household + food + school + toys)
    ("kartoffel",  "en", "kartoflen",   "kartofler",   "kartoflerne"),
    ("gulerod",    "en", "guleroden",   "gulerødder",  "gulerødderne"),
    ("appelsin",   "en", "appelsinen",  "appelsiner",  "appelsinerne"),
    ("citron",     "en", "citronen",    "citroner",    "citronerne"),
    ("kop",        "en", "koppen",      "kopper",      "kopperne"),
    ("tallerken",  "en", "tallerkenen", "tallerkener", "tallerkenerne"),
    ("skål",       "en", "skålen",      "skåle",       "skålene"),
    ("gaffel",     "en", "gaflen",      "gafler",      "gaflerne"),
    ("kniv",       "en", "kniven",      "knive",       "knivene"),
    ("ske",        "en", "skeen",       "skeer",       "skeerne"),
    ("glas",       "et", "glasset",     "glas",        "glassene"),
    ("pen",        "en", "pennen",      "penne",       "pennene"),
    ("blyant",     "en", "blyanten",    "blyanter",    "blyanterne"),
    ("lineal",     "en", "linealen",    "linealer",    "linealerne"),
    ("hæfte",      "et", "hæftet",      "hæfter",      "hæfterne"),
    ("tegning",    "en", "tegningen",   "tegninger",   "tegningerne"),
    ("ballon",     "en", "ballonen",    "balloner",    "ballonerne"),
    ("klods",      "en", "klodsen",     "klodser",     "klodserne"),
    ("sten",       "en", "stenen",      "sten",        "stenene"),
    ("musling",    "en", "muslingen",   "muslinger",   "muslingerne"),
    ("fugl",       "en", "fuglen",      "fugle",       "fuglene"),
    ("kanin",      "en", "kaninen",     "kaniner",     "kaninerne"),
    ("kat",        "en", "katten",      "katte",       "kattene"),
    ("hund",       "en", "hunden",      "hunde",       "hundene"),
    ("fisk",       "en", "fisken",      "fisk",        "fiskene"),
    ("lys",        "et", "lyset",       "lys",         "lysene"),
    ("ring",       "en", "ringen",      "ringe",       "ringene"),
    ("brik",       "en", "brikken",     "brikker",     "brikkerne"),
    ("kugle",      "en", "kuglen",      "kugler",      "kuglerne"),
    ("nøgle",      "en", "nøglen",      "nøgler",      "nøglerne"),
    ("pose",       "en", "posen",       "poser",       "poserne"),
    ("æg",         "et", "ægget",       "æg",          "æggene"),
    ("brød",       "et", "brødet",      "brød",        "brødene"),
    ("bolle",      "en", "bollen",      "boller",      "bollerne"),
]

# Commodities for percent problems. (indef, def) pairs — no morphology rules,
# just lookup. Def form is used when we refer back ("Hvad koster den nu?").
COMMODITIES = [
    ("en bog",          "bogen"),
    ("en cykel",        "cyklen"),
    ("en telefon",      "telefonen"),
    ("en jakke",        "jakken"),
    ("en billet",       "billetten"),
    ("en middag",       "middagen"),
    ("en kop kaffe",    "kaffen"),
    ("en kjole",        "kjolen"),
    ("en tablet",       "tabletten"),
    ("en computer",     "computeren"),
    ("en støvsuger",    "støvsugeren"),
    ("en sofa",         "sofaen"),
    ("et fjernsyn",     "fjernsynet"),
    ("et køleskab",     "køleskabet"),
    ("et abonnement",   "abonnementet"),
    ("et par bukser",   "bukserne"),
    ("et par sko",      "skoene"),
    ("en kaffemaskine", "kaffemaskinen"),
    ("en lampe",        "lampen"),
    ("en cykelhjelm",   "cykelhjelmen"),
    ("et par briller",  "brillerne"),
    ("en trøje",        "trøjen"),
    ("en skjorte",      "skjorten"),
    ("et par handsker", "handskerne"),
    ("et halstørklæde", "halstørklædet"),
    ("en hat",          "hatten"),
    ("et par støvler",  "støvlerne"),
    ("en pengepung",    "pengepungen"),
    ("en taske",        "tasken"),
    ("en rygsæk",       "rygsækken"),
    ("et kamera",       "kameraet"),
    ("et smykke",       "smykket"),
    ("en cykellygte",   "cykellygten"),
    ("en cykellås",     "cykellåsen"),
    ("et skateboard",   "skateboardet"),
    ("en mikroovn",     "mikroovnen"),
    ("en brødrister",   "brødristeren"),
    ("en radio",        "radioen"),
    ("et videospil",    "videospillet"),
    ("en pude",         "puden"),
    ("et tæppe",        "tæppet"),
    ("en spejlrefleks", "spejlrefleksen"),
    ("en el-scooter",   "el-scooteren"),
    ("en gasgrill",     "gasgrillen"),
    ("en havemøbelsæt", "havemøbelsættet"),
    ("en løbehjul",     "løbehjulet"),
    ("en musikafspiller","musikafspilleren"),
    ("en spillekonsol", "spillekonsollen"),
]

NAMES = [
    "Anders", "Anne", "Bo", "Britta", "Christian", "Cecilie",
    "Daniel", "Ditte", "Emil", "Emma", "Frederik", "Freja",
    "Gustav", "Grethe", "Henrik", "Helle", "Ivan", "Ida",
    "Jens", "Julie", "Kasper", "Karina", "Lars", "Louise",
    "Mads", "Mette", "Niels", "Nina", "Oliver", "Olivia",
    "Peter", "Pernille", "Rasmus", "Rikke", "Søren", "Sofie",
]

# Danish possessive: names ending in s/x/z take just an apostrophe (Jens' hus),
# not an -s (never "Jenss"). We rewrite `<name>s` → `<name>'` for known names.
_S_ENDING_NAMES = tuple(n for n in NAMES if n[-1].lower() in "sxz")


def fix_possessives(text: str) -> str:
    for n in _S_ENDING_NAMES:
        text = text.replace(f"{n}s", f"{n}'")
    return text


@dataclass
class Step:
    """One arithmetic step in a chain of reasoning.

    `expr` is a sympy-evaluable arithmetic string (e.g. "36 / 4"). `result`
    is the ground-truth value. `pre`/`post` are natural-language narration
    with the constraint that `pre` MUST NOT mention `result` (so the same
    step data can be rendered as a tool-call: the assistant emits `pre` +
    `expr`, the tool returns `result`, the assistant then emits `post`).
    """
    pre: str = ""
    expr: str = ""
    result: str = ""
    post: str = ""


def render_prose(steps: list[Step], final: str) -> str:
    """Inline prose: `{pre} {expr} = {result}. {post}` per step + `#### N`."""
    lines = []
    for s in steps:
        parts = []
        if s.pre:
            parts.append(s.pre)
        if s.expr:
            parts.append(f"{s.expr} = {s.result}.")
        if s.post:
            parts.append(s.post)
        line = " ".join(p.strip() for p in parts if p).strip()
        if line:
            lines.append(line)
    lines.append(f"#### {final}")
    return "\n".join(lines)


def render_funcall(question: str, steps: list[Step], final: str) -> list[dict]:
    """Multi-turn messages: system+user pose the problem; each Step becomes
    an assistant turn that emits `pre` + a tool_call, followed by a tool
    response with `result`, and the assistant continues with `post`. The
    final assistant turn concludes with the answer."""
    messages: list[dict] = [
        {"role": "user", "content": f"Spørgsmål: {question}"},
    ]
    for s in steps:
        assistant_pre = s.pre.strip()
        if assistant_pre:
            messages.append({"role": "assistant", "content": assistant_pre})
        messages.append({
            "role": "assistant",
            "tool_calls": [{"type": "calculator", "expr": s.expr}],
        })
        messages.append({
            "role": "tool", "name": "calculator", "content": s.result,
        })
        if s.post.strip():
            messages.append({"role": "assistant", "content": s.post.strip()})
    messages.append({"role": "assistant", "content": f"#### {final}"})
    return messages


@dataclass
class Problem:
    type: str
    question_da: str
    chain_da: str
    answer: str
    params: dict
    strategy: str
    # Compositional artifacts (may be empty for types not yet refactored)
    steps: list[dict] | None = None  # serialized [{pre,expr,result,post}]
    funcall: list[dict] | None = None


# ── RATIO type ──────────────────────────────────────────────────────────

# Preambles reused across all 5 ask kinds. `{sharers}` is a Danish name-list
# like "Anna og Bo" (2-way) or "Anna, Bo og Cecilie" (3-way). `{ratio}` is
# "3:5" or "1:2:3". The question tail depends on the ask kind.
_RATIO_PREAMBLES = [
    "{sharers} deler {total} {obj_pl} i forholdet {ratio}.",
    "Der er {total} {obj_pl}, som skal fordeles mellem {sharers} "
    "i forholdet {ratio}.",
    "{sharers} har tilsammen {total} {obj_pl}. De deler dem i forholdet {ratio}.",
    "En kasse med {total} {obj_pl} deles i forholdet {ratio} mellem {sharers}.",
    "Ved en fest deler {sharers} {total} {obj_pl} i forholdet {ratio}.",
    "På en skoleudflugt får {sharers} udleveret i alt {total} {obj_pl} "
    "i forholdet {ratio}.",
    "Efter et arrangement skal {sharers} dele {total} {obj_pl} mellem sig "
    "efter forholdet {ratio}.",
    "En pose indeholder {total} {obj_pl}, som {sharers} deler i forholdet "
    "{ratio}.",
    "Til en fælles fejring har {sharers} indkøbt {total} {obj_pl}. De deler "
    "dem i forholdet {ratio}.",
    "På markedet køber {sharers} sammen {total} {obj_pl} og deler dem "
    "i forholdet {ratio}.",
    "Efter en høst deler {sharers} {total} {obj_pl} i forholdet {ratio}.",
    "Et arvedelt bo omfatter {total} {obj_pl}, som fordeles mellem {sharers} "
    "i forholdet {ratio}.",
    "På en workshop deler {sharers} {total} {obj_pl} i forholdet {ratio}.",
    "{sharers} har vundet {total} {obj_pl} i en konkurrence. De aftaler at "
    "dele dem i forholdet {ratio}.",
]

_RATIO_TAILS = {
    "direct":    ["Hvor mange {obj_pl} får {who}?",
                  "Hvor mange får {who}?",
                  "Hvor mange {obj_pl} tilfalder {who}?"],
    "larger":    ["Hvor mange {obj_pl} får den, der får mest?",
                  "Hvad er den største andel i {obj_pl}?",
                  "Hvor mange {obj_pl} udgør den største andel?"],
    "smaller":   ["Hvor mange {obj_pl} får den, der får mindst?",
                  "Hvad er den mindste andel i {obj_pl}?",
                  "Hvor mange {obj_pl} udgør den mindste andel?"],
    "diff":      ["Hvor mange flere {obj_pl} får den ene end den anden?",
                  "Hvor stor er forskellen mellem den største og den "
                  "mindste andel (i {obj_pl})?",
                  "Hvor mange flere {obj_pl} har den, der får mest, "
                  "sammenlignet med den, der får mindst?"],
}

# given-one changes the givens: total is not stated, one person's share is.
_RATIO_GIVEN_ONE_TEMPLATES = [
    "{sharers} deler et antal {obj_pl} i forholdet {ratio}. {given_name} "
    "får {given_val} {obj_pl}. Hvor mange {obj_pl} deler de i alt?",
    "Der fordeles nogle {obj_pl} mellem {sharers} i forholdet {ratio}. "
    "{given_name} får {given_val} {obj_pl}. Hvor mange {obj_pl} er der i alt?",
    "{sharers} har delt et parti {obj_pl} i forholdet {ratio}. Vi ved, at "
    "{given_name} fik {given_val} {obj_pl}. Hvor mange {obj_pl} var der "
    "i alt at dele?",
    "Ved en deling af {obj_pl} i forholdet {ratio} mellem {sharers} fik "
    "{given_name} {given_val} {obj_pl}. Hvor mange {obj_pl} er der i alt?",
    "{sharers} fordeler et ukendt antal {obj_pl} i forholdet {ratio}. "
    "{given_name} får {given_val} {obj_pl}. Hvor mange {obj_pl} var der "
    "at dele?",
]


def _da_names_list(names: list[str]) -> str:
    """'Anna og Bo' or 'Anna, Bo og Cecilie'."""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} og {names[1]}"
    return ", ".join(names[:-1]) + f" og {names[-1]}"

@dataclass
class RatioParams:
    ratios: tuple           # e.g. (2, 3) or (1, 2, 3)
    names: tuple            # tuple of names, same length as ratios
    total: int              # total distributed (given except for given-one)
    per_part: int           # value of one ratio unit
    ask: str                # "direct" | "larger" | "smaller" | "diff" | "given-one"
    ask_idx: int            # for direct: which index; for given-one: which index we know
    given_val: int          # for given-one: parts[ask_idx]. else 0.
    obj_pl: str
    answer: int

    @property
    def parts(self):
        return [r * self.per_part for r in self.ratios]

    @property
    def sum_r(self):
        return sum(self.ratios)

    @property
    def n(self):
        return len(self.ratios)


def _ratio_num_word(n: int) -> str:
    return "1 del" if n == 1 else f"{n} dele"


def _ratio_steps_parts(p: RatioParams) -> tuple[list[Step], str]:
    """Sum of ratios → per part → dispatch to ask-specific final step."""
    ratios_str = " + ".join(str(r) for r in p.ratios)
    steps = [
        Step(pre="Summen af forholdstallene er",
             expr=ratios_str, result=str(p.sum_r),
             post=f"så vi har {p.sum_r} lige store dele."),
    ]
    if p.ask == "given-one":
        # We know one person's share and need the total. Skip the per-part step
        # since we can compute it directly from given_val.
        given_r = p.ratios[p.ask_idx]
        steps.append(Step(
            pre=f"{p.names[p.ask_idx]} får {given_r} af de {p.sum_r} dele, "
                f"altså er én del {p.given_val} divideret med {given_r}:",
            expr=f"{p.given_val} / {given_r}", result=str(p.per_part),
            post=f"altså {p.per_part} {p.obj_pl} per del."))
        steps.append(Step(
            pre="Totalen er én del gange antal dele:",
            expr=f"{p.per_part} * {p.sum_r}", result=str(p.answer),
            post=f"i alt {p.answer} {p.obj_pl}."))
        return steps, str(p.answer)

    steps.append(Step(
        pre="En del svarer til totalen divideret med antal dele:",
        expr=f"{p.total} / {p.sum_r}", result=str(p.per_part),
        post=f"altså {p.per_part} {p.obj_pl} per del."))

    if p.ask == "direct":
        r = p.ratios[p.ask_idx]
        steps.append(Step(
            pre=f"{p.names[p.ask_idx]} får {_ratio_num_word(r)}, altså",
            expr=f"{p.per_part} * {r}", result=str(p.answer),
            post=f"i alt {p.answer} {p.obj_pl}."))
    elif p.ask == "larger":
        rmax = max(p.ratios)
        steps.append(Step(
            pre=f"Den største andel er {rmax} af {p.sum_r} dele, altså",
            expr=f"{p.per_part} * {rmax}", result=str(p.answer),
            post=f"i alt {p.answer} {p.obj_pl}."))
    elif p.ask == "smaller":
        rmin = min(p.ratios)
        steps.append(Step(
            pre=f"Den mindste andel er {rmin} af {p.sum_r} dele, altså",
            expr=f"{p.per_part} * {rmin}", result=str(p.answer),
            post=f"i alt {p.answer} {p.obj_pl}."))
    elif p.ask == "diff":
        rmax, rmin = max(p.ratios), min(p.ratios)
        diff_r = rmax - rmin
        steps.append(Step(
            pre=f"Forskellen mellem den største og den mindste andel er "
                f"({rmax} - {rmin}) dele, altså",
            expr=f"{p.per_part} * ({rmax} - {rmin})", result=str(p.answer),
            post=f"altså {p.answer} {p.obj_pl}."))
    else:
        raise ValueError(p.ask)
    return steps, str(p.answer)


def _ratio_steps_fraction(p: RatioParams) -> tuple[list[Step], str]:
    """Direct fraction approach; for given-one we cannot use fraction (total unknown),
    fall back to parts."""
    if p.ask == "given-one":
        return _ratio_steps_parts(p)
    if p.ask == "direct":
        r = p.ratios[p.ask_idx]
        who = p.names[p.ask_idx]
        return [
            Step(pre=f"Da forholdet er {':'.join(map(str, p.ratios))}, får "
                     f"{who} en brøkdel af totalen på {r}/{p.sum_r}. "
                     f"Vi udregner:",
                 expr=f"{p.total} * {r} / {p.sum_r}", result=str(p.answer),
                 post=f"altså {p.answer} {p.obj_pl}."),
        ], str(p.answer)
    if p.ask == "larger":
        rmax = max(p.ratios)
        return [
            Step(pre=f"Den største andel udgør brøkdelen {rmax}/{p.sum_r} "
                     f"af totalen. Vi udregner:",
                 expr=f"{p.total} * {rmax} / {p.sum_r}",
                 result=str(p.answer),
                 post=f"altså {p.answer} {p.obj_pl}."),
        ], str(p.answer)
    if p.ask == "smaller":
        rmin = min(p.ratios)
        return [
            Step(pre=f"Den mindste andel udgør brøkdelen {rmin}/{p.sum_r} "
                     f"af totalen. Vi udregner:",
                 expr=f"{p.total} * {rmin} / {p.sum_r}",
                 result=str(p.answer),
                 post=f"altså {p.answer} {p.obj_pl}."),
        ], str(p.answer)
    if p.ask == "diff":
        rmax, rmin = max(p.ratios), min(p.ratios)
        return [
            Step(pre=f"Forskellen udgør brøkdelen ({rmax} - {rmin})/{p.sum_r} "
                     f"af totalen. Vi udregner:",
                 expr=f"{p.total} * ({rmax} - {rmin}) / {p.sum_r}",
                 result=str(p.answer),
                 post=f"altså {p.answer} {p.obj_pl}."),
        ], str(p.answer)
    raise ValueError(p.ask)


def _ratio_steps_algebra(p: RatioParams) -> tuple[list[Step], str]:
    """Algebra: let x be one part; solve sum_r * x = total (or use given_val)."""
    if p.ask == "given-one":
        given_r = p.ratios[p.ask_idx]
        who = p.names[p.ask_idx]
        return [
            Step(pre=f"Lad x være værdien af én del. Så får {who} = {given_r}x, "
                     f"og vi ved at {given_r}x = {p.given_val}. Vi løser for x:",
                 expr=f"{p.given_val} / {given_r}", result=str(p.per_part),
                 post=f"altså x = {p.per_part}."),
            Step(pre=f"Totalen er summen af alle andele = {p.sum_r}x:",
                 expr=f"{p.sum_r} * {p.per_part}", result=str(p.answer),
                 post=f"altså {p.answer} {p.obj_pl} i alt."),
        ], str(p.answer)

    coefs = " + ".join(f"{r}x" for r in p.ratios)
    steps = [
        Step(pre=f"Lad x være værdien af én del. Så er totalen "
                 f"{coefs} = {p.sum_r}x = {p.total}. Vi løser for x:",
             expr=f"{p.total} / {p.sum_r}", result=str(p.per_part),
             post=f"altså x = {p.per_part}."),
    ]
    if p.ask == "direct":
        r = p.ratios[p.ask_idx]
        who = p.names[p.ask_idx]
        steps.append(Step(pre=f"{who}s andel er {r}x:",
                          expr=f"{r} * {p.per_part}", result=str(p.answer),
                          post=f"i alt {p.answer} {p.obj_pl}."))
    elif p.ask == "larger":
        rmax = max(p.ratios)
        steps.append(Step(pre=f"Den største andel er {rmax}x:",
                          expr=f"{rmax} * {p.per_part}", result=str(p.answer),
                          post=f"i alt {p.answer} {p.obj_pl}."))
    elif p.ask == "smaller":
        rmin = min(p.ratios)
        steps.append(Step(pre=f"Den mindste andel er {rmin}x:",
                          expr=f"{rmin} * {p.per_part}", result=str(p.answer),
                          post=f"i alt {p.answer} {p.obj_pl}."))
    elif p.ask == "diff":
        rmax, rmin = max(p.ratios), min(p.ratios)
        steps.append(Step(pre=f"Forskellen mellem største og mindste andel "
                              f"er ({rmax} - {rmin})x:",
                          expr=f"({rmax} - {rmin}) * {p.per_part}",
                          result=str(p.answer),
                          post=f"altså {p.answer} {p.obj_pl}."))
    return steps, str(p.answer)


_RATIO_STEPS = {
    "parts": _ratio_steps_parts,
    "fraction": _ratio_steps_fraction,
    "algebra": _ratio_steps_algebra,
}


_RATIOS_2 = [(a, b) for a in range(1, 8) for b in range(1, 8) if a != b]
_RATIOS_3 = [(1, 2, 3), (2, 3, 5), (1, 1, 2), (1, 3, 4), (2, 3, 4), (1, 2, 4)]


def sample_ratio(rng: random.Random) -> Problem:
    obj = rng.choice(NOUNS)
    # 2-way biased over 3-way (matches EO taste of asymmetry).
    ratios = rng.choice(_RATIOS_2 * 3 + _RATIOS_3)
    n = len(ratios)
    names = tuple(rng.sample(NAMES, n))
    per_part = rng.randint(2, 50)
    total = per_part * sum(ratios)

    ask_choices = ["direct", "larger", "smaller", "given-one"]
    if n == 2:
        ask_choices.append("diff")
    ask = rng.choice(ask_choices)

    parts = [r * per_part for r in ratios]
    ask_idx = rng.randint(0, n - 1)

    if ask == "direct":
        answer = parts[ask_idx]
        given_val = 0
    elif ask == "larger":
        answer = max(parts)
        given_val = 0
    elif ask == "smaller":
        answer = min(parts)
        given_val = 0
    elif ask == "diff":
        answer = max(parts) - min(parts)
        given_val = 0
    elif ask == "given-one":
        answer = total
        given_val = parts[ask_idx]
    else:
        raise ValueError(ask)

    p = RatioParams(ratios=ratios, names=names, total=total, per_part=per_part,
                    ask=ask, ask_idx=ask_idx, given_val=given_val,
                    obj_pl=obj[3], answer=answer)

    sharers = _da_names_list(list(names))
    ratio_str = ":".join(str(r) for r in ratios)
    if ask == "given-one":
        q_tpl = rng.choice(_RATIO_GIVEN_ONE_TEMPLATES)
        question = q_tpl.format(sharers=sharers, obj_pl=obj[3],
                                ratio=ratio_str, given_name=names[ask_idx],
                                given_val=given_val)
    else:
        preamble = rng.choice(_RATIO_PREAMBLES)
        tail = rng.choice(_RATIO_TAILS[ask])
        who = names[ask_idx] if ask == "direct" else ""
        question = (preamble + " " + tail).format(
            sharers=sharers, total=total, obj_pl=obj[3],
            ratio=ratio_str, who=who,
        )

    strategy = rng.choice(list(_RATIO_STEPS))
    steps, final = _RATIO_STEPS[strategy](p)
    chain = render_prose(steps, final)
    funcall = render_funcall(question, steps, final)

    return Problem(
        type="ratio",
        question_da=question,
        chain_da=chain,
        answer=final,
        params={
            "ratios": list(ratios), "names": list(names),
            "total": total, "per_part": per_part,
            "ask": ask, "ask_idx": ask_idx, "given_val": given_val,
            "object": obj[0],
        },
        strategy=f"{strategy}_{ask}_n{n}",
        steps=[asdict(s) for s in steps],
        funcall=funcall,
    )


# ── PERCENT type ────────────────────────────────────────────────────────

PERCENT_QUESTION_TEMPLATES = {
    "discount": [
        "{item_indef_cap} koster normalt {price} kr. I dag er der {pct}% "
        "rabat. Hvad koster {item_def} nu?",
        "{name} vil købe {item_indef}. Prisen er {price} kr, men der er "
        "{pct}% rabat. Hvor meget skal {name} betale?",
        "{item_indef_cap} sælges med {pct}% rabat. Den normale pris er "
        "{price} kr. Hvad er tilbudsprisen?",
        "I forbindelse med januarudsalget nedsættes {item_indef} fra {price} "
        "kr med {pct}%. Hvad er den nye pris?",
        "{name} finder {item_indef} til {price} kr, og der er {pct}% rabat "
        "i kassen. Hvad kommer den til at koste?",
        "En butik giver {pct}% rabat på {item_indef} til vejledende pris "
        "{price} kr. Hvad koster {item_def}?",
        "{item_indef_cap} nedsættes med {pct}% i sommerudsalget. "
        "Den oprindelige pris var {price} kr. Hvad er den nye pris?",
        "Til Black Friday sælges {item_indef} med {pct}% rabat fra normalprisen "
        "{price} kr. Hvad koster {item_def} nu?",
        "{name} bruger en rabatkupon på {pct}% ved køb af {item_indef} til "
        "{price} kr. Hvad ender {name} med at betale?",
        "En medlemsrabat på {pct}% bringer prisen på {item_indef} ned fra "
        "{price} kr. Hvad koster {item_def} for medlemmet?",
    ],
    "markup": [
        "{item_indef_cap} har en indkøbspris på {price} kr. Butikken lægger "
        "{pct}% oveni. Hvad er salgsprisen?",
        "{name} vil videresælge {item_indef} med {pct}% fortjeneste. "
        "Indkøbsprisen var {price} kr. Hvad er salgsprisen?",
        "En importør køber {item_indef} for {price} kr og pålægger {pct}% "
        "avance. Hvad bliver udsalgsprisen?",
        "{name} sælger {item_indef} med {pct}% avance. Indkøbsprisen var "
        "{price} kr. Hvad koster {item_def} i {name}s butik?",
        "En grossist køber {item_indef} for {price} kr og videresælger den "
        "med {pct}% fortjeneste. Hvad er videresalgsprisen?",
        "{name} har hjemtaget {item_indef} for {price} kr og lægger {pct}% "
        "oveni til dækning af omkostninger. Hvad er den nye pris?",
        "Efter {pct}% avance på indkøbsprisen {price} kr sælges "
        "{item_indef} i butikken. Hvad er butikkens pris?",
    ],
    "tax": [
        "{item_indef_cap} koster {price} kr før moms. Der lægges {pct}% moms "
        "oveni. Hvad er den samlede pris?",
        "{name} køber {item_indef} til {price} kr eksklusive {pct}% moms. "
        "Hvad er den samlede pris?",
        "På en faktura står {item_indef} til {price} kr eksklusive moms. "
        "Momssatsen er {pct}%. Hvad er beløbet inklusive moms?",
        "{item_indef_cap} sælges business-to-business til {price} kr plus "
        "{pct}% moms. Hvad er den samlede pris?",
        "Efter tillæg af {pct}% moms på nettoprisen {price} kr, "
        "hvad koster {item_def} inklusive moms?",
        "Prisen på {item_indef} er {price} kr eksklusive moms på {pct}%. "
        "Hvad skal kunden betale?",
    ],
    "of_amount": [
        "Hvad er {pct}% af {price} kr?",
        "Beregn {pct}% af {price} kr.",
        "{name} skal udregne {pct}% af {price} kr. Hvad er resultatet?",
        "{pct}% af et beløb på {price} kr — hvor mange kroner er det?",
        "En andel på {pct}% af {price} kr — hvad svarer det til i kroner?",
        "Hvor stort er {pct}% af {price} kr?",
        "En fond bevilger {pct}% af en samlet ramme på {price} kr. "
        "Hvor mange kroner er det?",
    ],
    "saving": [
        "{item_indef_cap} nedsættes med {pct}% fra {price} kr. Hvor mange "
        "kroner sparer {name}?",
        "Ved et udsalg gives {pct}% rabat på {item_indef} til {price} kr. "
        "Hvor mange kroner spares?",
        "{name} køber {item_indef} med {pct}% rabat fra normalprisen "
        "{price} kr. Hvor stor er besparelsen i kroner?",
        "En medlemsrabat på {pct}% på {item_indef} til {price} kr. Hvor meget "
        "sparer {name} i kroner?",
        "Under sommerudsalget gives {pct}% rabat på {item_indef} til {price} kr. "
        "Hvor mange kroner spares?",
        "{name} bruger en kupon på {pct}% ved køb af {item_indef} til "
        "{price} kr. Hvor mange kroner spares på købet?",
    ],
}

@dataclass
class PercentParams:
    kind: str          # "discount" | "markup" | "tax" | "of_amount" | "saving"
    price: int
    pct: int
    change: int
    answer: int


def _percent_steps_direct(p: PercentParams) -> tuple[list[Step], str]:
    """Two-step: change = price*pct/100, then answer = price ± change.

    Also handles of-amount and saving: those return the change amount itself,
    so we stop after step 1 with the noun matched to the op.
    """
    if p.kind == "discount":
        change_noun = "Rabatten"
        op_word = "-"
        op_desc = "trukket fra"
    elif p.kind == "markup":
        change_noun = "Fortjenesten"
        op_word = "+"
        op_desc = "lagt til"
    elif p.kind == "tax":
        change_noun = "Momsen"
        op_word = "+"
        op_desc = "lagt til"
    elif p.kind == "of_amount":
        change_noun = "Beløbet"
        op_word = None
        op_desc = None
    elif p.kind == "saving":
        change_noun = "Besparelsen"
        op_word = None
        op_desc = None
    else:
        raise ValueError(p.kind)

    if op_word is None:
        # single step — the "change" is the answer
        return [
            Step(pre=f"{change_noun} udgør {p.pct}% af {p.price} kr:",
                 expr=f"{p.price} * {p.pct} / 100", result=str(p.change),
                 post=f"altså {p.change} kr."),
        ], str(p.answer)

    return [
        Step(pre=f"{change_noun} udgør {p.pct}% af {p.price} kr:",
             expr=f"{p.price} * {p.pct} / 100", result=str(p.change),
             post=f"altså {p.change} kr {op_desc}."),
        Step(pre="Den samlede pris bliver derfor",
             expr=f"{p.price} {op_word} {p.change}",
             result=str(p.answer),
             post=f"i alt {p.answer} kr."),
    ], str(p.answer)


def _percent_steps_compound(p: PercentParams) -> tuple[list[Step], str]:
    """One-step compound: answer = price * (100 ± pct) / 100.

    For of-amount / saving fall back to _percent_steps_direct (there is no
    natural compound form for those)."""
    if p.kind in ("of_amount", "saving"):
        return _percent_steps_direct(p)
    if p.kind == "discount":
        sign = "-"
        desc = f"Kunden skal betale (100 - {p.pct})% af {p.price} kr:"
    else:  # markup or tax
        sign = "+"
        desc = f"Slutbeløbet svarer til (100 + {p.pct})% af {p.price} kr:"
    steps = [
        Step(pre=desc,
             expr=f"{p.price} * (100 {sign} {p.pct}) / 100",
             result=str(p.answer),
             post=f"altså {p.answer} kr."),
    ]
    return steps, str(p.answer)


_PERCENT_STEPS = {
    "direct": _percent_steps_direct,
    "compound": _percent_steps_compound,
}


def sample_percent(rng: random.Random) -> Problem:
    kind = rng.choice(["discount", "markup", "tax", "of_amount", "saving"])
    while True:
        # Multiples of 10 kroner from 50 to 5000 — most divisibility hits fast.
        price = rng.randint(5, 500) * 10
        pct = rng.randint(2, 75)
        if price * pct % 100 == 0:
            break
    change = price * pct // 100
    if kind == "discount":
        answer = price - change
    elif kind in ("markup", "tax"):
        answer = price + change
    else:  # of_amount, saving
        answer = change

    item_indef, item_def = rng.choice(COMMODITIES)
    item_indef_cap = item_indef[0].upper() + item_indef[1:]
    name = rng.choice(NAMES)

    q_tpl = rng.choice(PERCENT_QUESTION_TEMPLATES[kind])
    question = q_tpl.format(
        item_indef=item_indef, item_indef_cap=item_indef_cap,
        item_def=item_def, price=price, pct=pct, name=name,
    )

    p = PercentParams(kind=kind, price=price, pct=pct,
                      change=change, answer=answer)
    strategy = rng.choice(list(_PERCENT_STEPS))
    steps, final = _PERCENT_STEPS[strategy](p)
    chain = render_prose(steps, final)
    funcall = render_funcall(question, steps, final)

    return Problem(
        type="percent",
        question_da=question,
        chain_da=chain,
        answer=final,
        params={"kind": kind, "price": price, "pct": pct,
                "change": change, "item": item_indef, "name": name},
        strategy=f"percent_{kind}_{strategy}",
        steps=[asdict(s) for s in steps],
        funcall=funcall,
    )


# ── INVERSE_RATE ────────────────────────────────────────────────────────

INVERSE_RATE_TEMPLATES = {
    # find-time: given n1,t1,n2 → find t2. answer = t2.
    "find_time": [
        "{n1} arbejdere kan udføre et stykke arbejde på {t1} timer. Hvor lang tid "
        "tager det, hvis {n2} arbejdere skal udføre samme arbejde?",
        "En opgave kan løses af {n1} personer på {t1} timer. Hvor lang tid tager "
        "det for {n2} personer at løse den samme opgave?",
        "Det tager {n1} malere {t1} dage at male et hus. Hvor mange dage tager "
        "det {n2} malere at male det samme hus?",
        "En gruppe på {n1} personer kan tømme en tank på {t1} minutter. Hvor "
        "mange minutter tager det {n2} personer?",
        "Hvis {n1} maskiner producerer en ordre på {t1} timer, hvor mange timer "
        "tager det så {n2} maskiner at producere den samme ordre?",
        "{n1} kokke kan tilberede en menu på {t1} timer. Hvor lang tid vil {n2} "
        "kokke bruge på den samme menu?",
        "{n1} landmænd kan høste en mark på {t1} dage. Hvor mange dage tager det "
        "{n2} landmænd at høste den samme mark?",
        "Et rengøringsfirma med {n1} medarbejdere rengør et hotel på {t1} timer. "
        "Hvor lang tid tager det med {n2} medarbejdere?",
        "{n1} bagere kan færdiggøre en ordre på {t1} timer. Hvor lang tid tager "
        "det, hvis {n2} bagere arbejder på samme ordre?",
        "En pumpe med {n1} indløb tømmer bassinet på {t1} minutter. Hvor mange "
        "minutter tager det med {n2} indløb?",
        "Med {n1} håndværkere kan et projekt afsluttes på {t1} dage. Hvor mange "
        "dage varer det med {n2} håndværkere?",
        "En avisrute betjenes af {n1} bude på {t1} timer. Hvor lang tid tager "
        "den samme rute for {n2} bude?",
        "Et budfirma med {n1} chauffører leverer alle pakker på {t1} timer. "
        "Hvor lang tid tager det med {n2} chauffører?",
        "{n1} musikere kan gennemføre et program på {t1} minutter. Hvis {n2} "
        "musikere skal spille det samme program, hvor lang tid tager det?",
        "Et bygefirma med {n1} arbejdere færdiggør et projekt på {t1} dage. "
        "Hvor mange dage tager samme projekt med {n2} arbejdere?",
        "En kontorafdeling med {n1} sagsbehandlere afvikler alle sager på {t1} "
        "timer. Hvor lang tid tager det med {n2} sagsbehandlere?",
        "Et kokketeam på {n1} personer laver en firmafrokost på {t1} timer. "
        "Hvor lang tid tager det for {n2} kokke?",
    ],
    # find-workers: given n1,t1,t2 → find n2 (workers needed).
    "find_workers": [
        "{n1} arbejdere kan udføre et stykke arbejde på {t1} timer. Hvor mange "
        "arbejdere skal der til for at gøre det på {t2} timer?",
        "En opgave kan løses af {n1} personer på {t1} timer. Hvor mange personer "
        "skal der til, hvis opgaven skal være færdig efter {t2} timer?",
        "Det tager {n1} malere {t1} dage at male et hus. Hvor mange malere skal "
        "der til for at male huset på {t2} dage?",
        "Hvis {n1} maskiner producerer en ordre på {t1} timer, hvor mange "
        "maskiner er nødvendige for at gøre det på {t2} timer?",
        "{n1} kokke kan tilberede en menu på {t1} timer. Hvor mange kokke kræver "
        "det for at gøre det færdigt på {t2} timer?",
        "Et rengøringsfirma med {n1} medarbejdere rengør et hotel på {t1} timer. "
        "Hvor mange medarbejdere skal der til for at gøre det på {t2} timer?",
        "{n1} bagere kan færdiggøre en ordre på {t1} timer. Hvor mange bagere "
        "skal der til for at gøre det færdigt på {t2} timer?",
        "En pumpe med {n1} indløb tømmer bassinet på {t1} minutter. Hvor mange "
        "indløb kræves for at tømme det på {t2} minutter?",
        "Med {n1} håndværkere kan et projekt afsluttes på {t1} dage. Hvor mange "
        "håndværkere skal der til for at afslutte det på {t2} dage?",
        "En avisrute betjenes af {n1} bude på {t1} timer. Hvor mange bude skal "
        "der til, hvis samme rute skal klares på {t2} timer?",
        "{n1} landmænd kan høste en mark på {t1} dage. Hvor mange landmænd skal "
        "der til, hvis marken skal høstes på {t2} dage?",
        "{n1} musikere kan gennemføre et program på {t1} minutter. Hvor mange "
        "musikere skal der til, hvis programmet skal gennemføres på {t2} minutter?",
        "Et budfirma med {n1} chauffører leverer alle pakker på {t1} timer. "
        "Hvor mange chauffører kræves for at gøre det på {t2} timer?",
        "Et kokketeam på {n1} personer laver en firmafrokost på {t1} timer. "
        "Hvor mange kokke skal der til for at gøre det på {t2} timer?",
    ],
}

@dataclass
class InvRateParams:
    n1: int
    t1: int
    n2: int         # "the other known" — count of workers for find_time; else derived
    t2: int         # "the other known" — target time for find_workers; else derived
    work: int
    ask: str        # "find_time" | "find_workers"
    answer: int


def _inv_steps_constant_product(p: InvRateParams) -> tuple[list[Step], str]:
    if p.ask == "find_time":
        unit = "time" if p.answer == 1 else "timer"
        return [
            Step(pre="Det samlede arbejde svarer til antal personer gange tid:",
                 expr=f"{p.n1} * {p.t1}", result=str(p.work),
                 post=f"altså {p.work} personetimer i alt."),
            Step(pre=f"Med {p.n2} personer bliver tiden det samlede arbejde "
                     "divideret med det nye antal:",
                 expr=f"{p.work} / {p.n2}", result=str(p.answer),
                 post=f"altså {p.answer} {unit}."),
        ], str(p.answer)
    # find_workers
    unit = "person" if p.answer == 1 else "personer"
    return [
        Step(pre="Det samlede arbejde svarer til antal personer gange tid:",
             expr=f"{p.n1} * {p.t1}", result=str(p.work),
             post=f"altså {p.work} personetimer i alt."),
        Step(pre=f"For at gøre det på {p.t2} tidsenheder skal antallet af "
                 "personer være arbejdet divideret med den nye tid:",
             expr=f"{p.work} / {p.t2}", result=str(p.answer),
             post=f"altså {p.answer} {unit}."),
    ], str(p.answer)


def _inv_steps_equation(p: InvRateParams) -> tuple[list[Step], str]:
    if p.ask == "find_time":
        unit = "time" if p.answer == 1 else "timer"
        return [
            Step(pre="Da arbejdet er omvendt proportionalt med antal personer, "
                     f"gælder {p.n1} * {p.t1} = {p.n2} * t. Vi løser for t:",
                 expr=f"{p.n1} * {p.t1} / {p.n2}", result=str(p.answer),
                 post=f"altså t = {p.answer} {unit}."),
        ], str(p.answer)
    # find_workers
    unit = "person" if p.answer == 1 else "personer"
    return [
        Step(pre="Da arbejdet er omvendt proportionalt med antal personer, "
                 f"gælder {p.n1} * {p.t1} = n * {p.t2}. Vi løser for n:",
             expr=f"{p.n1} * {p.t1} / {p.t2}", result=str(p.answer),
             post=f"altså n = {p.answer} {unit}."),
    ], str(p.answer)


_INV_STEPS = {
    "constant_product": _inv_steps_constant_product,
    "equation": _inv_steps_equation,
}


def sample_inverse_rate(rng: random.Random) -> Problem:
    ask = rng.choice(["find_time", "find_workers"])
    while True:
        n1 = rng.randint(2, 20)
        t1 = rng.randint(2, 40)
        work = n1 * t1
        if ask == "find_time":
            n2 = rng.randint(2, 20)
            if n2 == n1:
                continue
            if work % n2 == 0:
                answer = work // n2
                t2 = answer
                if 1 <= answer <= 200:
                    break
        else:  # find_workers
            t2 = rng.randint(2, 40)
            if t2 == t1:
                continue
            if work % t2 == 0:
                answer = work // t2
                n2 = answer
                if 1 <= answer <= 200:
                    break

    q_tpl = rng.choice(INVERSE_RATE_TEMPLATES[ask])
    question = q_tpl.format(n1=n1, t1=t1, n2=n2, t2=t2)
    p = InvRateParams(n1=n1, t1=t1, n2=n2, t2=t2, work=work,
                      ask=ask, answer=answer)
    strategy = rng.choice(list(_INV_STEPS))
    steps, final = _INV_STEPS[strategy](p)
    chain = render_prose(steps, final)
    funcall = render_funcall(question, steps, final)

    return Problem(
        type="inverse_rate",
        question_da=question,
        chain_da=chain,
        answer=final,
        params={"n1": n1, "t1": t1, "n2": n2, "t2": t2, "work": work,
                "ask": ask},
        strategy=f"{strategy}_{ask}",
        steps=[asdict(s) for s in steps],
        funcall=funcall,
    )


# ── CONSECUTIVE ─────────────────────────────────────────────────────────

CONSECUTIVE_TEMPLATES = {
    "any": [
        "Summen af {N} på hinanden følgende heltal er {S}. Hvad er det "
        "{ordinal}?",
        "{N} heltal, som følger lige efter hinanden, har summen {S}. "
        "Hvad er det {ordinal}?",
        "Find det {ordinal} af {N} på hinanden følgende heltal, der har "
        "summen {S}.",
        "{N} tal i træk lægger sammen til {S}. Hvilket tal er det {ordinal}?",
        "Hvis {N} på hinanden følgende heltal har en samlet sum på {S}, "
        "hvad er så det {ordinal} tal?",
        "Der er {N} heltal i træk, hvis sum udgør {S}. Angiv det {ordinal} "
        "tal i rækken.",
        "En række af {N} på hinanden følgende heltal summer til {S}. "
        "Bestem det {ordinal} tal.",
    ],
    "even": [
        "Summen af {N} på hinanden følgende lige tal er {S}. Hvad er det "
        "{ordinal}?",
        "{N} lige tal i træk giver tilsammen {S}. Hvad er det {ordinal}?",
        "Hvis {N} på hinanden følgende lige heltal har sum {S}, "
        "hvad er så det {ordinal} tal?",
        "Der findes {N} lige tal i træk med summen {S}. Angiv det {ordinal}.",
        "Bestem det {ordinal} af {N} på hinanden følgende lige tal, som "
        "sammenlagt giver {S}.",
    ],
    "odd": [
        "Summen af {N} på hinanden følgende ulige tal er {S}. Hvad er det "
        "{ordinal}?",
        "{N} ulige tal i træk giver tilsammen {S}. Hvad er det {ordinal}?",
        "Hvis {N} på hinanden følgende ulige heltal har sum {S}, "
        "hvad er så det {ordinal} tal?",
        "Der findes {N} ulige tal i træk med summen {S}. Angiv det {ordinal}.",
        "Bestem det {ordinal} af {N} på hinanden følgende ulige tal, som "
        "sammenlagt giver {S}.",
    ],
}

ORDINALS = ["første", "andet", "tredje", "fjerde", "femte"]


@dataclass
class ConsecParams:
    kind: str
    N: int
    step: int
    smallest: int
    largest: int
    mid: int
    half_span: int
    S: int
    ordinal: str
    ord_idx: int
    answer: int


def _consec_steps(p: ConsecParams) -> tuple[list[Step], str]:
    """Compute the middle, then locate the ordinal term.

    For even N with step=1 the true mean is a half-integer, so we use the
    algebraic derivation S = N*a + N(N-1)/2 → a = (S - N(N-1)/2) / N instead
    of the mean-based chain (which would round the mean and teach wrong math).
    """
    kind_word = {"any": "heltal", "even": "lige tal", "odd": "ulige tal"}[p.kind]

    if p.N % 2 == 0 and p.step == 1:
        offset = p.N * (p.N - 1) // 2
        num = p.S - offset
        steps = [
            Step(pre=f"For {p.N} på hinanden følgende heltal a, a+1, ..., a+{p.N - 1} "
                     f"er summen {p.N}·a + {offset}, så {p.N}·a = summen minus {offset}:",
                 expr=f"{p.S} - {offset}", result=str(num),
                 post=f"altså {p.N}·a = {num}."),
            Step(pre="Vi løser for det mindste tal a:",
                 expr=f"{num} / {p.N}", result=str(p.smallest),
                 post=f"altså det mindste er {p.smallest}."),
            Step(pre=f"Det {p.ordinal} tal er det mindste plus "
                     f"{p.ord_idx} * {p.step}:",
                 expr=f"{p.smallest} + {p.ord_idx} * {p.step}",
                 result=str(p.answer),
                 post=f"altså det {p.ordinal} er {p.answer}."),
        ]
        return steps, str(p.answer)

    steps = [
        Step(pre=f"Middeltallet af {p.N} på hinanden følgende {kind_word} er "
                 "summen divideret med antallet:",
             expr=f"{p.S} / {p.N}", result=str(p.mid),
             post=f"altså {p.mid}."),
    ]
    if p.step == 2:
        span_desc = ("Da tallene er lige og på hinanden følgende, er "
                     "afstanden mellem hvert tal 2.")
        if p.kind == "odd":
            span_desc = ("Da tallene er ulige og på hinanden følgende, er "
                         "afstanden mellem hvert tal 2.")
        steps.append(Step(pre=span_desc,
                          expr=f"{p.smallest} + ({p.N} - 1) * {p.step}",
                          result=str(p.largest),
                          post=f"altså det mindste er {p.smallest}, "
                               f"det største er {p.largest}."))
    else:
        steps.append(Step(pre="Det mindste tal er middeltallet minus halve "
                              "spændvidde:",
                          expr=f"{p.mid} - {p.half_span}",
                          result=str(p.smallest),
                          post=f"altså det mindste er {p.smallest}."))
    steps.append(Step(pre=f"Det {p.ordinal} tal er det mindste plus "
                          f"{p.ord_idx} * {p.step}:",
                      expr=f"{p.smallest} + {p.ord_idx} * {p.step}",
                      result=str(p.answer),
                      post=f"altså det {p.ordinal} er {p.answer}."))
    return steps, str(p.answer)


_CONSEC_STEPS = {
    "mean_and_span": _consec_steps,
}


def sample_consecutive(rng: random.Random) -> Problem:
    kind = rng.choice(["any", "even", "odd"])
    N = rng.choice([3, 4, 5])
    step = 1 if kind == "any" else 2
    # Wider midpoint range → much bigger param space
    if kind == "any":
        mid = rng.randint(5, 200)
    elif kind == "even":
        mid = rng.choice(range(6, 200, 2)) if N % 2 == 1 else rng.choice(range(5, 199, 2))
    else:  # odd
        mid = rng.choice(range(5, 201, 2)) if N % 2 == 1 else rng.choice(range(6, 200, 2))

    # Total = N * mid (if N is odd) OR = N * mid.5 (needs adjustment for even N)
    # Simplify: for odd N, middle is a real integer; for even N, "middle" is between two
    if N % 2 == 1:
        half_span = (N // 2) * step
        smallest = mid - half_span
        largest = mid + half_span
        S = N * mid
    else:
        # For even N, middle sits between term N/2 and term N/2+1
        half = N // 2
        smallest = mid - (half - 1) * step - step // 1  # approximate; regen below
        # simpler: build directly from smallest
        smallest = rng.randint(3, 30)
        if kind == "even" and smallest % 2 != 0:
            smallest += 1
        if kind == "odd" and smallest % 2 == 0:
            smallest += 1
        largest = smallest + (N - 1) * step
        S = sum(range(smallest, largest + 1, step))
        mid = (smallest + largest) // 2  # for reporting only
        half_span = (largest - smallest) // 2

    # For chain rendering we need integer half_span for even-N cases too; skip if not
    if smallest < 1 or largest > 500:
        # regenerate: return a fresh sample
        return sample_consecutive(rng)

    ord_idx = rng.randint(0, N - 1)
    ordinal = ORDINALS[ord_idx]
    answer = smallest + ord_idx * step

    q_tpl = rng.choice(CONSECUTIVE_TEMPLATES[kind])
    question = q_tpl.format(N=N, S=S, ordinal=ordinal)

    p = ConsecParams(kind=kind, N=N, step=step, smallest=smallest,
                     largest=largest, mid=mid, half_span=half_span,
                     S=S, ordinal=ordinal, ord_idx=ord_idx, answer=answer)
    strategy = "mean_and_span"
    steps, final = _CONSEC_STEPS[strategy](p)
    chain = render_prose(steps, final)
    funcall = render_funcall(question, steps, final)

    return Problem(
        type="consecutive",
        question_da=question,
        chain_da=chain,
        answer=final,
        params={"kind": kind, "N": N, "step": step, "smallest": smallest,
                "largest": largest, "sum": S, "which": ordinal},
        strategy=f"consec_{kind}_{strategy}",
        steps=[asdict(s) for s in steps],
        funcall=funcall,
    )


# ── COIN ────────────────────────────────────────────────────────────────

COIN_DENOMS = [
    (5, 10, "5-krone", "10-krone", "5-kroner", "10-kroner"),
    (10, 20, "10-krone", "20-krone", "10-kroner", "20-kroner"),
    (20, 50, "20-krone", "50-krone", "20-kroner", "50-kroner"),
    (50, 100, "50-krone", "100-krone", "50-kroner", "100-kroner"),
    (100, 500, "100-krone", "500-krone", "100-kroner", "500-kroner"),
]

COIN_TEMPLATES = [
    "{name} har tilsammen {C} mønter fordelt på {d1_pl} og {d2_pl}. Den "
    "samlede værdi er {V} kr. Hvor mange {ask_pl} har {name}?",
    "I en pengekasse ligger der {C} mønter, som enten er {d1_pl} eller "
    "{d2_pl}. Værdien er {V} kr. Hvor mange {ask_pl} er der?",
    "En kasserer optæller {C} mønter i {d1_pl} og {d2_pl}. Beløbet er "
    "{V} kr. Hvor mange {ask_pl} er der?",
    "{name} tømmer sin sparegris og finder {C} mønter i to slags: {d1_pl} "
    "og {d2_pl}. Værdien er {V} kr. Hvor mange {ask_pl} har {name}?",
    "En pose indeholder {C} mønter, som fordeler sig på {d1_pl} og {d2_pl}. "
    "Den samlede sum er {V} kr. Hvor mange {ask_pl} er der i posen?",
    "I en samling af {C} mønter — alle enten {d1_pl} eller {d2_pl} — "
    "er den samlede værdi {V} kr. Bestem antallet af {ask_pl}.",
    "{name} har sparet op i alt {V} kr fordelt på {C} mønter af to typer: "
    "{d1_pl} og {d2_pl}. Hvor mange {ask_pl} er det?",
    "En vekseler har {C} mønter i {d1_pl} og {d2_pl} med en samlet værdi på "
    "{V} kr. Hvor mange {ask_pl} er der?",
    "Efter en indsamling er der {C} mønter i {d1_pl} og {d2_pl}, i alt "
    "{V} kr. Hvor mange {ask_pl} er indsamlet?",
    "En butik har {C} mønter til byttepenge — kun {d1_pl} og {d2_pl}. "
    "Værdien er {V} kr. Hvor mange {ask_pl} er der?",
]

@dataclass
class CoinParams:
    d1: int
    d2: int
    d1_sg: str
    d2_sg: str
    d1_pl: str
    d2_pl: str
    C: int
    V: int
    d2C: int
    diff: int
    step: int
    a: int
    b: int
    ask: str      # "d1" | "d2"
    answer: int


def _coin_final_step(p: CoinParams) -> Step | None:
    """If we solved for a and the question asks for b (d2), add a step."""
    if p.ask == "d1":
        return None
    return Step(pre=f"Antal {p.d2_pl} = {p.C} - a:",
                expr=f"{p.C} - {p.a}", result=str(p.b),
                post=f"altså {p.b} {p.d2_pl}.")


def _coin_steps_substitution(p: CoinParams) -> tuple[list[Step], str]:
    steps = [
        Step(pre=f"Lad a være antal {p.d1_pl} og b antal {p.d2_pl}. "
                 f"Vi har a + b = {p.C} og {p.d1} * a + {p.d2} * b = {p.V}. "
                 f"Fra første ligning: b = {p.C} - a. Indsat:",
             expr=f"{p.d1} * a + {p.d2} * ({p.C} - a)",
             result=f"{p.V}",
             post=f"altså ligningen bliver: {p.d1}a + {p.d2C} - {p.d2}a = {p.V}."),
        Step(pre=f"Vi flytter over og isolerer a: "
                 f"({p.d1} - {p.d2}) * a = {p.V} - {p.d2C}. Vi udregner:",
             expr=f"({p.V} - {p.d2C}) / ({p.d1} - {p.d2})",
             result=str(p.a),
             post=f"altså a = {p.a}."),
    ]
    final = _coin_final_step(p)
    if final is not None:
        steps.append(final)
    return steps, str(p.answer)


def _coin_steps_if_all_d2(p: CoinParams) -> tuple[list[Step], str]:
    steps = [
        Step(pre=f"Hvis alle {p.C} mønter var {p.d2_pl}, ville værdien være",
             expr=f"{p.C} * {p.d2}", result=str(p.d2C),
             post=f"altså {p.d2C} kr."),
        Step(pre=f"Den faktiske værdi er {p.V} kr, så forskellen er",
             expr=f"{p.d2C} - {p.V}", result=str(p.diff),
             post=f"altså {p.diff} kr mindre end 'kun {p.d2_pl}'-scenariet."),
        Step(pre=f"Hver {p.d1_sg} bidrager med {p.d2} - {p.d1} = {p.step} kr "
                 f"mindre end en {p.d2_sg}. Antallet af {p.d1_pl}:",
             expr=f"{p.diff} / {p.step}", result=str(p.a),
             post=f"altså {p.a} {p.d1_pl}."),
    ]
    final = _coin_final_step(p)
    if final is not None:
        steps.append(final)
    return steps, str(p.answer)


_COIN_STEPS = {
    "substitution": _coin_steps_substitution,
    "if_all_d2": _coin_steps_if_all_d2,
}


def sample_coin(rng: random.Random) -> Problem:
    d1, d2, d1_sg, d2_sg, d1_pl, d2_pl = rng.choice(COIN_DENOMS)
    a = rng.randint(1, 30)
    b = rng.randint(1, 30)
    C = a + b
    V = d1 * a + d2 * b
    d2C = d2 * C
    diff = d2C - V
    step = d2 - d1
    name = rng.choice(NAMES)

    ask = rng.choice(["d1", "d2"])
    ask_pl = d1_pl if ask == "d1" else d2_pl
    answer = a if ask == "d1" else b

    q_tpl = rng.choice(COIN_TEMPLATES)
    question = q_tpl.format(name=name, C=C, V=V,
                            d1_pl=d1_pl, d2_pl=d2_pl, ask_pl=ask_pl)

    p = CoinParams(d1=d1, d2=d2, d1_sg=d1_sg, d2_sg=d2_sg,
                   d1_pl=d1_pl, d2_pl=d2_pl,
                   C=C, V=V, d2C=d2C, diff=diff, step=step,
                   a=a, b=b, ask=ask, answer=answer)
    strategy = rng.choice(list(_COIN_STEPS))
    steps, final = _COIN_STEPS[strategy](p)
    chain = render_prose(steps, final)
    funcall = render_funcall(question, steps, final)

    return Problem(
        type="coin",
        question_da=question,
        chain_da=chain,
        answer=final,
        params={"d1": d1, "d2": d2, "C": C, "V": V, "a": a, "b": b,
                "name": name, "ask": ask},
        strategy=f"{strategy}_{ask}",
        steps=[asdict(s) for s in steps],
        funcall=funcall,
    )


# ── AGE ─────────────────────────────────────────────────────────────────

# Simple-now templates: X = k * Y right now, ask about Y or X or a future age.
AGE_SIMPLE_NOW_TEMPLATES = {
    # ask "young": given X and k, find Y (the younger age). X is stated.
    "young": [
        "{name_a} er {k} gange så gammel som {name_b}. {name_a} er {X} år. "
        "Hvor gammel er {name_b}?",
        "{name_a} er {X} år, og {name_a}s alder er {k} gange {name_b}s. "
        "Hvor gammel er {name_b}?",
        "Forholdet mellem {name_a}s og {name_b}s alder er {k}:1. {name_a} er "
        "{X} år. Hvor gammel er {name_b}?",
        "{name_a} er præcis {k} gange så gammel som {name_b}. Hvis {name_a} "
        "er {X} år, hvor gammel er {name_b} så?",
        "Hvis {name_a} er {k} gange så gammel som {name_b} og {name_a} netop "
        "er fyldt {X} år, hvor gammel er {name_b}?",
    ],
    # ask "old": given Y and k, find X. Y is stated.
    "old": [
        "{name_a} er {k} gange så gammel som {name_b}. {name_b} er {Y} år. "
        "Hvor gammel er {name_a}?",
        "{name_b} er {Y} år, og {name_a}s alder er {k} gange {name_b}s. "
        "Hvor gammel er {name_a}?",
        "Forholdet mellem {name_a}s og {name_b}s alder er {k}:1. {name_b} er "
        "{Y} år. Hvor gammel er {name_a}?",
        "{name_a} er præcis {k} gange så gammel som {name_b}. Hvis {name_b} "
        "er {Y} år, hvor gammel er {name_a} så?",
        "Hvis {name_a} er {k} gange så gammel som {name_b} og {name_b} netop "
        "er fyldt {Y} år, hvor gammel er {name_a}?",
    ],
    # ask "future": given Y and k (and thus X = k*Y), ask "in T years, how old
    # will name_b be?" answer = Y + T
    "future": [
        "{name_a} er {k} gange så gammel som {name_b}. {name_b} er {Y} år i "
        "dag. Hvor gammel er {name_b} om {future_t} år?",
        "{name_b} er {Y} år, og {name_a} er {k} gange så gammel. Hvor gammel "
        "vil {name_b} være om {future_t} år?",
        "Forholdet mellem {name_a}s og {name_b}s alder er {k}:1. {name_b} er "
        "{Y} år. Hvor gammel er {name_b} om {future_t} år?",
        "Hvis {name_a} er {k} gange så gammel som {name_b}, og {name_b} er "
        "{Y} år i dag, hvor gammel er {name_b} så om {future_t} år?",
    ],
}

AGE_TEMPLATES = {
    # "t": ask for number of years until the condition holds
    "t": [
        "{name_a} er {X} år, og {name_b} er {Y} år. Om hvor mange år vil "
        "{name_a} være {k} gange så gammel som {name_b}?",
        "Lige nu er {name_a} {X} år, mens {name_b} er {Y} år gammel. Om hvor "
        "mange år er {name_a}s alder {k} gange {name_b}s?",
        "{name_a} er i dag {X} år, {name_b} er {Y} år. Hvornår (om hvor mange "
        "år) vil forholdet mellem deres aldre være {k}:1?",
        "{name_a}s alder er {X} år, og {name_b}s alder er {Y} år. Om hvor "
        "mange år vil {name_a} være {k} gange så gammel som {name_b}?",
        "For nuværende er {name_a} {X} år og {name_b} {Y} år. "
        "Efter hvor mange år bliver {name_a}s alder {k} gange {name_b}s?",
        "{name_a} og {name_b} er hhv. {X} og {Y} år gamle. Om hvor mange år "
        "vil {name_a} være præcis {k} gange så gammel som {name_b}?",
        "I dag er {name_a} {X} år, og {name_b} er {Y} år. Bestem antallet af "
        "år, indtil {name_a} er {k} gange så gammel som {name_b}.",
        "Nu er {name_a} {X} år og {name_b} {Y} år. Om hvor mange år er "
        "{name_a} nøjagtig {k} gange så gammel som {name_b}?",
        "Aldersforskellen mellem {name_a} ({X} år) og {name_b} ({Y} år) betyder, "
        "at {name_a} om nogle år vil være {k} gange så gammel som {name_b}. "
        "Hvor mange år er der tale om?",
        "{name_a} er {X} år og {name_b} er {Y} år i år. "
        "Hvor mange år går der, før {name_a}s alder er {k} gange {name_b}s?",
    ],
    # "age_a": ask A's age at the point the condition holds (X + t)
    "age_a": [
        "{name_a} er {X} år, og {name_b} er {Y} år. Hvor gammel vil {name_a} "
        "være, når {name_a} er {k} gange så gammel som {name_b}?",
        "Lige nu er {name_a} {X} år og {name_b} {Y} år. Hvor mange år vil "
        "{name_a} være, når forholdet mellem deres aldre er {k}:1?",
        "{name_a} er i dag {X} år og {name_b} {Y} år. Hvor gammel er "
        "{name_a} på det tidspunkt, hvor {name_a}s alder er {k} gange "
        "{name_b}s?",
        "I dag er {name_a} {X} år og {name_b} {Y} år. Bestem {name_a}s alder "
        "den dag, {name_a} bliver præcis {k} gange så gammel som {name_b}.",
        "For nuværende er {name_a} {X} år og {name_b} {Y} år. Hvor gammel "
        "vil {name_a} være, når {name_a}s alder er {k} gange {name_b}s?",
    ],
    # "age_b": ask B's age at the point the condition holds (Y + t)
    "age_b": [
        "{name_a} er {X} år, og {name_b} er {Y} år. Hvor gammel vil {name_b} "
        "være på det tidspunkt, hvor {name_a} er {k} gange så gammel som "
        "{name_b}?",
        "Lige nu er {name_a} {X} år og {name_b} {Y} år. Hvor gammel bliver "
        "{name_b}, når {name_a} er {k} gange så gammel som {name_b}?",
        "{name_a} er i dag {X} år og {name_b} {Y} år. Hvor mange år er "
        "{name_b}, når {name_a}s alder er {k} gange {name_b}s?",
        "I dag er {name_a} {X} år og {name_b} {Y} år. Angiv {name_b}s alder "
        "den dag, {name_a} bliver præcis {k} gange så gammel som {name_b}.",
        "For nuværende er {name_a} {X} år og {name_b} {Y} år. Hvor gammel "
        "vil {name_b} være, når {name_a}s alder er {k} gange {name_b}s?",
    ],
}

@dataclass
class AgeParams:
    name_a: str
    name_b: str
    X: int
    Y: int
    k: int
    kY: int
    num: int
    denom: int
    kind: str       # "time_shift" | "simple_now"
    t: int          # years until the condition holds (time_shift) OR future_t (simple_now future)
    ask: str        # time_shift: "t"|"age_a"|"age_b" ;  simple_now: "young"|"old"|"future"
    answer: int


def _age_final_step(p: AgeParams) -> Step | None:
    """After t is known, add a step to compute the requested quantity."""
    if p.ask == "t":
        return None
    if p.ask == "age_a":
        return Step(pre=f"{p.name_a}s alder på det tidspunkt er {p.X} + t:",
                    expr=f"{p.X} + {p.t}", result=str(p.answer),
                    post=f"altså {p.name_a} er {p.answer} år.")
    if p.ask == "age_b":
        return Step(pre=f"{p.name_b}s alder på det tidspunkt er {p.Y} + t:",
                    expr=f"{p.Y} + {p.t}", result=str(p.answer),
                    post=f"altså {p.name_b} er {p.answer} år.")
    raise ValueError(p.ask)


def _age_steps_expand(p: AgeParams) -> tuple[list[Step], str]:
    steps = [
        Step(pre=f"Lad t være antal år. Om t år er {p.name_a} {p.X} + t og "
                 f"{p.name_b} {p.Y} + t. Vi ønsker "
                 f"{p.X} + t = {p.k} * ({p.Y} + t). "
                 f"Højreside udregnes: {p.k}({p.Y} + t) =",
             expr=f"{p.k} * {p.Y}", result=str(p.kY),
             post=f"altså {p.X} + t = {p.kY} + {p.k}t."),
        Step(pre=f"Vi flytter så alle t-led til den ene side: "
                 f"({p.k} - 1)t = {p.X} - {p.kY}. Vi udregner:",
             expr=f"({p.X} - {p.kY}) / ({p.k} - 1)",
             result=str(p.t),
             post=f"altså t = {p.t}."),
    ]
    final_step = _age_final_step(p)
    if final_step is not None:
        steps.append(final_step)
    return steps, str(p.answer)


def _age_steps_ratio(p: AgeParams) -> tuple[list[Step], str]:
    steps = [
        Step(pre=f"Om t år skal forholdet ({p.X} + t) / ({p.Y} + t) = {p.k}. "
                 f"Vi ganger op og udregner {p.k} * {p.Y}:",
             expr=f"{p.k} * {p.Y}", result=str(p.kY),
             post=f"altså kY = {p.kY}."),
        Step(pre=f"Så {p.X} + t = {p.kY} + {p.k}t, hvilket giver "
                 f"({p.k} - 1)t = {p.X} - {p.kY}. Vi udregner:",
             expr=f"({p.X} - {p.kY}) / ({p.k} - 1)",
             result=str(p.t),
             post=f"altså t = {p.t}."),
    ]
    final_step = _age_final_step(p)
    if final_step is not None:
        steps.append(final_step)
    return steps, str(p.answer)


def _age_steps_simple_now(p: AgeParams) -> tuple[list[Step], str]:
    """X = k * Y. Given the two known quantities, compute the third.

    ask="young": know X and k, find Y = X / k.
    ask="old":   know Y and k, find X = k * Y.
    ask="future": know Y and k, add future_t. answer = Y + t.
    """
    if p.ask == "young":
        return [
            Step(pre=f"Da {p.name_a} er {p.k} gange så gammel som {p.name_b} "
                     f"og {p.name_a} er {p.X} år, er {p.name_b}s alder "
                     f"{p.X} divideret med {p.k}:",
                 expr=f"{p.X} / {p.k}", result=str(p.Y),
                 post=f"altså {p.name_b} er {p.Y} år."),
        ], str(p.answer)
    if p.ask == "old":
        return [
            Step(pre=f"Da {p.name_a} er {p.k} gange så gammel som {p.name_b} "
                     f"og {p.name_b} er {p.Y} år, er {p.name_a}s alder "
                     f"{p.k} gange {p.Y}:",
                 expr=f"{p.k} * {p.Y}", result=str(p.X),
                 post=f"altså {p.name_a} er {p.X} år."),
        ], str(p.answer)
    if p.ask == "future":
        return [
            Step(pre=f"{p.name_b} er {p.Y} år i dag. Om {p.t} år er "
                     f"{p.name_b}s alder {p.Y} + {p.t}:",
                 expr=f"{p.Y} + {p.t}", result=str(p.answer),
                 post=f"altså {p.name_b} er {p.answer} år om {p.t} år."),
        ], str(p.answer)
    raise ValueError(p.ask)


_AGE_STEPS = {
    "expand": _age_steps_expand,
    "ratio": _age_steps_ratio,
    "simple_now": _age_steps_simple_now,
}


def sample_age(rng: random.Random) -> Problem:
    kind = rng.choice(["time_shift", "time_shift", "simple_now"])  # bias existing
    name_a, name_b = rng.sample(NAMES, 2)

    if kind == "time_shift":
        while True:
            Y = rng.randint(2, 35)
            k = rng.randint(2, 5)
            t = rng.randint(1, 30)
            X = k * (Y + t) - t
            if 3 <= X <= 120 and X > Y and X != Y:
                break
        kY = k * Y
        num = X - kY
        denom = k - 1
        ask = rng.choice(["t", "age_a", "age_b"])
        answer = {"t": t, "age_a": X + t, "age_b": Y + t}[ask]
        q_tpl = rng.choice(AGE_TEMPLATES[ask])
        question = q_tpl.format(name_a=name_a, name_b=name_b, X=X, Y=Y, k=k)
        strategy = rng.choice(["expand", "ratio"])
    else:  # simple_now: X = k * Y right now
        while True:
            k = rng.randint(2, 6)
            Y = rng.randint(3, 40)
            X = k * Y
            if 6 <= X <= 120:
                break
        ask = rng.choice(["young", "old", "future"])
        future_t = rng.choice([3, 5, 10, 15, 20]) if ask == "future" else 0
        answer = {"young": Y, "old": X, "future": Y + future_t}[ask]
        # Fields not used by simple_now, keep zero for schema stability
        kY = k * Y
        num = 0
        denom = 0
        t = future_t
        q_tpl = rng.choice(AGE_SIMPLE_NOW_TEMPLATES[ask])
        question = q_tpl.format(name_a=name_a, name_b=name_b, X=X, Y=Y, k=k,
                                future_t=future_t)
        strategy = "simple_now"

    p = AgeParams(name_a=name_a, name_b=name_b, X=X, Y=Y, k=k, kY=kY,
                  num=num, denom=denom, kind=kind, t=t, ask=ask,
                  answer=answer)
    steps, final = _AGE_STEPS[strategy](p)
    chain = render_prose(steps, final)
    funcall = render_funcall(question, steps, final)

    return Problem(
        type="age",
        question_da=question,
        chain_da=chain,
        answer=final,
        params={"name_a": name_a, "name_b": name_b, "X": X, "Y": Y, "k": k,
                "kind": kind, "t": t, "ask": ask},
        strategy=f"{strategy}_{ask}",
        steps=[asdict(s) for s in steps],
        funcall=funcall,
    )


# ── MIXTURE ─────────────────────────────────────────────────────────────

MIXTURE_TEMPLATES = {}
MIXTURE_TEMPLATES["blend"] = [
    "En kemiker blander {V1} ml opløsning med {C1}% koncentration med "
    "{V2} ml opløsning med {C2}% koncentration. Hvad er koncentrationen "
    "i den samlede blanding (i procent)?",
    "{V1} ml af en {C1}%-opløsning blandes med {V2} ml af en {C2}%-"
    "opløsning. Hvad er koncentrationen i den nye blanding (%)?",
    "Man hælder {V1} ml væske med {C1}% saltindhold sammen med {V2} ml "
    "væske med {C2}% saltindhold. Hvad bliver saltindholdet i blandingen "
    "(%)?",
    "På et laboratorium blandes {V1} ml med {C1}% aktivt stof og {V2} ml "
    "med {C2}% aktivt stof. Hvad er indholdet af aktivt stof i den samlede "
    "blanding (%)?",
    "En bartender blander {V1} ml juice med {C1}% frugtindhold med {V2} ml "
    "af en anden juice med {C2}% frugtindhold. Hvad er frugtindholdet i "
    "den blandede drink (%)?",
    "I en pool tilsættes {V1} liter vand med {C1}% klorindhold til {V2} liter "
    "vand med {C2}% klorindhold. Hvad er klorindholdet i den samlede mængde "
    "(%)? (svar i procent)",
    "En landmand blander {V1} liter foder med {C1}% proteinindhold med {V2} "
    "liter foder med {C2}% proteinindhold. Hvad er proteinindholdet i den "
    "endelige blanding (%)?",
    "På et bryggeri kombineres {V1} liter øl med {C1}% alkohol med {V2} liter "
    "øl med {C2}% alkohol. Hvad er alkoholprocenten i blandingen?",
    "En parfumist rører {V1} ml essens med {C1}% duftkoncentration sammen med "
    "{V2} ml essens med {C2}% duftkoncentration. Hvad er duftkoncentrationen "
    "i den nye essens (%)?",
    "Et malingfirma blander {V1} ml maling med {C1}% pigment med {V2} ml "
    "maling med {C2}% pigment. Hvad er pigmentindholdet i den samlede "
    "maling (%)?",
]

# dilute: start with V1 ml at C1%, add water to reach C2%. C2 < C1. answer = ml water.
MIXTURE_TEMPLATES["dilute"] = [
    "En kemiker har {V1} ml opløsning med {C1}% koncentration. Hvor meget vand "
    "skal tilsættes for at reducere koncentrationen til {C2}%?",
    "{V1} ml af en {C1}%-opløsning skal fortyndes til {C2}%. Hvor mange "
    "milliliter vand skal tilsættes?",
    "På et laboratorium har man {V1} ml væske med {C1}% aktivt stof. Man "
    "ønsker at fortynde den til {C2}%. Hvor meget vand skal tilsættes?",
    "En løsning på {V1} ml indeholder {C1}% salt. Hvor mange ml rent vand "
    "skal tilsættes for at nå ned på {C2}%?",
    "Man har {V1} ml juice med {C1}% frugtindhold og vil fortynde den til "
    "{C2}%. Hvor meget vand skal tilsættes?",
    "En pool indeholder {V1} liter vand med {C1}% klor. Hvor mange liter "
    "rent vand skal tilsættes for at bringe klorindholdet ned til {C2}%?",
    "En bartender har {V1} ml drink med {C1}% alkohol og vil have den fortyndet "
    "til {C2}%. Hvor meget vand skal tilsættes?",
    "En parfumist har {V1} ml essens med {C1}% duftkoncentration og "
    "ønsker at få {C2}% koncentration. Hvor meget opløsningsmiddel skal "
    "tilsættes?",
]

# concentrate: start with V1 ml at C1%, add pure solute (100%) to reach C2%. C2 > C1. answer = amount of solute.
MIXTURE_TEMPLATES["concentrate"] = [
    "En kemiker har {V1} ml opløsning med {C1}% koncentration. Hvor meget "
    "rent stof skal tilsættes for at hæve koncentrationen til {C2}%?",
    "{V1} ml af en {C1}%-opløsning skal opkoncentreres til {C2}%. Hvor "
    "mange gram rent stof skal tilsættes?",
    "På et laboratorium har man {V1} ml væske med {C1}% aktivt stof. Man "
    "ønsker at hæve indholdet til {C2}%. Hvor meget rent stof skal tilsættes?",
    "En løsning på {V1} ml indeholder {C1}% salt. Hvor meget rent salt skal "
    "tilsættes for at nå op på {C2}%?",
    "Man har {V1} ml sirup med {C1}% sukker og vil have {C2}% sukker. Hvor "
    "meget rent sukker skal tilsættes?",
    "En parfumist har {V1} ml essens med {C1}% duftkoncentration og "
    "ønsker at opnå {C2}%. Hvor meget rent duftstof skal tilsættes?",
    "Man har {V1} ml opløsning med {C1}% aktivstof og skal have den "
    "koncentreret til {C2}%. Hvor meget rent aktivstof kræves?",
    "En brygger har {V1} liter brygsats med {C1}% sukker og vil hæve den "
    "til {C2}%. Hvor meget rent sukker skal tilsættes?",
]

@dataclass
class MixtureParams:
    kind: str        # "blend" | "dilute" | "concentrate"
    V1: int
    V2: int          # 0 for dilute/concentrate
    C1: int
    C2: int
    m1: int          # amount of solute in solution 1
    m2: int          # blend: solute in sol 2. dilute/concentrate: not used
    msum: int
    msum100: int
    v1c1: int
    v2c2: int
    vsum: int
    v1c1_full: int   # for dilute: V1*C1 (used in "target volume" step)
    diff: int        # for concentrate: V1*(C2-C1)
    denom: int       # for concentrate: 100 - C2
    answer: int


def _mixture_steps_amounts(p: MixtureParams) -> tuple[list[Step], str]:
    steps = [
        Step(pre="Mængden af stof i blanding 1 er volumen gange koncentration "
                 "divideret med 100:",
             expr=f"{p.V1} * {p.C1} / 100", result=str(p.m1),
             post=f"altså {p.m1} enheder stof."),
        Step(pre="Mængden af stof i blanding 2:",
             expr=f"{p.V2} * {p.C2} / 100", result=str(p.m2),
             post=f"altså {p.m2} enheder stof."),
        Step(pre="Samlet mængde stof:",
             expr=f"{p.m1} + {p.m2}", result=str(p.msum),
             post=f"altså {p.msum} enheder."),
        Step(pre="Samlet volumen:",
             expr=f"{p.V1} + {p.V2}", result=str(p.vsum),
             post=f"altså {p.vsum} ml."),
        Step(pre="Koncentrationen bliver den samlede mængde stof divideret "
                 "med den samlede volumen, ganget med 100:",
             expr=f"{p.msum} / {p.vsum} * 100", result=str(p.answer),
             post=f"altså {p.answer}%."),
    ]
    return steps, str(p.answer)


def _mixture_steps_weighted(p: MixtureParams) -> tuple[list[Step], str]:
    steps = [
        Step(pre="Vægtet gennemsnit af koncentrationerne udregnes som "
                 f"({p.V1}*{p.C1} + {p.V2}*{p.C2}) / ({p.V1} + {p.V2}). "
                 "Vi finder tælleren:",
             expr=f"{p.V1} * {p.C1} + {p.V2} * {p.C2}",
             result=str(p.msum100),
             post=f"altså tæller = {p.msum100}."),
        Step(pre="Nævneren:",
             expr=f"{p.V1} + {p.V2}", result=str(p.vsum),
             post=f"altså nævner = {p.vsum}."),
        Step(pre="Koncentrationen bliver:",
             expr=f"{p.msum100} / {p.vsum}", result=str(p.answer),
             post=f"altså {p.answer}%."),
    ]
    return steps, str(p.answer)


def _mixture_steps_dilute(p: MixtureParams) -> tuple[list[Step], str]:
    """V1 ml at C1% + x ml water → same solute, lower concentration C2.

    solute = V1 * C1 / 100. final volume = solute / (C2/100) = V1*C1/C2.
    water added = final volume - V1 = V1*C1/C2 - V1 = V1*(C1 - C2)/C2.
    We render with two integer steps (target volume, then delta) which
    guarantees exact arithmetic when C1*V1 is divisible by C2.
    """
    target_vol = p.v1c1_full // p.C2  # V1*C1 / C2 = target total volume
    return [
        Step(pre=f"Mængden af stof er uændret. For {p.V1} ml med {p.C1}% er "
                 "stofmængden volumen gange procent:",
             expr=f"{p.V1} * {p.C1}", result=str(p.v1c1_full),
             post=f"altså tæller = {p.v1c1_full} (i procent-ml-enheder)."),
        Step(pre=f"For at nå {p.C2}% skal totalvolumen være tæller divideret "
                 f"med den nye procent:",
             expr=f"{p.v1c1_full} / {p.C2}", result=str(target_vol),
             post=f"altså totalvolumen skal være {target_vol} ml."),
        Step(pre="Vandet der skal tilsættes er totalvolumen minus den "
                 "oprindelige volumen:",
             expr=f"{target_vol} - {p.V1}", result=str(p.answer),
             post=f"altså {p.answer} ml vand."),
    ], str(p.answer)


def _mixture_steps_concentrate(p: MixtureParams) -> tuple[list[Step], str]:
    """V1 ml at C1% + x g pure solute (100%) → C2%.

    From (V1*C1 + 100*x) / (V1 + x) = C2 →
    x = V1*(C2 - C1) / (100 - C2). We emit two integer steps: numerator = V1*diff
    and division by (100 - C2).
    """
    return [
        Step(pre=f"Tilsæt x gram rent stof. Ligningen "
                 f"({p.V1}·{p.C1} + 100·x) / ({p.V1} + x) = {p.C2} "
                 f"omskrives til x·(100 - {p.C2}) = {p.V1}·({p.C2} - {p.C1}). "
                 f"Vi finder højresiden:",
             expr=f"{p.V1} * ({p.C2} - {p.C1})", result=str(p.diff),
             post=f"altså tæller = {p.diff}."),
        Step(pre=f"Nævneren er 100 - {p.C2}:",
             expr=f"100 - {p.C2}", result=str(p.denom),
             post=f"altså nævner = {p.denom}."),
        Step(pre="Vi løser for x:",
             expr=f"{p.diff} / {p.denom}", result=str(p.answer),
             post=f"altså x = {p.answer} gram rent stof."),
    ], str(p.answer)


_MIX_STEPS = {
    "amounts": _mixture_steps_amounts,
    "weighted": _mixture_steps_weighted,
    "dilute": _mixture_steps_dilute,
    "concentrate": _mixture_steps_concentrate,
}


_MIXTURE_VOLS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
                 600, 700, 800, 1000]
_MIXTURE_CONCS = [2, 4, 5, 6, 8, 10, 12, 15, 16, 18, 20, 22, 25,
                  28, 30, 35, 40, 45, 50, 60, 70, 80]


def _sample_blend(rng):
    while True:
        V1 = rng.choice(_MIXTURE_VOLS)
        V2 = rng.choice(_MIXTURE_VOLS)
        C1 = rng.choice(_MIXTURE_CONCS)
        C2 = rng.choice(_MIXTURE_CONCS)
        if C1 == C2:
            continue
        v1c1 = V1 * C1
        v2c2 = V2 * C2
        vsum = V1 + V2
        if (v1c1 + v2c2) % vsum == 0:
            answer = (v1c1 + v2c2) // vsum
            if answer != C1 and answer != C2:
                break
    m1 = V1 * C1 // 100 if V1 * C1 % 100 == 0 else None
    m2 = V2 * C2 // 100 if V2 * C2 % 100 == 0 else None
    if m1 is None or m2 is None:
        return _sample_blend(rng)
    msum = m1 + m2
    msum100 = v1c1 + v2c2
    return MixtureParams(kind="blend", V1=V1, V2=V2, C1=C1, C2=C2,
                         m1=m1, m2=m2, msum=msum, msum100=msum100,
                         v1c1=v1c1, v2c2=v2c2, vsum=vsum,
                         v1c1_full=0, diff=0, denom=0, answer=answer)


def _sample_dilute(rng):
    """Add water: C2 < C1, answer = target_vol - V1 with integer arithmetic."""
    for _ in range(200):
        V1 = rng.choice(_MIXTURE_VOLS)
        C1 = rng.choice(_MIXTURE_CONCS)
        C2 = rng.choice(_MIXTURE_CONCS)
        if C2 >= C1:
            continue
        v1c1_full = V1 * C1                 # target volume numerator
        if v1c1_full % C2 != 0:
            continue
        target_vol = v1c1_full // C2
        answer = target_vol - V1
        if 10 <= answer <= 5000:
            return MixtureParams(kind="dilute", V1=V1, V2=0, C1=C1, C2=C2,
                                 m1=0, m2=0, msum=0, msum100=0,
                                 v1c1=v1c1_full, v2c2=0, vsum=0,
                                 v1c1_full=v1c1_full, diff=0, denom=0,
                                 answer=answer)
    # fallback
    return _sample_dilute(rng)


def _sample_concentrate(rng):
    """Add pure solute: C2 > C1, answer = V1*(C2-C1)/(100-C2)."""
    for _ in range(200):
        V1 = rng.choice(_MIXTURE_VOLS)
        C1 = rng.choice(_MIXTURE_CONCS)
        C2 = rng.choice(_MIXTURE_CONCS)
        if C2 <= C1 or C2 >= 100:
            continue
        diff_num = V1 * (C2 - C1)
        denom = 100 - C2
        if diff_num % denom != 0:
            continue
        answer = diff_num // denom
        if 5 <= answer <= 3000:
            return MixtureParams(kind="concentrate", V1=V1, V2=0, C1=C1, C2=C2,
                                 m1=0, m2=0, msum=0, msum100=0,
                                 v1c1=0, v2c2=0, vsum=0,
                                 v1c1_full=0, diff=diff_num, denom=denom,
                                 answer=answer)
    return _sample_concentrate(rng)


def sample_mixture(rng: random.Random) -> Problem:
    kind = rng.choice(["blend", "blend", "dilute", "concentrate"])
    if kind == "blend":
        p = _sample_blend(rng)
        strategy = rng.choice(["amounts", "weighted"])
        q_tpl = rng.choice(MIXTURE_TEMPLATES["blend"])
        question = q_tpl.format(V1=p.V1, V2=p.V2, C1=p.C1, C2=p.C2)
        params = {"kind": "blend", "V1": p.V1, "V2": p.V2, "C1": p.C1, "C2": p.C2}
    else:
        p = _sample_dilute(rng) if kind == "dilute" else _sample_concentrate(rng)
        strategy = kind
        q_tpl = rng.choice(MIXTURE_TEMPLATES[kind])
        question = q_tpl.format(V1=p.V1, C1=p.C1, C2=p.C2)
        params = {"kind": kind, "V1": p.V1, "C1": p.C1, "C2": p.C2}

    steps, final = _MIX_STEPS[strategy](p)
    chain = render_prose(steps, final)
    funcall = render_funcall(question, steps, final)

    return Problem(
        type="mixture",
        question_da=question,
        chain_da=chain,
        answer=final,
        params=params,
        strategy=f"mixture_{strategy}",
        steps=[asdict(s) for s in steps],
        funcall=funcall,
    )


# ── DISTANCE ────────────────────────────────────────────────────────────

DISTANCE_TEMPLATES = {
    # "simple" was single-ask (find d). Split into three asks. Each subdict has
    # its own templates so we can phrase the unknown correctly.
    "simple_d": [
        "En bil kører med {R} km/t i {T} timer. Hvor langt har den kørt?",
        "{name} cykler {R} km/t i {T} timer. Hvor mange kilometer har "
        "{name} tilbagelagt?",
        "Et tog kører i {T} timer med en gennemsnitsfart på {R} km/t. "
        "Hvor lang er strækningen?",
        "En bus kører {R} km/t i {T} timer. Hvor langt bevæger den sig?",
        "{name} løber i {T} timer med en fart på {R} km/t. Hvor mange "
        "kilometer har {name} løbet?",
        "Et fly holder en gennemsnitsfart på {R} km/t i {T} timer. "
        "Hvor lang er strækningen tilbagelagt?",
        "En lastbil kører {T} timer med {R} km/t. Hvor langt kommer den?",
        "En sportsudøver træner med en jævn fart på {R} km/t i {T} timer. "
        "Hvor mange kilometer tilbagelægges?",
    ],
    "simple_r": [
        "En bil tilbagelægger {D} km på {T} timer. Hvad er gennemsnitsfarten "
        "i km/t?",
        "{name} cykler {D} km på {T} timer. Hvad er {name}s gennemsnitsfart "
        "i km/t?",
        "Et tog tilbagelægger en strækning på {D} km på {T} timer. Hvad er "
        "farten i km/t?",
        "En bus kører {D} km på {T} timer. Hvad er gennemsnitsfarten i km/t?",
        "{name} løber {D} km på {T} timer. Hvad er farten i km/t?",
        "Et fly tilbagelægger {D} km på {T} timer. Hvad er den gennemsnitlige "
        "fart i km/t?",
        "En lastbil kører {D} km på {T} timer. Hvad er dens fart i km/t?",
    ],
    "simple_t": [
        "En bil skal køre {D} km med en fart på {R} km/t. Hvor mange timer "
        "tager det?",
        "{name} cykler {D} km med {R} km/t. Hvor mange timer tager turen?",
        "Et tog kører en strækning på {D} km med {R} km/t. Hvor lang tid "
        "tager turen?",
        "En bus skal tilbagelægge {D} km med en fart på {R} km/t. Hvor mange "
        "timer varer turen?",
        "{name} løber {D} km med en fart på {R} km/t. Hvor lang tid tager "
        "det?",
        "Et fly skal flyve {D} km med en gennemsnitsfart på {R} km/t. "
        "Hvor mange timer tager turen?",
        "En lastbil skal køre {D} km med {R} km/t. Hvor lang tid tager det?",
    ],
    "average": [
        "{name} kører {D1} km med {R1} km/t og derefter {D2} km med {R2} km/t. "
        "Hvad er gennemsnitsfarten for hele turen i km/t?",
        "En bilist tilbagelægger {D1} km med {R1} km/t og herefter {D2} km "
        "med {R2} km/t. Hvad er gennemsnitsfarten på hele strækningen?",
        "På første del af turen kører {name} {D1} km med {R1} km/t. På anden "
        "del kører {name} {D2} km med {R2} km/t. Hvad er gennemsnitsfarten "
        "for hele turen (km/t)?",
        "Et tog kører {D1} km med {R1} km/t og derefter {D2} km med {R2} km/t. "
        "Hvad er togets gennemsnitsfart over hele strækningen i km/t?",
        "En cyklist tilbagelægger {D1} km med {R1} km/t og herefter {D2} km "
        "med {R2} km/t. Hvad er den gennemsnitlige fart i km/t?",
        "{name} løber {D1} km med {R1} km/t og derefter {D2} km med {R2} km/t. "
        "Hvad er den gennemsnitlige fart i km/t?",
    ],
    "meeting": [
        "To biler starter samtidig fra hver sin ende af en {D} km lang "
        "strækning og kører mod hinanden. Den ene kører {Ra} km/t, den "
        "anden {Rb} km/t. Om hvor mange timer mødes de?",
        "{name_a} kører {Ra} km/t fra by A mod by B, og {name_b} kører "
        "{Rb} km/t fra by B mod by A. Byerne ligger {D} km fra hinanden. "
        "Om hvor mange timer mødes de?",
        "To tog kører imod hinanden på en {D} km lang strækning. "
        "Det ene tog kører {Ra} km/t, det andet {Rb} km/t. Om hvor mange "
        "timer mødes togene?",
        "{name_a} og {name_b} står {D} km fra hinanden og cykler mod hinanden "
        "med {Ra} km/t og {Rb} km/t. Hvor lang tid går der, før de møder "
        "hinanden?",
        "Fra hver sin ende af en {D} km lang landevej starter {name_a} og "
        "{name_b} på cykel mod hinanden med {Ra} km/t og {Rb} km/t. Om hvor "
        "mange timer mødes de?",
        "To skibe sejler mod hinanden på en {D} km lang rute. Farterne er "
        "{Ra} km/t og {Rb} km/t. Om hvor mange timer møder de hinanden?",
    ],
    "catchup": [
        "{name_a} går af sted med {Ra} km/t. Efter {T0} timer starter "
        "{name_b} fra samme sted og kører {Rb} km/t i samme retning. "
        "Om hvor mange timer indhenter {name_b} {name_a}?",
        "En cyklist kører {Ra} km/t. Efter {T0} timer starter en anden "
        "cyklist samme rute med {Rb} km/t. Hvor mange timer bruger "
        "den anden cyklist på at indhente den første?",
        "{name_a} tager af sted med {Ra} km/t. {T0} timer senere kører "
        "{name_b} samme rute med {Rb} km/t. Om hvor mange timer indhenter "
        "{name_b} {name_a}?",
        "En bus kører {Ra} km/t fra terminalen. {T0} timer senere kører en "
        "hurtigere bus fra samme terminal med {Rb} km/t. Hvor mange timer "
        "efter afgang indhenter den den første?",
        "Et fragtskib forlader havnen med {Ra} km/t. {T0} timer senere "
        "sætter et hurtigere skib af sted med {Rb} km/t i samme retning. "
        "Hvor lang tid går der, før det hurtigere skib indhenter det første?",
        "En motorcyklist kører {Ra} km/t. {T0} timer senere jagter en anden "
        "motorcyklist med {Rb} km/t. Om hvor mange timer indhenter den anden "
        "den første?",
    ],
}

@dataclass
class DistanceSimple:
    R: int
    T: int
    D: int
    ask: str        # "d" | "r" | "t"
    answer: int


@dataclass
class DistanceAverage:
    R1: int
    R2: int
    T1: int
    T2: int
    D1: int
    D2: int
    Dsum: int
    Tsum: int
    answer: int


@dataclass
class DistanceMeeting:
    D: int
    Ra: int
    Rb: int
    Rsum: int
    answer: int


@dataclass
class DistanceCatchup:
    Ra: int
    Rb: int
    T0: int
    Rdiff: int
    lead: int
    answer: int


def _dist_simple(p: DistanceSimple) -> tuple[list[Step], str]:
    if p.ask == "d":
        return [
            Step(pre="Distancen udregnes som fart gange tid:",
                 expr=f"{p.R} * {p.T}", result=str(p.D),
                 post=f"altså {p.D} km."),
        ], str(p.answer)
    if p.ask == "r":
        return [
            Step(pre="Farten udregnes som distance divideret med tid:",
                 expr=f"{p.D} / {p.T}", result=str(p.R),
                 post=f"altså {p.R} km/t."),
        ], str(p.answer)
    # ask == "t"
    unit = "time" if p.T == 1 else "timer"
    return [
        Step(pre="Tiden udregnes som distance divideret med fart:",
             expr=f"{p.D} / {p.R}", result=str(p.T),
             post=f"altså {p.T} {unit}."),
    ], str(p.answer)


def _dist_average(p: DistanceAverage) -> tuple[list[Step], str]:
    hr = lambda n: "time" if n == 1 else "timer"
    steps = [
        Step(pre="Tiden for første del er distance divideret med fart:",
             expr=f"{p.D1} / {p.R1}", result=str(p.T1),
             post=f"altså {p.T1} {hr(p.T1)}."),
        Step(pre="Tiden for anden del:",
             expr=f"{p.D2} / {p.R2}", result=str(p.T2),
             post=f"altså {p.T2} {hr(p.T2)}."),
        Step(pre="Samlet distance:",
             expr=f"{p.D1} + {p.D2}", result=str(p.Dsum),
             post=f"altså {p.Dsum} km."),
        Step(pre="Samlet tid:",
             expr=f"{p.T1} + {p.T2}", result=str(p.Tsum),
             post=f"altså {p.Tsum} {hr(p.Tsum)}."),
        Step(pre="Gennemsnitsfarten er samlet distance divideret med "
                 "samlet tid:",
             expr=f"{p.Dsum} / {p.Tsum}", result=str(p.answer),
             post=f"altså {p.answer} km/t."),
    ]
    return steps, str(p.answer)


def _dist_meeting(p: DistanceMeeting) -> tuple[list[Step], str]:
    unit = "time" if p.answer == 1 else "timer"
    steps = [
        Step(pre="De to køretøjer nærmer sig hinanden med en samlet fart på",
             expr=f"{p.Ra} + {p.Rb}", result=str(p.Rsum),
             post=f"altså {p.Rsum} km/t sammenlagt."),
        Step(pre="Mødetidspunktet findes ved at dele afstanden med den samlede fart:",
             expr=f"{p.D} / {p.Rsum}", result=str(p.answer),
             post=f"altså {p.answer} {unit}."),
    ]
    return steps, str(p.answer)


def _dist_catchup(p: DistanceCatchup) -> tuple[list[Step], str]:
    unit = "time" if p.answer == 1 else "timer"
    steps = [
        Step(pre="Ved den anden's afgang har den første et forspring på",
             expr=f"{p.Ra} * {p.T0}", result=str(p.lead),
             post=f"altså {p.lead} km."),
        Step(pre="Relativ fart mellem den anden og den første er",
             expr=f"{p.Rb} - {p.Ra}", result=str(p.Rdiff),
             post=f"altså {p.Rdiff} km/t netto ind på forspringet."),
        Step(pre="Indhentningstid = forspring / relativ fart:",
             expr=f"{p.lead} / {p.Rdiff}", result=str(p.answer),
             post=f"altså {p.answer} {unit}."),
    ]
    return steps, str(p.answer)


_DIST_STEPS = {
    "simple": _dist_simple,
    "meeting": _dist_meeting,
    "catchup": _dist_catchup,
    "average": _dist_average,
}


def sample_distance(rng: random.Random) -> Problem:
    kind = rng.choice(["simple", "meeting", "catchup", "average"])
    if kind == "simple":
        R = rng.choice([25, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90,
                        100, 110, 120])
        T = rng.randint(2, 15)
        D = R * T
        ask = rng.choice(["d", "r", "t"])
        answer = {"d": D, "r": R, "t": T}[ask]
        name = rng.choice(NAMES)
        tpl_key = f"simple_{ask}"
        q_tpl = rng.choice(DISTANCE_TEMPLATES[tpl_key])
        question = q_tpl.format(R=R, T=T, D=D, name=name)
        p = DistanceSimple(R=R, T=T, D=D, ask=ask, answer=answer)
        params = {"kind": "simple", "R": R, "T": T, "D": D, "ask": ask}
        strategy = f"simple_{ask}"
    elif kind == "average":
        # Two legs at different speeds. Need t1, t2 integer AND avg integer.
        for _ in range(200):
            R1 = rng.choice([20, 30, 40, 50, 60, 70, 80, 90, 100])
            R2 = rng.choice([20, 30, 40, 50, 60, 70, 80, 90, 100])
            if R1 == R2:
                continue
            T1 = rng.randint(1, 6)
            T2 = rng.randint(1, 6)
            D1 = R1 * T1
            D2 = R2 * T2
            Dsum = D1 + D2
            Tsum = T1 + T2
            if Dsum % Tsum == 0:
                answer = Dsum // Tsum
                if answer != R1 and answer != R2 and answer >= 10:
                    break
        else:
            return sample_distance(rng)
        name = rng.choice(NAMES)
        q_tpl = rng.choice(DISTANCE_TEMPLATES["average"])
        question = q_tpl.format(D1=D1, D2=D2, R1=R1, R2=R2, name=name)
        p = DistanceAverage(R1=R1, R2=R2, T1=T1, T2=T2, D1=D1, D2=D2,
                            Dsum=Dsum, Tsum=Tsum, answer=answer)
        params = {"kind": "average", "R1": R1, "R2": R2,
                  "D1": D1, "D2": D2, "T1": T1, "T2": T2, "name": name}
        strategy = "average"
    elif kind == "meeting":
        while True:
            Ra = rng.choice([25, 30, 40, 45, 50, 60, 70, 80, 90, 100])
            Rb = rng.choice([25, 30, 40, 45, 50, 60, 70, 80, 90, 100])
            if Ra == Rb:
                continue
            Rsum = Ra + Rb
            answer_num = rng.randint(2, 12)
            D = Rsum * answer_num
            if D <= 2000:
                answer = answer_num
                break
        name_a, name_b = rng.sample(NAMES, 2)
        q_tpl = rng.choice(DISTANCE_TEMPLATES["meeting"])
        question = q_tpl.format(D=D, Ra=Ra, Rb=Rb,
                                name_a=name_a, name_b=name_b)
        p = DistanceMeeting(D=D, Ra=Ra, Rb=Rb, Rsum=Rsum, answer=answer)
        params = {"kind": "meeting", "D": D, "Ra": Ra, "Rb": Rb,
                  "name_a": name_a, "name_b": name_b}
        strategy = "meeting"
    else:  # catchup
        while True:
            Ra = rng.choice([15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
            Rb = rng.choice([30, 40, 45, 50, 55, 60, 70, 75, 80, 90, 100, 110, 120])
            if Rb <= Ra:
                continue
            T0 = rng.randint(1, 10)
            Rdiff = Rb - Ra
            lead = Ra * T0
            if lead % Rdiff == 0 and 1 <= lead // Rdiff <= 24:
                answer = lead // Rdiff
                break
        name_a, name_b = rng.sample(NAMES, 2)
        q_tpl = rng.choice(DISTANCE_TEMPLATES["catchup"])
        question = q_tpl.format(Ra=Ra, Rb=Rb, T0=T0,
                                name_a=name_a, name_b=name_b)
        p = DistanceCatchup(Ra=Ra, Rb=Rb, T0=T0, Rdiff=Rdiff,
                            lead=lead, answer=answer)
        params = {"kind": "catchup", "Ra": Ra, "Rb": Rb, "T0": T0,
                  "name_a": name_a, "name_b": name_b}
        strategy = "catchup"

    steps, final = _DIST_STEPS[kind](p)
    chain = render_prose(steps, final)
    funcall = render_funcall(question, steps, final)

    return Problem(
        type="distance",
        question_da=question,
        chain_da=chain,
        answer=final,
        params=params,
        strategy=f"distance_{strategy}",
        steps=[asdict(s) for s in steps],
        funcall=funcall,
    )


# ── Driver ──────────────────────────────────────────────────────────────

SAMPLERS = {
    "ratio": sample_ratio,
    "percent": sample_percent,
    "inverse_rate": sample_inverse_rate,
    "consecutive": sample_consecutive,
    "coin": sample_coin,
    "age": sample_age,
    "mixture": sample_mixture,
    "distance": sample_distance,
}

# Master dispatch: type_name → strategy_name → step_fn(params) → (list[Step], str)
# Useful downstream if you want to regenerate steps from params without
# re-running the whole sample_X (e.g. for funcall variants at SFT time).
STEPS_BY_TYPE = {
    "ratio": _RATIO_STEPS,
    "percent": _PERCENT_STEPS,
    "inverse_rate": _INV_STEPS,
    "consecutive": _CONSEC_STEPS,
    "coin": _COIN_STEPS,
    "age": _AGE_STEPS,
    "mixture": _MIX_STEPS,
    "distance": _DIST_STEPS,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=list(SAMPLERS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-attempts-per-row", type=int, default=100,
                    help="give up after this many dedup misses per emitted row")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str, str]] = set()
    written = 0
    attempts_since_last = 0
    with args.out.open("w") as f:
        while written < args.n:
            p = SAMPLERS[args.type](rng)
            p.question_da = fix_possessives(p.question_da)
            p.chain_da = fix_possessives(p.chain_da)
            if p.funcall:
                for msg in p.funcall:
                    if isinstance(msg.get("content"), str):
                        msg["content"] = fix_possessives(msg["content"])
            key = (p.question_da[:60], p.answer)
            attempts_since_last += 1
            if key in seen:
                if attempts_since_last > args.max_attempts_per_row:
                    print(f"stopping early: dedup pool exhausted after {written}")
                    break
                continue
            seen.add(key)
            attempts_since_last = 0
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} → {args.out}")


if __name__ == "__main__":
    main()
