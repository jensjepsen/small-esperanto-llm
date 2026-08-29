"""Procedural word-problem generator. Math by construction; no Gemini in the loop.

Each type gets:
  - `sample(rng) -> Problem` — picks valid params and computes the answer
  - `render_question(p, rng) -> str` — picks one of N prose templates
  - `render_chain(p, strategy, rng) -> str` — walks the actual solver

Output: JSONL with the same schema as generate_word_problems.py so the two
sources are interchangeable downstream.

This is the proof-of-concept on `ratio`. If the pattern works, percent/etc.
are ~100 lines apiece.

Usage:
  uv run python scripts/word_problems_procedural.py \\
    --type ratio --n 1000 --out data/word_problems/ratio_proc.jsonl
"""
import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from esperanto_lm.funcall.tokens import (
    TOOL_CALL_OPEN as TC_O, TOOL_CALL_CLOSE as TC_C,
    TOOL_RESULT_OPEN as TR_O, TOOL_RESULT_CLOSE as TR_C,
)


@dataclass
class Step:
    """One arithmetic step. The model NEVER sees `result` before the call."""
    pre: str = ""    # narration before the call — must not mention `result`
    expr: str = ""   # sympy-evaluable arithmetic
    result: str = "" # ground-truth value (becomes <|tool_result|>)
    post: str = ""   # narration after the call — may name the result


def render_prose(steps: list[Step], final: str) -> str:
    """Inline prose form: `{pre} {expr} = {result}. {post}` per step."""
    lines = []
    for s in steps:
        parts = []
        if s.pre:
            parts.append(s.pre)
        if s.expr:
            parts.append(f"{s.expr} = {s.result}.")
        if s.post:
            parts.append(s.post)
        lines.append(" ".join(parts).strip())
    lines.append(f"#### {final}")
    return "\n".join(lines)


def render_funcall(steps: list[Step], final: str) -> list[dict]:
    """Multi-turn funcall form. Each Step → (assistant pre)(tool_call)(tool_result)(assistant post).
    Consecutive same-role turns are coalesced."""
    turns: list[tuple[str, str]] = []
    for s in steps:
        if s.pre:
            turns.append(("assistant", s.pre))
        if s.expr:
            turns.append(("assistant", f"{TC_O}{s.expr}{TC_C}"))
            turns.append(("tool", f"{TR_O}{s.result}{TR_C}"))
        if s.post:
            turns.append(("assistant", s.post))
    turns.append(("assistant", f"#### {final}"))
    out: list[list] = []
    for role, content in turns:
        if out and out[-1][0] == role:
            out[-1][1] = out[-1][1] + " " + content
        else:
            out.append([role, content])
    return [{"role": r, "content": c} for r, c in out]
_NAMES_FILE = PROJECT_ROOT / "src/esperanto_lm/ontology/sampler.py"
_CONCEPTS = PROJECT_ROOT / "src/esperanto_lm/ontology/data/concepts.jsonl"

_BAD_OBJECTS = {
    "brako", "dento", "dorso", "fingro", "kapo", "kolo", "korpo", "mano",
    "okulo", "orelo", "piedo", "ventro", "vosto", "ŝultro", "buŝo", "haŭto",
    "nazo", "lipo", "lango", "frunto", "mentono", "trunko", "kruro", "genuo",
    "kubuto", "ostoj", "sango", "haro", "ungo", "muskolo", "cerbo", "koro",
    "pulmo", "stomako", "rumpa", "hepato",
    "ĉielo", "ĉielarko", "aŭroro", "fajro", "flako", "vento", "vojo",
    "duno", "vulkano", "ŝtuparo",
}


def load_names() -> list[str]:
    src = _NAMES_FILE.read_text()
    m = re.search(r"PERSON_NAMES\s*=\s*\[(.*?)\]", src, re.DOTALL)
    names = re.findall(r'"([a-zćĉĝĥĵŝŭ]+)"', m.group(1)) if m else []
    return [n.capitalize() for n in names]


def load_objects() -> list[str]:
    out = []
    for line in _CONCEPTS.open():
        r = json.loads(line)
        et = r.get("entity_type", "")
        if isinstance(et, list):
            et = et[0] if et else ""
        lem = r.get("lemma", "")
        if (et in ("artifact", "natural_object", "inanimate")
                and lem.endswith("o")
                and 3 <= len(lem) <= 12
                and "-" not in lem and " " not in lem
                and lem not in _BAD_OBJECTS):
            out.append(lem)
    return sorted(set(out))


PERSON_NAMES = load_names()
OBJECT_POOL = load_objects()


# ── EO morphology helpers ─────────────────────────────────────────────────

def acc_pl(noun: str) -> str:
    """Accusative plural of a noun ending in -o → +jn (bombono → bombonojn)."""
    return noun + "jn"


def nom_pl(noun: str) -> str:
    """Nominative plural: -o → -oj (bombono → bombonoj)."""
    return noun + "j"


def acc_sg(noun: str) -> str:
    return noun + "n"


# ── RATIO ─────────────────────────────────────────────────────────────────

# Valid ratios. Two- and three-way. All small integers; sums kept ≤ 10
# so totals stay reasonable.
_RATIOS = [
    (1, 2), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5),
    (1, 2, 3), (2, 3, 5), (1, 1, 2), (1, 3, 4),
]

ASK_KINDS = ["direct", "larger", "smaller", "diff", "given-one"]


@dataclass
class Ratio:
    ratio: tuple[int, ...]
    total: int
    item: str             # nominative singular ("bombono")
    names: list[str]      # one per ratio component
    ask: str              # one of ASK_KINDS
    ask_idx: int = 0      # which person to ask about (for "direct")
    given_idx: int = 0    # which person's part was given (for "given-one")

    @property
    def n_parts(self) -> int:
        return sum(self.ratio)

    @property
    def unit(self) -> int:
        return self.total // self.n_parts

    @property
    def parts(self) -> list[int]:
        return [r * self.unit for r in self.ratio]

    @property
    def answer(self) -> int:
        if self.ask == "direct":
            return self.parts[self.ask_idx]
        if self.ask == "larger":
            return max(self.parts)
        if self.ask == "smaller":
            return min(self.parts)
        if self.ask == "diff":
            return max(self.parts) - min(self.parts)
        if self.ask == "given-one":
            # asks: given that person `given_idx` got parts[given_idx], find total
            return self.total
        raise ValueError(self.ask)


def sample_ratio(rng: random.Random) -> Ratio:
    ratio = rng.choice(_RATIOS)
    n_parts = sum(ratio)
    # multiplier so unit is in 2..40, hence total in [4 .. 400]
    multiplier = rng.randint(2, 40)
    total = n_parts * multiplier
    names = rng.sample(PERSON_NAMES, len(ratio))
    item = rng.choice(OBJECT_POOL)
    ask = rng.choice(ASK_KINDS)
    if ask == "diff" and len(ratio) > 2:
        # "diff" only makes sense for two-way ratios
        ask = "direct"
    ask_idx = rng.randint(0, len(ratio) - 1)
    given_idx = rng.randint(0, len(ratio) - 1) if ask == "given-one" else 0
    return Ratio(ratio=ratio, total=total, item=item, names=names,
                 ask=ask, ask_idx=ask_idx, given_idx=given_idx)


# ── Question prose templates ──────────────────────────────────────────────

_RATIO_PHRASES = [
    "en la proporcio {r}",
    "en proporcio {r}",
    "laŭ la proporcio {r}",
    "laŭ proporcio {r}",
    "en rilato {r}",
    "laŭ rilato {r}",
]


def _ratio_str(r: tuple[int, ...]) -> str:
    return ":".join(str(x) for x in r)


def _names_list(names: list[str]) -> str:
    """Anna, Bert kaj Klara — EO list with 'kaj' before last."""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} kaj {names[1]}"
    return ", ".join(names[:-1]) + f" kaj {names[-1]}"


def render_question(p: Ratio, rng: random.Random) -> str:
    """Pick one of several EO surface templates for the given ask."""
    rp = rng.choice(_RATIO_PHRASES).format(r=_ratio_str(p.ratio))
    names_str = _names_list(p.names)
    item_apl = acc_pl(p.item)
    item_npl = nom_pl(p.item)

    if p.ask == "direct":
        whom = p.names[p.ask_idx]
        return rng.choice([
            f"{names_str} dividas {p.total} {item_apl} {rp}. Kiom da {item_npl} ricevas {whom}?",
            f"{names_str} dividis {p.total} {item_apl} {rp}. Kiom da {item_npl} ricevis {whom}?",
            f"{p.total} {item_npl} estas dividitaj inter {names_str} {rp}. Kiom da {item_npl} ricevas {whom}?",
            f"Inter {names_str}, oni dividis {p.total} {item_apl} {rp}. Kiom da {item_npl} ricevas {whom}?",
            f"{names_str} kune havas {p.total} {item_apl}, kiujn ili dividas {rp}. Kiom da {item_npl} estas por {whom}?",
        ])

    if p.ask == "larger":
        return rng.choice([
            f"{names_str} dividas {p.total} {item_apl} {rp}. Kiom da {item_npl} ricevas la persono kun la PLI GRANDA parto?",
            f"{p.total} {item_npl} estas dividitaj inter {names_str} {rp}. Kiom ricevas tiu kun la pli granda parto?",
            f"Dividu {p.total} {item_apl} inter {names_str} {rp}. Kiom da {item_npl} ricevas la pli granda parto?",
        ])

    if p.ask == "smaller":
        return rng.choice([
            f"{names_str} dividas {p.total} {item_apl} {rp}. Kiom da {item_npl} ricevas la persono kun la PLI MALGRANDA parto?",
            f"Inter {names_str}, oni dividas {p.total} {item_apl} {rp}. Kiom ricevas la pli malgranda parto?",
            f"{p.total} {item_npl} estas dividitaj {rp} inter {names_str}. Kiom ricevas tiu kun la malpli granda parto?",
        ])

    if p.ask == "diff":
        a, b = p.names
        return rng.choice([
            f"{a} kaj {b} dividas {p.total} {item_apl} {rp}. Kiom pli da {item_npl} ricevas tiu kun la pli granda parto?",
            f"{p.total} {item_npl} estas dividitaj inter {a} kaj {b} {rp}. Kiom estas la diferenco?",
            f"Dividu {p.total} {item_apl} inter {a} kaj {b} {rp}. Kio estas la diferenco inter iliaj partoj?",
        ])

    if p.ask == "given-one":
        giver = p.names[p.given_idx]
        givers_part = p.parts[p.given_idx]
        return rng.choice([
            f"{names_str} dividis iom da {item_npl} {rp}. {giver} ricevis {givers_part} {item_apl}. Kiom da {item_npl} estis entute?",
            f"En dividado de {item_npl} {rp} inter {names_str}, {giver} ricevis {givers_part}. Kio estas la totalo?",
            f"{giver} ricevis {givers_part} {item_apl} kiam {names_str} dividis ilin {rp}. Kiom estis entute?",
        ])

    raise ValueError(p.ask)


# ── Chain strategies (each walks the solver step by step) ─────────────────

CHAIN_STRATEGIES = ["parts", "algebra", "fraction", "diff"]


def _steps_parts(p: Ratio) -> tuple[list[Step], str]:
    """ni dividu en N partojn → unu parto = total/N → person ricevas r*u"""
    r_sum = sum(p.ratio)
    sum_str = "+".join(str(x) for x in p.ratio)
    u = p.unit
    if p.ask == "given-one":
        r = p.ratio[p.given_idx]
        givers_part = p.parts[p.given_idx]
        steps = [
            Step(pre=f"se {r} partoj egalas {givers_part}, ni trovas unu parton:",
                 expr=f"{givers_part}/{r}", result=str(u),
                 post=f"do unu parto = {u}."),
            Step(pre="la totalo egalas la nombron de partoj fojigitan per unu parto:",
                 expr=f"{r_sum}*{u}", result=str(p.total)),
        ]
        return steps, str(p.total)

    steps = [
        Step(pre="ni dividu laŭ la rilatumo. la sumo de la partoj:",
             expr=sum_str, result=str(r_sum),
             post=f"do {r_sum} partoj entute."),
        Step(pre="unu parto valoras la totalon dividitan per la nombro de partoj:",
             expr=f"{p.total}/{r_sum}", result=str(u),
             post=f"do unu parto = {u}."),
    ]
    if p.ask == "direct":
        whom = p.names[p.ask_idx].lower()
        r = p.ratio[p.ask_idx]
        ans = r * u
        steps.append(Step(
            pre=f"{whom} ricevas {r} partojn:",
            expr=f"{r}*{u}", result=str(ans)))
        return steps, str(ans)
    if p.ask == "larger":
        r = max(p.ratio)
        ans = r * u
        steps.append(Step(pre="la pli granda parto:",
                          expr=f"{r}*{u}", result=str(ans)))
        return steps, str(ans)
    if p.ask == "smaller":
        r = min(p.ratio)
        ans = r * u
        steps.append(Step(pre="la pli malgranda parto:",
                          expr=f"{r}*{u}", result=str(ans)))
        return steps, str(ans)
    if p.ask == "diff":
        big, small = max(p.ratio), min(p.ratio)
        steps.append(Step(pre="la pli granda parto:",
                          expr=f"{big}*{u}", result=str(big*u)))
        steps.append(Step(pre="la pli malgranda parto:",
                          expr=f"{small}*{u}", result=str(small*u)))
        steps.append(Step(pre="diferenco:",
                          expr=f"{big*u}-{small*u}", result=str(big*u - small*u)))
        return steps, str(big*u - small*u)
    raise ValueError(p.ask)


def _steps_algebra(p: Ratio) -> tuple[list[Step], str]:
    """estu x la valoro de unu parto. r1*x + r2*x = total → solve."""
    r_sum = sum(p.ratio)
    u = p.unit
    if p.ask == "given-one":
        r = p.ratio[p.given_idx]
        givers_part = p.parts[p.given_idx]
        steps = [
            Step(pre=f"estu x la valoro de unu parto. laŭ la problemo: {r}x = {givers_part}. ni solvas:",
                 expr=f"{givers_part}/{r}", result=str(u),
                 post=f"do x = {u}."),
            Step(pre="totalo = (sumo de partoj) * x:",
                 expr=f"{r_sum}*{u}", result=str(p.total)),
        ]
        return steps, str(p.total)

    names_clause = " ".join(
        f"{n.lower()} ricevas {r}x," if i < len(p.names) - 1
        else f"{n.lower()} ricevas {r}x."
        for i, (n, r) in enumerate(zip(p.names, p.ratio))
    )
    pre_solve = (
        f"estu x la valoro de unu parto. {names_clause} "
        f"do la totalo: {'+'.join(f'{r}x' for r in p.ratio)} = {p.total}, "
        f"do {r_sum}x = {p.total}. ni solvas por x:"
    )
    steps = [
        Step(pre=pre_solve, expr=f"{p.total}/{r_sum}", result=str(u),
             post=f"do x = {u}."),
    ]
    if p.ask == "direct":
        r = p.ratio[p.ask_idx]
        whom = p.names[p.ask_idx].lower()
        ans = r * u
        steps.append(Step(pre=f"{whom} ricevas {r}x:",
                          expr=f"{r}*{u}", result=str(ans)))
        return steps, str(ans)
    if p.ask == "larger":
        r = max(p.ratio)
        ans = r * u
        steps.append(Step(pre="la pli granda parto = (pli granda koeficiento)*x:",
                          expr=f"{r}*{u}", result=str(ans)))
        return steps, str(ans)
    if p.ask == "smaller":
        r = min(p.ratio)
        ans = r * u
        steps.append(Step(pre="la pli malgranda parto = (pli malgranda koeficiento)*x:",
                          expr=f"{r}*{u}", result=str(ans)))
        return steps, str(ans)
    if p.ask == "diff":
        big, small = max(p.ratio), min(p.ratio)
        ans = (big - small) * u
        steps.append(Step(pre=f"diferenco = ({big}-{small})x = {big-small}x:",
                          expr=f"{big-small}*{u}", result=str(ans)))
        return steps, str(ans)
    raise ValueError(p.ask)


def _steps_fraction(p: Ratio) -> tuple[list[Step], str]:
    """frakcio de la totalo aliro."""
    r_sum = sum(p.ratio)
    sum_str = "+".join(str(x) for x in p.ratio)
    if p.ask == "given-one":
        r = p.ratio[p.given_idx]
        givers_part = p.parts[p.given_idx]
        steps = [
            Step(pre="la sumo de la proporciopartoj:",
                 expr=sum_str, result=str(r_sum),
                 post=f"do la donita parto estas frakcio {r}/{r_sum} de la totalo. "
                      f"do la totalo = (donita parto) * (sumo de partoj) / (donita koeficiento):"),
            Step(expr=f"{givers_part}*{r_sum}/{r}", result=str(p.total)),
        ]
        return steps, str(p.total)

    sum_step = Step(pre="la sumo de la proporciopartoj:",
                    expr=sum_str, result=str(r_sum),
                    post=f"do entute {r_sum} partoj.")
    if p.ask == "direct":
        r = p.ratio[p.ask_idx]
        whom = p.names[p.ask_idx].lower()
        ans = r * p.unit
        steps = [sum_step,
                 Step(pre=f"{whom} ricevas frakcion {r}/{r_sum} de la totalo:",
                      expr=f"{r}/{r_sum}*{p.total}", result=str(ans))]
        return steps, str(ans)
    if p.ask == "larger":
        r = max(p.ratio)
        ans = r * p.unit
        steps = [sum_step,
                 Step(pre=f"la pli granda parto = {r}/{r_sum} de la totalo:",
                      expr=f"{r}/{r_sum}*{p.total}", result=str(ans))]
        return steps, str(ans)
    if p.ask == "smaller":
        r = min(p.ratio)
        ans = r * p.unit
        steps = [sum_step,
                 Step(pre=f"la pli malgranda parto = {r}/{r_sum} de la totalo:",
                      expr=f"{r}/{r_sum}*{p.total}", result=str(ans))]
        return steps, str(ans)
    if p.ask == "diff":
        big, small = max(p.ratio), min(p.ratio)
        ans = (big - small) * p.unit
        steps = [sum_step,
                 Step(pre=f"frakcio-diferenco = ({big}-{small})/{r_sum} = {big-small}/{r_sum} de la totalo:",
                      expr=f"{big-small}/{r_sum}*{p.total}", result=str(ans))]
        return steps, str(ans)
    raise ValueError(p.ask)


def _steps_diff(p: Ratio) -> tuple[list[Step], str]:
    """Compute all parts explicitly, then derive the requested answer."""
    r_sum = sum(p.ratio)
    sum_str = "+".join(str(x) for x in p.ratio)
    u = p.unit
    if p.ask == "given-one":
        # given-one in 'diff' strategy degenerates to the parts approach
        return _steps_parts(p)

    steps = [
        Step(pre="ni dividu laŭ la rilatumo. la sumo de la partoj:",
             expr=sum_str, result=str(r_sum),
             post=f"do {r_sum} partoj entute."),
        Step(pre="unu parto valoras la totalon dividitan per la nombro de partoj:",
             expr=f"{p.total}/{r_sum}", result=str(u),
             post=f"do unu parto = {u}."),
    ]
    parts = p.parts
    for name, r in zip(p.names, p.ratio):
        steps.append(Step(pre=f"{name.lower()} ricevas {r} partojn:",
                          expr=f"{r}*{u}", result=str(r * u)))
    if p.ask == "direct":
        return steps, str(parts[p.ask_idx])
    if p.ask == "larger":
        return steps, str(max(parts))
    if p.ask == "smaller":
        return steps, str(min(parts))
    if p.ask == "diff":
        big, small = max(parts), min(parts)
        steps.append(Step(pre="diferenco:",
                          expr=f"{big}-{small}", result=str(big - small)))
        return steps, str(big - small)
    raise ValueError(p.ask)


_RATIO_STEPS = {
    "parts": _steps_parts,
    "algebra": _steps_algebra,
    "fraction": _steps_fraction,
    "diff": _steps_diff,
}

# Backward-compat: prose-returning dict the diverse generator + TYPES table use.
_RATIO_CHAINS = {
    name: (lambda p, _fn=fn: render_prose(*_fn(p)))
    for name, fn in _RATIO_STEPS.items()
}


def render_chain_steps(p: Ratio, strategy: str) -> tuple[list[Step], str]:
    return _RATIO_STEPS[strategy](p)


def render_chain(p: Ratio, strategy: str) -> str:
    """Backward-compatible prose form (used by diverse generator)."""
    steps, final = render_chain_steps(p, strategy)
    return render_prose(steps, final)


# Master step-table dispatch by problem type. Populated below after all
# per-type _*_STEPS dicts have been defined. Each entry: type_name → strategy → step_fn.
STEPS_BY_TYPE: dict[str, dict[str, callable]] = {}


def render_funcall_for(type_name: str, p, strategy: str) -> list[dict]:
    """Multi-turn funcall messages for any problem type. Use AFTER the
    _*_STEPS dicts are populated below — STEPS_BY_TYPE is wired at module bottom."""
    steps, final = STEPS_BY_TYPE[type_name][strategy](p)
    return render_funcall(steps, final)


def render_chain_funcall(p, strategy: str, type_name: str = "ratio") -> list[dict]:
    """Backward-compat: ratio-only signature still works. For other types,
    pass `type_name`."""
    return render_funcall_for(type_name, p, strategy)


# ══════════════════════════════════════════════════════════════════════════
# PERCENT
# ══════════════════════════════════════════════════════════════════════════

_PCT_PERCENTS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80]
PCT_OPS = ["discount", "markup", "tax", "of-amount", "saving"]


@dataclass
class Percent:
    base: int       # original price/amount
    pct: int        # percent value (integer)
    op: str         # discount | markup | tax | of-amount | saving
    name: str
    item: str       # nominative singular

    @property
    def amount(self) -> int:
        return self.base * self.pct // 100

    @property
    def answer(self) -> int:
        if self.op == "discount":
            return self.base - self.amount
        if self.op == "markup" or self.op == "tax":
            return self.base + self.amount
        if self.op == "of-amount" or self.op == "saving":
            return self.amount
        raise ValueError(self.op)


def sample_percent(rng: random.Random) -> Percent:
    pct = rng.choice(_PCT_PERCENTS)
    # base must satisfy: base * pct % 100 == 0 → base must be a multiple of 100/gcd(pct,100)
    from math import gcd
    step = 100 // gcd(pct, 100)
    # range: 2-50 multiples
    base = step * rng.randint(2, 50)
    op = rng.choice(PCT_OPS)
    name = rng.choice(PERSON_NAMES)
    item = rng.choice(OBJECT_POOL)
    return Percent(base=base, pct=pct, op=op, name=name, item=item)


def render_percent_q(p: Percent, rng: random.Random) -> str:
    item_sg = p.item
    item_acc = acc_sg(item_sg)
    if p.op == "discount":
        return rng.choice([
            f"{p.name} aĉetas {item_acc} kiu kostas {p.base} eŭrojn. La vendejo donas rabaton de {p.pct}%. Kiom kostas la {item_sg} nun?",
            f"{p.name} volas aĉeti {item_acc}. La origina prezo estas {p.base} eŭroj, sed estas rabato de {p.pct}%. Kiom da eŭroj {p.name} pagas?",
            f"En vendejo, {item_sg} kostas {p.base} eŭrojn. Hodiaŭ estas rabato de {p.pct}%. Kiu estas la nova prezo?",
        ])
    if p.op == "markup":
        return rng.choice([
            f"{p.name} havas {item_acc} kiu kostas {p.base} eŭrojn. La prezo pliiĝas je {p.pct}%. Kiom kostas la {item_sg} nun?",
            f"La prezo de {item_sg} estis {p.base} eŭroj, sed pliiĝis je {p.pct}%. Kiu estas la nova prezo?",
            f"{p.name} rimarkis ke la prezo de {item_sg} ŝanĝiĝis. La malnova prezo estis {p.base} eŭroj kaj pliiĝo estas {p.pct}%. Kiom kostas ĝi nun?",
        ])
    if p.op == "tax":
        return rng.choice([
            f"{p.name} aĉetas {item_acc} por {p.base} eŭroj. La imposto estas {p.pct}%. Kiom entute {p.name} pagas?",
            f"{item_sg} kostas {p.base} eŭrojn. Kun {p.pct}% impostoj, kiom estas la totala prezo?",
        ])
    if p.op == "of-amount":
        return rng.choice([
            f"Kiom estas {p.pct}% de {p.base}?",
            f"{p.name} kalkulis {p.pct}% de {p.base} eŭroj. Kiu estas la rezulto?",
            f"En klaso estas {p.base} studentoj. {p.pct}% el ili portas okulvitrojn. Kiom da studentoj portas okulvitrojn?",
        ])
    if p.op == "saving":
        return rng.choice([
            f"{p.name} aĉetis {item_acc} kiu kostis {p.base} eŭrojn kun rabato de {p.pct}%. Kiom da eŭroj {p.name} ŝparis?",
            f"La origina prezo de {item_sg} estas {p.base} eŭroj. Kun {p.pct}% rabato, kiom oni ŝparas?",
        ])
    raise ValueError(p.op)


def _steps_pct_direct(p: Percent) -> tuple[list[Step], str]:
    amt = p.amount
    steps = [Step(pre=f"ni trovas {p.pct}% de {p.base}:",
                  expr=f"{p.pct}/100*{p.base}", result=str(amt),
                  post=f"do {p.pct}% de {p.base} estas {amt}.")]
    if p.op == "discount":
        res = p.base - amt
        steps.append(Step(pre="nova prezo = bazo minus rabato:",
                          expr=f"{p.base}-{amt}", result=str(res)))
        return steps, str(res)
    if p.op in ("markup", "tax"):
        res = p.base + amt
        steps.append(Step(pre="nova prezo = bazo plus aldono:",
                          expr=f"{p.base}+{amt}", result=str(res)))
        return steps, str(res)
    return steps, str(amt)


def _steps_pct_decimal(p: Percent) -> tuple[list[Step], str]:
    amt = p.amount
    dec = p.pct / 100
    dec_str = f"{dec:.2f}".rstrip("0").rstrip(".") if dec else "0"
    steps = [
        Step(pre=f"unue ni transformas {p.pct}% al decimala formo:",
             expr=f"{p.pct}/100", result=dec_str,
             post=f"do {p.pct}% = {dec_str}."),
        Step(pre=f"nun {p.pct}% de {p.base} estas:",
             expr=f"{dec_str}*{p.base}", result=str(amt),
             post=f"do la rezulto estas {amt}."),
    ]
    if p.op == "discount":
        res = p.base - amt
        steps.append(Step(pre="finrezulto = bazo minus rabato:",
                          expr=f"{p.base}-{amt}", result=str(res)))
        return steps, str(res)
    if p.op in ("markup", "tax"):
        res = p.base + amt
        steps.append(Step(pre="finrezulto = bazo plus aldono:",
                          expr=f"{p.base}+{amt}", result=str(res)))
        return steps, str(res)
    return steps, str(amt)


def _steps_pct_multiplier(p: Percent) -> tuple[list[Step], str]:
    if p.op not in ("discount", "markup", "tax"):
        return _steps_pct_direct(p)
    sign = "-" if p.op == "discount" else "+"
    res = p.base + (-1 if p.op == "discount" else 1) * p.amount
    if p.op == "discount":
        mult = 1 - p.pct / 100
    else:
        mult = 1 + p.pct / 100
    mult_str = f"{mult:.2f}".rstrip("0").rstrip(".")
    steps = [
        Step(pre=f"ni trovas la multobligilon (1 {sign} {p.pct}/100):",
             expr=f"1{sign}{p.pct}/100", result=mult_str,
             post=f"do la multobligilo estas {mult_str}."),
        Step(pre="rezulto = bazo fojigita per la multobligilo:",
             expr=f"{p.base}*{mult_str}", result=str(res)),
    ]
    return steps, str(res)


_PCT_STEPS = {
    "direct": _steps_pct_direct,
    "decimal": _steps_pct_decimal,
    "multiplier": _steps_pct_multiplier,
}

_PCT_CHAINS = {
    name: (lambda p, _fn=fn: render_prose(*_fn(p)))
    for name, fn in _PCT_STEPS.items()
}


# ══════════════════════════════════════════════════════════════════════════
# INVERSE-RATE  (workers × time = constant)
# ══════════════════════════════════════════════════════════════════════════

_INV_SCENARIOS = [
    ("laboristoj", "farbas", "muron", "horoj"),
    ("pumpiloj", "plenigas", "naĝejon", "minutoj"),
    ("tuboj", "plenigas", "akvujon", "horoj"),
    ("maŝinoj", "presas", "libron", "horoj"),
    ("rikoltistoj", "rikoltas", "kampon", "tagoj"),
    ("fosistoj", "fosas", "tranĉeon", "tagoj"),
    ("kuiristoj", "preparas", "manĝon", "horoj"),
    ("tajpistoj", "tajpas", "manuskripton", "horoj"),
    ("robotoj", "kunmetas", "aŭton", "horoj"),
    ("ĝardenistoj", "plantas", "arbojn", "tagoj"),
]


@dataclass
class InverseRate:
    w1: int          # initial workers
    t1: int          # initial time
    w2: int          # new workers (divisor of w1*t1)
    scenario: tuple  # (worker_pl, verb, task, time_unit)
    ask: str         # "find-time" | "find-workers"

    @property
    def const(self) -> int:
        return self.w1 * self.t1

    @property
    def t2(self) -> int:
        return self.const // self.w2

    @property
    def answer(self) -> int:
        if self.ask == "find-time":
            return self.t2
        if self.ask == "find-workers":
            return self.w2
        raise ValueError(self.ask)


def sample_inverse_rate(rng: random.Random) -> InverseRate:
    while True:
        w1 = rng.randint(2, 12)
        t1 = rng.randint(2, 24)
        const = w1 * t1
        # find divisors of const that aren't w1
        divs = [d for d in range(1, const + 1) if const % d == 0 and d != w1 and 1 <= d <= 50]
        if not divs:
            continue
        w2 = rng.choice(divs)
        scenario = rng.choice(_INV_SCENARIOS)
        ask = rng.choice(["find-time", "find-workers"])
        return InverseRate(w1=w1, t1=t1, w2=w2, scenario=scenario, ask=ask)


def render_inv_q(p: InverseRate, rng: random.Random) -> str:
    workers, verb, task, unit = p.scenario
    if p.ask == "find-time":
        return rng.choice([
            f"{p.w1} {workers} {verb} {task} en {p.t1} {unit}. Kiom da {unit} bezonas {p.w2} {workers}?",
            f"Se {p.w1} {workers} povas {verb.replace('as','i')} {task} en {p.t1} {unit}, kiom da {unit} bezonas {p.w2} {workers} por la sama tasko?",
            f"{p.w1} {workers} finas {task} en {p.t1} {unit}. Kun {p.w2} {workers}, kiom da {unit} bezonatas?",
        ])
    if p.ask == "find-workers":
        return rng.choice([
            f"{p.w1} {workers} {verb} {task} en {p.t1} {unit}. Kiom da {workers} bezonatas por fini en {p.t2} {unit}?",
            f"Tasko: {verb} {task}. {p.w1} {workers} bezonas {p.t1} {unit}. Kiom da {workers} bezonatas por fini en {p.t2} {unit}?",
        ])
    raise ValueError(p.ask)


def _steps_inv_const(p: InverseRate) -> tuple[list[Step], str]:
    workers, _, _, unit = p.scenario
    steps = [Step(pre="konstanta produkto = laboristoj * tempo:",
                  expr=f"{p.w1}*{p.t1}", result=str(p.const),
                  post=f"do la konstanto estas {p.const} person-{unit}.")]
    if p.ask == "find-time":
        steps.append(Step(
            pre=f"por {p.w2} {workers}, ni solvas {p.w2}*t = {p.const}, do t =",
            expr=f"{p.const}/{p.w2}", result=str(p.t2)))
        return steps, str(p.t2)
    steps.append(Step(
        pre=f"por fini en {p.t2} {unit}, ni solvas w*{p.t2} = {p.const}, do w =",
        expr=f"{p.const}/{p.t2}", result=str(p.w2)))
    return steps, str(p.w2)


def _steps_inv_perunit(p: InverseRate) -> tuple[list[Step], str]:
    workers, _, _, unit = p.scenario
    steps = [Step(pre="totala laboro = laboristoj * tempo:",
                  expr=f"{p.w1}*{p.t1}", result=str(p.const),
                  post=f"do la totala laboro estas {p.const} person-{unit}.")]
    if p.ask == "find-time":
        steps.append(Step(
            pre=f"por {p.w2} {workers}, la bezonata tempo estas:",
            expr=f"{p.const}/{p.w2}", result=str(p.t2)))
        return steps, str(p.t2)
    steps.append(Step(
        pre=f"por fini en {p.t2} {unit}, ni bezonas:",
        expr=f"{p.const}/{p.t2}", result=str(p.w2)))
    return steps, str(p.w2)


def _steps_inv_proportion(p: InverseRate) -> tuple[list[Step], str]:
    if p.ask == "find-time":
        steps = [
            Step(pre=f"inversa proporcio: w1/w2 = t2/t1, do t2 = t1*w1/w2. "
                     f"ni komencu per t1*w1:",
                 expr=f"{p.t1}*{p.w1}", result=str(p.t1*p.w1),
                 post=f"do t1*w1 = {p.t1*p.w1}."),
            Step(pre=f"nun t2 = {p.t1*p.w1}/w2:",
                 expr=f"{p.t1*p.w1}/{p.w2}", result=str(p.t2)),
        ]
        return steps, str(p.t2)
    steps = [
        Step(pre=f"inversa proporcio: w1/w2 = t2/t1, do w2 = w1*t1/t2. "
                 f"ni komencu per w1*t1:",
             expr=f"{p.w1}*{p.t1}", result=str(p.w1*p.t1),
             post=f"do w1*t1 = {p.w1*p.t1}."),
        Step(pre=f"nun w2 = {p.w1*p.t1}/t2:",
             expr=f"{p.w1*p.t1}/{p.t2}", result=str(p.w2)),
    ]
    return steps, str(p.w2)


_INV_STEPS = {
    "constant-product": _steps_inv_const,
    "per-unit": _steps_inv_perunit,
    "inverse-proportion": _steps_inv_proportion,
}

_INV_CHAINS = {
    name: (lambda p, _fn=fn: render_prose(*_fn(p)))
    for name, fn in _INV_STEPS.items()
}


# ══════════════════════════════════════════════════════════════════════════
# CONSECUTIVE INTEGERS
# ══════════════════════════════════════════════════════════════════════════

_CONSEC_SCENARIOS = [
    "paĝnumeroj en libro",
    "jaroj de medaloj",
    "numeroj sur domoj",
    "numeroj de buslinioj",
    "numeroj de loĝejoj",
    "aĝoj de gefratoj",
    "numeroj de seĝoj",
    "numeroj de biletoj",
]


@dataclass
class Consec:
    count: int        # 3, 4, or 5
    start: int        # first integer
    step: int         # 1 for any, 2 for even/odd
    parity: str       # "any" | "even" | "odd"
    name: str
    scenario: str
    ask: str          # "smallest" | "largest" | "middle" | "sum"

    @property
    def values(self) -> list[int]:
        return [self.start + i * self.step for i in range(self.count)]

    @property
    def total(self) -> int:
        return sum(self.values)

    @property
    def answer(self) -> int:
        if self.ask == "smallest":
            return self.values[0]
        if self.ask == "largest":
            return self.values[-1]
        if self.ask == "middle":
            return self.values[self.count // 2]
        if self.ask == "sum":
            return self.total
        raise ValueError(self.ask)


def sample_consec(rng: random.Random) -> Consec:
    count = rng.choice([3, 3, 3, 4, 5])  # bias toward 3
    parity = rng.choice(["any", "any", "even", "odd"])
    step = 1 if parity == "any" else 2
    # pick start so all integers are positive and "small enough"
    if parity == "any":
        start = rng.randint(1, 30)
    elif parity == "even":
        start = rng.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
    else:  # odd
        start = rng.choice([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29])
    ask = rng.choice(["smallest", "largest", "middle"] if count % 2 == 1
                     else ["smallest", "largest"])
    name = rng.choice(PERSON_NAMES)
    scenario = rng.choice(_CONSEC_SCENARIOS)
    return Consec(count=count, start=start, step=step, parity=parity,
                  name=name, scenario=scenario, ask=ask)


def render_consec_q(p: Consec, rng: random.Random) -> str:
    par_adj = {"any": "", "even": "parajn ", "odd": "neparajn "}[p.parity]
    what = {"smallest": "plej malgranda", "largest": "plej granda",
            "middle": "meza", "sum": "sumo"}[p.ask]
    return rng.choice([
        f"{p.name} havas {p.count} sinsekvajn {par_adj}{p.scenario}, kies sumo estas {p.total}. Kio estas la {what}?",
        f"La sumo de {p.count} sinsekvaj {par_adj}entjeroj estas {p.total}. Trovu la {what}n el ili.",
        f"{p.name} rimarkas {p.count} sinsekvajn {par_adj}{p.scenario}. Ilia sumo estas {p.total}. Kiu estas la {what}?",
    ])


def _steps_consec_first(p: Consec) -> tuple[list[Step], str]:
    """estu x → solve for x. The algebra setup is in pre; the solving uses tool calls."""
    offsets = [i * p.step for i in range(p.count)]
    const_sum = sum(offsets)
    offsets_str = "+".join(str(o) for o in offsets) if len(offsets) > 1 else str(offsets[0])
    enum_str = ", ".join(["x"] + [f"x+{o}" for o in offsets[1:]])
    pre0 = (
        f"estu x la unua entjero. la {p.count} entjeroj estas: {enum_str}. "
        f"la sumo de la konstantaj ofsetoj:"
    )
    steps = [
        Step(pre=pre0, expr=offsets_str, result=str(const_sum),
             post=f"do la ofsetoj sumas al {const_sum}."),
        Step(pre=f"do {p.count}x = {p.total} - {const_sum}, do ni trovas {p.count}x:",
             expr=f"{p.total}-{const_sum}", result=str(p.total - const_sum),
             post=f"do {p.count}x = {p.total - const_sum}. ni solvas por x:"),
        Step(expr=f"{p.total - const_sum}/{p.count}", result=str(p.start),
             post=f"do x = {p.start}."),
    ]
    if p.ask == "smallest":
        return steps, str(p.start)
    if p.ask == "largest":
        off = (p.count - 1) * p.step
        steps.append(Step(pre=f"plej granda = x + {off}:",
                          expr=f"{p.start}+{off}", result=str(p.values[-1])))
        return steps, str(p.values[-1])
    if p.ask == "middle":
        off = (p.count // 2) * p.step
        steps.append(Step(pre=f"meza = x + {off}:",
                          expr=f"{p.start}+{off}", result=str(p.values[p.count//2])))
        return steps, str(p.values[p.count//2])
    if p.ask == "sum":
        return steps, str(p.total)
    raise ValueError(p.ask)


def _steps_consec_avg(p: Consec) -> tuple[list[Step], str]:
    if (p.total % p.count != 0) or (p.count % 2 == 0) or (p.parity != "any"):
        return _steps_consec_first(p)
    avg = p.total // p.count
    steps = [
        Step(pre="meznombro = sumo / kalkulo:",
             expr=f"{p.total}/{p.count}", result=str(avg),
             post=f"por {p.count} sinsekvaj entjeroj, la meza estas {avg}. "
                  f"do la entjeroj estas: {', '.join(str(v) for v in p.values)}."),
    ]
    if p.ask == "smallest":
        return steps, str(p.values[0])
    if p.ask == "largest":
        return steps, str(p.values[-1])
    if p.ask == "middle":
        return steps, str(avg)
    if p.ask == "sum":
        return steps, str(p.total)
    raise ValueError(p.ask)


_CONSEC_STEPS = {
    "first-as-x": _steps_consec_first,
    "average": _steps_consec_avg,
}

_CONSEC_CHAINS = {
    name: (lambda p, _fn=fn: render_prose(*_fn(p)))
    for name, fn in _CONSEC_STEPS.items()
}


# ══════════════════════════════════════════════════════════════════════════
# COIN
# ══════════════════════════════════════════════════════════════════════════

# (small_val, big_val, label, unit-currency, scenario noun)
_COIN_DENOMS = [
    (5, 10, "pencaj", "pencoj", "monero"),
    (10, 25, "cendaj", "cendoj", "monero"),
    (1, 2, "eŭraj", "eŭroj", "monero"),
    (5, 10, "eŭraj", "eŭroj", "bileto"),
    (20, 50, "cendaj", "cendoj", "monero"),
    (2, 5, "eŭraj", "eŭroj", "poŝtmarko"),
]


@dataclass
class Coin:
    small_val: int
    big_val: int
    denom_label: str      # adjective form: "pencaj"
    currency: str         # "pencoj"
    item: str             # "monero" | "bileto" | "poŝtmarko"
    total_count: int
    count_big: int        # number of big-value items
    name: str
    ask: str              # "find-big" | "find-small"

    @property
    def count_small(self) -> int:
        return self.total_count - self.count_big

    @property
    def total_value(self) -> int:
        return self.count_small * self.small_val + self.count_big * self.big_val

    @property
    def answer(self) -> int:
        if self.ask == "find-big":
            return self.count_big
        return self.count_small


def sample_coin(rng: random.Random) -> Coin:
    small, big, label, currency, item = rng.choice(_COIN_DENOMS)
    total_count = rng.randint(5, 30)
    count_big = rng.randint(1, total_count - 1)
    name = rng.choice(PERSON_NAMES)
    ask = rng.choice(["find-big", "find-small"])
    return Coin(small_val=small, big_val=big, denom_label=label,
                currency=currency, item=item,
                total_count=total_count, count_big=count_big,
                name=name, ask=ask)


def render_coin_q(p: Coin, rng: random.Random) -> str:
    item_acc_pl = acc_pl(p.item)
    item_npl = nom_pl(p.item)
    target = p.big_val if p.ask == "find-big" else p.small_val
    return rng.choice([
        f"{p.name} havas {p.total_count} {item_acc_pl}, ĉiuj aŭ {p.small_val}-{p.currency[:-1]}-aj aŭ {p.big_val}-{p.currency[:-1]}-aj. La totala valoro estas {p.total_value} {p.currency}. Kiom da {target}-{p.currency[:-1]}-aj {item_npl} havas {p.name}?",
        f"En sia poŝo {p.name} havas {p.total_count} {item_acc_pl} de du valoroj: {p.small_val} kaj {p.big_val} {p.currency}. La sumo estas {p.total_value} {p.currency}. Kiom estas {target}-{p.currency[:-1]}-aj?",
        f"{p.name} kolektis {p.total_count} {item_acc_pl} kun totala valoro {p.total_value} {p.currency}. Iuj valoras {p.small_val} {p.currency} kaj la aliaj {p.big_val} {p.currency}. Kiom da {target}-{p.currency[:-1]}-aj {item_npl} estas?",
    ])


def _steps_coin_subst(p: Coin) -> tuple[list[Step], str]:
    cur = p.currency[:-1]
    if p.ask == "find-big":
        # x = big-count; equation: B*x + S*(N - x) = V → x = (V - S*N)/(B - S)
        SN = p.small_val * p.total_count
        diff = p.total_value - SN
        denom = p.big_val - p.small_val
        steps = [
            Step(pre=f"estu x la nombro de {p.big_val}-{cur}-aj {nom_pl(p.item)}. "
                     f"do ({p.total_count} - x) estas {p.small_val}-{cur}-aj. "
                     f"totala valoro: {p.big_val}x + {p.small_val}*({p.total_count} - x) = {p.total_value}. "
                     f"unue ni kalkulu {p.small_val}*{p.total_count}:",
                 expr=f"{p.small_val}*{p.total_count}", result=str(SN),
                 post=f"do la ekvacio iĝas {p.big_val}x + {SN} - {p.small_val}x = {p.total_value}, "
                      f"do {denom}x = {p.total_value} - {SN}. ni kalkulu la dekstron:"),
            Step(expr=f"{p.total_value}-{SN}", result=str(diff),
                 post=f"do {denom}x = {diff}. ni solvas:"),
            Step(expr=f"{diff}/{denom}", result=str(p.count_big)),
        ]
        return steps, str(p.count_big)
    # find-small
    BN = p.big_val * p.total_count
    diff = BN - p.total_value
    denom = p.big_val - p.small_val
    steps = [
        Step(pre=f"estu x la nombro de {p.small_val}-{cur}-aj {nom_pl(p.item)}. "
                 f"do ({p.total_count} - x) estas {p.big_val}-{cur}-aj. "
                 f"totala valoro: {p.small_val}x + {p.big_val}*({p.total_count} - x) = {p.total_value}. "
                 f"unue ni kalkulu {p.big_val}*{p.total_count}:",
             expr=f"{p.big_val}*{p.total_count}", result=str(BN),
             post=f"do la ekvacio iĝas {p.small_val}x + {BN} - {p.big_val}x = {p.total_value}, "
                  f"do -{denom}x = {p.total_value} - {BN}, do {denom}x = {BN} - {p.total_value}. "
                  f"ni kalkulu la dekstron:"),
        Step(expr=f"{BN}-{p.total_value}", result=str(diff),
             post=f"do {denom}x = {diff}. ni solvas:"),
        Step(expr=f"{diff}/{denom}", result=str(p.count_small)),
    ]
    return steps, str(p.count_small)


def _steps_coin_assume(p: Coin) -> tuple[list[Step], str]:
    cur = p.currency[:-1]
    assumed = p.total_count * p.small_val
    diff = p.total_value - assumed
    step = p.big_val - p.small_val
    big = diff // step
    small = p.total_count - big
    steps = [
        Step(pre=f"se ĉiuj {p.total_count} estus {p.small_val}-{cur}-aj, totalo estus:",
             expr=f"{p.small_val}*{p.total_count}", result=str(assumed),
             post=f"do la supozita totalo estas {assumed}."),
        Step(pre=f"sed la vera totalo estas {p.total_value}, do la manko estas:",
             expr=f"{p.total_value}-{assumed}", result=str(diff),
             post=f"do manko = {diff}."),
        Step(pre=f"ĉiu {p.big_val}-{cur}-a anstataŭ {p.small_val}-{cur}-a aldonas:",
             expr=f"{p.big_val}-{p.small_val}", result=str(step),
             post=f"do ĉiu anstataŭigo aldonas {step}."),
        Step(pre=f"nombro de {p.big_val}-{cur}-aj = manko / pliigo:",
             expr=f"{diff}/{step}", result=str(big)),
    ]
    if p.ask == "find-big":
        return steps, str(big)
    steps.append(Step(pre=f"nombro de {p.small_val}-{cur}-aj = totala kalkulo - {p.big_val}-{cur}-aj:",
                      expr=f"{p.total_count}-{big}", result=str(small)))
    return steps, str(small)


_COIN_STEPS = {
    "substitution": _steps_coin_subst,
    "assume-then-correct": _steps_coin_assume,
}

_COIN_CHAINS = {
    name: (lambda p, _fn=fn: render_prose(*_fn(p)))
    for name, fn in _COIN_STEPS.items()
}


# ══════════════════════════════════════════════════════════════════════════
# AGE
# ══════════════════════════════════════════════════════════════════════════

_AGE_RELATIONS = [
    ("patrino", "filino"), ("patro", "filo"), ("avo", "nepo"),
    ("avino", "nepino"), ("onklo", "nevo"), ("onklino", "nevino"),
    ("instruisto", "studento"), ("frato", "fratino"),
    ("mentoro", "lernanto"),
]


@dataclass
class Age:
    young: int          # young person's current age
    old: int            # old person's current age
    ratio_now: int      # old = ratio_now * young
    relation: tuple     # (old_role, young_role)
    name_young: str
    name_old: str
    kind: str           # "simple-now" | "time-shift"
    sum_now: int = 0    # for simple-now: given
    t: int = 0          # time shift (years)
    ratio_later: int = 0  # for time-shift
    ask: str = "young"  # "young" | "old" | "future"
    ask_t: int = 0      # years for "future" ask

    @property
    def answer(self) -> int:
        if self.ask == "young":
            return self.young
        if self.ask == "old":
            return self.old
        if self.ask == "future":
            target = self.young if self.t == 0 else self.young  # default to young
            return self.young + self.ask_t
        raise ValueError(self.ask)


_AGE_TIME_SHIFT_CFGS = [
    # (r_now, t, r_later) — must satisfy r_now > r_later AND
    # t*(r_later-1) divisible by (r_now-r_later) for integer ages.
    (rn, t, rl) for (rn, t, rl) in [
        (3, 10, 2), (5, 4, 3), (4, 10, 3), (7, 6, 4), (4, 6, 2),
        (6, 10, 4), (3, 6, 2), (5, 8, 3), (4, 9, 3), (5, 6, 4),
        (6, 5, 5), (3, 4, 2), (4, 12, 2),
    ] if rn > rl and (t * (rl - 1)) % (rn - rl) == 0
]


def sample_age(rng: random.Random) -> Age:
    relation = rng.choice(_AGE_RELATIONS)
    names = rng.sample(PERSON_NAMES, 2)
    kind = rng.choice(["simple-now", "simple-now", "time-shift"])  # bias simple
    if kind == "simple-now":
        ratio = rng.choice([2, 3, 4, 5])
        young = rng.randint(4, 20)
        old = ratio * young
        sum_now = young + old
        ask = rng.choice(["young", "old", "future"])
        ask_t = rng.choice([3, 5, 10]) if ask == "future" else 0
        return Age(young=young, old=old, ratio_now=ratio, relation=relation,
                   name_young=names[1], name_old=names[0], kind=kind,
                   sum_now=sum_now, ask=ask, ask_t=ask_t)
    else:  # time-shift
        rn, t, rl = rng.choice(_AGE_TIME_SHIFT_CFGS)
        young = t * (rl - 1) // (rn - rl)
        old = rn * young
        return Age(young=young, old=old, ratio_now=rn, relation=relation,
                   name_young=names[1], name_old=names[0], kind=kind,
                   t=t, ratio_later=rl, ask="young")


def render_age_q(p: Age, rng: random.Random) -> str:
    old_role, young_role = p.relation
    ny, no = p.name_young, p.name_old
    if p.kind == "simple-now":
        mul_word = {2: "dufoje", 3: "trifoje", 4: "kvarfoje", 5: "kvinfoje"}[p.ratio_now]
        if p.ask == "young":
            return rng.choice([
                f"{no} estas {p.ratio_now}-foje pli aĝa ol {ny}. Kune ili havas {p.sum_now} jarojn. Kiom da jaroj havas {ny}?",
                f"{no} kaj {ny} estas {old_role} kaj {young_role}. {no} estas {mul_word} pli aĝa, kaj iliaj aĝoj sumiĝas al {p.sum_now}. Kiom aĝa estas {ny}?",
            ])
        if p.ask == "old":
            return rng.choice([
                f"{no} estas {mul_word} pli aĝa ol {ny}. Kune ili havas {p.sum_now} jarojn. Kiom da jaroj havas {no}?",
                f"La aĝo de {no} estas {p.ratio_now}-foje tiu de {ny}, kaj kune ili havas {p.sum_now} jarojn. Kiom aĝa estas {no}?",
            ])
        if p.ask == "future":
            return rng.choice([
                f"{no} estas {mul_word} pli aĝa ol {ny}, kaj iliaj aĝoj sumiĝas al {p.sum_now}. Kiom aĝa estos {ny} post {p.ask_t} jaroj?",
            ])
    else:  # time-shift
        mul_now = {2:"dufoje",3:"trifoje",4:"kvarfoje",5:"kvinfoje",6:"sesfoje",7:"sepfoje"}[p.ratio_now]
        mul_later = {2:"dufoje",3:"trifoje",4:"kvarfoje",5:"kvinfoje"}[p.ratio_later]
        return rng.choice([
            f"Nun {no} estas {mul_now} pli aĝa ol {ny}. Post {p.t} jaroj, {no} estos {mul_later} pli aĝa ol {ny}. Kiom da jaroj havas {ny} nun?",
            f"{no} kaj {ny} estas {old_role} kaj {young_role}. Hodiaŭ {no} estas {p.ratio_now}-foje pli aĝa, sed post {p.t} jaroj nur {p.ratio_later}-foje. Kio estas la nuna aĝo de {ny}?",
        ])
    raise ValueError(p.ask)


def _steps_age_simple(p: Age) -> tuple[list[Step], str]:
    ny = p.name_young.lower()
    coef = p.ratio_now + 1
    setup = (f"estu x la aĝo de {ny}. la maljuna persono havas {p.ratio_now}*x jarojn. "
             f"sumo: x + {p.ratio_now}*x = {p.sum_now}, do {coef}x = {p.sum_now}. ni solvas por x:")
    steps = [Step(pre=setup, expr=f"{p.sum_now}/{coef}", result=str(p.young),
                  post=f"do x = {p.young}.")]
    if p.ask == "young":
        return steps, str(p.young)
    if p.ask == "old":
        steps.append(Step(pre=f"la maljuna persono = {p.ratio_now}*x:",
                          expr=f"{p.ratio_now}*{p.young}", result=str(p.old)))
        return steps, str(p.old)
    if p.ask == "future":
        ans = p.young + p.ask_t
        steps.append(Step(pre=f"post {p.ask_t} jaroj:",
                          expr=f"{p.young}+{p.ask_t}", result=str(ans)))
        return steps, str(ans)
    raise ValueError(p.ask)


def _steps_age_time_shift(p: Age) -> tuple[list[Step], str]:
    ny = p.name_young.lower()
    diff_coef = p.ratio_now - p.ratio_later
    diff_const = p.ratio_later * p.t - p.t
    setup = (
        f"estu d la aĝo de {ny} nun. la maljuna persono havas {p.ratio_now}*d jarojn. "
        f"post {p.t} jaroj: juna estos d+{p.t}, maljuna estos {p.ratio_now}d+{p.t}, "
        f"kaj maljuna estos {p.ratio_later} foje juna. el la ekvacio: "
        f"({p.ratio_now}-{p.ratio_later})d = ({p.ratio_later}*{p.t}-{p.t}). "
        f"diferenco de koeficientoj:"
    )
    steps = [
        Step(pre=setup,
             expr=f"{p.ratio_now}-{p.ratio_later}", result=str(diff_coef),
             post=f"do la koeficiento estas {diff_coef}. diferenco de konstantoj:"),
        Step(expr=f"{p.ratio_later}*{p.t}-{p.t}", result=str(diff_const),
             post=f"do la konstanto estas {diff_const}. ni solvas por d:"),
        Step(expr=f"{diff_const}/{diff_coef}", result=str(p.young)),
    ]
    return steps, str(p.young)


_AGE_STEPS = {
    "simple-now": _steps_age_simple,
    "time-shift": _steps_age_time_shift,
}

_AGE_CHAINS = {
    name: (lambda p, _fn=fn: render_prose(*_fn(p)))
    for name, fn in _AGE_STEPS.items()
}


# ══════════════════════════════════════════════════════════════════════════
# MIXTURE
# ══════════════════════════════════════════════════════════════════════════

_MIX_SCENARIOS = [
    ("salakvo", "salo"), ("sukerakvo", "sukero"),
    ("acida solvaĵo", "acido"), ("alkohola solvaĵo", "alkoholo"),
    ("kafosolvaĵo", "kafopulvoro"), ("frostiga solvaĵo", "antifrosto"),
    ("fertilizilo", "nutraĵo"),
]


@dataclass
class Mixture:
    kind: str         # "dilute" | "concentrate" | "blend"
    name: str
    sol_name: str     # "salakvo"
    solute: str       # "salo"
    # dilute/concentrate:
    v1: int = 0
    p1: int = 0
    p2: int = 0
    add: int = 0      # ml of water (dilute) or g of solute (concentrate)
    # blend:
    v2: int = 0
    p2_blend: int = 0
    p_avg: int = 0

    @property
    def answer(self) -> int:
        if self.kind in ("dilute", "concentrate"):
            return self.add
        if self.kind == "blend":
            return self.p_avg
        raise ValueError(self.kind)


def sample_mixture(rng: random.Random) -> Mixture:
    sol_name, solute = rng.choice(_MIX_SCENARIOS)
    name = rng.choice(PERSON_NAMES)
    kind = rng.choice(["dilute", "concentrate", "blend"])
    if kind == "dilute":
        # find (v1, p1, p2) such that v1 * p1 / p2 - v1 is positive integer
        # AND v1*p1 % 100 == 0 so per-side chain step is integer
        for _ in range(50):
            p1 = rng.choice([10, 15, 20, 25, 30, 40, 50])
            p2 = rng.choice([5, 8, 10, 15, 20])
            if p2 >= p1:
                continue
            v1 = rng.choice([100, 150, 200, 250, 300, 400, 500])
            if (v1 * p1) % 100 != 0:
                continue
            if (v1 * p1) % p2 == 0:
                add = v1 * p1 // p2 - v1
                if add > 0:
                    return Mixture(kind="dilute", name=name, sol_name=sol_name,
                                   solute=solute, v1=v1, p1=p1, p2=p2, add=add)
        # fallback: known good
        return Mixture(kind="dilute", name=name, sol_name=sol_name, solute=solute,
                       v1=200, p1=10, p2=5, add=200)
    if kind == "concentrate":
        # X = V*(P2-P1)/(100-P2)
        for _ in range(50):
            p1 = rng.choice([5, 8, 10, 15, 20])
            p2 = rng.choice([20, 25, 30, 40, 50])
            if p2 <= p1 or p2 >= 100:
                continue
            v = rng.choice([100, 150, 200, 300, 400])
            if (v * p1) % 100 != 0:  # per-side chain step must be integer
                continue
            num = v * (p2 - p1)
            den = 100 - p2
            if num % den == 0:
                add = num // den
                if add > 0:
                    return Mixture(kind="concentrate", name=name, sol_name=sol_name,
                                   solute=solute, v1=v, p1=p1, p2=p2, add=add)
        return Mixture(kind="concentrate", name=name, sol_name=sol_name, solute=solute,
                       v1=200, p1=10, p2=20, add=25)
    # blend
    for _ in range(50):
        p1 = rng.choice([10, 15, 20, 25, 30, 40, 50])
        p2 = rng.choice([5, 10, 15, 20])
        if p1 == p2:
            continue
        v1 = rng.choice([50, 100, 150, 200, 300])
        v2 = rng.choice([50, 100, 150, 200, 300])
        # Require each side's solute amount to be a whole number too,
        # so the chain steps "v1 * p1 / 100" eval cleanly.
        if (v1 * p1) % 100 or (v2 * p2) % 100:
            continue
        num = v1 * p1 + v2 * p2
        den = v1 + v2
        if num % den == 0:
            p_avg = num // den
            return Mixture(kind="blend", name=name, sol_name=sol_name, solute=solute,
                           v1=v1, p1=p1, v2=v2, p2_blend=p2, p_avg=p_avg)
    return Mixture(kind="blend", name=name, sol_name=sol_name, solute=solute,
                   v1=100, p1=20, v2=100, p2_blend=10, p_avg=15)


def render_mix_q(p: Mixture, rng: random.Random) -> str:
    if p.kind == "dilute":
        return rng.choice([
            f"{p.name} havas {p.v1} ml da {p.sol_name} kun koncentriĝo de {p.p1}%. Kiom da pura akvo {p.name} devas aldoni por atingi koncentriĝon de {p.p2}%?",
            f"En glaso estas {p.v1} ml de {p.p1}-procenta {p.sol_name}. {p.name} volas dilui ĝin al {p.p2}%. Kiom da akvo aldoni?",
        ])
    if p.kind == "concentrate":
        return rng.choice([
            f"{p.name} havas {p.v1} ml da {p.sol_name} kun koncentriĝo de {p.p1}%. Li volas plialtigi la koncentriĝon al {p.p2}% per aldono de pura {p.solute}. Kiom da {p.solute} li devas aldoni?",
            f"En {p.v1} ml de {p.p1}-procenta {p.sol_name}, {p.name} aldonas puran {p.solute} ĝis la koncentriĝo iĝas {p.p2}%. Kiom da {p.solute} li aldonis?",
        ])
    # blend
    return rng.choice([
        f"{p.name} miksas {p.v1} ml de {p.p1}-procenta {p.sol_name} kun {p.v2} ml de {p.p2_blend}-procenta {p.sol_name}. Kio estos la fina procento de la miksaĵo?",
        f"Du solvaĵoj estas miksitaj: {p.v1} ml de {p.p1}% kaj {p.v2} ml de {p.p2_blend}%. Kio estas la koncentriĝo de la rezulto?",
    ])


def _steps_mix_dilute(p: Mixture) -> tuple[list[Step], str]:
    solute_amt = p.v1 * p.p1 // 100
    v_final = p.v1 + p.add
    steps = [
        Step(pre=f"kvanto de {p.solute} = volumo * koncentriĝo / 100:",
             expr=f"{p.v1}*{p.p1}/100", result=str(solute_amt),
             post=f"do estas {solute_amt} ml de {p.solute}."),
        Step(pre=f"la {p.solute}-kvanto restos sama. fina koncentriĝo {p.p2}% donas finan volumon:",
             expr=f"{solute_amt}*100/{p.p2}", result=str(v_final),
             post=f"do la fina volumo estas {v_final} ml."),
        Step(pre="aldonenda akvo = fina volumo - nuna volumo:",
             expr=f"{v_final}-{p.v1}", result=str(p.add)),
    ]
    return steps, str(p.add)


def _steps_mix_concentrate(p: Mixture) -> tuple[list[Step], str]:
    solute_amt = p.v1 * p.p1 // 100
    rhs = p.p2 * p.v1 - 100 * solute_amt
    denom = 100 - p.p2
    steps = [
        Step(pre=f"nuna {p.solute}: {p.v1} * {p.p1} / 100:",
             expr=f"{p.v1}*{p.p1}/100", result=str(solute_amt),
             post=f"do estas {solute_amt} ml de {p.solute}. ni aldonas X ml puran {p.solute}. "
                  f"el la ekvacio ({solute_amt}+X)/({p.v1}+X) = {p.p2}/100, "
                  f"ni ekspandas al {100-p.p2}X = {p.p2}*{p.v1} - 100*{solute_amt}. "
                  f"ni kalkulu la dekstron:"),
        Step(expr=f"{p.p2}*{p.v1}-100*{solute_amt}", result=str(rhs),
             post=f"do {denom}X = {rhs}. ni solvas por X:"),
        Step(expr=f"{rhs}/{denom}", result=str(p.add)),
    ]
    return steps, str(p.add)


def _steps_mix_blend(p: Mixture) -> tuple[list[Step], str]:
    s1 = p.v1 * p.p1 // 100
    s2 = p.v2 * p.p2_blend // 100
    s_total = s1 + s2
    v_total = p.v1 + p.v2
    steps = [
        Step(pre=f"{p.solute} en unua parto = volumo * koncentriĝo / 100:",
             expr=f"{p.v1}*{p.p1}/100", result=str(s1),
             post=f"do unua parto enhavas {s1} ml {p.solute}."),
        Step(pre=f"{p.solute} en dua parto:",
             expr=f"{p.v2}*{p.p2_blend}/100", result=str(s2),
             post=f"do dua parto enhavas {s2} ml {p.solute}."),
        Step(pre=f"totala {p.solute}:",
             expr=f"{s1}+{s2}", result=str(s_total),
             post=f"do entute {s_total} ml {p.solute}."),
        Step(pre="totala volumo:",
             expr=f"{p.v1}+{p.v2}", result=str(v_total),
             post=f"do entute {v_total} ml. finita koncentriĝo = (totala {p.solute}) / (totala volumo) * 100:"),
        Step(expr=f"{s_total}/{v_total}*100", result=str(p.p_avg)),
    ]
    return steps, str(p.p_avg)


_MIX_STEPS = {
    "dilute": _steps_mix_dilute,
    "concentrate": _steps_mix_concentrate,
    "blend": _steps_mix_blend,
}

_MIX_CHAINS = {
    name: (lambda p, _fn=fn: render_prose(*_fn(p)))
    for name, fn in _MIX_STEPS.items()
}


# ══════════════════════════════════════════════════════════════════════════
# DISTANCE / RATE / TIME
# ══════════════════════════════════════════════════════════════════════════

_DIST_VEHICLES = ["aŭto", "biciklo", "trajno", "motorciklo", "kamiono",
                  "buso", "ŝipo", "boato"]


@dataclass
class Distance:
    kind: str          # "direct" | "catch-up" | "meeting" | "average"
    name: str
    vehicle: str
    # direct: D = R * T
    d: int = 0
    r: int = 0
    t: int = 0
    ask: str = "d"     # for direct: which of d/r/t to ask
    # catch-up: A at ra for h hours, B at rb catches up
    ra: int = 0
    rb: int = 0
    h: int = 0
    catch_t: int = 0
    # meeting: r1 + r2 toward each other over D
    r1: int = 0
    r2: int = 0
    meet_d: int = 0
    meet_t: int = 0
    # average: round-trip r1 out, r2 back
    rout: int = 0
    rback: int = 0
    ravg: int = 0
    name2: str = ""

    @property
    def answer(self) -> int:
        if self.kind == "direct":
            return {"d": self.d, "r": self.r, "t": self.t}[self.ask]
        if self.kind == "catch-up":
            return self.catch_t
        if self.kind == "meeting":
            return self.meet_t
        if self.kind == "average":
            return self.ravg
        raise ValueError(self.kind)


def sample_distance(rng: random.Random) -> Distance:
    name = rng.choice(PERSON_NAMES)
    vehicle = rng.choice(_DIST_VEHICLES)
    kind = rng.choice(["direct", "catch-up", "meeting", "average"])
    if kind == "direct":
        r = rng.choice([40, 50, 60, 70, 75, 80, 90, 100, 120])
        t = rng.randint(2, 8)
        d = r * t
        ask = rng.choice(["d", "r", "t"])
        return Distance(kind=kind, name=name, vehicle=vehicle, d=d, r=r, t=t, ask=ask)
    if kind == "catch-up":
        for _ in range(50):
            ra = rng.choice([40, 50, 60, 70, 75, 80])
            rb = ra + rng.choice([20, 30, 40, 50])
            h = rng.randint(2, 6)
            num = ra * h
            den = rb - ra
            if num % den == 0:
                ct = num // den
                if 1 <= ct <= 12:
                    name2 = rng.choice([n for n in PERSON_NAMES if n != name])
                    return Distance(kind=kind, name=name, vehicle=vehicle,
                                    ra=ra, rb=rb, h=h, catch_t=ct, name2=name2)
        return Distance(kind=kind, name=name, vehicle=vehicle,
                        ra=80, rb=120, h=2, catch_t=4, name2="Petro")
    if kind == "meeting":
        for _ in range(30):
            r1 = rng.choice([30, 40, 50, 60, 70])
            r2 = rng.choice([40, 50, 60, 80, 90, 120])
            if r1 == r2:
                continue
            t = rng.randint(2, 6)
            d = (r1 + r2) * t
            name2 = rng.choice([n for n in PERSON_NAMES if n != name])
            return Distance(kind=kind, name=name, vehicle=vehicle,
                            r1=r1, r2=r2, meet_d=d, meet_t=t, name2=name2)
    # average (harmonic mean)
    for _ in range(50):
        rout = rng.choice([40, 50, 60, 75, 80, 90, 120])
        rback = rng.choice([30, 40, 50, 60, 75, 80])
        if rout == rback:
            continue
        num = 2 * rout * rback
        den = rout + rback
        if num % den == 0:
            return Distance(kind="average", name=name, vehicle=vehicle,
                            rout=rout, rback=rback, ravg=num // den)
    return Distance(kind="average", name=name, vehicle=vehicle,
                    rout=60, rback=40, ravg=48)


def render_dist_q(p: Distance, rng: random.Random) -> str:
    if p.kind == "direct":
        if p.ask == "d":
            return rng.choice([
                f"{p.name} veturas per {acc_sg(p.vehicle)} kun rapideco de {p.r} km/h. Se {p.name} veturas dum {p.t} horoj, kiom da kilometroj {p.name} kovras?",
                f"Per sia {p.vehicle}, {p.name} veturas {p.t} horojn je {p.r} km/h. Kiu estas la distanco?",
            ])
        if p.ask == "r":
            return rng.choice([
                f"{p.name} veturas {p.d} km per {acc_sg(p.vehicle)} en {p.t} horoj. Kiu estas la rapideco?",
                f"En {p.t} horoj, {p.name}'s {p.vehicle} kovras {p.d} km. Kiu estas la rapideco?",
            ])
        if p.ask == "t":
            return rng.choice([
                f"{p.name} veturas per {acc_sg(p.vehicle)} je {p.r} km/h. Kiom da horoj necesas por kovri {p.d} km?",
                f"Per {p.vehicle} je {p.r} km/h, kiom da horoj bezonas {p.name} por veturi {p.d} km?",
            ])
    if p.kind == "catch-up":
        return rng.choice([
            f"{p.name} ekveturis per sia {p.vehicle} kun rapideco de {p.ra} km/h. Post {p.h} horoj, {p.name2} ekiris de la sama loko en la sama direkto kun rapideco de {p.rb} km/h. Post kiom da horoj de la ekveturo de {p.name2}, li atingos {p.name}n?",
            f"{p.name} forveturis je {p.ra} km/h. {p.h} horojn poste, {p.name2} ekiris persekutante je {p.rb} km/h. Kiom da horoj post la ekveturo de {p.name2}?",
        ])
    if p.kind == "meeting":
        return rng.choice([
            f"{p.name} stiras sian {p.vehicle} de urbo A al B, dum {p.name2} stiras sian {p.vehicle} de B al A. La distanco inter la urboj estas {p.meet_d} km. Ili ekiras samtempe je {p.r1} km/h kaj {p.r2} km/h respektive. Post kiom da horoj ili renkontiĝos?",
            f"Du veturiloj ekiras samtempe de du urboj distancaj je {p.meet_d} km. Unu veturas je {p.r1} km/h, la alia je {p.r2} km/h en la kontraŭa direkto. Kiam ili renkontiĝos?",
        ])
    # average
    return rng.choice([
        f"{p.name} veturas per sia {p.vehicle} de urbo A al urbo B je rapideco de {p.rout} km/h. Por reveni de B al A, {p.name} veturas je rapideco de {p.rback} km/h. Kiun mezuman rapidecon {p.name} atingis por la tuta rondiro?",
        f"En rondiro, {p.name} iras je {p.rout} km/h kaj revenas je {p.rback} km/h. Kio estas la mezuma rapideco?",
    ])


def _steps_dist_direct(p: Distance) -> tuple[list[Step], str]:
    if p.ask == "d":
        steps = [Step(pre="distanco = rapideco * tempo:",
                      expr=f"{p.r}*{p.t}", result=str(p.d))]
        return steps, str(p.d)
    if p.ask == "r":
        steps = [Step(pre="rapideco = distanco / tempo:",
                      expr=f"{p.d}/{p.t}", result=str(p.r))]
        return steps, str(p.r)
    steps = [Step(pre="tempo = distanco / rapideco:",
                  expr=f"{p.d}/{p.r}", result=str(p.t))]
    return steps, str(p.t)


def _steps_dist_catchup(p: Distance) -> tuple[list[Step], str]:
    """{rb}t = {ra}*(t+h) → t = ra*h / (rb-ra)."""
    numer = p.ra * p.h
    denom = p.rb - p.ra
    setup = (
        f"distanco de {p.name.lower()} post t horoj de la ekveturo de {p.name2.lower()}: "
        f"{p.ra}*(t+{p.h}). distanco de {p.name2.lower()} post t horoj: {p.rb}*t. "
        f"egalu: {p.rb}t = {p.ra}*(t+{p.h}), do {p.rb}t = {p.ra}t + {p.ra}*{p.h}, "
        f"do ({p.rb}-{p.ra})t = {p.ra}*{p.h}. unue ni kalkulu la dekstron:"
    )
    steps = [
        Step(pre=setup, expr=f"{p.ra}*{p.h}", result=str(numer),
             post=f"do {denom}t = {numer}. ni solvas por t:"),
        Step(expr=f"{numer}/{denom}", result=str(p.catch_t)),
    ]
    return steps, str(p.catch_t)


def _steps_dist_meeting(p: Distance) -> tuple[list[Step], str]:
    r_sum = p.r1 + p.r2
    steps = [
        Step(pre="sumo de rapidecoj (ili moviĝas unu al la alia):",
             expr=f"{p.r1}+{p.r2}", result=str(r_sum),
             post=f"do la kombina rapideco estas {r_sum} km/h. "
                  f"tempo = distanco / kombina rapideco:"),
        Step(expr=f"{p.meet_d}/{r_sum}", result=str(p.meet_t)),
    ]
    return steps, str(p.meet_t)


def _steps_dist_avg(p: Distance) -> tuple[list[Step], str]:
    numer = 2 * p.rout * p.rback
    denom = p.rout + p.rback
    steps = [
        Step(pre="mezuma rapideco por rondiro = 2*r1*r2/(r1+r2). unue 2*r1*r2:",
             expr=f"2*{p.rout}*{p.rback}", result=str(numer),
             post=f"do la nombritoro estas {numer}. nun r1+r2:"),
        Step(expr=f"{p.rout}+{p.rback}", result=str(denom),
             post=f"do la denominatoro estas {denom}. ni dividas:"),
        Step(expr=f"{numer}/{denom}", result=str(p.ravg)),
    ]
    return steps, str(p.ravg)


_DIST_STEPS = {
    "direct": _steps_dist_direct,
    "catch-up": _steps_dist_catchup,
    "meeting": _steps_dist_meeting,
    "average": _steps_dist_avg,
}

_DIST_CHAINS = {
    name: (lambda p, _fn=fn: render_prose(*_fn(p)))
    for name, fn in _DIST_STEPS.items()
}


# ══════════════════════════════════════════════════════════════════════════
# EQUATION-SOLVE  (bare "Solvu: ax + b = c")
# ══════════════════════════════════════════════════════════════════════════
# Covers the failure mode: probe shows v6 solves the same equation in word form
# (when it matches a trained template) but fails in bare "Solvu: …" form.
# We add the bare-equation shape so the equation-solving subroutine has its own
# direct trigger.

@dataclass
class EquationSolve:
    a: int          # coefficient of x
    b: int          # constant term, can be negative
    c: int          # RHS
    style: str      # "ax+b=c" | "ax-b=c" | "a(x+b)=c" | "a(x-b)=c" | "x/a+b=c" | "x/a=c"

    @property
    def answer(self) -> int:
        if self.style == "ax+b=c":   return (self.c - self.b) // self.a
        if self.style == "ax-b=c":   return (self.c + self.b) // self.a
        if self.style == "a(x+b)=c": return self.c // self.a - self.b
        if self.style == "a(x-b)=c": return self.c // self.a + self.b
        if self.style == "x/a+b=c":  return (self.c - self.b) * self.a
        if self.style == "x/a=c":    return self.c * self.a
        raise ValueError(self.style)


def sample_equation_solve(rng: random.Random) -> EquationSolve:
    style = rng.choice(["ax+b=c", "ax-b=c", "a(x+b)=c", "a(x-b)=c",
                        "x/a+b=c", "x/a=c"])
    for _ in range(50):
        if style in ("ax+b=c", "ax-b=c"):
            a = rng.choice([2, 3, 4, 5, 6, 7, 8])
            x = rng.randint(-10, 20)
            b = rng.randint(1, 30)
            c = a * x + (b if style == "ax+b=c" else -b)
            inst = EquationSolve(a=a, b=b, c=c, style=style)
        elif style in ("a(x+b)=c", "a(x-b)=c"):
            a = rng.choice([2, 3, 4, 5])
            x = rng.randint(-5, 15)
            b = rng.randint(1, 12)
            c = a * (x + (b if style == "a(x+b)=c" else -b))
            inst = EquationSolve(a=a, b=b, c=c, style=style)
        elif style == "x/a+b=c":
            a = rng.choice([2, 3, 4, 5])
            x = rng.randint(1, 12) * a  # multiple of a so x/a is integer
            b = rng.randint(1, 20)
            c = x // a + b
            inst = EquationSolve(a=a, b=b, c=c, style=style)
        else:  # x/a=c
            a = rng.choice([2, 3, 4, 5, 6, 8])
            x = rng.randint(1, 15) * a
            inst = EquationSolve(a=a, b=0, c=x // a, style=style)
        # sanity: answer is integer and reasonable
        ans = inst.answer
        if -50 <= ans <= 100:
            return inst
    # fallback
    return EquationSolve(a=2, b=5, c=11, style="ax+b=c")


def _eq_expr(p: EquationSolve) -> str:
    """The equation as it appears in the prompt."""
    if p.style == "ax+b=c":   return f"{p.a}x + {p.b} = {p.c}"
    if p.style == "ax-b=c":   return f"{p.a}x - {p.b} = {p.c}"
    if p.style == "a(x+b)=c": return f"{p.a}(x + {p.b}) = {p.c}"
    if p.style == "a(x-b)=c": return f"{p.a}(x - {p.b}) = {p.c}"
    if p.style == "x/a+b=c":  return f"x / {p.a} + {p.b} = {p.c}"
    if p.style == "x/a=c":    return f"x / {p.a} = {p.c}"
    raise ValueError(p.style)


def render_eq_q(p: EquationSolve, rng: random.Random) -> str:
    expr = _eq_expr(p)
    return rng.choice([
        f"Solvu: {expr}",
        f"Solvu por x: {expr}",
        f"Trovu la valoron de x: {expr}",
        f"Kio estas x se {expr}?",
        f"Solvu la ekvacion: {expr}",
    ])


def _steps_eq_isolate(p: EquationSolve) -> tuple[list[Step], str]:
    """Standard isolate-then-divide chain."""
    ans = p.answer
    if p.style == "ax+b=c":
        diff = p.c - p.b
        steps = [
            Step(pre=f"subtrahi {p.b} de ambaŭ flankoj. dekstra flanko:",
                 expr=f"{p.c}-{p.b}", result=str(diff),
                 post=f"do {p.a}-foje x egalas al {diff}. dividi per {p.a}:"),
            Step(expr=f"{diff}/{p.a}", result=str(ans)),
        ]
    elif p.style == "ax-b=c":
        plus = p.c + p.b
        steps = [
            Step(pre=f"aldoni {p.b} al ambaŭ flankoj. dekstra flanko:",
                 expr=f"{p.c}+{p.b}", result=str(plus),
                 post=f"do {p.a}-foje x egalas al {plus}. dividi per {p.a}:"),
            Step(expr=f"{plus}/{p.a}", result=str(ans)),
        ]
    elif p.style == "a(x+b)=c":
        quo = p.c // p.a
        steps = [
            Step(pre=f"dividi ambaŭ flankojn per {p.a}. dekstra flanko:",
                 expr=f"{p.c}/{p.a}", result=str(quo),
                 post=f"do x plus {p.b} egalas al {quo}. subtrahi {p.b}:"),
            Step(expr=f"{quo}-{p.b}", result=str(ans)),
        ]
    elif p.style == "a(x-b)=c":
        quo = p.c // p.a
        steps = [
            Step(pre=f"dividi ambaŭ flankojn per {p.a}. dekstra flanko:",
                 expr=f"{p.c}/{p.a}", result=str(quo),
                 post=f"do x minus {p.b} egalas al {quo}. aldoni {p.b}:"),
            Step(expr=f"{quo}+{p.b}", result=str(ans)),
        ]
    elif p.style == "x/a+b=c":
        diff = p.c - p.b
        steps = [
            Step(pre=f"subtrahi {p.b} de ambaŭ flankoj. dekstra flanko:",
                 expr=f"{p.c}-{p.b}", result=str(diff),
                 post=f"do x/{p.a} egalas al {diff}. multobligi per {p.a}:"),
            Step(expr=f"{diff}*{p.a}", result=str(ans)),
        ]
    else:  # x/a=c
        steps = [
            Step(pre=f"multobligi ambaŭ flankojn per {p.a}:",
                 expr=f"{p.c}*{p.a}", result=str(ans)),
        ]
    return steps, str(ans)


_EQ_STEPS = {"isolate": _steps_eq_isolate}

_EQ_CHAINS = {
    name: (lambda p, _fn=fn: render_prose(*_fn(p)))
    for name, fn in _EQ_STEPS.items()
}


# ══════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════

# (sample_fn, render_q_fn, chain_dict, strategy_keys_or_kind_attr)
# strategy_keys_or_kind_attr: if a list, picked randomly; if a string,
# extracted from the sampled instance's attribute (for kind-based dispatch).
GENERATORS = {
    "ratio":        (sample_ratio,       render_question,  _RATIO_CHAINS,  ["parts", "algebra", "fraction", "diff"]),
    "percent":      (sample_percent,     render_percent_q, _PCT_CHAINS,    ["direct", "decimal", "multiplier"]),
    "inverse-rate": (sample_inverse_rate, render_inv_q,    _INV_CHAINS,    ["constant-product", "per-unit", "inverse-proportion"]),
    "consecutive":  (sample_consec,      render_consec_q,  _CONSEC_CHAINS, ["first-as-x", "average"]),
    "coin":         (sample_coin,        render_coin_q,    _COIN_CHAINS,   ["substitution", "assume-then-correct"]),
    "age":          (sample_age,         render_age_q,     _AGE_CHAINS,    "kind"),
    "mixture":      (sample_mixture,     render_mix_q,     _MIX_CHAINS,    "kind"),
    "distance":     (sample_distance,    render_dist_q,    _DIST_CHAINS,   "kind"),
    "equation":     (sample_equation_solve, render_eq_q,   _EQ_CHAINS,     ["isolate"]),
}

# Step-based dispatchers per type. Used by render_funcall_for() to emit
# multi-turn tool-call training data.
STEPS_BY_TYPE.update({
    "ratio":        _RATIO_STEPS,
    "percent":      _PCT_STEPS,
    "inverse-rate": _INV_STEPS,
    "consecutive":  _CONSEC_STEPS,
    "coin":         _COIN_STEPS,
    "age":          _AGE_STEPS,
    "mixture":      _MIX_STEPS,
    "distance":     _DIST_STEPS,
    "equation":     _EQ_STEPS,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=list(GENERATORS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sample_fn, render_q_fn, chains, strat_spec = GENERATORS[args.type]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    t0 = time.time()
    n_written = 0
    with args.out.open("w") as f:
        for _ in range(args.n):
            p = sample_fn(rng)
            if isinstance(strat_spec, list):
                strat = rng.choice(strat_spec)
            else:
                strat = getattr(p, strat_spec)
            q = render_q_fn(p, rng)
            c = chains[strat](p)
            row = {
                "type": args.type,
                "question_eo": q,
                "chain_eo": c,
                "answer": p.answer,
                "strategy": strat,
                "params": {k: v for k, v in p.__dict__.items()
                            if not k.startswith("_") and not callable(v)},
            }
            # JSON can't serialize tuples-as-keys etc — convert tuples to lists
            row["params"] = {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in row["params"].items()}
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n_written += 1
    dt = time.time() - t0
    print(f"wrote {n_written} {args.type} → {args.out}")
    print(f"  {dt*1000:.0f}ms ({n_written/dt:.0f}/sec)")


if __name__ == "__main__":
    main()
