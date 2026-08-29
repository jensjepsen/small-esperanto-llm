"""Proof-of-concept: 2-layer ratio generator.

Layer 1: hand-authored MATH FRAGMENTS — atomic ways to state "X gets a parts,
Y gets b parts" in different mathematical languages (proporcio / frakcio /
procento / unit-rate / implicit-multiplicative). These carry the math
meaning and stay tight + verified.

Layer 2: LLM-PRE-GENERATED WRAPPER TEMPLATES — narrative scaffolds with
{MATH_STATEMENT} + {QUESTION} placeholders. Provides verbose/tense/context
diversity without touching the numbers.

At runtime: sample params + language + wrapper, compose, render. Chain
stays canonical procedural — verifier guards math integrity.

This is a proof on ratio only. If it works, the pattern extends to
percent/age/inverse-rate/etc.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from esperanto_lm.data import _morpheme_preprocess  # noqa
from word_problems_procedural import (
    Ratio, sample_ratio, PERSON_NAMES, OBJECT_POOL,
    _RATIO_CHAINS, render_chain,
)

# ── Layer 1: math-language fragments ──────────────────────────────────────
# Each fragment must express the math fully and unambiguously. Picks at
# runtime are constrained by the problem's (ratio, total) — e.g. `procento`
# only fires when a+b divides 100 cleanly.
#
# Placeholders: {a}, {b}, {sum} (=a+b), {total}, {name_a}, {name_b},
#               {item_acc_pl}, {item_npl}, {pct_a}, {pct_b}

MATH_FRAGMENTS = {
    "proporcio": [
        "{name_a} kaj {name_b} dividis {total} {item_acc_pl} en proporcio {a}:{b}",
        "{name_a} kaj {name_b} dividis {total} {item_acc_pl} laŭ rilato {a}-al-{b}",
        "{total} {item_npl} estas dividitaj inter {name_a} kaj {name_b} en proporcio {a}:{b}",
    ],
    "frakcio": [
        "{name_a} ricevis {a}/{sum} de {total} {item_npl} kaj {name_b} la ceteran {b}/{sum}",
        "el la totalo de {total} {item_npl}, {name_a} prenis {a}-onon (parton de {sum}) kaj {name_b} {b}-onon",
    ],
    "procento": [
        "{name_a} ricevis {pct_a}% el {total} {item_npl}, kaj {name_b} la reston",
        "el {total} {item_npl}, {pct_a}% iris al {name_a} kaj {pct_b}% al {name_b}",
    ],
    "implicit-multiplicative": [
        "{name_a} kaj {name_b} dividis {total} {item_acc_pl} tiel ke por ĉiu {a} {item_npl} ricevitaj de {name_a}, {name_b} ricevis {b}",
        "el {total} {item_acc_pl}, {name_a} ricevis {a} {item_npl} por ĉiu {b} de {name_b}",
    ],
    "verbose": [
        "{name_a} kaj {name_b} kunhavigis {total} {item_acc_pl}. Ili konsentis ke {name_a} ricevu {a} partojn por ĉiu {b} partoj ricevitaj de {name_b}",
    ],
}


def render_math_statement(p: Ratio, lang: str, rng: random.Random) -> str | None:
    """Pick a fragment from the requested language; return None if the
    problem's params don't admit that language (e.g. procento needs clean %)."""
    # Only 2-way ratios for this PoC
    if len(p.ratio) != 2:
        return None
    a, b = p.ratio
    rsum = a + b
    if lang == "procento" and 100 % rsum != 0:
        return None  # ratio not cleanly convertible to integer %
    pct_a = (a * 100) // rsum if 100 % rsum == 0 else None
    pct_b = (b * 100) // rsum if 100 % rsum == 0 else None
    item_npl = p.item + "j"
    item_acc_pl = p.item + "jn"
    fragment = rng.choice(MATH_FRAGMENTS[lang])
    return fragment.format(
        a=a, b=b, sum=rsum, total=p.total,
        name_a=p.names[0], name_b=p.names[1],
        item_npl=item_npl, item_acc_pl=item_acc_pl,
        pct_a=pct_a, pct_b=pct_b,
    )


# ── Layer 2: wrapper templates ────────────────────────────────────────────
# These get LOADED from a JSON pool. For this PoC we hand-author 12 to
# demonstrate; in production they're Gemini-generated (~200 per type for ~$1).

DEFAULT_WRAPPERS = [
    # Bare — no narrative, just the math statement + question
    {"tone": "bare", "template": "{MATH_STATEMENT}. {QUESTION}"},

    # Bake-sale narrative
    {"tone": "school", "template":
     "Antaŭ la lerneja bazaroj, du gelernantoj kolektis monon por la kuirklubo. {MATH_STATEMENT}. Post la fino, {QUESTION}"},

    # Family inheritance
    {"tone": "family", "template":
     "La avo testamentis al la nepoj la enhavon de sia kabineto. {MATH_STATEMENT}. {QUESTION}"},

    # Sports tournament
    {"tone": "sport", "template":
     "Post la futbalturniro, la trejnisto dividis la premiojn. {MATH_STATEMENT}. {QUESTION}"},

    # Shop scenario
    {"tone": "shop", "template":
     "En la vendejo, du klientoj samtempe aĉetis pakaĵon kaj decidis kunhavigi ĝian enhavon. {MATH_STATEMENT}, kaj {QUESTION}"},

    # Past tense, longer
    {"tone": "story-past", "template":
     "Estis varma somera tago. {MATH_STATEMENT}. Kiam la posttagmezo finiĝis, {QUESTION}"},

    # Hypothetical conditional
    {"tone": "hypothetical", "template":
     "Se {MATH_STATEMENT_LOWER_RESUME}, tiam {QUESTION}"},

    # Newspaper-style impersonal
    {"tone": "news", "template":
     "Laŭ ĵusa raporto: {MATH_STATEMENT}. La demando: {QUESTION}"},

    # Game/puzzle framing
    {"tone": "puzzle", "template":
     "Logika rompvazo: {MATH_STATEMENT}. {QUESTION}"},

    # Direct prompt, terse
    {"tone": "terse", "template":
     "{MATH_STATEMENT}. {QUESTION}"},

    # Wrap with extra context BEFORE math
    {"tone": "context-first", "template":
     "Dum la printempa kunveno de la legoklubo, la membroj decidis distribui sian kolekton. {MATH_STATEMENT}. {QUESTION}"},

    # Question first
    {"tone": "question-first", "template":
     "{QUESTION}? Vi scias ke {MATH_STATEMENT}."},
]


QUESTION_FORMS = {
    "direct": "kiom da {item_npl} ricevis {name_target}?",
    "nominative": "kio estas la parto de {name_target}?",
    "imperative": "kalkulu la kvanton de {item_npl} kiun ricevis {name_target}.",
    "passive": "kiom da {item_npl} estis donitaj al {name_target}?",
    "completion": "la kvanto de {item_npl} ricevita de {name_target} egalas al ___.",
}


def render_question(p: Ratio, qform: str, target_idx: int) -> str:
    item_npl = p.item + "j"
    return QUESTION_FORMS[qform].format(
        item_npl=item_npl, name_target=p.names[target_idx]
    )


def render_diverse(p: Ratio, lang: str, wrapper: dict, qform: str,
                    rng: random.Random) -> str | None:
    """Compose a diverse-form ratio question. Returns None if the chosen
    language doesn't admit this problem."""
    math_stmt = render_math_statement(p, lang, rng)
    if math_stmt is None:
        return None
    # Lower-case the math statement when it appears mid-sentence
    math_stmt_low = math_stmt[0].lower() + math_stmt[1:]
    if "MATH_STATEMENT_LOWER_RESUME" in wrapper["template"]:
        math_stmt_field = math_stmt_low.rstrip(".")
    else:
        math_stmt_field = math_stmt
    # `direct` only (so we can match Ratio.answer with ask='direct')
    if p.ask != "direct":
        return None
    question = render_question(p, qform, p.ask_idx)
    text = wrapper["template"].format(
        MATH_STATEMENT=math_stmt_field,
        MATH_STATEMENT_LOWER_RESUME=math_stmt_field,
        QUESTION=question,
    )
    # Capitalize first letter of final
    text = text[0].upper() + text[1:]
    return text


# ── Driver ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("/tmp/wp_ratio_diverse.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    langs = list(MATH_FRAGMENTS.keys())
    qforms = list(QUESTION_FORMS.keys())

    written = 0
    attempts = 0
    with args.out.open("w") as f:
        while written < args.n and attempts < args.n * 10:
            attempts += 1
            p = sample_ratio(rng)
            # force ask=direct + 2-way for this PoC
            if p.ask != "direct" or len(p.ratio) != 2:
                continue
            lang = rng.choice(langs)
            wrapper = rng.choice(DEFAULT_WRAPPERS)
            qform = rng.choice(qforms)
            text = render_diverse(p, lang, wrapper, qform, rng)
            if text is None:
                continue
            strat = rng.choice(list(_RATIO_CHAINS))
            chain = render_chain(p, strat)
            row = {
                "type": "ratio-diverse",
                "question_eo": text,
                "chain_eo": chain,
                "answer": p.answer,
                "strategy": strat,
                "math_language": lang,
                "wrapper_tone": wrapper["tone"],
                "question_form": qform,
                "params": {"ratio": list(p.ratio), "total": p.total,
                            "names": p.names, "item": p.item},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} diverse ratio problems → {args.out}")


if __name__ == "__main__":
    main()
