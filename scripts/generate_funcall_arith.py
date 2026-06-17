"""Synthesize Esperanto arithmetic word problems with function-call
answers. The model identifies operands + operator but does NOT compute
— output is one or more `[[expr]]` calls.

Reuses the lexicon, person names, and scene-building helpers from
`generate_arith_icl` so morphology (case, plural, gender pronouns) and
verb/concept compatibility (manĝi+edible, etc.) stay consistent with
the existing arith CoT pipeline. Only the answer-side renderer is new:
GSM `<<a OP b = c>>c noun` is replaced by bare `[[a OP b]]` with `#N`
back-references for multi-step.

Format:
  Single op:    `[[5+3]]`
  Multi-step:   `[[5+3]] [[#1-2]]`  (#N = result of the Nth call)

NOTE: Brackets are `[[ ]]` not `<< >>` because `train_sft.py` has a
`_clean_gsm8k_markers` step that strips every `<<[^>]*>>` from message
content before training. Square brackets sail through unchanged.

Includes an extractor + per-call grader.

Usage:
    uv run python scripts/generate_funcall_arith.py \\
        --n-per-pattern 5000 \\
        --out data/sft/funcall_arith.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esperanto_lm.ontology.loader import load_lexicon
from esperanto_lm.ontology.realize.render import to_plural, to_accusative


def _pl_acc(lemma):
    """Plural accusative ('pomojn'). For 'donis siajn pomojn' constructions."""
    return to_accusative(to_plural(lemma))

# Pull the scene helpers that already handle Eo morphology, verb-concept
# compatibility, etc. Avoids re-implementing the lexicon's `applies_to`
# filtering, accusative agreement, name picking, etc.
import generate_arith_icl as ga
_num_noun = ga._num_noun
_pick_name = ga._pick_name
_pick_distinct_names = ga._pick_distinct_names
_countable_concepts = ga._countable_concepts
_build_verb_concept_index = ga._build_verb_concept_index

# Force digit-form numbers for the function-call mode. Reasons:
# - call syntax is `[[5+3]]`, mixing digit calls with word-form prose
#   ("kvin pomoj... [[5+3]]") would be jarring;
# - GSM-style training already exposes the model to digit operands.
ga._USE_DIGITS = True
ga._GSM_STYLE = False  # we render our own answer; disable the GSM CoT.


INSTRUCTION_PREFIX = ("Solvu paŝon post paŝo per kalkulado. "
                      "Skribu ĉiun paŝon kiel [[esprimo]]:")


# Distractor signal comes from GSM8K data in the mix, not from injected
# extra-concept sentences in synthetic problems. Reason: the existing
# arith generator separates concerns (single-concept arith vs multi-
# concept extract); we follow the same architectural split here.


# ---------- single-op generators ----------

def _gen_add(rng, lex, concepts, verb_index):
    """Actor has N, then receives M more."""
    name = _pick_name(lex, rng)
    lemma = rng.choice(concepts)
    n = rng.randint(2, 100)
    m = rng.randint(2, 100)
    prose = (f"{name} havis {_num_noun(n, lemma, acc=True)}. "
             f"Poste {name} ricevis {_num_noun(m, lemma, acc=True)} pliajn. "
             f"Kiom da {to_plural(lemma)} {name} havas nun?")
    calls = [f"[[{n}+{m}]]"]
    return prose, calls

def _gen_sub(rng, lex, concepts, verb_index):
    """Actor has N, gives M away (uses lexicon verb pairs when available)."""
    use_pair = rng.random() < 0.5 and verb_index
    if use_pair:
        verb_past, matches = rng.choice(verb_index)
        lemma = rng.choice(matches)
    else:
        verb_past = None
        lemma = rng.choice(concepts)
    n = rng.randint(3, 150)
    m = rng.randint(1, n - 1)
    actor = _pick_name(lex, rng)
    if verb_past is not None:
        prose = (f"{actor} havis {_num_noun(n, lemma, acc=True)}. "
                 f"{actor} {verb_past} {_num_noun(m, lemma, acc=True)}. "
                 f"Kiom da {to_plural(lemma)} restas?")
    else:
        names = _pick_distinct_names(lex, rng, 2)
        if names is None:
            return None
        actor, recipient = names
        prose = (f"{actor} havis {_num_noun(n, lemma, acc=True)}. "
                 f"{actor} donis {_num_noun(m, lemma, acc=True)} al {recipient}. "
                 f"Kiom da {to_plural(lemma)} {actor} havas nun?")
    calls = [f"[[{n}-{m}]]"]
    return prose, calls

def _gen_mul(rng, lex, concepts, verb_index):
    """K containers, each with M of the concept."""
    lemma = rng.choice(concepts)
    K = rng.randint(2, 12)
    M = rng.randint(2, 30)
    cont = rng.choice(("skatoloj", "korboj", "sakoj", "kestoj"))
    prose = (f"Estis {K} {cont}, kaj en ĉiu estis "
             f"{_num_noun(M, lemma)}. "
             f"Kiom da {to_plural(lemma)} entute?")
    calls = [f"[[{K}*{M}]]"]
    return prose, calls

def _gen_div(rng, lex, concepts, verb_index):
    """N items split equally among K recipients."""
    lemma = rng.choice(concepts)
    K = rng.randint(2, 12)
    per = rng.randint(2, 25)
    N = K * per
    rec = rng.choice(("infanoj", "amikoj", "studentoj", "ludantoj"))
    prose = (f"{N} {to_plural(lemma)} estis dividitaj egale inter "
             f"{K} {rec}. "
             f"Kiom da {to_plural(lemma)} ricevis ĉiu?")
    calls = [f"[[{N}/{K}]]"]
    return prose, calls


# ---------- multi-step generators ----------

def _gen_add_then_sub(rng, lex, concepts, verb_index):
    """Had N, received M, gave away K."""
    name = _pick_name(lex, rng)
    lemma = rng.choice(concepts)
    n = rng.randint(2, 50)
    m = rng.randint(2, 50)
    total = n + m
    k = rng.randint(1, total - 1)
    prose = (f"{name} havis {_num_noun(n, lemma, acc=True)}. "
             f"{name} ricevis {_num_noun(m, lemma, acc=True)} pliajn. "
             f"Poste {name} donis {_num_noun(k, lemma, acc=True)} al amiko. "
             f"Kiom da {to_plural(lemma)} restas al {name}?")
    calls = [f"[[{n}+{m}]]", f"[[#1-{k}]]"]
    return prose, calls

def _gen_sub_then_add(rng, lex, concepts, verb_index):
    """Had N, gave M, then received K."""
    name = _pick_name(lex, rng)
    lemma = rng.choice(concepts)
    n = rng.randint(3, 100)
    m = rng.randint(1, n - 1)
    k = rng.randint(2, 50)
    prose = (f"{name} havis {_num_noun(n, lemma, acc=True)}. "
             f"{name} donis {_num_noun(m, lemma, acc=True)} al frato. "
             f"Poste {name} ricevis {_num_noun(k, lemma, acc=True)} novajn. "
             f"Kiom da {to_plural(lemma)} {name} havas nun?")
    calls = [f"[[{n}-{m}]]", f"[[#1+{k}]]"]
    return prose, calls

def _gen_mul_then_sub(rng, lex, concepts, verb_index):
    """K containers of M, then lost L."""
    lemma = rng.choice(concepts)
    K = rng.randint(2, 10)
    M = rng.randint(2, 12)
    total = K * M
    L = rng.randint(1, max(1, total - 1))
    cont = rng.choice(("skatoloj", "korboj", "sakoj", "kestoj"))
    prose = (f"Estis {K} {cont}, ĉiu kun {_num_noun(M, lemma)}. "
             f"Poste {_num_noun(L, lemma)} perdiĝis. "
             f"Kiom da {to_plural(lemma)} restas?")
    calls = [f"[[{K}*{M}]]", f"[[#1-{L}]]"]
    return prose, calls

def _gen_mul_then_add(rng, lex, concepts, verb_index):
    """K containers of M, then L more added."""
    lemma = rng.choice(concepts)
    K = rng.randint(2, 10)
    M = rng.randint(2, 12)
    L = rng.randint(2, 50)
    cont = rng.choice(("skatoloj", "korboj", "sakoj", "kestoj"))
    prose = (f"En ĉiu el {K} {cont} estis {_num_noun(M, lemma)}. "
             f"Poste oni aldonis {_num_noun(L, lemma, acc=True)} pliajn. "
             f"Kiom da {to_plural(lemma)} entute?")
    calls = [f"[[{K}*{M}]]", f"[[#1+{L}]]"]
    return prose, calls

def _gen_div_then_sub(rng, lex, concepts, verb_index):
    """N items split among K; one person used L."""
    lemma = rng.choice(concepts)
    K = rng.randint(2, 10)
    per = rng.randint(3, 20)
    N = K * per
    L = rng.randint(1, per - 1)
    rec_sg, rec_pl = rng.choice((
        ("infano", "infanoj"), ("amiko", "amikoj"),
        ("studento", "studentoj"), ("ludanto", "ludantoj"),
    ))
    name = _pick_name(lex, rng)
    prose = (f"{N} {to_plural(lemma)} estis dividitaj egale inter "
             f"{K} {rec_pl}. {name}, unu el la {rec_pl}, uzis "
             f"{_num_noun(L, lemma, acc=True)}. "
             f"Kiom da {to_plural(lemma)} {name} havas nun?")
    calls = [f"[[{N}/{K}]]", f"[[#1-{L}]]"]
    return prose, calls


# ---------- three-step ----------

def _gen_add_sub_mul(rng, lex, concepts, verb_index):
    """Had N, received M, gave L, then K-fold."""
    name = _pick_name(lex, rng)
    lemma = rng.choice(concepts)
    n = rng.randint(2, 30)
    m = rng.randint(2, 30)
    total = n + m
    l = rng.randint(1, total - 1)
    k = rng.randint(2, 8)
    prose = (f"{name} havis {_num_noun(n, lemma, acc=True)}. "
             f"{name} ricevis {_num_noun(m, lemma, acc=True)} pliajn. "
             f"Poste {name} donis {_num_noun(l, lemma, acc=True)} al amiko. "
             f"Fine, {name} aĉetis {k} fojojn pli da {to_plural(lemma)} ol restis. "
             f"Kiom da {to_plural(lemma)} {name} aĉetis fine?")
    calls = [f"[[{n}+{m}]]", f"[[#1-{l}]]", f"[[#2*{k}]]"]
    return prose, calls


# ---------- hidden-op patterns ----------
# Targets the failure mode where GSM8K's verbal quantifiers
# ("duoble", "triono", "fojojn pli") don't surface as the right
# operator in the model's output. Each pattern surfaces ONE quantifier
# form and binds it to its arithmetic op.

def _gen_n_fold_more(rng, lex, concepts, verb_index):
    """'Petro havis N pomojn. Anna havas K fojojn pli.' -> [[N*K]]."""
    names = _pick_distinct_names(lex, rng, 2)
    if names is None:
        return None
    a, b = names
    lemma = rng.choice(concepts)
    n = rng.randint(2, 30)
    k = rng.randint(2, 8)
    prose = (f"{a} havis {_num_noun(n, lemma, acc=True)}. "
             f"{b} havas {k} fojojn pli da {to_plural(lemma)} ol {a}. "
             f"Kiom da {to_plural(lemma)} havas {b}?")
    calls = [f"[[{n}*{k}]]"]
    return prose, calls

def _gen_double(rng, lex, concepts, verb_index):
    """'X had N. X duobligis sian havaĵon.' -> [[N*2]]."""
    name = _pick_name(lex, rng)
    lemma = rng.choice(concepts)
    n = rng.randint(2, 80)
    prose = (f"{name} havis {_num_noun(n, lemma, acc=True)}. "
             f"{name} duobligis siajn {_pl_acc(lemma)}. "
             f"Kiom da {to_plural(lemma)} {name} havas nun?")
    calls = [f"[[{n}*2]]"]
    return prose, calls

def _gen_half_of(rng, lex, concepts, verb_index):
    """'Duono de N estas ...' -> [[N/2]]. N must be even."""
    name = _pick_name(lex, rng)
    lemma = rng.choice(concepts)
    n = rng.randint(2, 60) * 2
    prose = (f"{name} havis {_num_noun(n, lemma, acc=True)}. "
             f"Duono de la {to_plural(lemma)} estis ruĝaj. "
             f"Kiom da ruĝaj {to_plural(lemma)} {name} havis?")
    calls = [f"[[{n}/2]]"]
    return prose, calls

def _gen_third_of(rng, lex, concepts, verb_index):
    """'Triono de N ...' -> [[N/3]]. N must be divisible by 3."""
    name = _pick_name(lex, rng)
    lemma = rng.choice(concepts)
    n = rng.randint(2, 40) * 3
    prose = (f"{name} havis {_num_noun(n, lemma, acc=True)}. "
             f"Triono de la {to_plural(lemma)} estis verdaj. "
             f"Kiom da verdaj {to_plural(lemma)} {name} havis?")
    calls = [f"[[{n}/3]]"]
    return prose, calls


# ---------- multi-op chain patterns ----------
# Surfaces the GSM8K convention where a sequence of operations on the
# same running total collapses to one [[a OP b OP c OP d]] call.

def _gen_chain_sub(rng, lex, concepts, verb_index):
    """'Marko havis N. Li elspezis A, B, kaj C.' -> [[N-A-B-C]]."""
    name = _pick_name(lex, rng)
    lemma = rng.choice(concepts)
    K = rng.randint(2, 4)  # 2-4 deductions -> total 3-5 operands
    parts = [rng.randint(1, 20) for _ in range(K)]
    n = sum(parts) + rng.randint(1, 50)  # ensure non-negative result
    listed = ", ".join(str(p) for p in parts[:-1]) + f", kaj {parts[-1]}"
    if K == 1:
        listed = str(parts[0])
    prose = (f"{name} havis {_num_noun(n, lemma, acc=True)}. "
             f"{name} elspezis {listed} {_pl_acc(lemma)}. "
             f"Kiom da {to_plural(lemma)} restas?")
    expr = str(n) + "".join(f"-{p}" for p in parts)
    calls = [f"[[{expr}]]"]
    return prose, calls

def _gen_chain_add(rng, lex, concepts, verb_index):
    """'Marko ricevis A, poste B, kaj fine C.' -> [[A+B+C]]."""
    name = _pick_name(lex, rng)
    lemma = rng.choice(concepts)
    K = rng.randint(3, 5)
    parts = [rng.randint(2, 25) for _ in range(K)]
    listed = ", ".join(str(p) for p in parts[:-1]) + f", kaj {parts[-1]}"
    prose = (f"{name} ricevis {listed} {_pl_acc(lemma)} en pluraj fojoj. "
             f"Kiom da {to_plural(lemma)} {name} ricevis entute?")
    expr = "+".join(str(p) for p in parts)
    calls = [f"[[{expr}]]"]
    return prose, calls


# ---------- distractor patterns ----------
# Adds an irrelevant-count sentence about a DIFFERENT lexicon concept.
# Targets the failure mode where the model picks operands by encounter
# order rather than by which concept the question asks about.

def _gen_add_distractor(rng, lex, concepts, verb_index):
    """Add task with an unrelated count of a different concept inserted."""
    name = _pick_name(lex, rng)
    if len(concepts) < 2:
        return None
    lemma = rng.choice(concepts)
    other = rng.choice([c for c in concepts if c != lemma])
    n = rng.randint(2, 80)
    m = rng.randint(2, 80)
    distractor_n = rng.randint(2, 80)
    sentences = [
        f"{name} havis {_num_noun(n, lemma, acc=True)}.",
        f"{name} ankaŭ havis {_num_noun(distractor_n, other, acc=True)}.",
        f"{name} ricevis {_num_noun(m, lemma, acc=True)} pliajn.",
    ]
    rng.shuffle(sentences)
    prose = " ".join(sentences) + f" Kiom da {to_plural(lemma)} {name} havas nun?"
    calls = [f"[[{n}+{m}]]"]
    return prose, calls

def _gen_sub_distractor(rng, lex, concepts, verb_index):
    """Sub task with an unrelated count inserted."""
    name = _pick_name(lex, rng)
    if len(concepts) < 2:
        return None
    lemma = rng.choice(concepts)
    other = rng.choice([c for c in concepts if c != lemma])
    n = rng.randint(5, 100)
    m = rng.randint(1, n - 1)
    distractor_n = rng.randint(2, 80)
    sentences = [
        f"{name} havis {_num_noun(n, lemma, acc=True)}.",
        f"{name} ankaŭ havis {_num_noun(distractor_n, other, acc=True)}.",
        f"{name} donis {_num_noun(m, lemma, acc=True)} al amiko.",
    ]
    rng.shuffle(sentences)
    prose = " ".join(sentences) + f" Kiom da {to_plural(lemma)} restas al {name}?"
    calls = [f"[[{n}-{m}]]"]
    return prose, calls


# ---------- combine two back-refs ----------

def _gen_combine_results(rng, lex, concepts, verb_index):
    """'A skatoloj × B + C skatoloj × D' -> [[A*B]] [[C*D]] [[#1+#2]].

    Two independent multiplications then summed via [[#1+#2]]. This is
    the canonical 'never seen in synth' pattern — two back-refs in one
    call.
    """
    lemma = rng.choice(concepts)
    a = rng.randint(2, 9); b = rng.randint(2, 9)
    c = rng.randint(2, 9); d = rng.randint(2, 9)
    cont = rng.choice(("skatoloj", "korboj", "sakoj"))
    prose = (
        f"En {a} {cont} estis po {b} {to_plural(lemma)}. "
        f"En aliaj {c} {cont} estis po {d} {to_plural(lemma)}. "
        f"Kiom da {to_plural(lemma)} entute?"
    )
    calls = [f"[[{a}*{b}]]", f"[[{c}*{d}]]", "[[#1+#2]]"]
    return prose, calls


# ---------- longer chains (4, 5 steps) ----------

def _gen_chain_4(rng, lex, concepts, verb_index):
    """4-step chain: had N, +M, -A, *B, then -C."""
    name = _pick_name(lex, rng)
    lemma = rng.choice(concepts)
    n = rng.randint(2, 20)
    m = rng.randint(2, 20)
    a = rng.randint(1, n + m - 1)
    b = rng.randint(2, 4)
    after_mul = (n + m - a) * b
    c = rng.randint(1, max(1, after_mul - 1))
    prose = (
        f"{name} havis {_num_noun(n, lemma, acc=True)}. "
        f"{name} ricevis {_num_noun(m, lemma, acc=True)} pliajn. "
        f"Poste {name} donis {_num_noun(a, lemma, acc=True)} al amiko. "
        f"Fine {name} multobligis siajn {_pl_acc(lemma)} per {b}. "
        f"Sed {c} {to_plural(lemma)} perdiĝis. "
        f"Kiom da {to_plural(lemma)} {name} havas nun?"
    )
    calls = [f"[[{n}+{m}]]", f"[[#1-{a}]]", f"[[#2*{b}]]", f"[[#3-{c}]]"]
    return prose, calls

def _gen_chain_5(rng, lex, concepts, verb_index):
    """5-step chain: +, -, *, +, -."""
    name = _pick_name(lex, rng)
    lemma = rng.choice(concepts)
    n = rng.randint(2, 15)
    m = rng.randint(2, 15)
    a = rng.randint(1, n + m - 1)
    b = rng.randint(2, 3)
    after_mul = (n + m - a) * b
    d = rng.randint(2, 15)
    e = rng.randint(1, max(1, after_mul + d - 1))
    prose = (
        f"{name} havis {_num_noun(n, lemma, acc=True)}. "
        f"{name} ricevis {_num_noun(m, lemma, acc=True)} pliajn. "
        f"Poste {name} donis {_num_noun(a, lemma, acc=True)} al amiko. "
        f"Fine {name} multobligis siajn {_pl_acc(lemma)} per {b}. "
        f"Tiam {name} ricevis {_num_noun(d, lemma, acc=True)} pliajn. "
        f"Sed {e} {to_plural(lemma)} perdiĝis. "
        f"Kiom da {to_plural(lemma)} {name} havas nun?"
    )
    calls = [f"[[{n}+{m}]]", f"[[#1-{a}]]", f"[[#2*{b}]]",
             f"[[#3+{d}]]", f"[[#4-{e}]]"]
    return prose, calls


# ---------- percentage (multi-op with decimals) ----------

def _gen_percent(rng, lex, concepts, verb_index):
    """'X% el N -> [[X/100*N]]'. Multi-op chain with decimals comes up
    naturally in GSM8K's percentage problems."""
    lemma = rng.choice(concepts)
    pct = rng.choice((10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90))
    total = rng.choice((50, 100, 150, 200, 300, 400, 500, 1000))
    color = rng.choice(("ruĝaj", "verdaj", "bluaj", "flavaj"))
    prose = (
        f"El {total} {to_plural(lemma)}, {pct}% estas {color}. "
        f"Kiom da {color} {to_plural(lemma)} estas?"
    )
    calls = [f"[[{pct}/100*{total}]]"]
    return prose, calls


# ---------- rate per unit × duration ----------

def _gen_rate_total(rng, lex, concepts, verb_index):
    """'Produktas X per unit, dum Y unitoj' -> [[X*Y]]."""
    name = _pick_name(lex, rng)
    lemma = rng.choice(concepts)
    unit_sg, unit_acc_sg, unit_acc_pl = rng.choice((
        ("horo", "horon", "horojn"),
        ("tago", "tagon", "tagojn"),
        ("minuto", "minuton", "minutojn"),
        ("semajno", "semajnon", "semajnojn"),
    ))
    rate = rng.randint(2, 25)
    duration = rng.randint(2, 30)
    prose = (
        f"{name} produktas {_num_noun(rate, lemma, acc=True)} "
        f"ĉiun {unit_acc_sg}. "
        f"{name} laboris {duration} {unit_acc_pl}. "
        f"Kiom da {to_plural(lemma)} {name} produktis entute?"
    )
    calls = [f"[[{rate}*{duration}]]"]
    return prose, calls


# ---------- longer chains (GSM8K distribution coverage) ----------
# GSM8K test set has answers up to 8 steps deep. Existing chain_4 and
# chain_5 cap at 5 steps; these add 6/7/8 to match the long-chain tail.
# Same alternating-op shape (+ − * + − * + −) as chain_5 — operators
# cycle deterministically, operands are bounded to keep intermediate
# values from blowing up beyond the model's number-token vocab.


def _chain_step(prev, op, x, name, lemma):
    """Render one chain step in a varied natural Esperanto sentence
    keyed off the operator. Picking from a small phrasing pool per
    op-kind so the long chains don't read as identical templates."""
    if op == "+":
        return rng_pick(name, lemma, x, [
            "{name} ricevis {nx} pliajn.",
            "Poste {name} aldonis {nx}.",
            "{name} aĉetis {nx} pliajn.",
        ])
    if op == "-":
        return rng_pick(name, lemma, x, [
            "Poste {name} donis {nx} al amiko.",
            "{name} perdis {nx}.",
            "{name} vendis {nx}.",
            "Sed {x} {pl} perdiĝis.",
        ])
    if op == "*":
        return rng_pick(name, lemma, x, [
            "{name} multobligis siajn {pl_acc} per {x}.",
            "{name} {x}-obligis siajn {pl_acc}.",
        ])
    if op == "/":
        return rng_pick(name, lemma, x, [
            "{name} dividis siajn {pl_acc} egale en {x} grupojn, kaj prenis unu.",
            "{name} havis nur {fraction} de siaj {pl_acc}.",
        ], extra={"fraction": {2: "duonon", 3: "trionon",
                                4: "kvaronon", 5: "kvinonon"}.get(x, "parton")})
    raise ValueError(f"unknown op {op}")


def rng_pick(name, lemma, x, templates, extra=None):
    """Module-level random pick for chain phrasing. Uses a fresh
    Random instance only to avoid leaking state into the caller's rng —
    caller passes a global seed via the script's --seed."""
    import random as _r
    t = _r.choice(templates)
    fields = dict(
        name=name,
        x=x,
        nx=_num_noun(x, lemma, acc=True),
        pl=to_plural(lemma),
        pl_acc=_pl_acc(lemma),
    )
    if extra:
        fields.update({k: v for k, v in extra.items()})
    return t.format(**fields)


_CHAIN_OPS = ("+", "-", "*", "+", "-", "*", "+", "-", "*", "+", "-")


def _gen_chain_n(rng, lex, concepts, verb_index, n_steps):
    """Generic chain generator. Builds a chain of `n_steps` operators
    (alternating + − * cycle), validates intermediate values stay in
    [1, 500] to keep operands tokenizer-friendly, and renders a
    chained-narrative prose. Used by chain_6/7/8 below."""
    for _attempt in range(20):
        name = _pick_name(lex, rng)
        lemma = rng.choice(concepts)
        a = rng.randint(3, 40)
        b = rng.randint(2, 40)
        cur = a + b
        operands = [a, b]
        ops = ["+"]
        prose_steps = [f"{name} ricevis {_num_noun(b, lemma, acc=True)} pliajn."]
        ok = True
        for i in range(n_steps - 1):
            op = _CHAIN_OPS[i + 1]
            if op == "*":
                x = rng.randint(2, 3)
                new_val = cur * x
            elif op == "-":
                if cur < 3:
                    ok = False; break
                x = rng.randint(1, cur - 1)
                new_val = cur - x
            elif op == "+":
                x = rng.randint(2, 30)
                new_val = cur + x
            elif op == "/":
                # pick a divisor of cur
                divisors = [d for d in (2, 3, 4, 5) if cur % d == 0]
                if not divisors:
                    ok = False; break
                x = rng.choice(divisors)
                new_val = cur // x
            if new_val < 1 or new_val > 500:
                ok = False; break
            ops.append(op)
            operands.append(x)
            prose_steps.append(_chain_step(cur, op, x, name, lemma))
            cur = new_val
        if not ok:
            continue
        prose = (
            f"{name} havis {_num_noun(a, lemma, acc=True)}. "
            + " ".join(prose_steps)
            + f" Kiom da {to_plural(lemma)} {name} havas nun?"
        )
        calls = [f"[[{operands[0]}{ops[0]}{operands[1]}]]"]
        for i in range(1, n_steps):
            calls.append(f"[[#{i}{ops[i]}{operands[i+1]}]]")
        return prose, calls
    return None


def _gen_chain_6(rng, lex, concepts, verb_index):
    return _gen_chain_n(rng, lex, concepts, verb_index, 6)


def _gen_chain_7(rng, lex, concepts, verb_index):
    return _gen_chain_n(rng, lex, concepts, verb_index, 7)


def _gen_chain_8(rng, lex, concepts, verb_index):
    return _gen_chain_n(rng, lex, concepts, verb_index, 8)


# ---------- multi-operand single calls ----------
# GSM8K has 710 non-simple expressions like [[16-3-4]] (3-arg sub) or
# [[5+3-1]] (mixed +/- in one call). chain_add and chain_sub cover the
# uniform-op case (`[[a+b+c+d]]`, `[[a-b-c-d]]`). This adds the MIXED
# case where + and − interleave in a single call, the most common
# GSM8K non-simple-expression shape.


def _gen_mixed_seq(rng, lex, concepts, verb_index):
    """'X had N. Gave A. Received B. Gave C.' -> [[N-A+B-C]].
    Alternating +/− in a single call. Mirrors the GSM8K
    'inventory-flow' word problem shape that doesn't decompose into
    back-referenced steps."""
    name = _pick_name(lex, rng)
    lemma = rng.choice(concepts)
    n = rng.randint(20, 150)
    K = rng.randint(2, 4)  # 2-4 alternating moves after the initial value
    # Start with subtract so first move drains the inventory
    ops = ["-"] + [rng.choice(("+", "-")) for _ in range(K - 1)]
    operands = []
    cur = n
    for op in ops:
        if op == "-":
            x = rng.randint(1, max(1, cur - 1))
            cur -= x
        else:
            x = rng.randint(2, 30)
            cur += x
        operands.append(x)
        if cur < 0 or cur > 500:
            return None
    parts = []
    for op, x in zip(ops, operands):
        verb = (
            rng.choice(("donis", "elspezis", "perdis", "vendis"))
            if op == "-" else
            rng.choice(("ricevis", "aldonis", "aĉetis"))
        )
        parts.append(f"{name} {verb} {_num_noun(x, lemma, acc=True)}.")
    prose = (
        f"{name} havis {_num_noun(n, lemma, acc=True)}. "
        + " Poste ".join(parts)
        + f" Kiom da {to_plural(lemma)} restas al {name}?"
    )
    expr = str(n) + "".join(f"{op}{x}" for op, x in zip(ops, operands))
    calls = [f"[[{expr}]]"]
    return prose, calls


# ---------- monetary / unit price ----------
# GSM8K's "Janet sells N eggs at $2 each" pattern. Money is a natural
# multiplicative context that doesn't surface much in the ontology
# (which models physical state, not currency). Adding it broadens
# the operand space without compromising other capabilities.


def _gen_unit_price(rng, lex, concepts, verb_index):
    """'X bought N items at K dollars each. How much?' -> [[N*K]].
    Or 'X sold all and earned Y, how many sold?' -> [[Y/K]]."""
    name = _pick_name(lex, rng)
    lemma = rng.choice(concepts)
    n = rng.randint(2, 50)
    price = rng.randint(2, 30)
    direction = rng.random()
    if direction < 0.5:
        # forward: N items × $K each = ?
        prose = (
            f"{name} aĉetis {_num_noun(n, lemma, acc=True)} "
            f"po ${price} ĉiu. "
            f"Kiom da dolaroj {name} pagis entute?"
        )
        calls = [f"[[{n}*{price}]]"]
    else:
        # backward: earned $T, $K each → how many?
        total = n * price
        prose = (
            f"{name} vendis siajn {_pl_acc(lemma)} "
            f"po ${price} ĉiu kaj gajnis ${total}. "
            f"Kiom da {to_plural(lemma)} {name} vendis?"
        )
        calls = [f"[[{total}/{price}]]"]
    return prose, calls


def _gen_unit_price_then_sub(rng, lex, concepts, verb_index):
    """'X bought N items at $K each, then spent $M of the rest.'
    -> [[N*K]] [[#1-M]]. Two-step money pattern."""
    name = _pick_name(lex, rng)
    lemma = rng.choice(concepts)
    n = rng.randint(3, 30)
    price = rng.randint(2, 25)
    earned = n * price
    spent = rng.randint(1, earned - 1)
    prose = (
        f"{name} vendis {_num_noun(n, lemma, acc=True)} "
        f"po ${price} ĉiu. "
        f"Poste {name} elspezis ${spent}. "
        f"Kiom da dolaroj restas al {name}?"
    )
    calls = [f"[[{n}*{price}]]", f"[[#1-{spent}]]"]
    return prose, calls


# ---------- multi-actor combined inventory ----------
# GSM8K has 'Anna kaj Petro KUNE havas...' style problems mixing two
# actors' inventories. Existing patterns are all single-actor. This
# uses the lexicon's name pool for both names (distinct via
# _pick_distinct_names).


def _gen_multi_actor_combine(rng, lex, concepts, verb_index):
    """'Anna had A, Petro had B. Together gave C. How many remain?'
    -> [[A+B]] [[#1-C]]. Two-actor combine into shared inventory."""
    n1, n2 = _pick_distinct_names(lex, rng, 2)
    lemma = rng.choice(concepts)
    a = rng.randint(3, 50)
    b = rng.randint(3, 50)
    total = a + b
    c = rng.randint(1, total - 1)
    prose = (
        f"{n1} havis {_num_noun(a, lemma, acc=True)}. "
        f"{n2} havis {_num_noun(b, lemma, acc=True)}. "
        f"Kune ili donis {_num_noun(c, lemma, acc=True)} al amikoj. "
        f"Kiom da {to_plural(lemma)} restas al ili?"
    )
    return prose, [f"[[{a}+{b}]]", f"[[#1-{c}]]"]


def _gen_multi_actor_compare(rng, lex, concepts, verb_index):
    """'Anna has A. Petro has B more than Anna. How many does Petro have?'
    -> [[A+B]]. Comparative framing GSM8K uses heavily."""
    n1, n2 = _pick_distinct_names(lex, rng, 2)
    lemma = rng.choice(concepts)
    a = rng.randint(2, 80)
    b = rng.randint(2, 50)
    prose = (
        f"{n1} havas {_num_noun(a, lemma, acc=True)}. "
        f"{n2} havas {b} pli da {to_plural(lemma)} ol {n1}. "
        f"Kiom da {to_plural(lemma)} havas {n2}?"
    )
    return prose, [f"[[{a}+{b}]]"]


# ---------- registry ----------

PATTERNS = {
    "add":              _gen_add,
    "sub":              _gen_sub,
    "mul":              _gen_mul,
    "div":              _gen_div,
    "add_then_sub":     _gen_add_then_sub,
    "sub_then_add":     _gen_sub_then_add,
    "mul_then_sub":     _gen_mul_then_sub,
    "mul_then_add":     _gen_mul_then_add,
    "div_then_sub":     _gen_div_then_sub,
    "add_sub_mul":      _gen_add_sub_mul,
    # hidden-op patterns
    "n_fold_more":      _gen_n_fold_more,
    "double":           _gen_double,
    "half_of":          _gen_half_of,
    "third_of":         _gen_third_of,
    # multi-op chains
    "chain_sub":        _gen_chain_sub,
    "chain_add":        _gen_chain_add,
    # distractor variants
    "add_distractor":   _gen_add_distractor,
    "sub_distractor":   _gen_sub_distractor,
    # two-back-ref combine
    "combine_results":  _gen_combine_results,
    # longer chains
    "chain_4":          _gen_chain_4,
    "chain_5":          _gen_chain_5,
    "chain_6":          _gen_chain_6,
    "chain_7":          _gen_chain_7,
    "chain_8":          _gen_chain_8,
    # multi-operand mixed-op single calls (GSM8K [[5+3-1]] shape)
    "mixed_seq":        _gen_mixed_seq,
    # monetary patterns
    "unit_price":       _gen_unit_price,
    "unit_price_then_sub": _gen_unit_price_then_sub,
    # multi-actor combine + compare
    "multi_actor_combine": _gen_multi_actor_combine,
    "multi_actor_compare": _gen_multi_actor_compare,
    # percent + rate
    "percent":          _gen_percent,
    "rate_total":       _gen_rate_total,
}


# ---------- grader ----------

# Space-tolerant: the morpheme tokenizer inserts <w> tokens between
# every char of `[[5+3]]`, so decoded output is `[ [ 5 + 3 ] ]` with
# spaces between every character. Operand sub-regexes also tolerate
# internal whitespace (e.g. `# 1` for `#1`) which gets stripped in
# extract_calls.
_CALL_RE = re.compile(r"\[\s*\[\s*([^\]]+?)\s*\]\s*\]")
# Operand: back-ref (#1), or signed/unsigned decimal/integer.
# .5 form (no leading digit) and 1.5 form both accepted.
_OPERAND_FIRST_RE = re.compile(
    r"^\s*(#\s*\d+|-?\d+(?:\.\d+)?|-?\.\d+)\s*"
)
# After an operator, leading `-` becomes ambiguous with binary minus,
# so disallow it here. (GSM8K traces never have explicit negative
# literals as later operands.)
_OPERAND_NEXT_RE = re.compile(
    r"^\s*(#\s*\d+|\d+(?:\.\d+)?|\.\d+)\s*"
)
_OP_RE = re.compile(r"^\s*([+\-*/])\s*")


def _parse_inner(text):
    """Parse the inside of `[[ ... ]]` into (operands, ops) tuples.

    Accepts:
      - simple binary: `5+3`, `5*3`, `#1-2`, `1.5*4`, `.5*10`, `-7+3`
      - left-associative chain: `100-50-30-15`, `80/100*10`, `5+3-1`

    Returns (operands, ops) where len(operands) == len(ops) + 1.
    Returns None on parse failure.
    """
    s = text
    m = _OPERAND_FIRST_RE.match(s)
    if not m:
        return None
    operands = [m.group(1).replace(" ", "")]
    s = s[m.end():]
    ops = []
    while s.strip():
        m = _OP_RE.match(s)
        if not m:
            return None
        ops.append(m.group(1))
        s = s[m.end():]
        m = _OPERAND_NEXT_RE.match(s)
        if not m:
            return None
        operands.append(m.group(1).replace(" ", ""))
        s = s[m.end():]
    if len(operands) != len(ops) + 1:
        return None
    return tuple(operands), tuple(ops)

def _resolve(operand, prior_results):
    if operand.startswith("#"):
        idx = int(operand[1:]) - 1
        if 0 <= idx < len(prior_results) and prior_results[idx] is not None:
            return prior_results[idx]
        raise ValueError(f"bad ref {operand}")
    if "." in operand:
        return float(operand)
    return int(operand)


def _apply(op, a, b):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/":
        if b == 0: raise ZeroDivisionError
        if isinstance(a, int) and isinstance(b, int) and a % b == 0:
            return a // b
        return a / b
    raise ValueError(f"bad op {op}")


def execute_calls(calls):
    """Run the call sequence left-associatively as an external executor
    would. Each call is a (operands, ops) tuple from `extract_calls`.

    Returns a list of numeric results, one per input call. A call whose
    operands can't be resolved (bad `#N` ref, division by zero, etc.)
    returns None; any later call that references it via `#N` will also
    return None.

    Multi-op calls like `100-50-30-15` evaluate left-to-right (no
    precedence) — that's how GSM8K's CoT markers are written. Integer
    division returns int when exact, else float.
    """
    results = []
    for operands, ops in calls:
        try:
            val = _resolve(operands[0], results)
            for i, op in enumerate(ops):
                rhs = _resolve(operands[i + 1], results)
                val = _apply(op, val, rhs)
            results.append(val)
        except Exception:
            results.append(None)
    return results


def extract_calls(text):
    """Return list of (operands, ops) tuples — one per `[[ ... ]]` group.

    Both `operands` and `ops` are tuples of strings. For a simple
    binary call like `[[5+3]]` -> (('5','3'), ('+',)). For multi-op
    `[[100-50-30-15]]` -> (('100','50','30','15'), ('-','-','-')).
    `#N` placeholders are preserved, normalized to no internal
    whitespace. Malformed groups are dropped silently — graded as
    missing by `grade()`.
    """
    out = []
    for raw in _CALL_RE.findall(text):
        parsed = _parse_inner(raw)
        if parsed is None:
            continue
        out.append(parsed)
    return out


def _format_call(operands, ops):
    """Render a (operands, ops) pair back to `[[...]]` form."""
    parts = [operands[0]]
    for i, op in enumerate(ops):
        parts.append(op)
        parts.append(operands[i + 1])
    return f"[[{''.join(parts)}]]"

def grade(expected_text, actual_text, *, commutative_ops=("+", "*")):
    """Per-call grading of actual vs expected.

    Each call is the parsed (operands, ops) tuple from `extract_calls`.
    A pair matches `exact` when ops match in sequence AND operands match
    in sequence. For single-op commutative calls (one op, in `+`/`*`),
    operand order is ignored. Multi-op chains require exact sequence
    match for both operands and ops — left-associative evaluation makes
    operand order load-bearing for `-` and `/` chains, and the chain
    structure itself encodes evaluation order.

    Returns dict with totals + per-call breakdown.
    """
    exp = extract_calls(expected_text)
    act = extract_calls(actual_text)
    n = min(len(exp), len(act))
    per_call = []
    exact = op_only = structural = 0
    for i in range(n):
        e_operands, e_ops = exp[i]
        a_operands, a_ops = act[i]
        ops_match = (e_ops == a_ops)
        if ops_match and len(e_ops) == 1 and e_ops[0] in commutative_ops:
            operands_match = set(e_operands) == set(a_operands)
        else:
            operands_match = (e_operands == a_operands)
        if ops_match and operands_match:
            exact += 1
        elif ops_match:
            op_only += 1
        elif operands_match:
            structural += 1
        per_call.append({
            "expected": _format_call(e_operands, e_ops),
            "actual":   _format_call(a_operands, a_ops),
            "op_match": ops_match,
            "operands_match": operands_match,
        })
    return {
        "n_expected": len(exp),
        "n_actual":   len(act),
        "exact":      exact,
        "op_only":    op_only,
        "structural": structural,
        "missing":    max(0, len(exp) - len(act)),
        "extra":      max(0, len(act) - len(exp)),
        "per_call":   per_call,
    }


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", default=",".join(PATTERNS.keys()),
                    help="comma-separated pattern names")
    ap.add_argument("--n-per-pattern", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-prefix", action="store_true",
                    help="omit the 'Solvu paŝon...' instruction prefix")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    lex = load_lexicon()
    concepts = _countable_concepts(lex)
    verb_index = _build_verb_concept_index(lex, concepts)

    names = [n.strip() for n in args.patterns.split(",")]
    for n in names:
        if n not in PATTERNS:
            sys.exit(f"unknown pattern: {n}; have {sorted(PATTERNS)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = Counter()
    rejected = Counter()
    seen = set()

    with out.open("w") as f:
        for name in names:
            fn = PATTERNS[name]
            attempts = 0
            cap = args.n_per_pattern * 10
            while kept[name] < args.n_per_pattern:
                attempts += 1
                if attempts > cap:
                    print(f"  {name}: gave up at {kept[name]}/{args.n_per_pattern}",
                          file=sys.stderr)
                    break
                result = fn(rng, lex, concepts, verb_index)
                if result is None:
                    rejected[name] += 1
                    continue
                prose, calls = result
                key = (prose, " ".join(calls))
                if key in seen:
                    rejected[name] += 1
                    continue
                seen.add(key)
                user_msg = prose if args.no_prefix else f"{INSTRUCTION_PREFIX}\n\n{prose}"
                assistant_msg = " ".join(calls)
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ],
                    "category": f"funcall_arith:{name}",
                    "expected_calls": calls,
                    "n_steps": len(calls),
                }, ensure_ascii=False) + "\n")
                kept[name] += 1

    for n in names:
        print(f"  {n:20s} kept={kept[n]}  rejected={rejected[n]}")
    print(f"wrote {sum(kept.values())} records -> {out}")


if __name__ == "__main__":
    main()
