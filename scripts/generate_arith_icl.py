"""Generate synthetic arithmetic ICL Q/A from the ontology lexicon.

Bypasses the regression/forward-walk pipeline. Builds minimal scenes
from `lex.concepts` (countable nouns) and `lex.person_names`, picks
an arithmetic pattern, renders prose + Q + CoT answer.

Reuses CoT rendering from `generate_icl_from_traces._math_phrase`
so the output style matches the trace-derived arithmetic the model
already trains on.

Patterns: sum, subtract, multistep, comparison, difference,
distribution.

Output schema (same as `generate_icl_from_traces.py`):
  {"messages": [{"role":"user","content": prose+"\\nDemando: ..."},
                {"role":"assistant","content": cot_answer}]}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esperanto_lm.ontology.loader import load_lexicon
from esperanto_lm.ontology.realize.render import (
    past_tense, to_accusative, to_plural,
)
from esperanto_lm.ontology.schemas import CountDeltaEffect

# Pull the trace-derived ICL helpers so CoT formatting stays identical.
import generate_icl_from_traces as gicl
_math_phrase = gicl._math_phrase
_int_to_eo = gicl._int_to_eo


# ----- vocabulary selection -----

def _countable_concepts(lex):
    """Pick concept lemmas whose `count` slot is in-scope. Body parts,
    persons, abstracts and locations are filtered out.

    Returns a list of lemmas. Verb-pairing happens via
    `_count_subtract_verbs` which derives the role's property filter
    from the action schema (no hardcoded edibility)."""
    count_slot = lex.slots.get("count")
    applies = count_slot.applies_to if count_slot else ("physical",)
    out = []
    for lemma, c in lex.concepts.items():
        if c.entity_type in ("person", "abstract", "location"):
            continue
        if getattr(c, "is_body_part", False):
            continue
        if "count" in c.properties:
            out.append(lemma)
            continue
        if any(lex.types.is_subtype(c.entity_type, t) for t in applies):
            out.append(lemma)
    return out


def _count_subtract_verbs(lex):
    """Walk `lex.actions` and find every verb whose first effect is a
    `CountDeltaEffect` with `op="subtract"` on the theme role. For each,
    return `(verb_past, theme_role_spec)` — the role spec carries the
    `type` and `properties` filters a concept must satisfy to be a
    grammatical theme of this verb. Source of truth: the action
    schema itself; nothing hardcoded.

    Skip pure-transfer verbs like `doni` whose subtract is just
    book-keeping for a transfer event — those work for any countable
    theme without affordance constraints, and the generic
    'give X to Y' template handles them separately."""
    out = []
    for verb, action in lex.actions.items():
        if verb in ("doni", "vendi", "montri", "aĉeti", "ricevi"):
            continue
        # Walk every effect; CountDeltaEffect often sits behind a
        # primary property change (manĝi: presence=manĝita THEN count-=).
        eff = next((e for e in action.effects
                    if isinstance(e, CountDeltaEffect)
                    and e.op == "subtract"), None)
        if eff is None:
            continue
        theme_spec = next((r for r in action.roles
                           if r.name == eff.target_role), None)
        if theme_spec is None:
            continue
        out.append((past_tense(verb), theme_spec))
    return out


def _concept_matches_role(concept, role_spec, lex) -> bool:
    """True iff `concept` can fill `role_spec` as a theme: subtype of
    the role's declared type AND satisfies every property filter
    (intersect concept.properties[k] with role_spec.properties[k])."""
    if role_spec.type and not lex.types.is_subtype(
            concept.entity_type, role_spec.type):
        return False
    for slot, allowed in (role_spec.properties or {}).items():
        vals = concept.properties.get(slot, [])
        if not any(v in allowed for v in vals):
            return False
    return True


def _build_verb_concept_index(lex, concepts):
    """For each subtract-verb in the action lexicon, list which of our
    countable concepts can serve as its theme. Pre-computed so the
    subtract generator can pick a verb and then sample a matching
    concept in O(1) at runtime."""
    verbs = _count_subtract_verbs(lex)
    index = []
    for verb_past, role_spec in verbs:
        matches = [c for c in concepts
                   if _concept_matches_role(
                       lex.concepts[c], role_spec, lex)]
        if matches:
            index.append((verb_past, matches))
    return index


def _arith_phrase(
    addends: list[tuple[int, str]], op_kind: str,
    result_count: int, result_lemma: str, rng,
) -> str:
    """Like `_math_phrase` but agrees number (n=1 → singular).
    `result_lemma` is the singular nominative; we render the result
    in nominative (singular when n==1, plural otherwise). When
    `_USE_DIGITS` is set, numbers render as digits and the operator
    biases to the symbol form (`+`/`-`/`=`)."""
    if op_kind == "add":
        op_word, op_sym = "plus", "+"
    elif op_kind == "mul":
        op_word, op_sym = "fojojn", "×"
    elif op_kind == "div":
        op_word, op_sym = "dividita per", "/"
    else:
        op_word, op_sym = "minus", "-"
    # In digit mode bias toward symbols (matches eval surface).
    if _USE_DIGITS:
        op, equals = rng.choice([(op_sym, "="), (op_sym, "=")])
    else:
        op, equals = rng.choice([(op_word, "egalas"), (op_sym, "=")])

    def render(n, lemma):
        if _USE_DIGITS:
            return f"{n} {lemma if n == 1 else to_plural(lemma)}"
        if n == 1:
            return f"unu {lemma}"
        return f"{_int_to_eo(n)} {to_plural(lemma)}"

    chain = f" {op} ".join(render(n, lem) for n, lem in addends)
    return f"{chain} {equals} {render(result_count, result_lemma)}."


def _pluralize(lemma: str) -> str:
    return to_plural(lemma)


def _plural_acc(lemma: str) -> str:
    return to_accusative(to_plural(lemma))


# Per-record digit form toggle. Set by `main` before calling each
# generator so the SAME record uses one surface form consistently
# (mixing word/digit within a record reads as a typo). Across the
# corpus this gives the model exposure to both surface forms — so
# eval prompts that use digits ("15 pomoj") parse correctly even
# though our default training form is word-form ("dek kvin pomoj").
_USE_DIGITS: bool = False

# When True, generators emit GSM8K-style narratives with inline
# <<a OP b = c>>c noun markers and a trailing `#### N` final-answer
# line. Matches the format the model is trained on for GSM8K.
# Forces digits regardless of `_USE_DIGITS` since GSM8K is
# digit-heavy.
_GSM_STYLE: bool = True


_OP_SYM = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def _gsm_eq(a: int, op: str, b: int, result: int, noun: str = "") -> str:
    """GSM8K-style inline equation: '5 - 2 = <<5-2=3>>3 pomoj'.
    `noun` (if given) is appended in plural-nominative; '1 pomo' for
    singular result. Empty noun → no trailing noun phrase."""
    sym = _OP_SYM[op]
    expr = f"{a}{sym}{b}={result}"
    head = f"{a} {sym} {b} = <<{expr}>>{result}"
    if not noun:
        return head
    noun_form = noun if result == 1 else to_plural(noun)
    return f"{head} {noun_form}"


def _gsm_answer(line: str, result: int) -> str:
    """Append the canonical `#### N` final-answer marker."""
    return f"{line}\n#### {result}"


def _num_noun(n: int, lemma: str, *, acc: bool = False) -> str:
    """`tri pomoj` or `tri pomojn`. n=1 stays singular."""
    if _USE_DIGITS:
        # Numeric form: "3 pomoj" / "1 pomo" / "3 pomojn"
        if n == 1:
            return f"1 {to_accusative(lemma) if acc else lemma}"
        return f"{n} {_plural_acc(lemma) if acc else _pluralize(lemma)}"
    if n == 1:
        return f"unu {to_accusative(lemma) if acc else lemma}"
    return f"{_int_to_eo(n)} {_plural_acc(lemma) if acc else _pluralize(lemma)}"


def _pick_name(lex, rng):
    pn = rng.choice(lex.person_names)
    # Use first name; full names are noisier for short arithmetic prose.
    return pn.name.split()[0].capitalize() if pn.name else "Maria"


# ----- generators -----
# Each generator returns a dict matching the ICL JSONL schema, or None
# when the random pick lands on an infeasible combination.

_LOCATIONS = ("kuirejo", "salono", "dormĉambro", "ĝardeno",
              "manĝejo", "ĉambro", "korto", "balkono")
_CONTAINERS = ("tablo", "breto", "korbo", "telero", "ŝranko")
_PREPS = ("en", "sur")


def _pick_distinct_names(lex, rng, k):
    """Sample `k` distinct first names. Returns None if it can't find
    enough distinct picks within a bounded retry budget."""
    names = []
    for _ in range(k * 8):
        n = _pick_name(lex, rng)
        if n not in names:
            names.append(n)
            if len(names) == k:
                return names
    return None


def _gen_sum(rng, lex, concepts, verb_index):
    """K stacks of the same concept, sum across them. Keep numbers small
    (≤20 total) so the model has dense exposure to each cardinal —
    sparse data at larger magnitudes was hurting arithmetic precision."""
    lemma = rng.choice(concepts)
    K = rng.randint(2, 3)
    parts = [rng.randint(1, 8) for _ in range(K)]
    total = sum(parts)
    if total > 50:
        return None
    sentences = []
    for c in parts:
        loc = rng.choice(_LOCATIONS)
        prep = rng.choice(_PREPS)
        if prep == "sur":
            host = rng.choice(_CONTAINERS)
            sentences.append(
                f"Sur la {host} en la {loc} estis {_num_noun(c, lemma)}.")
        else:
            sentences.append(
                f"En la {loc} estis {_num_noun(c, lemma)}.")
    prose = " ".join(sentences)
    q = f"Kiom da {to_plural(lemma)} estis sume?"
    if _GSM_STYLE:
        # Chain partial sums into a running narrative.
        running = parts[0]
        steps = []
        for p in parts[1:]:
            new = running + p
            steps.append(_gsm_eq(running, "add", p, new, lemma))
            running = new
        cot = _gsm_answer(
            f"Entute estis {' . Sume '.join(steps)}.", total)
    else:
        cot = _arith_phrase(
            [(c, lemma) for c in parts], "add", total, lemma, rng)
    return {"messages": [
        {"role": "user", "content": f"{prose}\nDemando: {q}"},
        {"role": "assistant", "content": cot},
    ]}


def _gen_subtract(rng, lex, concepts, verb_index):
    """Actor has N, then a subtract-verb consumes/transfers M units of
    a concept that satisfies the verb's role spec. Mixes:
      - lexicon-derived verb+concept pairs (manĝi+edible, ŝteli+anything…)
      - generic transfer ("donis al X")
    so consumption affordances stay grammatical without hardcoding."""
    use_pair = rng.random() < 0.5 and verb_index
    if use_pair:
        verb_past, matches = rng.choice(verb_index)
        lemma = rng.choice(matches)
    else:
        verb_past = None
        lemma = rng.choice(concepts)
    n = rng.randint(2, 15)
    m = rng.randint(1, n - 1)
    actor = _pick_name(lex, rng)
    if verb_past is not None:
        action = (f"{actor} havis {_num_noun(n, lemma, acc=True)}. "
                  f"Li {verb_past} {_num_noun(m, lemma, acc=True)}.")
    else:
        names = _pick_distinct_names(lex, rng, 2)
        if names is None:
            return None
        actor, recipient = names
        action = (f"{actor} havis {_num_noun(n, lemma, acc=True)}. "
                  f"Li donis {_num_noun(m, lemma, acc=True)} al {recipient}.")
    q = f"Kiom da {to_plural(lemma)} restas?"
    if _GSM_STYLE:
        cot = _gsm_answer(
            f"{actor} havas {_gsm_eq(n, 'sub', m, n - m, lemma)}.",
            n - m)
    else:
        cot = _arith_phrase([(n, lemma), (m, lemma)], "sub",
                            n - m, lemma, rng)
    return {"messages": [
        {"role": "user", "content": f"{action}\nDemando: {q}"},
        {"role": "assistant", "content": cot},
    ]}


def _gen_multi_op(rng, lex, concepts, verb_index):
    """K consecutive operations, each independently add (ricevis ... de)
    or sub (donis ... al). Trains both consecutive subtraction
    ("donis tri kaj du") and add+sub chains.

    Range scales with surface form (word-form ≤50, digit ≤200). 60%
    chance the next op SWITCHES from the previous one — counters the
    "model repeats last operator" failure mode seen on GSM probes."""
    lemma = rng.choice(concepts)
    n = rng.randint(4, 12)
    K = rng.randint(2, 3)
    actor = _pick_name(lex, rng)
    ops = []        # (op, amount, other)
    cur = n
    prev_op = None
    for _ in range(K):
        # 60% switch from previous op; 40% pick freely. First step is free.
        if prev_op is not None and rng.random() < 0.6:
            op = "sub" if prev_op == "add" else "add"
        else:
            op = rng.choice(("add", "sub"))
        if op == "sub":
            if cur <= 1:
                op = "add"
        if op == "sub":
            amt = rng.randint(1, cur - 1)
        else:
            headroom = 50 - cur
            if headroom < 1:
                return None
            amt = rng.randint(1, min(6, headroom))
        prev_op = op
        # Distinct other-person per step
        for _retry in range(6):
            other = _pick_name(lex, rng)
            if other != actor:
                break
        else:
            return None
        ops.append((op, amt, other))
        cur = cur + amt if op == "add" else cur - amt

    # Prose
    sentences = [f"{actor} havis {_num_noun(n, lemma, acc=True)}."]
    for i, (op, amt, other) in enumerate(ops):
        amt_noun = _num_noun(amt, lemma, acc=True)
        if i == 0:
            lead = "Li "
        else:
            lead = "Poste li "
        verb_phrase = (f"ricevis {amt_noun} de {other}"
                       if op == "add" else
                       f"donis {amt_noun} al {other}")
        sentences.append(f"{lead}{verb_phrase}.")
    prose = " ".join(sentences)
    q = f"Kiom da {to_plural(lemma)} havas {actor} nun?"

    # CoT chain
    if _GSM_STYLE:
        steps = []
        run = n
        for i, (op, amt, _) in enumerate(ops):
            new = run + amt if op == "add" else run - amt
            lead = "Komence" if i == 0 else "Poste"
            steps.append(
                f"{lead} {actor} havas {_gsm_eq(run, op, amt, new, lemma)}.")
            run = new
        cot = _gsm_answer(" ".join(steps), cur)
    else:
        cot_parts: list[str] = []
        run = n
        for op, amt, _ in ops:
            new = run + amt if op == "add" else run - amt
            cot_parts.append(_arith_phrase(
                [(run, lemma), (amt, lemma)], op, new, lemma, rng))
            run = new
        final = _num_noun(cur, lemma, acc=True)
        cot = " ".join(cot_parts) + f" {actor} havas {final}."
    return {"messages": [
        {"role": "user", "content": f"{prose}\nDemando: {q}"},
        {"role": "assistant", "content": cot},
    ]}


def _gen_compound_kaj(rng, lex, concepts, verb_index):
    """Compound single-sentence form matching the eval:
        'Maria havis ok krajonojn. Ŝi donis tri al Petro kaj du al Anna.'
    Two ops of the SAME type joined by 'kaj', second amount given bare
    (no repeated noun) about half the time — both eval surface forms.

    The training corpus already has multi_op in the 'Poste'-chained
    form; this fills the gap so the model parses 'kaj' compounds as
    two operations rather than one."""
    lemma = rng.choice(concepts)
    op = rng.choice(("sub", "add"))
    n = rng.randint(4, 14)
    if op == "sub":
        amt1 = rng.randint(1, n - 2)
        amt2 = rng.randint(1, n - amt1 - 1)
        final = n - amt1 - amt2
    else:
        amt1 = rng.randint(1, 5)
        headroom = 50 - n - amt1
        if headroom < 1:
            return None
        amt2 = rng.randint(1, min(5, headroom))
        final = n + amt1 + amt2

    names = _pick_distinct_names(lex, rng, 3)
    if names is None:
        return None
    actor, p1, p2 = names

    verb = "donis" if op == "sub" else "ricevis"
    prep = "al" if op == "sub" else "de"

    a1_phrase = _num_noun(amt1, lemma, acc=True)
    # Half the time, drop the noun in the second amount (eval-style:
    # "donis tri al Petro kaj du al Anna" — no "krajonojn" after du).
    if rng.random() < 0.5:
        a2_phrase = _int_to_eo(amt2)
    else:
        a2_phrase = _num_noun(amt2, lemma, acc=True)

    prose = (
        f"{actor} havis {_num_noun(n, lemma, acc=True)}. "
        f"Li {verb} {a1_phrase} {prep} {p1} kaj "
        f"{a2_phrase} {prep} {p2}."
    )
    if op == "sub":
        q = f"Kiom da {to_plural(lemma)} restas ĉe {actor}?"
    else:
        q = f"Kiom da {to_plural(lemma)} havas {actor} nun?"

    mid = n - amt1 if op == "sub" else n + amt1
    if _GSM_STYLE:
        cot = _gsm_answer(
            f"Post la unua, {actor} havas "
            f"{_gsm_eq(n, op, amt1, mid, lemma)}. "
            f"Post la dua, {actor} havas "
            f"{_gsm_eq(mid, op, amt2, final, lemma)}.",
            final)
    else:
        cot1 = _arith_phrase(
            [(n, lemma), (amt1, lemma)], op, mid, lemma, rng)
        cot2 = _arith_phrase(
            [(mid, lemma), (amt2, lemma)], op, final, lemma, rng)
        cot = f"{cot1} {cot2} {actor} havas {_num_noun(final, lemma, acc=True)}."
    return {"messages": [
        {"role": "user", "content": f"{prose}\nDemando: {q}"},
        {"role": "assistant", "content": cot},
    ]}


def _gen_selective_extract(rng, lex, concepts, verb_index):
    """Prose has 2-4 distinct concepts (each with a count); the question
    asks about ONE of them; the answer is just that count. Direct
    intervention against the model's over-application of sum templates
    to single-category questions.

    Varied surface forms so the model can't shortcut to a single
    template:
      - K ∈ {2, 3, 4} concepts (the eval's failing pattern is K=2)
      - prose: with/without location, with/without 'sume X' addendum,
        single- or multi-sentence
      - question: bare 'Kiom da X?', or 'Kiom da X estis?', or
        'Kiom da X estis en la L?'
      - answer: bare number-noun, or 'Estis X foo.', or 'X foo.'"""
    if len(concepts) < 2:
        return None
    K = rng.choices((2, 3, 4), weights=(50, 30, 20))[0]
    picked: list[str] = []
    for _retry in range(K * 8):
        c = rng.choice(concepts)
        if c not in picked:
            picked.append(c)
            if len(picked) == K:
                break
    if len(picked) < 2:
        return None
    counts = [rng.randint(1, 8) for _ in picked]
    parts = [_num_noun(c, lem) for c, lem in zip(counts, picked)]

    # ---- prose form ----
    loc = rng.choice(_LOCATIONS) if rng.random() < 0.6 else None
    if K == 2:
        # Eval's failing form. Mostly "X kaj Y" without intermediate
        # commas; sometimes with a "sume Z" addendum that names a
        # super-category and the total — the model should still pick
        # the single sub-category answer when asked, not the sum.
        joined = f"{parts[0]} kaj {parts[1]}"
        if loc is not None:
            base = rng.choice([
                f"En la {loc} estis {joined}.",
                f"Sur la tablo en la {loc} estis {joined}.",
                f"{joined} estis en la {loc}.",
            ])
        else:
            base = rng.choice([
                f"Estis {joined}.",
                f"{joined} estis sur la tablo.",
            ])
        prose = base
    else:
        # K=3 or 4: comma-listed.
        listed = ", ".join(parts[:-1]) + f" kaj {parts[-1]}"
        if loc is not None:
            prose = rng.choice([
                f"En la {loc} estis {listed}.",
                f"Sur la tablo estis {listed}.",
                f"{listed} estis en la {loc}.",
            ])
        else:
            prose = f"Estis {listed}."

    # ---- question ----
    idx = rng.randrange(len(picked))
    target_lemma, target_count = picked[idx], counts[idx]
    target_plural = to_plural(target_lemma)
    q_form = rng.choices((
        f"Kiom da {target_plural}?",
        f"Kiom da {target_plural} estis?",
        (f"Kiom da {target_plural} estis en la {loc}?"
         if loc is not None else None),
        f"Kiom estis la {target_plural}?",
    ), weights=(35, 30, 20 if loc else 0, 15))[0]
    if q_form is None:  # location-bound but no loc picked
        q_form = f"Kiom da {target_plural}?"

    # ---- answer ----
    ans_lemma_phrase = _num_noun(target_count, target_lemma)
    cot = rng.choice([
        f"{ans_lemma_phrase}.",
        f"Estis {ans_lemma_phrase}.",
        f"{_int_to_eo(target_count) if target_count != 1 else 'unu'}.",
    ])
    return {"messages": [
        {"role": "user", "content": f"{prose}\nDemando: {q_form}"},
        {"role": "assistant", "content": cot},
    ]}


# Sub-concept → super-category aggregations the model should learn to
# sum across (frato + fratino = gefratoj, pomo + banano = fruktoj, ...).
# Curated; could be derived from `concept.category` later.
_CROSS_CATEGORY = [
    (("frato", "fratino"), "gefrato"),
    (("knabo", "knabino"), "geknabo"),
    (("patro", "patrino"), "gepatro"),
    (("filo", "filino"), "gefilo"),
    (("amiko", "amikino"), "geamiko"),
    (("pomo", "banano"), "frukto"),
    (("pomo", "oranĝo"), "frukto"),
    (("frago", "ĉerizo"), "frukto"),
    (("karoto", "cepo"), "legomo"),
    (("seĝo", "tablo"), "meblo"),
    (("hundo", "kato"), "besto"),
    (("kuracisto", "instruisto"), "persono"),
]


def _gen_cross_category(rng, lex, concepts, verb_index):
    """Two sub-concepts of a known super-category, summed.
    'Petro havis kvar fratojn kaj du fratinojn' → 'kvar fratoj plus du
    fratinoj egalas ses gefratoj.' Trains cross-category aggregation."""
    pair = rng.choice(_CROSS_CATEGORY)
    (sub_a, sub_b), super_c = pair
    if sub_a not in lex.concepts or sub_b not in lex.concepts:
        return None
    na, nb = rng.randint(1, 8), rng.randint(1, 8)
    if na + nb > 15:
        return None
    actor = _pick_name(lex, rng)
    a_phrase = _num_noun(na, sub_a, acc=True)
    b_phrase = _num_noun(nb, sub_b, acc=True)
    prose = f"{actor} havis {a_phrase} kaj {b_phrase}."
    q = f"Kiom da {to_plural(super_c)} havas {actor}?"
    if _GSM_STYLE:
        cot = _gsm_answer(
            f"{actor} havas {_gsm_eq(na, 'add', nb, na + nb, super_c)}.",
            na + nb)
    else:
        cot = _arith_phrase(
            [(na, sub_a), (nb, sub_b)], "add", na + nb, super_c, rng)
    return {"messages": [
        {"role": "user", "content": f"{prose}\nDemando: {q}"},
        {"role": "assistant", "content": cot},
    ]}


def _gen_comparison(rng, lex, concepts, verb_index):
    """Two actors hold counts; which has more?"""
    lemma = rng.choice(concepts)
    names = _pick_distinct_names(lex, rng, 2)
    if names is None:
        return None
    a, b = names
    na, nb = rng.randint(1, 15), rng.randint(1, 15)
    if na == nb:
        return None
    prose = (f"{a} havas {_num_noun(na, lemma, acc=True)}. "
             f"{b} havas {_num_noun(nb, lemma, acc=True)}.")
    q = f"Kiu havas pli da {to_plural(lemma)}?"
    winner = a if na > nb else b
    cot = (f"{_int_to_eo(max(na, nb))} estas pli ol "
           f"{_int_to_eo(min(na, nb))}. {winner} havas pli.")
    return {"messages": [
        {"role": "user", "content": f"{prose}\nDemando: {q}"},
        {"role": "assistant", "content": cot},
    ]}


def _gen_difference(rng, lex, concepts, verb_index):
    """Two actors hold counts; by how many does one exceed the other?"""
    lemma = rng.choice(concepts)
    names = _pick_distinct_names(lex, rng, 2)
    if names is None:
        return None
    a, b = names
    na, nb = rng.randint(2, 15), rng.randint(1, 14)
    if na <= nb:
        na, nb = nb + rng.randint(1, 5), nb
    if na > 50:
        return None
    prose = (f"{a} havas {_num_noun(na, lemma, acc=True)}. "
             f"{b} havas {_num_noun(nb, lemma, acc=True)}.")
    q = f"Per kiom da {to_plural(lemma)} {a} havas pli ol {b}?"
    if _GSM_STYLE:
        cot = _gsm_answer(
            f"La diferenco estas {_gsm_eq(na, 'sub', nb, na - nb, lemma)}.",
            na - nb)
    else:
        cot = _arith_phrase([(na, lemma), (nb, lemma)], "sub",
                            na - nb, lemma, rng)
    return {"messages": [
        {"role": "user", "content": f"{prose}\nDemando: {q}"},
        {"role": "assistant", "content": cot},
    ]}


def _gen_distribution(rng, lex, concepts, verb_index):
    """N items divided equally between K persons."""
    k = rng.randint(2, 5)
    per = rng.randint(1, 4)
    n = k * per
    if n > 50:
        return None
    lemma = rng.choice(concepts)
    actors = _pick_distinct_names(lex, rng, k)
    if actors is None:
        return None
    actor_phrase = ", ".join(actors[:-1]) + f" kaj {actors[-1]}"
    prose = (f"Estis {_num_noun(n, lemma)}. "
             f"{actor_phrase} dividis ilin egale.")
    q = f"Kiom da {to_plural(lemma)} ricevas ĉiu?"
    if _GSM_STYLE:
        cot = _gsm_answer(
            f"Ĉiu ricevas {_gsm_eq(n, 'div', k, per, lemma)}.", per)
    else:
        op_word, op_sym = "dividita per", "/"
        op, equals = rng.choice([(op_word, "egalas"), (op_sym, "=")])
        cot = (f"{_int_to_eo(n)} {to_plural(lemma)} {op} {_int_to_eo(k)} "
               f"{equals} {_int_to_eo(per)}. Ĉiu ricevas "
               f"{_num_noun(per, lemma, acc=True)}.")
    return {"messages": [
        {"role": "user", "content": f"{prose}\nDemando: {q}"},
        {"role": "assistant", "content": cot},
    ]}


def _gen_multiplication(rng, lex, concepts, verb_index):
    """Multiplication via the "po" idiom or "kun ... en ĉiu" phrasing:
        "Karlo havis 3 monerojn po 5 dolaroj." → 3 × 5 = 15
        "4 vicoj kun 7 plantoj en ĉiu vico." → 4 × 7 = 28
    Trains GSM8K-style multiplicative composition. Range scales with
    surface form (word ≤50, digit ≤200)."""
    lemma = rng.choice(concepts)
    # Both factors up to 9 so the corpus actually covers
    # everyday products like 3×5, 4×7, 2×10. Rejection on
    # product>20 then keeps the result in our cardinal range.
    a = rng.randint(2, 10)
    b = rng.randint(2, 10)
    product = a * b
    if product > 50:
        return None
    actor = _pick_name(lex, rng)
    style = rng.choice(("po", "vicoj"))
    if style == "po":
        unit = rng.choice(("monero", "skatolo", "saketo", "korbo"))
        prose = (f"{actor} havas {_num_noun(a, unit, acc=True)} "
                 f"po {_num_noun(b, lemma)}.")
        q = f"Kiom da {to_plural(lemma)} havas {actor} entute?"
    else:
        unit = rng.choice(("vico", "linio", "tablo", "breto"))
        prose = (f"Estis {_num_noun(a, unit)} kun "
                 f"{_num_noun(b, lemma)} en ĉiu {unit}.")
        q = f"Kiom da {to_plural(lemma)} estis entute?"
    if _GSM_STYLE:
        cot = _gsm_answer(
            f"Entute {actor if style == 'po' else 'estas'} "
            f"{_gsm_eq(a, 'mul', b, product, lemma)}.", product)
    else:
        cot = _arith_phrase([(a, lemma), (b, lemma)], "mul",
                            product, lemma, rng)
    return {"messages": [
        {"role": "user", "content": f"{prose}\nDemando: {q}"},
        {"role": "assistant", "content": cot},
    ]}


_HALVES = {"duono": 2, "triono": 3, "kvarono": 4}


def _gen_division_fraction(rng, lex, concepts, verb_index):
    """Division via fractional consumption ("duonon", "trionon"):
        "Estis 12 kuketoj. Maria manĝis duonon. Kiom restas?"
    The answer is N - N/k, rendered as a sub-CoT. Trains the
    fractional-quantifier pattern GSM8K-style halving uses."""
    fraction, divisor = rng.choice(list(_HALVES.items()))
    # Widen multiple so divisor=2 reaches n=20 (giving 12/2, 14/2 etc.).
    # Rejection on n>20 keeps the result in cardinal range.
    multiple = rng.randint(2, 10)
    n = divisor * multiple
    if n > 50:
        return None
    portion = n // divisor
    remaining = n - portion
    lemma = rng.choice(concepts)
    actor = _pick_name(lex, rng)
    prose = (f"Estis {_num_noun(n, lemma)}. {actor} manĝis "
             f"{fraction}n.")
    q = f"Kiom da {to_plural(lemma)} restas?"
    if _GSM_STYLE:
        cot = _gsm_answer(
            f"{actor} manĝis {_gsm_eq(n, 'div', divisor, portion, lemma)}. "
            f"Restas {_gsm_eq(n, 'sub', portion, remaining, lemma)}.",
            remaining)
    else:
        cot = (f"{_arith_phrase([(n, lemma), (divisor, lemma)], 'div', portion, lemma, rng)} "
               f"{_arith_phrase([(n, lemma), (portion, lemma)], 'sub', remaining, lemma, rng)} "
               f"Restas {_num_noun(remaining, lemma)}.")
    return {"messages": [
        {"role": "user", "content": f"{prose}\nDemando: {q}"},
        {"role": "assistant", "content": cot},
    ]}


def _gen_three_step(rng, lex, concepts, verb_index):
    """Three consecutive ±-operations on the same actor's stack.
    Generalises `_gen_multi_op` to K=3 forced (vs its random 2-3).
    60% chance each step switches op from the previous one (fights the
    "repeat last op" failure mode); range scales with surface form."""
    lemma = rng.choice(concepts)
    n = rng.randint(5, 12)
    K = 3
    actor = _pick_name(lex, rng)
    ops = []
    cur = n
    prev_op = None
    for _ in range(K):
        if prev_op is not None and rng.random() < 0.6:
            op = "sub" if prev_op == "add" else "add"
        else:
            op = rng.choice(("add", "sub"))
        if op == "sub":
            if cur <= 1: op = "add"
        if op == "sub":
            amt = rng.randint(1, cur - 1)
        else:
            headroom = 50 - cur
            if headroom < 1: return None
            amt = rng.randint(1, min(5, headroom))
        prev_op = op
        for _r in range(6):
            other = _pick_name(lex, rng)
            if other != actor: break
        else:
            return None
        ops.append((op, amt, other))
        cur = cur + amt if op == "add" else cur - amt
    if cur < 0 or cur > 50: return None

    sentences = [f"{actor} havis {_num_noun(n, lemma, acc=True)}."]
    for i, (op, amt, other) in enumerate(ops):
        lead = "Li " if i == 0 else "Poste li "
        amt_noun = _num_noun(amt, lemma, acc=True)
        verb_phrase = (f"ricevis {amt_noun} de {other}"
                       if op == "add" else
                       f"donis {amt_noun} al {other}")
        sentences.append(f"{lead}{verb_phrase}.")
    prose = " ".join(sentences)
    q = f"Kiom da {to_plural(lemma)} havas {actor} nun?"
    if _GSM_STYLE:
        steps = []
        run = n
        for i, (op, amt, _) in enumerate(ops):
            new = run + amt if op == "add" else run - amt
            lead = "Komence" if i == 0 else ("Poste" if i == 1 else "Fine")
            steps.append(
                f"{lead} {actor} havas {_gsm_eq(run, op, amt, new, lemma)}.")
            run = new
        cot = _gsm_answer(" ".join(steps), cur)
    else:
        cot_parts: list[str] = []
        run = n
        for op, amt, _ in ops:
            new = run + amt if op == "add" else run - amt
            cot_parts.append(_arith_phrase(
                [(run, lemma), (amt, lemma)], op, new, lemma, rng))
            run = new
        final = _num_noun(cur, lemma, acc=True)
        cot = " ".join(cot_parts) + f" {actor} havas {final}."
    return {"messages": [
        {"role": "user", "content": f"{prose}\nDemando: {q}"},
        {"role": "assistant", "content": cot},
    ]}


_RATE_UNITS = ("monero", "skatolo", "saketo", "korbo", "boteloj")
_RATE_VALUES = ("dolaro", "centimo", "punkto", "grado")


def _gen_two_rate_mul(rng, lex, concepts, verb_index):
    """Two-rate multiplication + sum (GSM8K classic):
        "Karlo havis 3 monerojn po 5 dolaroj kaj 2 monerojn po 10 dolaroj.
         Kiom da dolaroj?" → 3×5 + 2×10 = 35.
    Distinct units (monero/skatolo) hold distinct value-rates; total is
    sum of products. Trains the compositional shape current
    `_gen_multiplication` (single-rate only) misses."""
    # Wider factor range — the original [2,4] never produced 3×5, 4×7
    # etc. which are exactly the GSM-style probes. Total still capped
    # at 20 by rejection.
    a1, b1 = rng.randint(2, 10), rng.randint(2, 10)
    a2, b2 = rng.randint(2, 10), rng.randint(2, 10)
    p1, p2 = a1 * b1, a2 * b2
    total = p1 + p2
    if total > 50 or p1 > 50 or p2 > 50:
        return None
    actor = _pick_name(lex, rng)
    unit1, unit2 = rng.sample(_RATE_UNITS, 2)
    value = rng.choice(_RATE_VALUES)
    prose = (f"{actor} havas {_num_noun(a1, unit1, acc=True)} "
             f"po {_num_noun(b1, value)} kaj "
             f"{_num_noun(a2, unit2, acc=True)} "
             f"po {_num_noun(b2, value)}.")
    q = f"Kiom da {to_plural(value)} havas {actor} entute?"
    if _GSM_STYLE:
        unit1_pl = to_plural(unit1)
        unit2_pl = to_plural(unit2)
        cot = _gsm_answer(
            f"La {unit1_pl} valoras {_gsm_eq(a1, 'mul', b1, p1, value)}. "
            f"La {unit2_pl} valoras {_gsm_eq(a2, 'mul', b2, p2, value)}. "
            f"Entute {actor} havas "
            f"{_gsm_eq(p1, 'add', p2, total, value)}.",
            total)
    else:
        cot1 = _arith_phrase([(a1, unit1), (b1, value)], "mul", p1, value, rng)
        cot2 = _arith_phrase([(a2, unit2), (b2, value)], "mul", p2, value, rng)
        cot3 = _arith_phrase([(p1, value), (p2, value)], "add", total, value, rng)
        cot = (f"{cot1} {cot2} {cot3} "
               f"{actor} havas {_num_noun(total, value, acc=True)}.")
    return {"messages": [
        {"role": "user", "content": f"{prose}\nDemando: {q}"},
        {"role": "assistant", "content": cot},
    ]}


_DAYS = ("lundo", "mardo", "merkredo", "ĵaŭdo", "vendredo",
         "sabato", "dimanĉo")
_TIMES = ("matene", "tagmeze", "vespere", "nokte")


def _gen_labeled_sum(rng, lex, concepts, verb_index):
    """Sum across labeled contexts (different days/times of same lemma):
        "Anna legis 8 paĝojn lundo, 12 paĝojn mardo, 5 paĝojn merkredo."
         → 8 + 12 + 5 = 25.
    The repeating-day prose is what GSM8K uses for cumulative reading/
    earning/spending — `_gen_sum` uses location-varying prose instead
    and the model doesn't generalize the per-context pattern."""
    K = rng.randint(2, 3)
    parts = [rng.randint(2, 8) for _ in range(K)]
    total = sum(parts)
    if total > 50:
        return None
    lemma = rng.choice(concepts)
    actor = _pick_name(lex, rng)
    verb_past = rng.choice(("legis", "manĝis", "vidis", "kalkulis",
                            "kolektis", "trovis"))
    labels = rng.sample(_DAYS, K)
    sents = []
    for cnt, label in zip(parts, labels):
        sents.append(f"{_num_noun(cnt, lemma, acc=True)} {label}")
    # Comma list with a kaj before the last
    if K == 2:
        addends = f"{sents[0]} kaj {sents[1]}"
    else:
        addends = ", ".join(sents[:-1]) + f" kaj {sents[-1]}"
    prose = f"{actor} {verb_past} {addends}."
    q = f"Kiom da {to_plural(lemma)} {actor} {verb_past} sume?"
    if _GSM_STYLE:
        running = parts[0]
        steps = []
        for i, p in enumerate(parts[1:], start=1):
            new = running + p
            steps.append(_gsm_eq(running, "add", p, new, lemma))
            running = new
        cot = _gsm_answer(
            f"Sume {actor} {verb_past} {' . Poste '.join(steps)}.", total)
    else:
        cot = _arith_phrase(
            [(c, lemma) for c in parts], "add", total, lemma, rng)
    return {"messages": [
        {"role": "user", "content": f"{prose}\nDemando: {q}"},
        {"role": "assistant", "content": cot},
    ]}


GENERATORS = [
    ("sum", _gen_sum, 10),
    ("subtract", _gen_subtract, 12),
    ("multi_op", _gen_multi_op, 10),
    ("compound_kaj", _gen_compound_kaj, 12),
    ("selective", _gen_selective_extract, 17),
    ("cross_category", _gen_cross_category, 8),
    ("comparison", _gen_comparison, 3),
    ("difference", _gen_difference, 3),
    ("multiplication", _gen_multiplication, 8),
    ("two_rate_mul", _gen_two_rate_mul, 8),
    ("labeled_sum", _gen_labeled_sum, 8),
    ("division", _gen_division_fraction, 5),
    ("three_step", _gen_three_step, 8),
    ("distribution", _gen_distribution, 10),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20000,
                    help="Target number of records")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max-per-q", type=int, default=200,
                    help="Skip a record if its exact Q string has "
                         "already been emitted this many times — "
                         "prevents memorization of repeat questions.")
    ap.add_argument("--stratify-result", action="store_true",
                    help="Reject records to enforce uniform distribution "
                         "of final numeric answer across [1, --result-max]. "
                         "Counters the natural skew toward 1-4 in random "
                         "arithmetic (which biases the model toward 'du' "
                         "as a low-confidence default).")
    ap.add_argument("--result-max", type=int, default=20,
                    help="Upper bound for result-stratification bucket. "
                         "Each value 1..result-max gets ~n/result-max "
                         "records when --stratify-result is set.")
    args = ap.parse_args()

    lex = load_lexicon()
    concepts = _countable_concepts(lex)
    verb_index = _build_verb_concept_index(lex, concepts)
    print(f"Countable concepts: {len(concepts)}", file=sys.stderr)
    print(f"Subtract-verb pairs: "
          f"{[(v, len(ms)) for v, ms in verb_index]}", file=sys.stderr)
    if not concepts:
        print("No countable concepts found — aborting", file=sys.stderr)
        return

    rng = random.Random(args.seed)
    gens = [g[1] for g in GENERATORS]
    weights = [g[2] for g in GENERATORS]
    names = [g[0] for g in GENERATORS]

    from collections import Counter
    import re as _re
    q_counts: Counter = Counter()
    gen_counts: Counter = Counter()
    result_counts: Counter = Counter()
    result_skips: Counter = Counter()
    per_result_cap = (args.n // args.result_max
                      if args.stratify_result else None)

    # Build word→int lookup so we can read both digit and word answers.
    word_to_int: dict[str, int] = {}
    for i, w in enumerate(_int_to_eo(j) for j in range(args.result_max + 1)):
        if w:
            word_to_int[w] = i
            if " " not in w:
                word_to_int[w + "n"] = i

    def _final_int(text: str):
        """Last integer in `text` — digit or Esperanto word form."""
        digits = _re.findall(r"-?\d+", text)
        if digits:
            return int(digits[-1])
        for tok in reversed(_re.findall(r"[a-zĉĝĥĵŝŭ]+", text.lower())):
            if tok in word_to_int:
                return word_to_int[tok]
        return None

    emitted = 0
    attempts = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        while emitted < args.n:
            attempts += 1
            if attempts > args.n * 20:
                print(f"Too many failed attempts ({attempts}) "
                      f"for {emitted} emitted — stopping", file=sys.stderr)
                break
            idx = rng.choices(range(len(gens)), weights=weights)[0]
            # GSM-style is digit-first (matches GSM8K corpus). Legacy
            # path still mixes 30% digit / 70% word — keeps the
            # surface-form variation that helped pretrain transfer.
            global _USE_DIGITS
            _USE_DIGITS = True if _GSM_STYLE else (rng.random() < 0.3)
            rec = gens[idx](rng, lex, concepts, verb_index)
            if rec is None:
                continue
            q = rec["messages"][0]["content"]
            if q_counts[q] >= args.max_per_q:
                continue
            if per_result_cap is not None:
                asst = next((m["content"] for m in rec["messages"]
                             if m["role"] == "assistant"), "")
                r = _final_int(asst)
                if r is None or r < 1 or r > args.result_max:
                    result_skips["out_of_range"] += 1
                    continue
                if result_counts[r] >= per_result_cap:
                    result_skips[r] += 1
                    continue
                result_counts[r] += 1
            q_counts[q] += 1
            gen_counts[names[idx]] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            emitted += 1
    print(f"Wrote {emitted} records to {out_path}", file=sys.stderr)
    for name in names:
        c = gen_counts[name]
        print(f"  {name}: {c} ({c/emitted*100:.1f}%)", file=sys.stderr)
    if per_result_cap is not None:
        print(f"\nResult-stratified (cap={per_result_cap} per value):",
              file=sys.stderr)
        for v in range(1, args.result_max + 1):
            c = result_counts.get(v, 0)
            print(f"  {v:>3}: {c}", file=sys.stderr)
        oor = result_skips.get("out_of_range", 0)
        cap_skips = sum(v for k, v in result_skips.items() if k != "out_of_range")
        print(f"  Skipped (out of range): {oor}", file=sys.stderr)
        print(f"  Skipped (cap full):     {cap_skips}", file=sys.stderr)


if __name__ == "__main__":
    main()
