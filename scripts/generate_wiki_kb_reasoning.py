"""Cross-entity reasoning Q/A from the Wikidata KB subgraph.

Patterns are data-driven from property metadata, so adding a new
pattern means adding one function and a @register entry.

Only `filter_subset` is implemented here as a template. The other
seven patterns (count_with_prop, groupby_count, multi_hop,
negative_filter, intersection, set_difference, shared_attribute)
follow the same (kb, rng) -> dict|None shape.

Usage:
    uv run --python pypy3.11 --with pydantic --no-project python \\
        scripts/generate_wiki_kb_reasoning.py \\
        --patterns filter_subset \\
        --n-per-pattern 5000 \\
        --out data/sft/wiki_reasoning.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from esperanto_lm.ontology.wiki_kb.load import load_kb

KB_PATH = Path(__file__).parent.parent / "src/esperanto_lm/ontology/wiki_kb/data/kb.json"

# Property classification — the only hand-written table.
# All values in EntityRec.facts are entity QIDs in the current extract,
# so the ORDERABLE class is empty; compare-by-date patterns don't apply
# until date facts get added to the kept-props list.
CATEGORICAL = {"ŝtataneco", "naskiĝloko", "mortloko", "lando", "kontinento"}
ENTITY_REF  = {"naskiĝloko", "mortloko", "patro", "patrino", "infano",
               "lando", "ĉefurbo", "troviĝas en administra unuo",
               "kontinento", "parto de", "nomita laŭ", "verko"}
# Occupation lives in EntityRec.eo_tags, not facts — special-cased per pattern.


PATTERNS = {}

def register(name):
    def deco(fn):
        PATTERNS[name] = fn
        return fn
    return deco


# ---------- shared helpers ----------

def label_of(kb, qid):
    rec = kb.by_id.get(qid)
    if rec:
        return rec.label
    return kb.extra_labels.get(qid, qid)


FACT_TEMPLATES = {
    "naskiĝloko": "{e} naskiĝis en {v}.",
    "mortloko":   "{e} mortis en {v}.",
    "ŝtataneco":  "{e} estas civitano de {v}.",
    "patro":      "La patro de {e} estas {v}.",
    "patrino":    "La patrino de {e} estas {v}.",
    "lando":      "{e} situas en {v}.",
    "kontinento": "{e} situas sur {v}.",
    "ĉefurbo":    "La ĉefurbo de {e} estas {v}.",
}

def fact_sentence(kb, ent, prop):
    vals = ent.facts.get(prop, ())
    if not vals:
        return None
    v = label_of(kb, vals[0])
    tmpl = FACT_TEMPLATES.get(prop, "{e}: " + prop + " estas {v}.")
    return tmpl.format(e=ent.label, v=v)

def render_passage(kb, ents, props, rng):
    sents = [s for e in ents for p in props if (s := fact_sentence(kb, e, p))]
    rng.shuffle(sents)
    return " ".join(sents)


def join_eo(parts):
    """Idiomatic Esperanto list: 'X' / 'X kaj Y' / 'X, Y kaj Z'."""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " kaj " + parts[-1]


def primary_value(ent, pivot):
    """Return the entity's primary (first) value for `pivot`, or None.

    fact_sentence / render_passage always render vals[0], so all hot
    paths must filter on vals[0] too — otherwise the passage and the
    answer can disagree when an entity has multiple values for a fact
    (1.2% of facts in this KB; e.g. Richard Wagner has dual ŝtataneco
    [Germanio, Aŭstrio], and a city in disputed territory may have
    multiple `lando` QIDs that label to different country names).
    """
    vals = ent.facts.get(pivot, ())
    return vals[0] if vals else None


# Shared type/pivot table. Each entry is (entity_type, pivot_prop,
# verb_phrase) — verb_phrase is in plural agreement so it works
# regardless of how many entities the answer covers (the matrix verbs
# situas/naskiĝis/mortis don't conjugate for number, and "civitanoj"
# stays plural in the Q since we're asking about a subset of a list).
# Each pattern reads from this list and embeds verb_phrase into its
# own Q skeleton ("Kiuj el ... ?", "Kiom el ... ?", "Kiuj el ... kaj
# ... ?"). Adding a new type/pivot here gives all three patterns a new
# combo for free.
TYPE_PIVOTS = [
    ("persono", "ŝtataneco",  "estas civitanoj de"),
    ("persono", "naskiĝloko", "naskiĝis en"),
    ("persono", "mortloko",   "mortis en"),
    ("urbo",    "lando",      "situas en"),
    ("urbo",    "kontinento", "situas sur"),
    ("rivero",  "lando",      "situas en"),
    ("rivero",  "kontinento", "situas sur"),
    ("monto",   "lando",      "situas en"),
    ("monto",   "kontinento", "situas sur"),
    ("lago",    "lando",      "situas en"),
]

# Index by type for patterns that need two independent pivots from the
# same type (intersection).
PIVOTS_BY_TYPE = {}
for _t, _p, _v in TYPE_PIVOTS:
    PIVOTS_BY_TYPE.setdefault(_t, []).append((_p, _v))


# ---------- template pattern: filter_subset ----------

@register("filter_subset")
def filter_subset(kb, rng):
    """4 same-type entities, pick which share a target value on `pivot`.

    Rotates over (type, pivot) combos via TYPE_PIVOTS so the model
    sees subset-filter Qs about persons, cities, rivers, mountains,
    and lakes — not only persons.

    Constraints:
      - 2-3 entities hold the target value (non-trivial subset)
      - non-holders carry the pivot with a different value, so the
        passage shows all four values and the answer requires reading
    """
    ent_type, pivot, verb = rng.choice(TYPE_PIVOTS)

    holders_by_value = {}
    for qid in kb.by_type.get(ent_type, ()):
        v = primary_value(kb.by_id[qid], pivot)
        if v is not None:
            holders_by_value.setdefault(v, []).append(qid)

    eligible = [(v, hs) for v, hs in holders_by_value.items() if 2 <= len(hs) <= 6]
    if not eligible:
        return None
    target_value, target_holders = rng.choice(eligible)
    n_holders = rng.randint(2, min(3, len(target_holders)))
    holders = rng.sample(target_holders, n_holders)

    pool = [q for q in kb.by_type.get(ent_type, ())
            if (v := primary_value(kb.by_id[q], pivot)) is not None
            and v != target_value]
    if len(pool) < 4 - n_holders:
        return None
    non = rng.sample(pool, 4 - n_holders)

    ent_qids = holders + non
    rng.shuffle(ent_qids)
    ents = [kb.by_id[q] for q in ent_qids]

    aux = [p for p in CATEGORICAL if p != pivot]
    passage = render_passage(kb, ents, [pivot] + aux, rng)

    if not all(e.label in passage for e in ents):
        return None

    target_label = label_of(kb, target_value)
    names = ", ".join(e.label for e in ents)
    q = f"Kiuj el {names} {verb} {target_label}?"

    correct = [e.label for e in ents if primary_value(e, pivot) == target_value]
    if len(correct) in (0, len(ents)):
        return None
    a = join_eo(correct) + "."

    return {
        "passage": passage, "q": q, "a": a,
        "meta": {"pattern": "filter_subset", "ent_type": ent_type,
                 "pivot": pivot, "target": target_label,
                 "n_correct": len(correct)},
    }


# ---------- pattern: intersection (two facts) ----------

# Pre-build a (val_A, val_B) -> [qids] index once per (type, pivot pair),
# since this is the hot loop's bottleneck. Keyed by (type, frozenset of
# pivots) so either pivot order shares the cache.
_INTERSECTION_CACHE = {}

def _intersection_index(kb, ent_type, pivot_a, pivot_b):
    key = (ent_type, frozenset((pivot_a, pivot_b)))
    if key in _INTERSECTION_CACHE:
        return _INTERSECTION_CACHE[key]
    idx = {}
    for qid in kb.by_type.get(ent_type, ()):
        ent = kb.by_id[qid]
        va = primary_value(ent, pivot_a)
        vb = primary_value(ent, pivot_b)
        if va is None or vb is None:
            continue
        idx.setdefault((va, vb), []).append(qid)
    _INTERSECTION_CACHE[key] = idx
    return idx


@register("intersection")
def intersection(kb, rng):
    """4 same-type entities, pick which satisfy BOTH (pivot_a=val_a)
    AND (pivot_b=val_b).

    Persons-only — the geographic types' two-pivot combos (lando +
    kontinento) are entailment-correlated (every French river is in
    Europe), so the strict-subset gate rejects them ~100%. Persons
    have three pivots (ŝtataneco, naskiĝloko, mortloko), giving
    three independent two-combos that produce real intersections.

    Constraints:
      - target (val_a, val_b) has 1-3 holders (non-trivial subset)
      - non-holders have both pivots populated (passage shows their
        actual values rather than relying on absence)
      - intersection must be strictly smaller than each single-pivot
        match over the chosen entities, else the second pivot is
        decorative
    """
    ent_type = "persono"
    pivots = PIVOTS_BY_TYPE[ent_type]
    (pivot_a, verb_a), (pivot_b, verb_b) = rng.sample(pivots, 2)

    idx = _intersection_index(kb, ent_type, pivot_a, pivot_b)
    candidates = [(k, qs) for k, qs in idx.items() if 1 <= len(qs) <= 3]
    if not candidates:
        return None
    (target_a, target_b), target_holders = rng.choice(candidates)

    n_holders = rng.randint(1, len(target_holders))
    holders = rng.sample(target_holders, n_holders)

    pool_partial = []  # match exactly one of the two
    pool_other = []    # match neither
    for qid in kb.by_type.get(ent_type, ()):
        if qid in target_holders:
            continue
        ent = kb.by_id[qid]
        va = primary_value(ent, pivot_a)
        vb = primary_value(ent, pivot_b)
        if va is None or vb is None:
            continue
        a_match = va == target_a
        b_match = vb == target_b
        if a_match ^ b_match:
            pool_partial.append(qid)
        else:
            pool_other.append(qid)

    n_non = 4 - n_holders
    if len(pool_partial) + len(pool_other) < n_non:
        return None
    n_partial = min(len(pool_partial), max(1, n_non // 2 + 1))
    n_partial = min(n_partial, n_non)
    n_other = n_non - n_partial
    if n_other > len(pool_other):
        n_other, n_partial = len(pool_other), n_non - len(pool_other)
        if n_partial > len(pool_partial):
            return None
    non = rng.sample(pool_partial, n_partial) + rng.sample(pool_other, n_other)

    ent_qids = holders + non
    rng.shuffle(ent_qids)
    ents = [kb.by_id[q] for q in ent_qids]

    passage = render_passage(kb, ents, [pivot_a, pivot_b], rng)

    if not all(e.label in passage for e in ents):
        return None

    target_a_label = label_of(kb, target_a)
    target_b_label = label_of(kb, target_b)
    names = ", ".join(e.label for e in ents)
    q = (f"Kiuj el {names} {verb_a} {target_a_label} "
         f"kaj {verb_b} {target_b_label}?")

    correct = [e.label for e in ents
               if primary_value(e, pivot_a) == target_a
               and primary_value(e, pivot_b) == target_b]
    if len(correct) in (0, len(ents)):
        return None
    only_a = sum(1 for e in ents if primary_value(e, pivot_a) == target_a)
    only_b = sum(1 for e in ents if primary_value(e, pivot_b) == target_b)
    if len(correct) >= only_a or len(correct) >= only_b:
        return None

    a = join_eo(correct) + "."
    return {
        "passage": passage, "q": q, "a": a,
        "meta": {"pattern": "intersection", "ent_type": ent_type,
                 "pivots": [pivot_a, pivot_b],
                 "targets": [target_a_label, target_b_label],
                 "n_correct": len(correct)},
    }


# ---------- pattern: multi_hop ----------

# Cached chain index: list of (person_qid, city_qid, country_qid)
# triples where person->naskiĝloko->city, city->lando->country.
_MULTI_HOP_CHAINS = None

def _multi_hop_chains(kb):
    global _MULTI_HOP_CHAINS
    if _MULTI_HOP_CHAINS is not None:
        return _MULTI_HOP_CHAINS
    chains = []
    for qid in kb.by_type.get("persono", ()):
        ent = kb.by_id[qid]
        for city_qid in ent.facts.get("naskiĝloko", ()):
            city = kb.by_id.get(city_qid)
            if not city:
                continue
            country_vals = city.facts.get("lando", ())
            if not country_vals:
                continue
            # Drop degenerate chains where "city" IS the country (e.g.
            # naskiĝloko=Germanio, lando(Germanio)=Germanio). Those
            # render as "Germanio situas en Germanio." and let the
            # model shortcut on direct extraction.
            if city_qid == country_vals[0]:
                continue
            chains.append((qid, city_qid, country_vals[0]))
            break
    _MULTI_HOP_CHAINS = chains
    return chains


@register("multi_hop")
def multi_hop(kb, rng):
    """N persons, each with chain person->birthplace(city)->country.

    'En kiu lando naskiĝis X?' — answer requires composing two facts
    rendered as separate sentences (no direct person->country fact).

    Constraints:
      - all N persons have distinct chain-derived countries (so
        guessing one country isn't a hedge across multiple targets)
      - passage renders both link sentences per chain (person->city
        and city->country)
      - target person is one of the N at random
    """
    chains = _multi_hop_chains(kb)
    if len(chains) < 3:
        return None

    n = rng.randint(3, 4)
    # Pick chains with distinct countries
    rng.shuffle(chains := list(chains))  # rebind to a shuffled copy
    chosen = []
    seen_countries = set()
    seen_persons = set()
    for p_qid, c_qid, country_qid in chains:
        if country_qid in seen_countries or p_qid in seen_persons:
            continue
        chosen.append((p_qid, c_qid, country_qid))
        seen_countries.add(country_qid)
        seen_persons.add(p_qid)
        if len(chosen) == n:
            break
    if len(chosen) < n:
        return None

    target_p, target_c, target_country = rng.choice(chosen)

    # Render: person->city and city->country sentences, all shuffled
    sents = []
    for p_qid, c_qid, country_qid in chosen:
        person = kb.by_id[p_qid]
        city = kb.by_id[c_qid]
        country_label = label_of(kb, country_qid)
        sents.append(f"{person.label} naskiĝis en {city.label}.")
        sents.append(f"{city.label} situas en {country_label}.")
    rng.shuffle(sents)
    passage = " ".join(sents)

    target_person_label = kb.by_id[target_p].label
    q = f"En kiu lando naskiĝis {target_person_label}?"
    a = label_of(kb, target_country) + "."

    return {
        "passage": passage, "q": q, "a": a,
        "meta": {"pattern": "multi_hop",
                 "chain": "person->city->country",
                 "n_persons": len(chosen),
                 "target_person": target_person_label,
                 "target_country": label_of(kb, target_country)},
    }


# ---------- pattern: count_with_prop ----------

# Esperanto number words 0..10. Counts beyond 10 are rejected at sample
# time because longer-form numbers ("dek du") add tokenizer noise.
EO_NUMBERS = ("nul", "unu", "du", "tri", "kvar", "kvin",
              "ses", "sep", "ok", "naŭ", "dek")

@register("count_with_prop")
def count_with_prop(kb, rng):
    """N same-type entities, count how many hold pivot=target.

    'Kiom el A, B, C, D, E estas civitanoj de Francio?' -> 'Tri.'
    Output is a spelled-out Esperanto number word. Forces counting
    over the passage rather than listing names. Rotates over (type,
    pivot) combos via TYPE_PIVOTS so the model sees counting Qs about
    persons, cities, rivers, mountains, and lakes.

    Constraints:
      - N in [5, 8] entities (so the count range is interesting)
      - count strictly in (0, N) (not all-or-nothing)
      - non-holders carry the pivot prop with a different value, so
        the passage shows all N values
    """
    ent_type, pivot, verb = rng.choice(TYPE_PIVOTS)

    holders_by_value = {}
    for qid in kb.by_type.get(ent_type, ()):
        v = primary_value(kb.by_id[qid], pivot)
        if v is not None:
            holders_by_value.setdefault(v, []).append(qid)

    # Pick the desired count first (uniform over [1, n_total-1]) and
    # only then look for a target_value with enough holders. Sampling
    # the target_value first biases counts low because most values
    # have only 2-3 holders. With this order, the count distribution
    # is uniform modulo eligibility — values with >=k holders shrink
    # as k grows, but the main loop retries on None so the kept
    # samples reflect the actual feasible distribution evenly.
    n_total = rng.randint(5, 8)
    n_holders = rng.randint(1, n_total - 1)
    eligible = [(v, hs) for v, hs in holders_by_value.items()
                if len(hs) >= n_holders]
    if not eligible:
        return None
    target_value, target_holders = rng.choice(eligible)
    holders = rng.sample(target_holders, n_holders)

    pool = [q for q in kb.by_type.get(ent_type, ())
            if (v := primary_value(kb.by_id[q], pivot)) is not None
            and v != target_value]
    if len(pool) < n_total - n_holders:
        return None
    non = rng.sample(pool, n_total - n_holders)

    ent_qids = holders + non
    rng.shuffle(ent_qids)
    ents = [kb.by_id[q] for q in ent_qids]

    aux = [p for p in CATEGORICAL if p != pivot]
    passage = render_passage(kb, ents, [pivot] + aux, rng)

    if not all(e.label in passage for e in ents):
        return None

    target_label = label_of(kb, target_value)
    names = ", ".join(e.label for e in ents)
    q = f"Kiom el {names} {verb} {target_label}?"

    count = sum(1 for e in ents if primary_value(e, pivot) == target_value)
    if count in (0, len(ents)):
        return None
    a = EO_NUMBERS[count].capitalize() + "."

    return {
        "passage": passage, "q": q, "a": a,
        "meta": {"pattern": "count_with_prop", "ent_type": ent_type,
                 "pivot": pivot, "target": target_label,
                 "n_total": n_total, "count": count},
    }


# ---------- pattern: groupby_count ----------

@register("groupby_count")
def groupby_count(kb, rng):
    """N same-type entities, per-group count by pivot value.

    Q: 'Grupigu la jenajn personojn laŭ ŝtataneco: A, B, C, D, E.'
    A: 'Francio: du. Germanio: unu. Hispanio: du.'

    Output is a multi-line per-group count, alphabetical by group
    label, each count spelled out as an Esperanto number word. Tests
    grouping + counting in one shot — a different output shape from
    count_with_prop's single-number answer.

    Constraints:
      - 2-4 distinct pivot values represented (single-group is trivial)
      - 1-3 entities per group (per-group counts stay in [1, 3])
      - N total in [4, 8]
      - alphabetical group order so the answer is canonical and the
        matcher doesn't penalize different orderings of the same set
    """
    ent_type, pivot, verb = rng.choice(TYPE_PIVOTS)

    holders_by_value = {}
    for qid in kb.by_type.get(ent_type, ()):
        v = primary_value(kb.by_id[qid], pivot)
        if v is not None:
            holders_by_value.setdefault(v, []).append(qid)

    candidates = [(v, hs) for v, hs in holders_by_value.items() if hs]
    n_groups = rng.randint(2, 4)
    if len(candidates) < n_groups:
        return None
    chosen_groups = rng.sample(candidates, n_groups)

    chosen = []  # list of (qid, target_value)
    for target_value, holders in chosen_groups:
        k = rng.randint(1, min(3, len(holders)))
        for qid in rng.sample(holders, k):
            chosen.append((qid, target_value))

    if not (4 <= len(chosen) <= 8):
        return None

    rng.shuffle(chosen)
    ents = [kb.by_id[q] for q, _ in chosen]

    aux = [p for p in CATEGORICAL if p != pivot]
    passage = render_passage(kb, ents, [pivot] + aux, rng)

    if not all(e.label in passage for e in ents):
        return None

    # Accusative plural — `grupigu` is transitive imperative so the
    # object is `-jn` (plural + acc), not `-j` (plural nominative).
    type_acc_pl = ent_type + "jn"  # persono -> personojn
    names = ", ".join(e.label for e in ents)
    q = f"Grupigu la jenajn {type_acc_pl} laŭ {pivot}: {names}."

    counts = {}
    for qid, target_value in chosen:
        counts[target_value] = counts.get(target_value, 0) + 1
    # Alphabetical by group label for canonical ordering
    sorted_groups = sorted(counts.items(), key=lambda x: label_of(kb, x[0]))
    if any(c >= len(EO_NUMBERS) for _, c in sorted_groups):
        return None
    a_parts = [f"{label_of(kb, v)}: {EO_NUMBERS[c]}" for v, c in sorted_groups]
    a = ". ".join(a_parts) + "."

    return {
        "passage": passage, "q": q, "a": a,
        "meta": {"pattern": "groupby_count", "ent_type": ent_type,
                 "pivot": pivot, "n_total": len(chosen),
                 "n_groups": len(counts)},
    }


# ---------- pattern: negative_filter ----------

@register("negative_filter")
def negative_filter(kb, rng):
    """4 same-type entities, pick which DO NOT hold pivot=target.

    'Kiuj el A, B, C, D NE estas civitanoj de Francio?'
    Inversion of filter_subset. Tests handling of `NE` (negation) over
    the same passage shape.

    Constraints:
      - 1-2 entities hold the target (so answer = 2-3 non-holders)
      - non-holders have the pivot populated with a different value,
        so the passage shows all four values
    """
    ent_type, pivot, verb = rng.choice(TYPE_PIVOTS)

    holders_by_value = {}
    for qid in kb.by_type.get(ent_type, ()):
        v = primary_value(kb.by_id[qid], pivot)
        if v is not None:
            holders_by_value.setdefault(v, []).append(qid)

    # Target value carries 1-2 holders → answer = 2-3 non-holders
    eligible = [(v, hs) for v, hs in holders_by_value.items() if 1 <= len(hs) <= 3]
    if not eligible:
        return None
    target_value, target_holders = rng.choice(eligible)
    n_holders = rng.randint(1, min(2, len(target_holders)))
    holders = rng.sample(target_holders, n_holders)

    pool = [q for q in kb.by_type.get(ent_type, ())
            if (v := primary_value(kb.by_id[q], pivot)) is not None
            and v != target_value]
    n_non = 4 - n_holders
    if len(pool) < n_non:
        return None
    non = rng.sample(pool, n_non)

    ent_qids = holders + non
    rng.shuffle(ent_qids)
    ents = [kb.by_id[q] for q in ent_qids]

    aux = [p for p in CATEGORICAL if p != pivot]
    passage = render_passage(kb, ents, [pivot] + aux, rng)
    if not all(e.label in passage for e in ents):
        return None

    target_label = label_of(kb, target_value)
    names = ", ".join(e.label for e in ents)
    q = f"Kiuj el {names} NE {verb} {target_label}?"

    correct = [e.label for e in ents if primary_value(e, pivot) != target_value]
    if len(correct) in (0, len(ents)):
        return None
    a = join_eo(correct) + "."

    return {
        "passage": passage, "q": q, "a": a,
        "meta": {"pattern": "negative_filter", "ent_type": ent_type,
                 "pivot": pivot, "excluded_target": target_label,
                 "n_correct": len(correct)},
    }


# ---------- pattern: set_difference ----------

@register("set_difference")
def set_difference(kb, rng):
    """4 persons, answer: those with pivot_a=target_a AND pivot_b != target_b.

    'Kiuj el A, B, C, D estas civitanoj de Francio sed NE naskiĝis en Parizo?'
    Positive + negative composition over two facts. Persons-only for
    the same reason as intersection (geographic two-pivot combos are
    entailment-correlated).

    Constraints:
      - 1-2 entities are A-holders who are NOT B-holders (the answer)
      - 1 entity matches BOTH A and B (the trap — must be excluded)
      - rest are non-A-holders (distractors)
      - answer strictly in (0, 4)
    """
    ent_type = "persono"
    pivots = PIVOTS_BY_TYPE[ent_type]
    (pivot_a, verb_a), (pivot_b, verb_b) = rng.sample(pivots, 2)

    # Group A-holders by their B value
    a_by_b = {}  # target_a -> {val_b: [qids]}
    for qid in kb.by_type.get(ent_type, ()):
        ent = kb.by_id[qid]
        va = primary_value(ent, pivot_a)
        vb = primary_value(ent, pivot_b)
        if va is None or vb is None:
            continue
        a_by_b.setdefault(va, {}).setdefault(vb, []).append(qid)

    # Need target_a with >=2 distinct B-values (one to exclude, one or more for answers)
    candidates = []
    for target_a, b_groups in a_by_b.items():
        if len(b_groups) < 2:
            continue
        for target_b, trap_qids in b_groups.items():
            answer_qids = [q for vb, qs in b_groups.items() if vb != target_b for q in qs]
            if 1 <= len(answer_qids) <= 3 and trap_qids:
                candidates.append((target_a, target_b, answer_qids, trap_qids))
    if not candidates:
        return None
    target_a, target_b, answer_qids, trap_qids = rng.choice(candidates)

    n_answer = rng.randint(1, min(2, len(answer_qids)))
    answers = rng.sample(answer_qids, n_answer)
    trap = rng.sample(trap_qids, 1)

    # Distractors: non-A-holders with both pivots populated
    chosen_set = set(answers) | set(trap)
    non_a = [q for q in kb.by_type.get(ent_type, ())
             if q not in chosen_set
             and (vva := primary_value(kb.by_id[q], pivot_a)) is not None
             and primary_value(kb.by_id[q], pivot_b) is not None
             and vva != target_a]
    n_other = 4 - n_answer - 1
    if len(non_a) < n_other:
        return None
    others = rng.sample(non_a, n_other)

    ent_qids = answers + trap + others
    rng.shuffle(ent_qids)
    ents = [kb.by_id[q] for q in ent_qids]

    passage = render_passage(kb, ents, [pivot_a, pivot_b], rng)
    if not all(e.label in passage for e in ents):
        return None

    target_a_label = label_of(kb, target_a)
    target_b_label = label_of(kb, target_b)
    names = ", ".join(e.label for e in ents)
    q = (f"Kiuj el {names} {verb_a} {target_a_label} "
         f"sed NE {verb_b} {target_b_label}?")

    correct = [e.label for e in ents
               if primary_value(e, pivot_a) == target_a
               and primary_value(e, pivot_b) != target_b]
    if len(correct) in (0, len(ents)):
        return None
    a = join_eo(correct) + "."

    return {
        "passage": passage, "q": q, "a": a,
        "meta": {"pattern": "set_difference", "ent_type": ent_type,
                 "pivots": [pivot_a, pivot_b],
                 "target_a": target_a_label, "excluded_b": target_b_label,
                 "n_correct": len(correct)},
    }


# ---------- pattern: shared_attribute ----------

@register("shared_attribute")
def shared_attribute(kb, rng):
    """4 same-type entities, exactly 2 share a pivot value, the other
    2 have unique distinct values.

    'Kiuj du el A, B, C, D havas la saman ŝtatanecon?'
    Discovery rather than given-target — model must scan the four
    pivot values, find the pair that matches, and name them.

    Constraints:
      - exactly one pair shares (no triples or two pairs)
      - the two unique entities have different values from each
        other and from the shared value
    """
    ent_type, pivot, _ = rng.choice(TYPE_PIVOTS)

    holders_by_value = {}
    for qid in kb.by_type.get(ent_type, ()):
        v = primary_value(kb.by_id[qid], pivot)
        if v is not None:
            holders_by_value.setdefault(v, []).append(qid)

    pair_candidates = [(v, hs) for v, hs in holders_by_value.items() if len(hs) >= 2]
    if not pair_candidates:
        return None
    shared_value, sharers = rng.choice(pair_candidates)
    shared_two = rng.sample(sharers, 2)

    # Two unique entities, with values distinct from each other and shared_value.
    # Pick from distinct value buckets (one entity per bucket).
    unique_buckets = [(v, hs) for v, hs in holders_by_value.items()
                      if v != shared_value]
    if len(unique_buckets) < 2:
        return None
    two_buckets = rng.sample(unique_buckets, 2)
    unique_qids = [rng.choice(hs) for _, hs in two_buckets]

    ent_qids = shared_two + unique_qids
    rng.shuffle(ent_qids)
    ents = [kb.by_id[q] for q in ent_qids]

    aux = [p for p in CATEGORICAL if p != pivot]
    passage = render_passage(kb, ents, [pivot] + aux, rng)
    if not all(e.label in passage for e in ents):
        return None

    # Accusative form. All KB pivots end in -o, so always append -n.
    pivot_acc = pivot + "n"

    names = ", ".join(e.label for e in ents)
    q = f"Kiuj du el {names} havas la saman {pivot_acc}?"

    correct = sorted(kb.by_id[q].label for q in shared_two)
    a = join_eo(correct) + "."

    return {
        "passage": passage, "q": q, "a": a,
        "meta": {"pattern": "shared_attribute", "ent_type": ent_type,
                 "pivot": pivot,
                 "shared_value": label_of(kb, shared_value)},
    }


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", default="filter_subset",
                    help="comma-separated subset of registered pattern names")
    ap.add_argument("--n-per-pattern", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kb", default=str(KB_PATH))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kb = load_kb(args.kb)
    names = [n.strip() for n in args.patterns.split(",")]
    for n in names:
        if n not in PATTERNS:
            sys.exit(f"unknown pattern: {n}; have {sorted(PATTERNS)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = Counter()
    rejected = Counter()
    seen = set()  # dedup by (passage, q)

    with out.open("w") as f:
        for name in names:
            fn = PATTERNS[name]
            attempts = 0
            cap = args.n_per_pattern * 30
            while kept[name] < args.n_per_pattern:
                attempts += 1
                if attempts > cap:
                    print(f"  {name}: gave up at {kept[name]}/{args.n_per_pattern}",
                          file=sys.stderr)
                    break
                rec = fn(kb, rng)
                if rec is None:
                    rejected[name] += 1
                    continue
                key = (rec["passage"], rec["q"])
                if key in seen:
                    rejected[name] += 1
                    continue
                seen.add(key)
                f.write(json.dumps({
                    "messages": [
                        {"role": "user",
                         "content": f"{rec['passage']}\n\n{rec['q']}"},
                        {"role": "assistant", "content": rec["a"]},
                    ],
                    "category": f"wiki_reasoning:{name}",
                    **rec["meta"],
                }, ensure_ascii=False) + "\n")
                kept[name] += 1

    for n in names:
        print(f"  {n}: kept={kept[n]}  rejected={rejected[n]}")
    print(f"wrote {sum(kept.values())} records -> {out}")


if __name__ == "__main__":
    main()
