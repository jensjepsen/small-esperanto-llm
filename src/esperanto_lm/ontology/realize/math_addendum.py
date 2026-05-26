"""Quantitative addendum sentences appended to realized prose.

A small post-pass that scans the trace for numeric facts already
encoded in the schema (count slot, maso, alteco, …) and emits 1-2
math sentences in Esperanto. Trains the LM on light commonsense
math grounded in scene state.

Conservative: each statement type runs at most once per trace, and
only when the trace has enough material to make the math non-
trivial (≥2 same-category entities for aggregation, ≥2 mass-bearing
entities for comparison).
"""
from __future__ import annotations

import random
from typing import Optional

from ..causal import Trace
from ..loader import Lexicon


# Esperanto cardinals 1..20. Compositions (dudek tri, cent, mil) for
# the rare cases mass comparisons need them are computed by
# `_cardinal_eo`.
_CARDINALS = [
    "nul", "unu", "du", "tri", "kvar", "kvin",
    "ses", "sep", "ok", "naŭ", "dek",
    "dek unu", "dek du", "dek tri", "dek kvar", "dek kvin",
    "dek ses", "dek sep", "dek ok", "dek naŭ", "dudek",
]


def _cardinal_eo(n: int) -> str:
    """Esperanto cardinal number in words."""
    if 0 <= n < len(_CARDINALS):
        return _CARDINALS[n]
    parts = []
    if n >= 100:
        h = n // 100
        parts.append("cent" if h == 1 else f"{_CARDINALS[h]}cent")
        n %= 100
    if n >= 10:
        tens, units = divmod(n, 10)
        head = _CARDINALS[tens] if tens == 1 else f"{_CARDINALS[tens]}dek"
        parts.append(head)
        if units > 0:
            parts.append(_CARDINALS[units])
    elif n > 0:
        parts.append(_CARDINALS[n])
    return " ".join(parts) if parts else "nul"


def _plural_form(noun: str) -> str:
    """Add -j plural suffix. Esperanto morphology — noun must end in
    -o (the substantive ending) for this to be safe; the categories
    we read (frukto, legomo, manĝaĵo, bero) all qualify."""
    return noun + "j" if noun.endswith("o") else noun


def _number_phrase(n: int, noun: str) -> str:
    """Esperanto number + noun agreement: `unu` keeps the noun in
    singular (-o), `du+` pluralizes (-oj)."""
    return f"{_cardinal_eo(n)} {noun if n == 1 else _plural_form(noun)}"


def _aggregate_by_category(
    trace: Trace, lex: Lexicon,
) -> Optional[tuple[str, int, list[tuple[str, int]]]]:
    """Find a category with ≥2 distinct concept instances summing to
    ≥3, return (category_lemma, total_count, [(concept_lemma, count),
    …]) for the largest such bundle. Returns None if no qualifying
    bundle exists. Used to emit `Sume estas N fruktoj.`-style
    aggregation sentences.

    Skips parts (entities that appear as the second arg of
    `havas_parton`), `mondo`, and entities whose host concept lacks
    a category. The category lemma is picked among the concept's
    declared categories — prefer the first one for stability."""
    part_eids = {
        r.args[1] for r in trace.relations
        if r.relation == "havas_parton" and len(r.args) == 2
    }
    by_cat: dict[str, dict[str, int]] = {}
    for eid, ent in trace.entities.items():
        if eid in ("mondo",) or eid in part_eids:
            continue
        if ent.entity_type == "location":
            continue
        concept = lex.concepts.get(ent.concept_lemma)
        if concept is None or not concept.category:
            continue
        # Prefer a non-stub category. concept.category is a list of
        # lemmas; skip categories that aren't themselves concepts (rare
        # but possible — `arbo` is a category but also a stub).
        category = concept.category[0]
        # Read count from properties (set by realizer/seeder); default 1.
        count_vals = ent.properties.get("count")
        try:
            n = int(count_vals[0]) if count_vals else 1
        except (ValueError, TypeError):
            n = 1
        if n < 1:
            continue
        bucket = by_cat.setdefault(category, {})
        bucket[ent.concept_lemma] = bucket.get(ent.concept_lemma, 0) + n
    candidates: list[tuple[str, int, list[tuple[str, int]]]] = []
    for cat, bucket in by_cat.items():
        if len(bucket) < 2:
            continue
        total = sum(bucket.values())
        if total < 3:
            continue
        candidates.append((cat, total, sorted(bucket.items())))
    if not candidates:
        return None
    # Pick the bundle with the most instances, ties broken by total.
    candidates.sort(key=lambda c: (len(c[2]), c[1]), reverse=True)
    return candidates[0]


def math_addendum(
    trace: Trace, lex: Lexicon, *,
    rng: Optional[random.Random] = None,
) -> str:
    """Return a string of 0-2 math sentences derived from the trace's
    quantitative state. Empty when nothing interesting fits.

    Currently emits one type: category aggregation
    (`Du pomoj kaj tri oranĝoj — sume kvin fruktoj.`). Other types
    (mass comparison, height comparison) will plug in here as siblings.
    """
    sentences: list[str] = []
    bundle = _aggregate_by_category(trace, lex)
    if bundle is not None:
        cat, total, items = bundle
        # Aggregation: "Du pomoj kaj tri oranĝoj — sume kvin fruktoj."
        parts = [_number_phrase(n, lemma) for lemma, n in items]
        if len(parts) == 1:
            phrase = parts[0]
        elif len(parts) == 2:
            phrase = f"{parts[0]} kaj {parts[1]}"
        else:
            phrase = ", ".join(parts[:-1]) + f", kaj {parts[-1]}"
        sentences.append(
            f"{phrase[0].upper() + phrase[1:]} — sume "
            f"{_number_phrase(total, cat)}."
        )
        # Subtraction: derive from the same bundle by removing one
        # concept's contribution. "Se oni forprenas la N X-ojn,
        # restas M Y-oj." Only emit when the bundle has exactly two
        # distinct concepts (the math stays clean) and the rng
        # consents — half the time we just leave the aggregation,
        # half the time we extend with subtraction, so the corpus
        # gets a mix of pure-sum and sum-with-followup.
        if rng is None or rng.random() < 0.5:
            # Pick one concept to remove. For 2-item bundles, the
            # remaining is a single concept (so we name it). For 3+
            # item bundles, the remaining is mixed, so we fall back
            # to the parent category name.
            idx = 0 if rng is None else rng.randrange(len(items))
            removed_lemma, removed_n = items[idx]
            remaining = total - removed_n
            from .render import to_accusative
            removed_acc = to_accusative(
                "la " + _number_phrase(removed_n, removed_lemma))
            if len(items) == 2:
                kept_lemma, _kept_n = next(
                    it for it in items if it[0] != removed_lemma)
                remaining_phrase = _number_phrase(remaining, kept_lemma)
            else:
                remaining_phrase = _number_phrase(remaining, cat)
            sentences.append(
                f"Se oni forprenas {removed_acc}, restas "
                f"{remaining_phrase}."
            )
    return " ".join(sentences)
