"""Constraint alias + incompatibility table for IF combo sampling.

Companion to `build_grpo_if_rewrite.py`. Replaces the coarse `_kind_of()`
substring-matching approach with two explicit tables:

  1. ALIAS     — namespace duplicate map (custom name → canonical google name)
  2. FORBID    — explicit pair-level incompatibility edges (asymmetric OK)
  3. PARAM_BAD — param-level sanity predicates (e.g. num_sentences < num_paragraphs)

Public API:

  normalize(names)                  → list, applies ALIAS
  duplicates(names)                 → set, names appearing >1 after normalize
  pair_conflicts(names)             → list of (a, b) forbidden pairs present
  param_conflicts(names, params)    → list of str reasons
  is_valid(names, params)           → (bool, list[str])   # convenience

Design notes:
  - `FORBID` is stored as `{canonical_name: {set_of_forbidden_partners}}`.
    Semantics: if `a` is in the combo and any of `FORBID[a]` is also in the
    combo, the combo is invalid.
  - Both directions are populated by _seal_symmetric so callers don't have
    to think about which side to declare.
  - `constrained_response` gets a large forbidden set because it locks the
    answer to one of a handful of fixed short strings, which is
    incompatible with every constraint that demands additional content
    shape.
"""
from __future__ import annotations

from typing import Iterable, Mapping


# ═══════════════════════════════════════════════════════════════════════════
# 1) Namespace alias map — custom our-46 name → canonical google name
# ═══════════════════════════════════════════════════════════════════════════
ALIAS: dict[str, str] = {
    # Duplicates confirmed present in danish-if-grpo-combined-v2:
    "entire_in_quotes":            "google:startend:quotation",
    "ifeval_two_responses_6star":  "google:combination:two_responses",
    "two_responses_split":         "google:combination:two_responses",
    "ifeval_json_format":          "google:detectable_format:json_format",
    "ifeval_postscript":           "google:detectable_content:postscript",
    "no_commas":                   "google:punctuation:no_comma",
    "exactly_n_words":             "google:length_constraints:number_words",
    "at_least_n_words":            "google:length_constraints:number_words",
    "at_most_n_words":             "google:length_constraints:number_words",
    "exactly_n_sentences":         "google:length_constraints:number_sentences",
    "at_least_n_sentences":        "google:length_constraints:number_sentences",
    "n_paragraphs":                "google:length_constraints:number_paragraphs",
    "single_paragraph":            "google:length_constraints:number_paragraphs",
    "n_placeholders":              "google:detectable_content:number_placeholders",
    "n_italic_sections":           "google:detectable_format:number_highlighted_sections",
    "section_headers":             "google:detectable_format:multiple_sections",
    "capital_word_frequency":      "google:change_case:capital_word_frequency",
    "all_lowercase":               "google:change_case:english_lowercase",
    "include_keyword":             "google:keywords:existence",
    "include_all_keywords":        "google:keywords:existence",
    "keyword_exactly_n_times":     "google:keywords:frequency",
    "letter_exactly_n_times":      "google:keywords:letter_frequency",
    "letter_frequency":            "google:keywords:letter_frequency",
    "exclude_word":                "google:keywords:forbidden_words",
    "ends_with_phrase":            "google:startend:end_checker",
    # Rules with no google-namespace equivalent (identity):
    "no_first_person":             "no_first_person",
    "in_second_person":            "in_second_person",
    "first_sentence_max_words":    "first_sentence_max_words",
    "contains_year":               "contains_year",
    "contains_percentage":         "contains_percentage",
    "ends_with_punctuation":       "ends_with_punctuation",
    "markdown_table":              "markdown_table",
    "starts_with_phrase":          "starts_with_phrase",
    "numbered_list_n_items":       "numbered_list_n_items",  # bullet-ish but distinct verifier
    "n_bullets":                   "google:detectable_format:number_bullet_lists",
}


# ═══════════════════════════════════════════════════════════════════════════
# 2) Pair-level incompatibility edges
# ═══════════════════════════════════════════════════════════════════════════

# Any constraint that demands additional CONTENT SHAPE in the response
# (length, structure, wrapping, specific tokens). `constrained_response`
# says the WHOLE answer must equal one of a fixed set of ~5 short strings,
# so nothing on this list can co-exist with it.
_CONTENT_SHAPE = frozenset({
    "google:length_constraints:number_words",
    "google:length_constraints:number_sentences",
    "google:length_constraints:number_paragraphs",
    "google:length_constraints:nth_paragraph_first_word",
    "google:detectable_format:number_bullet_lists",
    "google:detectable_format:number_highlighted_sections",
    "google:detectable_format:multiple_sections",
    "google:detectable_format:json_format",
    "google:detectable_format:title",
    "google:detectable_content:number_placeholders",
    "google:detectable_content:postscript",
    "google:keywords:existence",
    "google:keywords:frequency",
    "google:keywords:letter_frequency",
    "google:keywords:forbidden_words",
    "google:combination:repeat_prompt",
    "google:combination:two_responses",
    "google:startend:quotation",
    "google:startend:end_checker",
    "google:change_case:capital_word_frequency",
    "in_second_person",
    "no_first_person",
    "first_sentence_max_words",
    "contains_year",
    "contains_percentage",
    "markdown_table",
    "starts_with_phrase",
    "ends_with_punctuation",
    "numbered_list_n_items",
})

# JSON output locks surface — anything that adds non-JSON structure conflicts.
_JSON_INCOMPAT = frozenset({
    "google:detectable_format:number_bullet_lists",
    "google:detectable_format:number_highlighted_sections",
    "google:detectable_format:multiple_sections",
    "google:startend:quotation",
    "google:detectable_format:constrained_response",
    "google:combination:two_responses",
    "google:detectable_format:title",
    "markdown_table",
    "numbered_list_n_items",
})

# Whole response wrapped in "…" — can't co-exist with alternative wrappers.
_QUOT_INCOMPAT = frozenset({
    "google:detectable_format:json_format",
    "google:combination:two_responses",
    "markdown_table",
})

# Two responses separated by ****** — can't co-exist with wrappers or a
# single-answer lock.
_TWO_INCOMPAT = frozenset({
    "google:detectable_format:json_format",
    "google:startend:quotation",
    "google:detectable_format:constrained_response",
    "google:combination:repeat_prompt",
})

# Repeat the prompt unchanged — can't force an all-caps or all-lower
# response because the prompt is mixed-case; enforcing case would violate
# "unchanged".
_REPEAT_INCOMPAT = frozenset({
    "google:change_case:english_capital",
    "google:change_case:english_lowercase",
})

# Raw declarations (asymmetric OK; _seal_symmetric mirrors).
_RAW: dict[str, frozenset[str]] = {
    "google:detectable_format:constrained_response": _CONTENT_SHAPE,
    "google:detectable_format:json_format":         _JSON_INCOMPAT,
    "google:startend:quotation":                    _QUOT_INCOMPAT,
    "google:combination:two_responses":             _TWO_INCOMPAT,
    "google:combination:repeat_prompt":             _REPEAT_INCOMPAT,
    "google:change_case:english_capital":           frozenset({"google:change_case:english_lowercase"}),
}


def _seal_symmetric(raw: Mapping[str, frozenset[str]]) -> dict[str, frozenset[str]]:
    """Symmetrize: if A→B is forbidden, add B→A too."""
    out: dict[str, set[str]] = {k: set(v) for k, v in raw.items()}
    for a, partners in raw.items():
        for b in partners:
            out.setdefault(b, set()).add(a)
    return {k: frozenset(v) for k, v in out.items()}


FORBID: dict[str, frozenset[str]] = _seal_symmetric(_RAW)


# ═══════════════════════════════════════════════════════════════════════════
# 3) Param-level sanity checks
# ═══════════════════════════════════════════════════════════════════════════

def _num_sentences_lt_num_paragraphs(names: set[str], p: Mapping) -> str | None:
    if not ({"google:length_constraints:number_sentences",
             "google:length_constraints:number_paragraphs"} <= names):
        return None
    ns, np = p.get("num_sentences"), p.get("num_paragraphs")
    if ns is not None and np is not None and ns < np:
        return f"num_sentences={ns} < num_paragraphs={np}"
    return None


def _nth_paragraph_out_of_range(names: set[str], p: Mapping) -> str | None:
    if not ({"google:length_constraints:nth_paragraph_first_word",
             "google:length_constraints:number_paragraphs"} <= names):
        return None
    nth, np = p.get("nth_paragraph"), p.get("num_paragraphs")
    if nth is not None and np is not None and nth > np:
        return f"nth_paragraph={nth} > num_paragraphs={np}"
    return None


def _keyword_required_and_forbidden(names: set[str], p: Mapping) -> str | None:
    if not ({"google:keywords:existence",
             "google:keywords:forbidden_words"} <= names):
        return None
    req = {str(k).lower() for k in (p.get("keywords") or [])}
    forb = {str(k).lower() for k in (p.get("forbidden_words") or [])}
    overlap = req & forb
    if overlap:
        return f"keywords required and forbidden: {sorted(overlap)}"
    return None


PARAM_CHECKS = [
    _num_sentences_lt_num_paragraphs,
    _nth_paragraph_out_of_range,
    _keyword_required_and_forbidden,
]


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def normalize(names: Iterable[str]) -> list[str]:
    """Apply ALIAS to each name. Unknown names pass through unchanged."""
    return [ALIAS.get(n, n) for n in names]


def duplicates(names: Iterable[str]) -> set[str]:
    """Names appearing >1 time after normalize."""
    seen: set[str] = set()
    dupes: set[str] = set()
    for n in names:
        if n in seen:
            dupes.add(n)
        seen.add(n)
    return dupes


def pair_conflicts(names: Iterable[str]) -> list[tuple[str, str]]:
    """List of (a, b) pairs present in `names` that are on each other's FORBID list.
    Each pair appears once (sorted)."""
    name_set = set(names)
    out: set[tuple[str, str]] = set()
    for a in name_set:
        for b in FORBID.get(a, ()):
            if b in name_set and a != b:
                out.add(tuple(sorted((a, b))))
    return sorted(out)


def param_conflicts(names: Iterable[str], params: Mapping) -> list[str]:
    """Run PARAM_CHECKS; return list of human-readable failure reasons."""
    name_set = set(names)
    reasons: list[str] = []
    for check in PARAM_CHECKS:
        r = check(name_set, params)
        if r is not None:
            reasons.append(r)
    return reasons


def is_valid(names: Iterable[str], params: Mapping) -> tuple[bool, list[str]]:
    """One-shot: normalize, then check duplicates + pair + param. Returns (ok, reasons)."""
    canonical = normalize(names)
    reasons: list[str] = []
    dupes = duplicates(canonical)
    if dupes:
        reasons.append(f"duplicate after alias: {sorted(dupes)}")
    for a, b in pair_conflicts(canonical):
        reasons.append(f"forbidden pair: {a} × {b}")
    reasons.extend(param_conflicts(canonical, params))
    return (not reasons), reasons


# ═══════════════════════════════════════════════════════════════════════════
# Merger helper for the combo-of-dicts shape used by sample_combined_combo
# ═══════════════════════════════════════════════════════════════════════════

def merge_combo_params(combo: Iterable[Mapping]) -> dict:
    """combo is a list of {name, params, ...}. Merge params dicts into one
    flat map, keeping first non-None value per key (matches how the dataset
    stores its `params` list-of-dicts too)."""
    out: dict = {}
    for rule in combo:
        p = rule.get("params") or {}
        for k, v in p.items():
            if v is not None:
                out.setdefault(k, v)
    return out
