"""Tests for ContainmentFact loading + the pattern resolver."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from esperanto_lm.ontology import (
    Lexicon,
    containment_relation_for,
    load_lexicon,
    reachable_from,
    resolve_containment,
)


# Helper: build a tiny lexicon at tmp_path with a containment.jsonl payload.
def _write_lexicon(tmp: Path, *, containment: list[dict] | None = None,
                   extra_concepts: list[dict] | None = None,
                   extra_slots: list[dict] | None = None) -> None:
    types = {
        "physical": None,
        "inanimate": "physical",
        "animate": "physical",
        "location": "physical",
        "artifact": "inanimate",
        "substance": "inanimate",
        "person": "animate",
    }
    slots = [
        {"name": "indoor_outdoor",
         "vocabulary": ["indoor", "outdoor"],
         "applies_to": ["location"], "scalar": True},
    ]
    if extra_slots:
        slots.extend(extra_slots)
    concepts = [
        {"lemma": "kuirejo", "entity_type": "location",
         "properties": {"indoor_outdoor": ["indoor"]}},
        {"lemma": "ĝardeno", "entity_type": "location",
         "properties": {"indoor_outdoor": ["outdoor"]}},
        {"lemma": "tablo", "entity_type": "artifact", "properties": {}},
        {"lemma": "pordo", "entity_type": "artifact", "properties": {}},
        {"lemma": "muro", "entity_type": "artifact", "properties": {}},
        {"lemma": "akvo", "entity_type": "substance", "properties": {}},
    ]
    if extra_concepts:
        concepts.extend(extra_concepts)
    relations = [
        {"name": "en", "arity": 2, "arg_types": ["physical", "physical"],
         "arg_names": ["contained", "container"],
         "inverse": None, "symmetric": False},
        {"name": "sur", "arity": 2, "arg_types": ["physical", "physical"],
         "arg_names": ["contained", "container"],
         "inverse": None, "symmetric": False},
    ]
    (tmp / "types.json").write_text(json.dumps(types))
    for name, payload in [
        ("slots", slots), ("concepts", concepts), ("relations", relations),
        ("qualities", []), ("actions", []), ("affixes", []),
    ]:
        with open(tmp / f"{name}.jsonl", "w") as f:
            for entry in payload:
                f.write(json.dumps(entry) + "\n")
    if containment is not None:
        with open(tmp / "containment.jsonl", "w") as f:
            for entry in containment:
                f.write(json.dumps(entry) + "\n")


# ---- loader normalization + validation -----------------------------------

def test_specific_container_normalizes_to_pattern(tmp_path):
    _write_lexicon(tmp_path, containment=[
        {"container": "kuirejo", "contained": "tablo", "relation": "en"},
    ])
    lex = load_lexicon(tmp_path)
    assert len(lex.containment) == 1
    fact = lex.containment[0]
    assert fact.container is None  # normalized away
    assert fact.container_pattern is not None
    assert fact.container_pattern.sense_id == "kuirejo"
    assert fact.contained == "tablo"
    assert fact.relation == "en"


def test_pattern_with_suffix_loads(tmp_path):
    _write_lexicon(tmp_path, containment=[
        {"container_pattern": {"suffix": "ej"}, "contained": "tablo",
         "relation": "en"},
    ])
    lex = load_lexicon(tmp_path)
    assert lex.containment[0].container_pattern.suffix == "ej"


def test_setting_both_container_and_pattern_fails(tmp_path):
    _write_lexicon(tmp_path, containment=[
        {"container": "kuirejo",
         "container_pattern": {"sense_id": "kuirejo"},
         "contained": "tablo", "relation": "en"},
    ])
    with pytest.raises(ValueError, match="exactly one"):
        load_lexicon(tmp_path)


def test_empty_pattern_fails(tmp_path):
    _write_lexicon(tmp_path, containment=[
        {"container_pattern": {}, "contained": "tablo", "relation": "en"},
    ])
    with pytest.raises(ValueError, match="at least one"):
        load_lexicon(tmp_path)


def test_unknown_relation_fails(tmp_path):
    _write_lexicon(tmp_path, containment=[
        {"container": "kuirejo", "contained": "tablo", "relation": "ŝvebas"},
    ])
    with pytest.raises(ValueError, match="unknown relation"):
        load_lexicon(tmp_path)


def test_unknown_sense_id_fails(tmp_path):
    _write_lexicon(tmp_path, containment=[
        {"container": "no_such_concept", "contained": "tablo",
         "relation": "en"},
    ])
    with pytest.raises(ValueError, match="not a known concept"):
        load_lexicon(tmp_path)


def test_unknown_entity_type_fails(tmp_path):
    _write_lexicon(tmp_path, containment=[
        {"container_pattern": {"entity_type": "no_such_type"},
         "contained": "tablo", "relation": "en"},
    ])
    with pytest.raises(ValueError, match="not in the type spine"):
        load_lexicon(tmp_path)


def test_contained_can_be_a_type_not_a_concept(tmp_path):
    _write_lexicon(tmp_path, containment=[
        {"container_pattern": {"entity_type": "location"},
         "contained": "artifact", "relation": "en"},
    ])
    lex = load_lexicon(tmp_path)  # should succeed
    assert lex.containment[0].contained == "artifact"


# ---- resolver + reachability --------------------------------------------

def test_sense_id_pattern_matches_only_named_concept(tmp_path):
    _write_lexicon(tmp_path, containment=[
        {"container": "kuirejo", "contained": "tablo", "relation": "en"},
    ])
    lex = load_lexicon(tmp_path)
    idx = resolve_containment(lex)
    assert "kuirejo" in idx
    assert len(idx["kuirejo"]) == 1
    assert idx["kuirejo"][0].contained == "tablo"
    # ĝardeno is also a location but only kuirejo was named.
    assert "ĝardeno" not in idx


def test_entity_type_pattern_matches_all_subtypes(tmp_path):
    _write_lexicon(tmp_path, containment=[
        {"container_pattern": {"entity_type": "location"},
         "contained": "pordo", "relation": "en"},
    ])
    lex = load_lexicon(tmp_path)
    idx = resolve_containment(lex)
    assert "pordo" in {f.contained for f in idx.get("kuirejo", [])}
    assert "pordo" in {f.contained for f in idx.get("ĝardeno", [])}


def test_property_pattern_matches_concepts_with_value(tmp_path):
    _write_lexicon(tmp_path, containment=[
        {"container_pattern": {"property": {"indoor_outdoor": "indoor"}},
         "contained": "muro", "relation": "en"},
    ])
    lex = load_lexicon(tmp_path)
    idx = resolve_containment(lex)
    # kuirejo is indoor_outdoor=indoor; ĝardeno is outdoor.
    assert "muro" in {f.contained for f in idx.get("kuirejo", [])}
    assert "muro" not in {f.contained for f in idx.get("ĝardeno", [])}


def test_conjunction_requires_all_fields_match(tmp_path):
    """Pattern with both entity_type AND property — both must match."""
    _write_lexicon(tmp_path, containment=[
        {"container_pattern": {
            "entity_type": "location",
            "property": {"indoor_outdoor": "outdoor"},
         },
         "contained": "muro", "relation": "en"},
    ])
    lex = load_lexicon(tmp_path)
    idx = resolve_containment(lex)
    # Only ĝardeno is location AND outdoor.
    assert "muro" not in {f.contained for f in idx.get("kuirejo", [])}
    assert "muro" in {f.contained for f in idx.get("ĝardeno", [])}


def test_reachable_from_walks_transitively(tmp_path):
    _write_lexicon(
        tmp_path,
        extra_concepts=[
            {"lemma": "pano", "entity_type": "substance", "properties": {}},
            {"lemma": "glaso", "entity_type": "artifact", "properties": {}},
        ],
        containment=[
            {"container": "kuirejo", "contained": "tablo", "relation": "en"},
            {"container": "kuirejo", "contained": "glaso", "relation": "en"},
            {"container": "tablo", "contained": "pano", "relation": "sur"},
            {"container": "glaso", "contained": "akvo", "relation": "en"},
        ],
    )
    lex = load_lexicon(tmp_path)
    idx = resolve_containment(lex)
    reach = reachable_from("kuirejo", idx, lex)
    # Direct: tablo, glaso. Indirect: pano (via tablo), akvo (via glaso).
    assert {"kuirejo", "tablo", "glaso", "pano", "akvo"} <= reach


def test_containment_relation_for_lookup(tmp_path):
    _write_lexicon(tmp_path, containment=[
        {"container": "kuirejo", "contained": "tablo", "relation": "en"},
    ])
    lex = load_lexicon(tmp_path)
    idx = resolve_containment(lex)
    assert containment_relation_for("kuirejo", "tablo", idx, lex) == "en"
    assert containment_relation_for("kuirejo", "muro", idx, lex) is None


def test_contains_pattern_second_order(tmp_path):
    """Second-order pattern: container matches if its first-order
    reachability includes the named lemma. Two-pass resolver."""
    _write_lexicon(
        tmp_path,
        extra_concepts=[
            {"lemma": "ŝlosilo", "entity_type": "artifact", "properties": {}},
        ],
        containment=[
            # Pass 1: every location can contain a door.
            {"container_pattern": {"entity_type": "location"},
             "contained": "pordo", "relation": "en"},
            # Pass 2: anywhere with a door also has a key.
            {"container_pattern": {"contains": "pordo"},
             "contained": "ŝlosilo", "relation": "en"},
        ],
    )
    lex = load_lexicon(tmp_path)
    idx = resolve_containment(lex)
    # Both kuirejo and ĝardeno (both locations) contain pordo via pass 1,
    # so both should also contain ŝlosilo via pass 2.
    assert "ŝlosilo" in {f.contained for f in idx.get("kuirejo", [])}
    assert "ŝlosilo" in {f.contained for f in idx.get("ĝardeno", [])}
    # tablo is an artifact, not a location; doesn't contain pordo via pass 1,
    # so the second-order pattern doesn't apply.
    assert "ŝlosilo" not in {f.contained for f in idx.get("tablo", [])}


def test_contains_pattern_does_not_see_other_second_order(tmp_path):
    """No nested second-order: pass 2 facts can only reference pass 1
    reachability, not each other."""
    _write_lexicon(
        tmp_path,
        extra_concepts=[
            {"lemma": "ŝlosilo", "entity_type": "artifact", "properties": {}},
            {"lemma": "lampo", "entity_type": "artifact", "properties": {}},
        ],
        containment=[
            {"container_pattern": {"entity_type": "location"},
             "contained": "pordo", "relation": "en"},
            # Pass 2: where doors are, there are keys.
            {"container_pattern": {"contains": "pordo"},
             "contained": "ŝlosilo", "relation": "en"},
            # Pass 2 again: where keys are, there are lamps.
            # This MUST NOT fire because ŝlosilo was added in pass 2,
            # so ŝlosilo isn't in pass-1 reachability of any concept.
            {"container_pattern": {"contains": "ŝlosilo"},
             "contained": "lampo", "relation": "en"},
        ],
    )
    lex = load_lexicon(tmp_path)
    idx = resolve_containment(lex)
    # ŝlosilo gets in via pass 2 (good).
    assert "ŝlosilo" in {f.contained for f in idx.get("kuirejo", [])}
    # lampo does NOT get in, because the contains:ŝlosilo pattern can only
    # see pass 1 (where ŝlosilo isn't reachable from any container).
    assert "lampo" not in {f.contained for f in idx.get("kuirejo", [])}


def test_contains_pattern_unknown_lemma_fails(tmp_path):
    _write_lexicon(tmp_path, containment=[
        {"container_pattern": {"contains": "no_such_concept"},
         "contained": "tablo", "relation": "en"},
    ])
    with pytest.raises(ValueError, match="not a known concept"):
        load_lexicon(tmp_path)


def test_specific_and_pattern_entries_union_for_same_container(tmp_path):
    _write_lexicon(tmp_path, containment=[
        {"container": "kuirejo", "contained": "tablo", "relation": "en"},
        {"container_pattern": {"entity_type": "location"},
         "contained": "pordo", "relation": "en"},
    ])
    lex = load_lexicon(tmp_path)
    idx = resolve_containment(lex)
    contained_for_kuirejo = {f.contained for f in idx["kuirejo"]}
    # Specific (tablo) + pattern (pordo) both apply to kuirejo.
    assert contained_for_kuirejo == {"tablo", "pordo"}
