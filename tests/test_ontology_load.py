"""Tests for lexicon load + compositional derivation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from esperanto_lm.ontology import (
    FUNCTIONAL_SIGNATURE,
    load_lexicon,
)
from esperanto_lm.ontology.loader import resolve_signature


# Use the real data directory for the prototype's golden cases. Structural
# tests (failure paths) write a temporary tree. Tests use the real
# DefaultMorphParser so we catch upstream parser changes; the StubMorphParser
# remains available for fixtures that don't want the root dictionary.
DATA_DIR = Path("data/ontology")


def test_load_real_lexicon():
    lex = load_lexicon(DATA_DIR)
    assert "persono" in lex.concepts
    assert "pano" in lex.concepts
    assert "tranĉi" in lex.actions


def test_trancxilo_is_derived_not_authored():
    lex = load_lexicon(DATA_DIR)
    knife = lex.concepts["tranĉilo"]
    assert knife.derived is True
    assert knife.derived_from == {"verb": "tranĉi", "affix": "il"}
    assert knife.entity_type == "artifact"
    # Functional signature points at tranĉi.
    sig = knife.properties.get(FUNCTIONAL_SIGNATURE)
    assert sig == ["tranĉi"]
    verb = resolve_signature(lex, knife)
    assert verb is not None
    assert verb.lemma == "tranĉi"
    # ŝlosilo also derived (from ŝlosi which has derives_instrument=True).
    assert "ŝlosilo" in lex.concepts
    assert lex.concepts["ŝlosilo"].derived is True


def test_quality_applies_to_derived_from_slot(tmp_path):
    lex = load_lexicon(DATA_DIR)
    # rompebla names slot=fragility, which applies_to=[inanimate]
    assert lex.quality_applies_to["rompebla"] == ["inanimate"]
    # varma names slot=temperature, which applies_to=[physical]
    assert lex.quality_applies_to["varma"] == ["physical"]


def _write_min_ontology(root: Path, **overrides):
    """Write a minimal ontology to `root`. `overrides` may set
    'slots'/'concepts'/'qualities'/'actions'/'affixes'/'relations'/'types'
    to a list of dicts (or dict for types) which will replace the default."""
    defaults = {
        "types": {"physical": None, "artifact": "physical"},
        "slots": [
            {"name": "color", "vocabulary": ["red", "blue"],
             "applies_to": ["physical"], "scalar": True},
        ],
        "concepts": [
            {"lemma": "domo", "entity_type": "artifact",
             "properties": {"color": ["red"]}},
        ],
        "qualities": [],
        "relations": [],
        "actions": [],
        "affixes": [],
    }
    defaults.update(overrides)
    (root / "types.json").write_text(json.dumps(defaults["types"]))
    for name in ("slots", "concepts", "qualities", "relations",
                 "actions", "affixes"):
        with open(root / f"{name}.jsonl", "w") as f:
            for d in defaults[name]:
                f.write(json.dumps(d) + "\n")


def test_quality_with_unknown_slot_fails(tmp_path):
    _write_min_ontology(
        tmp_path,
        qualities=[{"lemma": "ruĝa", "slot": "no_such_slot", "value": "red"}],
    )
    with pytest.raises(ValueError, match="no_such_slot"):
        load_lexicon(tmp_path)


def test_concept_value_outside_slot_vocabulary_fails(tmp_path):
    _write_min_ontology(
        tmp_path,
        concepts=[{"lemma": "domo", "entity_type": "artifact",
                   "properties": {"color": ["chartreuse"]}}],
    )
    with pytest.raises(ValueError, match="chartreuse"):
        load_lexicon(tmp_path)


def test_concept_with_slot_not_applicable_to_type_fails(tmp_path):
    # Slot only applies to 'artifact'; concept has type 'physical' which
    # is the parent. is_subtype(physical, artifact) is False -> reject.
    _write_min_ontology(
        tmp_path,
        slots=[{"name": "color", "vocabulary": ["red"],
                "applies_to": ["artifact"], "scalar": True}],
        concepts=[{"lemma": "io", "entity_type": "physical",
                   "properties": {"color": ["red"]}}],
    )
    with pytest.raises(ValueError, match="does not apply"):
        load_lexicon(tmp_path)


def test_unknown_entity_type_fails(tmp_path):
    _write_min_ontology(
        tmp_path,
        concepts=[{"lemma": "io", "entity_type": "ufo", "properties": {}}],
    )
    with pytest.raises(ValueError, match="ufo"):
        load_lexicon(tmp_path)


def test_derives_instrument_without_effects_fails(tmp_path):
    # A verb flagged for instrument derivation but with no non-instrument
    # effects has nothing to project as functional signature.
    _write_min_ontology(
        tmp_path,
        types={"physical": None, "artifact": "physical",
               "person": "physical"},
        slots=[{"name": "functional_signature", "vocabulary": None,
                "applies_to": ["artifact"], "scalar": True}],
        concepts=[],
        actions=[
            {"lemma": "stari", "transitivity": "intransitive",
             "aspect": "state",
             "roles": [{"name": "agent", "type": "person",
                        "properties": {}}],
             "effects": [],
             "derives_instrument": True},
        ],
        affixes=[
            {"form": "il", "kind": "suffix", "attaches_to": "verb",
             "produces": "noun", "output_type": "artifact",
             "signature_source": "effect", "noun_ending": "o"},
        ],
    )
    with pytest.raises(ValueError, match="no effects"):
        load_lexicon(tmp_path)
