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


# The real data directory lives next to the ontology module — tests
# use the loader's default. Structural tests (failure paths) write a
# temporary tree. Tests use the real DefaultMorphParser so we catch
# upstream parser changes.


def test_load_real_lexicon():
    lex = load_lexicon()
    assert "persono" in lex.concepts
    assert "pano" in lex.concepts
    assert "tranĉi" in lex.actions


def test_trancxilo_is_derived_not_authored():
    lex = load_lexicon()
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


def _write_min_ontology(root: Path, **overrides):
    """Write a minimal ontology to `root`. `overrides` may set
    'slots'/'concepts'/'actions'/'affixes'/'relations'/'types'
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
        "relations": [],
        "actions": [],
        "affixes": [],
    }
    defaults.update(overrides)
    (root / "types.json").write_text(json.dumps(defaults["types"]))
    for name in ("slots", "concepts", "relations", "actions", "affixes"):
        with open(root / f"{name}.jsonl", "w") as f:
            for d in defaults[name]:
                f.write(json.dumps(d) + "\n")


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


def test_derives_instrument_without_effects_succeeds(tmp_path):
    # A verb flagged derives_instrument but lacking property effects is
    # valid: the functional_signature is just the verb lemma, and the
    # verb's semantics may live in a causal rule (e.g. skribi creates a
    # skribaĵo entity rather than mutating a property). The loader
    # trusts the author.
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
             "trigger_flag": "derives_instrument",
             "signature_source": "effect", "noun_ending": "o"},
        ],
    )
    lex = load_lexicon(tmp_path)
    assert "starilo" in lex.concepts
    assert lex.concepts["starilo"].properties.get(
        "functional_signature") == ["stari"]
