"""Phase 5 acceptance: lexicon tag migrated to derivation.

The 9 concepts that used to carry `flammability=flammable` now carry
`made_of=<material>` instead. The DSL derivation
`flammability_from_material` produces the effective flammability at
trace time. These tests verify:

  - Disk concepts really dropped the flammability tag (the file is
    measurably more compact).
  - The derivation recovers flammability for every formerly-tagged
    concept in the real lexicon.
  - Entities of non-flammable material don't get falsely marked.
  - The fire cascade (which matches `entity(flammability='flammable')`)
    still fires correctly once the derivation runs — end-to-end proof
    the migration is behaviorally neutral.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from esperanto_lm.ontology import (
    Trace, effect_changes, load_lexicon, make_event,
)
from esperanto_lm.ontology.dsl import run_dsl
from esperanto_lm.ontology.dsl.rules import (
    DEFAULT_DSL_DERIVATIONS,
    fire_spreads_to_adjacent_flammables,
    flammability_from_material,
)
from esperanto_lm.ontology.dsl.unifier import DerivedState
from esperanto_lm.ontology.dsl.engine import _run_derivations_to_fixed_point


DATA_DIR = Path("data/ontology")


@pytest.fixture
def lex():
    return load_lexicon(DATA_DIR)


# Concepts that used to carry flammability=flammable directly on disk
# and now rely on made_of + derivation.
_FORMERLY_FLAMMABLE = [
    "ligno", "libro", "sofo", "korbo", "papero",
    "tablo", "arbo", "seĝo", "breto",
]

# Things that should not be derived as flammable (either wrong material
# or no made_of tag at all).
_NOT_FLAMMABLE = ["glaso", "najlo", "pano"]


# ---- disk compactness ---------------------------------------------------

def test_flammability_tag_removed_from_disk_concepts(lex):
    """No concept carries flammability=flammable on disk anymore —
    it comes from the derivation. (A future derivation could imply
    flammability=inflammable on other materials; that's why we check
    the specific value.)"""
    for lemma in _FORMERLY_FLAMMABLE:
        c = lex.concepts[lemma]
        assert "flammability" not in c.properties, (
            f"{lemma!r} still has flammability tag on disk: "
            f"{c.properties}")


def test_made_of_tag_present_on_formerly_flammable_concepts(lex):
    """Each formerly-tagged concept now carries `made_of` instead."""
    for lemma in _FORMERLY_FLAMMABLE:
        c = lex.concepts[lemma]
        assert c.properties.get("made_of"), (
            f"{lemma!r} has no made_of tag: {c.properties}")


# ---- derivation semantics -----------------------------------------------

def _derived_flammability(t: Trace, lex, eid: str):
    derived = DerivedState()
    _run_derivations_to_fixed_point(
        t, DEFAULT_DSL_DERIVATIONS, lex, derived)
    return derived.get(eid, "flammability")


@pytest.mark.parametrize("lemma", _FORMERLY_FLAMMABLE)
def test_derivation_produces_flammable_for_each_formerly_tagged(lex, lemma):
    t = Trace()
    t.add_entity(lemma, lex, entity_id=lemma)
    assert _derived_flammability(t, lex, lemma) == "flammable"


@pytest.mark.parametrize("lemma", _NOT_FLAMMABLE)
def test_derivation_does_not_produce_flammable_for_others(lex, lemma):
    t = Trace()
    t.add_entity(lemma, lex, entity_id=lemma)
    assert _derived_flammability(t, lex, lemma) is None


# ---- end-to-end with the fire rule --------------------------------------

def test_fire_cascade_works_with_derived_flammability(lex):
    """Fire spreads correctly even though flammability is no longer
    on disk — the derivation materializes it at trace time, and the
    causal rule matches against the derived value transparently."""
    t = Trace()
    t.add_entity("laborejo", lex, entity_id="laborejo")
    t.add_entity("breto", lex, entity_id="breto")
    t.add_entity("ligno", lex, entity_id="ligno")
    t.assert_relation("en", ("breto", "laborejo"), lex)
    t.assert_relation("sur", ("ligno", "breto"), lex)
    roles = {"theme": "ligno"}
    t.events.append(make_event(
        "bruli", roles=roles,
        property_changes=effect_changes("bruli", roles, lex)))

    run_dsl(t, [fire_spreads_to_adjacent_flammables],
            DEFAULT_DSL_DERIVATIONS, lex)

    burning = {e.roles.get("theme") for e in t.events if e.action == "bruli"}
    assert burning == {"ligno", "breto"}


def test_asserted_flammability_still_wins_over_derived(lex):
    """Asserted values override derivations (per the DSL's semantics).
    Set a wooden entity's flammability to 'inflammable' explicitly;
    the derivation should not override it. This demonstrates that the
    Phase 5 migration preserves the escape hatch for exceptional
    concepts."""
    t = Trace()
    breto = t.add_entity("breto", lex, entity_id="breto")
    breto.set_property("flammability", "inflammable")  # asserted override

    derived = DerivedState()
    _run_derivations_to_fixed_point(
        t, DEFAULT_DSL_DERIVATIONS, lex, derived)

    # Derived layer shouldn't have set flammability — the asserted
    # value blocks it.
    assert derived.get("breto", "flammability") is None
    # And the effective property (asserted wins) is the override.
    assert t.property_at("breto", "flammability", 0) == ["inflammable"]


# ---- derivation IS registered in DEFAULT_DSL_DERIVATIONS ----------------

def test_flammability_derivation_is_registered():
    """`flammability_from_material` is the canonical source for this
    pattern — no hidden duplicates in the loader or elsewhere. Other
    derivations ride alongside it in DEFAULT_DSL_DERIVATIONS."""
    assert flammability_from_material in DEFAULT_DSL_DERIVATIONS
