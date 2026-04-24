"""Tests for the causal trace + rule engine."""
from __future__ import annotations

from pathlib import Path

import pytest

from esperanto_lm.ontology import (
    Trace,
    load_lexicon,
    make_event,
)
from esperanto_lm.ontology.dsl import run_dsl
from esperanto_lm.ontology.dsl.rules import (
    DEFAULT_DSL_RULES,
    make_use_instrument_rules,
)


def _run(t, lex):
    return run_dsl(
        t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex), [], lex)


DATA_DIR = Path("data/ontology")



@pytest.fixture
def lex():
    return load_lexicon(DATA_DIR)


def _kitchen_trace(lex):
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("tablo", lex, entity_id="tablo")
    t.add_entity("pano", lex, entity_id="pano")
    t.add_entity("tranĉilo", lex, entity_id="tranĉilo")
    t.assert_relation("en", ("petro", "kuirejo"), lex)
    t.assert_relation("havi", ("petro", "tranĉilo"), lex)
    t.assert_relation("sur", ("pano", "tablo"), lex)
    return t


def test_use_instrument_synthesizes_cut_event(lex):
    t = _kitchen_trace(lex)
    seed = make_event("uzi", roles={
        "agent": "petro", "instrument": "tranĉilo", "theme": "pano"})
    t.add_event(seed)

    iters = _run(t, lex)

    actions = [e.action for e in t.events]
    assert "uzi" in actions
    assert "tranĉi" in actions, "generic rule should fire tranĉi"
    cut = next(e for e in t.events if e.action == "tranĉi")
    assert seed.id in cut.caused_by
    # Engine should converge in 2 passes (1 to fire tranĉi, 1 to confirm
    # no more rules apply). Tolerate one extra to be conservative.
    assert iters <= 3


def test_cut_event_mutates_pano_integrity(lex):
    t = _kitchen_trace(lex)
    assert t.entities["pano"].properties.get("integrity") == ["intact"]
    t.add_event(make_event("uzi", roles={
        "agent": "petro", "instrument": "tranĉilo", "theme": "pano"}))
    _run(t, lex)
    assert t.property_at("pano", "integrity", len(t.events)) == "severed"


def test_theme_type_mismatch_yields_no_synthesis(lex):
    # Try to "cut" the kitchen (location, not physical-substance-ish).
    # tranĉi's theme role accepts 'physical', and 'location' is a child of
    # 'physical' in our spine, so the type check passes — switch to a
    # mismatch the spine actually rejects: try to use the knife on an
    # abstract 'event'... but we don't have any abstract entities. Use
    # the lock instead, which is wrong-typed for tranĉi's theme? Actually
    # tranĉi.theme = physical, and pordo is artifact -> physical, so it'd
    # pass. The real mismatch test: synthesize an entity with type
    # 'abstract' and confirm rejection.
    t = _kitchen_trace(lex)
    # Hand-craft an abstract entity bypassing concept lookup, since our
    # demo concepts are all physical.
    from esperanto_lm.ontology import EntityInstance
    t.entities["ideo"] = EntityInstance(
        id="ideo", concept_lemma="ideo", entity_type="abstract")

    t.add_event(make_event("uzi", roles={
        "agent": "petro", "instrument": "tranĉilo", "theme": "ideo"}))
    _run(t, lex)
    actions = [e.action for e in t.events]
    assert "tranĉi" not in actions, \
        "tranĉi should not fire on a non-physical theme"


def test_event_id_is_content_addressed(lex):
    t = _kitchen_trace(lex)
    e1 = make_event("uzi", roles={
        "agent": "petro", "instrument": "tranĉilo", "theme": "pano"})
    e2 = make_event("uzi", roles={
        "agent": "petro", "instrument": "tranĉilo", "theme": "pano"})
    assert e1.id == e2.id
    t.add_event(e1)
    assert t.add_event(e2) is False, \
        "duplicate-by-content event should not be re-added"


def test_lock_via_slosilo_uses_same_generic_rule(lex):
    # Adding a second derived instrument requires zero new code: the same
    # use_instrument rule fires for ŝlosilo because it carries a
    # functional_signature too.
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("pordo", lex, entity_id="pordo")
    t.add_entity("ŝlosilo", lex, entity_id="ŝlosilo")
    t.assert_relation("havi", ("petro", "ŝlosilo"), lex)
    # A pordo starts locked per concept defaults; unlock first to make the
    # effect visible.
    t.entities["pordo"].set_property("lock_state", "unlocked")
    t.add_event(make_event("uzi", roles={
        "agent": "petro", "instrument": "ŝlosilo", "theme": "pordo"}))
    _run(t, lex)
    actions = [e.action for e in t.events]
    assert "ŝlosi" in actions
    assert t.property_at("pordo", "lock_state", len(t.events)) == "locked"
