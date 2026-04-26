"""Tests for the causal trace + rule engine."""
from __future__ import annotations

from pathlib import Path

import pytest

from esperanto_lm.ontology import (
    Trace,
    effect_changes,
    load_lexicon,
    make_event,
)
from esperanto_lm.ontology.dsl import run_dsl
from esperanto_lm.ontology.dsl.rules import DEFAULT_DSL_RULES


def _run(t, lex):
    return run_dsl(t, list(DEFAULT_DSL_RULES), [], lex)


@pytest.fixture
def lex():
    return load_lexicon()


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


def _seed(action, roles, lex):
    """Build a seed event with the action's intrinsic effects baked in."""
    return make_event(
        action, roles=roles,
        property_changes=effect_changes(action, roles, lex))


def test_cut_event_mutates_pano_integrity(lex):
    """Direct tranĉi event mutates pano.integrity via its intrinsic effect."""
    t = _kitchen_trace(lex)
    assert t.entities["pano"].properties.get("integrity") == ["tuta"]
    t.add_event(_seed("tranĉi", {
        "agent": "petro", "theme": "pano", "instrument": "tranĉilo"}, lex))
    _run(t, lex)
    assert t.property_at("pano", "integrity", len(t.events)) == "tranĉita"


def test_lock_event_mutates_pordo_lock_state(lex):
    """Direct ŝlosi event mutates pordo.lock_state."""
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("pordo", lex, entity_id="pordo")
    t.add_entity("ŝlosilo", lex, entity_id="ŝlosilo")
    t.assert_relation("havi", ("petro", "ŝlosilo"), lex)
    # A pordo starts locked per concept defaults; unlock first to make the
    # effect visible.
    t.entities["pordo"].set_property("lock_state", "malŝlosita")
    t.add_event(_seed("ŝlosi", {
        "agent": "petro", "theme": "pordo", "instrument": "ŝlosilo"}, lex))
    _run(t, lex)
    assert t.property_at("pordo", "lock_state", len(t.events)) == "ŝlosita"


def test_event_id_is_content_addressed(lex):
    """Same action+roles+causes ⇒ same id; duplicate add is idempotent."""
    t = _kitchen_trace(lex)
    e1 = make_event("tranĉi", roles={
        "agent": "petro", "theme": "pano", "instrument": "tranĉilo"})
    e2 = make_event("tranĉi", roles={
        "agent": "petro", "theme": "pano", "instrument": "tranĉilo"})
    assert e1.id == e2.id
    t.add_event(e1)
    assert t.add_event(e2) is False, \
        "duplicate-by-content event should not be re-added"
