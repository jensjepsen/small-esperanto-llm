"""Tests for the new property-pattern rules and the second derived
instrument (ŝlosilo)."""
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
from esperanto_lm.ontology.dsl.rules import (
    DEFAULT_DSL_RULES,
    make_use_instrument_rules,
)


def _run(t, lex):
    return run_dsl(
        t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex), [], lex)





@pytest.fixture
def lex():
    return load_lexicon()


# ---- fragile_falls_breaks ------------------------------------------------

def test_fragile_glass_falls_then_breaks(lex):
    t = Trace()
    glaso = t.add_entity("glaso", lex, entity_id="glaso")
    t.add_event(make_event("fali", roles={"theme": glaso.id}))
    _run(t, lex)
    actions = [e.action for e in t.events]
    assert "rompiĝi" in actions
    assert t.property_at("glaso", "integrity", len(t.events)) == "broken"


def test_sturdy_table_falls_does_not_break(lex):
    t = Trace()
    tablo = t.add_entity("tablo", lex, entity_id="tablo")
    t.add_event(make_event("fali", roles={"theme": tablo.id}))
    _run(t, lex)
    actions = [e.action for e in t.events]
    assert "rompiĝi" not in actions
    # Sturdy tablo has no integrity slot; it stays whatever it was.


# ---- container_falls_contents_fall + cascade -----------------------------

def test_glass_with_water_drops_water_falls_too(lex):
    t = Trace()
    glaso = t.add_entity("glaso", lex, entity_id="glaso")
    akvo = t.add_entity("akvo", lex, entity_id="akvo")
    t.assert_relation("en", (akvo.id, glaso.id), lex)
    t.add_event(make_event("fali", roles={"theme": glaso.id}))
    _run(t, lex)
    actions = [e.action for e in t.events]
    fali_themes = [e.roles.get("theme") for e in t.events if e.action == "fali"]
    assert "glaso" in fali_themes and "akvo" in fali_themes
    # Glass also breaks (fragile).
    assert "rompiĝi" in actions
    # Water doesn't break (no fragility) and isn't a container of anything.


def test_kitchen_does_not_fall_when_person_in_it(lex):
    """If a person is `en kuirejo` and something else falls, the kitchen
    must not be a target of a synthesized fali. Locations are not contents."""
    t = Trace()
    petro = t.add_entity("persono", lex, entity_id="petro")
    kuirejo = t.add_entity("kuirejo", lex, entity_id="kuirejo")
    glaso = t.add_entity("glaso", lex, entity_id="glaso")
    t.assert_relation("en", (petro.id, kuirejo.id), lex)
    t.assert_relation("en", (glaso.id, kuirejo.id), lex)
    t.add_event(make_event("fali", roles={"theme": glaso.id}))
    _run(t, lex)
    fali_themes = [e.roles.get("theme") for e in t.events if e.action == "fali"]
    # glaso fell; kuirejo did NOT (only contents of glaso would fall, and it
    # has no contents in this test).
    assert "glaso" in fali_themes
    assert "kuirejo" not in fali_themes


# ---- hungry_eats_sated ----------------------------------------------------

def test_hungry_person_eats_becomes_sated_and_consumes_food(lex):
    t = Trace()
    petro = t.add_entity("persono", lex, entity_id="petro")
    pano = t.add_entity("pano", lex, entity_id="pano")
    petro.set_property("hunger", "hungry")
    roles = {"agent": petro.id, "theme": pano.id}
    t.add_event(make_event(
        "manĝi", roles=roles,
        property_changes=effect_changes("manĝi", roles, lex)))
    _run(t, lex)
    actions = [e.action for e in t.events]
    assert "satiĝi" in actions, "satiation rule should fire"
    assert t.property_at("petro", "hunger", len(t.events)) == "sated"
    # manĝi's intrinsic effect consumed the food.
    assert t.property_at("pano", "presence", len(t.events)) == "consumed"


def test_already_sated_person_eats_no_satiation_event(lex):
    t = Trace()
    petro = t.add_entity("persono", lex, entity_id="petro")
    pano = t.add_entity("pano", lex, entity_id="pano")
    # Default hunger unset (not 'hungry'); rule should not fire.
    roles = {"agent": petro.id, "theme": pano.id}
    t.add_event(make_event(
        "manĝi", roles=roles,
        property_changes=effect_changes("manĝi", roles, lex)))
    _run(t, lex)
    actions = [e.action for e in t.events]
    assert "satiĝi" not in actions
    # Food is still consumed by manĝi's own effect — that's intrinsic, not
    # gated on hunger.
    assert t.property_at("pano", "presence", len(t.events)) == "consumed"


# ---- second derived instrument: ŝlosilo (acceptance) ----------------------

def test_slosilo_locks_pordo_via_generic_rule(lex):
    """Acceptance: the same generic instrument-use rule that fires
    tranĉilo→tranĉi also fires ŝlosilo→ŝlosi without modification."""
    t = Trace()
    maria = t.add_entity("persono", lex, entity_id="maria")
    pordo = t.add_entity("pordo", lex, entity_id="pordo")
    slosilo = t.add_entity("ŝlosilo", lex, entity_id="ŝlosilo")
    t.assert_relation("havi", (maria.id, slosilo.id), lex)
    pordo.set_property("lock_state", "unlocked")
    t.add_event(make_event("uzi", roles={
        "agent": maria.id, "instrument": slosilo.id, "theme": pordo.id}))
    _run(t, lex)
    assert any(e.action == "ŝlosi" for e in t.events)
    assert t.property_at("pordo", "lock_state", len(t.events)) == "locked"
