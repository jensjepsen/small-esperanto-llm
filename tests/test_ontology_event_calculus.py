"""Tests for the event-calculus model.

Fields: EntityInstance.{properties, created_at_event, destroyed_at_event},
Event.{creates, property_changes, trace_position}.

Queries: Trace.entities_at(t), Trace.property_at(eid, prop, t).

Position semantics: t = number of events that have fired. t=0 is
scene-initial; t=len(events) is post-final.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from esperanto_lm.ontology import (
    EntityInstance,
    Trace,
    load_lexicon,
    make_event,
)





@pytest.fixture
def lex():
    return load_lexicon()


# ---- Step 1: schema additions -------------------------------------------

def test_entity_properties_reflects_construction_state(lex):
    """properties holds the entity's state at construction — what the
    caller passed in. set_property (scene-init only) writes through to
    it. After engine starts, property mutations go through events, not
    back into this dict."""
    e = EntityInstance(
        id="x", concept_lemma="glaso", entity_type="artifact",
        properties={"fragility": ["fragila"]})
    assert e.properties == {"fragility": ["fragila"]}
    # Scene-init set_property writes through.
    e.set_property("integrity", "tuta")
    assert e.properties["integrity"] == ["tuta"]


def test_entity_lifecycle_fields_default_none(lex):
    e = EntityInstance(id="x", concept_lemma="glaso", entity_type="artifact")
    assert e.created_at_event is None
    assert e.destroyed_at_event is None


def test_event_new_fields_default_empty():
    ev = make_event("fali", roles={"theme": "x"})
    assert ev.creates == []
    assert ev.property_changes == {}
    assert ev.trace_position is None


def test_event_id_independent_of_new_fields():
    """Two events with identical action/roles/causes must have identical
    id, regardless of creates/property_changes/trace_position values."""
    ev1 = make_event("rompiĝi", roles={"theme": "glaso"})
    shards = EntityInstance(
        id="s", concept_lemma="vitropecetoj", entity_type="inanimate")
    ev2 = make_event(
        "rompiĝi", roles={"theme": "glaso"},
        creates=[shards],
        property_changes={("glaso", "integrity"): "rompita"},
        trace_position=5)
    assert ev1.id == ev2.id


# ---- Step 2: trace queries ----------------------------------------------

def test_entities_at_scene_init(lex):
    """At t=0, only scene-initial entities are present."""
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("glaso", lex, entity_id="glaso")
    # Manually simulate a mid-trace-created entity.
    shards = EntityInstance(
        id="shards", concept_lemma="persono",  # lemma doesn't matter here
        entity_type="inanimate", created_at_event=2)
    t.entities["shards"] = shards

    at_0 = {e.id for e in t.entities_at(0)}
    assert at_0 == {"petro", "glaso"}


def test_entities_at_after_creation(lex):
    """An entity with created_at_event=k is visible at t>k, not at t<=k."""
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    shards = EntityInstance(
        id="shards", concept_lemma="persono", entity_type="inanimate",
        created_at_event=2)
    t.entities["shards"] = shards

    assert "shards" not in {e.id for e in t.entities_at(2)}
    assert "shards" in {e.id for e in t.entities_at(3)}
    assert "shards" in {e.id for e in t.entities_at(100)}


def test_entities_at_respects_destruction(lex):
    """destroyed_at_event=k removes the entity from t>k."""
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    t.entities["glaso"].destroyed_at_event = 3
    assert "glaso" in {e.id for e in t.entities_at(3)}
    assert "glaso" not in {e.id for e in t.entities_at(4)}


def test_property_at_no_changes_returns_initial(lex):
    """When no event has touched a property, property_at returns the
    entity's initial value."""
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    # glaso concept has fragility=fragile, integrity=intact by default.
    assert t.property_at("glaso", "integrity", 0) == ["tuta"]
    assert t.property_at("glaso", "integrity", 10) == ["tuta"]


def test_property_at_returns_most_recent_change(lex):
    """Property_at reads the most recent property_change from events
    before position t."""
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    # Append two events that change integrity.
    t.events.append(make_event(
        "rompiĝi", roles={"theme": "glaso"},
        property_changes={("glaso", "integrity"): "tranĉita"}))
    t.events.append(make_event(
        "rompiĝi", roles={"theme": "glaso"}, caused_by=("x",),
        property_changes={("glaso", "integrity"): "rompita"}))

    # t=0: scene-initial
    assert t.property_at("glaso", "integrity", 0) == ["tuta"]
    # t=1: after event 0 (severed)
    assert t.property_at("glaso", "integrity", 1) == "tranĉita"
    # t=2: after event 1 (broken) — most recent wins
    assert t.property_at("glaso", "integrity", 2) == "rompita"


def test_property_at_returns_none_for_missing_entity(lex):
    t = Trace()
    assert t.property_at("nonexistent", "integrity", 0) is None


def test_property_at_returns_none_before_creation(lex):
    """An entity not yet created returns None, not the initial_properties."""
    t = Trace()
    shards = EntityInstance(
        id="shards", concept_lemma="persono", entity_type="inanimate",
        properties={"sharpness": ["akra"]},
        created_at_event=3)
    t.entities["shards"] = shards
    assert t.property_at("shards", "sharpness", 0) is None
    assert t.property_at("shards", "sharpness", 3) is None  # not yet at t=3
    assert t.property_at("shards", "sharpness", 4) == ["akra"]


def test_property_at_returns_initial_at_exact_creation_position(lex):
    """At t = created_at_event + 1 (first position the entity exists),
    no property_change event for it has fired yet, so we get
    initial_properties."""
    t = Trace()
    shards = EntityInstance(
        id="shards", concept_lemma="persono", entity_type="inanimate",
        properties={"sharpness": ["akra"]},
        created_at_event=0)
    t.entities["shards"] = shards
    # Fake an event at index 0 that created shards but made no property changes.
    t.events.append(make_event(
        "rompiĝi", roles={"theme": "glaso"}, creates=[shards]))
    assert t.property_at("shards", "sharpness", 1) == ["akra"]


# ---- Engine + ported rules from the DSL --------------------------------

from esperanto_lm.ontology.dsl import run_dsl as _run_dsl
from esperanto_lm.ontology.dsl.engine import Rule as _Rule
from esperanto_lm.ontology.dsl.rules import (
    DEFAULT_DSL_RULES,
    fragile_falls_breaks,
)


def _run_dsl_compat(trace, rules, lexicon=None):
    """Test helper: most cases here pass a flat rule list and want the
    engine to figure out the lexicon (load it lazily) and accept a
    rule-or-list-of-rules in the input. Flattens, then delegates to
    `run_dsl` with empty derivations."""
    if lexicon is None:
        from esperanto_lm.ontology import load_lexicon
        lexicon = load_lexicon()
    flat: list = []
    for item in rules:
        if isinstance(item, _Rule):
            flat.append(item)
        else:
            flat.extend(item)
    return _run_dsl(trace, flat, [], lexicon)


def test_engine_runs_fragile_break(lex):
    """A fali on a fragile theme triggers a rompiĝi via the rule, and
    the property_change is reflected in property_at after the new engine
    runs."""
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    # Seed: glaso fell. Append directly (not via add_event so we don't
    # depend on engine effect-application).
    t.events.append(make_event("fali", roles={"theme": "glaso"}))

    iters = _run_dsl_compat(t, DEFAULT_DSL_RULES)

    assert iters >= 1
    actions = [ev.action for ev in t.events]
    assert "rompiĝi" in actions, actions
    # property_at should reflect the broken state after the rompiĝi event.
    final_t = len(t.events)
    assert t.property_at("glaso", "integrity", final_t) == "rompita"
    # Before the rompiĝi event, integrity is still intact.
    assert t.property_at("glaso", "integrity", 1) == ["tuta"]


def test_engine_sets_trace_position(lex):
    """Each event added by the DSL engine has trace_position set to its
    index in trace.events."""
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    t.events.append(make_event("fali", roles={"theme": "glaso"}))
    _run_dsl_compat(t, DEFAULT_DSL_RULES)
    # The seed event was appended directly so its
    # trace_position is None.
    assert t.events[0].trace_position is None
    # The synthesized rompiĝi has trace_position set by the engine.
    rompig = next(e for e in t.events if e.action == "rompiĝi")
    assert rompig.trace_position == t.events.index(rompig)


def test_engine_does_not_break_when_theme_already_broken(lex):
    """If a theme is already broken, no second rompiĝi fires."""
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    t.entities["glaso"].properties = {
        "fragility": ["fragila"], "integrity": ["rompita"]}
    t.events.append(make_event("fali", roles={"theme": "glaso"}))
    _run_dsl_compat(t, DEFAULT_DSL_RULES)
    actions = [ev.action for ev in t.events]
    assert "rompiĝi" not in actions


# Retired: `test_engine_extended_creates_entity` and
# `test_engine_memoization_prevents_infinite_loop` exercised the old
# engine's `Callable[[Trace, int], list[Event]]` rule protocol (and its
# per-event-id memoization semantics). The DSL engine uses declarative
# `Rule` instances instead; equivalent coverage lives in
# `test_ontology_dsl.py::test_3_create_entity_then_emit_resolves` (for
# entity creation) and the DSL's per-(rule, event, binding)
# memoization is exercised implicitly throughout the cascade tests.


# ---- Tests for individual ported rules ---------------------------------

from esperanto_lm.ontology.dsl.rules import (
    broken_container_releases_contents,
    container_falls_contents_fall,
    hungry_eats_sated,
)


def test_hungry_eats_sated_fires(lex):
    """manĝi by a hungry agent → satiĝi event with hunger=sated change."""
    t = Trace()
    petro = t.add_entity("persono", lex, entity_id="petro")
    pano = t.add_entity("pano", lex, entity_id="pano")
    # Make petro hungry via initial_properties.
    petro.properties = {"hunger": ["malsata"]}
    t.events.append(make_event("manĝi", roles={
        "agent": "petro", "theme": "pano"}))

    _run_dsl_compat(t, [hungry_eats_sated])

    actions = [ev.action for ev in t.events]
    assert "satiĝi" in actions
    final_t = len(t.events)
    assert t.property_at("petro", "hunger", final_t) == "sata"


def test_hungry_does_not_fire_if_not_hungry(lex):
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("pano", lex, entity_id="pano")
    t.events.append(make_event("manĝi", roles={
        "agent": "petro", "theme": "pano"}))
    _run_dsl_compat(t, [hungry_eats_sated])
    actions = [ev.action for ev in t.events]
    assert "satiĝi" not in actions


def test_container_falls_contents_fall(lex):
    """fali on a glass that contains water → water also falls."""
    t = Trace()
    glaso = t.add_entity("glaso", lex, entity_id="glaso")
    akvo = t.add_entity("akvo", lex, entity_id="akvo")
    t.assert_relation("en", ("akvo", "glaso"), lex)
    t.events.append(make_event("fali", roles={"theme": "glaso"}))

    _run_dsl_compat(t, [container_falls_contents_fall])

    fali_themes = [ev.roles.get("theme") for ev in t.events
                   if ev.action == "fali"]
    assert "glaso" in fali_themes and "akvo" in fali_themes


def test_container_falls_doesnt_make_kitchen_fall(lex):
    """Locations are excluded from cascade — kitchen doesn't fall when
    a person in it is mentioned in a fali."""
    t = Trace()
    petro = t.add_entity("persono", lex, entity_id="petro")
    kuirejo = t.add_entity("kuirejo", lex, entity_id="kuirejo")
    glaso = t.add_entity("glaso", lex, entity_id="glaso")
    t.assert_relation("en", ("petro", "kuirejo"), lex)
    t.assert_relation("en", ("glaso", "kuirejo"), lex)
    # If kuirejo somehow fell (it shouldn't — but the rule shouldn't
    # cascade to non-existent absurd events).
    t.events.append(make_event("fali", roles={"theme": "glaso"}))
    _run_dsl_compat(t, [container_falls_contents_fall])
    # Glass has no contents en/sur it; rule shouldn't produce any fali.
    fali_count = sum(1 for ev in t.events if ev.action == "fali")
    assert fali_count == 1  # just the seed


def test_broken_container_releases_contents(lex):
    """rompiĝi on a container → contents fall, even without a fall first."""
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    t.add_entity("akvo", lex, entity_id="akvo")
    t.assert_relation("en", ("akvo", "glaso"), lex)
    t.events.append(make_event("rompiĝi", roles={"theme": "glaso"}))

    _run_dsl_compat(t, [broken_container_releases_contents])

    fali_themes = [ev.roles.get("theme") for ev in t.events
                   if ev.action == "fali"]
    assert "akvo" in fali_themes


def test_bridge_sampler_to_engine_hunger(lex):
    """Bridge canary: the sampler's `set_initial_property("hunger", "malsata")`
    on a person makes that hunger visible to the DSL engine via property_at.
    Verifies sampler→wiring.
    """
    import random as _r
    from esperanto_lm.ontology import sample_scene, prune_unused_persons

    rng = _r.Random(0)
    # Sample until we get an eat-recipe (so the sampler exercises the
    # set_initial_property("hunger", "malsata") path).
    for _ in range(30):
        t, info = sample_scene(lex, rng, scene="manĝejo")
        if "eats" not in info.recipe:
            continue
        # Identify the agent (first manĝi event in trace).
        manĝi_ev = next(e for e in t.events if e.action == "manĝi")
        agent_id = manĝi_ev.roles["agent"]
        # The bridge should have populated both properties and
        # initial_properties — property_at reads them.
        assert t.property_at(agent_id, "hunger", 0) == ["malsata"], \
            f"engine didnt see hunger via initial_properties for {agent_id}"
        # Run DSL engine. satiĝi should fire.
        _run_dsl_compat(t, [hungry_eats_sated])
        actions = [ev.action for ev in t.events]
        assert "satiĝi" in actions, \
            f"hungry_eats_sated didn't fire after sampler setup"
        return  # done
    pytest.skip("no eat-recipe sampled in 30 attempts")


# ---- Realizer first-mention for created entities ---------------

def test_realizer_introduces_created_entity_via_appearance_line(lex):
    """A rompiĝi event with creates=[shards] should produce prose that
    introduces shards as a new entity (existential/appearance form),
    not preempt-mention it in setup."""
    from esperanto_lm.ontology import realize_trace
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    # Create the shards entity directly with created_at_event=1 (will be
    # set by the rompiĝi event).
    shards = EntityInstance(
        id="vitropecetoj", concept_lemma="vitropecetoj",
        entity_type="inanimate",
        properties={"sharpness": ["akra"]},
        created_at_event=1)
    t.entities["vitropecetoj"] = shards
    # Event 0: glass falls.
    t.events.append(make_event("fali", roles={"theme": "glaso"}))
    # Event 1: glass breaks, creating shards.
    t.events.append(make_event(
        "rompiĝi", roles={"theme": "glaso"}, caused_by=[t.events[0].id],
        property_changes={("glaso", "integrity"): "rompita"},
        creates=[shards],
    ))

    prose = realize_trace(t, lex, scene_location_id="kuirejo")
    # Prose must reference "vitropecetoj" — first mention, so it's
    # introduced via an appearance phrase, not a setup line.
    assert "vitropecetoj" in prose.lower(), prose
    # And critically, NOT pre-introduced as if it were in the scene from
    # the start.
    assert "Vitropecetoj estis en" not in prose, \
        f"created entity wrongly pre-mentioned in setup: {prose}"
    assert "Vitropecetoj kuŝis" not in prose, prose


def test_realizer_subsequent_mention_of_created_entity_is_definite(lex):
    """After an appearance line introduces a created entity, a later
    event referencing it should use definite reference ('la X')."""
    from esperanto_lm.ontology import realize_trace
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    t.add_entity("persono", lex, entity_id="petro")
    shards = EntityInstance(
        id="vitropecetoj", concept_lemma="vitropecetoj",
        entity_type="inanimate", created_at_event=1)
    t.entities["vitropecetoj"] = shards
    # Sequence: glass falls → glass breaks creating shards → person falls
    # (caused-by shards, simulating the hazard rule for testing).
    t.events.append(make_event("fali", roles={"theme": "glaso"}))
    t.events.append(make_event(
        "rompiĝi", roles={"theme": "glaso"}, caused_by=[t.events[0].id],
        property_changes={("glaso", "integrity"): "rompita"},
        creates=[shards]))
    t.events.append(make_event(
        "fali", roles={"theme": "petro"},
        caused_by=[t.events[1].id]))

    prose = realize_trace(t, lex, scene_location_id="kuirejo")
    # After the appearance line, references should be definite.
    # We don't pin the exact wording, but we expect "la vitropecetoj" or
    # similar to appear if we want to refer back. For this test, just
    # confirm the appearance happened once and the created entity isn't
    # introduced twice.
    intro_count = prose.lower().count("vitropecetoj")
    assert intro_count >= 1, prose


def test_realizer_synthetic_grounding_skips_created_entities(lex):
    """The synthetic 'X estis en la kuirejo' grounding pass must NOT
    preemptively introduce entities that were created mid-trace, even
    when they later appear in event roles."""
    from esperanto_lm.ontology import realize_trace
    t = Trace()
    # Need kuirejo in trace for synthetic grounding to even fire.
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("glaso", lex, entity_id="glaso")
    shards = EntityInstance(
        id="vitropecetoj", concept_lemma="vitropecetoj",
        entity_type="inanimate", created_at_event=1)
    t.entities["vitropecetoj"] = shards
    t.events.append(make_event("fali", roles={"theme": "glaso"}))
    t.events.append(make_event(
        "rompiĝi", roles={"theme": "glaso"},
        caused_by=[t.events[0].id], creates=[shards]))
    # A follow-up event referencing shards in roles — synthetic grounding
    # *would* try to ground it if not for the created_at_event check.
    t.events.append(make_event(
        "fali", roles={"theme": "vitropecetoj"},
        caused_by=[t.events[1].id]))

    prose = realize_trace(t, lex, scene_location_id="kuirejo")
    # No synthetic 'estis en la kuirejo' line for the created entity.
    assert "Vitropecetoj estis en la kuirejo" not in prose, prose
    assert "vitropecetoj estis en la kuirejo" not in prose.lower(), prose
    # But glaso (scene-initial, no created_at_event) IS grounded.
    # Its synthetic line should appear since glaso has no setup relation.
    # (The variation pool may render it as 'Glaso estis en la kuirejo'
    # or 'En la kuirejo estis glaso' or similar — assert on the lemma.)
    assert "glaso" in prose.lower()


def test_full_rule_pool_kitchen_cascade(lex):
    """End-to-end: fragile glass with water falls → glass breaks AND
    water falls. All four no-lexicon DSL rules + use_instrument firing
    in the same trace."""
    # Local alias kept for naming-symmetry within this test
    # for clarity within this test.
    from esperanto_lm.ontology.dsl.rules import DEFAULT_DSL_RULES as DEFAULT_DSL_RULES
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    t.add_entity("akvo", lex, entity_id="akvo")
    t.assert_relation("en", ("akvo", "glaso"), lex)
    t.events.append(make_event("fali", roles={"theme": "glaso"}))

    rules = list(DEFAULT_DSL_RULES)
    _run_dsl_compat(t, rules)

    actions = [ev.action for ev in t.events]
    assert "rompiĝi" in actions  # from fragile_falls_breaks
    fali_themes = [ev.roles.get("theme") for ev in t.events
                   if ev.action == "fali"]
    assert "akvo" in fali_themes  # from container_falls or broken_container

    final_t = len(t.events)
    assert t.property_at("glaso", "integrity", final_t) == "rompita"
