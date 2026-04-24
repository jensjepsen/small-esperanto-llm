"""Tests for the procedural sampler and the surface realizer."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from esperanto_lm.ontology import (
    Trace,
    load_lexicon,
    make_event,
    realize_trace,
    sample_scene,
)
from esperanto_lm.ontology.dsl import run_dsl
from esperanto_lm.ontology.dsl.rules import (
    DEFAULT_DSL_DERIVATIONS,
    DEFAULT_DSL_RULES,
    make_use_instrument_rules,
)
from esperanto_lm.ontology.realize import past_tense, to_accusative





@pytest.fixture
def lex():
    return load_lexicon()


# ---- realize.py morphology helpers ---------------------------------------

@pytest.mark.parametrize("inp,expected", [
    ("tranĉi", "tranĉis"),
    ("ŝlosi", "ŝlosis"),
    ("rompiĝi", "rompiĝis"),
    ("satiĝi", "satiĝis"),
    ("malŝlosi", "malŝlosis"),
])
def test_past_tense(inp, expected):
    assert past_tense(inp) == expected


@pytest.mark.parametrize("inp,expected", [
    ("pano", "panon"),
    ("la pano", "la panon"),
    ("tranĉilo", "tranĉilon"),
    ("la tranĉilo", "la tranĉilon"),
    ("panoj", "panojn"),
])
def test_accusative(inp, expected):
    assert to_accusative(inp) == expected


# ---- sampler validity ----------------------------------------------------

def test_sampler_produces_valid_kitchen_scenes(lex):
    """Every sampled scene must satisfy: (1) at least one person in kuirejo,
    (2) at least one event seeded, (3) lexicon validation passed (would have
    raised at relation/event insertion if not)."""
    rng = random.Random(0)
    for _ in range(30):
        trace, info = sample_scene(lex, rng)
        # Kitchen present.
        assert "kuirejo" in trace.entities
        # At least one person.
        assert any(e.entity_type == "person" for e in trace.entities.values())
        # Persons are en kuirejo.
        for pid in info.persons:
            assert any(
                r.relation == "en"
                and len(r.args) == 2
                and r.args[0] == pid
                and r.args[1] == "kuirejo"
                for r in trace.relations
            )
        # At least one seed event.
        assert len(trace.events) >= 1


def test_sampler_recipes_cover_all_paths_over_many_runs(lex):
    """Sample enough times that every recipe label appears at least once.
    Confirms the recipe pool isn't stuck on one branch."""
    rng = random.Random(1)
    seen_recipes = set()
    for _ in range(200):
        _, info = sample_scene(lex, rng)
        seen_recipes.add(info.recipe)
    # We have 7 distinct recipes; expect to hit all in 200 samples (well
    # above coupon-collector expectation of ~14).
    assert len(seen_recipes) >= 7, f"only saw {seen_recipes}"


def test_sampler_seed_is_deterministic(lex):
    """Same seed → same scene + same recipe."""
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    a_trace, a_info = sample_scene(lex, rng_a)
    b_trace, b_info = sample_scene(lex, rng_b)
    assert a_info.recipe == b_info.recipe
    assert a_info.persons == b_info.persons
    assert [e.id for e in a_trace.events] == [e.id for e in b_trace.events]


# ---- end-to-end realization ----------------------------------------------

def test_realize_canonical_kitchen_trace(lex):
    """End-to-end check on a known scene. uzi gets fused out per the
    realizer; only the synthesized verb appears."""
    t = Trace()
    petro = t.add_entity("persono", lex, entity_id="petro")
    kuirejo = t.add_entity("kuirejo", lex, entity_id="kuirejo")
    pano = t.add_entity("pano", lex, entity_id="pano")
    knife = t.add_entity("tranĉilo", lex, entity_id="tranĉilo")
    t.assert_relation("en", (petro.id, kuirejo.id), lex)
    t.assert_relation("havi", (petro.id, knife.id), lex)
    t.add_event(make_event("uzi", roles={
        "agent": petro.id, "instrument": knife.id, "theme": pano.id}))
    run_dsl(t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex),
            DEFAULT_DSL_DERIVATIONS, lex)

    prose = realize_trace(t, lex)
    assert "tranĉis" in prose
    assert " uzis " not in prose, "uzi should be fused out"
    assert "panon" in prose
    assert "en kuirejo" in prose, "kuirejo first-mention is bare"
    assert "Petro" in prose


def test_realize_returns_string_for_random_traces(lex):
    """Smoke: pump 30 samples through the realizer and confirm it returns
    non-empty strings ending with a period."""
    rng = random.Random(123)
    for _ in range(30):
        t, _ = sample_scene(lex, rng)
        run_dsl(t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex),
            DEFAULT_DSL_DERIVATIONS, lex)
        prose = realize_trace(t, lex)
        assert isinstance(prose, str) and prose
        assert prose.endswith(".")


# ---- use-instrument fusion -----------------------------------------------

def test_use_instrument_event_collapses_in_prose(lex):
    """The redundant 'X uzis Y per Z. Tial X verbis Yn per Z.' becomes just
    the second sentence."""
    t = Trace()
    sara = t.add_entity("persono", lex, entity_id="sara")
    pano = t.add_entity("pano", lex, entity_id="pano")
    knife = t.add_entity("tranĉilo", lex, entity_id="tranĉilo")
    t.assert_relation("havi", (sara.id, knife.id), lex)
    t.add_event(make_event("uzi", roles={
        "agent": sara.id, "instrument": knife.id, "theme": pano.id}))
    run_dsl(t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex),
            DEFAULT_DSL_DERIVATIONS, lex)

    prose = realize_trace(t, lex)
    assert "tranĉis" in prose, "the synthesized verb should still be rendered"
    assert " uzis " not in prose, \
        f"uzi should be collapsed; got: {prose!r}"
    # No connective should land before the surviving verb sentence — its
    # prose-cause was just collapsed.
    assert "Tial Sara tranĉis" not in prose, \
        f"no Tial when cause is fused out; got: {prose!r}"


# ---- article tracking ----------------------------------------------------

def test_first_mention_indefinite_subsequent_definite(lex):
    """Bare on first mention, 'la X' on subsequent — including locations,
    including across setup→event boundary."""
    t = Trace()
    sara = t.add_entity("persono", lex, entity_id="sara")
    kuirejo = t.add_entity("kuirejo", lex, entity_id="kuirejo")
    pano = t.add_entity("pano", lex, entity_id="pano")
    knife = t.add_entity("tranĉilo", lex, entity_id="tranĉilo")
    t.assert_relation("en", (sara.id, kuirejo.id), lex)
    t.assert_relation("havi", (sara.id, knife.id), lex)
    t.add_event(make_event("uzi", roles={
        "agent": sara.id, "instrument": knife.id, "theme": pano.id}))
    run_dsl(t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex),
            DEFAULT_DSL_DERIVATIONS, lex)

    prose = realize_trace(t, lex)
    # First-mention bare:
    assert "en kuirejo" in prose, \
        f"first mention of kuirejo should be bare; got: {prose!r}"
    assert "havis tranĉilon" in prose, \
        f"first mention of tranĉilo should be bare (acc); got: {prose!r}"
    # Subsequent-mention definite:
    assert "per la tranĉilo" in prose, \
        f"second mention of tranĉilo should be 'la tranĉilo'; got: {prose!r}"


def test_no_la_on_location_first_mention(lex):
    """Specifically guard against the previous force_definite-on-location bug."""
    t = Trace()
    sara = t.add_entity("persono", lex, entity_id="sara")
    kuirejo = t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.assert_relation("en", (sara.id, kuirejo.id), lex)

    prose = realize_trace(t, lex)
    assert prose == "Sara estis en kuirejo.", prose


def test_setup_introduces_then_event_uses_definite(lex):
    """Akvo + glaso both first-mentioned in setup; subsequent fall events
    use 'la X'. Glaso's fali + rompiĝi aggregate onto one verb phrase
    via the same-subject combiner — the article rule must still
    resolve glaso as definite in that coordinated sentence."""
    t = Trace()
    glaso = t.add_entity("glaso", lex, entity_id="glaso")
    akvo = t.add_entity("akvo", lex, entity_id="akvo")
    t.assert_relation("en", (akvo.id, glaso.id), lex)
    t.add_event(make_event("fali", roles={"theme": glaso.id}))
    run_dsl(t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex),
            DEFAULT_DSL_DERIVATIONS, lex)

    prose = realize_trace(t, lex)
    # Setup line: both indefinite (akvo bare, glaso bare).
    assert "Akvo estis en glaso." in prose, \
        f"setup line should have both bare; got: {prose!r}"
    # Subsequent glaso reference uses definite. Aggregation collapses
    # fali + rompiĝi into one clause: "La glaso falis kaj rompiĝis."
    assert "La glaso falis kaj rompiĝis." in prose, prose
    # Subsequent akvo reference definite too.
    assert "la akvo falis." in prose


# ---- participant-guaranteed sampling -------------------------------------

def test_prune_removes_lone_persons_and_their_relations(lex):
    """A person who never participates in an event is pruned, along with
    any relations involving them."""
    from esperanto_lm.ontology import prune_unused_persons

    t = Trace()
    sara = t.add_entity("persono", lex, entity_id="sara")
    lidia = t.add_entity("persono", lex, entity_id="lidia")
    kuirejo = t.add_entity("kuirejo", lex, entity_id="kuirejo")
    pano = t.add_entity("pano", lex, entity_id="pano")
    t.assert_relation("en", (sara.id, kuirejo.id), lex)
    t.assert_relation("en", (lidia.id, kuirejo.id), lex)
    t.assert_relation("havi", (sara.id, pano.id), lex)
    # Only Sara participates.
    t.add_event(make_event("manĝi", roles={"agent": sara.id, "theme": pano.id}))

    pruned = prune_unused_persons(t)
    assert pruned == ["lidia"]
    assert "lidia" not in t.entities
    # Relation involving lidia is gone, sara's relations remain.
    assert all("lidia" not in r.args for r in t.relations)
    assert any(r.relation == "en" and r.args[0] == "sara" for r in t.relations)


def test_prune_keeps_unused_non_persons(lex):
    """Tables, kitchens etc. may legitimately be set-dressing without
    participating in any event."""
    from esperanto_lm.ontology import prune_unused_persons

    t = Trace()
    sara = t.add_entity("persono", lex, entity_id="sara")
    tablo = t.add_entity("tablo", lex, entity_id="tablo")
    pano = t.add_entity("pano", lex, entity_id="pano")
    t.assert_relation("havi", (sara.id, pano.id), lex)
    t.assert_relation("sur", (pano.id, tablo.id), lex)
    t.add_event(make_event("manĝi", roles={"agent": sara.id, "theme": pano.id}))

    pruned = prune_unused_persons(t)
    assert pruned == []
    assert "tablo" in t.entities  # never in an event but kept
    assert any(r.relation == "sur" for r in t.relations)


def test_scene_location_definite_from_first_mention(lex):
    """When realize_trace is given scene_location_id, the named entity gets
    'la X' even on its first mention."""
    t = Trace()
    sara = t.add_entity("persono", lex, entity_id="sara")
    kuirejo = t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.assert_relation("en", (sara.id, kuirejo.id), lex)

    prose = realize_trace(t, lex, scene_location_id="kuirejo")
    assert prose == "Sara estis en la kuirejo.", prose


def test_synthetic_grounding_for_agentless_drop(lex):
    """An entity that participates in events but has no setup relation
    grounding it gets a synthetic 'X estis en la SCENE.' line. This is
    the contextless-drop fix."""
    t = Trace()
    # Note: no person, no relations. Just a glass and the kitchen.
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    glaso = t.add_entity("glaso", lex, entity_id="glaso")
    t.add_event(make_event("fali", roles={"theme": glaso.id}))
    run_dsl(t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex),
            DEFAULT_DSL_DERIVATIONS, lex)

    prose = realize_trace(t, lex, scene_location_id="kuirejo")
    # Synthetic setup: glaso grounded to la kuirejo.
    assert prose.startswith("Glaso estis en la kuirejo."), prose
    # Then the events.
    assert "La glaso falis" in prose
    assert "rompiĝis" in prose


def test_synthetic_grounding_skips_entities_already_in_relations(lex):
    """If an entity appears anywhere in a setup relation (subject or object),
    it's considered grounded and gets no synthetic 'estis en la kuirejo'
    line. The relation chain provides the scene context."""
    t = Trace()
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    glaso = t.add_entity("glaso", lex, entity_id="glaso")
    akvo = t.add_entity("akvo", lex, entity_id="akvo")
    t.assert_relation("en", (akvo.id, glaso.id), lex)
    t.add_event(make_event("fali", roles={"theme": glaso.id}))
    run_dsl(t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex),
            DEFAULT_DSL_DERIVATIONS, lex)

    prose = realize_trace(t, lex, scene_location_id="kuirejo")
    # Both akvo (subject) and glaso (object) are in the en relation, so
    # neither gets a synthetic kitchen-grounding line.
    assert "estis en la kuirejo" not in prose, prose
    assert "Akvo estis en glaso." in prose
    assert "La glaso falis" in prose


def test_no_synthetic_grounding_when_explicit_relation_present(lex):
    """When the agent isn't pruned, all entities are introduced via
    relations; no synthetic lines should appear."""
    t = Trace()
    sara = t.add_entity("persono", lex, entity_id="sara")
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    pano = t.add_entity("pano", lex, entity_id="pano")
    t.assert_relation("en", (sara.id, "kuirejo"), lex)
    t.assert_relation("havi", (sara.id, pano.id), lex)
    t.add_event(make_event("manĝi", roles={
        "agent": sara.id, "theme": pano.id}))
    run_dsl(t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex),
            DEFAULT_DSL_DERIVATIONS, lex)

    prose = realize_trace(t, lex, scene_location_id="kuirejo")
    # Sara's `havi pano` introduces pano; no synthetic 'Pano estis en la kuirejo'.
    assert "Pano estis en la kuirejo." not in prose, prose
    # Kitchen mentioned just once via the en relation.
    assert prose.count("kuirejo") == 1, prose


def test_sampler_works_across_all_four_scenes(lex):
    """Each of the four authored scenes must produce a valid trace with at
    least one event. No per-scene code paths in the sampler — variation
    comes purely from containment-graph reachability."""
    rng = random.Random(0)
    for scene in ("kuirejo", "laborejo", "ĝardeno", "manĝejo"):
        t, info = sample_scene(lex, rng, scene=scene)
        assert info.scene_location_id == scene
        assert scene in t.entities
        assert any(e.entity_type == "person" for e in t.entities.values())
        assert len(t.events) >= 1, f"scene {scene} produced no events"


def test_garden_only_produces_garden_appropriate_recipes(lex):
    """Garden has narrow reachability: only plant/water/door recipes fire.
    No knife use, no eating, no key-locking — those concepts are
    unreachable from ĝardeno via containment."""
    rng = random.Random(0)
    seen = set()
    for _ in range(40):
        _, info = sample_scene(lex, rng, scene="ĝardeno")
        seen.add(info.recipe)
    # No food, no cutting, no key in the garden's containment.
    # Gardens contain doors (every location does), so ŝlosilo is now
    # reachable in the garden via the `contains: pordo` second-order
    # pattern. That's correct (gates can be locked), so use_ŝlosilo
    # is allowed. The forbidden set is everything that genuinely has no
    # path from ĝardeno: edibles, knives, hammers, fragile-container drops.
    forbidden_prefixes = ("eat_", "use_tranĉilo_", "use_najlilo_",
                          "drop_glaso", "drop_botelo", "drop_ovo")
    bad = [r for r in seen
           if any(r.startswith(p) for p in forbidden_prefixes)]
    assert not bad, f"unreachable-concept recipe fired in garden: {bad}"


def test_unknown_scene_lemma_fails(lex):
    rng = random.Random(0)
    with pytest.raises(ValueError, match="unknown scene"):
        sample_scene(lex, rng, scene="no_such_scene")


def test_non_location_scene_fails(lex):
    rng = random.Random(0)
    with pytest.raises(ValueError, match="not a location"):
        sample_scene(lex, rng, scene="pano")  # pano is a substance


def test_recipe_required_concepts_must_be_reachable(lex):
    """Verify a known constraint: floro is unreachable from kuirejo, so no
    plant_floro recipes should fire when sampling the kitchen."""
    rng = random.Random(0)
    seen = set()
    for _ in range(40):
        _, info = sample_scene(rng=rng, lex=lex, scene="kuirejo")
        seen.add(info.recipe)
    assert all(not r.startswith(("plant_", "water_")) for r in seen), \
        f"garden recipe in kitchen: {seen}"


def test_realize_variation_changes_with_rng_seed(lex):
    """Different RNG seeds should produce visibly different prose for the
    same trace. Otherwise the variation isn't doing anything."""
    t = Trace()
    sara = t.add_entity("persono", lex, entity_id="sara")
    kuirejo = t.add_entity("kuirejo", lex, entity_id="kuirejo")
    pano = t.add_entity("pano", lex, entity_id="pano")
    t.assert_relation("en", (sara.id, kuirejo.id), lex)
    t.assert_relation("havi", (sara.id, pano.id), lex)
    sara.set_property("hunger", "hungry")
    t.add_event(make_event("manĝi", roles={
        "agent": sara.id, "theme": pano.id}))
    run_dsl(t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex),
            DEFAULT_DSL_DERIVATIONS, lex)

    outputs = set()
    for seed in range(10):
        # Re-resolve trace context for each render: render_trace is pure,
        # the trace itself isn't mutated by realization.
        prose = realize_trace(t, lex, scene_location_id="kuirejo",
                              rng=random.Random(seed))
        outputs.add(prose)
    assert len(outputs) >= 3, \
        f"variation didn't happen: only {len(outputs)} unique outputs:\n" + "\n".join(outputs)


def test_realize_pronoun_substitution_for_lone_person(lex):
    """A solo person should sometimes appear as li/ŝi in subsequent
    mentions across a sample of seeds."""
    t = Trace()
    sara = t.add_entity("persono", lex, entity_id="sara")
    kuirejo = t.add_entity("kuirejo", lex, entity_id="kuirejo")
    pano = t.add_entity("pano", lex, entity_id="pano")
    t.assert_relation("en", (sara.id, kuirejo.id), lex)
    t.assert_relation("havi", (sara.id, pano.id), lex)
    sara.set_property("hunger", "hungry")
    t.add_event(make_event("manĝi", roles={
        "agent": sara.id, "theme": pano.id}))
    run_dsl(t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex),
            DEFAULT_DSL_DERIVATIONS, lex)

    saw_pronoun = False
    for seed in range(30):
        prose = realize_trace(t, lex, scene_location_id="kuirejo",
                              rng=random.Random(seed))
        if " Ŝi " in prose or "ŝi " in prose.split(". ", 1)[-1]:
            saw_pronoun = True
            break
    assert saw_pronoun, "no pronoun substitution observed across 30 seeds"


def test_realize_no_pronoun_when_two_same_gender_persons(lex):
    """With two ŝi-using persons in the trace, the pronoun is ambiguous
    and must NOT be substituted. Names must always be used."""
    t = Trace()
    sara = t.add_entity("persono", lex, entity_id="sara")
    maria = t.add_entity("persono", lex, entity_id="maria")
    kuirejo = t.add_entity("kuirejo", lex, entity_id="kuirejo")
    pano = t.add_entity("pano", lex, entity_id="pano")
    t.assert_relation("en", (sara.id, kuirejo.id), lex)
    t.assert_relation("en", (maria.id, kuirejo.id), lex)
    t.assert_relation("havi", (sara.id, pano.id), lex)
    sara.set_property("hunger", "hungry")
    t.add_event(make_event("manĝi", roles={
        "agent": sara.id, "theme": pano.id}))
    run_dsl(t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex),
            DEFAULT_DSL_DERIVATIONS, lex)

    for seed in range(30):
        prose = realize_trace(t, lex, scene_location_id="kuirejo",
                              rng=random.Random(seed))
        # No ambiguous pronoun should appear.
        assert " Ŝi " not in prose and " ŝi " not in prose, \
            f"ambiguous ŝi appeared at seed {seed}: {prose}"


def test_realize_variation_uses_alternate_connectives(lex):
    """Across many seeds, multiple connective variants should appear."""
    t = Trace()
    glaso = t.add_entity("glaso", lex, entity_id="glaso")
    t.add_event(make_event("fali", roles={"theme": glaso.id}))
    run_dsl(t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex),
            DEFAULT_DSL_DERIVATIONS, lex)

    seen_connectives = set()
    for seed in range(40):
        prose = realize_trace(t, lex, scene_location_id="kuirejo",
                              rng=random.Random(seed))
        for c in ("Tial", "Sekve", "Pro tio"):
            if c in prose:
                seen_connectives.add(c)
    # Across 40 seeds we expect at least 2 of the 3 named connectives
    # (the 4th option is empty juxtaposition).
    assert len(seen_connectives) >= 2, \
        f"only saw connectives: {seen_connectives}"


def test_realize_deterministic_when_rng_is_none(lex):
    """rng=None must produce the exact same output on repeated calls."""
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("pano", lex, entity_id="pano")
    t.assert_relation("en", ("petro", "kuirejo"), lex)
    t.assert_relation("havi", ("petro", "pano"), lex)
    t.entities["petro"].set_property("hunger", "hungry")
    t.add_event(make_event("manĝi", roles={
        "agent": "petro", "theme": "pano"}))
    run_dsl(t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex),
            DEFAULT_DSL_DERIVATIONS, lex)

    a = realize_trace(t, lex, scene_location_id="kuirejo")
    b = realize_trace(t, lex, scene_location_id="kuirejo")
    assert a == b, "rng=None should be deterministic"


def test_sampler_plus_prune_yields_only_participating_persons(lex):
    """Across many samples, after engine + prune, every remaining person
    appears in at least one event's roles."""
    from esperanto_lm.ontology import prune_unused_persons

    rng = random.Random(7)
    for _ in range(40):
        t, _ = sample_scene(lex, rng)
        run_dsl(t, DEFAULT_DSL_RULES + make_use_instrument_rules(lex),
            DEFAULT_DSL_DERIVATIONS, lex)
        prune_unused_persons(t)
        used = set()
        for ev in t.events:
            used.update(v for v in ev.roles.values() if isinstance(v, str))
        for eid, ent in t.entities.items():
            if ent.entity_type == "person":
                assert eid in used, \
                    f"person {eid!r} survived prune but isn't in any event"
