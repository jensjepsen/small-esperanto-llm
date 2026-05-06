"""Pattern-matching invariants for the production lexicon.

These tests pin the observable behavior of `compute_derived_state` on
real-lexicon traces. They're meant to catch regressions when
optimizing pattern matching (entity_type / relation indexing). Each
test builds a small trace and asserts a specific derived fact —
adding indexing must not change which facts derive.
"""
from __future__ import annotations

import pytest

from esperanto_lm.ontology.causal import EntityInstance, Trace
from esperanto_lm.ontology.dsl.engine import collect_rules, compute_derived_state
from esperanto_lm.ontology.loader import load_lexicon
import esperanto_lm.ontology.dsl.rules as R


@pytest.fixture(scope="module")
def lex():
    return load_lexicon()


@pytest.fixture(scope="module")
def derivations(lex):
    # Use the explicit DEFAULT_DSL_DERIVATIONS + RUNTIME_DERIVATIONS
    # lists rather than `collect_rules` — the latter returns source-
    # definition order, which is fragile when adding new derivations
    # (e.g. samloke chain rules added at the bottom of the source must
    # run BEFORE the lighting rules they feed). RUNTIME-only rules
    # (lit_state via tempo_de_tago, posture, etc.) need to fire too
    # for the assertions about derived state to be meaningful.
    return list(R.DEFAULT_DSL_DERIVATIONS) + list(R.RUNTIME_DERIVATIONS)


def _make_trace(lex, entities: list[tuple[str, str]],
                en_relations: list[tuple[str, str]] = (),
                konas: list[tuple[str, str]] = (),
                havas_parton: list[tuple[str, str]] = ()) -> Trace:
    """Build a trace with the given (concept, entity_id) entities and
    `en` relations. Entities take their initial properties from the
    concept (no randomization, no parts materialization)."""
    t = Trace()
    for concept_lemma, eid in entities:
        c = lex.concepts[concept_lemma]
        t.entities[eid] = EntityInstance(
            id=eid, concept_lemma=concept_lemma,
            entity_type=c.entity_type,
            properties={k: list(v) for k, v in c.properties.items()})
    for a, b in en_relations:
        t.assert_relation("en", (a, b), lex)
    for a, b in konas:
        t.assert_relation("konas", (a, b), lex)
    for a, b in havas_parton:
        t.assert_relation("havas_parton", (a, b), lex)
    return t


def test_outdoor_location_is_luma(lex, derivations):
    """Outdoor location's lit_state=luma is now RUNTIME-derived from
    the trace-wide `mondo` singleton's `tempo_de_tago`. With mondo
    set to a non-night value, the outdoor scene is luma and the
    actor is illuminated."""
    t = _make_trace(lex,
        entities=[("mondo", "mondo"),
                  ("parko", "parko"), ("persono", "Mikael")],
        en_relations=[("Mikael", "parko")])
    t.entities["mondo"].set_property("tempo_de_tago", "tago")
    d = compute_derived_state(t, derivations, lex)
    assert d.properties.get(("parko", "lit_state")) == "luma"
    assert d.properties.get(("Mikael", "illuminated")) == "yes"


def test_outdoor_luma_during_morning_evening(lex, derivations):
    """All non-night `tempo_de_tago` values keep outdoor luma.
    Verifies the day-pattern's NotPattern (`~ tempo_de_tago=nokto`)
    isn't accidentally restricted to one daytime value."""
    for tdt in ("mateno", "tago", "vespero"):
        t = _make_trace(lex,
            entities=[("mondo", "mondo"), ("parko", "parko")])
        t.entities["mondo"].set_property("tempo_de_tago", tdt)
        d = compute_derived_state(t, derivations, lex)
        assert d.properties.get(("parko", "lit_state")) == "luma", (
            f"expected luma at tempo_de_tago={tdt!r}")


def test_outdoor_dark_at_night(lex, derivations):
    """When `mondo.tempo_de_tago=nokto`, outdoor lit_state flips to
    malluma and an actor outside is no longer illuminated. This is
    the night-darkness branch — mutually exclusive with the day
    branch on the night condition."""
    t = _make_trace(lex,
        entities=[("mondo", "mondo"),
                  ("parko", "parko"), ("persono", "Mikael")],
        en_relations=[("Mikael", "parko")])
    t.entities["mondo"].set_property("tempo_de_tago", "nokto")
    d = compute_derived_state(t, derivations, lex)
    assert d.properties.get(("parko", "lit_state")) == "malluma"
    assert d.properties.get(("Mikael", "illuminated")) is None


def test_sleep_preference_flips_at_night(lex):
    """At nokto, animates prefer to be dormanta — `effective_preferences`
    overrides the static `sleep_state=vekita` baseline. During the day
    the static preference holds. Verified by walking
    `displeased_slots` against the same vekita actor in two worlds."""
    from esperanto_lm.ontology.agent.planner import displeased_slots
    from esperanto_lm.ontology.agent.preferences import effective_preferences

    # Day world: vekita actor is satisfied on sleep_state.
    day = _make_trace(lex,
        entities=[("mondo", "mondo"),
                  ("biblioteko", "biblioteko"),
                  ("persono", "Mikael")])
    day.entities["mondo"].set_property("tempo_de_tago", "tago")
    day.entities["Mikael"].set_property("sleep_state", "vekita")
    assert effective_preferences(day)["sleep_state"] == "vekita"
    day_displeased = dict(displeased_slots(day.entities["Mikael"], day))
    assert "sleep_state" not in day_displeased

    # Night world: vekita actor is displeased — preference is dormanta.
    night = _make_trace(lex,
        entities=[("mondo", "mondo"),
                  ("biblioteko", "biblioteko"),
                  ("persono", "Mikael")])
    night.entities["mondo"].set_property("tempo_de_tago", "nokto")
    night.entities["Mikael"].set_property("sleep_state", "vekita")
    assert effective_preferences(night)["sleep_state"] == "dormanta"
    night_displeased = dict(displeased_slots(night.entities["Mikael"], night))
    assert night_displeased.get("sleep_state") == "dormanta"


def test_weather_preamble_renders(lex):
    """`mondo.weather=pluva` produces "Pluvis." preamble (verb form
    derived from adjective root). `serena` (unmarked) stays silent.
    Tense matches the trace's narrative tense."""
    import random
    from esperanto_lm.ontology import realize_trace
    ĉambro_c = lex.concepts["ĉambro"]
    for weather, expected in [("serena", ""), ("pluva", "Pluvis."),
                                ("neĝa", "Neĝis.")]:
        t = Trace()
        t.entities["mondo"] = EntityInstance(
            id="mondo", concept_lemma="mondo", entity_type="abstract",
            properties={"tempo_de_tago": ["tago"], "weather": [weather]})
        t.entities["ĉambro"] = EntityInstance(
            id="ĉambro", concept_lemma="ĉambro", entity_type="location",
            properties={k: list(v) for k, v in ĉambro_c.properties.items()})
        prose = realize_trace(t, lex, scene_location_id="ĉambro",
                               rng=random.Random(0), tense="is")
        if expected:
            assert prose.startswith(expected), (weather, prose)
        else:
            assert "Pluv" not in prose and "Neĝ" not in prose, (weather, prose)


def test_pluva_emits_pluvi_on_outdoor_locations(lex):
    """`_emit_weather_events` walks `mondo.weather`, derives the verb
    via adjective→infinitive (`pluva` → `pluvi`), and fires it for
    locations matching the action's role spec. Indoor places are
    filtered out by `pluvi.location.indoor_outdoor=ekstera`. With
    `serena` (unmarked) no event fires."""
    from esperanto_lm.ontology.sampler import _emit_weather_events
    parko_c = lex.concepts["parko"]
    ĉambro_c = lex.concepts["ĉambro"]

    # serena: no events.
    t = Trace()
    t.entities["mondo"] = EntityInstance(
        id="mondo", concept_lemma="mondo", entity_type="abstract",
        properties={"weather": ["serena"]})
    t.entities["parko"] = EntityInstance(
        id="parko", concept_lemma="parko", entity_type="location",
        properties={k: list(v) for k, v in parko_c.properties.items()})
    _emit_weather_events(t, lex)
    assert not t.events

    # pluva + outdoor: pluvi event fires.
    t = Trace()
    t.entities["mondo"] = EntityInstance(
        id="mondo", concept_lemma="mondo", entity_type="abstract",
        properties={"weather": ["pluva"]})
    t.entities["parko"] = EntityInstance(
        id="parko", concept_lemma="parko", entity_type="location",
        properties={k: list(v) for k, v in parko_c.properties.items()})
    t.entities["ĉambro"] = EntityInstance(
        id="ĉambro", concept_lemma="ĉambro", entity_type="location",
        properties={k: list(v) for k, v in ĉambro_c.properties.items()})
    _emit_weather_events(t, lex)
    fired = [(e.action, e.roles) for e in t.events]
    assert ("pluvi", {"location": "parko"}) in fired
    assert all(r.get("location") != "ĉambro" for _, r in fired)


def test_cooking_gate_forces_kuiri_for_raw_food(lex):
    """`manĝi` on a `requires_cooking=yes` theme requires
    `cooking_state=kuirita`. Bare `manĝi(viando=kruda)` would violate
    the IfPropertyPrecondition; the planner must subgoal `kuiri` first.

    Verifies the schema-level gate, not the planner — pin the concept
    flag, the slot weights, and the precondition presence so the
    cooking chain can't silently regress."""
    viando = lex.concepts.get("viando")
    assert viando is not None
    assert "yes" in viando.properties.get("requires_cooking", [])
    assert "kruda" in viando.properties.get("cooking_state", [])

    manĝi = lex.actions.get("manĝi")
    assert manĝi is not None
    from esperanto_lm.ontology.schemas import IfPropertyPrecondition
    gates = [pc for pc in manĝi.preconditions
             if isinstance(pc, IfPropertyPrecondition)
             and pc.role == "theme"
             and pc.if_property == "requires_cooking"]
    assert gates, "manĝi must gate raw cookable food on cooking_state"
    assert gates[0].then_property == "cooking_state"
    assert gates[0].then_value == "kuirita"

    kuiri = lex.actions.get("kuiri")
    assert any(eff.property == "cooking_state" and eff.value == "kuirita"
               for eff in kuiri.effects)


def test_self_slot_seeder_includes_dormi(lex):
    """`_self_slot_drive_pairs` should include `(sleep_state, dormanta)`
    via source 3 — direct-effect verbs writing the agent's slot. Without
    that path, sleep chains never surface in regression coverage and the
    night-time preference flip has nowhere to attach.

    Also exercises `regress_for_self_slot` to confirm the agent-direct
    seeder builds a valid scene and the world-state nudge forces
    tempo_de_tago to a value where `effective_preferences` matches."""
    import random
    from esperanto_lm.ontology.regression.seeders import (
        _self_slot_drive_pairs, regress_for_self_slot,
    )
    from esperanto_lm.ontology.agent.preferences import effective_preferences
    rules = list(R.DEFAULT_DSL_RULES)

    pairs = set(_self_slot_drive_pairs(rules, lex))
    assert ("sleep_state", "dormanta") in pairs, pairs

    rng = random.Random(0)
    result = regress_for_self_slot("sleep_state", "dormanta", lex, rng, rules)
    assert result is not None
    trace, _scene_id, drive = result
    assert drive[0] == "self_slot"
    assert drive[2] == "sleep_state" and drive[3] == "dormanta"
    actor = trace.entities[drive[1]]
    # Pre-state must be the opposite — the drive needs to do something.
    assert "dormanta" not in actor.properties.get("sleep_state", [])
    # Mondo nudged so the preference matches the drive.
    assert effective_preferences(trace).get("sleep_state") == "dormanta"


def test_outdoor_lit_by_lamp_at_night(lex, derivations):
    """An active lamp in an outdoor scene at night lights it — a
    torch in a dark park IS a light source. `location_lit_by_active_lamp`
    fires before `outdoor_dark_at_night` in RUNTIME_DERIVATIONS so
    luma wins on the first-write-wins scalar slot. Without the
    lamp, the same scene goes malluma (verified above)."""
    t = _make_trace(lex,
        entities=[("mondo", "mondo"),
                  ("parko", "parko"), ("lampo", "lampo"),
                  ("persono", "Mikael")],
        en_relations=[("Mikael", "parko"), ("lampo", "parko")])
    t.entities["mondo"].set_property("tempo_de_tago", "nokto")
    t.entities["lampo"].set_property("power_state", "aktiva")
    d = compute_derived_state(t, derivations, lex)
    assert d.properties.get(("parko", "lit_state")) == "luma"
    assert d.properties.get(("Mikael", "illuminated")) == "yes"


def test_indoor_with_aktiva_lamp_derives_luma(lex, derivations):
    """Indoor + lamp en'd + lamp.power_state=aktiva → location.lit_state=luma
    (via indoor_lit_by_active_lamp), agent illuminated."""
    t = _make_trace(lex,
        entities=[("biblioteko", "biblioteko"),
                  ("persono", "Mikael"),
                  ("lampo", "lampo")],
        en_relations=[("Mikael", "biblioteko"),
                      ("lampo", "biblioteko")])
    t.entities["lampo"].set_property("power_state", "aktiva")
    d = compute_derived_state(t, derivations, lex)
    assert d.properties.get(("biblioteko", "lit_state")) == "luma"
    assert d.properties.get(("Mikael", "illuminated")) == "yes"


def test_indoor_with_neaktiva_lamp_derives_malluma(lex, derivations):
    """Indoor + lamp en'd + lamp.power_state=neaktiva →
    location.lit_state=malluma (via indoor_dark_without_active_lamp),
    agent NOT illuminated."""
    t = _make_trace(lex,
        entities=[("biblioteko", "biblioteko"),
                  ("persono", "Mikael"),
                  ("lampo", "lampo")],
        en_relations=[("Mikael", "biblioteko"),
                      ("lampo", "biblioteko")])
    t.entities["lampo"].set_property("power_state", "neaktiva")
    d = compute_derived_state(t, derivations, lex)
    assert d.properties.get(("biblioteko", "lit_state")) == "malluma"
    assert d.properties.get(("Mikael", "illuminated")) is None


def test_indoor_no_lamp_is_malluma(lex, derivations):
    """Indoor location with no aktiva lamp →
    indoor_dark_without_active_lamp fires → lit_state=malluma."""
    t = _make_trace(lex,
        entities=[("biblioteko", "biblioteko"),
                  ("persono", "Mikael")],
        en_relations=[("Mikael", "biblioteko")])
    d = compute_derived_state(t, derivations, lex)
    assert d.properties.get(("biblioteko", "lit_state")) == "malluma"


def test_samloke_from_shared_en(lex, derivations):
    """Two entities en the same container → samloke derives both
    directions."""
    t = _make_trace(lex,
        entities=[("kuirejo", "kuirejo"),
                  ("persono", "Mikael"),
                  ("pordo", "pordo")],
        en_relations=[("Mikael", "kuirejo"), ("pordo", "kuirejo")])
    d = compute_derived_state(t, derivations, lex)
    assert d.has_relation("samloke", ("Mikael", "pordo"))
    assert d.has_relation("samloke", ("pordo", "Mikael"))


def test_scias_lokon_via_konas_en(lex, derivations):
    """konas a (en) fakto → scias_lokon derives on the fakto's subjekto."""
    t = _make_trace(lex,
        entities=[("kuirejo", "kuirejo"),
                  ("persono", "Mikael"),
                  ("ŝlosilo", "ŝlosilo")])
    fakto = lex.concepts["fakto"]
    fid = "fakto_from_en_ŝlosilo_kuirejo"
    t.entities[fid] = EntityInstance(
        id=fid, concept_lemma="fakto",
        entity_type=fakto.entity_type,
        properties={"pri_relacio": ["en"]})
    t.assert_relation("subjekto", (fid, "ŝlosilo"), lex)
    t.assert_relation("objekto", (fid, "kuirejo"), lex)
    t.assert_relation("konas", ("Mikael", fid), lex)
    d = compute_derived_state(t, derivations, lex)
    assert d.has_relation("scias_lokon", ("Mikael", "ŝlosilo"))


def test_animate_knows_self_subject(lex, derivations):
    """An animate is the subjekto of some fakto → konas it (self-knowledge)."""
    t = _make_trace(lex,
        entities=[("kuirejo", "kuirejo"),
                  ("persono", "Mikael")])
    fakto = lex.concepts["fakto"]
    fid = "fakto_from_en_Mikael_kuirejo"
    t.entities[fid] = EntityInstance(
        id=fid, concept_lemma="fakto",
        entity_type=fakto.entity_type,
        properties={"pri_relacio": ["en"]})
    t.assert_relation("subjekto", (fid, "Mikael"), lex)
    t.assert_relation("objekto", (fid, "kuirejo"), lex)
    d = compute_derived_state(t, derivations, lex)
    assert d.has_relation("konas", ("Mikael", fid))


def test_baked_facts_present(lex):
    """Concept-level baked facts persist in concept.properties."""
    # Person-family lifts:
    persono = lex.concepts["persono"]
    assert "yes" in persono.properties.get("can_use_tools", [])
    # -isto person inherits parts via person_has_human_parts:
    kuiristo = lex.concepts["kuiristo"]
    parts = [p.concept for p in kuiristo.parts]
    assert "mano" in parts
    assert "yes" in kuiristo.properties.get("can_use_tools", [])
    # pordo lifts lock_capable from its seruro part. lock_state itself
    # is varies=true so it's deliberately NOT baked — the runtime
    # derivation re-fires per-instance based on the actual seruro's
    # randomized lock_state, rather than freezing to the seruro's
    # default at concept-load time.
    pordo = lex.concepts["pordo"]
    assert pordo.properties.get("lock_capable") == ["yes"]
    assert pordo.properties.get("lock_state") is None
    # Outdoor lit_state is no longer bake-time — it now depends on
    # the trace-wide `mondo` singleton's `tempo_de_tago` (varies=true).
    # Both indoor and outdoor lit_state are RUNTIME-only:
    parko = lex.concepts["parko"]
    assert parko.properties.get("lit_state") is None
    biblioteko = lex.concepts["biblioteko"]
    assert biblioteko.properties.get("lit_state") is None
