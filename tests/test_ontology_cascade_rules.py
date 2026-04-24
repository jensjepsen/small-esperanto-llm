"""Tests for the deep-cascade rules added in Step 7.

Rules exercised:
  broken_fragile_creates_shards  — rompiĝi(glaso) → aperi(vitropecetoj)
  wet_liquid_container_tips      — fali(akvo)     → aperi(flako)
  person_walks_on_hazard_falls   — aperi(hazard)  → fali(person)
  carried_fragile_falls_when_carrier_falls
                                 — fali(person)   → fali(carried fragile)
"""
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
    DEFAULT_DSL_DERIVATIONS,
    DEFAULT_DSL_RULES,
    broken_fragile_creates_shards,
    carried_fragile_falls_when_carrier_falls,
    fire_spreads_to_adjacent_flammables,
    make_use_instrument_rules,
    person_slips_on_wet,
    wet_liquid_container_tips,
)


DATA_DIR = Path("data/ontology")


@pytest.fixture
def lex():
    return load_lexicon(DATA_DIR)


def _all_rules(lex):
    return DEFAULT_DSL_RULES + make_use_instrument_rules(lex)


# ---- broken_fragile_creates_shards ---------------------------------------

def test_broken_glaso_spawns_vitropecetoj(lex):
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    t.events.append(make_event("rompiĝi", roles={"theme": "glaso"},
                               property_changes={("glaso", "integrity"): "broken"}))

    run_dsl(t, [broken_fragile_creates_shards], [], lex)

    aperi_events = [e for e in t.events if e.action == "aperi"]
    assert len(aperi_events) == 1
    shards_id = aperi_events[0].roles["theme"]
    assert shards_id in t.entities
    assert t.entities[shards_id].concept_lemma == "vitropecetoj"
    assert t.entities[shards_id].properties.get("hazard") == ["sharp"]


def test_broken_non_transforming_does_not_spawn(lex):
    """ovo is fragile but has no transforms_on_break — no shards appear."""
    t = Trace()
    t.add_entity("ovo", lex, entity_id="ovo")
    t.events.append(make_event("rompiĝi", roles={"theme": "ovo"}))
    run_dsl(t, [broken_fragile_creates_shards], [], lex)
    assert not any(e.action == "aperi" for e in t.events)


# ---- wet_liquid_container_tips -------------------------------------------

def test_fallen_akvo_spawns_flako(lex):
    t = Trace()
    t.add_entity("akvo", lex, entity_id="akvo")
    t.events.append(make_event("fali", roles={"theme": "akvo"}))
    run_dsl(t, [wet_liquid_container_tips], [], lex)
    aperi = [e for e in t.events if e.action == "aperi"]
    assert len(aperi) == 1
    puddle_id = aperi[0].roles["theme"]
    assert t.entities[puddle_id].concept_lemma == "flako"
    assert t.entities[puddle_id].properties.get("hazard") == ["slippery"]


def test_fallen_pano_does_not_spawn_puddle(lex):
    """Solids don't puddle. pano has no transforms_on_spill."""
    t = Trace()
    t.add_entity("pano", lex, entity_id="pano")
    t.events.append(make_event("fali", roles={"theme": "pano"}))
    run_dsl(t, [wet_liquid_container_tips], [], lex)
    assert not any(e.action == "aperi" for e in t.events)


# ---- person_slips_on_wet -------------------------------------------------

def test_person_in_same_location_slips_on_puddle(lex):
    """Full chain: glass (with water) falls → breaks → water falls →
    puddle appears → person (in kitchen) slips. Sharp shards appear
    too but do NOT cause a slip — only slippery (wet) things do."""
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("glaso", lex, entity_id="glaso")
    t.add_entity("akvo", lex, entity_id="akvo")
    t.assert_relation("en", ("petro", "kuirejo"), lex)
    t.assert_relation("en", ("glaso", "kuirejo"), lex)
    t.assert_relation("en", ("akvo", "glaso"), lex)
    t.events.append(make_event("fali", roles={"theme": "glaso"}))

    run_dsl(t, _all_rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    petro_falls = [e for e in t.events
                   if e.action == "fali" and e.roles.get("theme") == "petro"]
    assert len(petro_falls) == 1
    # The fall is caused by the puddle-appearance aperi, not the shards.
    by_id = {e.id: e for e in t.events}
    cause = by_id[petro_falls[0].caused_by[0]]
    assert cause.action == "aperi"
    assert cause.roles["theme"].startswith("flako_"), (
        f"expected puddle-caused slip, got cause={cause.roles}")


def test_shards_alone_do_not_cause_slip(lex):
    """A dry glass breaking creates shards but no puddle; nobody slips."""
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("glaso", lex, entity_id="glaso")  # no water inside
    t.assert_relation("en", ("petro", "kuirejo"), lex)
    t.assert_relation("en", ("glaso", "kuirejo"), lex)
    t.events.append(make_event("fali", roles={"theme": "glaso"}))

    run_dsl(t, _all_rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    petro_falls = [e for e in t.events
                   if e.action == "fali" and e.roles.get("theme") == "petro"]
    assert not petro_falls, (
        "petro should not fall from dry shards — only wet surfaces slip")
    # But shards did appear.
    aperi_themes = {e.roles.get("theme") for e in t.events if e.action == "aperi"}
    assert any(s.startswith("vitropecetoj_") for s in aperi_themes)


def test_person_in_different_location_does_not_slip(lex):
    """Person in another room doesn't slip on a remote puddle."""
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("salono", lex, entity_id="salono")
    t.add_entity("glaso", lex, entity_id="glaso")
    t.add_entity("akvo", lex, entity_id="akvo")
    t.assert_relation("en", ("petro", "salono"), lex)
    t.assert_relation("en", ("glaso", "kuirejo"), lex)
    t.assert_relation("en", ("akvo", "glaso"), lex)
    t.events.append(make_event("fali", roles={"theme": "glaso"}))

    run_dsl(t, _all_rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    petro_falls = [e for e in t.events
                   if e.action == "fali" and e.roles.get("theme") == "petro"]
    assert not petro_falls


# ---- carried_fragile_falls_when_carrier_falls ----------------------------

def test_carrier_falls_then_carried_fragile_falls(lex):
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("botelo", lex, entity_id="botelo")
    t.assert_relation("havi", ("petro", "botelo"), lex)
    # Hand-craft a fali on petro (normally this would come from a hazard).
    t.events.append(make_event("fali", roles={"theme": "petro"}))

    run_dsl(t, [carried_fragile_falls_when_carrier_falls], [], lex)

    botelo_falls = [e for e in t.events
                    if e.action == "fali" and e.roles.get("theme") == "botelo"]
    assert len(botelo_falls) == 1


def test_carrier_falls_sturdy_object_does_not_fall(lex):
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("libro", lex, entity_id="libro")   # sturdy
    t.assert_relation("havi", ("petro", "libro"), lex)
    t.events.append(make_event("fali", roles={"theme": "petro"}))
    run_dsl(t, [carried_fragile_falls_when_carrier_falls], [], lex)
    libro_falls = [e for e in t.events
                   if e.action == "fali" and e.roles.get("theme") == "libro"]
    assert not libro_falls


# ---- end-to-end depth ----------------------------------------------------

def test_deep_cascade_reaches_depth_3(lex):
    """Canonical deep cascade via the wet-slip path. Depths are edge
    counts from the seed:
      fali(glaso)              depth 0  (seed)
      → fali(akvo)             depth 1  (container_falls_contents_fall)
      → aperi(flako)           depth 2  (wet_liquid_container_tips)
      → fali(petro)            depth 3  (person_slips_on_wet)
    The glaso→rompiĝi→aperi(shards) branch runs in parallel at depth
    ≤2 but the shards path doesn't extend the chain (shards don't
    cause slips under the new semantics).
    """
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("glaso", lex, entity_id="glaso")
    t.add_entity("akvo", lex, entity_id="akvo")
    t.assert_relation("en", ("petro", "kuirejo"), lex)
    t.assert_relation("en", ("glaso", "kuirejo"), lex)
    t.assert_relation("en", ("akvo", "glaso"), lex)
    t.events.append(make_event("fali", roles={"theme": "glaso"}))

    run_dsl(t, _all_rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    by_id = {ev.id: ev for ev in t.events}

    def depth(ev, memo={}):
        if ev.id in memo:
            return memo[ev.id]
        if not ev.caused_by:
            memo[ev.id] = 0
        else:
            memo[ev.id] = 1 + max(
                depth(by_id[c]) for c in ev.caused_by if c in by_id)
        return memo[ev.id]

    max_depth = max(depth(ev) for ev in t.events)
    assert max_depth >= 3, f"expected depth ≥3, got {max_depth}"


# ---- fire_spreads_to_adjacent_flammables ---------------------------------

def _bruli_seed(lex, theme_id):
    roles = {"theme": theme_id}
    return make_event(
        "bruli", roles=roles,
        property_changes=effect_changes("bruli", roles, lex))


def test_fire_spreads_via_sur_contact(lex):
    t = Trace()
    t.add_entity("laborejo", lex, entity_id="laborejo")
    t.add_entity("breto", lex, entity_id="breto")
    t.add_entity("ligno", lex, entity_id="ligno")
    t.assert_relation("en", ("breto", "laborejo"), lex)
    t.assert_relation("sur", ("ligno", "breto"), lex)
    t.events.append(_bruli_seed(lex, "ligno"))

    run_dsl(t, [fire_spreads_to_adjacent_flammables], DEFAULT_DSL_DERIVATIONS, lex)

    burning = {e.roles.get("theme") for e in t.events if e.action == "bruli"}
    assert burning == {"ligno", "breto"}


def test_fire_does_not_spread_to_nonflammables(lex):
    """Glass and water nearby don't burn."""
    t = Trace()
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("papero", lex, entity_id="papero")
    t.add_entity("glaso", lex, entity_id="glaso")
    t.add_entity("akvo", lex, entity_id="akvo")
    t.assert_relation("en", ("papero", "kuirejo"), lex)
    t.assert_relation("en", ("glaso", "kuirejo"), lex)
    t.assert_relation("en", ("akvo", "glaso"), lex)
    t.events.append(_bruli_seed(lex, "papero"))

    run_dsl(t, [fire_spreads_to_adjacent_flammables], DEFAULT_DSL_DERIVATIONS, lex)

    burning = {e.roles.get("theme") for e in t.events if e.action == "bruli"}
    # kuirejo is a location (skipped); glaso/akvo aren't flammable.
    assert burning == {"papero"}


def test_fire_does_not_spread_to_location(lex):
    """A room doesn't catch fire even though things in it do."""
    t = Trace()
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("papero", lex, entity_id="papero")
    t.assert_relation("en", ("papero", "kuirejo"), lex)
    t.events.append(_bruli_seed(lex, "papero"))
    run_dsl(t, [fire_spreads_to_adjacent_flammables], DEFAULT_DSL_DERIVATIONS, lex)
    assert not any(e.roles.get("theme") == "kuirejo"
                   for e in t.events if e.action == "bruli")


def test_fire_reaches_depth_3_via_nested_containment(lex):
    """papero en korbo sur tablo, libro sur tablo.
      bruli(papero)   d0
      → bruli(korbo)  d1  (papero en korbo)
      → bruli(tablo)  d2  (korbo sur tablo)
      → bruli(libro)  d3  (libro sur tablo)
    """
    t = Trace()
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("tablo", lex, entity_id="tablo")
    t.add_entity("korbo", lex, entity_id="korbo")
    t.add_entity("papero", lex, entity_id="papero")
    t.add_entity("libro", lex, entity_id="libro")
    t.assert_relation("en", ("tablo", "kuirejo"), lex)
    t.assert_relation("sur", ("korbo", "tablo"), lex)
    t.assert_relation("en", ("papero", "korbo"), lex)
    t.assert_relation("sur", ("libro", "tablo"), lex)
    t.events.append(_bruli_seed(lex, "papero"))

    run_dsl(t, [fire_spreads_to_adjacent_flammables], DEFAULT_DSL_DERIVATIONS, lex)

    by_id = {ev.id: ev for ev in t.events}

    def depth(ev, memo={}):
        if ev.id in memo:
            return memo[ev.id]
        if not ev.caused_by:
            memo[ev.id] = 0
        else:
            memo[ev.id] = 1 + max(
                depth(by_id[c]) for c in ev.caused_by if c in by_id)
        return memo[ev.id]

    assert max(depth(ev) for ev in t.events) >= 3


def test_deep_cascade_with_carried_fragile_reaches_depth_5(lex):
    """Longest chain: a wet glass falls and shatters, puddle forms,
    person slips on the puddle, drops their bottle, which breaks.
      fali(glaso)            depth 0
      → rompiĝi(glaso)       depth 1
      → fali(akvo)           depth 2
      → aperi(flako)         depth 3
      → fali(petro)          depth 4
      → fali(botelo)         depth 5
      → rompiĝi(botelo)      depth 6
      → aperi(shards_botelo) depth 7
    """
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("glaso", lex, entity_id="glaso")
    t.add_entity("akvo", lex, entity_id="akvo")
    t.add_entity("botelo", lex, entity_id="botelo")
    t.assert_relation("en", ("petro", "kuirejo"), lex)
    t.assert_relation("en", ("glaso", "kuirejo"), lex)
    t.assert_relation("en", ("akvo", "glaso"), lex)
    t.assert_relation("havi", ("petro", "botelo"), lex)
    t.events.append(make_event("fali", roles={"theme": "glaso"}))

    run_dsl(t, _all_rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    # petro slipped and dropped botelo.
    actions_by_theme = [(e.action, e.roles.get("theme")) for e in t.events]
    assert ("fali", "petro") in actions_by_theme
    assert ("fali", "botelo") in actions_by_theme
    assert ("rompiĝi", "botelo") in actions_by_theme

    by_id = {ev.id: ev for ev in t.events}

    def depth(ev, memo={}):
        if ev.id in memo:
            return memo[ev.id]
        if not ev.caused_by:
            memo[ev.id] = 0
        else:
            memo[ev.id] = 1 + max(
                depth(by_id[c]) for c in ev.caused_by if c in by_id)
        return memo[ev.id]

    max_depth = max(depth(ev) for ev in t.events)
    assert max_depth >= 5, f"expected depth ≥5, got {max_depth}"
