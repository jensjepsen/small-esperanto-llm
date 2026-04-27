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
    return collect_rules(R)[1]


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
    """Outdoor location's lit_state=luma is BAKED on parko's concept
    via outdoor_is_luma; agent in such a location derives illuminated=yes."""
    t = _make_trace(lex,
        entities=[("parko", "parko"), ("persono", "Mikael")],
        en_relations=[("Mikael", "parko")])
    d = compute_derived_state(t, derivations, lex)
    # parko's lit_state is in concept.properties (baked), so derived
    # state doesn't add it — but agent_illuminated DOES need to fire
    # against the baked value.
    assert "luma" in t.entities["parko"].properties.get("lit_state", [])
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
    # pordo lifts lock_state from seruro part:
    pordo = lex.concepts["pordo"]
    assert pordo.properties.get("lock_state") == ["ŝlosita"]
    assert pordo.properties.get("lock_capable") == ["yes"]
    # Outdoor location is_luma baked:
    parko = lex.concepts["parko"]
    assert parko.properties.get("lit_state") == ["luma"]
    # Indoor location is NOT baked malluma (NotPattern derivation
    # excluded from bake; runtime decides):
    biblioteko = lex.concepts["biblioteko"]
    assert biblioteko.properties.get("lit_state") is None
