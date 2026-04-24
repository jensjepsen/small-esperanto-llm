"""Tests for the message-IR realizer rebuild.

Three focused groups:
  - Aggregation: same-subject events combine with "kaj".
  - Relation-change narration: preni/doni emit "ne plu havis".
  - Destruction narration: manĝi emits "malaperis".

Plan-level and render-level both covered — some tests inspect the
intermediate message list, others just the final prose.
"""
from __future__ import annotations

import pytest

from esperanto_lm.ontology import (
    Trace, effect_changes, load_lexicon, make_event, realize_trace,
)
from esperanto_lm.ontology.dsl import run_dsl
from esperanto_lm.ontology.dsl.rules import (
    DEFAULT_DSL_DERIVATIONS, DEFAULT_DSL_RULES, make_use_instrument_rules,
)
from esperanto_lm.ontology.realize import (
    CoordinatedMessage, DestructionMessage, EventMessage,
    RelationRemovedMessage, SubordinatedMessage,
    aggregate_same_subject, plan_messages, subordinate_creations,
)


@pytest.fixture
def lex():
    return load_lexicon()


def _rules(lex):
    return DEFAULT_DSL_RULES + make_use_instrument_rules(lex)


# ========================== aggregation ==========================

def test_same_subject_events_aggregate_into_coordinated_message(lex):
    """fali(glaso) causes rompiĝi(glaso) — same subject, direct cause.
    They collapse into one CoordinatedMessage."""
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    t.events.append(make_event("fali", roles={"theme": "glaso"}))
    run_dsl(t, _rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    messages = plan_messages(t, lex)
    messages = aggregate_same_subject(messages, lex)

    coord = [m for m in messages if isinstance(m, CoordinatedMessage)]
    assert len(coord) == 1, [type(m).__name__ for m in messages]
    # Children are the fali and rompiĝi EventMessages.
    child_actions = [
        c.event.action for c in coord[0].children
        if isinstance(c, EventMessage)
    ]
    assert child_actions == ["fali", "rompiĝi"]


def test_aggregated_prose_uses_kaj(lex):
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    t.events.append(make_event("fali", roles={"theme": "glaso"}))
    run_dsl(t, _rules(lex), DEFAULT_DSL_DERIVATIONS, lex)
    prose = realize_trace(t, lex)
    assert "falis kaj rompiĝis" in prose, prose


def test_different_subjects_do_not_aggregate(lex):
    """fali(glaso) cascades into fali(akvo) — different subjects,
    should NOT aggregate."""
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    t.add_entity("akvo", lex, entity_id="akvo")
    t.assert_relation("en", ("akvo", "glaso"), lex)
    t.events.append(make_event("fali", roles={"theme": "glaso"}))
    run_dsl(t, _rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    prose = realize_trace(t, lex)
    # glaso's fali + rompiĝi coordinate; akvo's fali stays separate.
    assert "glaso falis kaj rompiĝis" in prose, prose
    assert "akvo falis" in prose
    # The akvo sentence is not joined with glaso's via kaj.
    assert "rompiĝis kaj" not in prose, prose


def test_single_event_not_wrapped_as_coordinated(lex):
    """A fali that doesn't trigger any cascade stays as a plain
    EventMessage — no coordination of length 1."""
    t = Trace()
    t.add_entity("tablo", lex, entity_id="tablo")  # sturdy, doesn't break
    t.events.append(make_event("fali", roles={"theme": "tablo"}))
    run_dsl(t, _rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    messages = aggregate_same_subject(plan_messages(t, lex), lex)
    coords = [m for m in messages if isinstance(m, CoordinatedMessage)]
    assert coords == []
    events = [m for m in messages if isinstance(m, EventMessage)]
    assert len(events) == 1 and events[0].event.action == "fali"


# ==================== relation-change narration ====================

def test_doni_suppresses_redundant_ownership_narration(lex):
    """`doni X al Y` already conveys the transfer — don't add
    "Maria ne plu havas la libron. Nun Petro havas la libron." The
    planner suppresses both sides for transfer verbs."""
    t = Trace()
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("libro",   lex, entity_id="libro")
    t.assert_relation("havi", ("maria", "libro"), lex)
    t.events.append(make_event("doni", roles={
        "agent": "maria", "theme": "libro", "recipient": "petro"}))
    setup = t.snapshot_relations()
    run_dsl(t, _rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    messages = plan_messages(t, lex, setup_relations=setup)
    removed = [m for m in messages if isinstance(m, RelationRemovedMessage)]
    assert not removed, "transfer verbs shouldn't emit ownership-loss"

    prose = realize_trace(t, lex, setup_relations=setup)
    assert "ne plu" not in prose, prose
    # Main event still renders; tense depends on rng defaulting.
    assert "donis" in prose or "donas" in prose


def test_iri_does_not_narrate_en_removal(lex):
    """Moving between rooms: `iri` implies the location change via
    `al <dest>`. Narrating "ne plu estis en kuirejo" would be
    redundant, so the realizer skips it."""
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("salono",  lex, entity_id="salono")
    t.assert_relation("en", ("petro", "kuirejo"), lex)
    t.events.append(make_event("iri", roles={
        "agent": "petro", "destination": "salono"}))
    setup = t.snapshot_relations()
    run_dsl(t, _rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    prose = realize_trace(t, lex, setup_relations=setup)
    assert "ne plu" not in prose, (
        f"en-removal shouldn't narrate (implied by iri...al); got: {prose!r}")
    # And the move still reads correctly.
    assert "al salono" in prose, prose


# ==================== destruction narration ====================

def test_manĝi_narrates_destruction(lex):
    """Kato manĝas muson → 'La muso malaperis.'"""
    t = Trace()
    t.add_entity("kato", lex, entity_id="kato")
    t.add_entity("muso", lex, entity_id="muso")
    roles = {"agent": "kato", "theme": "muso"}
    t.events.append(make_event(
        "manĝi", roles=roles,
        property_changes=effect_changes("manĝi", roles, lex)))
    run_dsl(t, _rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    messages = plan_messages(t, lex)
    dests = [m for m in messages if isinstance(m, DestructionMessage)]
    assert any(m.entity_id == "muso" for m in dests)

    prose = realize_trace(t, lex)
    assert "malaperis" in prose, prose


# ==================== object elision ====================

def test_coordinated_children_share_theme_elide(lex):
    """Two cooks-and-eats on la pano: theme renders once, not twice.
    'Ŝi kuiras kaj manĝas la panon.'"""
    t = Trace()
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("pano",    lex, entity_id="pano")
    t.assert_relation("havi", ("maria", "pano"), lex)
    t.entities["maria"].set_property("hunger", "hungry")
    roles = {"agent": "maria", "theme": "pano"}
    t.events.append(make_event(
        "kuiri", roles=roles,
        property_changes=effect_changes("kuiri", roles, lex)))
    t.events.append(make_event(
        "manĝi", roles=roles,
        property_changes=effect_changes("manĝi", roles, lex)))
    run_dsl(t, _rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    prose = realize_trace(t, lex)
    # Both verbs render, but "la panon" only once in the coordinated
    # phrase (initial havi setup also mentions panon).
    coord_part = prose.split(".")[2] if len(prose.split(".")) > 2 else ""
    # A looser check: in the whole prose, "la panon" appears exactly once.
    # (initial setup is "panon" bare, the coordination is "la panon".)
    assert prose.count("la panon") == 1, prose
    assert "kuir" in prose and "manĝ" in prose


# ==================== source narration ====================

def test_preni_from_owner_narrates_de_source(lex):
    """Klara prenas la libron → the libro was Maria's → 'de Maria'."""
    t = Trace()
    t.add_entity("persono", lex, entity_id="klara")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("libro",   lex, entity_id="libro")
    t.assert_relation("havi", ("maria", "libro"), lex)
    t.events.append(make_event(
        "preni", roles={"agent": "klara", "theme": "libro"}))
    setup = t.snapshot_relations()
    run_dsl(t, _rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    prose = realize_trace(t, lex, setup_relations=setup)
    assert "de Maria" in prose or "de ŝi" in prose, prose


def test_kapti_from_nobody_has_no_source(lex):
    """If no one owns the theme when the catch happens, no 'de X'.
    ĵeti removes ownership first, so a subsequent kapti catches an
    ownerless ball."""
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("hundo",   lex, entity_id="hundo")
    t.add_entity("pilko",   lex, entity_id="pilko")
    t.assert_relation("havi", ("petro", "pilko"), lex)
    t.events.append(make_event(
        "ĵeti",  roles={"agent": "petro", "theme": "pilko"}))
    t.events.append(make_event(
        "kapti", roles={"agent": "hundo", "theme": "pilko"}))
    setup = t.snapshot_relations()
    run_dsl(t, _rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    prose = realize_trace(t, lex, setup_relations=setup)
    assert " de " not in prose, prose


# ==================== subordination ====================

def test_cascade_creation_subordinates_with_el_kio(lex):
    """rompiĝi → aperi becomes one sentence with 'el kio'.
    'La glaso falis kaj rompiĝis, el kio aperis vitropecetoj.'"""
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    t.events.append(make_event("fali", roles={"theme": "glaso"}))
    run_dsl(t, _rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    messages = plan_messages(t, lex)
    messages = aggregate_same_subject(messages, lex)
    messages = subordinate_creations(messages)

    subs = [m for m in messages if isinstance(m, SubordinatedMessage)]
    assert len(subs) == 1, [type(m).__name__ for m in messages]
    assert subs[0].conjunction == "el kio"

    prose = realize_trace(t, lex)
    assert "el kio" in prose, prose
    assert "vitropecetoj" in prose
    # The main and subordinate share one sentence, not two.
    assert prose.count("aperis") == 1 or prose.count("aperas") == 1


def test_appearance_without_matching_cause_does_not_subordinate(lex):
    """If an appearance's cause isn't the immediately preceding
    message's contained event, don't subordinate — leave as separate
    sentences."""
    # Pretend aperi is manually seeded (not cascade-produced).
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    t.events.append(make_event("fali", roles={"theme": "glaso"}))
    # Free-floating appearance with no cause event.
    t.events.append(make_event("aperi", roles={"theme": "glaso"}))
    # Don't run DSL — just test the planner on hand-crafted events.

    messages = subordinate_creations(
        aggregate_same_subject(plan_messages(t, lex), lex))
    subs = [m for m in messages if isinstance(m, SubordinatedMessage)]
    assert subs == []


def test_destruction_attached_to_eating_event(lex):
    """DestructionMessage should land in the plan right after the
    manĝi EventMessage, not drift to the end."""
    t = Trace()
    t.add_entity("kato", lex, entity_id="kato")
    t.add_entity("muso", lex, entity_id="muso")
    roles = {"agent": "kato", "theme": "muso"}
    t.events.append(make_event(
        "manĝi", roles=roles,
        property_changes=effect_changes("manĝi", roles, lex)))
    run_dsl(t, _rules(lex), DEFAULT_DSL_DERIVATIONS, lex)

    messages = plan_messages(t, lex)
    # Find manĝi then first DestructionMessage.
    idx_eat = next(
        i for i, m in enumerate(messages)
        if isinstance(m, EventMessage) and m.event.action == "manĝi")
    idx_dest = next(
        i for i, m in enumerate(messages)
        if isinstance(m, DestructionMessage))
    assert idx_dest == idx_eat + 1, (
        f"destruction should immediately follow eating; "
        f"got messages {[type(m).__name__ for m in messages]}")
