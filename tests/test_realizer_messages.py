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
    RelationRemovedMessage, aggregate_same_subject, plan_messages,
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

def test_doni_narrates_previous_owner_lost(lex):
    """After Maria gives the book to Petro, we want a sentence saying
    she no longer has it."""
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
    assert any(m.relation == "havi" and m.args == ("maria", "libro")
               for m in removed), [type(m).__name__ for m in messages]

    prose = realize_trace(t, lex, setup_relations=setup)
    assert "ne plu hav" in prose, prose


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
