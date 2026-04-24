"""Plan + transform phases.

`plan_messages(trace, lexicon, ...)` walks a trace once and produces
a flat list of Message objects. Mostly 1:1 with events and relations
in this slice — a future document planner replaces this function
wholesale without touching the IR or the renderer.

`aggregate_same_subject(messages)` is the one transform this slice
ships. Adjacent messages that share a subject and are on the same
causal chain collapse into a `CoordinatedMessage`; the renderer
joins them with 'kaj'. Future transforms (subordination, participial
backgrounding, relative clauses) plug into the same stage — each is
an independent function taking and returning `list[Message]`.
"""
from __future__ import annotations

from typing import Optional

from ..causal import Event, RelationAssertion, Trace
from ..loader import Lexicon, resolve_signature
from .messages import (
    AppearanceMessage,
    CoordinatedMessage,
    DestructionMessage,
    EventMessage,
    Message,
    RelationAddedMessage,
    RelationMessage,
    RelationRemovedMessage,
    SceneGroundingMessage,
)


# --------------------------- planning ---------------------------

def plan_messages(
    trace: Trace, lexicon: Lexicon, *,
    scene_location_id: Optional[str] = None,
    setup_relations: Optional[list[RelationAssertion]] = None,
) -> list[Message]:
    """Build the flat list of messages for a trace.

    Order: synthetic grounding → scene-setup relations → per-event
    group (event + appearance lines + state-change narration for that
    event). Entity destructions from `manĝi` etc. get a DestructionMessage
    attached to the destroying event's group.
    """
    messages: list[Message] = []
    skip_uzi_ids = _use_instrument_skip_set(trace, lexicon)

    # 1. Scene grounding — implicit "X estis en la SCENE." lines for
    # non-person event participants without an explicit relation.
    for eid in _synthetic_grounding_targets(
            trace, scene_location_id, setup_relations):
        messages.append(SceneGroundingMessage(entity_id=eid))

    # 2. Scene-setup relations — from the snapshot if provided, else
    # the (possibly mutated) current relations.
    relations_for_setup = (setup_relations if setup_relations is not None
                           else list(trace.relations))
    for rel in relations_for_setup:
        messages.append(RelationMessage(relation=rel))

    # 3. Relation changes between setup and final — only computed when
    # the caller provided a snapshot. Index by plausible causing event
    # so each change lands after the right event in the output.
    rel_changes_by_event = _attribute_relation_changes(
        trace, setup_relations)

    # 4. Events, with per-event trailers.
    for ev in trace.events:
        if ev.id in skip_uzi_ids:
            continue
        # Events synthesized by relation-only rules (bare preni/meti/doni
        # with a synthetic "_wet" action etc.) don't render — see the
        # renderer's dispatch. But their relation effects still narrate.
        messages.append(EventMessage(
            event=ev,
            cause_event_id=(ev.caused_by[0] if ev.caused_by else None)))

        # Appearance lines for entities created by this event.
        for created in ev.creates:
            messages.append(AppearanceMessage(
                entity_id=created.id, cause_event_id=ev.id))

        # Relation changes attributed to this event.
        for msg in rel_changes_by_event.get(ev.id, []):
            messages.append(msg)

        # Destructions whose destroyed_at_event index matches this event.
        for dmsg in _destruction_messages_for_event(trace, ev):
            messages.append(dmsg)

    return messages


# -------------------- helpers: synthetic grounding -------------------

def _synthetic_grounding_targets(
    trace: Trace,
    scene_location_id: Optional[str],
    setup_relations: Optional[list[RelationAssertion]],
) -> list[str]:
    """Entity ids that need 'X estis en la SCENE.' grounding.

    Skips: scene itself, persons, mid-trace-created entities, and
    anything already in setup_relations.
    """
    if scene_location_id is None or scene_location_id not in trace.entities:
        return []

    in_relations: set[str] = set()
    rels = setup_relations if setup_relations is not None else trace.relations
    for r in rels:
        in_relations.update(r.args)

    in_events: set[str] = set()
    for ev in trace.events:
        for v in ev.roles.values():
            if isinstance(v, str):
                in_events.add(v)

    out: list[str] = []
    for eid in sorted(in_events):
        if eid == scene_location_id:
            continue
        if eid in in_relations:
            continue
        ent = trace.entities.get(eid)
        if ent is None or ent.entity_type == "person":
            continue
        if ent.created_at_event is not None:
            continue
        out.append(eid)
    return out


# ------------------- helpers: relation change attribution -----------

def _attribute_relation_changes(
    trace: Trace,
    setup_relations: Optional[list[RelationAssertion]],
) -> dict[str, list[Message]]:
    """Diff setup vs final `trace.relations` and attribute each add/
    remove to a plausible event. Returns `{event_id: [messages]}`.

    Heuristic: walk events forward; a removal of `havi(A, X)` pairs
    with the first event whose action is one of the 'ownership-moving'
    verbs (doni, preni, ĵeti, kapti) and whose roles reference A or X.
    Similarly `en(A, L1)` removal pairs with the first iri/veturi
    event for A.

    This is a deliberately shallow heuristic — tight enough for the
    current rule set, loose enough that future rules may need their
    own attribution entries. The full pipeline replaces this with
    per-event relation-delta tracking in the engine itself.
    """
    if setup_relations is None:
        return {}

    setup_set = {(r.relation, tuple(r.args)) for r in setup_relations}
    final_set = {(r.relation, tuple(r.args)) for r in trace.relations}
    added = final_set - setup_set
    removed = setup_set - final_set
    if not added and not removed:
        return {}

    out: dict[str, list[Message]] = {}
    unattributed_adds = set(added)
    unattributed_removes = set(removed)

    for ev in trace.events:
        # Candidate actions for ownership changes.
        if ev.action in {"doni", "preni", "ĵeti", "kapti", "iri", "veturi",
                         "meti"}:
            ev_referents = {v for v in ev.roles.values()
                            if isinstance(v, str)}
            claims_rem: list = []
            claims_add: list = []
            for (rel, args) in list(unattributed_removes):
                if set(args) & ev_referents:
                    claims_rem.append((rel, args))
            for (rel, args) in list(unattributed_adds):
                if set(args) & ev_referents:
                    claims_add.append((rel, args))
            for (rel, args) in claims_rem:
                unattributed_removes.discard((rel, args))
                out.setdefault(ev.id, []).append(
                    RelationRemovedMessage(
                        relation=rel, args=args, cause_event_id=ev.id))
            for (rel, args) in claims_add:
                unattributed_adds.discard((rel, args))
                # RelationAddedMessage is only emitted for havi — the
                # location-changing verbs already narrate destination
                # via `al`, making addition narration redundant.
                if rel == "havi":
                    out.setdefault(ev.id, []).append(
                        RelationAddedMessage(
                            relation=rel, args=args, cause_event_id=ev.id))
    return out


# ----------------- helpers: destruction narration -------------------

def _destruction_messages_for_event(
    trace: Trace, ev: Event,
) -> list[Message]:
    """Entities destroyed AT this event's position get a
    DestructionMessage attached to it. Matching via trace_position
    when set; falling back to events.index otherwise."""
    try:
        ev_pos = (ev.trace_position if ev.trace_position is not None
                  else trace.events.index(ev))
    except ValueError:
        return []
    out: list[Message] = []
    for eid, ent in trace.entities.items():
        if ent.destroyed_at_event == ev_pos:
            out.append(DestructionMessage(
                entity_id=eid, cause_event_id=ev.id))
    return out


# ------------------- helpers: use-instrument fusion -----------------

def _use_instrument_skip_set(
    trace: Trace, lexicon: Lexicon,
) -> set[str]:
    """Identify `uzi` events whose synthesized verb-event we'll render
    instead. Same logic as the old realizer — kept here because it's a
    planning-stage concern (decide which events to drop) rather than a
    rendering concern."""
    by_id = {e.id: e for e in trace.events}
    skip: set[str] = set()
    for e2 in trace.events:
        if not e2.caused_by or len(e2.caused_by) != 1:
            continue
        e1 = by_id.get(e2.caused_by[0])
        if e1 is None or e1.action != "uzi":
            continue
        instr_id = e1.roles.get("instrument")
        if not instr_id:
            continue
        instr = trace.entity(instr_id)
        if instr is None:
            continue
        instr_concept = lexicon.concepts.get(instr.concept_lemma)
        if instr_concept is None:
            continue
        source = resolve_signature(lexicon, instr_concept)
        if source is None or source.lemma != e2.action:
            continue
        skip.add(e1.id)
    return skip


# --------------------------- transforms -------------------------

def aggregate_same_subject(
    messages: list[Message], lexicon: Lexicon,
) -> list[Message]:
    """Combine adjacent messages that share a grammatical subject
    and fall on the same causal chain into a `CoordinatedMessage`.

    Example: 'Petro falas. Petro rompiĝas la glason.' (same subject)
    on the same cause-chain collapses to 'Petro falas kaj rompas la
    glason.'

    Scope for this iteration:
      - Only combines EventMessage with EventMessage.
      - Subjects must be identical (resolved from the verb's role
        structure — agent if transitive, theme if intransitive).
      - Second message must have cause_event_id pointing to an event
        in the first's chain (including the first message's event).
      - A run of ≥2 such messages becomes one CoordinatedMessage.

    State-change messages (RelationRemoved, DestructionMessage etc.)
    are intentionally not aggregated yet — they're rendered as follow-
    on clauses, not coordinated verb phrases.
    """
    if not messages:
        return messages

    out: list[Message] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if not isinstance(m, EventMessage):
            out.append(m)
            i += 1
            continue

        run: list[EventMessage] = [m]
        base_subject = _event_subject(m.event, lexicon)
        j = i + 1
        while j < len(messages) and isinstance(messages[j], EventMessage):
            nxt: EventMessage = messages[j]
            if _event_subject(nxt.event, lexicon) != base_subject:
                break
            if base_subject is None:
                break
            # Causal link: the next event must be caused by something
            # in the growing run (transitively). Conservative: require
            # direct causation by one of the run's events.
            run_ids = {r.event.id for r in run}
            if not any(cid in run_ids for cid in nxt.event.caused_by):
                break
            run.append(nxt)
            j += 1

        if len(run) >= 2:
            out.append(CoordinatedMessage(
                children=list(run),
                cause_event_id=run[0].cause_event_id))
            i = j
        else:
            out.append(m)
            i += 1

    return out


def _event_subject(ev: Event, lexicon: Lexicon) -> Optional[str]:
    """Which entity id is the grammatical subject of this event?
    Agent if the verb declares one, theme if intransitive, None for
    impersonals (pluvi et al.)."""
    action = lexicon.actions.get(ev.action)
    if action is None:
        return None
    role_names = {r.name for r in action.roles}
    if "agent" in role_names and ev.roles.get("agent"):
        return ev.roles["agent"]
    if "theme" in role_names and ev.roles.get("theme"):
        return ev.roles["theme"]
    return None
