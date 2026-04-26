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
from ..loader import Lexicon
from .messages import (
    AppearanceMessage,
    CoordinatedMessage,
    DestructionMessage,
    EntityQualityMessage,
    EventMessage,
    GroupedRelationMessage,
    Message,
    RelationAddedMessage,
    RelationMessage,
    RelationRemovedMessage,
    SceneGroundingMessage,
    SubordinatedMessage,
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

    # 1. Scene grounding — implicit "X estis en la SCENE." lines for
    # non-person event participants without an explicit relation.
    for eid in _synthetic_grounding_targets(
            trace, scene_location_id, setup_relations):
        messages.append(SceneGroundingMessage(entity_id=eid))

    # 1b. Compute event-attached preconditions (active causal triggers
    # to fold into event sentences as "ĉar ..." clauses).
    event_preconds = _event_preconditions(trace, lexicon)
    # Entities whose precondition will be inlined — suppress separate
    # quality grounding for those.
    inlined_entities = {eid for (eid, _) in event_preconds.values()}

    # 1c. Quality grounding — surface CAUSALLY RELEVANT transient
    # states as standalone sentences ("La pordo estas fermita."), but
    # skip entities whose state will be folded into a `ĉar` clause.
    for msg in _quality_grounding_messages(trace, lexicon, inlined_entities):
        messages.append(msg)

    # 2. Scene-setup relations — from the snapshot if provided, else
    # the (possibly mutated) current relations.
    relations_for_setup = (setup_relations if setup_relations is not None
                           else list(trace.relations))
    for rel in relations_for_setup:
        messages.append(RelationMessage(relation=rel))

    # 3. Relation changes between setup and final + source annotations
    # for acquisition verbs. Both computed from the same diff pass.
    rel_changes_by_event, source_by_event = _attribute_relation_changes(
        trace, setup_relations)

    # 4. Events, with per-event trailers.
    for ev in trace.events:
        messages.append(EventMessage(
            event=ev,
            cause_event_id=(ev.caused_by[0] if ev.caused_by else None),
            source_entity_id=source_by_event.get(ev.id),
            precondition=event_preconds.get(ev.id)))

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


# -------------------- helpers: salience-driven quality grounding -----

def _extract_entity_constraints(patt) -> dict[str, object]:
    """Walk a pattern tree, return slot-value constraints from any
    EntityPattern under it. Skips type/concept/has_suffix (not slot
    values). Conservatively under-extracts on Or/Not — those are
    rare in our preconditions and over-extraction would surface
    irrelevant slots."""
    from ..dsl.patterns import (
        AndPattern, BindPattern, EntityPattern,
    )
    if isinstance(patt, EntityPattern):
        return {k: v for k, v in patt.constraints.items()
                if k not in ("type", "concept", "has_suffix")}
    if isinstance(patt, AndPattern):
        out = dict(_extract_entity_constraints(patt.left))
        out.update(_extract_entity_constraints(patt.right))
        return out
    if isinstance(patt, BindPattern):
        return {}
    # OrPattern, NotPattern, etc. — skip
    return {}


def _event_preconditions(
    trace: Trace, lexicon: Lexicon,
) -> dict[str, tuple[str, str]]:
    """For each event, find one ACTIVE precondition: an (entity_id,
    quality_lemma) pair where some verb-role or rule-pattern conditions
    on a slot AND the entity actually has the constraint's value.

    Returns {event_id: (entity_id, quality_lemma)} for events that have
    such a precondition. Used to:
      (1) suppress separate quality grounding for that entity (the
          quality is already going to be inlined),
      (2) attach the precondition to the EventMessage so the renderer
          emits a `ĉar` clause.

    For events with multiple satisfied preconditions, picks
    deterministically: alphabetical by role name, then by slot name.
    """
    from ..dsl.patterns import EventPattern
    from ..dsl.rules import DEFAULT_DSL_RULES
    rules = list(DEFAULT_DSL_RULES)

    out: dict[str, tuple[str, str]] = {}
    for ev in trace.events:
        # Synthesized events (cause is non-empty) inherit causal
        # connectives from their cause; skip ĉar for them.
        if ev.caused_by:
            continue
        action = lexicon.actions.get(ev.action)
        candidates: list[tuple[str, str, str]] = []  # (role, slot, expected)

        # (a) Verb-level role property constraints.
        if action is not None:
            for role in action.roles:
                if not role.properties:
                    continue
                if role.name not in ev.roles:
                    continue
                for slot, vals in role.properties.items():
                    if vals:
                        candidates.append((role.name, slot, vals[0]))

        # (b) Rule-level constraints from any rule whose `when` is an
        # EventPattern matching this event's action.
        for rule in rules:
            when = rule.when
            if not isinstance(when, EventPattern):
                continue
            if when.action != ev.action:
                continue
            for role_name, role_patt in when.role_patterns.items():
                if role_name not in ev.roles:
                    continue
                for slot, expected in _extract_entity_constraints(
                        role_patt).items():
                    if isinstance(expected, str):
                        candidates.append((role_name, slot, expected))

        # Filter to candidates whose actual entity value matches
        # AND whose slot is varies-flagged (otherwise the precondition
        # is identity, surfacing it as ĉar reads odd: "tranĉis ĉar
        # estas solida"). Pick one deterministically.
        candidates.sort()
        for role_name, slot, expected in candidates:
            slot_def = lexicon.slots.get(slot)
            if slot_def is None or not getattr(slot_def, "varies", False):
                continue
            eid = ev.roles[role_name]
            if not isinstance(eid, str):
                continue
            ent = trace.entities.get(eid)
            if ent is None:
                continue
            actual = ent.properties.get(slot, [])
            if isinstance(actual, list) and expected in actual:
                out[ev.id] = (eid, expected)
                break
            if actual == expected:
                out[ev.id] = (eid, expected)
                break
    return out


def _salient_entity_slots(
    trace: Trace, lexicon: Lexicon,
) -> set[tuple[str, str]]:
    """For each event in the trace, collect the (entity_id, slot_name)
    pairs that are causally relevant — meaning either:
      (a) the verb's role had a property constraint on that slot
          (verb-level precondition), or
      (b) a causal rule's `when` event matched the verb and constrained
          a role on that slot (rule-level precondition like
          hungry_eats_sated requiring agent.hunger=malsata).

    Effects (slots the verb writes) are NOT included — those become
    visible through the event narration and any state-change messages
    that follow; pre-grounding them would be redundant.
    """
    relevant: set[tuple[str, str]] = set()

    # (a) Verb-level role property constraints.
    for ev in trace.events:
        action = lexicon.actions.get(ev.action)
        if action is None:
            continue
        for role in action.roles:
            if not role.properties:
                continue
            if role.name not in ev.roles:
                continue
            eid = ev.roles[role.name]
            if not isinstance(eid, str):
                continue
            for slot in role.properties.keys():
                relevant.add((eid, slot))

    # (b) Rule-level role property constraints. Late-import to avoid
    # cycles with the realizer.
    from ..dsl.patterns import EventPattern
    from ..dsl.rules import DEFAULT_DSL_RULES
    rules = list(DEFAULT_DSL_RULES)
    for ev in trace.events:
        for rule in rules:
            when = rule.when
            if not isinstance(when, EventPattern):
                continue
            if when.action != ev.action:
                continue
            for role_name, role_patt in when.role_patterns.items():
                if role_name not in ev.roles:
                    continue
                eid = ev.roles[role_name]
                if not isinstance(eid, str):
                    continue
                for slot in _extract_entity_constraints(role_patt).keys():
                    relevant.add((eid, slot))

    return relevant


def _quality_grounding_messages(
    trace: Trace, lexicon: Lexicon,
    skip_entities: Optional[set[str]] = None,
) -> list[Message]:
    """Surface qualities that are CAUSALLY RELEVANT to the trace's
    events: only (entity, slot) pairs that some verb or rule's
    preconditions reference get pre-ground sentences.

    `skip_entities`: entities whose precondition will be inlined into
    an event sentence as a `ĉar` clause — skip standalone grounding
    for them to avoid duplication.

    This filters out atmospheric noise like "La sako estas malpura"
    when nothing in the trace actually reads cleanliness. What survives
    is preconditions that explain the events: "La pordo estas fermita."
    landing before "Maria malfermis la pordon" only if some verb/rule
    conditions on the theme's openness AND the precondition isn't
    already being folded into the event sentence.

    Skips persons (predicative form like "Maria estas malsata" reads
    stilted in Esperanto; attributive "Hungra Maria" is the natural
    shape but hasn't been built yet) and mid-trace creations (those
    get AppearanceMessage instead)."""
    salient = _salient_entity_slots(trace, lexicon)
    skip = skip_entities or set()

    out: list[Message] = []
    seen: set[str] = set()  # at most one quality per entity
    for (eid, slot_name) in sorted(salient):
        if eid in seen or eid in skip:
            continue
        ent = trace.entities.get(eid)
        if ent is None:
            continue
        if ent.entity_type == "person":
            continue
        if ent.created_at_event is not None:
            continue
        slot = lexicon.slots.get(slot_name)
        if slot is None or not getattr(slot, "varies", False):
            continue
        value = ent.properties.get(slot_name)
        if not value:
            continue
        quality = value[0] if isinstance(value, list) else value
        out.append(EntityQualityMessage(
            entity_id=eid, quality_lemma=quality))
        seen.add(eid)
    return out


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

    # Entities that are sub-entity-parts of something else (body parts,
    # locks, etc.) shouldn't be grounded as independent scene contents
    # — they surface via their host. Walk havas_parton, collect the
    # parto-side ids.
    is_part: set[str] = set()
    for r in rels:
        if r.relation == "havas_parton" and len(r.args) == 2:
            is_part.add(r.args[1])

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
        # Don't ground other locations as scene contents — they're
        # destinations of motion (iri/veni/veturi) or otherwise
        # parallel to the scene, not inside it. Saying "Estas oficejo
        # en la insulo" is wrong both spatially and narratively.
        if ent.entity_type == "location":
            continue
        # Skip abstract entities (faktos and the like) — they don't
        # live in the scene's physical containment graph; saying
        # "Fakto estis en la dormejo" is incoherent. Knowledge of the
        # fakto surfaces via konas-relation prose instead.
        if ent.entity_type == "abstract":
            continue
        # Skip sub-entity parts (body parts, locks). They surface via
        # their host's prose, not as standalone scene contents.
        if eid in is_part:
            continue
        if ent.created_at_event is not None:
            continue
        out.append(eid)
    return out


# ------------------- helpers: relation change attribution -----------

def _attribute_relation_changes(
    trace: Trace,
    setup_relations: Optional[list[RelationAssertion]],
) -> tuple[dict[str, list[Message]], dict[str, str]]:
    """Diff setup vs final `trace.relations` and attribute each add/
    remove to a plausible event.

    Returns `(change_messages, source_for_event)`:
      change_messages[event_id] — RelationRemoved/Added messages to
        emit after the event's sentence.
      source_for_event[event_id] — for acquisition verbs (preni,
        kapti) whose rule consumed a havi-removal from some entity
        other than the agent, the previous owner's id. Renders as
        "de <source>" after the theme.

    Heuristic: walk events forward; a removal of `havi(A, X)` pairs
    with the first event whose action is one of the 'ownership-moving'
    verbs and whose roles reference A or X. Similarly `en(A, L1)`
    removal pairs with the first iri/veturi event for A.

    This is a deliberately shallow heuristic — tight enough for the
    current rule set, loose enough that future rules may need their
    own attribution entries. The full pipeline replaces this with
    per-event relation-delta tracking in the engine itself.
    """
    if setup_relations is None:
        return {}, {}

    setup_set = {(r.relation, tuple(r.args)) for r in setup_relations}
    final_set = {(r.relation, tuple(r.args)) for r in trace.relations}
    added = final_set - setup_set
    removed = setup_set - final_set
    if not added and not removed:
        return {}, {}

    # Actions whose semantics already communicate ownership transfer
    # (doni X al Y implies Y now has X; kapti implies the agent now
    # has the theme; ĵeti implies the agent no longer does).
    # Narrating "Maria ne plu havas la libron" after "Maria donas la
    # libron al Petro" is redundant — the realizer suppresses both
    # sides of the transfer for these verbs.
    # Transfer verbs already convey ownership change in the verb
    # itself. Acquisition verbs (a subset) can optionally narrate
    # the previous owner as "de <owner>" on the event itself.
    TRANSFER_VERBS = {"doni", "preni", "ĵeti", "kapti"}
    ACQUISITION_VERBS = {"preni", "kapti"}
    MOVEMENT_VERBS = {"iri", "veturi"}
    PLACEMENT_VERBS = {"meti"}
    candidate_actions = TRANSFER_VERBS | MOVEMENT_VERBS | PLACEMENT_VERBS

    out: dict[str, list[Message]] = {}
    source_for_event: dict[str, str] = {}
    unattributed_adds = set(added)
    unattributed_removes = set(removed)

    for ev in trace.events:
        if ev.action not in candidate_actions:
            continue
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

        narrate_ownership = ev.action not in TRANSFER_VERBS
        # Acquisition verbs: if we consumed a havi-removal whose
        # owner differs from the event's agent, that owner is the
        # "source" to narrate via "de <source>" on the event itself.
        if ev.action in ACQUISITION_VERBS:
            agent = ev.roles.get("agent")
            theme = ev.roles.get("theme")
            for (rel, args) in claims_rem:
                if (rel == "havi" and len(args) == 2
                        and args[1] == theme and args[0] != agent):
                    source_for_event[ev.id] = args[0]
                    break

        for (rel, args) in claims_rem:
            unattributed_removes.discard((rel, args))
            if rel == "havi" and not narrate_ownership:
                continue
            out.setdefault(ev.id, []).append(
                RelationRemovedMessage(
                    relation=rel, args=args, cause_event_id=ev.id))
        for (rel, args) in claims_add:
            unattributed_adds.discard((rel, args))
            if rel == "havi" and narrate_ownership:
                out.setdefault(ev.id, []).append(
                    RelationAddedMessage(
                        relation=rel, args=args, cause_event_id=ev.id))
    return out, source_for_event


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


# --------------------------- transforms -------------------------

def aggregate_same_subject(
    messages: list[Message], lexicon: Lexicon,
) -> list[Message]:
    """Combine adjacent EventMessages that share a grammatical subject
    into a `CoordinatedMessage`.

    The causal-link requirement has been dropped — adjacent same-
    subject events coordinate naturally in prose even without a
    direct cause relation: "La hundo kuras kaj kaptas la pilkon"
    reads better than two sentences regardless of whether the run
    causes the catch.

    Scope:
      - Only combines EventMessage with EventMessage.
      - Subjects must be identical (resolved from the verb's role
        structure — agent if transitive, theme if intransitive).
      - Subject must be resolvable (impersonals like `pluvi` never
        aggregate — nothing to share between them).

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
        if base_subject is None:
            out.append(m)
            i += 1
            continue

        j = i + 1
        while j < len(messages) and isinstance(messages[j], EventMessage):
            nxt: EventMessage = messages[j]
            if _event_subject(nxt.event, lexicon) != base_subject:
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


def subordinate_creations(messages: list[Message]) -> list[Message]:
    """Fuse cascade event → aperi-event pairs into a "el kio" clause.

    Pattern: an EventMessage or CoordinatedMessage whose contained
    events include the cause of the *immediately following* `aperi`
    EventMessage (or an AppearanceMessage). The follow-on collapses
    into a subordinate clause on the cascade event:

        "La glaso falis kaj rompiĝis. Vitropecetoj aperis."
        →  "La glaso falis kaj rompiĝis, el kio aperis vitropecetoj."

    Runs after aggregation — so coordinated chains get the
    subordinated clause too. Conservative: only single-appearance
    subordination; a cascade producing two appearances keeps one
    subordinated and the other standalone.
    """
    if not messages:
        return messages
    out: list[Message] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if isinstance(m, (EventMessage, CoordinatedMessage)):
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            follow_cause = _appearance_cause(nxt)
            if (follow_cause is not None
                    and follow_cause in _contained_event_ids(m)):
                out.append(SubordinatedMessage(
                    main=m,
                    subordinate=nxt,                 # type: ignore[arg-type]
                    conjunction="el kio",
                    cause_event_id=m.cause_event_id,
                ))
                i += 2
                continue
        out.append(m)
        i += 1
    return out


def aggregate_relations(messages: list[Message]) -> list[Message]:
    """Group setup-phase RelationMessages by (relation, container).

    Walks the leading run of RelationMessages — the scene-setup phase
    — re-orders them so siblings sharing a container appear together,
    then collapses each `(en|sur, container)` group into one
    `GroupedRelationMessage`. Result: instead of

      "En la kuirejo estas Petro. En la kuirejo estas tablo. Sur la
       tablo estas glaso. En la kuirejo estas korbo."

    the planner emits

      "En la kuirejo estas Petro, tablo, kaj korbo. Sur la tablo
       estas glaso."

    Only the leading run is touched — once an EventMessage or any
    non-RelationMessage appears, that's the boundary; the rest of
    the message stream stays in original order. `havi` and `apud`
    relations are left as individual RelationMessages because
    they don't aggregate as cleanly.
    """
    if not messages:
        return messages
    # Find the leading run of RelationMessages (the setup phase).
    cut = 0
    while cut < len(messages) and isinstance(messages[cut], RelationMessage):
        cut += 1
    if cut == 0:
        return messages

    setup = messages[:cut]
    rest = messages[cut:]

    # Bucket by (relation, container). Track first-appearance order
    # so the output preserves the rough scene-introduction sequence.
    buckets: dict[tuple[str, str], list[str]] = {}
    bucket_order: list[tuple[str, str]] = []
    leftover: list[RelationMessage] = []

    for m in setup:
        rel = m.relation
        if rel.relation in ("en", "sur") and len(rel.args) == 2:
            key = (rel.relation, rel.args[1])  # (relation, container)
            if key not in buckets:
                buckets[key] = []
                bucket_order.append(key)
            buckets[key].append(rel.args[0])
        else:
            leftover.append(m)

    out: list[Message] = []
    for key in bucket_order:
        rel_name, container_id = key
        contained_ids = buckets[key]
        if len(contained_ids) >= 2:
            out.append(GroupedRelationMessage(
                relation=rel_name,
                container_id=container_id,
                contained_ids=contained_ids,
            ))
        else:
            # Singleton: fall back to a regular RelationMessage so
            # the existing renderer with its template variation kicks
            # in.
            from ..causal import RelationAssertion
            out.append(RelationMessage(
                relation=RelationAssertion(
                    relation=rel_name, args=(contained_ids[0], container_id))))
    out.extend(leftover)
    out.extend(rest)
    return out


def _appearance_cause(m: Optional[Message]) -> Optional[str]:
    """If `m` represents an appearance (either an AppearanceMessage or
    an EventMessage whose verb is `aperi`), return the id of its
    causing event — otherwise None. Both shapes can occur depending
    on whether the rule attached the new entity via `Event.creates`
    or emitted a standalone aperi event."""
    if isinstance(m, AppearanceMessage):
        return m.cause_event_id
    if isinstance(m, EventMessage) and m.event.action == "aperi":
        return m.event.caused_by[0] if m.event.caused_by else None
    return None


def _contained_event_ids(m: Message) -> set[str]:
    """Event ids that this message renders. Used to decide whether a
    follow-on message whose `cause_event_id` names one of them should
    subordinate into it."""
    if isinstance(m, EventMessage):
        return {m.event.id}
    if isinstance(m, CoordinatedMessage):
        ids: set[str] = set()
        for c in m.children:
            if isinstance(c, EventMessage):
                ids.add(c.event.id)
        return ids
    return set()
