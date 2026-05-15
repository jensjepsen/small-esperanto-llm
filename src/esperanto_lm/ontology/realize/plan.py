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

    Layout: each event becomes a "group" — its grounding preamble
    (scene-grounding / setup-relations / quality-grounding for entities
    making their first appearance at this event), the EventMessage
    itself, then per-event trailers (appearances, relation-change
    narration, destructions). Grounding messages are anchored to the
    event where their entity is first referenced, so "En la kuirejo
    estis pomo" sits right before the event that touches the apple
    instead of upfront with everything else. For traces with no events
    (static description / tests), groundings emit upfront unchanged.
    """
    rel_changes_by_event, source_by_event = _attribute_relation_changes(
        trace, setup_relations, lexicon)

    # Salience filter — see `_referenced_entity_ids` for details.
    referenced = _referenced_entity_ids(
        trace, scene_location_id, setup_relations, source_by_event)

    # Per-event preconditions for the ĉar-clause attachment.
    event_preconds = _event_preconditions(trace, lexicon)
    inlined_entities = {eid for (eid, _, _) in event_preconds.values()}

    # Compute first-event index per entity. Drives the lazy-grounding
    # anchor: a grounding/setup-relation message attaches to the
    # earliest event where its entity participates.
    first_event_idx = _first_event_index(trace, source_by_event)

    relations_for_setup = (setup_relations if setup_relations is not None
                           else list(trace.relations))

    # No-events fallback: emit groundings upfront in the original
    # order. Static traces (Sara estis en kuirejo, no events) stay
    # readable without any anchoring machinery.
    if not trace.events:
        messages: list[Message] = []
        for eid in _synthetic_grounding_targets(
                trace, scene_location_id, setup_relations):
            if eid in referenced:
                messages.append(SceneGroundingMessage(entity_id=eid))
        for msg in _quality_grounding_messages(
                trace, lexicon, inlined_entities):
            if getattr(msg, "entity_id", None) in referenced:
                messages.append(msg)
        for rel in relations_for_setup:
            if all(arg in referenced for arg in rel.args):
                messages.append(RelationMessage(relation=rel))
        return messages

    # Bucket grounding messages by their anchor event index. Entities
    # never referenced (anchor < 0) get dropped — already filtered by
    # salience above; the `< 0` guard is belt-and-suspenders.
    pre_event: dict[int, list[Message]] = {
        i: [] for i in range(len(trace.events))}

    # Show-not-tell motivation: each event's active precondition
    # surfaces as an `EntityQualityMessage` rather than a `ĉar` clause.
    # Anchor at the precondition entity's FIRST event (not the
    # specific event that needs it) — the state is established once,
    # upfront, and the action chain that follows uses it implicitly.
    # Anchoring at the precondition event itself would mid-sentence
    # interrupt action coordination ("Maria kuiris la panon. Maria
    # estis malsata. Maria manĝis la panon" vs the cleaner "Maria
    # estis malsata. Maria kuiris kaj manĝis la panon").
    # `inlined_entities` then filters the generic quality-grounding
    # pass so we don't double-emit.
    seen_precond: set[tuple[str, str]] = set()
    for ev_id, (entity_id, slot, quality_lemma) in event_preconds.items():
        if entity_id not in referenced:
            continue
        # Skip preconditions on entities that don't yet exist at scene
        # init (construct stubs, mid-trace creations). Surfacing "the
        # chair was clean" before the build sentence reads as the chair
        # pre-existing; the construct verb's adjective rendering carries
        # the initial state instead.
        ent = trace.entities.get(entity_id)
        if ent is not None and ent.created_at_event is not None:
            continue
        key = (entity_id, quality_lemma)
        if key in seen_precond:
            continue
        seen_precond.add(key)
        anchor = first_event_idx.get(entity_id, 0)
        pre_event[anchor].append(EntityQualityMessage(
            entity_id=entity_id, slot=slot,
            quality_lemma=quality_lemma))

    for eid in _synthetic_grounding_targets(
            trace, scene_location_id, setup_relations):
        if eid not in referenced:
            continue
        anchor = first_event_idx.get(eid, 0)
        pre_event[anchor].append(SceneGroundingMessage(entity_id=eid))

    for msg in _quality_grounding_messages(
            trace, lexicon, inlined_entities):
        eid = getattr(msg, "entity_id", None)
        if eid not in referenced:
            continue
        anchor = first_event_idx.get(eid, 0)
        pre_event[anchor].append(msg)

    for rel in relations_for_setup:
        if not all(arg in referenced for arg in rel.args):
            continue
        anchor = _relation_anchor(rel, first_event_idx, trace)
        pre_event[anchor].append(RelationMessage(relation=rel))

    messages = []
    for idx, ev in enumerate(trace.events):
        for m in pre_event[idx]:
            messages.append(m)
        # Precondition is now surfaced as a pre-event quality message
        # (above); don't also pass it to the EventMessage so the
        # renderer doesn't emit a ĉar clause.
        messages.append(EventMessage(
            event=ev,
            cause_event_id=(ev.caused_by[0] if ev.caused_by else None),
            source_entity_id=source_by_event.get(ev.id)))
        for created in ev.creates:
            messages.append(AppearanceMessage(
                entity_id=created.id, cause_event_id=ev.id))
        for msg in rel_changes_by_event.get(ev.id, []):
            messages.append(msg)
        for dmsg in _destruction_messages_for_event(trace, ev):
            messages.append(dmsg)
    return messages


def _relation_anchor(
    rel: RelationAssertion, first_event_idx: dict[str, int],
    trace: Trace,
) -> int:
    """Pick the event index a setup relation should attach to.

    Special case: `en(thing, location)` defers to the thing's first
    event. The reader meets the thing on arrival ("Anna venis al la
    valo. Estis ĉapelo en valo. Anna prenis la ĉapelon"). Anchoring
    at the location's first event would render "Estis ĉapelo en valo"
    upfront, before the agent has reason to care.

    `en(contents, container)` for a non-location container falls back
    to min — "akvo en glaso" matters from the start because the glass
    is something that breaks (transformative event), not just an
    arrival destination.

    Other relations (havi, apud, sur, konas, havas_parton, …) anchor
    at min over participants — both sides are narratively relevant
    whenever either first appears."""
    if rel.relation == "en" and len(rel.args) == 2:
        contained, container = rel.args
        container_ent = trace.entities.get(container)
        if (container_ent is not None
                and container_ent.entity_type == "location"
                and contained in first_event_idx):
            return first_event_idx[contained]
    anchors = [first_event_idx[a] for a in rel.args
               if a in first_event_idx]
    return min(anchors) if anchors else 0


def _first_event_index(
    trace: Trace, source_by_event: dict[str, str],
) -> dict[str, int]:
    """For each entity that appears in the trace's events (as a role
    value, property_change target, or source-attributed prior owner),
    record the index of the earliest event that references it.
    Entities only present in scene-setup relations get the index of
    the earliest event that mentions any of their relation neighbors —
    handled implicitly because the relation message itself anchors to
    the min over its participants."""
    out: dict[str, int] = {}
    for idx, ev in enumerate(trace.events):
        for v in ev.roles.values():
            if isinstance(v, str) and v not in out:
                out[v] = idx
        for (eid, _slot), _val in ev.property_changes.items():
            if isinstance(eid, str) and eid not in out:
                out[eid] = idx
    # Source-attributed prior owners anchor at the event that took
    # from them — keeps "En la dormejo havis Maria la pomon" right
    # before the take event.
    for ev_id, src_eid in source_by_event.items():
        if not isinstance(src_eid, str) or src_eid in out:
            continue
        for idx, ev in enumerate(trace.events):
            if ev.id == ev_id:
                out[src_eid] = idx
                break
    return out


# -------------------- helpers: salience filter ----------------------

def _referenced_entity_ids(
    trace: Trace,
    scene_location_id: Optional[str],
    setup_relations: Optional[list[RelationAssertion]],
    source_by_event: dict[str, str],
) -> set[str]:
    """Entity ids the action chain references, plus their 1-hop
    relation neighbors and the scene location. Used to filter
    background grounding so scenes only describe entities that
    matter to the unfolding action.

    Sources of "referenced":
      - event roles (agent, theme, recipient, instrument, location, ...)
      - event property_changes targets
      - source attributions for acquisition verbs (prior owner)
      - the scene location itself (always grounded)

    1-hop expansion through scene-setup relations preserves narrative
    context: if Maria is the prior owner of an apple Petro takes,
    `havi(Maria, apple)` keeps Maria visible even though she has no
    role in any event."""
    # No events means the trace is pure scene description (a static
    # snapshot for tests / appendix prose) — nothing to filter against,
    # so keep everything visible.
    if not trace.events:
        return set(trace.entities.keys())
    referenced: set[str] = set()
    if scene_location_id is not None:
        referenced.add(scene_location_id)
    for ev in trace.events:
        for v in ev.roles.values():
            if isinstance(v, str):
                referenced.add(v)
        for (eid, _slot), _val in ev.property_changes.items():
            if isinstance(eid, str):
                referenced.add(eid)
    for src in source_by_event.values():
        if isinstance(src, str):
            referenced.add(src)
    # Expansion through setup relations.
    # For most relations expansion is symmetric and SINGLE-PASS: if
    # either arg is in `seeds`, add both. This keeps prior owners
    # (havi), parts (havas_parton), seating containers (sur),
    # neighbors (apud) visible when any side is narratively referenced.
    #
    # `en` is asymmetric AND iterated to fixed point: a referenced
    # entity's container matters (forno → kuirejo → domo), but a
    # referenced location doesn't make every other thing in it
    # relevant (a location collects unrelated bystanders). The
    # contained→container direction is iterated so multi-level
    # containment hierarchies (forno en kuirejo en domo) all stay
    # in scope — one pass would only catch the immediate container.
    rels = (setup_relations if setup_relations is not None
            else list(trace.relations))
    seeds = frozenset(referenced)
    for r in rels:
        args = tuple(r.args)
        if r.relation == "en":
            continue
        if any(a in seeds for a in args):
            referenced.update(args)
    en_rels = [r for r in rels if r.relation == "en" and len(r.args) == 2]
    while True:
        added = False
        for r in en_rels:
            contained, container = r.args
            if contained in referenced and container not in referenced:
                referenced.add(container)
                added = True
        if not added:
            break
    return referenced


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
) -> dict[str, tuple[str, str, str]]:
    """For each event, find one ACTIVE precondition: an (entity_id,
    slot, quality_lemma) triple where some verb-role or rule-pattern
    conditions on a slot AND the entity actually has the constraint's
    value.

    Returns {event_id: (entity_id, slot, quality_lemma)} for events
    that have such a precondition. The slot is what the renderer
    needs to look up `lexicon.state_verbs[(slot, value)]` and decide
    whether the predicate can be contracted to a verbal form. Used to:
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

    out: dict[str, tuple[str, str, str]] = {}
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
                out[ev.id] = (eid, slot, expected)
                break
            if actual == expected:
                out[ev.id] = (eid, slot, expected)
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
            entity_id=eid, slot=slot_name, quality_lemma=quality))
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
    lexicon: Optional[Lexicon] = None,
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
    # Transfer / acquisition verbs are derived from rule structure
    # (see dsl/introspect.py): any verb whose rule fires `transfer_n`
    # is a TRANSFER verb; the subset where theme moves to the agent
    # are the ACQUISITION verbs (those get prior-owner attribution).
    # ĵeti is hand-added because it's the one transfer verb whose
    # rule uses remove_relation directly rather than transfer_n.
    from ..dsl.introspect import (
        acquisition_verbs as _acq_verbs,
        transfer_verbs as _trans_verbs,
        verb_relation_kinds as _verb_rel_kinds,
    )
    from ..dsl.rules import DEFAULT_DSL_RULES
    _rules = list(DEFAULT_DSL_RULES)
    TRANSFER_VERBS = set(_trans_verbs(_rules)) | {"ĵeti"}
    ACQUISITION_VERBS = set(_acq_verbs(_rules))
    MOVEMENT_VERBS = {"iri", "veturi"}
    PLACEMENT_VERBS = {"meti"}
    candidate_actions = TRANSFER_VERBS | MOVEMENT_VERBS | PLACEMENT_VERBS
    # Per-verb relation-kind gate. Without it, iri(agent, destination)
    # would claim a same-event havi(agent, item) add (because "agent"
    # is in iri's role values) and narrate it as a side effect of
    # going somewhere. The havi belongs to the adjacent preni event.
    # Movement / placement verbs use add_relation/remove_relation
    # directly so they fall out of introspection — re-add them here.
    _MANUAL_RELATION_KINDS: dict[str, frozenset[str]] = {
        "iri": frozenset({"en"}),
        "veturi": frozenset({"en"}),
        "ĵeti": frozenset({"havi"}),
        "meti": frozenset({"en"}),
    }
    VERB_RELATION_KINDS: dict[str, frozenset[str]] = {
        **_verb_rel_kinds(_rules),
        **_MANUAL_RELATION_KINDS,
    }

    out: dict[str, list[Message]] = {}
    source_for_event: dict[str, str] = {}
    unattributed_adds = set(added)
    unattributed_removes = set(removed)

    for ev in trace.events:
        if ev.action not in candidate_actions:
            continue
        ev_referents = {v for v in ev.roles.values()
                        if isinstance(v, str)}
        relation_kinds = VERB_RELATION_KINDS.get(ev.action, frozenset())
        claims_rem: list = []
        claims_add: list = []
        for (rel, args) in list(unattributed_removes):
            if rel not in relation_kinds:
                continue
            if set(args) & ev_referents:
                claims_rem.append((rel, args))
        for (rel, args) in list(unattributed_adds):
            if rel not in relation_kinds:
                continue
            if set(args) & ev_referents:
                claims_add.append((rel, args))

        narrate_ownership = ev.action not in TRANSFER_VERBS
        # Acquisition verbs: if we consumed a havi-removal whose
        # owner differs from the event's agent, that owner is the
        # "source" to narrate via "de <source>" on the event itself.
        # Fallback to the rule's precondition `havi(<role>, theme)`
        # when no havi-removal surfaced — catches partial transfers
        # (source keeps a smaller stack, so its havi isn't removed)
        # and lets aĉeti read as "aĉetis ... de SELLER" even when
        # only some of SELLER's goods moved.
        if ev.action in ACQUISITION_VERBS:
            agent = ev.roles.get("agent")
            theme = ev.roles.get("theme")
            found_source = False
            for (rel, args) in claims_rem:
                if (rel == "havi" and len(args) == 2
                        and args[1] == theme and args[0] != agent):
                    source_for_event[ev.id] = args[0]
                    found_source = True
                    break
            if not found_source and lexicon is not None:
                action_def = lexicon.actions.get(ev.action)
                if action_def is not None:
                    for pc in action_def.preconditions:
                        if (pc.kind == "relation" and pc.rel == "havi"
                                and len(pc.roles) == 2
                                and pc.roles[1] == "theme"
                                and pc.roles[0] != "agent"):
                            src_id = ev.roles.get(pc.roles[0])
                            if src_id is not None:
                                source_for_event[ev.id] = src_id
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
        # Events with a kind="created" role (fari and any future
        # construction verb) stay standalone — their role structure
        # (created theme, list parts, optional instrument) is too
        # rich to elide into a coordinated "faris X kaj Yis Z" phrase.
        if _has_created_role(m.event, lexicon):
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
            if _has_created_role(nxt.event, lexicon):
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


def _has_created_role(ev: Event, lexicon: Lexicon) -> bool:
    """True if this event's action declares any role of kind="created"
    (fari today, future construction verbs). Used by aggregation to
    keep construction events as standalone sentences — their role
    structure is too rich (created theme, list parts, optional tool)
    to elide gracefully into a `X faris ... kaj Yis ...` coordination."""
    action = lexicon.actions.get(ev.action)
    if action is None:
        return False
    return any(
        getattr(r, "kind", "single") == "created" for r in action.roles)


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
    # Find the leading run of setup-phase messages — the prefix before
    # the first EventMessage. Scene grounding ("X estis en SCENE")
    # semantically equals an `en(X, scene_location_id)` relation, so
    # we treat both kinds as bucketable. EntityQualityMessage and
    # other interstitials are kept in place via `leftover`.
    cut = 0
    while cut < len(messages) and not isinstance(
            messages[cut], EventMessage):
        cut += 1
    if cut == 0:
        return messages

    setup = messages[:cut]
    rest = messages[cut:]

    # Find the implicit scene-location id from any SceneGroundingMessage.
    scene_location_id: Optional[str] = None
    for m in setup:
        if isinstance(m, SceneGroundingMessage):
            for r in messages:
                # Walk other messages to find the scene id — it's
                # stored implicitly via the trace in render time, but
                # for grouping we need it explicit. SceneGrounding's
                # entity is `en` the scene by definition; we infer the
                # scene from any en-relation whose container appears
                # repeatedly, OR fall back to ungrouped if ambiguous.
                pass
            break

    # Bucket by (relation, container). Track first-appearance order
    # so the output preserves the rough scene-introduction sequence.
    buckets: dict[tuple[str, str], list[str]] = {}
    bucket_order: list[tuple[str, str]] = []
    leftover: list[Message] = []

    # First pass: gather candidate scene container from RelationMessages.
    # Most-frequent en-container wins as the inferred scene id for
    # SceneGroundingMessage attachment.
    container_counts: dict[str, int] = {}
    for m in setup:
        if isinstance(m, RelationMessage):
            r = m.relation
            if r.relation == "en" and len(r.args) == 2:
                container_counts[r.args[1]] = (
                    container_counts.get(r.args[1], 0) + 1)
    inferred_scene = (
        max(container_counts, key=container_counts.get)
        if container_counts else None)

    for m in setup:
        if isinstance(m, RelationMessage):
            rel = m.relation
            if rel.relation in ("en", "sur") and len(rel.args) == 2:
                key = (rel.relation, rel.args[1])
                if key not in buckets:
                    buckets[key] = []
                    bucket_order.append(key)
                buckets[key].append(rel.args[0])
            elif rel.relation == "havi" and len(rel.args) == 2:
                # Same shape as en/sur but flipped: havi(owner, theme)
                # — bucket by owner so "Kantisto havis pano, viando
                # kaj fromaĝo" collapses three sequential lines.
                key = ("havi", rel.args[0])
                if key not in buckets:
                    buckets[key] = []
                    bucket_order.append(key)
                buckets[key].append(rel.args[1])
            else:
                leftover.append(m)
        elif isinstance(m, SceneGroundingMessage) and inferred_scene:
            key = ("en", inferred_scene)
            if key not in buckets:
                buckets[key] = []
                bucket_order.append(key)
            buckets[key].append(m.entity_id)
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
            # in. For havi the arg order is (owner, theme); en/sur is
            # (contained, container) — swap accordingly.
            from ..causal import RelationAssertion
            args = ((container_id, contained_ids[0])
                    if rel_name == "havi"
                    else (contained_ids[0], container_id))
            out.append(RelationMessage(
                relation=RelationAssertion(
                    relation=rel_name, args=args)))
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
