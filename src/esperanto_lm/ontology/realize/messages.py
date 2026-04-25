"""Message IR — one sentence per Message (after transforms).

The realizer is a pipeline: trace → plan (build messages) → transform
(aggregate) → render (surface prose). Messages are the handoff
between stages. Each kind corresponds to a distinct sentence shape;
the renderer dispatches on type.

This module defines message shapes only — no logic. The planner
populates them; the renderer reads them.

Types in this slice (v1):

  SceneGroundingMessage  — "X estis en la SCENE." synthetic placement.
  RelationMessage        — "X estis en Y." / "X havis Y." (scene-setup).
  EventMessage           — "X faris Y." one event's sentence.
  AppearanceMessage      — "Aperis X." created-entity introduction.
  RelationRemovedMessage — "X ne plu havis Y." ownership/location loss.
  RelationAddedMessage   — "X ricevis Y." ownership acquisition.
  DestructionMessage     — "X malaperis." entity-lifecycle end.
  CoordinatedMessage     — children joined with "kaj", shared subject.

Shapes reserved for later phases (sentence planner, document planner):
  SubordinatedMessage    — "Kiam X faris A, Y faris B."
  ParticipleMessage      — "Falinta X rompis Y."
  RelativeClauseMessage  — "X, kiu falis, rompiĝis."

Those are additions, not replacements — the renderer dispatch table
extends, the existing types stay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..causal import Event, RelationAssertion


@dataclass(kw_only=True)
class Message:
    """Base class. `cause_event_id` hooks into connective selection —
    if the preceding rendered sentence's event is a cause, the next
    sentence gets a `Tial` / `Sekve` / `Pro tio` prefix."""
    cause_event_id: Optional[str] = None


@dataclass(kw_only=True)
class SceneGroundingMessage(Message):
    """Synthetic 'X estis en la SCENE.' for entities that appear in
    events but have no explicit containment relation. Skipped for
    entities that came into existence mid-trace (those get an
    AppearanceMessage instead)."""
    entity_id: str


@dataclass(kw_only=True)
class RelationMessage(Message):
    """Scene-setup relation. `relation` is a live RelationAssertion
    from the initial world — either `trace.relations` or the snapshot
    the caller passed via `setup_relations`."""
    relation: RelationAssertion


@dataclass(kw_only=True)
class EventMessage(Message):
    """An event's sentence. The renderer picks subject/object shape
    from the action's role structure — agent-first for transitives,
    theme-first for intransitives, location-fronted for impersonals.

    `source_entity_id` (optional): for acquisition verbs (preni,
    kapti) whose rule consumed a havi-removal from some entity M
    other than the event's agent. Rendered as "de <M>" after the
    theme — "Klara prenas la libron de Maria" — so the previous
    owner is narrated even though we suppress the separate
    "Maria ne plu havas" line.
    """
    event: Event
    source_entity_id: Optional[str] = None


@dataclass(kw_only=True)
class AppearanceMessage(Message):
    """'Aperis X.' for an entity created mid-trace. Attached to the
    event whose `creates` list included the entity, so the appearance
    line lands right after its causing event."""
    entity_id: str


@dataclass(kw_only=True)
class RelationRemovedMessage(Message):
    """'Maria ne plu havis la libron.' or 'X ne plu estis en Y.'
    Narrates a relation that held before an event and no longer does.
    Attached to the event that caused the removal (via cause_event_id)."""
    relation: str
    args: tuple[str, ...]


@dataclass(kw_only=True)
class RelationAddedMessage(Message):
    """'Petro ricevis la libron.' Optional counterpart to
    RelationRemovedMessage. Usually redundant when paired with the
    triggering event ('Petro prenis la libron' implies havi is added),
    so the planner narrates this selectively."""
    relation: str
    args: tuple[str, ...]


@dataclass(kw_only=True)
class DestructionMessage(Message):
    """'La muso malaperis.' Fires when an entity's destroyed_at_event
    is set — currently only by `manĝi_destroys_theme`. The rendering
    doesn't repeat the event that caused the destruction; that's
    already been emitted via EventMessage."""
    entity_id: str


@dataclass(kw_only=True)
class CoordinatedMessage(Message):
    """Children share a subject and collapse into one sentence joined
    with 'kaj'. Produced by the aggregation transform; renderer takes
    the shared subject once and renders just the verb phrases for
    children after the first.

    `children` may contain any Message type the renderer knows how to
    emit a verb phrase for (currently EventMessage and the state-change
    kinds). Mixing coordination with subordination — nested
    CoordinatedMessage / future SubordinatedMessage — is deferred to
    the full sentence planner.
    """
    children: list[Message] = field(default_factory=list)


@dataclass(kw_only=True)
class GroupedRelationMessage(Message):
    """Multiple entities in the same containment relation with the
    same container, rendered as a single list sentence:

      "En la kuirejo estas tablo, glaso, kaj korbo."
      "Sur la breto estas libro kaj papero."

    Produced by `aggregate_relations` from a run of RelationMessages
    that share `(relation, container_id)`. Only `en` and `sur` —
    `havi` and `apud` keep one-per-line for now (different rhetorical
    weight; havi is rarely chained, apud is usually emphatic).
    """
    relation: str             # "en" or "sur"
    container_id: str
    contained_ids: list[str]


@dataclass(kw_only=True)
class SubordinatedMessage(Message):
    """`main, conjunction subordinate.` — one sentence with the
    subordinate clause tacked on after a comma.

    Produced by the subordination transform. Currently only generated
    with `conjunction="el kio"` (Esperanto's result subordinator —
    "from which"), combining a cascade event with the appearance it
    triggered: "La glaso rompiĝis, el kio aperis vitropecetoj."

    Future conjunctions the same render path handles: "post kiam"
    (temporal), "ĉar" (causal), "tiel ke" (result-degree). Each is a
    separate transform that produces a SubordinatedMessage — the
    renderer doesn't care which transform built it.
    """
    main: Message
    subordinate: Message
    conjunction: str
