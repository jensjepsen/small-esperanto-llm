"""Causal trace + event-calculus engine.

Trace primitives:
  EntityInstance      — a concept materialized in the world. Has
                        `properties` (state at the moment it entered the
                        trace — either scene-init or mid-trace creation)
                        and a lifecycle marker `created_at_event`.
  RelationAssertion   — an instance of a Relation between entities.
  Event               — an action firing with role bindings. Carries
                        `property_changes` (state transitions it makes),
                        `creates` (entities it brings into existence),
                        `caused_by` (causal DAG), and `trace_position`.

Engine:
  Rule signature: `rule(trace, t) -> list[Event]`. Rules inspect state
  via `trace.entities_at(t)` and `trace.property_at(eid, prop, t)`. The
  engine iterates each rule at every position, appends produced events,
  and registers created entities. Memoization is per event id; re-firing
  the same event content is prevented automatically, while state-driven
  re-firing (e.g. person falls twice with different causes) works
  because distinct `caused_by` lists produce distinct ids.

Event calculus:
  `entity.properties` is the state at the moment of entity creation and
  NEVER mutated thereafter. Current-at-time-t state is read by walking
  events backward from position t-1, finding the most recent
  `property_changes[(eid, prop)]`, falling back to `entity.properties`.
  See `Trace.property_at`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .loader import Lexicon


# ----------------------------- entities -----------------------------

@dataclass
class EntityInstance:
    """A concept materialized in the trace.

    `properties` holds the entity's state at the moment it entered the
    trace — scene-init for entities with `created_at_event is None`,
    or the state at the position they were created for mid-trace
    entities. The dict is conceptually immutable after construction;
    any subsequent state transitions live in events' `property_changes`
    and are reconstructed by `Trace.property_at`.

    The `set_property` helper writes to `properties` — useful at scene
    setup (before the engine runs) to tweak scene-initial state. It
    should not be called after the engine starts running; current-state
    mutations go through events.
    """
    id: str
    concept_lemma: str
    entity_type: str
    properties: dict[str, list[str]] = field(default_factory=dict)
    created_at_event: Optional[int] = None
    destroyed_at_event: Optional[int] = None

    def set_property(self, slot: str, value: str, scalar: bool = True) -> None:
        """Set scene-initial state. Do not call after engine starts."""
        if scalar:
            self.properties[slot] = [value]
        else:
            existing = self.properties.setdefault(slot, [])
            if value not in existing:
                existing.append(value)

    def get_property(self, slot: str) -> list[str]:
        """Read scene-initial state (shortcut for `properties.get(...)`).
        For current-at-time-t state, use `trace.property_at(id, slot, t)`.
        """
        return list(self.properties.get(slot, []))


@dataclass
class RelationAssertion:
    relation: str
    args: tuple[str, ...]               # entity ids


# ------------------------------ events ------------------------------

@dataclass(frozen=True)
class Event:
    """An event in the trace.

    Event-calculus fields (Step 1 of migration):
      creates: entities brought into existence by this event. Empty for
        events the old engine produces; populated by new-engine rules
        that synthesize new entities (e.g. broken_fragile_creates_shards).
      property_changes: {(entity_id, prop_name): value} state changes
        attributed to this event. Empty for old-engine events; populated
        by new-engine rules.
      trace_position: position at which this event fired. Set by the new
        engine's run_to_fixed_point; None for old-engine events.

    Note: frozen=True prevents reassignment of fields, but the mutable
    defaults (creates, property_changes) can still be appended/updated
    by code that holds an Event reference. That's intentional during the
    migration; once the old engine is gone we'll lock these down further.
    """
    id: str
    action: str
    roles: dict[str, str]               # role_name -> entity_id
    caused_by: tuple[str, ...] = ()
    creates: list["EntityInstance"] = field(default_factory=list)
    property_changes: dict[tuple[str, str], Any] = field(default_factory=dict)
    trace_position: Optional[int] = None


def effect_changes(
    action_lemma: str, roles: dict[str, str], lexicon: "Lexicon",
) -> dict[tuple[str, str], Any]:
    """Compute the `property_changes` dict that an event's verb effect
    spec would produce, given the role bindings. Replaces the old
    engine's `_apply_effects` mechanism — now the changes are baked into
    events rather than mutated onto entities at apply time.

    Callers: sampler seed functions (so seed events carry their verb's
    intrinsic state transitions) and any rule whose synthesized event
    should inherit the target verb's effects (e.g. use_instrument
    computing the signature verb's property_changes).
    """
    action = lexicon.actions.get(action_lemma)
    if action is None:
        return {}
    out: dict[tuple[str, str], Any] = {}
    for eff in action.effects:
        tid = roles.get(eff.target_role)
        if tid is not None:
            out[(tid, eff.property)] = eff.value
    return out


def make_event(
    action: str, roles: dict[str, str], caused_by: Iterable[str] = (),
    *,
    creates: Optional[Iterable["EntityInstance"]] = None,
    property_changes: Optional[dict[tuple[str, str], Any]] = None,
    trace_position: Optional[int] = None,
) -> Event:
    """Content-addressed event id so rules are idempotent.

    Note: id is derived from action/roles/causes only. The new fields
    (creates, property_changes, trace_position) intentionally do NOT
    contribute to the id, because they're side effects of *what fired*,
    not part of the event's identity. Two rules producing 'fragile_falls
    on glaso at position 3' should be the same event regardless of
    whether one of them attaches a property_change and the other doesn't.
    The new engine's memoization (rule_name, arg_key, position) replaces
    content hashing for re-firing prevention anyway.
    """
    causes = tuple(sorted(caused_by))
    role_str = ",".join(f"{k}={v}" for k, v in sorted(roles.items()))
    h = hashlib.sha1(
        f"{action}|{role_str}|{','.join(causes)}".encode("utf-8")
    ).hexdigest()[:12]
    return Event(
        id=h, action=action, roles=dict(roles), caused_by=causes,
        creates=list(creates or []),
        property_changes=dict(property_changes or {}),
        trace_position=trace_position,
    )


# ------------------------------ trace ------------------------------

@dataclass
class Trace:
    entities: dict[str, EntityInstance] = field(default_factory=dict)
    relations: list[RelationAssertion] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    _event_ids: set[str] = field(default_factory=set)
    _next_entity_id: int = 1

    def snapshot_relations(self) -> list[RelationAssertion]:
        """Shallow copy of the current relations. Useful before running
        the engine when the caller plans to pass the initial-state
        relations to `realize_trace(..., setup_relations=...)` — rules
        that modify relations (e.g. `preni`) otherwise leave
        `trace.relations` in the post-run state."""
        return list(self.relations)

    # ---------- entity helpers ----------
    def add_entity(
        self, concept_lemma: str, lexicon: Lexicon,
        entity_id: str | None = None,
    ) -> EntityInstance:
        concept = lexicon.concept(concept_lemma)
        eid = entity_id or f"e{self._next_entity_id}"
        if eid in self.entities:
            raise ValueError(f"entity id {eid!r} already in trace")
        self._next_entity_id += 1
        ent = EntityInstance(
            id=eid, concept_lemma=concept_lemma,
            entity_type=concept.entity_type,
            properties={k: list(v) for k, v in concept.properties.items()},
        )
        self.entities[eid] = ent
        return ent

    def entity(self, eid: str | None) -> EntityInstance | None:
        if eid is None:
            return None
        return self.entities.get(eid)

    # ---------- relation helpers ----------
    def assert_relation(
        self, name: str, args: tuple[str, ...], lexicon: Lexicon,
    ) -> RelationAssertion:
        rel = lexicon.relations.get(name)
        if rel is None:
            raise KeyError(f"unknown relation {name!r}")
        if len(args) != rel.arity:
            raise ValueError(
                f"relation {name!r}: expected {rel.arity} args, got {len(args)}")
        for arg, expected_type in zip(args, rel.arg_types):
            ent = self.entities.get(arg)
            if ent is None:
                raise KeyError(f"unknown entity {arg!r}")
            if not lexicon.types.is_subtype(ent.entity_type, expected_type):
                raise ValueError(
                    f"relation {name!r}: entity {arg!r} type "
                    f"{ent.entity_type!r} not a {expected_type!r}")
        ra = RelationAssertion(relation=name, args=args)
        self.relations.append(ra)
        return ra

    # ---------- event helpers ----------
    def add_event(self, event: Event) -> bool:
        """Returns True if the event was newly added, False if it was a
        duplicate (same content hash)."""
        if event.id in self._event_ids:
            return False
        self.events.append(event)
        self._event_ids.add(event.id)
        return True

    # ---------- event-calculus queries (Step 2) ----------
    #
    # Position semantics: t is the number of events that have fired. t=0
    # is scene-initial; t=len(events) is the post-final state. An entity
    # with `created_at_event = k` is the result of events[k] firing, so
    # it's visible from position k+1 onward (i.e. at any t > k).

    def entities_at(self, t: int) -> list[EntityInstance]:
        """Entities present at position t.

        Includes scene-initial entities (`created_at_event is None`) and
        entities created by events at indices < t. Excludes entities
        whose `destroyed_at_event` index is < t.
        """
        out: list[EntityInstance] = []
        for ent in self.entities.values():
            if ent.created_at_event is not None and ent.created_at_event >= t:
                continue
            if ent.destroyed_at_event is not None and ent.destroyed_at_event < t:
                continue
            out.append(ent)
        return out

    def property_at(self, entity_id: str, prop: str, t: int) -> Any:
        """Value of `prop` on entity `entity_id` at position t.

        Walks events[0..t-1] backward; returns the value from the most
        recent `property_changes[(entity_id, prop)]` entry. If no event
        in that range changed the property, falls back to the entity's
        scene-init `properties[prop]`.

        Liveness semantics: if the entity wasn't yet created at t
        (`created_at_event >= t`), returns None — it didn't exist then.
        `destroyed_at_event` is intentionally NOT checked: history
        outlives the thing. Asking "what was the mouse's state after
        the cat ate it?" returns the last recorded values (e.g.
        presence=consumed); use `entities_at(t)` for liveness queries.

        Naive O(t) per call. Acceptable for traces under ~20 events;
        cache as a sorted (position, value) list per (entity_id, prop)
        if it becomes a bottleneck.
        """
        ent = self.entities.get(entity_id)
        if ent is None:
            return None
        if ent.created_at_event is not None and ent.created_at_event >= t:
            return None

        key = (entity_id, prop)
        # Walk backward from events[t-1] down to events[0].
        last_idx = min(t, len(self.events)) - 1
        for i in range(last_idx, -1, -1):
            ev = self.events[i]
            if key in ev.property_changes:
                return ev.property_changes[key]

        return ent.properties.get(prop)


# The imperative `run_to_fixed_point` engine and its `Rule =
# Callable[[Trace, int], list[Event]]` protocol used to live here.
# Retired in Phase 5 of the ontology migration; replaced by
# `esperanto_lm.ontology.dsl.run_dsl` which runs declarative DSL rules
# plus a derivation layer. See `ontology/rules.py` for the redirect
# note and `dsl/CLAUDE.md` for the new model.
