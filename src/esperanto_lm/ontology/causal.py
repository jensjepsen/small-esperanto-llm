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

_MISS = object()  # property_at cache sentinel — None is a valid stored value

# Containment-index cache for `assert_relation`'s en/sur validation.
# Keyed by id(lex). Resolves once per lexicon — the index is the same
# data the sampler uses (see `containment.resolve_containment`), so
# this just deduplicates work between scene placement and assertion
# validation.
_CONTAINMENT_IDX_CACHE: dict[int, dict] = {}
# Relations that have at least one rule in containment.jsonl. Computed
# from the lexicon's facts so validate_relation knows which relations
# to consult containment rules for — no hardcoded "en"/"sur" names.
# A new containment-validated relation (e.g. a future `pendi`) joins
# this set automatically by virtue of having rules added.
_CONTAINMENT_RELS_CACHE: dict[int, frozenset[str]] = {}


def _containment_index_for(lex):
    key = id(lex)
    cached = _CONTAINMENT_IDX_CACHE.get(key)
    if cached is None:
        from .containment import resolve_containment
        cached = resolve_containment(lex)
        _CONTAINMENT_IDX_CACHE[key] = cached
    return cached


def _containment_validated_relations(lex) -> frozenset[str]:
    """Set of relation names that have rules in containment.jsonl.
    `validate_relation` consults containment rules iff the relation is
    in this set — discovered from the data, not hardcoded."""
    key = id(lex)
    cached = _CONTAINMENT_RELS_CACHE.get(key)
    if cached is None:
        cached = frozenset(f.relation for f in lex.containment)
        _CONTAINMENT_RELS_CACHE[key] = cached
    return cached


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

    @classmethod
    def from_concept(
        cls, concept, eid: str, lexicon,
        *,
        created_at_event: Optional[int] = None,
        extra_props: Optional[dict[str, list[str]]] = None,
    ) -> "EntityInstance":
        """Canonical EntityInstance constructor. Copies the concept's
        on-disk properties, applies any `extra_props` overrides, and
        populates `slot.unmarked` defaults for in-scope, non-varies
        slots the concept didn't declare. Single place that knows
        about defaults — used by `Trace.add_entity` and the engine's
        mid-trace creation paths.

        The bake's proto-trace construction intentionally bypasses
        this (uses the raw `EntityInstance(...)` constructor): the
        bake itself is where derived properties like `is_body_part`
        get computed, so populating defaults there would block the
        derivation under asserted-wins semantics."""
        props = {k: list(v) for k, v in concept.properties.items()}
        if extra_props:
            for slot, vals in extra_props.items():
                props[slot] = list(vals)
        # Note: slot.unmarked is NOT auto-populated here. It's a
        # rendering hint ("if value equals this, skip the
        # adjective"), not a default to inject onto every
        # applicable entity. Auto-populating would make
        # `concept_models_slot` ambiguous (every entity would
        # carry every applicable slot) and phantom-match role-
        # property filters like ŝalti(theme.power_state=neaktiva)
        # against entities whose concept never declares power_state.
        # Non-derived slots take values from concept declarations
        # or engine actions; derived slots (is_part, ...) take
        # values from the runtime derivation engine.
        return cls(
            id=eid,
            concept_lemma=concept.lemma,
            entity_type=concept.entity_type,
            properties=props,
            created_at_event=created_at_event,
        )

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
        attributed to this event. Populated by rules whose `then`
        includes `emit(...).changing(...)`.
      trace_position: position at which this event fired in `trace.events`.
        Set by the engine when an Emit effect produces the event; None
        on hand-built seed events the caller appends directly.

    Note: frozen=True prevents reassignment of fields, but the mutable
    defaults (creates, property_changes) can still be appended/updated
    by code that holds an Event reference.
    """
    id: str
    action: str
    # Role values are entity ids, OR a list of entity ids for variadic
    # roles (kind="list" in the action schema — e.g. fari.parts holds
    # one eid per declared part of the constructed concept).
    roles: dict[str, str | list[str]]
    caused_by: tuple[str, ...] = ()
    creates: list["EntityInstance"] = field(default_factory=list)
    property_changes: dict[tuple[str, str], Any] = field(default_factory=dict)
    trace_position: Optional[int] = None
    # Number of units this event acts on. Default 1 (one bite, one
    # transfer, one unlock). Consumption verbs (manĝi/trinki) and
    # transfer verbs (doni/preni) read this to decrement/transfer
    # the right amount in one event. The realizer surfaces it in
    # prose: quantity > 1 → "Maria manĝis du pomojn".
    quantity: int = 1


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
    quantity: int = 1,
) -> Event:
    """Content-addressed event id so rules are idempotent.

    Note: id is derived from action/roles/causes/quantity. Quantity is
    part of the event's identity because two manĝi events with
    different quantities are semantically distinct (one bite vs. two
    bites) and should fire the consume cascade independently. Other
    side-effect fields (creates, property_changes, trace_position) are
    NOT in the id — they're outputs of what fired.
    """
    causes = tuple(sorted(caused_by))
    # List-valued roles (variadic parts on fari) hash by joining
    # elements with `+`; the brackets prevent ambiguity with scalar
    # ids that happen to contain a "+" character.
    def _role_to_str(v):
        if isinstance(v, (list, tuple)):
            return "[" + "+".join(str(x) for x in v) + "]"
        return str(v)
    role_str = ",".join(
        f"{k}={_role_to_str(v)}" for k, v in sorted(roles.items()))
    qty_str = "" if quantity == 1 else f"|q={quantity}"
    h = hashlib.sha1(
        f"{action}|{role_str}|{','.join(causes)}{qty_str}".encode("utf-8")
    ).hexdigest()[:12]
    return Event(
        id=h, action=action, roles=dict(roles), caused_by=causes,
        creates=list(creates or []),
        property_changes=dict(property_changes or {}),
        trace_position=trace_position,
        quantity=quantity,
    )


# ------------------------------ trace ------------------------------

@dataclass
class Trace:
    entities: dict[str, EntityInstance] = field(default_factory=dict)
    relations: list[RelationAssertion] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    _event_ids: set[str] = field(default_factory=set)
    _next_entity_id: int = 1
    # property_at memo, valid only between add_event calls. Engine
    # fixpoint passes hammer property_at on a stable events list — the
    # cache turns the inner O(t) walk into O(1) for repeat queries
    # within a single pass. Profiled at ~9% self-time pre-cache.
    _property_at_cache: dict = field(default_factory=dict)
    # Set of eids that appear as the `parto` (args[1]) of any
    # `havas_parton` assertion. Maintained inline by `assert_relation`
    # so the relation schema's `arg_not_part` check is O(1) per
    # assertion instead of an O(relations) scan. Forks copy by value.
    _parts_index: set[str] = field(default_factory=set)
    # Current-state cache for `property_at` at `t == len(events)`.
    # Maps (entity_id, prop) → last-seen value, materialized lazily.
    # Empirically every property_at call queries current state; the
    # prior per-(eid,prop,t) cache wiped itself on every add_event,
    # negating the benefit when many events fire in a planning loop.
    # `_current_props_version` records `len(events)` at last sync;
    # `property_at` resyncs by walking the unincorporated tail before
    # serving a current-state query, so direct `events.append(...)`
    # paths (the engine has several) stay correct without going
    # through `add_event`. Forks copy both by value.
    _current_props: dict[tuple[str, str], Any] = field(default_factory=dict)
    _current_props_version: int = 0

    def snapshot_relations(self) -> list[RelationAssertion]:
        """Shallow copy of the current relations. Useful before running
        the engine when the caller plans to pass the initial-state
        relations to `realize_trace(..., setup_relations=...)` — rules
        that modify relations (e.g. `preni`) otherwise leave
        `trace.relations` in the post-run state."""
        return list(self.relations)

    def fork(self) -> "Trace":
        """Lightweight copy used by the planner's _simulate_from_scratch
        instead of copy.deepcopy. Safe because:
          - EntityInstance is conceptually immutable post-construction
            (per the docstring); rules never mutate entity.properties.
            New entities created by rules go into the new dict only.
          - RelationAssertion objects are immutable; the list itself
            gets appended/filtered, so we duplicate the list but share
            the objects.
          - Event objects are immutable; events list is append-only.
        Hot path: ~10× faster than deepcopy in practice because Python's
        deepcopy walks every nested object via reflection."""
        new = Trace.__new__(Trace)
        new.entities = dict(self.entities)
        new.relations = list(self.relations)
        new.events = list(self.events)
        new._event_ids = set(self._event_ids)
        new._next_entity_id = self._next_entity_id
        new._property_at_cache = {}
        new._parts_index = set(self._parts_index)
        new._current_props = dict(self._current_props)
        new._current_props_version = self._current_props_version
        return new

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
        ent = EntityInstance.from_concept(concept, eid, lexicon)
        self.entities[eid] = ent
        return ent

    def entity(self, eid: str | None) -> EntityInstance | None:
        if eid is None:
            return None
        return self.entities.get(eid)

    # ---------- relation helpers ----------
    def validate_relation(
        self, name: str, args: tuple[str, ...], lexicon: Lexicon,
    ) -> Optional[str]:
        """Run every check `assert_relation` would do; return None when
        the assertion is permissible, or a short reason string when it
        isn't. No mutation, no exceptions for predictable rejections.
        Raises only on programming errors (unknown relation/entity).

        The full check set is unconditional — containment.jsonl rules
        are always consulted. Single source of truth: the planner, the
        seeder, and assert_relation all reach the same verdict via
        this function (or its boolean wrapper `is_relation_permitted`).
        Anything that wants to bypass containment should not be using
        `en`/`sur` at all."""
        rel = lexicon.relations.get(name)
        if rel is None:
            raise KeyError(f"unknown relation {name!r}")
        if len(args) != rel.arity:
            return (f"relation {name!r}: expected {rel.arity} args, "
                    f"got {len(args)}")
        # arg_kinds: per-position "what does this arg refer to?". Empty
        # = all entity (legacy default). See schemas.Relation.arg_kinds.
        kinds = (list(rel.arg_kinds) if rel.arg_kinds
                 else ["entity"] * rel.arity)
        for i, (arg, expected_type, kind) in enumerate(
                zip(args, rel.arg_types, kinds)):
            if kind == "literal":
                # Opaque string; no validation.
                continue
            if kind == "slot":
                if arg not in lexicon.slots:
                    return (f"relation {name!r}: arg {i} ({arg!r}) "
                            f"declared kind=slot but not in lexicon.slots")
                continue
            # kind == "entity" (default)
            ent = self.entities.get(arg)
            if ent is None:
                raise KeyError(f"unknown entity {arg!r}")
            if not lexicon.types.is_subtype(ent.entity_type, expected_type):
                return (f"relation {name!r}: entity {arg!r} type "
                        f"{ent.entity_type!r} not a {expected_type!r}")
        if rel.arg_excludes:
            for i, arg in enumerate(args):
                if i >= len(rel.arg_excludes):
                    continue
                if kinds[i] != "entity":
                    continue
                forbidden_list = rel.arg_excludes[i]
                if not forbidden_list:
                    continue
                ent = self.entities[arg]
                for forbidden in forbidden_list:
                    if lexicon.types.is_subtype(
                            ent.entity_type, forbidden):
                        return (f"relation {name!r}: arg {i} ({arg!r} "
                                f"type {ent.entity_type!r}) is excluded "
                                f"subtype {forbidden!r}")
        if rel.arg_not_part:
            for i, arg in enumerate(args):
                if i >= len(rel.arg_not_part) or not rel.arg_not_part[i]:
                    continue
                if kinds[i] != "entity":
                    continue
                if arg in self._parts_index:
                    return (f"relation {name!r}: arg {i} ({arg!r}) is a "
                            f"part of another entity, can't appear here")
        # arg_patterns: per-arg Pattern (NotPattern, EntityPattern,
        # And/Or) evaluated against the entity. Lets schema-level
        # invariants like "havi.theme cannot be nemovebla=yes" live
        # next to the relation definition. Static evaluator — no
        # trace/derived state — so the same gate is used both at
        # assert time and (via introspect.relation_arg_excludes) at
        # planner grounding time.
        if rel.arg_patterns:
            from .dsl.patterns import entity_matches_static
            for i, arg in enumerate(args):
                if i >= len(rel.arg_patterns):
                    continue
                if kinds[i] != "entity":
                    continue
                pat = rel.arg_patterns[i]
                if pat is None:
                    continue
                ent = self.entities[arg]
                if not entity_matches_static(ent, pat, lexicon):
                    return (f"relation {name!r}: arg {i} ({arg!r} "
                            f"concept {ent.concept_lemma!r}) violates "
                            f"arg pattern")
        # arg_compare: numeric cross-arg comparisons (e.g. havi's
        # theme.maso <= owner.lift_capacity carry-capacity gate). Same
        # vacuous-on-missing-data semantics as the precondition kind.
        # Skip when any compared arg is non-entity — slot/literal kinds
        # have no entity to read properties from.
        if rel.arg_compare:
            from .dsl.patterns import numeric_args_compare
            for spec in rel.arg_compare:
                if (kinds[spec["left_arg"]] != "entity"
                        or kinds[spec["right_arg"]] != "entity"):
                    continue
                ent_tuple = tuple(
                    self.entities[a] if kinds[idx] == "entity" else None
                    for idx, a in enumerate(args))
                if not numeric_args_compare(ent_tuple, spec):
                    return (
                        f"relation {name!r}: arg_compare failed "
                        f"({ent_tuple[spec['left_arg']].id}.{spec['left_property']} "
                        f"{spec['op']} {ent_tuple[spec['right_arg']].id}."
                        f"{spec['right_property']})")
        # Containment registry: source of truth for what can plausibly
        # be in/on what. Two-tier check (no required violation AND
        # afforded by at least one entry). The set of relations this
        # applies to is data-driven — any relation with rules in
        # containment.jsonl is validated here, no hardcoded names.
        if name in _containment_validated_relations(lexicon):
            from .containment import (
                containment_relations_for, required_fact_violations,
            )
            contained_ent = self.entities.get(args[0])
            container_ent = self.entities.get(args[1])
            if (contained_ent is not None and container_ent is not None
                    and contained_ent.concept_lemma in lexicon.concepts
                    and container_ent.concept_lemma in lexicon.concepts):
                idx = _containment_index_for(lexicon)
                contained_lemma = contained_ent.concept_lemma
                container_lemma = container_ent.concept_lemma
                req_violations = required_fact_violations(
                    container_lemma, contained_lemma, name, idx, lexicon)
                allowed = containment_relations_for(
                    container_lemma, contained_lemma, idx, lexicon)
                if req_violations:
                    return (f"containment requirement violation: "
                            f"{contained_lemma} {name} {container_lemma} "
                            f"violates {len(req_violations)} required "
                            f"entr{'y' if len(req_violations)==1 else 'ies'} "
                            f"in containment.jsonl")
                if name not in allowed:
                    return (f"containment violation: "
                            f"{contained_lemma} {name} {container_lemma} "
                            f"not declared in containment.jsonl "
                            f"(allowed: {sorted(allowed) or 'none'})")
        return None

    def is_relation_permitted(
        self, name: str, args: tuple[str, ...], lexicon: Lexicon,
    ) -> bool:
        """Bool view of `validate_relation`. Single principled answer
        — see `validate_relation` for the unified semantics."""
        return self.validate_relation(name, args, lexicon) is None

    def assert_relation(
        self, name: str, args: tuple[str, ...], lexicon: Lexicon,
    ) -> RelationAssertion:
        # All validity checks raise on failure — single principled
        # answer with no mode-dependent dispatch. Callers that want to
        # detect-without-raising consult validate_relation or
        # is_relation_permitted first.
        reason = self.validate_relation(name, args, lexicon)
        if reason is not None:
            raise ValueError(reason)
        ra = RelationAssertion(relation=name, args=args)
        self.relations.append(ra)
        # Maintain the parts index for the arg_not_part check above.
        # Only havas_parton(host, parto) feeds it.
        if name == "havas_parton" and len(args) == 2:
            self._parts_index.add(args[1])
        return ra

    # ---------- event helpers ----------
    def add_event(self, event: Event) -> bool:
        """Returns True if the event was newly added, False if it was a
        duplicate (same content hash)."""
        if event.id in self._event_ids:
            return False
        self.events.append(event)
        self._event_ids.add(event.id)
        if self._property_at_cache:
            self._property_at_cache.clear()
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

        Current-state queries (t == len(events)) hit the incrementally-
        maintained `_current_props` map in O(1); the version sync
        below brings it up to date if anything appended events
        outside `add_event`. Historical queries fall through to the
        per-(eid, prop, t) cache + backward walk.
        """
        ent = self.entities.get(entity_id)
        if ent is None:
            return None
        if ent.created_at_event is not None and ent.created_at_event >= t:
            return None

        n_events = len(self.events)
        # Fast path: current-state (every observed call site queries
        # this position exclusively).
        if t == n_events:
            if self._current_props_version != n_events:
                # Direct events.append callers (engine, samplers) bypass
                # add_event; resync by folding any unincorporated tail.
                # Cheap when nothing's pending — usually a length check.
                cur = self._current_props
                for i in range(self._current_props_version, n_events):
                    for k, v in self.events[i].property_changes.items():
                        cur[k] = v
                self._current_props_version = n_events
            key = (entity_id, prop)
            cur = self._current_props
            if key in cur:
                return cur[key]
            # First lookup for this (eid, prop): no event has touched
            # it, so the value is the entity's scene-init property.
            # Cache for repeat lookups.
            val = ent.properties.get(prop)
            cur[key] = val
            return val

        # Historical-query path: keep the original (eid, prop, t) cache.
        cache_key = (entity_id, prop, t)
        cache = self._property_at_cache
        cached = cache.get(cache_key, _MISS)
        if cached is not _MISS:
            return cached

        key = (entity_id, prop)
        last_idx = min(t, n_events) - 1
        for i in range(last_idx, -1, -1):
            ev = self.events[i]
            if key in ev.property_changes:
                val = ev.property_changes[key]
                cache[cache_key] = val
                return val

        val = ent.properties.get(prop)
        cache[cache_key] = val
        return val


# The imperative `run_to_fixed_point` engine and its `Rule =
# Callable[[Trace, int], list[Event]]` protocol used to live here.
# Retired in Phase 5 of the ontology migration; replaced by
# `esperanto_lm.ontology.dsl.run_dsl` which runs declarative DSL rules
# plus a derivation layer. See `ontology/rules.py` for the redirect
# note and `dsl/CLAUDE.md` for the new model.
