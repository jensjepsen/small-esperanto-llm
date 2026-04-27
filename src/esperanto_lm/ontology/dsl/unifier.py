"""Match-clause evaluation.

The core primitive is `enumerate_bindings(when, given, ctx)`: yield
each consistent (Var → value) binding that satisfies the clauses. Used
by both causal and derivation rules — only the caller differs (causal
rules pass a `focus_event` in the context; derivations don't).

Execution: `when` is searched first, then each clause of `given` in
order. Later clauses see bindings from earlier ones; since `search()`
on composed patterns threads bindings through `&`, `|`, `~`, this
works out to conjunctive normal form with per-clause variable
scoping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from ..causal import Event, Trace
from ..loader import Lexicon
from .patterns import Bindings, Pattern, Var


# ------------------- derived-state side table ---------------------

@dataclass
class DerivedState:
    """Tracks facts materialized by derivation rules. Two layers:

      properties — keyed by (entity_id, slot), value is the derived
        slot value. Distinct from `Trace.entities[*].properties` and
        event `property_changes`, which are asserted state.

      relations — set of (relation_name, args_tuple) tuples. Distinct
        from `Trace.relations`, which is the asserted relation list.
        Used for relations whose existence follows from other state
        (e.g. `samloke(A, B)` from shared `en` container).

    Fully rebuilt at the start of each derivation phase — see
    `engine.run_dsl`. Consumers should check asserted first and fall
    back to derived (see `effective_property` /
    `effective_has_relation` on MatchContext)."""
    properties: dict[tuple[str, str], Any] = field(default_factory=dict)
    relations: set[tuple[str, tuple[str, ...]]] = field(default_factory=set)

    def clear(self) -> None:
        self.properties.clear()
        self.relations.clear()

    def get(self, eid: str, slot: str) -> Any:
        return self.properties.get((eid, slot))

    def set(self, eid: str, slot: str, value: Any,
            scalar: bool = True) -> bool:
        """Set a derived value. Returns True if the value changed.

        For scalar slots, replaces any prior value. For multi-valued
        slots (`scalar=False` in the slot definition), accumulates a
        sorted list — multiple derivations can contribute distinct
        values to the same slot (e.g. has_paws_can_walk +
        has_wings_can_fly both writing `locomotion` for a bird
        produces `[fly, walk]`, not a ping-pong)."""
        key = (eid, slot)
        prev = self.properties.get(key)
        if scalar:
            if prev == value:
                return False
            self.properties[key] = value
            return True
        # Multi-valued: accumulate as a sorted list of distinct values.
        if isinstance(prev, list):
            existing = list(prev)
        elif prev is None:
            existing = []
        else:
            existing = [prev]
        if value in existing:
            return False
        existing.append(value)
        existing.sort()
        self.properties[key] = existing
        return True

    def add_relation(self, name: str, args: tuple[str, ...]) -> bool:
        """Assert a derived relation. Returns True if it's new."""
        key = (name, tuple(args))
        if key in self.relations:
            return False
        self.relations.add(key)
        return True

    def has_relation(self, name: str, args: tuple[str, ...]) -> bool:
        return (name, tuple(args)) in self.relations

    def snapshot(self) -> dict[tuple[str, str], Any]:
        return dict(self.properties)


# ---------------------------- context ------------------------------

@dataclass
class MatchContext:
    """Everything a Pattern needs to evaluate: the trace (entities,
    relations, events), the lexicon (for subtype checks and concept
    field lookups), the derived-property table, and — for causal
    rules — the current focus event.

    `effective_property(eid, slot)` is the unified-view accessor: asserted
    wins over derived. Patterns use this when matching slot constraints.

    Indexes (lazy, scoped to one engine cycle):
      `entities_by_type` — exact entity_type → list of (eid, ent)
      `relations_by_name` — relation name → list of args tuples,
        union of asserted + derived.
    Pattern matching consults these to avoid O(N) scans of the full
    entities dict / relations list per pattern × per derivation cycle."""
    trace: Trace
    lexicon: Lexicon
    derived: DerivedState
    focus_event: Optional[Event] = None
    _entities_by_type_cache: Optional[dict] = None
    _relations_by_name_cache: Optional[dict] = None

    def effective_property(self, eid: str, slot: str) -> Any:
        """Read asserted-then-derived. Used by EntityPattern constraint
        checks so derivations are transparent to causal matching."""
        asserted = self.trace.property_at(eid, slot, len(self.trace.events))
        if asserted is not None:
            return asserted
        return self.derived.get(eid, slot)

    def entities_of_type(self, type_name: str) -> list:
        """Return [(eid, ent)] where ent.entity_type is-a `type_name`.
        Cached: built on first call by walking trace.entities once and
        bucketing by every ancestor type. Subsequent lookups are O(1)
        in the bucket."""
        if self._entities_by_type_cache is None:
            spine = self.lexicon.types
            buckets: dict[str, list] = {}
            # For each entity, add it to the bucket for its exact type
            # AND every ancestor type. is_subtype handles the spine
            # walk once per (entity_type × candidate-ancestor) but we
            # only call it for ancestors actually queried — so do
            # exact-type bucketing here, expand on lookup.
            for eid, ent in self.trace.entities.items():
                buckets.setdefault(ent.entity_type, []).append((eid, ent))
            self._entities_by_type_cache = buckets
        cache = self._entities_by_type_cache
        # Direct hit
        direct = cache.get(type_name, [])
        # Walk other buckets to find subtype matches and merge.
        # Memoize the merged result under the queried type so future
        # queries are O(1).
        merged_key = ("__merged__", type_name)
        if merged_key in cache:
            return cache[merged_key]
        spine = self.lexicon.types
        merged = []
        for et, ents in cache.items():
            if isinstance(et, tuple):
                continue
            try:
                if spine.is_subtype(et, type_name):
                    merged.extend(ents)
            except KeyError:
                continue
        cache[merged_key] = merged
        return merged

    def relations_of(self, relation_name: str) -> list:
        """Return [args_tuple] for asserted + derived relations with
        the given name. Cached, but the cache is invalidated when
        `derived.relations` grows — derivations chain (e.g. samloke
        propagates through parts after shared_container fires) and
        later derivations within the same cycle must see earlier ones'
        relation outputs."""
        derived_version = len(self.derived.relations)
        if (self._relations_by_name_cache is None
                or getattr(self, "_relations_cache_version", None)
                != derived_version):
            buckets: dict[str, list] = {}
            for r in self.trace.relations:
                buckets.setdefault(r.relation, []).append(r.args)
            for (name, args) in self.derived.relations:
                buckets.setdefault(name, []).append(args)
            self._relations_by_name_cache = buckets
            self._relations_cache_version = derived_version
        return self._relations_by_name_cache.get(relation_name, [])


# --------------------- enumerate bindings -------------------------

def enumerate_bindings(
    when: Pattern,
    given: tuple[Pattern, ...],
    ctx: MatchContext,
    initial: Optional[Bindings] = None,
) -> Iterator[Bindings]:
    """Yield every binding that satisfies `when` plus each `given`
    clause in order. Caller is responsible for packing a focus event
    into `ctx` when using a causal rule's `when`."""
    bindings: Bindings = {} if initial is None else dict(initial)
    # `when` first.
    for b_after_when in when.search(ctx, bindings):
        yield from _given_chain(given, 0, ctx, b_after_when)


def _given_chain(
    given: tuple[Pattern, ...], i: int,
    ctx: MatchContext, bindings: Bindings,
) -> Iterator[Bindings]:
    if i >= len(given):
        yield bindings
        return
    clause = given[i]
    for b1 in clause.search(ctx, bindings):
        yield from _given_chain(given, i + 1, ctx, b1)


# --------------------------- resolve ------------------------------

def resolve(value: Any, bindings: Bindings) -> Any:
    """Resolve a Var against bindings, or pass through a literal."""
    if isinstance(value, Var):
        if value not in bindings:
            raise KeyError(f"unbound variable ${value.name} at resolution")
        return bindings[value]
    return value
