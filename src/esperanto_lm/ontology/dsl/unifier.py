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

from ..causal import EntityInstance, Event, Trace
from ..loader import Lexicon
from .patterns import Bindings, Pattern, Var


# Shared empty mapping for `relations_position_bucket` when a
# relation has no entries — avoids handing back a fresh `{}` each
# call (and avoids `None`-checks in compiled enums).
_EMPTY_BUCKET: dict = {}


# ------------------- derived-state side table ---------------------

@dataclass
class DerivedState:
    """Tracks facts materialized by derivation rules. Three layers:

      properties — keyed by (entity_id, slot), value is the derived
        slot value. Distinct from `Trace.entities[*].properties` and
        event `property_changes`, which are asserted state.

      relations — set of (relation_name, args_tuple) tuples. Distinct
        from `Trace.relations`, which is the asserted relation list.
        Used for relations whose existence follows from other state
        (e.g. `samloke(A, B)` from shared `en` container).

      categories — entity_id → set of contextual category lemmas. A
        viro entity participating as the parent in a gepatro relation
        gets `patro` added here; the renderer's alias chain consults
        it on top of the static `concept.category` chain. Pattern
        matching against `entity(concept=...)` is unaffected — the
        derived category is realizer-only sugar.

    Fully rebuilt at the start of each derivation phase — see
    `engine.run_dsl`. Consumers should check asserted first and fall
    back to derived (see `effective_property` /
    `effective_has_relation` on MatchContext)."""
    properties: dict[tuple[str, str], Any] = field(default_factory=dict)
    # Ordered list of (name, args) — primary store. Insertion order is
    # stable, len() gives a monotonic version for incremental cache
    # consumers like `MatchContext.relations_of`. The parallel
    # `_relation_set` is the O(1) dedup index.
    relations: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    _relation_set: set[tuple[str, tuple[str, ...]]] = field(
        default_factory=set)
    # Hides relations from effective-state lookups. Populated by
    # RemoveRelationImplication. Asserted relations stay in the trace
    # untouched; the removal is a derived-layer mask. `relations_of`
    # in MatchContext filters returned tuples through this set so
    # both downstream consumers and chained derivations see the
    # effective state.
    removals: set[tuple[str, tuple[str, ...]]] = field(default_factory=set)
    categories: dict[str, set[str]] = field(default_factory=dict)

    def clear(self) -> None:
        self.properties.clear()
        self.relations.clear()
        self._relation_set.clear()
        self.removals.clear()
        self.categories.clear()

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
        if key in self._relation_set:
            return False
        self._relation_set.add(key)
        self.relations.append(key)
        return True

    def has_relation(self, name: str, args: tuple[str, ...]) -> bool:
        return (name, tuple(args)) in self._relation_set

    def add_removal(self, name: str, args: tuple[str, ...]) -> bool:
        """Mark a relation as hidden in effective state. Returns True
        if this is a new removal (drives the fixed-point delta loop)."""
        key = (name, tuple(args))
        if key in self.removals:
            return False
        self.removals.add(key)
        return True

    def has_removal(self, name: str, args: tuple[str, ...]) -> bool:
        return (name, tuple(args)) in self.removals

    def add_category(self, eid: str, lemma: str) -> bool:
        """Tag entity with a contextual category lemma. Returns True
        if the label is new for this entity."""
        bucket = self.categories.get(eid)
        if bucket is None:
            self.categories[eid] = {lemma}
            return True
        if lemma in bucket:
            return False
        bucket.add(lemma)
        return True

    def categories_for(self, eid: str) -> set[str]:
        """Read-only view of derived categories on an entity. Returns
        empty set when nothing is derived."""
        return self.categories.get(eid, set())

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
    _entities_by_type_cache: dict = field(default_factory=dict)
    _relations_by_name_cache: Optional[dict] = None
    _entities_by_concept_cache: dict = field(default_factory=dict)
    # Cached EntityIndex handle — avoids the per-call
    # `getattr(trace, _fwd_entity_index, None)` lookup the
    # `entity_index_for` helper does. Inside a unifier, the index
    # never changes identity.
    _entity_idx: Any = None
    # Per-(rel_name, position) → dict[eid, list[args]] secondary
    # index. Lets compiled enums anchor a rel iteration to a known
    # eid (pre-bound var or pool member) instead of scanning every
    # tuple of the relation. Built lazily on first
    # `relations_with_arg` call; same incremental-version pattern
    # as `_relations_by_name_cache`. Symmetric arity-2 relations
    # are stored bidirectionally so the codegen symmetric-expansion
    # dance disappears for callers that use this entry point.
    _rel_pos_index_cache: Optional[dict] = None
    _rel_pos_cache_version: int = 0

    def effective_property(self, eid: str, slot: str) -> Any:
        """Read asserted-then-derived. Used by EntityPattern constraint
        checks so derivations are transparent to causal matching."""
        asserted = self.trace.property_at(eid, slot, len(self.trace.events))
        if asserted is not None:
            return asserted
        return self.derived.get(eid, slot)

    def _idx(self):
        if self._entity_idx is None:
            from ..entity_index import entity_index_for
            object.__setattr__(
                self, "_entity_idx",
                entity_index_for(self.trace, self.lexicon))
        return self._entity_idx

    def entities_of_type(self, type_name: str) -> list:
        """Return [(eid, ent)] where ent.entity_type is-a `type_name`.
        Backed by the per-trace `EntityIndex` — its type bitmap is
        subtype-closed and shared across unifier instances built
        against the same trace (engine fixed-point loop, multiple
        plan_for_goal calls). Result list is memoized per type."""
        cached = self._entities_by_type_cache.get(type_name)
        if cached is not None:
            return cached
        result = [(eid, self.trace.entities[eid])
                  for eid in self._idx().entities_matching(type_name)]
        self._entities_by_type_cache[type_name] = result
        return result

    def entities_of_concept(self, concept_lemma: str) -> list:
        """Return [(eid, ent)] where ent.concept_lemma == `concept_lemma`.
        Backed by the per-trace `EntityIndex.entities_of_concept`
        bitmap. Pattern matching for `entity(concept=X)` uses this
        instead of scanning every entity — frequent in host-derivations
        like `host_lock_state_*_from_seruro` that gate on a part's
        concept."""
        cached = self._entities_by_concept_cache.get(concept_lemma)
        if cached is not None:
            return cached
        result = [(eid, self.trace.entities[eid])
                  for eid in self._idx().entities_of_concept(concept_lemma)]
        self._entities_by_concept_cache[concept_lemma] = result
        return result

    def relations_of(self, relation_name: str) -> list:
        """Return [args_tuple] for asserted + derived relations with
        the given name. Incrementally maintained: built once with the
        full asserted+derived state, then appended-to as new derived
        relations land within the same fixpoint. `derived.relations`
        is now an ordered list (see `DerivedState`), so we can index
        the delta by length and avoid the O(N) rebuild on every
        invalidation. This was the leader at 2.1s self-time before."""
        target_version = len(self.derived.relations)
        if self._relations_by_name_cache is None:
            buckets: dict[str, list] = {}
            for r in self.trace.relations:
                buckets.setdefault(r.relation, []).append(r.args)
            for (name, args) in self.derived.relations:
                buckets.setdefault(name, []).append(args)
            self._relations_by_name_cache = buckets
            self._relations_cache_version = target_version
        elif self._relations_cache_version != target_version:
            buckets = self._relations_by_name_cache
            for i in range(self._relations_cache_version, target_version):
                name, args = self.derived.relations[i]
                buckets.setdefault(name, []).append(args)
            self._relations_cache_version = target_version
        raw = self._relations_by_name_cache.get(relation_name, [])
        # Subtract removals (derived-layer relation masks). Skip the
        # filter loop in the common case where nothing's been removed,
        # since the bucket can be large and the removal set is small.
        removals = self.derived.removals
        if removals:
            return [args for args in raw
                    if (relation_name, args) not in removals]
        return raw

    def _ensure_rel_pos_index(self) -> dict:
        """Build / incrementally maintain the per-(name, position)
        index against `derived.relations`. Returns the index dict
        `(name, pos) → {eid → [args, ...]}`. Symmetric arity-2
        relations are stored bidirectionally."""
        target_version = len(self.derived.relations)
        idx = self._rel_pos_index_cache
        if idx is None:
            idx = {}
            for r in self.trace.relations:
                self._add_to_pos_index(
                    idx, r.relation, tuple(r.args))
            for (name, args) in self.derived.relations:
                self._add_to_pos_index(idx, name, args)
            self._rel_pos_index_cache = idx
            self._rel_pos_cache_version = target_version
        elif self._rel_pos_cache_version != target_version:
            for i in range(
                    self._rel_pos_cache_version, target_version):
                name, args = self.derived.relations[i]
                self._add_to_pos_index(idx, name, args)
            self._rel_pos_cache_version = target_version
        return idx

    def relations_position_bucket(
        self, relation_name: str, position: int,
    ) -> dict:
        """Return the `{eid → [args, ...]}` bucket for
        `(relation_name, position)`. Suitable for hoisting to a
        compiled enum's prelude — the (name, position) pair is
        loop-invariant; only the eid lookup varies. Per-query cost
        becomes one `dict.get`.

        Returns `{}` if no relations of this name have been
        asserted/derived. The compiled enum still has to filter
        `derived.removals` itself when that set is non-empty."""
        idx = self._ensure_rel_pos_index()
        return idx.get((relation_name, position), _EMPTY_BUCKET)

    def relations_with_arg(
        self, relation_name: str, position: int, eid: str,
    ) -> list:
        """Convenience wrapper: bucket lookup + removal filter +
        eid index. Kept for non-codegen callers that don't have a
        prelude to hoist into.

        Cuts compiled enums from O(|relations[name]|) per query to
        O(|tuples involving eid|) — usually an order of magnitude
        smaller. Symmetric arity-2 relations are stored
        bidirectionally so a query at any position returns every
        tuple involving `eid`."""
        bucket = self.relations_position_bucket(relation_name, position)
        raw = bucket.get(eid, [])
        removals = self.derived.removals
        if removals:
            return [args for args in raw
                    if (relation_name, args) not in removals]
        return raw

    def _add_to_pos_index(
        self, idx: dict, name: str, args: tuple,
    ) -> None:
        """Insert a relation into the position index at every arg
        position. For symmetric arity-2 (non-reflexive) tuples,
        also insert the swapped form — query at any position will
        then find every tuple involving the queried eid without a
        downstream symmetric-expansion step."""
        for pos, eid in enumerate(args):
            idx.setdefault((name, pos), {}).setdefault(eid, []).append(
                args)
        rel_def = self.lexicon.relations.get(name)
        if (rel_def is not None and rel_def.symmetric
                and len(args) == 2 and args[0] != args[1]):
            swapped = (args[1], args[0])
            for pos, eid in enumerate(swapped):
                idx.setdefault(
                    (name, pos), {}).setdefault(eid, []).append(swapped)



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
    """Resolve a Var against bindings, or pass through a literal.

    EntityInstance bindings (compiled enum can resolve event roles to
    entities when a given clause needs cross-arg matching) are reduced
    to their .id since effects operate on entity ids."""
    if isinstance(value, Var):
        if value not in bindings:
            raise KeyError(f"unbound variable ${value.name} at resolution")
        v = bindings[value]
        return v.id if isinstance(v, EntityInstance) else v
    return value
