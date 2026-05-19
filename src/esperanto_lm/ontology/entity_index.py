"""Per-trace bitmap index of entities by type + initial slot values.

Mirrors `concept_index.ConceptIndex` at trace level — same query
shape (`(type, properties)` intersection), same frozenset
operations. Replaces per-call `[eid for eid, ent in trace.entities
if is_subtype(...)]` loops at planner / spawner / drive-sampler
sites.

Invariants (per `ontology/CLAUDE.md`):
  - `trace.entities` is append-only — entities are added, never
    removed mid-plan.
  - `EntityInstance.properties` is immutable after creation.
    Mutable state lives in `Event.property_changes`; CURRENT state
    at position t is `trace.property_at(eid, prop, t)`.

The index reflects INITIAL properties only. Callers needing
current state should query the engine's derived state. The cache
key is `len(trace.entities)` — additions invalidate, properties
are stable so no further check needed.
"""
from __future__ import annotations

from typing import Any, Optional


_EMPTY: frozenset = frozenset()
_ENTITY_INDEX_ATTR = "_fwd_entity_index"


class EntityIndex:
    """See module docstring."""

    __slots__ = ("_trace", "_lex", "_size", "_bms")

    def __init__(self, trace, lex):
        self._trace = trace
        self._lex = lex
        self._size = -1
        self._bms: dict = {}

    def _ensure_fresh(self) -> None:
        cur = len(self._trace.entities)
        if cur == self._size:
            return
        bms_mut: dict = {}
        for eid, ent in self._trace.entities.items():
            for t in self._lex.types._ancestors.get(ent.entity_type, ()):
                bms_mut.setdefault(("type", t), set()).add(eid)
            bms_mut.setdefault(
                ("concept", ent.concept_lemma), set()).add(eid)
            for slot, values in ent.properties.items():
                bms_mut.setdefault(("slot", slot, None), set()).add(eid)
                for v in values:
                    bms_mut.setdefault(("slot", slot, v), set()).add(eid)
        self._bms = {k: frozenset(v) for k, v in bms_mut.items()}
        self._size = cur

    def entities_of_concept(self, concept_lemma: str) -> frozenset:
        """Entities whose `concept_lemma` matches exactly."""
        self._ensure_fresh()
        return self._bms.get(("concept", concept_lemma), _EMPTY)

    def entities_matching(
        self, role_type: Optional[str] = None,
        properties: Optional[dict] = None,
    ) -> frozenset:
        """Intersection of: type pool + per-(slot, value) pool for
        each `(slot, [value])` in `properties`. Mirrors
        `ConceptIndex.concepts_matching` shape so callers can compose
        the two interchangeably."""
        self._ensure_fresh()
        pools: list = []
        if role_type is not None:
            pool = self._bms.get(("type", role_type))
            if pool is None:
                return _EMPTY
            pools.append(pool)
        if properties:
            for slot, values in properties.items():
                if values:
                    pool = self._bms.get(("slot", slot, values[0]))
                else:
                    pool = self._bms.get(("slot", slot, None))
                if pool is None:
                    return _EMPTY
                pools.append(pool)
        if not pools:
            all_eids: set = set()
            for (kind, *_), pool in self._bms.items():
                if kind == "type":
                    all_eids.update(pool)
            return frozenset(all_eids)
        return frozenset.intersection(*pools)

    def matches_constraints(self, constraints: dict) -> frozenset:
        """Bitmap pool of entities satisfying ALL literal constraints
        in `constraints`. Mirrors the per-entity matcher that the
        planner's `_ground_derivations` used to call inline — keys
        are `"type"` / `"concept"` / slot names; list/tuple values on
        a slot use "any-of" (union) semantics, matching the raw
        EntityPattern constraint shape. `Var` values are treated as
        unconstrained (they bind at use time, not match time).

        Empty constraints return the full entity universe; an empty
        list value on a slot returns the empty set (mirrors the
        `not any(x in vals for x in v)` short-circuit in the
        former matcher).
        """
        self._ensure_fresh()
        # Local import keeps entity_index.py independent of dsl/.
        from .dsl.patterns import Var
        pools: list = []
        for k, v in constraints.items():
            if isinstance(v, Var):
                continue
            if k == "type":
                pool = self._bms.get(("type", v))
                if pool is None:
                    return _EMPTY
                pools.append(pool)
            elif k == "concept":
                pool = self._bms.get(("concept", v))
                if pool is None:
                    return _EMPTY
                pools.append(pool)
            elif isinstance(v, (list, tuple)):
                # "any-of" semantics: union of per-value pools.
                if not v:
                    return _EMPTY
                union: frozenset = _EMPTY
                for x in v:
                    p = self._bms.get(("slot", k, x))
                    if p is not None:
                        union = union | p
                if not union:
                    return _EMPTY
                pools.append(union)
            else:
                pool = self._bms.get(("slot", k, v))
                if pool is None:
                    return _EMPTY
                pools.append(pool)
        if not pools:
            # No literal constraints — everything matches.
            all_eids: set = set()
            for (kind, *_), pool in self._bms.items():
                if kind == "type":
                    all_eids.update(pool)
            return frozenset(all_eids)
        return frozenset.intersection(*pools)


def entity_index_for(trace, lex) -> EntityIndex:
    """Cached per-trace EntityIndex. Stored on the trace via
    `object.__setattr__` so dataclass-frozen traces still work.
    Invalidates only on lex switch (different worker process)."""
    cached = getattr(trace, _ENTITY_INDEX_ATTR, None)
    if cached is None or cached._lex is not lex:
        cached = EntityIndex(trace, lex)
        try:
            object.__setattr__(trace, _ENTITY_INDEX_ATTR, cached)
        except Exception:
            pass
    return cached
