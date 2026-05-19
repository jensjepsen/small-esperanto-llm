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
