"""Per-trace fact-ID table for the forward planner's bitmap state
representation.

NOT YET WIRED IN. The table itself is complete; integration into
the planner's grounding / heuristic / search loop is the next
focused planner refactor. Lands as the foundation so the follow-up
bitmap-state work can focus on the operator/loop rewrites without
also having to introduce the translation layer.

(`dsl/unifier` migration to `EntityIndex` already landed — the
trace-level bitmap pattern that motivates this fact-level one.)

Maps fact tuples (the three shapes used by the planner:
`("prop", eid, slot, value)`, `("rel", relation, args)`,
`("event_fired", verb, frozenset_bindings)`) to/from densely-packed
integer IDs starting at 0. The integer ID doubles as a bit position
in the bitmap state representation: `state = state | (1 << fact_id)`
sets the fact, `(pres_mask & state) == pres_mask` checks subset.

Lifetime: one table per trace, cached on the trace via
`object.__setattr__` so multiple `plan_for_goal` calls on the same
trace reuse the same IDs. Append-only — IDs never recycled, the
table just grows as new facts get emitted by grounding or by the
spawner adding entities mid-plan.

The forward planner is the only intended consumer. Other Trace
clients (realizer, sampler, scene_builder, engine) work in the
original fact-tuple representation; the table lives in `_fwd_*`
namespace to signal it's a planner-internal optimization.
"""
from __future__ import annotations

from typing import Iterable


class FactTable:
    """See module docstring."""

    __slots__ = ("_t2i", "_i2t")

    def __init__(self):
        self._t2i: dict = {}
        self._i2t: list = []

    def id_for(self, fact: tuple) -> int:
        """Return the int ID for `fact`, allocating a new one if
        unseen. Stable for the table's lifetime."""
        i = self._t2i.get(fact)
        if i is None:
            i = len(self._i2t)
            self._t2i[fact] = i
            self._i2t.append(fact)
        return i

    def ids_for(self, facts: Iterable[tuple]) -> frozenset:
        """Bulk forward translation: facts → frozenset of IDs.
        Convenience for grounding sites that emit a set at a time."""
        return frozenset(self.id_for(f) for f in facts)

    def mask_for(self, facts: Iterable[tuple]) -> int:
        """Bitmap form: facts → int with one bit set per fact.
        Phase-2 callers use this in place of `ids_for` once they're
        on the bitmap representation."""
        mask = 0
        for f in facts:
            mask |= 1 << self.id_for(f)
        return mask

    def tuple_for(self, fact_id: int) -> tuple:
        """Reverse lookup; O(1) via list index. Used at the
        plan→event boundary and for any debug rendering."""
        return self._i2t[fact_id]

    def tuples_for_mask(self, mask: int) -> list:
        """Decode a bitmap state back to a list of fact tuples.
        Used for `repr_state` / debug dumps. O(bits set)."""
        out = []
        while mask:
            lo = mask & -mask          # isolate lowest set bit
            out.append(self._i2t[lo.bit_length() - 1])
            mask ^= lo
        return out

    def __len__(self) -> int:
        return len(self._i2t)


_FACT_TABLE_ATTR = "_fwd_fact_table"


def fact_table_for(trace) -> FactTable:
    """Cached per-trace FactTable. Lazy-initialized on first call.
    Stored on the trace via `object.__setattr__` so frozen-dataclass
    traces still work."""
    cached = getattr(trace, _FACT_TABLE_ATTR, None)
    if cached is None:
        cached = FactTable()
        try:
            object.__setattr__(trace, _FACT_TABLE_ATTR, cached)
        except Exception:
            pass
    return cached
