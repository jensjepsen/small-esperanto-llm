"""Per-lexicon concept index: O(1) frozenset lookup of concept
lemmas matching a (type, properties) filter.

Built once per lex and reused by every site that previously did
`is_subtype(...) + per-concept property iteration`. Pure-Python
frozensets — no C extension, runs identical on CPython and PyPy.
At ~800 concepts, intersection of small sets is microseconds; the
gain over the imperative loop is 40-60× on the type+property bench.

Two-axis index:
  - type bitmap: `bms[("type", T)]` → concepts whose entity_type is
    subtype of T (transitive closure via TypeSpine ancestors).
  - slot bitmap: `bms[("slot", name, value)]` → concepts whose
    `properties[name]` contains `value`. Use `value=None` for
    "concepts that declare this slot at all".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


_EMPTY: frozenset = frozenset()


@dataclass(frozen=True)
class ConceptIndex:
    bms: dict
    """Keys are tuples:
      ("type", type_name)              → frozenset[lemma]
      ("slot", slot_name, value_or_None) → frozenset[lemma]
    """

    @classmethod
    def build(cls, concepts: dict, types) -> "ConceptIndex":
        bms_mut: dict = {}
        for lemma, concept in concepts.items():
            # Type bitmap — transitive via TypeSpine._ancestors.
            for t in types._ancestors.get(concept.entity_type, ()):
                bms_mut.setdefault(("type", t), set()).add(lemma)
            # Slot bitmaps — both presence (value=None) and per-value.
            for slot, values in concept.properties.items():
                bms_mut.setdefault(("slot", slot, None), set()).add(lemma)
                for v in values:
                    bms_mut.setdefault(("slot", slot, v), set()).add(lemma)
        return cls(bms={k: frozenset(v) for k, v in bms_mut.items()})

    def concepts_matching(
        self, role_type: Optional[str] = None,
        properties: Optional[dict] = None,
    ) -> frozenset:
        """Intersection of: type pool + per-(slot, value) pool for
        each slot in `properties`.

        `properties[slot]` shape mirrors RoleSpec.properties: an
        empty list means "presence only" (uses the slot-any bitmap);
        a non-empty list pre-filters on the FIRST value (matches the
        existing role.properties iteration semantics — multi-value
        role.properties are degenerate in current schema).

        Callers that want to skip certain slots (e.g. mutable ones
        the planner achieves via effects) should filter `properties`
        themselves before calling."""
        pools: list = []
        if role_type is not None:
            pool = self.bms.get(("type", role_type))
            if pool is None:
                return _EMPTY
            pools.append(pool)
        if properties:
            for slot, values in properties.items():
                if values:
                    pool = self.bms.get(("slot", slot, values[0]))
                else:
                    pool = self.bms.get(("slot", slot, None))
                if pool is None:
                    return _EMPTY
                pools.append(pool)
        if not pools:
            # No constraints — union of all type bitmaps. Rare path.
            all_concepts: set = set()
            for (kind, *_), pool in self.bms.items():
                if kind == "type":
                    all_concepts.update(pool)
            return frozenset(all_concepts)
        return frozenset.intersection(*pools)


