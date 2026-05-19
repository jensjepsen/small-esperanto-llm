"""Per-lexicon concept index: O(1) frozenset lookup of concept
lemmas matching a (type, properties) filter.

Built once per lex at load time and attached as `Lex.concept_index`.
Pure-Python frozensets — no C extension, runs identical on CPython
and PyPy.

Two query surfaces:

  - `concepts_matching(type, properties)`: literal intersection of
    type + per-(slot, value) bitmaps. Use when the caller knows
    exactly which (slot, value) pairs the concept must declare —
    no varies/derivable awareness. Fast, generic.

  - `concepts_matching_role(role_spec)`: applies the schema's slot
    semantics (varies, pervasive, runtime-derivable) so a role-spec
    with a pervasive-slot constraint accepts any type-eligible
    concept, etc. Use when filtering by an Action role's
    `properties` dict — replaces the historical
    `_concepts_matching_role` predicate from seeders.py.

Implementation: two-axis bitmap.
  - bms[("type", T)]            → frozenset[lemma] (subtype-closed)
  - bms[("slot", name, value)]  → frozenset[lemma]
  - bms[("slot", name, None)]   → frozenset[lemma] (presence only)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


_EMPTY: frozenset = frozenset()


@dataclass
class ConceptIndex:
    bms: dict
    """Keys are tuples:
      ("type", type_name)              → frozenset[lemma]
      ("slot", slot_name, value_or_None) → frozenset[lemma]
    """
    # Populated by `with_role_semantics` so `concepts_matching_role`
    # and `concepts_modeling_slot` can apply slot semantics. When
    # None, only the literal `concepts_matching` query is available.
    slots: Optional[dict] = None
    derivable: Optional[dict] = None
    types: Any = None
    stubs: frozenset = field(default_factory=frozenset)
    # Concepts that appear as some other concept's `parts[*].concept`
    # (body parts, components, recipe ingredients tagged as parts).
    # Used by the regression standalone-target picker to exclude
    # body parts ("la dorso estis en salono") from scene targets.
    parts_used: frozenset = field(default_factory=frozenset)
    # Per-category transitive closure of concept membership. Built at
    # `build()` time by walking each concept's `category` chain
    # (reflexive: every concept is in its own category). Replaces
    # the recursive `_concept_in_category` walk with an O(1)
    # set lookup.
    _category_map: dict = field(default_factory=dict)
    # Held for lazy `concepts_modeling_slot` queries (which need to
    # call `concept_models_slot` against the lex's parts).
    _lex: Any = None
    _derivations: Any = None
    # Per-query caches.
    _role_cache: dict = field(default_factory=dict)
    _modeling_cache: dict = field(default_factory=dict)

    @classmethod
    def build(cls, concepts: dict, types) -> "ConceptIndex":
        bms_mut: dict = {}
        stubs: set = set()
        parts_used: set = set()
        for lemma, concept in concepts.items():
            if getattr(concept, "is_category_stub", False):
                stubs.add(lemma)
            for part in (concept.parts or ()):
                parts_used.add(part.concept)
            for t in types._ancestors.get(concept.entity_type, ()):
                bms_mut.setdefault(("type", t), set()).add(lemma)
            for slot, values in concept.properties.items():
                bms_mut.setdefault(("slot", slot, None), set()).add(lemma)
                for v in values:
                    bms_mut.setdefault(("slot", slot, v), set()).add(lemma)
        # Category transitive closure. For each concept, walk its
        # category chain (cycle-safe). Every visited category gets
        # the concept added to its membership set; the reflexive
        # case (concept is in its own category) is captured by
        # always seeding the closure with the concept's own lemma.
        cat_map_mut: dict = {}
        for lemma, concept in concepts.items():
            closure: set = {lemma}
            visited: set = set()
            queue: list = list(concept.category or ())
            while queue:
                cat = queue.pop()
                if cat in visited:
                    continue
                visited.add(cat)
                closure.add(cat)
                cat_concept = concepts.get(cat)
                if cat_concept is not None:
                    queue.extend(cat_concept.category or ())
            for cat in closure:
                cat_map_mut.setdefault(cat, set()).add(lemma)
        return cls(
            bms={k: frozenset(v) for k, v in bms_mut.items()},
            stubs=frozenset(stubs),
            parts_used=frozenset(parts_used),
            types=types,
            _category_map={k: frozenset(v) for k, v in cat_map_mut.items()},
        )

    def concepts_in_category(self, category: str) -> frozenset:
        """Concepts whose transitive `category` chain includes
        `category` — reflexive (a concept is in its own category).
        Replaces the recursive `_concept_in_category` walk with an
        O(1) frozenset lookup; pre-built at index time."""
        return self._category_map.get(category, _EMPTY)

    def with_role_semantics(
        self, lex, derivations,
    ) -> "ConceptIndex":
        """Attach slot metadata, the runtime-derivable map, and a
        lazy handle on (lex, derivations) for
        `concepts_modeling_slot`. Idempotent — sets attributes in
        place."""
        # Lazy import to break loader↔dsl cycle when called from the
        # loader at import time.
        from .dsl.introspect import fully_derivable_slots
        self.slots = lex.slots
        self.derivable = fully_derivable_slots(lex, derivations)
        self._lex = lex
        self._derivations = derivations
        self._role_cache.clear()
        self._modeling_cache.clear()
        return self

    def concepts_modeling_slot(self, slot: str) -> frozenset:
        """Concepts where `slot` is meaningfully tracked: declared,
        pervasive, or part-derivable via some derivation whose
        `given` is satisfiable by the concept's parts. Same predicate
        as `dsl.introspect.concept_models_slot`, vectorized + cached
        per-slot for use in hot loops (e.g. action-grounding's
        effect-target validity check).

        Requires `with_role_semantics(lex, derivations)`."""
        cached = self._modeling_cache.get(slot)
        if cached is not None:
            return cached
        assert self._lex is not None, (
            "concepts_modeling_slot requires with_role_semantics()")
        from .dsl.introspect import concept_models_slot
        result = frozenset(
            lemma for lemma, concept in self._lex.concepts.items()
            if concept_models_slot(
                concept, slot, self._lex, self._derivations))
        self._modeling_cache[slot] = result
        return result

    def concepts_matching(
        self, role_type: Optional[str] = None,
        properties: Optional[dict] = None,
    ) -> frozenset:
        """Literal intersection. `properties[slot]` shape mirrors
        RoleSpec.properties: empty list means "presence only"
        (uses the slot-any bitmap); non-empty list pre-filters on
        the first value."""
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

    def concepts_matching_role(self, role_spec) -> frozenset:
        """Apply Action role-spec semantics:
          - subtype-correct entity_type;
          - not a category stub (besto, planto, …);
          - for each (slot, vals) in role_spec.properties:
            * if the slot is fully derivable by a runtime derivation
              whose target type covers this concept's type, the
              constraint is considered satisfied (planner can
              subgoal it);
            * if the slot is varies+pervasive, considered satisfied
              (default derivation fills it for every applicable concept);
            * if the slot is varies+non-pervasive, concept must
              DECLARE the slot (any value);
            * else (immutable), concept's declared values must
              intersect `vals`.

        Requires `with_role_semantics(slots, derivable)` to have been
        called. Cached per (role.type, frozenset(role.properties))."""
        assert self.slots is not None and self.derivable is not None, (
            "concept_matching_role requires with_role_semantics()")
        # Hashable cache key.
        props = role_spec.properties or {}
        prop_key = frozenset(
            (s, tuple(v) if v else None) for s, v in props.items())
        cache_key = (role_spec.type, prop_key)
        cached = self._role_cache.get(cache_key)
        if cached is not None:
            return cached
        # Start with type pool minus stubs.
        type_pool = self.bms.get(("type", role_spec.type))
        if type_pool is None:
            self._role_cache[cache_key] = _EMPTY
            return _EMPTY
        pools: list = [type_pool - self.stubs]
        for slot, vals in props.items():
            slot_def = self.slots.get(slot)
            if slot_def is None:
                continue
            # Concepts whose type is covered by some derivation that
            # produces this slot dynamically — the slot constraint is
            # considered satisfied via runtime subgoaling, regardless
            # of declared value. Union of the type-pools for each
            # derivation-target type.
            deriv_covered: frozenset = _EMPTY
            for t in self.derivable.get(slot, ()):
                deriv_covered = deriv_covered | self.bms.get(
                    ("type", t), _EMPTY)
            if slot_def.varies:
                if getattr(slot_def, "pervasive", False):
                    continue  # default derivation fills it for all.
                # Non-pervasive varies: concept must declare the slot
                # OR be in the deriv-covered set.
                slot_any = self.bms.get(("slot", slot, None), _EMPTY)
                combined = slot_any | deriv_covered
                if not combined:
                    self._role_cache[cache_key] = _EMPTY
                    return _EMPTY
                pools.append(combined)
                continue
            # Immutable: concept's declared values must intersect
            # `vals`, OR concept must be in the deriv-covered set.
            union = deriv_covered
            for v in vals:
                union = union | self.bms.get(("slot", slot, v), _EMPTY)
            if not union:
                self._role_cache[cache_key] = _EMPTY
                return _EMPTY
            pools.append(union)
        result = frozenset.intersection(*pools)
        self._role_cache[cache_key] = result
        return result
