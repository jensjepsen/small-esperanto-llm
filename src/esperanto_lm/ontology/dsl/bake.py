"""Materialize concept-compatible derivations at lex-load time.

The runtime DSL engine fires derivations against entities each cycle.
The sampler, however, picks role candidates by scanning
`concept.properties` directly — it never sees the runtime-derived
layer. That's a single-source-of-truth gap: facts that should be
visible to both layers (e.g. "all animals are solid") only land in
one.

`bake_concept_derivations(concepts, derivations, lex)` closes the
gap by running each derivation once at load time, against each
concept treated as a proto-entity. Implied properties get written
back into the concept's `properties` dict. The sampler now sees
them; the runtime engine does too (it reads the same data).

Scope: only "concept-compatible" derivations are baked — those
whose `when` is built from EntityPattern / BindPattern / And / Or /
Not, no relations and no events. Any derivation whose match depends
on relations or causal context is left to runtime; baking it would
require running the engine against a fully-populated trace, which
isn't available at load.

Iterates to fixed point so chained derivations resolve in one pass:
animate → made_of=meat → edibility=edible all land on cat in a
single bake.

Asserted-wins parity with the runtime engine: if a concept already
has the property explicitly set on disk, the bake won't override.
"""
from __future__ import annotations

from typing import Iterable

from ..schemas import Concept
from .patterns import (
    AndPattern, BindPattern, EntityPattern, NotPattern, OrPattern, Pattern, Var,
)


def _concept_compatible(when: Pattern) -> bool:
    """True iff `when` can be evaluated against a single concept
    (no relations, no events, no closures, no concept-field reads)."""
    if isinstance(when, EntityPattern):
        return True
    if isinstance(when, BindPattern):
        return True
    if isinstance(when, (AndPattern, OrPattern)):
        return _concept_compatible(when.left) and _concept_compatible(when.right)
    if isinstance(when, NotPattern):
        return _concept_compatible(when.inner)
    return False


def _matches_concept(
    pattern: Pattern, concept: Concept, props: dict[str, list[str]],
    types,
) -> bool:
    """Evaluate a concept-compatible pattern against a concept."""
    if isinstance(pattern, EntityPattern):
        for key, expected in pattern.constraints.items():
            if key == "type":
                # Tolerate unknown types/ancestors gracefully — the
                # bake is best-effort; a derivation referencing a type
                # that isn't in this lexicon's spine just doesn't match.
                # This matters for test fixtures with minimal spines.
                try:
                    if not types.is_subtype(concept.entity_type, expected):
                        return False
                except KeyError:
                    return False
            elif key == "has_suffix":
                if not concept.lemma.endswith(expected):
                    return False
            elif key == "concept":
                if concept.lemma != expected:
                    return False
            else:
                actual = props.get(key)
                if actual is None:
                    return False
                if expected is Ellipsis:
                    if not actual:
                        return False
                elif isinstance(expected, list):
                    if not any(v in actual for v in expected):
                        return False
                else:
                    if expected not in actual:
                        return False
        return True
    if isinstance(pattern, BindPattern):
        return True   # binding succeeds if a value is in scope
    if isinstance(pattern, AndPattern):
        return (_matches_concept(pattern.left, concept, props, types)
                and _matches_concept(pattern.right, concept, props, types))
    if isinstance(pattern, OrPattern):
        return (_matches_concept(pattern.left, concept, props, types)
                or _matches_concept(pattern.right, concept, props, types))
    if isinstance(pattern, NotPattern):
        return not _matches_concept(pattern.inner, concept, props, types)
    return False


def bake_concept_derivations(
    concepts: dict[str, Concept], derivations: Iterable, lex_types,
    slots: dict | None = None,
) -> dict[str, Concept]:
    """Apply each concept-compatible derivation to each concept,
    iterating to fixed point. Returns a new concepts dict — the input
    is not mutated.

    `lex_types` is the TypeSpine for subtype checks. Passed in rather
    than imported because the loader hasn't finished building the
    Lexicon when this runs.
    """
    from .implications import PropertyImplication

    # Filter to derivations whose when is concept-compatible AND whose
    # given is empty (relations don't exist at concept time).
    eligible = [
        d for d in derivations
        if _concept_compatible(d.when) and not d.given
    ]
    if not eligible:
        return dict(concepts)

    # Mutable working props per concept. Start from the on-disk values.
    work: dict[str, dict[str, list[str]]] = {
        lemma: {k: list(v) for k, v in c.properties.items()}
        for lemma, c in concepts.items()
    }

    # Iterate to fixed point — chained derivations need it.
    for _ in range(50):
        changed = False
        for d in eligible:
            for lemma, c in concepts.items():
                props = work[lemma]
                if not _matches_concept(d.when, c, props, lex_types):
                    continue
                for imp in d.implies:
                    if not isinstance(imp, PropertyImplication):
                        continue
                    value = imp.value
                    if isinstance(value, Var):
                        continue   # can't resolve free vars at concept time
                    slot_def = slots.get(imp.slot) if slots else None
                    is_scalar = slot_def is None or slot_def.scalar
                    if imp.slot in props:
                        if is_scalar:
                            continue   # asserted wins for scalar slots
                        # Multi-valued slot: APPEND if not already present.
                        # Lets multiple derivations contribute (e.g.
                        # person_can_walk + person_can_swim both writing
                        # to locomotion).
                        if value not in props[imp.slot]:
                            props[imp.slot] = props[imp.slot] + [value]
                            changed = True
                    else:
                        props[imp.slot] = [value]
                        changed = True
        if not changed:
            break

    out: dict[str, Concept] = {}
    for lemma, c in concepts.items():
        out[lemma] = c.model_copy(update={"properties": work[lemma]})
    return out
