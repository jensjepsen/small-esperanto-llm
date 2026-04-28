"""Materialize concept-compatible derivations at lex-load time.

The runtime DSL engine fires derivations against entities each cycle.
The sampler, however, picks role candidates by scanning
`concept.properties` directly — it never sees the runtime-derived
layer. That's a single-source-of-truth gap: facts that should be
visible to both layers (e.g. "all animals are solid") only land in
one.

`bake_concept_derivations(concepts, derivations, lex)` closes the
gap by running derivations at load time and writing implied
properties back into each concept's `properties` dict. The sampler
now sees them; the runtime engine does too (it reads the same data).

Two passes:

  1. Concept-only pass: derivations with no `given` clauses, evaluated
     against each concept treated as a standalone proto-entity. Cheap
     and sufficient for type-keyed defaults like `animate_has_hunger`.

  2. Parts-aware pass (when `relations` is provided): derivations whose
     `given` walks `havas_parton` to a part. For each concept C with
     parts, build a synthetic Trace containing C + its parts +
     `havas_parton` relations between them, then run the actual
     runtime engine matchers against it. Property implications that
     fire on the host C get baked back. This handles `pordo` getting
     `lock_state` from its `seruro` part and `persono` getting
     `can_use_tools` from its `mano` part — without re-implementing
     pattern evaluation for relations.

Iterates to fixed point so chained derivations resolve in one pass:
animate → made_of=meat → edibility=edible all land on cat in a
single bake.

Asserted-wins parity with the runtime engine: if a concept already
has the property explicitly set on disk, the bake won't override.
"""
from __future__ import annotations

from typing import Iterable

from ..schemas import Concept, ConceptPart
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
    slots: dict | None = None, relations: dict | None = None,
) -> dict[str, Concept]:
    """Apply derivations to each concept, iterating to fixed point.
    Returns a new concepts dict — the input is not mutated.

    `lex_types` is the TypeSpine for subtype checks. Passed in rather
    than imported because the loader hasn't finished building the
    Lexicon when this runs.

    `relations` enables the parts-aware second pass. Without it, only
    the concept-only pass runs (derivations with empty `given`). With
    it, derivations whose `given` walks `havas_parton` also fire,
    materialized via per-concept proto-traces.
    """
    from .implications import PartImplication, PropertyImplication

    derivations = list(derivations)

    # Mutable working props per concept. Start from the on-disk values.
    work: dict[str, dict[str, list[str]]] = {
        lemma: {k: list(v) for k, v in c.properties.items()}
        for lemma, c in concepts.items()
    }
    # Mutable working parts per concept, also from on-disk.
    parts_work: dict[str, list[ConceptPart]] = {
        lemma: list(c.parts) for lemma, c in concepts.items()
    }

    # ---- pass 1: concept-only (no given) ----
    eligible = [
        d for d in derivations
        if _concept_compatible(d.when) and not d.given
    ]
    for _ in range(50):
        changed = False
        for d in eligible:
            for lemma, c in concepts.items():
                props = work[lemma]
                if not _matches_concept(d.when, c, props, lex_types):
                    continue
                for imp in d.implies:
                    if isinstance(imp, PropertyImplication):
                        value = imp.value
                        if isinstance(value, Var):
                            continue   # can't resolve free vars at concept time
                        slot_def = slots.get(imp.slot) if slots else None
                        # varies=true slots get re-derived per-entity at
                        # runtime (the host's openness depends on its
                        # specific pordo's randomized openness, etc.).
                        # Baking would freeze them to the concept default
                        # and asserted-wins blocks runtime updates.
                        if slot_def is not None and slot_def.varies:
                            continue
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
                    elif isinstance(imp, PartImplication):
                        # Append a ConceptPart unless one with the same
                        # (concept, relation) already exists. Authored
                        # parts win; identical derived parts dedupe.
                        existing = {(p.concept, p.relation)
                                    for p in parts_work[lemma]}
                        key = (imp.part_concept, imp.relation)
                        if key in existing:
                            continue
                        parts_work[lemma].append(ConceptPart(
                            concept=imp.part_concept,
                            relation=imp.relation))
                        changed = True
        if not changed:
            break

    # Materialize parts updates before pass 2 so the parts-aware pass
    # sees the inherited parts (e.g. kuiristo gets mano → can lift
    # can_use_tools through has_hands_can_use_tools).
    concepts = {
        lemma: c.model_copy(update={
            "properties": work[lemma],
            "parts": parts_work[lemma],
        })
        for lemma, c in concepts.items()
    }

    # ---- pass 2: parts-aware (synthetic trace per concept) ----
    if relations is not None:
        _bake_parts_aware(
            concepts, work, derivations, lex_types, slots, relations)

    out: dict[str, Concept] = {}
    for lemma, c in concepts.items():
        out[lemma] = c.model_copy(update={"properties": work[lemma]})
    return out


def _has_negation(derivation) -> bool:
    """True if any pattern in the derivation's `when` or `given`
    contains a NotPattern. Closed-world derivations (`X because Y is
    absent`) shouldn't be baked — the proto-trace's emptiness gives
    the wrong answer."""
    def walk(p) -> bool:
        if isinstance(p, NotPattern):
            return True
        for child_attr in ("left", "right", "inner", "to_", "where"):
            child = getattr(p, child_attr, None)
            if child is not None and isinstance(child, Pattern) and walk(child):
                return True
        for d_attr in ("role_patterns", "arg_patterns"):
            d = getattr(p, d_attr, None)
            if isinstance(d, dict):
                for v in d.values():
                    if isinstance(v, Pattern) and walk(v):
                        return True
        return False
    if walk(derivation.when):
        return True
    return any(walk(g) for g in derivation.given)


def _references_known_symbols(derivation, lex_types, relations: dict) -> bool:
    """Return False if the derivation references a type or relation not
    present in the given spine/relations dict. Used to silently drop
    default derivations on minimal test-fixture lexicons."""
    from .patterns import (
        AndPattern, ClosurePattern, EntityPattern, NotPattern, OrPattern,
        RelPattern,
    )

    def walk(p) -> bool:
        if isinstance(p, EntityPattern):
            t = p.constraints.get("type")
            if t is not None and not isinstance(t, str):
                return True
            if t is not None and not lex_types.known(t):
                return False
            return True
        if isinstance(p, RelPattern):
            if p.relation not in relations:
                return False
            return all(walk(v) for v in p.arg_patterns.values())
        if isinstance(p, ClosurePattern):
            if any(r not in relations for r in p.relations):
                return False
            return walk(p.to_) and (p.where is None or walk(p.where))
        if isinstance(p, (AndPattern, OrPattern)):
            return walk(p.left) and walk(p.right)
        if isinstance(p, NotPattern):
            return walk(p.inner)
        return True

    if not walk(derivation.when):
        return False
    return all(walk(g) for g in derivation.given)


def _bake_parts_aware(
    concepts: dict[str, Concept],
    work: dict[str, dict[str, list[str]]],
    derivations: list,
    lex_types, slots: dict | None, relations: dict,
) -> None:
    """For each concept with parts, build a proto-trace of host + parts +
    `havas_parton` edges, run the runtime derivation engine against it,
    and merge property implications targeting the host into `work`.

    Mutates `work` in place. Iterates to fixed point at the outer level
    so a property baked onto a host can unblock further derivations
    (host_lock_state firing then enabling something keyed on the host
    having lock_state, etc.).

    Parts of parts are NOT recursively materialized — current
    derivations only need one-hop `havas_parton`. If a future rule
    walks deeper, materialize accordingly here.
    """
    from ..causal import EntityInstance, RelationAssertion, Trace
    from ..loader import Lexicon
    from .engine import _run_derivations_to_fixed_point
    from .unifier import DerivedState

    HOST_ID = "__host"

    proto_lex = Lexicon(
        types=lex_types,
        slots=slots or {},
        concepts=concepts,
        relations=relations,
        actions={}, affixes={}, containment=[], qualities={},
    )

    # Filter out derivations referencing types or relations not in this
    # lexicon — minimal test-fixture spines may legitimately lack
    # `animate` etc. The runtime engine raises on unknown ancestors;
    # the bake should silently skip, matching the concept-only pass's
    # try/except-on-unknown-type philosophy.
    #
    # Also skip derivations whose `given` contains a NotPattern: these
    # are closed-world assertions ("X holds because Y is absent"). At
    # bake time the proto-trace is artificially sparse, so absence is
    # spurious — `indoor_dark_without_lamp` would fire on every indoor
    # concept and bake `lit_state=malluma`, which then beats the
    # runtime indoor-lit-by-lamp derivation via asserted-wins. Leave
    # closed-world derivations to runtime where the trace is real.
    eligible_derivations = [
        d for d in derivations
        if _references_known_symbols(d, lex_types, relations)
        and not _has_negation(d)
    ]

    for _ in range(50):
        changed = False
        for host_lemma, host_concept in concepts.items():
            # Run for every concept — even parts-less ones — so closed-
            # world negation derivations like artifact_without_seruro
            # fire (they assert the absence of a part).
            trace = Trace()
            trace.entities[HOST_ID] = EntityInstance(
                id=HOST_ID, concept_lemma=host_lemma,
                entity_type=host_concept.entity_type,
                properties={k: list(v) for k, v in work[host_lemma].items()},
            )
            for i, part in enumerate(host_concept.parts):
                part_concept = concepts.get(part.concept)
                if part_concept is None:
                    continue
                pid = f"__part_{i}"
                trace.entities[pid] = EntityInstance(
                    id=pid, concept_lemma=part.concept,
                    entity_type=part_concept.entity_type,
                    properties={k: list(v) for k, v in work[part.concept].items()},
                )
                trace.relations.append(
                    RelationAssertion(relation=part.relation, args=(HOST_ID, pid)))

            derived = DerivedState()
            _run_derivations_to_fixed_point(
                trace, eligible_derivations, proto_lex, derived)

            for (eid, slot), value in derived.properties.items():
                if eid != HOST_ID:
                    continue
                slot_def = slots.get(slot) if slots else None
                # varies=true slots: same reasoning as the simple-pass —
                # baking freezes them to the concept default and blocks
                # runtime re-derivation per-instance.
                if slot_def is not None and slot_def.varies:
                    continue
                is_scalar = slot_def is None or slot_def.scalar
                props = work[host_lemma]
                if slot in props:
                    if is_scalar:
                        continue
                    existing = props[slot]
                    new_vals = value if isinstance(value, list) else [value]
                    for v in new_vals:
                        if v not in existing:
                            props[slot] = existing + [v]
                            existing = props[slot]
                            changed = True
                else:
                    props[slot] = (
                        list(value) if isinstance(value, list) else [value])
                    changed = True
        if not changed:
            return
