"""Containment pattern resolver.

Given a Lexicon with ContainmentFact entries (some specific, some
pattern-based), produce, for each concept C, the list of ContainmentFact
entries that apply to C when C is treated as a container.

This lets the sampler ask "what can be inside X?" without caring whether
the answer came from a specific `{container: "kuirejo"}` entry or from a
pattern like `{suffix: "ej"}`. Specific entries and all matching patterns
union together.

Rules (modal possibility, not frequency):
  - An entry asserts "this containment is possible"
  - Absence asserts nothing
  - The sampler decides how many reachable entities to place

Pattern semantics:
  - sense_id: exact lemma match
  - entity_type: concept.entity_type is-a pattern.entity_type (subtype)
  - suffix: morpheme parse of concept.lemma contains this suffix
  - property: concept.properties[slot] includes value (for every pair)
  - Multiple set fields: conjunction (all must match)
"""
from __future__ import annotations

from collections import defaultdict

from .loader import Lexicon
from .morph import DefaultMorphParser, MorphParser
from .schemas import Concept, ContainmentFact, ContainmentPattern


def _is_first_order(pattern: ContainmentPattern) -> bool:
    """A pattern is first-order if it has no relational fields (currently
    just `contains`). First-order patterns can be evaluated against the
    raw lexicon; second-order patterns need pass-1 reachability."""
    return pattern.contains is None


def _concept_matches_intrinsic(
    concept: Concept, pattern: ContainmentPattern,
    lexicon: Lexicon, parser: MorphParser,
) -> bool:
    """Conjunction of intrinsic (first-order) pattern fields. Skips
    relational fields like `contains` — those are checked separately by
    the second pass."""
    if pattern.sense_id is not None:
        if concept.lemma != pattern.sense_id:
            return False
    if pattern.entity_type is not None:
        if not lexicon.types.is_subtype(
                concept.entity_type, pattern.entity_type):
            return False
    if pattern.suffix is not None:
        parse = parser.parse(concept.lemma)
        if pattern.suffix not in parse.suffixes:
            return False
    if pattern.property is not None:
        for slot_name, value in pattern.property.items():
            if value not in concept.properties.get(slot_name, []):
                return False
    return True


# Backwards-compat alias used by the contained-side matcher (which is
# always first-order — we don't support contained_pattern.contains).
_concept_matches_pattern = _concept_matches_intrinsic


def resolve_containment(
    lexicon: Lexicon, parser: MorphParser | None = None,
) -> dict[str, list[ContainmentFact]]:
    """Return {container_concept_lemma: [facts that treat it as container]}.

    Two passes:
      1. Intrinsic / first-order facts (sense_id / entity_type / suffix /
         property). A fact applies to concept C if C's intrinsic
         properties match the pattern.
      2. Second-order facts (currently only `contains:`). For each
         candidate concept, we check both intrinsic conjuncts AND that
         the named lemma is reachable in pass-1 results.

    Pass 2 cannot see pass-2-added facts — `contains:` only sees the
    first-order graph. This is intentional: avoids fixpoint complications
    and keeps the schema's reasoning easy to follow. A future relational
    field (`affords:`, `instance_of:`) follows the same two-pass
    structure rather than introducing nested second-order semantics.
    """
    parser = parser or DefaultMorphParser()
    out: dict[str, list[ContainmentFact]] = defaultdict(list)

    first_order: list[ContainmentFact] = []
    second_order: list[ContainmentFact] = []
    for fact in lexicon.containment:
        pat = fact.container_pattern
        if pat is None:
            continue
        if _is_first_order(pat):
            first_order.append(fact)
        else:
            second_order.append(fact)

    # Pass 1: intrinsic patterns.
    for fact in first_order:
        pat = fact.container_pattern
        for lemma, concept in lexicon.concepts.items():
            if _concept_matches_intrinsic(concept, pat, lexicon, parser):
                out[lemma].append(fact)

    if not second_order:
        return dict(out)

    # Pre-compute pass-1 reachability per concept once.
    pass1_index = dict(out)
    pass1_reach = {
        lemma: reachable_from(lemma, pass1_index, lexicon, parser)
        for lemma in lexicon.concepts
    }

    # Pass 2: second-order patterns evaluated against pass-1 reachability.
    for fact in second_order:
        pat = fact.container_pattern
        for lemma, concept in lexicon.concepts.items():
            # Intrinsic conjuncts must still hold.
            if not _concept_matches_intrinsic(concept, pat, lexicon, parser):
                continue
            if pat.contains is not None:
                if pat.contains not in pass1_reach[lemma]:
                    continue
            out[lemma].append(fact)

    return dict(out)


def expand_contained(
    fact: ContainmentFact, lexicon: Lexicon,
    parser: MorphParser | None = None,
) -> list[str]:
    """Resolve a fact's contained side to a list of concrete concept lemmas.

    Handles all three forms:
      - contained = concept lemma → [lemma]
      - contained = type name → all concepts whose entity_type is-a this
      - contained_pattern → all concepts matching the pattern
    """
    parser = parser or DefaultMorphParser()
    if fact.contained is not None:
        if fact.contained in lexicon.concepts:
            return [fact.contained]
        if lexicon.types.known(fact.contained):
            return [
                lemma for lemma, c in lexicon.concepts.items()
                if lexicon.types.is_subtype(c.entity_type, fact.contained)
            ]
        return []
    if fact.contained_pattern is not None:
        return [
            lemma for lemma, c in lexicon.concepts.items()
            if _concept_matches_pattern(c, fact.contained_pattern,
                                        lexicon, parser)
        ]
    return []


def reachable_from(
    scene_lemma: str, containment_index: dict[str, list[ContainmentFact]],
    lexicon: Lexicon, parser: MorphParser | None = None,
) -> set[str]:
    """Set of concept lemmas reachable from `scene_lemma` by walking
    containment edges. Includes the scene itself.

    Type-contained and pattern-contained facts are expanded to their
    matching concepts. The walk terminates when no new concepts are
    discovered.
    """
    parser = parser or DefaultMorphParser()
    visited: set[str] = set()
    frontier = [scene_lemma]
    while frontier:
        cur = frontier.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for fact in containment_index.get(cur, []):
            for lemma in expand_contained(fact, lexicon, parser):
                if lemma not in visited:
                    frontier.append(lemma)
    return visited


def containment_relation_for(
    container_lemma: str, contained_lemma: str,
    containment_index: dict[str, list[ContainmentFact]],
    lexicon: Lexicon, parser: MorphParser | None = None,
) -> str | None:
    """Look up the relation name for a specific container/contained pair.

    A fact covers the pair if `contained_lemma` is in its expansion
    (whether by direct name match, type subsumption, or pattern match).
    Returns None if no fact covers this pair.

    Used by the sampler when materializing a concrete containment
    instance — it needs to know whether to assert `en` or `sur`.
    """
    parser = parser or DefaultMorphParser()
    facts = containment_index.get(container_lemma, [])
    for fact in facts:
        if contained_lemma in expand_contained(fact, lexicon, parser):
            return fact.relation
    return None
