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


def _concept_in_category(
    concept: Concept, category_lemma: str, lexicon: Lexicon,
    visited: set[str] | None = None,
) -> bool:
    """True iff concept's transitive `category` chain contains
    `category_lemma`. Reflexive: a concept is considered to be in its
    own category (a viro IS-A viro), so callers can use `category=X`
    to match both X-categorized concepts AND X itself when X is a
    real instantiable concept (not just a stub). Walks up through
    `concept.category` ↦ `<parent>.category`, cycle-safe via
    `visited`."""
    if visited is None:
        visited = set()
    if concept.lemma in visited:
        return False
    if concept.lemma == category_lemma:
        return True
    visited.add(concept.lemma)
    for cat in concept.category:
        if cat == category_lemma:
            return True
        cat_concept = lexicon.concepts.get(cat)
        if cat_concept is not None and _concept_in_category(
                cat_concept, category_lemma, lexicon, visited):
            return True
    return False


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
    if pattern.category is not None:
        if not _concept_in_category(concept, pattern.category, lexicon):
            return False
    return True


# Backwards-compat alias used by the contained-side matcher (which is
# always first-order — we don't support contained_pattern.contains).
_concept_matches_pattern = _concept_matches_intrinsic


_RESOLVE_CONTAINMENT_CACHE_ATTR = "_resolve_containment_cache"


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

    Result is cached per-lexicon-id: the lexicon is treated as immutable
    after load, and the seeders/planner call this hundreds of times per
    scene under containers_for/_place_respecting_containment. Without
    the cache the quadratic walk dominates regression sample time.
    """
    cached = getattr(lexicon, _RESOLVE_CONTAINMENT_CACHE_ATTR, None)
    if cached is not None:
        return cached
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
        result = dict(out)
        try:
            setattr(lexicon, _RESOLVE_CONTAINMENT_CACHE_ATTR, result)
        except (AttributeError, TypeError):
            pass
        return result

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

    result = dict(out)
    try:
        setattr(lexicon, _RESOLVE_CONTAINMENT_CACHE_ATTR, result)
    except (AttributeError, TypeError):
        pass
    return result


# `expand_contained` is called hundreds of thousands of times during
# planning and scene construction; its inputs (a frozen ContainmentFact
# + a treated-as-immutable Lexicon) never vary across a process. Cache
# the result lemmas. Keyed by `(id(lexicon), id(fact))`; bounded by
# the number of facts × the number of lexicon instances loaded (≤2
# in practice). Memo is process-global so workers share it.
_EXPAND_CONTAINED_CACHE: dict[tuple[int, int], list[str]] = {}


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
    cache_key = (id(lexicon), id(fact))
    cached = _EXPAND_CONTAINED_CACHE.get(cache_key)
    if cached is not None:
        return cached
    parser = parser or DefaultMorphParser()
    if fact.contained is not None:
        if fact.contained in lexicon.concepts:
            result = [fact.contained]
        elif lexicon.types.known(fact.contained):
            result = [
                lemma for lemma, c in lexicon.concepts.items()
                if lexicon.types.is_subtype(c.entity_type, fact.contained)
            ]
        else:
            result = []
    elif fact.contained_pattern is not None:
        result = [
            lemma for lemma, c in lexicon.concepts.items()
            if _concept_matches_pattern(c, fact.contained_pattern,
                                        lexicon, parser)
        ]
    else:
        result = []
    _EXPAND_CONTAINED_CACHE[cache_key] = result
    return result


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


def containers_for(
    contained_lemma: str,
    containment_index: dict[str, list[ContainmentFact]],
    lexicon: Lexicon, parser: MorphParser | None = None,
) -> list[tuple[str, str]]:
    """Inverse of `containment_relation_for`: given a contained lemma,
    return [(container_lemma, relation)] pairs where the contained
    could be placed. Walks every container's facts looking for ones
    whose `expand_contained` includes our lemma. Used by the sampler
    to lazily materialize a plausible container when nothing already
    in the scene fits — e.g. butero needs a tablo or korbo, neither
    in scene → look up "what could hold butero?" → pick tablo →
    materialize it (recursively place IT in scene), then put butero
    on it. Without this, butero would fall back to placement
    directly under the scene location, missing the natural
    'on a table' / 'in a basket' framing."""
    parser = parser or DefaultMorphParser()
    out: list[tuple[str, str]] = []
    for container_lemma, facts in containment_index.items():
        for fact in facts:
            if contained_lemma in expand_contained(fact, lexicon, parser):
                out.append((container_lemma, fact.relation))
                break
    return out


def _concept_matches_fact_container(
    concept_lemma: str, fact: ContainmentFact,
    lexicon: Lexicon, parser: MorphParser,
) -> bool:
    """True iff concept_lemma satisfies fact's container side."""
    concept = lexicon.concepts.get(concept_lemma)
    if concept is None:
        return False
    if fact.container is not None:
        return fact.container == concept_lemma
    if fact.container_pattern is not None:
        return _concept_matches_intrinsic(
            concept, fact.container_pattern, lexicon, parser)
    return False


def _concept_matches_fact_contained(
    concept_lemma: str, fact: ContainmentFact,
    lexicon: Lexicon, parser: MorphParser,
) -> bool:
    """True iff concept_lemma satisfies fact's contained side."""
    concept = lexicon.concepts.get(concept_lemma)
    if concept is None:
        return False
    if fact.contained is not None:
        target = fact.contained
        if target == concept_lemma:
            return True
        if lexicon.types.known(target):
            return lexicon.types.is_subtype(concept.entity_type, target)
        return False
    if fact.contained_pattern is not None:
        return _concept_matches_intrinsic(
            concept, fact.contained_pattern, lexicon, parser)
    return False


def required_fact_violations(
    container_lemma: str, contained_lemma: str, relation: str,
    containment_index: dict[str, list[ContainmentFact]],
    lexicon: Lexicon, parser: MorphParser | None = None,
) -> list[ContainmentFact]:
    """Return required ContainmentFact entries violated by this
    (contained, relation, container) triple. Empty list = all
    applicable requirements satisfied.

    Two requirement kinds:
      - Pattern requirement (default): "if contained matches
        contained_pattern, container MUST match container_pattern."
        Composes by AND.
      - Slot-overlap requirement (`slot_overlap`): "for each named
        slot, contained's values must intersect container's values
        for that slot, or one side must lack the slot entirely."
        Used for terrain (fish→water, vehicle→matching ground) where
        we want set-intersection, not value-equality."""
    parser = parser or DefaultMorphParser()
    violations: list[ContainmentFact] = []
    contained_concept = lexicon.concepts.get(contained_lemma)
    container_concept = lexicon.concepts.get(container_lemma)
    # Walk lexicon.containment directly — covers both pattern-anchored
    # required facts AND universal slot-overlap facts (which have no
    # container_pattern and so don't appear in `containment_index`).
    for fact in lexicon.containment:
        if not fact.required:
            continue
        if fact.relation != relation:
            continue
        # Optional gates: if patterns are set, both sides must
        # match before the requirement applies. Empty patterns
        # mean "applies to all".
        if (fact.contained is not None
                or fact.contained_pattern is not None):
            if not _concept_matches_fact_contained(
                    contained_lemma, fact, lexicon, parser):
                continue
        if (fact.container is not None
                or fact.container_pattern is not None):
            if not _concept_matches_fact_container(
                    container_lemma, fact, lexicon, parser):
                violations.append(fact)
                continue
        # Slot-overlap check: for each named slot, vacuously
        # satisfied if either side lacks values; otherwise must
        # intersect. Only run if the fact opted into it.
        if fact.slot_overlap and contained_concept and container_concept:
            bad = False
            for slot in fact.slot_overlap:
                cv = set(contained_concept.properties.get(slot, []))
                bv = set(container_concept.properties.get(slot, []))
                if cv and bv and not (cv & bv):
                    bad = True
                    break
            if bad:
                violations.append(fact)
    return violations


def containment_relations_for(
    container_lemma: str, contained_lemma: str,
    containment_index: dict[str, list[ContainmentFact]],
    lexicon: Lexicon, parser: MorphParser | None = None,
) -> set[str]:
    """All relations declared between this specific container and
    contained pair. Used by `Trace.assert_relation` to validate that a
    relation assertion is permitted by the containment registry: e.g.
    `en(tablo, sofo)` is rejected if no entry in containment.jsonl
    permits a tablo to be `en` a sofo. Returns a set so the same pair
    can have multiple relations declared (rare but legal — `tablo`
    could be both `en` and `sur` something)."""
    parser = parser or DefaultMorphParser()
    out: set[str] = set()
    for fact in containment_index.get(container_lemma, []):
        if contained_lemma in expand_contained(fact, lexicon, parser):
            out.add(fact.relation)
    return out


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
