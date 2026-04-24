"""Esperanto lexical-semantic ontology + DSL rule engine.

Phase 5 retired the imperative `run_to_fixed_point` engine and the
hand-coded rule callables. The DSL under `esperanto_lm.ontology.dsl`
is now the one and only runtime — see its package docstring for the
pattern vocabulary, derivation layer, and fixed-point loop.
"""
from .containment import (
    containment_relation_for,
    reachable_from,
    resolve_containment,
)
from .causal import (
    EntityInstance,
    Event,
    RelationAssertion,
    Trace,
    effect_changes,
    make_event,
)
from .loader import (
    FUNCTIONAL_SIGNATURE,
    Lexicon,
    load_lexicon,
    resolve_signature,
    signature_effects,
)
from .morph import DefaultMorphParser, MorphParse, MorphParser, StubMorphParser
from .realize import realize_trace
from .sampler import (
    PERSON_NAMES,
    Recipe,
    RoleBinding,
    SceneInfo,
    prune_unused_persons,
    recipes_for,
    sample_scene,
)
from .schemas import (
    Action,
    Affix,
    Concept,
    ContainmentFact,
    ContainmentPattern,
    Effect,
    PropertySlot,
    Relation,
    RoleSpec,
)
from .types import TypeSpine

__all__ = [
    # schemas
    "Action", "Affix", "Concept", "ContainmentFact", "ContainmentPattern",
    "Effect", "PropertySlot", "Relation", "RoleSpec",
    # loader
    "FUNCTIONAL_SIGNATURE", "Lexicon", "load_lexicon",
    "resolve_signature", "signature_effects",
    # morph
    "DefaultMorphParser", "MorphParse", "MorphParser", "StubMorphParser",
    # causal: state primitives only; rule execution lives in `dsl`.
    "EntityInstance", "Event", "RelationAssertion", "Trace",
    "effect_changes", "make_event",
    # realize
    "realize_trace",
    # sampler
    "PERSON_NAMES", "Recipe", "RoleBinding", "SceneInfo",
    "sample_scene", "prune_unused_persons", "recipes_for",
    # types
    "TypeSpine",
    # containment
    "containment_relation_for", "reachable_from", "resolve_containment",
]
