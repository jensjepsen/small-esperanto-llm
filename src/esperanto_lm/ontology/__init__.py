"""Esperanto lexical-semantic ontology + causal rule engine."""
from .containment import (
    containment_relation_for,
    reachable_from,
    resolve_containment,
)
from .causal import (
    EntityInstance,
    Event,
    RelationAssertion,
    Rule,
    Trace,
    effect_changes,
    make_event,
    run_to_fixed_point,
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
from .rules import (
    DEFAULT_RULES,
    broken_container_releases_contents,
    carried_fragile_falls_when_carrier_falls,
    container_falls_contents_fall,
    fire_spreads_to_adjacent_flammables,
    fragile_falls_breaks,
    hungry_eats_sated,
    make_broken_fragile_creates_shards,
    make_use_instrument,
    make_wet_liquid_container_tips,
    person_walks_on_hazard_falls,
)
from .schemas import (
    Action,
    Affix,
    Concept,
    ContainmentFact,
    ContainmentPattern,
    Effect,
    PropertySlot,
    Quality,
    Relation,
    RoleSpec,
)
from .types import TypeSpine

__all__ = [
    # schemas
    "Action", "Affix", "Concept", "ContainmentFact", "ContainmentPattern",
    "Effect", "PropertySlot", "Quality", "Relation", "RoleSpec",
    # loader
    "FUNCTIONAL_SIGNATURE", "Lexicon", "load_lexicon",
    "resolve_signature", "signature_effects",
    # morph
    "DefaultMorphParser", "MorphParse", "MorphParser", "StubMorphParser",
    # causal
    "EntityInstance", "Event", "RelationAssertion", "Rule", "Trace",
    "effect_changes", "make_event", "run_to_fixed_point",
    # rules
    "DEFAULT_RULES", "make_use_instrument",
    "make_broken_fragile_creates_shards", "make_wet_liquid_container_tips",
    "fragile_falls_breaks", "hungry_eats_sated",
    "container_falls_contents_fall", "broken_container_releases_contents",
    "person_walks_on_hazard_falls",
    "carried_fragile_falls_when_carrier_falls",
    "fire_spreads_to_adjacent_flammables",
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
