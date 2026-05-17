"""Declarative matching DSL for causal + derivation rules.

Two rule forms share the same pattern vocabulary:

  - `rule(when=event(...), then=..., given=...)` — causal. Triggers on
    events, produces new events/entities/state changes. Forward-
    chaining reactive layer.

  - `derive(when=entity(...), implies=property(...), given=...)` —
    derivation. Holds over state, materializes implied properties.
    Re-evaluated every cycle, so reversing a precondition reverses
    the derivation.

Pattern primitives: `event`, `entity`, `rel`, `closure`,
`has_concept_field`. Logical composition: `&`, `|`, `~`.

Variable style: walrus-declared Vars. First use inlines the
declaration; subsequent uses are bare.

    T = var("T")              # standalone declaration, or
    bind(T := var("T"))       # inline at first use

Engine: `run_dsl(trace, rules, derivations, lexicon)`. Each outer
cycle runs derivations to internal fixed point, then fires causal
rules on pending events, and repeats until a cycle produces no
changes.
"""
from .effects import (
    AddRelation, Change, ConsumeOne, CreateEntity, DestroyEntity, Effect, Emit,
    ForEach, RemoveRelation, TransferN,
    add_relation, change, consume_one, create_entity, destroy_entity, emit,
    for_each, remove_relation, transfer_n,
)
from .engine import (
    Derivation, Rule, collect_rules, compute_derived_state, derive, rule,
    run_dsl, validate_against_lexicon,
)
from .implications import (
    CategoryImplication, Implication, PartImplication, PropertyImplication,
    RelationImplication,
    category, part, property, relation,
)
from .patterns import (
    AndPattern, BindPattern, CausedByPattern, ClosurePattern, Compare,
    EntityPattern, EventPattern, HasConceptFieldPattern, NotPattern, OrPattern,
    PastEventPattern, Pattern, RelPattern, Var, VarList, VarProp,
    bind, bind_list, caused_by, closure, entity, event, has_concept_field,
    past_event, rel, var, var_list,
)
from .unifier import DerivedState, MatchContext, enumerate_bindings, resolve

__all__ = [
    # patterns + constructors
    "AndPattern", "BindPattern", "CausedByPattern", "ClosurePattern",
    "Compare", "EntityPattern", "EventPattern", "HasConceptFieldPattern",
    "NotPattern", "OrPattern", "PastEventPattern", "Pattern", "RelPattern",
    "Var", "VarList", "VarProp",
    "bind", "bind_list", "caused_by", "closure", "entity", "event",
    "has_concept_field", "past_event", "rel", "var", "var_list",
    # effects
    "AddRelation", "Change", "ConsumeOne", "CreateEntity", "DestroyEntity",
    "Effect", "Emit", "ForEach", "RemoveRelation", "TransferN",
    "add_relation", "change", "consume_one", "create_entity",
    "destroy_entity", "emit", "for_each", "remove_relation", "transfer_n",
    # implications
    "CategoryImplication", "Implication", "PartImplication",
    "PropertyImplication", "RelationImplication",
    "category", "part", "property", "relation",
    # engine
    "Derivation", "Rule", "collect_rules", "compute_derived_state",
    "derive", "rule", "run_dsl",
    "validate_against_lexicon",
    # unifier
    "DerivedState", "MatchContext", "enumerate_bindings", "resolve",
]
