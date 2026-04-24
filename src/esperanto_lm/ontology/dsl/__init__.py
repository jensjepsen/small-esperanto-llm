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
    AddRelation, Change, CreateEntity, Effect, Emit, RemoveRelation,
    add_relation, change, create_entity, emit, remove_relation,
)
from .engine import (
    Derivation, Rule, collect_rules, derive, rule, run_dsl,
    validate_against_lexicon,
)
from .implications import (
    Implication, PropertyImplication, property,
)
from .patterns import (
    AndPattern, BindPattern, CausedByPattern, ClosurePattern, EntityPattern,
    EventPattern, HasConceptFieldPattern, NotPattern, OrPattern,
    PastEventPattern, Pattern, RelPattern, Var,
    bind, caused_by, closure, entity, event, has_concept_field, past_event,
    rel, var,
)
from .unifier import DerivedState, MatchContext, enumerate_bindings, resolve

__all__ = [
    # patterns + constructors
    "AndPattern", "BindPattern", "CausedByPattern", "ClosurePattern",
    "EntityPattern", "EventPattern", "HasConceptFieldPattern", "NotPattern",
    "OrPattern", "PastEventPattern", "Pattern", "RelPattern", "Var",
    "bind", "caused_by", "closure", "entity", "event", "has_concept_field",
    "past_event", "rel", "var",
    # effects
    "AddRelation", "Change", "CreateEntity", "Effect", "Emit", "RemoveRelation",
    "add_relation", "change", "create_entity", "emit", "remove_relation",
    # implications
    "Implication", "PropertyImplication", "property",
    # engine
    "Derivation", "Rule", "collect_rules", "derive", "rule", "run_dsl",
    "validate_against_lexicon",
    # unifier
    "DerivedState", "MatchContext", "enumerate_bindings", "resolve",
]
