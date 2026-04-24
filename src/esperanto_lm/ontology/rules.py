"""Retired.

The old imperative rule set used to live here — `fragile_falls_breaks`,
`hungry_eats_sated`, the factory `make_use_instrument`, and friends —
each a plain `(trace, t) -> list[Event]` callable dispatched by
`causal.run_to_fixed_point`.

Phases 1–4 of the migration moved every rule to the declarative DSL
(`esperanto_lm.ontology.dsl`). Phase 5 retired the imperative engine
and this module alongside it. The DSL replacement:

    from esperanto_lm.ontology.dsl import run_dsl
    from esperanto_lm.ontology.dsl.rules import (
        DEFAULT_DSL_RULES, make_use_instrument_rules,
    )
    rules = DEFAULT_DSL_RULES + make_use_instrument_rules(lex)
    run_dsl(trace, rules, derivations, lex)

Kept empty rather than deleted so explicit imports (`from
esperanto_lm.ontology.rules import ...`) fail loudly pointing at this
note instead of a mystery ImportError on a missing file.
"""
