"""Goal index — maps state-change goals to the verbs that achieve them.

A Goal is one of:
  ("property", slot, value)         — achieve `entity.slot = value`
  ("relation", rel_name, role_tuple) — achieve `rel(*role_args)`

For each verb, postconditions come from two sources:
  1. `Action.effects` — direct property changes the verb declares.
  2. DSL `Rule.then` clauses where `rule.when` is an `EventPattern`
     matching this verb. Rule-mediated `Change` and `AddRelation`
     effects also count — that's how effect-less verbs like vidi
     (which `vidi_learns_en` mutates konas/scias_lokon) and iri
     (which an en-changing rule mutates the agent's location)
     enter the index.

The goal-first regression sampler picks a goal weighted by chain-
richness of its producers, then chooses a verb to drive toward
it, then constructs the supporting scene. This subsumes the verb-
pool walk used by `regress_for_verb` (effect-bearing) plus the
dedicated `regress_for_movement` / `regress_for_knowledge_verb`
seeders (effect-less), unifying them under one goal-driven flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..dsl.effects import AddRelation, Change, CreateEntity
from ..dsl.engine import Rule
from ..dsl.introspect import _role_vars
from ..dsl.patterns import EventPattern, Var


# Sentinel role name used in `role_args` when a relation argument is
# bound to an entity the rule itself creates (via CreateEntity in the
# same `then`). The vidi/flari/audi pattern: rule creates a fakto and
# asserts konas(agent, fakto) — the fakto isn't a role binding but a
# freshly-spawned entity. Drives targeting this goal mean "agent ends
# up in <relation> with something the verb creates."
CREATED_ROLE = "<created>"


# Goal kinds are tagged tuples; the regression sampler dispatches
# on the tag.
PropertyGoal = tuple[str, str, str]      # ("property", slot, value)
RelationGoal = tuple[str, str, tuple[str, ...]]  # ("relation", rel, role_tuple)
ConstructGoal = tuple[str, str]          # ("construct", theme_concept)
Goal = PropertyGoal | RelationGoal | ConstructGoal


@dataclass(frozen=True)
class VerbPostcondition:
    """One thing a verb does. `role_args` is filled only for
    relation postconditions (and contains role-name strings); for
    property postconditions, `target_role` carries the role name
    whose entity gets the slot=value assignment."""
    kind: str                          # "property" or "relation"
    verb_lemma: str
    target_role: str | None            # property: role name; relation: None
    slot: str | None                   # property only
    value: Any | None                  # property only
    relation: str | None               # relation only
    role_args: tuple[str, ...] | None  # relation only


def verb_postconditions(verb_lemma: str, rules: list, lex) -> list[VerbPostcondition]:
    """All postconditions attributed to a verb — direct effects plus
    rule-mediated implications.

    A rule contributes iff its `when` is an `EventPattern(action=
    verb_lemma)`. Each `Change`/`AddRelation` in its `then` is
    translated into a postcondition by mapping its Var arguments
    back to role names via the event pattern's role_patterns.

    Postconditions whose vars don't map to a verb role (e.g. an
    intermediate var bound only in `given`) are skipped — they're
    not directly actionable as drive targets.
    """
    out: list[VerbPostcondition] = []

    # 1. Direct action effects.
    action = lex.actions.get(verb_lemma)
    if action is not None:
        for eff in action.effects:
            out.append(VerbPostcondition(
                kind="property", verb_lemma=verb_lemma,
                target_role=eff.target_role,
                slot=eff.property, value=eff.value,
                relation=None, role_args=None,
            ))

    # 2. Rule-mediated implications: walk DSL rules whose when matches.
    for rule in rules:
        when = rule.when
        if not isinstance(when, EventPattern):
            continue
        if when.action != verb_lemma:
            continue
        var_to_role = {id(v): name for name, v in _role_vars(when).items()}
        effects = rule.then if isinstance(rule.then, (list, tuple)) else [rule.then]
        # First pass: collect CreateEntity as_var vars so subsequent
        # AddRelation effects in the same `then` can refer to the
        # created entity via CREATED_ROLE.
        for eff in effects:
            if isinstance(eff, CreateEntity):
                v = getattr(eff, "as_var", None)
                if isinstance(v, Var):
                    var_to_role[id(v)] = CREATED_ROLE
        for eff in effects:
            if isinstance(eff, Change):
                if not isinstance(eff.entity, Var):
                    continue
                role_name = var_to_role.get(id(eff.entity))
                if role_name is None:
                    continue
                out.append(VerbPostcondition(
                    kind="property", verb_lemma=verb_lemma,
                    target_role=role_name,
                    slot=eff.slot, value=eff.value,
                    relation=None, role_args=None,
                ))
            elif isinstance(eff, AddRelation):
                # All args must be Vars that map to roles. If any
                # arg is a literal or unmapped Var, skip — we can't
                # express a clean Goal for it.
                role_args = []
                ok = True
                for a in eff.args:
                    if not isinstance(a, Var):
                        ok = False
                        break
                    rn = var_to_role.get(id(a))
                    if rn is None:
                        ok = False
                        break
                    role_args.append(rn)
                if not ok:
                    continue
                out.append(VerbPostcondition(
                    kind="relation", verb_lemma=verb_lemma,
                    target_role=None, slot=None, value=None,
                    relation=eff.relation,
                    role_args=tuple(role_args),
                ))
    # Dedup: multiple rules can contribute the same postcondition for
    # the same verb (vidi has vidi_learns_en, vidi_learns_sur,
    # vidi_learns_havi_owner — all asserting konas(agent, <created>)).
    # Preserve insertion order for stable downstream weighting.
    seen: set = set()
    deduped: list[VerbPostcondition] = []
    for pc in out:
        if pc not in seen:
            seen.add(pc)
            deduped.append(pc)
    return deduped


def build_goal_index(lex, rules) -> dict[Goal, list[str]]:
    """Reverse-map verb postconditions: Goal → [verbs that achieve it].

    For property postconditions, target_role is omitted from the Goal
    key — drive picking only cares about (slot, value); the role
    mapping is recovered later from `verb_postconditions` when the
    seeder builds the scene.

    Relation Goals keep their role_args shape so locomotion-style
    drives like `("relation", "en", ("agent", "destination"))`
    distinguish from `("relation", "en", ("theme", "container"))`
    if the same relation appears in different role configurations.
    """
    out: dict[Goal, list[str]] = {}
    for verb_lemma in lex.actions:
        for pc in verb_postconditions(verb_lemma, rules, lex):
            if pc.kind == "property":
                key: Goal = ("property", pc.slot, pc.value)
            else:
                key = ("relation", pc.relation, pc.role_args)
            out.setdefault(key, [])
            if verb_lemma not in out[key]:
                out[key].append(verb_lemma)
    # Construct goals: one entry per constructable concept, produced
    # by fari. Lets `regress_for_goal` pick construction the same way
    # it picks property/relation goals — weighted by producer count
    # (always 1 here, since fari is the only producer) and uniform
    # across constructables. Without this, construction would only
    # surface via a separate hardcoded gate in the sampler.
    if "fari" in lex.actions:
        for lemma, concept in lex.concepts.items():
            if "yes" not in concept.properties.get("constructable", ()):
                continue
            if not concept.parts:
                continue
            out[("construct", lemma)] = ["fari"]
    return out
