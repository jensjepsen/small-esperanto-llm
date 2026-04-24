"""Two-phase fixed-point engine.

Each cycle:
  1. Run derivations to internal fixed point. Clear the derived layer
     and re-evaluate every derivation rule, materializing implied
     properties — but only when an asserted value isn't already in
     place (asserted wins over derived). Repeat until a sub-cycle
     produces no new derivations.
  2. Run causal rules over pending events. Each rule looks at events
     that haven't yet triggered it (per-(rule, event) memoization),
     enumerates bindings, and applies effects in order.
  3. If anything changed in step 2, return to step 1.

Convergence: per-(rule, event) memoization plus per-event-id dedup on
emitted events bounds the work. The asserted state grows monotonically
during a run; the derived state is rebuilt fresh each cycle.

Rule-construction validation runs at module load when rules are passed
to `validate_rules`, but the constructors themselves call into the
validator when given a slot registry, so import-time failure is
opt-in via `validate_rules(module, lex)`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..causal import EntityInstance, Event, Trace, make_event
from ..loader import Lexicon
from .effects import (
    AddRelation, Change, CreateEntity, DestroyEntity, Effect, Emit,
    RemoveRelation,
)
from .implications import Implication, PropertyImplication
from .patterns import (
    EventPattern, Pattern, Var,
)
from .unifier import (
    DerivedState, MatchContext, enumerate_bindings, resolve,
)


# ---------------------------- rules --------------------------------

@dataclass
class Rule:
    """A causal rule: trigger on `when` events, filter via `given`,
    apply `then` effects sequentially per binding."""
    name: str
    when: Pattern
    then: tuple[Effect, ...]
    given: tuple[Pattern, ...] = ()


@dataclass
class Derivation:
    """A derivation rule: `when` matches entities, `implies` produces
    derived properties. Re-evaluated every cycle."""
    name: str
    when: Pattern
    implies: tuple[Implication, ...]
    given: tuple[Pattern, ...] = ()


# --------------------------- builders ------------------------------

def rule(
    when: Pattern, then: Effect | list[Effect] | tuple[Effect, ...],
    given: list[Pattern] | tuple[Pattern, ...] = (),
    *, name: str | None = None,
) -> Rule:
    """Construct a causal rule. `when` must be exactly one event pattern;
    `then` is one effect or a sequence of effects; `given` is an
    optional conjunction of additional clauses."""
    if not isinstance(when, EventPattern):
        raise TypeError(
            "rule(when=...): causal rule when must be an event(...) pattern, "
            f"got {type(when).__name__}")
    then_t = (then,) if isinstance(then, Effect) else tuple(then)
    given_t = tuple(given)
    rule_name = name or "<unnamed_rule>"
    _validate_rule(rule_name, when, then_t, given_t)
    return Rule(rule_name, when, then_t, given_t)


def derive(
    when: Pattern, implies: Implication | list[Implication] | tuple[Implication, ...],
    given: list[Pattern] | tuple[Pattern, ...] = (),
    *, name: str | None = None,
) -> Derivation:
    """Construct a derivation rule. `when` must NOT contain any event
    patterns (derivations hold over state, not events)."""
    if _contains_event_pattern(when):
        raise TypeError(
            "derive(when=...): derivation when must not contain event(); "
            "derivations hold over state, not events.")
    for clause in given:
        if _contains_event_pattern(clause):
            raise TypeError(
                "derive(given=...): derivation given clauses cannot "
                "contain event() patterns.")
    implies_t = (implies,) if isinstance(implies, Implication) else tuple(implies)
    given_t = tuple(given)
    rule_name = name or "<unnamed_derivation>"
    _validate_derivation(rule_name, when, implies_t, given_t)
    return Derivation(rule_name, when, implies_t, given_t)


# --------------------- construction-time validation ----------------

def _contains_event_pattern(pattern: Pattern) -> bool:
    if isinstance(pattern, EventPattern):
        return True
    for child_attr in ("left", "right", "inner", "to_", "where"):
        child = getattr(pattern, child_attr, None)
        if isinstance(child, Pattern) and _contains_event_pattern(child):
            return True
    for d_attr in ("role_patterns", "arg_patterns"):
        d = getattr(pattern, d_attr, None)
        if isinstance(d, dict):
            for v in d.values():
                if isinstance(v, Pattern) and _contains_event_pattern(v):
                    return True
    return False


def _validate_rule(
    name: str, when: Pattern, then: tuple[Effect, ...],
    given: tuple[Pattern, ...],
) -> None:
    bound: set[Var] = set(when.variables())
    for g in given:
        bound |= g.variables()
    for eff in then:
        unbound = eff.reads() - bound
        if unbound:
            names = ", ".join(
                f"${v.name}" for v in sorted(unbound, key=lambda v: v.name))
            raise ValueError(
                f"rule {name!r}: effect {type(eff).__name__} reads "
                f"unbound variables: {names}. Variables must be bound "
                f"in `when`, `given`, or by an earlier effect.")
        # Writes (e.g. create_entity's as_var) become available for
        # subsequent effects.
        bound |= eff.writes()


def _validate_derivation(
    name: str, when: Pattern, implies: tuple[Implication, ...],
    given: tuple[Pattern, ...],
) -> None:
    bound: set[Var] = set(when.variables())
    for g in given:
        bound |= g.variables()
    for imp in implies:
        unbound = imp.reads() - bound
        if unbound:
            names = ", ".join(f"${v.name}" for v in sorted(unbound, key=lambda v: v.name))
            raise ValueError(
                f"derivation {name!r}: implication "
                f"{type(imp).__name__} reads unbound variables: {names}. "
                f"Variables must be bound in `when` or `given`.")


def validate_against_lexicon(
    rules_or_derivations: list[Rule | Derivation], lex: Lexicon,
) -> None:
    """Optional second-pass validation: check that all slot names,
    relation names, and concept lemmas referenced in patterns and
    implications actually exist in the lexicon. Run separately because
    rules are constructed at module load (before lex is available)."""
    slot_names = set(lex.slots.keys())
    rel_names = set(lex.relations.keys())
    for r in rules_or_derivations:
        _check_pattern_against_lex(r.name, r.when, slot_names, rel_names)
        for g in r.given:
            _check_pattern_against_lex(r.name, g, slot_names, rel_names)
        if isinstance(r, Derivation):
            for imp in r.implies:
                if isinstance(imp, PropertyImplication):
                    if imp.slot not in slot_names:
                        raise ValueError(
                            f"derivation {r.name!r}: property() refers to "
                            f"unknown slot {imp.slot!r}.")


def _check_pattern_against_lex(
    rule_name: str, pattern: Pattern, slots: set[str], rels: set[str],
) -> None:
    from .patterns import (
        AndPattern, CausedByPattern, ClosurePattern, EntityPattern,
        NotPattern, OrPattern, PastEventPattern, RelPattern,
    )
    if isinstance(pattern, EntityPattern):
        for k in pattern.constraints:
            if k in ("type", "has_suffix"):
                continue
            if k not in slots:
                raise ValueError(
                    f"rule {rule_name!r}: entity() refers to unknown slot {k!r}.")
    elif isinstance(pattern, RelPattern):
        if pattern.relation not in rels:
            raise ValueError(
                f"rule {rule_name!r}: rel() refers to unknown relation "
                f"{pattern.relation!r}.")
        for v in pattern.arg_patterns.values():
            _check_pattern_against_lex(rule_name, v, slots, rels)
    elif isinstance(pattern, ClosurePattern):
        for r in pattern.relations:
            if r not in rels:
                raise ValueError(
                    f"rule {rule_name!r}: closure() includes unknown "
                    f"relation {r!r}.")
        _check_pattern_against_lex(rule_name, pattern.to_, slots, rels)
        if pattern.where is not None:
            _check_pattern_against_lex(rule_name, pattern.where, slots, rels)
    elif isinstance(pattern, (AndPattern, OrPattern)):
        _check_pattern_against_lex(rule_name, pattern.left, slots, rels)
        _check_pattern_against_lex(rule_name, pattern.right, slots, rels)
    elif isinstance(pattern, NotPattern):
        _check_pattern_against_lex(rule_name, pattern.inner, slots, rels)
    elif isinstance(pattern, (EventPattern, PastEventPattern, CausedByPattern)):
        for v in pattern.role_patterns.values():
            _check_pattern_against_lex(rule_name, v, slots, rels)


# --------------------------- engine --------------------------------

@dataclass
class _CreatedEntities:
    """Counter for auto-generating ids when CreateEntity has no
    explicit id."""
    n: int = 0

    def next_id(self, concept: str) -> str:
        self.n += 1
        return f"{concept}_dsl_{self.n}"


def run_dsl(
    trace: Trace, rules: list[Rule], derivations: list[Derivation],
    lexicon: Lexicon, *, max_cycles: int = 50,
) -> int:
    """Run the engine to fixed point. Returns the number of outer
    cycles taken. Raises if convergence isn't reached within
    `max_cycles`."""
    derived = DerivedState()
    counter = _CreatedEntities()
    # Memoize (rule, event, exact-binding) firings: a rule fires once
    # per matching binding (per the spec). New bindings — opened by
    # derivations re-materializing or by new entities/relations — get
    # to fire freely.
    fired_bindings: set[tuple[str, str, frozenset]] = set()
    for cycle in range(1, max_cycles + 1):
        _run_derivations_to_fixed_point(
            trace, derivations, lexicon, derived)
        changed = _run_causal_phase(
            trace, rules, lexicon, derived, fired_bindings, counter)
        if not changed:
            return cycle
    raise RuntimeError(
        f"DSL engine did not converge within {max_cycles} cycles")


def _run_derivations_to_fixed_point(
    trace: Trace, derivations: list[Derivation],
    lexicon: Lexicon, derived: DerivedState,
    *, max_subcycles: int = 50,
) -> None:
    """Re-materialize the derived layer. Cleared first so reversed
    preconditions cause derived properties to disappear automatically.
    Inner loop iterates until no new derivations fire — handles
    chained derivations (A's output is B's input)."""
    derived.clear()
    ctx = MatchContext(
        trace=trace, lexicon=lexicon, derived=derived, focus_event=None)
    for _ in range(max_subcycles):
        changed = False
        for d in derivations:
            for bindings in enumerate_bindings(d.when, d.given, ctx):
                for imp in d.implies:
                    if isinstance(imp, PropertyImplication):
                        eid = bindings.get(imp.entity)
                        if eid is None:
                            continue
                        # Asserted wins over derived.
                        asserted = trace.property_at(
                            eid, imp.slot, len(trace.events))
                        if asserted is not None:
                            continue
                        value = resolve(imp.value, bindings)
                        if derived.set(eid, imp.slot, value):
                            changed = True
        if not changed:
            return
    raise RuntimeError(
        f"derivation phase did not converge within {max_subcycles} subcycles")


def _run_causal_phase(
    trace: Trace, rules: list[Rule], lexicon: Lexicon,
    derived: DerivedState, fired_bindings: set[tuple[str, str, frozenset]],
    counter: _CreatedEntities,
) -> bool:
    """Fire each rule on each pending event, once per matching binding.
    Returns True if any new event/entity/relation was added."""
    changed = False
    for r in rules:
        # when is an EventPattern (validated at construction). Early-
        # skip on action mismatch to avoid pointless enumeration.
        assert isinstance(r.when, EventPattern)
        for ev in list(trace.events):
            if ev.action != r.when.action:
                continue
            ctx = MatchContext(
                trace=trace, lexicon=lexicon, derived=derived,
                focus_event=ev)
            # Materialize bindings before applying effects — effects
            # can mutate `trace.entities`/`trace.relations` (e.g.
            # create_entity, add_relation), which would crash the
            # still-running enumeration generator with "dictionary
            # changed size during iteration".
            for bindings in list(enumerate_bindings(r.when, r.given, ctx)):
                key = (r.name, ev.id, frozenset(bindings.items()))
                if key in fired_bindings:
                    continue
                fired_bindings.add(key)
                if _apply_effects(r, bindings, trace, lexicon, counter, ev):
                    changed = True
    return changed


def _apply_effects(
    r: Rule, bindings: dict[Var, Any], trace: Trace,
    lexicon: Lexicon, counter: _CreatedEntities, cause_event: Event,
) -> bool:
    """Apply a rule's effects sequentially. Returns True if any new
    state was added.

    Effects share an extending bindings dict — `create_entity(as_var=S)`
    binds S so the next effect can use it.
    """
    b = dict(bindings)
    changed = False
    for eff in r.then:
        if isinstance(eff, Emit):
            roles = {k: resolve(v, b) for k, v in eff.role_vars.items()}
            pc = {
                (resolve(ent_or_id, b), slot): resolve(value, b)
                for (ent_or_id, slot), value in eff.property_changes.items()
            }
            pos = len(trace.events)
            new_ev = make_event(
                eff.action, roles=roles, caused_by=[cause_event.id],
                property_changes=pc if pc else None,
                trace_position=pos)
            if trace.add_event(new_ev):
                changed = True
        elif isinstance(eff, CreateEntity):
            concept_lemma = resolve(eff.concept, b)
            concept = lexicon.concepts.get(concept_lemma)
            if concept is None:
                continue
            if eff.entity_id is not None:
                eid = eff.entity_id
            elif eff.id_from is not None:
                from_value = resolve(eff.id_from, b)
                eid = f"{concept_lemma}_from_{from_value}"
            else:
                eid = counter.next_id(concept_lemma)
            if eid in trace.entities:
                b[eff.as_var] = eid
                continue
            ent = EntityInstance(
                id=eid,
                concept_lemma=concept_lemma,
                entity_type=concept.entity_type,
                properties={k: list(v) for k, v in concept.properties.items()},
                created_at_event=len(trace.events),
            )
            trace.entities[eid] = ent
            b[eff.as_var] = eid
            changed = True
        elif isinstance(eff, DestroyEntity):
            target = resolve(eff.target, b)
            ent = trace.entities.get(target)
            if ent is None or ent.destroyed_at_event is not None:
                continue
            ent.destroyed_at_event = len(trace.events) - 1
            changed = True
        elif isinstance(eff, Change):
            target = resolve(eff.entity, b)
            ent = trace.entities.get(target)
            if ent is None:
                continue
            # Asserted change: stored as a property_change on a
            # zero-effect synthetic event so the trace's event-calculus
            # reads pick it up. Could equivalently mutate ent.properties,
            # but we keep mutations off the entity per the v2 model.
            new_ev = make_event(
                "_change", roles={"theme": target},
                caused_by=[cause_event.id],
                property_changes={(target, eff.slot): eff.value})
            if trace.add_event(new_ev):
                changed = True
        elif isinstance(eff, AddRelation):
            args = tuple(resolve(a, b) for a in eff.args)
            existing = any(
                rl.relation == eff.relation and tuple(rl.args) == args
                for rl in trace.relations)
            if not existing:
                trace.assert_relation(eff.relation, args, lexicon)
                changed = True
        elif isinstance(eff, RemoveRelation):
            args = tuple(resolve(a, b) for a in eff.args)
            new_rels = [
                rl for rl in trace.relations
                if not (rl.relation == eff.relation
                        and tuple(rl.args) == args)
            ]
            if len(new_rels) != len(trace.relations):
                trace.relations = new_rels
                changed = True
    return changed


# ----------------------- module introspection ---------------------

def collect_rules(module) -> tuple[list[Rule], list[Derivation]]:
    """Pull all module-level Rule and Derivation values from a module,
    using the order they appear in the module's namespace. Used by
    callers who want to drive the engine from a rules.py-style module
    without naming each rule."""
    rules: list[Rule] = []
    derivations: list[Derivation] = []
    for name, value in vars(module).items():
        if isinstance(value, Rule):
            if value.name == "<unnamed_rule>":
                value.name = name
            rules.append(value)
        elif isinstance(value, Derivation):
            if value.name == "<unnamed_derivation>":
                value.name = name
            derivations.append(value)
    return rules, derivations
