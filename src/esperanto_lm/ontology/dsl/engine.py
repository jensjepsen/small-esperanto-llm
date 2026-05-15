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
    AddRelation, Change, ConsumeOne, CreateEntity, DestroyEntity, Effect, Emit,
    ForEach, RemoveRelation, TransferN,
)
from .implications import (
    CategoryImplication, Implication, PartImplication, PropertyImplication,
    RelationImplication,
)
from .patterns import (
    EventPattern, Pattern, Var,
)
from .unifier import (
    DerivedState, MatchContext, enumerate_bindings, resolve,
)

import os as _os

# Codegen toggle — set ESPLLM_DSL_COMPILE=0 to force the interpreted
# match path (used for parity testing and bisecting suspected codegen
# bugs). Read once at import time.
_DSL_COMPILE_ENABLED = _os.environ.get("ESPLLM_DSL_COMPILE", "1") != "0"


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
    concept_names = set(lex.concepts.keys())
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
                elif isinstance(imp, RelationImplication):
                    if imp.name not in rel_names:
                        raise ValueError(
                            f"derivation {r.name!r}: relation() refers to "
                            f"unknown relation {imp.name!r}.")
                elif isinstance(imp, PartImplication):
                    if imp.part_concept not in concept_names:
                        raise ValueError(
                            f"derivation {r.name!r}: part() refers to "
                            f"unknown concept {imp.part_concept!r}.")
                    if imp.relation not in rel_names:
                        raise ValueError(
                            f"derivation {r.name!r}: part() refers to "
                            f"unknown relation {imp.relation!r}.")


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


def compute_derived_state(
    trace: Trace, derivations: list[Derivation], lexicon: Lexicon,
) -> DerivedState:
    """Materialize the derived layer for the given trace and return it.
    Pure read-only — does not fire causal rules or mutate the trace.

    Use this when consumers outside the engine cycle (planners,
    inspectors, debug tools) need to see derived properties like
    `locomotion=walk` that are computed by Derivations rather than
    baked into `entity.properties`."""
    derived = DerivedState()
    _run_derivations_to_fixed_point(trace, derivations, lexicon, derived)
    return derived


_DEP_CACHE: dict[int, tuple[frozenset, frozenset]] = {}


def _derivation_dependencies(derivation):
    """Static analysis: what relations and property-slots does this
    derivation read in its when/given patterns? Used by the RETE-lite
    delta-tracking loop to skip re-evaluating derivations whose
    dependencies didn't change in the prior iteration. Cached by
    id(derivation) — derivation patterns are constructed once at
    module load and reused."""
    cached = _DEP_CACHE.get(id(derivation))
    if cached is not None:
        return cached
    from .patterns import (
        AndPattern, BindPattern, EntityPattern, NotPattern,
        RelPattern,
    )
    rels: set[str] = set()
    slots: set[str] = set()

    def walk(p):
        if p is None:
            return
        if isinstance(p, RelPattern):
            rels.add(p.relation)
            for arg_pat in p.arg_patterns.values():
                walk(arg_pat)
        elif isinstance(p, EntityPattern):
            for k in p.constraints:
                if k not in ("type", "concept", "has_suffix"):
                    slots.add(k)
        elif isinstance(p, AndPattern):
            walk(p.left)
            walk(p.right)
        elif isinstance(p, NotPattern):
            walk(p.inner)
        elif isinstance(p, BindPattern):
            walk(getattr(p, "inner", None))
        # OrPattern, ClosurePattern fall through — caller treats them
        # conservatively (no narrowing), which is correct.

    walk(derivation.when)
    for g in derivation.given:
        walk(g)
    result = (frozenset(rels), frozenset(slots))
    _DEP_CACHE[id(derivation)] = result
    return result


def _run_derivations_to_fixed_point(
    trace: Trace, derivations: list[Derivation],
    lexicon: Lexicon, derived: DerivedState,
    *, max_subcycles: int = 50,
) -> None:
    """Re-materialize the derived layer. Cleared first so reversed
    preconditions cause derived properties to disappear automatically.
    Inner loop iterates until no new derivations fire — handles
    chained derivations (A's output is B's input).

    RETE-lite optimization: between iterations, only re-evaluate
    derivations whose dependency tags overlap with what the previous
    iteration newly added to derived state. Pure correctness no-op
    relative to the naive "re-run everything until stable" loop —
    a derivation whose inputs didn't change can't produce new
    output. Saves ~70% of re-evaluation work on typical scenes."""
    from .compile import get_compiled_deriv_enum

    derived.clear()
    ctx = MatchContext(
        trace=trace, lexicon=lexicon, derived=derived, focus_event=None)
    deps = [_derivation_dependencies(d) for d in derivations]
    # Pre-resolve compiled enums (or None for fall-back) once per call.
    compiled_enums = (
        [get_compiled_deriv_enum(d) for d in derivations]
        if _DSL_COMPILE_ENABLED else [None] * len(derivations))
    # Iteration 1: evaluate all derivations.
    pending = list(range(len(derivations)))
    for cycle in range(max_subcycles):
        delta_rels: set[str] = set()
        delta_slots: set[str] = set()
        for i in pending:
            d = derivations[i]
            enum_fn = compiled_enums[i]
            binding_iter = (
                enum_fn(ctx) if enum_fn is not None
                else enumerate_bindings(d.when, d.given, ctx))
            for bindings in binding_iter:
                for imp in d.implies:
                    if isinstance(imp, PropertyImplication):
                        eid = bindings.get(imp.entity)
                        if eid is None:
                            continue
                        slot_def = lexicon.slots.get(imp.slot)
                        scalar = (slot_def.scalar
                                  if slot_def is not None else True)
                        asserted = trace.property_at(
                            eid, imp.slot, len(trace.events))
                        value = resolve(imp.value, bindings)
                        if scalar:
                            if asserted is not None:
                                continue
                            # First-write-wins for scalar slots within
                            # a derivation cycle: if another derivation
                            # already set this (entity, slot), skip so
                            # specific derivations (e.g. sittable →
                            # sidanta) aren't clobbered by defaults
                            # (animate → staranta) that fire later in
                            # the same iteration.
                            if derived.get(eid, imp.slot) is not None:
                                continue
                        else:
                            asserted_list = (
                                asserted if isinstance(asserted, list)
                                else [asserted] if asserted is not None
                                else [])
                            if value in asserted_list:
                                continue
                        if derived.set(eid, imp.slot, value,
                                        scalar=scalar):
                            delta_slots.add(imp.slot)
                    elif isinstance(imp, RelationImplication):
                        resolved_args = []
                        ok = True
                        for arg in imp.args:
                            v = resolve(arg, bindings)
                            if v is None:
                                ok = False
                                break
                            resolved_args.append(v)
                        if not ok:
                            continue
                        if derived.add_relation(
                                imp.name, tuple(resolved_args)):
                            delta_rels.add(imp.name)
                    elif isinstance(imp, CategoryImplication):
                        eid = bindings.get(imp.entity)
                        if eid is None:
                            continue
                        # Categories don't drive other derivations
                        # (renderer-only state), so no delta tracking
                        # needed — re-evaluation hooks read deltas in
                        # rels/slots and treat categories as quiet.
                        derived.add_category(eid, imp.category)
        if not delta_rels and not delta_slots:
            return
        # Next iteration: only derivations whose deps overlap with
        # the newly-added slots/relations need re-evaluation.
        pending = [
            i for i, (rels, slots) in enumerate(deps)
            if (rels & delta_rels) or (slots & delta_slots)
        ]
        if not pending:
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
    from .compile import get_compiled_enum

    changed = False
    for r in rules:
        # when is an EventPattern (validated at construction). Early-
        # skip on action mismatch to avoid pointless enumeration.
        assert isinstance(r.when, EventPattern)
        compiled_enum = (
            get_compiled_enum(r) if _DSL_COMPILE_ENABLED else None)
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
            # Capture each binding by copy at materialization time —
            # BindPattern.apply_to_value uses mutate-and-restore on a
            # shared dict, so a bare `list(...)` would yield N refs to
            # the same (now-empty) dict. Compiled enums yield fresh
            # dicts already; the dict(b) is then redundant but cheap.
            if compiled_enum is not None:
                binding_iter = compiled_enum(ev, ctx)
            else:
                binding_iter = enumerate_bindings(r.when, r.given, ctx)
            for bindings in [dict(b) for b in binding_iter]:
                # List-valued bindings (variadic event roles like
                # fari.parts) are unhashable in the dedup frozenset;
                # tuple-coerce just for the key. EntityInstance values
                # (compiled enum can resolve to instances when given
                # clauses are present) are reduced to their .id since
                # only the identity matters for dedup. The bindings
                # dict itself keeps the original values so ForEach can
                # mutate-iterate and effects can read attributes.
                def _k(v):
                    if isinstance(v, list):
                        return tuple(_k(x) for x in v)
                    if isinstance(v, EntityInstance):
                        return v.id
                    return v
                key = (r.name, ev.id, frozenset(
                    (k, _k(v)) for k, v in bindings.items()))
                if key in fired_bindings:
                    continue
                fired_bindings.add(key)
                if _apply_effects(r, bindings, trace, lexicon, counter, ev):
                    changed = True
    return changed


_UNSET = object()


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
    return _apply_effect_list(
        r.then, b, trace, lexicon, counter, cause_event)


def _apply_effect_list(
    effects, b: dict[Var, Any], trace: Trace,
    lexicon: Lexicon, counter: _CreatedEntities, cause_event: Event,
) -> bool:
    """Inner dispatcher — applies an arbitrary list of effects with a
    given (shared) bindings dict. Extracted from `_apply_effects` so
    ForEach can recursively dispatch its inner effect list per
    iteration."""
    changed = False
    for eff in effects:
        if isinstance(eff, ForEach):
            # Iterate the list binding, running inner effects with
            # item_var locally bound. Mutate-and-restore on the shared
            # bindings dict (same pattern BindPattern uses to avoid
            # copying bindings per iteration).
            items = b.get(eff.list_var, ())
            if isinstance(items, str):
                items = (items,)
            elif not isinstance(items, (list, tuple)):
                items = (items,)
            prior = b.get(eff.item_var, _UNSET)
            try:
                for item in items:
                    b[eff.item_var] = item
                    if _apply_effect_list(
                            eff.effects, b, trace, lexicon, counter,
                            cause_event):
                        changed = True
            finally:
                if prior is _UNSET:
                    b.pop(eff.item_var, None)
                else:
                    b[eff.item_var] = prior
            continue
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
            elif eff.id_parts is not None:
                parts = [
                    str(resolve(p, b)) if isinstance(p, Var) else str(p)
                    for p in eff.id_parts
                ]
                eid = f"{concept_lemma}_from_{'_'.join(parts)}"
            elif eff.id_from is not None:
                from_value = resolve(eff.id_from, b)
                eid = f"{concept_lemma}_from_{from_value}"
            else:
                eid = counter.next_id(concept_lemma)
            if eid in trace.entities:
                b[eff.as_var] = eid
                continue
            props = {k: list(v) for k, v in concept.properties.items()}
            if eff.initial_properties:
                for slot, val in eff.initial_properties.items():
                    resolved_val = resolve(val, b) if isinstance(val, Var) else val
                    props[slot] = [resolved_val]
            ent = EntityInstance(
                id=eid,
                concept_lemma=concept_lemma,
                entity_type=concept.entity_type,
                properties=props,
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
        elif isinstance(eff, ConsumeOne):
            target = resolve(eff.target, b)
            ent = trace.entities.get(target)
            if ent is None or ent.destroyed_at_event is not None:
                continue
            current_raw = trace.property_at(
                target, "count", len(trace.events))
            # Read consumption quantity from the firing event. Defaults
            # to 1 (one bite, one transfer); larger values come from
            # explicit "Maria ate 3 apples" events.
            qty = getattr(cause_event, "quantity", 1)
            # No count slot → single-unit theme; destroy outright
            # regardless of quantity (you can't eat 3 of something
            # that has no notion of count).
            if current_raw is None:
                ent.destroyed_at_event = len(trace.events) - 1
                changed = True
                continue
            if isinstance(current_raw, list):
                current_raw = current_raw[0] if current_raw else "1"
            try:
                current = int(current_raw)
            except (TypeError, ValueError):
                ent.destroyed_at_event = len(trace.events) - 1
                changed = True
                continue
            new = current - qty
            if new <= 0:
                ent.destroyed_at_event = len(trace.events) - 1
                changed = True
            else:
                new_ev = make_event(
                    "_change", roles={"theme": target},
                    caused_by=[cause_event.id],
                    property_changes={(target, "count"): str(new)})
                if trace.add_event(new_ev):
                    changed = True
        elif isinstance(eff, TransferN):
            source = resolve(eff.source, b)
            target = resolve(eff.target, b)
            src_ent = trace.entities.get(source)
            if src_ent is None or src_ent.destroyed_at_event is not None:
                continue
            # Find current owner (if any) of source via havi.
            prior_owner = None
            for rl in trace.relations:
                if rl.relation == "havi" and len(rl.args) == 2 \
                        and rl.args[1] == source:
                    prior_owner = rl.args[0]
                    break
            qty = getattr(cause_event, "quantity", 1)
            current_raw = trace.property_at(
                source, "count", len(trace.events))
            # Single-unit theme (no count slot) → full ownership swap.
            single_unit = current_raw is None
            if not single_unit:
                if isinstance(current_raw, list):
                    current_raw = current_raw[0] if current_raw else "1"
                try:
                    current = int(current_raw)
                except (TypeError, ValueError):
                    current = 1
                    single_unit = True
            # Full transfer when single-unit OR qty covers the stack.
            if single_unit or qty >= current:
                if prior_owner is not None and prior_owner != target:
                    new_rels = [
                        rl for rl in trace.relations
                        if not (rl.relation == "havi"
                                and tuple(rl.args) == (prior_owner, source))
                    ]
                    if len(new_rels) != len(trace.relations):
                        trace.relations = new_rels
                        changed = True
                already = any(
                    rl.relation == "havi"
                    and tuple(rl.args) == (target, source)
                    for rl in trace.relations)
                if not already:
                    # Relation schema may reject (location/person/part
                    # exclusions on havi.theme) — sampler-fired events
                    # with invalid theme bindings shouldn't crash the
                    # engine. Treat as no-op for that rule firing.
                    try:
                        trace.assert_relation(
                            "havi", (target, source), lexicon)
                        changed = True
                    except (ValueError, KeyError):
                        pass
                continue
            # Partial split: source keeps current-qty units; target gets
            # a new stack of qty units.
            new_count = current - qty
            new_eid = (
                f"{src_ent.concept_lemma}_from_{cause_event.id[:12]}")
            if new_eid not in trace.entities:
                concept = lexicon.concepts.get(src_ent.concept_lemma)
                props = (
                    {k: list(v) for k, v in concept.properties.items()}
                    if concept is not None
                    else {k: list(v) for k, v in src_ent.properties.items()}
                )
                props["count"] = [str(qty)]
                new_ent = EntityInstance(
                    id=new_eid,
                    concept_lemma=src_ent.concept_lemma,
                    entity_type=src_ent.entity_type,
                    properties=props,
                    created_at_event=len(trace.events),
                )
                trace.entities[new_eid] = new_ent
                changed = True
            # Decrement source's count via synthetic _change.
            decrement_ev = make_event(
                "_change", roles={"theme": source},
                caused_by=[cause_event.id],
                property_changes={(source, "count"): str(new_count)})
            if trace.add_event(decrement_ev):
                changed = True
            # Add havi(target, new_eid). Source's existing havi stays.
            already = any(
                rl.relation == "havi"
                and tuple(rl.args) == (target, new_eid)
                for rl in trace.relations)
            if not already:
                try:
                    trace.assert_relation(
                        "havi", (target, new_eid), lexicon)
                    changed = True
                except (ValueError, KeyError):
                    pass
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
                try:
                    trace.assert_relation(eff.relation, args, lexicon)
                    changed = True
                except (ValueError, KeyError):
                    pass
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
