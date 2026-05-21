"""Shared precondition-evaluation helpers.

`_has_relation` and `_pc_holds` need to behave identically across
the backward planner (`agent.planner`), the forward planner
(`agent.forward_planner`), and the forward sampler
(`scripts/forward_sampler.py`). Drift between independent
implementations has bitten us:
  - Sampler's `_has_relation` missed symmetric-relation swaps
    (apud(B,A) when apud(A,B) was asserted).
  - Sampler's `_pc_holds` rejected RelationPrecondition with
    unbound roles, while the planner treated them as vacuous —
    relevant for verbs with optional roles whose precondition
    is "only when bound".
  - Sampler's IfPropertyPrecondition read raw `ent.properties`,
    missing mid-trace state changes that the planner caught via
    `_entity_property_values`.

This module is the single source of truth. The planner / sampler
expose thin wrappers around these functions for back-compat
(`_has_relation`, `_pc_holds` are re-imported where they used
to live)."""
from __future__ import annotations

from typing import Optional

# Hoisted to module-level to avoid the per-call import overhead
# inside `_pc_holds` — `sys.modules` lookup + attribute resolution
# adds up over the planner's millions-of-calls hot path. Safe
# despite the agent.planner → agent.precondition_eval cycle
# because in planner.py the `from .precondition_eval import ...`
# happens AFTER `_entity_property_values` is defined; loading
# precondition_eval here while planner is mid-load still
# resolves `_entity_property_values` correctly. Same for the
# schema types — they have no transitive dependency on this
# module, so the import is a clean one-way reference.
from ..schemas import (
    HasPropertyPrecondition, IfPropertyPrecondition, MatchPrecondition,
    NotPropertyPrecondition, NotRelationPrecondition,
    OrPrecondition, RelationPrecondition,
)
from .planner import _entity_property_values


def _has_relation(
    relation: str,
    args: tuple,
    trace,
    derived=None,
    lex=None,
) -> bool:
    """True if `relation(*args)` currently holds in asserted state
    or in `derived`. For arity-2 symmetric relations declared in
    the lex (apud, samloke, frato, edzo, amiko, najbaro), also
    checks the swapped argument pair — the engine stores one
    canonical direction at assert-time and doesn't materialize the
    swap, so without this check the same fact would be visible
    from one ordering but not the other."""
    target = tuple(args)
    for r in trace.relations:
        if r.relation == relation and tuple(r.args) == target:
            return True
    if derived is not None and derived.has_relation(relation, target):
        return True
    is_symmetric = (
        lex is not None
        and relation in lex.relations
        and getattr(lex.relations[relation], "symmetric", False)
        and lex.relations[relation].arity == 2)
    if is_symmetric and len(target) == 2 and target[0] != target[1]:
        swapped = (target[1], target[0])
        for r in trace.relations:
            if r.relation == relation and tuple(r.args) == swapped:
                return True
        if derived is not None and derived.has_relation(relation, swapped):
            return True
    return False


def _pc_holds(pc, roles, trace, derived, lex) -> bool:
    """Evaluate a single precondition against the current trace +
    derived state with the given role bindings. Recursive over
    OrPrecondition.

    Semantics aligned to the engine's behavior:
      - Unbound roles in a positive RelationPrecondition →
        vacuous (the precondition only fires when the role is
        bound; matches the planner's
        `_ground_facts_from_template` skip-on-None).
      - Property reads consult `_entity_property_values` (the
        union of `trace.property_at(eid, slot, pos)`,
        `ent.properties`, and `derived.properties`) — catches
        mid-trace state changes (`malŝalti` having flipped
        `power_state` to neaktiva, `iri` having moved an
        entity, runtime derivations like is_part) that raw
        `ent.properties` misses.
      - Negative preconditions (NotPropertyPrecondition,
        NotRelationPrecondition) treat unbound roles as
        vacuously-passing (you can't violate a constraint on a
        role you didn't bind).
      - Unknown precondition kinds vacuously hold (forward-
        compat for new shapes).
    """
    if isinstance(pc, RelationPrecondition):
        eids = tuple(roles.get(r) for r in pc.roles)
        if any(e is None for e in eids):
            return True   # unbound role — vacuous
        return _has_relation(pc.rel, eids, trace, derived, lex)
    if isinstance(pc, IfPropertyPrecondition):
        eid = roles.get(pc.role)
        if eid is None:
            return True
        ent = trace.entities.get(eid)
        if ent is None:
            return False
        gate = _entity_property_values(
            ent, pc.if_property, trace, derived)
        if pc.if_value not in gate:
            return True   # gate not active — vacuous
        then_vals = _entity_property_values(
            ent, pc.then_property, trace, derived)
        return pc.then_value in then_vals
    if isinstance(pc, MatchPrecondition):
        ea = trace.entities.get(roles.get(pc.role_a))
        eb = trace.entities.get(roles.get(pc.role_b))
        if ea is None or eb is None:
            return False
        va = set(ea.properties.get(pc.slot_a, []))
        vb = set(eb.properties.get(pc.slot_b, []))
        return bool(va & vb)
    if isinstance(pc, HasPropertyPrecondition):
        eid = roles.get(pc.role)
        if eid is None:
            return False
        ent = trace.entities.get(eid)
        if ent is None:
            return False
        return pc.value in _entity_property_values(
            ent, pc.property, trace, derived)
    if isinstance(pc, NotPropertyPrecondition):
        eid = roles.get(pc.role)
        if eid is None:
            return True   # unbound — vacuous
        ent = trace.entities.get(eid)
        if ent is None:
            return False
        return pc.value not in _entity_property_values(
            ent, pc.property, trace, derived)
    if isinstance(pc, NotRelationPrecondition):
        eids = tuple(roles.get(r) for r in pc.roles)
        if any(e is None for e in eids):
            return True   # unbound — vacuous
        return not _has_relation(pc.rel, eids, trace, derived, lex)
    if isinstance(pc, OrPrecondition):
        return any(_pc_holds(alt, roles, trace, derived, lex)
                   for alt in pc.alternatives)
    return True   # unknown kind — vacuous


__all__ = ["_has_relation", "_pc_holds"]
