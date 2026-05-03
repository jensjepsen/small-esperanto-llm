"""Drive → planner dispatch.

`plan_for_drive(drive, ...)` is the single entry point: a drive tuple
goes in, an action sequence comes out (or `None` if the planner can't
satisfy it). Per-drive-kind dispatch routes to the right planner
primitive (`plan_to_achieve` for self/entity slots, `plan_to_reach_count`
for count drives, etc.) and threads a per-call simulation budget +
RNG via contextvars defined in `planner`.

`_drive_summary(drive)` produces a human-readable label used by the
coverage harness for prose listings.
"""
from __future__ import annotations

from .planner import (
    _BudgetExceeded, _DERIVED_CACHE, _PLANNER_RNG, _SIM_BUDGET,
    _SIM_CACHE, _SimulationBudget, _cached_compute_derived_state,
    _count_owned, plan_to_achieve, plan_to_establish_relation,
    plan_to_reach_count,
)


def _dedupe_adjacent_steps(plan):
    """Drop adjacent (verb, roles) pairs that are identical. Symptom
    treatment for a planner-redundancy bug: independent subgoal
    paths sometimes both emit the same en-establishing event,
    yielding chains like `eniri(X, salono) → eniri(X, salono)` where
    the second is a no-op (the rule's `given` clause fails because
    the first eniri already removed the apud). The duplicate event
    still lands in the trace and the realizer dutifully prints
    'eniris la salonon kaj eniris la salonon'. Adjacent dedup
    cleans up the prose without masking distant non-no-op repeats."""
    if not plan:
        return plan
    out = [plan[0]]
    for step in plan[1:]:
        if step != out[-1]:
            out.append(step)
    return out


def plan_for_drive(drive, t, lex, rules, derivations, *, max_depth=8,
                    rng=None, simulation_budget=5000):
    """Dispatch one drive to the right planner entry. Returns the
    plan or None. Computes derived state once. Doesn't fire — caller
    is responsible for executing the plan against the trace.

    `rng` (optional): when given, the planner shuffles candidate verbs
    at each enumeration site so different runs surface different
    chain shapes — e.g. rakonti vs respondi vs montri for konas
    goals. None preserves the deterministic enumeration order.

    `simulation_budget`: cap on `_simulate_from_scratch` calls per
    drive. Returns None if exhausted — bounds wall-clock latency on
    pathological searches without rejecting plans that have a
    reachable solution. Cache hits are free; only fresh simulations
    consume the budget. Default 5000 covers all observed plan sizes
    with ~5× headroom; lower for stricter latency guarantees."""
    kind = drive[0]
    token = _PLANNER_RNG.set(rng)
    budget = _SimulationBudget(simulation_budget)
    btoken = _SIM_BUDGET.set(budget)
    _SIM_CACHE.clear()
    _DERIVED_CACHE.clear()
    derived = _cached_compute_derived_state(t, derivations, lex)
    plan = None
    try:
        if kind == "self_slot":
            _, actor, slot, value = drive
            plan = plan_to_achieve(
                actor, slot, value, actor, t, lex, rules,
                derived=derived, derivations=derivations,
                max_depth=max_depth)
        elif kind == "entity_slot":
            _, actor, target, slot, value = drive
            plan = plan_to_achieve(
                target, slot, value, actor, t, lex, rules,
                derived=derived, derivations=derivations,
                max_depth=max_depth)
        elif kind == "location":
            _, actor, loc = drive
            plan = plan_to_establish_relation(
                "en", (actor, loc), actor, t, lex, rules,
                derived=derived, derivations=derivations,
                max_depth=max_depth)
        elif kind == "possession":
            _, actor, item = drive
            plan = plan_to_establish_relation(
                "havi", (actor, item), actor, t, lex, rules,
                derived=derived, derivations=derivations,
                max_depth=max_depth)
        elif kind == "knowledge":
            _, actor, knower, fakto_id = drive
            plan = plan_to_establish_relation(
                "konas", (knower, fakto_id), actor, t, lex, rules,
                derived=derived, derivations=derivations,
                max_depth=max_depth)
        elif kind == "wearing":
            _, actor, garment = drive
            plan = plan_to_establish_relation(
                "vestita", (actor, garment), actor, t, lex, rules,
                derived=derived, derivations=derivations,
                max_depth=max_depth)
        elif kind == "count":
            _, actor, concept_lemma, target_count = drive
            plan = plan_to_reach_count(
                actor, concept_lemma, target_count, t, lex, rules,
                derived=derived, derivations=derivations,
                max_depth=max_depth)
        elif kind == "give_count":
            _, donor, recipient, concept_lemma, target_count = drive
            plan = plan_to_reach_count(
                recipient, concept_lemma, target_count, t, lex, rules,
                planner_actor_id=donor,
                derived=derived, derivations=derivations,
                max_depth=max_depth)
        elif kind == "more_than":
            _, actor, concept_lemma, reference_id = drive
            # Resolve target from current state: one more than the
            # reference's count. If reference holds nothing, target=1
            # (one is "more than zero").
            ref_count = _count_owned(reference_id, concept_lemma, t)
            target_count = ref_count + 1
            plan = plan_to_reach_count(
                actor, concept_lemma, target_count, t, lex, rules,
                derived=derived, derivations=derivations,
                max_depth=max_depth)
        else:
            plan = None
        return _dedupe_adjacent_steps(plan) if plan else plan
    except _BudgetExceeded:
        # Planner ran out of simulation budget. Surface as a None plan
        # — caller (e.g. regression sampler) treats it as "this scene
        # couldn't be planned" and moves on. The exception keeps the
        # mid-search recursion from polluting the return path.
        return None
    finally:
        _PLANNER_RNG.reset(token)
        _SIM_BUDGET.reset(btoken)



def _drive_summary(drive):
    """Short human-readable form for the report."""
    kind = drive[0]
    if kind == "self_slot":
        return f"{drive[1]}.{drive[2]}={drive[3]}"
    if kind == "entity_slot":
        return f"{drive[1]} wants {drive[2]}.{drive[3]}={drive[4]}"
    if kind == "location":
        return f"{drive[1]} wants to be in {drive[2]}"
    if kind == "knowledge":
        actor, knower, fakto_id = drive[1], drive[2], drive[3]
        if actor == knower:
            return f"{actor} wants to know fact {fakto_id}"
        return f"{actor} wants {knower} to know fact {fakto_id}"
    if kind == "possession":
        return f"{drive[1]} wants to have {drive[2]}"
    if kind == "wearing":
        return f"{drive[1]} wants to be wearing {drive[2]}"
    if kind == "count":
        return f"{drive[1]} wants {drive[3]} {drive[2]}(s)"
    if kind == "give_count":
        return f"{drive[1]} wants {drive[2]} to have {drive[4]} {drive[3]}(s)"
    if kind == "more_than":
        return f"{drive[1]} wants more {drive[2]}(s) than {drive[3]}"
    return repr(drive)
