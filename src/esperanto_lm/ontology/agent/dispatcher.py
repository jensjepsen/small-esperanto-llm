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

from typing import Optional

from .planner import (
    _BudgetExceeded, _DERIVED_CACHE, _ENTITY_RESOLVER, _PLANNER_RNG,
    _SIM_BUDGET, _SIM_CACHE, _SimulationBudget,
    _cached_compute_derived_state,
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
                    rng=None, simulation_budget=5000,
                    entity_resolver=None):
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
    from .planner import _FAILURE_REASON, _TERMINAL_DEPTH
    kind = drive[0]
    token = _PLANNER_RNG.set(rng)
    btoken = _SIM_BUDGET.set(_SimulationBudget(simulation_budget))
    rtoken = (_ENTITY_RESOLVER.set(entity_resolver)
              if entity_resolver is not None else None)
    # Reset failure reason at entry. Don't reset at exit — leave the
    # final reason readable by the caller via
    # `get_planner_failure_reason()`. Persistence is fine because the
    # contextvar lives in the caller's context.
    _FAILURE_REASON.set(None)
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
        _FAILURE_REASON.set((_TERMINAL_DEPTH, ("budget",)))
        return None
    finally:
        _PLANNER_RNG.reset(token)
        _SIM_BUDGET.reset(btoken)
        if rtoken is not None:
            _ENTITY_RESOLVER.reset(rtoken)



FOLLOWUP_P: float = 0.5
FOLLOWUP_DECAY: float = 0.5
FOLLOWUP_MAX_PHASES: int = 3


def execute_drive(
    drive, t, lex, rules, derivations, *,
    scene_id, rng,
    max_states: int = 1200, max_plan_length: int = 16,
    spawn_budget: int = 6, prefer_scene_p: float = 1.0,
) -> Optional[list]:
    """Single runner: plan + execute the seeded drive into `t`, then
    with probability `FOLLOWUP_P` re-enter `regress_for_goal` on the
    post-plan trace to chain a follow-up drive on top. Loops up to
    `FOLLOWUP_MAX_PHASES` total, with `FOLLOWUP_DECAY` taper on the
    per-step probability. Followup mode biases actor + role-fillers
    toward in-scene entities, so chained drives naturally re-use the
    just-spawned NPCs / items and sometimes hand the protagonist
    role to a different person.

    Construct (event_fire on fari) phase-2 plans get fari excluded
    so they don't re-fire the same build mid-chain.

    Returns the concatenated plan-step list across all phases, or
    None if the seeded drive couldn't be planned. Late imports keep
    agent.__init__ from pulling regression at module-load time."""
    from .forward_planner import plan_for_goal
    from .planner import _step_to_event
    from ..dsl import run_dsl
    from ..regression.spawner import make_spawner

    def _run_phase(d, *, extra_exclude=None):
        """Plan + execute one drive into t. Returns list of plan
        steps (possibly empty if no plan), or None on planner error."""
        seeder_exclude = (
            getattr(t, "_planner_exclude_verbs", None) or set())
        exclude = set(seeder_exclude)
        if extra_exclude:
            exclude |= set(extra_exclude)
        sp = make_spawner(
            scene_id, lex, rng,
            budget=spawn_budget, prefer_scene_p=prefer_scene_p)
        p = plan_for_goal(
            d, t, lex, rules, derivations,
            max_states=max_states, max_plan_length=max_plan_length,
            entity_resolver=sp, rng=rng,
            exclude_verbs=exclude or None)
        if not p:
            return None
        for step in p:
            event = _step_to_event(step, lex)
            t.events.append(event)
            run_dsl(t, rules, derivations, lex)
        return list(p)

    first = _run_phase(drive)
    if first is None:
        return None
    full_plan = list(first)

    # Construct's phase-2 must exclude fari — see _construct_goal_scene
    # for the parts-cascade rationale; phase-2 re-firing fari would
    # rebuild the same stub mid-chain.
    last_was_construct = (
        drive[0] == "event_fire" and len(drive) >= 3
        and drive[2] == "fari")

    # Chain on itself: with probability p (tapered by DECAY per step),
    # re-sample a goal on the existing trace and run it. Up to
    # MAX_PHASES total events.
    from ..regression.goal_sampler import regress_for_goal
    p_followup = FOLLOWUP_P
    for _ in range(FOLLOWUP_MAX_PHASES - 1):
        if rng.random() >= p_followup:
            break
        followup_sample = regress_for_goal(
            lex, rng, rules,
            existing_trace=t, existing_scene_id=scene_id)
        if followup_sample is None:
            break
        _ft, _fscene, fdrive = followup_sample
        try:
            fplan = _run_phase(
                fdrive,
                extra_exclude={"fari"} if last_was_construct else None)
        except Exception:
            break
        if not fplan:
            break
        full_plan.extend(fplan)
        last_was_construct = (
            fdrive[0] == "event_fire" and len(fdrive) >= 3
            and fdrive[2] == "fari")
        p_followup *= FOLLOWUP_DECAY
    return full_plan


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
