"""Agent simulation loop: tick the world forward, asking each animate
in turn for a plan.

`run_simulation` is the simplest possible runner — pick the first
animate that has a satisfiable multi-step plan, fire the whole plan,
repeat until quiescent or `max_ticks` reached. The agent commits to
its sequence; a more conservative loop would re-plan after each step.

Used by the forward-sampler / one-agent demo paths. The regression
coverage harness in `agent.coverage` calls into the dispatcher
directly without this loop.
"""
from __future__ import annotations

from ..dsl import compute_derived_state, run_dsl
from .planner import _step_to_event, plan_with_subgoals


def run_simulation(trace, lex, rules, derivations, max_ticks=8):
    """Tick loop. Each tick: find the first animate entity with a
    satisfiable plan (possibly multi-step), fire ALL steps of the plan
    in order, run the engine after each, repeat. Terminate when no
    agent can act or max_ticks reached.

    Multi-step plans count as a single tick — the agent commits to a
    sequence and executes it. (A more conservative loop would re-plan
    after each step; the current shape is fine for POC.)

    Derived state is recomputed per tick (cheap — pure function of
    trace + derivations) so role specs that gate on derived properties
    work transparently.
    """
    for tick in range(max_ticks):
        fired = False
        derived = compute_derived_state(trace, derivations, lex)
        for eid, ent in list(trace.entities.items()):
            if not lex.types.is_subtype(ent.entity_type, "animate"):
                continue
            if ent.destroyed_at_event is not None:
                continue
            plan = plan_with_subgoals(
                eid, trace, lex, rules, derived=derived)
            if not plan:
                continue
            for step in plan:
                event = _step_to_event(step, lex)
                trace.events.append(event)
                run_dsl(trace, rules, derivations, lex)
            fired = True
            break
        if not fired:
            return tick
    return max_ticks
