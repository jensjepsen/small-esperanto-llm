"""Spawn-on-demand resolver for the planner's role-binding callback.

The seeder calls `make_spawner(scene_id, lex, rng)` to get a closure
suitable for the planner's `_ENTITY_RESOLVER` ContextVar. When the
planner's `_find_role_filler` exhausts in-trace candidates, it invokes
the resolver with the unsatisfied role_spec — and, crucially, with
the calling `action` and `role_name`. That verb context lets the
spawner do the same setup work the verb-first seeder
(`regress_for_verb`) does inline: gate-biased concept picking,
non-target initial state on effect targets, conditional-gate
forcing on cross-role preconditions.

Placement uses `_place_respecting_containment` so spawned entities
land in a recursively-constructed sub-scene (a pomo → pomarbo →
outdoor location apud the scene), not just an arbitrary "away"
room.

This is the unified entity-materialization point in the three-layer
seeder redesign — see `project_seeder_redesign` memory.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def make_spawner(
    scene_id: str,
    lex,
    rng,
    *,
    budget: int = 6,
) -> Callable:
    """Return a resolver closure for the planner's _ENTITY_RESOLVER.

    Budget caps total spawns per planning session at 6; once
    exhausted, the resolver returns None and the planner falls back
    to failing the candidate.

    The closure is verb-aware via the `action` and `role_name`
    kwargs the planner passes per invocation. When set, it:
      - biases the role-matching concept pool toward those that
        trigger conditional preconditions (gate-aware weighting,
        the same `_candidate_weights` regress_for_verb uses);
      - if the role is the action's effect target, sets the
        spawned entity's effect-slot to a non-target value so the
        verb has work to do;
      - walks the action's `if_property` preconditions and forces
        the gate-firing state on the spawned entity when its role
        matches.

    Placement: `_place_respecting_containment` walks the containment
    graph recursively (fruit → tree → outdoor, food → kitchen, etc.),
    materializing intermediate containers as needed.
    """
    state = {"spawned": 0}

    def resolver(role_spec, trace, lex_arg, exclude,
                 action=None, role_name=None):
        if state["spawned"] >= budget:
            return None
        from .seeders import (
            _candidate_weights, _concepts_matching_role,
            _place_respecting_containment,
        )

        # Concept pick. Verb-aware bias when caller supplied action +
        # role_name; uniform random otherwise.
        candidates = _concepts_matching_role(lex_arg, role_spec)
        # When this is the effect target of `action`, require the
        # concept to declare the effect slot (pervasive slots
        # excepted — for those, every applies-to-type qualifies).
        eff = None
        if action is not None and role_name is not None:
            for e in action.effects:
                if e.target_role == role_name:
                    eff = e
                    break
        if eff is not None:
            slot_def = lex_arg.slots.get(eff.property)
            if slot_def is None or not getattr(slot_def, "pervasive", False):
                candidates = [c for c in candidates
                              if eff.property in lex_arg.concepts[c].properties]
        if not candidates:
            return None
        weights = None
        if action is not None and role_name is not None:
            weights = _candidate_weights(
                candidates, role_name, action, lex_arg)
        if weights is not None:
            concept = rng.choices(candidates, weights=weights, k=1)[0]
        else:
            concept = rng.choice(candidates)

        # Materialize via recursive containment placement. This may
        # spawn intermediate containers (a pomarbo for a pomo, a
        # kuirejo for a forno) as siblings of the scene.
        eid = _place_respecting_containment(
            trace, lex_arg, scene_id, concept, rng,
            preferred_id=scene_id)
        if eid is None:
            return None

        # Verb-aware setup: non-target initial state for effect targets,
        # if_property gate forcing on this role's preconditions.
        ent = trace.entities.get(eid)
        if ent is not None and action is not None and role_name is not None:
            if eff is not None:
                slot_def = lex_arg.slots.get(eff.property)
                if slot_def is not None and slot_def.vocabulary:
                    non_target = [v for v in slot_def.vocabulary
                                  if v != eff.value]
                    if non_target:
                        ent.set_property(eff.property, rng.choice(non_target))
            # Walk preconditions for if_property gates referencing this
            # role; force the gate to fire (antecedent holds) AND its
            # consequent to a non-target value so the planner subgoals
            # a producer.
            for pc in action.preconditions:
                if getattr(pc, "kind", None) != "if_property":
                    continue
                if pc.role != role_name:
                    continue
                # Force antecedent.
                ent.set_property(pc.if_property, pc.if_value)
                # Force consequent to a non-then value.
                then_slot_def = lex_arg.slots.get(pc.then_property)
                if then_slot_def is not None and then_slot_def.vocabulary:
                    other = [v for v in then_slot_def.vocabulary
                             if v != pc.then_value]
                    if other:
                        ent.set_property(
                            pc.then_property, rng.choice(other))

        state["spawned"] += 1
        return eid

    return resolver
