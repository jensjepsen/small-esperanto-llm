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
    prefer_scene_p: float = 1.0,
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
                 action=None, role_name=None, *,
                 prefer_scene: bool | None = None):
        # `prefer_scene` resolution:
        #   - explicit True/False from caller: honored
        #   - None (default): coin flip against `prefer_scene_p` —
        #     1.0 (default) is "always prefer scene" (BP behavior);
        #     0.0 is "always natural habitat"; 0.5 mixes
        if prefer_scene is None:
            prefer_scene = rng.random() < prefer_scene_p
        # Budget exhausted: bail before any expensive work
        # (`_concepts_matching_role` walks the full concept dict, and
        # `_place_respecting_containment` recursively walks containment).
        # The planner can call us many times per plan; without this
        # short-circuit, post-budget calls keep paying the walk cost
        # just to return None at the end.
        if state["spawned"] >= budget:
            return None
        from .seeders import (
            _candidate_weights, _concepts_matching_role,
            _place_respecting_containment,
        )

        # Concept pick. Verb-aware bias when caller supplied action +
        # role_name; uniform random otherwise. The role spec's
        # type+properties already constrain candidates to concepts
        # compatible with the verb (animate for sidi.agent, lock_capable
        # for ŝlosi.theme). The slot's `applies_to` adds an entity-type
        # check downstream. We deliberately DON'T require the concept
        # to declare the effect slot in `c.properties` — that gate
        # would reject every host whose state slot is lifted from a
        # part by a derivation (pordo inheriting lock_state from
        # seruro), and every concept whose `varies` slot value lands
        # at instance creation (lock_state itself).
        eff = None
        if action is not None and role_name is not None:
            for e in action.effects:
                if e.target_role == role_name:
                    eff = e
                    break
        candidates = _concepts_matching_role(lex_arg, role_spec)
        if not candidates:
            return None
        weights = None
        if action is not None and role_name is not None:
            weights = _candidate_weights(
                candidates, role_name, action, lex_arg)
        # Weighted shuffle: draw the entire candidate list ordered by
        # weight, so we can retry the next-best concept when the first
        # pick has no valid containment host (seruro for ŝlosi.theme,
        # motoro for ŝalti.theme — part-only concepts that satisfy the
        # role spec but can't be placed standalone). Without this
        # retry, a single unlucky `rng.choice` aborts the whole
        # spawn; with it, we naturally fall back to the host concept.
        if weights is not None:
            order = []
            pool = list(zip(candidates, weights))
            while pool:
                pick = rng.choices(pool, weights=[w for _, w in pool],
                                    k=1)[0]
                order.append(pick[0])
                pool.remove(pick)
        else:
            order = list(candidates)
            rng.shuffle(order)

        eid = None
        for concept in order:
            # Materialize via recursive containment placement. This
            # may spawn intermediate containers (a pomarbo for a pomo,
            # a kuirejo for a forno) as siblings of the scene.
            # `prefer_scene=False` makes the spawner skip BOTH Tier 1
            # (preferred=scene) and Tier 2's fallback to scene, so the
            # concept lands in its natural containment habitat
            # (purigilo in laborejo, forno in kuirejo, najlilo in
            # laborejo, ...). The forward planner's pre-spawn uses this
            # to recreate the spatial spread BP's lazy spawning produced
            # as a side effect of being called in different trace states.
            eid = _place_respecting_containment(
                trace, lex_arg, scene_id, concept, rng,
                preferred_id=scene_id if prefer_scene else None,
                avoid=None if prefer_scene else frozenset({scene_id}))
            if eid is not None:
                break
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
