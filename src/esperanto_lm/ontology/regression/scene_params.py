"""Scene parameters — named knobs that bias chain shape.

The seeder samples a `SceneParameters` per scene, then `apply_scene_
parameters` materializes the sampled choices into the trace before
the planner runs. This formalizes the previously-scattered randomness
in `regress_for_verb` (away-placement flip, conditional-gate forcing,
chain-dependency scaffolding, lamp seeding) as a single schema both
the verb-first and goal-first samplers consume.

Layer 2 of the three-layer redesign — see project_seeder_redesign
memory. The goal-first sampler picks WHAT goal to drive; this layer
picks HOW the scene around it is shaped; the planner with the spawner
realizes the chain.

Each parameter is a small categorical or boolean with explicit
probabilities. Sampling produces a concrete config; applying mutates
the trace. Both stages are pure-from-rng so reproducible across runs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class SceneParameters:
    """Sampled scene-shape choices. None of these change WHICH goal
    drives the scene; they bias the chain length and the auxiliary
    setup the planner has to discover."""
    # Where the effect-target entity lives: in the scene (co-located
    # with the actor; simple verb fires with no locomotion) or away
    # (forces the planner to subgoal samloke via iri/kuri/veturi).
    # Default 0.85 away matches the previous regress_for_verb
    # behavior — chain richness over the bare-execute case.
    theme_in_away: bool = True

    # Force any IfPropertyPrecondition's "if_property=if_value" to
    # hold AND "then_property" to a non-target value, so the gate
    # FIRES and forces the planner to subgoal a producer for the
    # gate's consequent. e.g. pordo's lock_capable=yes is forced;
    # lock_state is set to non-malŝlosita so malfermi must chain
    # through malŝlosi first.
    force_conditional_gates: bool = True

    # Scaffold the action's preconditions: pre-place required
    # instruments in the away location so the planner can subgoal
    # `havi(agent, instrument)` via iri→preni. Without this, tool-
    # using verbs (ŝlosi, lavi, kuiri) often dead-end on missing
    # instruments.
    seed_chain_dependencies: bool = True

    # Place an active lamp in indoor scenes when the verb's chain
    # might involve vidi — the perception chain needs illuminated
    # locations, derivable only from an active lamp.
    seed_indoor_lamp: bool = True


def sample_scene_parameters(rng: random.Random) -> SceneParameters:
    """Draw one scene config. All params are independent per-scene
    flips; entanglement (e.g. "if instrument is away then lamp
    matters") is left for the apply pass to resolve. Probabilities
    chosen to match the previous regress_for_verb defaults so the
    refactor introduces no behavior change in isolation."""
    return SceneParameters(
        theme_in_away=rng.random() < 0.85,
        force_conditional_gates=True,
        seed_chain_dependencies=True,
        seed_indoor_lamp=True,
    )


def apply_scene_parameters(
    trace, scene_id: str, away_id: str,
    action, role_eids: dict, lex, rng: random.Random,
    params: SceneParameters,
) -> bool:
    """Materialize sampled parameters into the trace.

    Idempotently places each role-bound entity (effect target goes
    to `away_id` when `theme_in_away` AND away differs from scene),
    sets the effect-slot to a non-target value, then layers on the
    optional seeders (conditional gates, chain dependencies, indoor
    lamp). The role bindings + scene/away are expected to be
    pre-populated by the caller; this function completes the scene
    shape that previously lived inline in `regress_for_verb`.

    Returns True on success, False when any role-entity has no valid
    placement under containment.jsonl — caller should abort the scene
    so the planner never sees an orphan. The previous silent-pass
    behavior produced orphans that the planner couldn't reason about
    (no `apud`/`en` producer for `(artifact, location)`), accounting
    for ~half of the failed-drive cases. Containment is now the
    single source of truth: a concept that has no valid host is
    excluded from the scene altogether.
    """
    from .seeders import (
        _action_might_need_light, _force_conditional_gates,
        _place_respecting_containment, _route_through_container,
        _seed_chain_dependencies, _seed_indoor_lamp,
    )

    eff = action.effects[0] if action.effects else None

    # Place each role-entity. Effect target gets away placement when
    # the param flips and there's a distinct away. Other roles stay
    # in scene_id; the planner's spawner handles role-fillers it
    # needs to materialize.
    for role_name, eid in role_eids.items():
        if (eff is not None
                and role_name == eff.target_role
                and away_id != scene_id
                and params.theme_in_away):
            placement = away_id
        else:
            placement = scene_id
        ent = trace.entities.get(eid)
        concept_lemma = ent.concept_lemma if ent is not None else None
        if _route_through_container(
                trace, eid, concept_lemma, placement, lex, rng):
            continue
        # Strict placement via containment.jsonl. Tries the preferred
        # location first, then any in-trace entity, then materializes
        # a valid container as a sibling of the scene. Returns None
        # iff containment offers no host — surface the gap as scene
        # rejection rather than silently orphaning the entity.
        if concept_lemma is None or _place_respecting_containment(
                trace, lex, scene_id, concept_lemma, rng,
                preferred_id=placement, existing_eid=eid) is None:
            return False

    # Effect-target non-target initial state.
    if eff is not None:
        target_eid = role_eids.get(eff.target_role)
        if target_eid:
            slot_def = lex.slots.get(eff.property)
            if slot_def is not None and slot_def.vocabulary:
                non_target = [v for v in slot_def.vocabulary if v != eff.value]
                if non_target:
                    trace.entities[target_eid].set_property(
                        eff.property, rng.choice(non_target))

    if params.force_conditional_gates:
        _force_conditional_gates(trace, action, role_eids, lex, rng)

    if params.seed_chain_dependencies:
        _seed_chain_dependencies(
            trace, action, role_eids, scene_id, lex, rng, away_id=away_id)

    if params.seed_indoor_lamp and _action_might_need_light(action):
        _seed_indoor_lamp(trace, scene_id, away_id, lex, rng)

    return True
