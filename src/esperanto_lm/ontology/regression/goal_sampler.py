"""Goal-first regression sampler.

Picks a goal from the lexicon-derived index, picks a verb that
achieves it, then hands off to the spawner-enabled planner with the
absolute minimum pre-built scene: just a scene location + the actor.
Everything else — the drive's theme, any instrument the verb wants,
intermediate containers, away locations — is materialized by the
spawner when the planner asks for it. The spawner's verb-aware
setup (gate-biased concept pick, non-target initial state, gate
forcing, recursive containment placement) does the work the
verb-first seeder used to do up front.

This is the user-visible payoff of the three-layer redesign — see
project_seeder_redesign memory. Keep `regress_for_verb` for
comparison; this is the goal-first path.
"""
from __future__ import annotations

import random
from typing import Optional

from ..causal import Trace
from .goals import build_goal_index


# Cache the index per lexicon — building it walks every action and
# rule once.
_GOAL_INDEX_CACHE: dict[int, dict] = {}


def _cached_goal_index(lex, rules):
    key = id(lex)
    cached = _GOAL_INDEX_CACHE.get(key)
    if cached is None:
        cached = build_goal_index(lex, rules)
        _GOAL_INDEX_CACHE[key] = cached
    return cached


def regress_for_goal(lex, rng: random.Random, rules) -> Optional[tuple]:
    """Pick a goal, pick a verb from its producers, build a minimal
    trace, return (trace, scene_id, drive). The spawner — invoked by
    the planner via the entity_resolver hook — materializes everything
    else with verb-aware setup.

    Two goal kinds:
      - Property goals: ("property", slot, value). Drive shape is
        `entity_slot`. The verb's effect mutates the target's slot.
      - CREATED-role relation goals: ("relation", rel, role_tuple)
        where role_tuple includes the `<created>` sentinel. Used for
        verbs whose effect is CreateEntity + AddRelation (skribi,
        vidi, aŭdi, flari, montri, krii, ludi, …). Drive shape is
        `event_fire`; the planner just has to fire the verb with the
        appropriate role bindings — the rule creates the entity.
    """
    from ..sampler import _add_entity_randomized, _ensure_world
    from .seeders import _concepts_matching_role
    from .goals import CREATED_ROLE

    index = _cached_goal_index(lex, rules)
    all_goals = []
    for g, verbs in index.items():
        if not verbs:
            continue
        if g[0] == "property":
            all_goals.append((g, verbs))
        elif g[0] == "relation":
            # Only include relation goals where some role is
            # `<created>` — those resolve to event_fire drives.
            _, _rel, role_args = g
            if CREATED_ROLE in role_args:
                all_goals.append((g, verbs))
    if not all_goals:
        return None
    # Weight by producer count.
    weights = [len(verbs) for _, verbs in all_goals]
    chosen_goal, producers = rng.choices(
        all_goals, weights=weights, k=1)[0]
    verb_lemma = rng.choice(producers)
    action = lex.actions.get(verb_lemma)
    if action is None:
        return None

    if chosen_goal[0] == "property":
        _, slot, target_value = chosen_goal
        if not action.effects:
            return None
        eff = next((e for e in action.effects
                    if e.property == slot and e.value == target_value),
                   action.effects[0])
    else:
        # event_fire path — no slot/value/eff; we just need to fire
        # the verb. Setup spawns non-agent roles via setup_spawner.
        slot = target_value = eff = None

    # Pick an actor concept. Normally that's the "agent" role; for
    # intransitive verbs (fali, bruli, satiĝi, morti, pluvi, …) there
    # is no agent and the action's sole role IS the subject of the
    # sentence — "la folio falas" / "la virino satiĝas". Fall back to
    # that role.
    agent_role_spec = next(
        (r for r in action.roles if r.name == "agent"), None)
    if agent_role_spec is None and len(action.roles) == 1:
        agent_role_spec = action.roles[0]
    if agent_role_spec is None:
        return None
    actor_role_name = agent_role_spec.name
    agent_candidates = _concepts_matching_role(lex, agent_role_spec)
    if not agent_candidates:
        return None
    agent_concept = rng.choice(agent_candidates)

    # Scene location. Retry with different locations on placement
    # failure — actor or target may be incompatible with the
    # randomly-picked scene's containment affordances.
    locations = [l for l, c in lex.concepts.items()
                 if lex.types.is_subtype(c.entity_type, "location")
                 and not getattr(c, "is_category_stub", False)]
    if not locations:
        return None
    tried: set = set()
    for _ in range(5):
        remaining = [l for l in locations if l not in tried]
        if not remaining:
            break
        scene_lemma = rng.choice(remaining)
        tried.add(scene_lemma)

        # Minimal trace: mondo + scene + actor. No theme, no instrument,
        # no away location — the spawner builds those as the planner
        # discovers it needs them.
        t = Trace()
        _ensure_world(t, lex, rng)
        try:
            _add_entity_randomized(t, scene_lemma, lex, rng,
                                    entity_id=scene_lemma)
        except (KeyError, ValueError):
            continue
        scene_id = scene_lemma

        actor_eid = agent_concept
        suffix = 0
        while actor_eid in t.entities:
            suffix += 1
            actor_eid = (
                f"{agent_concept}_actor{suffix if suffix > 1 else ''}")
        try:
            _add_entity_randomized(t, agent_concept, lex, rng,
                                    entity_id=actor_eid)
            t.assert_relation("en", (actor_eid, scene_id), lex)
        except (KeyError, ValueError):
            continue

        if chosen_goal[0] == "property":
            if eff.target_role == actor_role_name:
                target_eid = actor_eid
                slot_def = lex.slots.get(eff.property)
                if slot_def is not None and slot_def.vocabulary:
                    non_target = [v for v in slot_def.vocabulary
                                  if v != eff.value]
                    if non_target:
                        t.entities[actor_eid].set_property(
                            eff.property, rng.choice(non_target))
            else:
                from .spawner import make_spawner
                setup_spawner = make_spawner(
                    scene_id, lex, rng, budget=1)
                target_role_spec = next(
                    (r for r in action.roles
                     if r.name == eff.target_role), None)
                if target_role_spec is None:
                    return None  # schema issue, not placement
                target_eid = setup_spawner(
                    target_role_spec, t, lex,
                    set(t.entities.keys()),
                    action=action, role_name=eff.target_role)
                if target_eid is None:
                    continue  # try another scene location
            drive = ("entity_slot", actor_eid, target_eid,
                     slot, target_value)
            return t, scene_id, drive
        else:
            # event_fire: spawn each non-actor role separately via
            # setup_spawner, collect bindings, emit event_fire drive.
            from .spawner import make_spawner
            setup_spawner = make_spawner(
                scene_id, lex, rng,
                budget=max(1, len(action.roles) - 1))
            bindings: dict = {actor_role_name: actor_eid}
            failed = False
            for role_spec in action.roles:
                if role_spec.name == actor_role_name:
                    continue
                filler = setup_spawner(
                    role_spec, t, lex, set(t.entities.keys()),
                    action=action, role_name=role_spec.name)
                if filler is None:
                    failed = True
                    break
                bindings[role_spec.name] = filler
            if failed:
                continue
            drive = ("event_fire", actor_eid, verb_lemma,
                     tuple(sorted(bindings.items())))
            return t, scene_id, drive
    return None
