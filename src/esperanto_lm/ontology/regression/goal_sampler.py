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
    """Pick a property goal, pick a verb from its producers, build a
    minimal trace (scene location + actor), return (trace, scene_id,
    drive). The spawner — invoked by the planner via the entity_
    resolver hook — materializes everything else with verb-aware
    setup.

    Property goals only for now; relation goals (locomotion, knowledge
    via create-relate) will join when their drive shapes get wired
    into the dispatcher's goal interpretation."""
    from ..sampler import _add_entity_randomized, _ensure_world
    from .seeders import _concepts_matching_role

    index = _cached_goal_index(lex, rules)
    property_goals = [
        (g, verbs) for g, verbs in index.items()
        if g[0] == "property" and verbs
    ]
    if not property_goals:
        return None
    # Weight goals by producer count: cleanliness=pura with 5 verbs
    # outweighs lock_state=ŝlosita with 1, surfacing multi-producer
    # goals more often. Within a goal, the planner's chain-richness
    # weight handles verb selection at runtime.
    weights = [len(verbs) for _, verbs in property_goals]
    chosen_goal, producers = rng.choices(
        property_goals, weights=weights, k=1)[0]
    _, slot, target_value = chosen_goal
    verb_lemma = rng.choice(producers)

    action = lex.actions.get(verb_lemma)
    if action is None or not action.effects:
        return None
    eff = next((e for e in action.effects
                if e.property == slot and e.value == target_value),
               action.effects[0])

    # Pick an agent concept. The actor is the one role we pre-bind:
    # everything else flows from the planner asking the spawner.
    agent_role_spec = next(
        (r for r in action.roles if r.name == "agent"), None)
    if agent_role_spec is None:
        return None
    agent_candidates = _concepts_matching_role(lex, agent_role_spec)
    if not agent_candidates:
        return None
    agent_concept = rng.choice(agent_candidates)

    # Scene location.
    locations = [l for l, c in lex.concepts.items()
                 if lex.types.is_subtype(c.entity_type, "location")
                 and not getattr(c, "is_category_stub", False)]
    if not locations:
        return None
    scene_lemma = rng.choice(locations)

    # Minimal trace: mondo + scene + actor. No theme, no instrument,
    # no away location — the spawner builds those as the planner
    # discovers it needs them.
    t = Trace()
    _ensure_world(t, lex, rng)
    try:
        _add_entity_randomized(t, scene_lemma, lex, rng, entity_id=scene_lemma)
    except (KeyError, ValueError):
        return None
    scene_id = scene_lemma

    actor_eid = agent_concept
    suffix = 0
    while actor_eid in t.entities:
        suffix += 1
        actor_eid = f"{agent_concept}_actor{suffix if suffix > 1 else ''}"
    try:
        _add_entity_randomized(t, agent_concept, lex, rng, entity_id=actor_eid)
        t.assert_relation("en", (actor_eid, scene_id), lex)
    except (KeyError, ValueError):
        return None

    # Drive: the planner needs a target eid. The spawner will fill
    # it in when invoked. To wire this, we leave the target as a
    # placeholder concept-lemma; the planner will lookup by role_spec.
    # NB: we DO pre-spawn the theme entity here, because the drive
    # tuple needs a concrete eid, AND we want the spawner to set its
    # non-target state. Easiest: invoke the spawner directly to pick
    # the theme with full verb-awareness, then bind it to the drive.
    from .spawner import make_spawner
    setup_spawner = make_spawner(scene_id, lex, rng, budget=1)
    target_role_spec = next(
        (r for r in action.roles if r.name == eff.target_role), None)
    if target_role_spec is None:
        return None
    target_eid = setup_spawner(
        target_role_spec, t, lex, set(t.entities.keys()),
        action=action, role_name=eff.target_role)
    if target_eid is None:
        return None

    drive = ("entity_slot", actor_eid, target_eid, slot, target_value)
    return t, scene_id, drive
