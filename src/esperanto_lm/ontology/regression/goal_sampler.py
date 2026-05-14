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


def _construct_goal_scene(lex, rng: random.Random) -> Optional[tuple]:
    """Build a scene for a construction drive: pick a constructable
    concept, materialize the actor + each part + (optional) instrument
    + a stub theme entity, then return the drive that fires `fari`
    with all roles pre-bound.

    Roles in the drive's bindings tuple:
      - agent: actor_eid
      - theme: stub_eid for the to-be-constructed entity
      - parts: tuple of part eids (one per concept.parts entry)
      - instrument: tool eid (only when concept.crafted_with is set)

    Returns None if any setup step fails (incompatible scene location,
    missing ingredient concept, etc.). Caller falls back to property/
    relation goals.
    """
    from ..causal import Trace
    from ..sampler import _add_entity_randomized, _ensure_world

    fari = lex.actions.get("fari")
    if fari is None:
        return None
    # Pick a constructable concept (one whose properties carry
    # constructable=yes and that declares parts).
    constructables = [
        l for l, c in lex.concepts.items()
        if "yes" in c.properties.get("constructable", ())
        and c.parts]
    if not constructables:
        return None
    theme_concept = rng.choice(constructables)
    theme_def = lex.concepts[theme_concept]
    part_concepts = [p.concept for p in theme_def.parts]
    crafted_with = list(theme_def.crafted_with)

    # Pick agent + scene location.
    agent_role = next(
        (r for r in fari.roles if r.name == "agent"), None)
    if agent_role is None:
        return None
    from .seeders import _concepts_matching_role
    agent_candidates = _concepts_matching_role(lex, agent_role)
    if not agent_candidates:
        return None
    agent_concept = rng.choice(agent_candidates)
    locations = [l for l, c in lex.concepts.items()
                 if lex.types.is_subtype(c.entity_type, "location")
                 and not getattr(c, "is_category_stub", False)]
    if not locations:
        return None
    for _ in range(5):
        scene_lemma = rng.choice(locations)
        t = Trace()
        _ensure_world(t, lex, rng)
        try:
            _add_entity_randomized(t, scene_lemma, lex, rng,
                                    entity_id=scene_lemma)
        except (KeyError, ValueError):
            continue
        scene_id = scene_lemma
        actor_eid = agent_concept
        try:
            _add_entity_randomized(t, agent_concept, lex, rng,
                                    entity_id=actor_eid)
            t.assert_relation("en", (actor_eid, scene_id), lex)
        except (KeyError, ValueError):
            continue
        # Spawn one entity per part concept, owned by the actor (havi).
        # Each part is placed by _place_respecting_containment to honor
        # containment rules — fadeno/pasto may need a kitchen, etc.
        from .seeders import _place_respecting_containment
        part_eids: list = []
        ok = True
        for i, pc in enumerate(part_concepts):
            part_eid = f"{pc}_part{i}"
            placed = _place_respecting_containment(
                t, lex, scene_id, pc, rng,
                preferred_id=scene_id, existing_eid=None)
            if placed is None:
                ok = False
                break
            # Rename to deterministic eid? `_place_respecting_containment`
            # uses concept_lemma as eid by default; if the concept is
            # already placed it returns that eid, so duplicates collapse.
            # That's fine — each part_concept appears once per recipe.
            part_eids.append(placed)
        if not ok:
            continue
        # Agent owns each part (havi).
        try:
            for peid in part_eids:
                t.assert_relation("havi", (actor_eid, peid), lex)
        except (KeyError, ValueError):
            continue
        # Pre-stage the to-be-created theme as a bare entity. The
        # fari rule transfers havi to the agent and attaches parts;
        # the entity exists before firing so the planner can reason
        # about it.
        stub_eid = f"{theme_concept}_planned"
        if stub_eid not in t.entities:
            try:
                _add_entity_randomized(
                    t, theme_concept, lex, rng, entity_id=stub_eid)
            except (KeyError, ValueError):
                continue
        # Optional instrument: pick the first concept from crafted_with
        # we can place. Skip when crafted_with is empty.
        instrument_eid = None
        if crafted_with:
            for tool_concept in crafted_with:
                placed = _place_respecting_containment(
                    t, lex, scene_id, tool_concept, rng,
                    preferred_id=scene_id, existing_eid=None)
                if placed is not None:
                    instrument_eid = placed
                    break
            if instrument_eid is None:
                continue  # no tool slot worked

        # Downstream drive: agent USES the constructed thing. Pick
        # the use-verb's target slot+value from a small menu sized
        # by what's plausible for the concept's declared/pervasive
        # slots. The planner discovers fari as a producer of
        # havi(agent, theme) via the construct-aware grounding pass
        # and chains it into the use-verb.
        #
        # Each entry: (weight, slot, value, gate-predicate). Filtered
        # to viable options: the gate-predicate runs against
        # theme_def.properties to keep e.g. integrity drives off
        # concepts that don't declare integrity.
        is_edible = "manĝebla" in theme_def.properties.get(
            "edibility", ())
        has_integrity = "integrity" in theme_def.properties
        # temperature is pervasive on physical → always available
        use_drives: list = []
        if is_edible:
            use_drives.append((40, "presence", "manĝita"))
        if has_integrity:
            use_drives.append((25, "integrity", "tranĉita"))
            use_drives.append((10, "integrity", "rompita"))
        # Warm-the-constructed-thing — works for any physical.
        use_drives.append((15, "temperature", "varmega"))
        use_drives.append((10, "cleanliness", "malpura"))
        if not use_drives:
            return None
        weights = [w for w, *_ in use_drives]
        _w, slot, value = rng.choices(use_drives, weights=weights, k=1)[0]
        drive = ("entity_slot", actor_eid, stub_eid, slot, value)
        return t, scene_id, drive
    return None


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

    # With probability ~15%, emit a construction drive instead of a
    # property/relation goal. Construction has its own path because
    # the recipe (parts list + optional crafted_with tool) is concept-
    # driven rather than effect-driven, so it doesn't slot into
    # build_goal_index's effect-walking logic.
    if "fari" in lex.actions and rng.random() < 0.15:
        result = _construct_goal_scene(lex, rng)
        if result is not None:
            return result
        # fall through if construction setup failed

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
    # When the actor role is itself a location (pluvi, where the verb
    # operates on a location), don't spawn a separate location entity
    # — use the scene itself. The scene picker constrains to locations
    # that satisfy the role's properties (ekstera for pluvi).
    actor_is_scene = lex.types.is_subtype(
        agent_role_spec.type, "location")
    agent_candidates = _concepts_matching_role(lex, agent_role_spec)
    if not agent_candidates:
        return None
    agent_concept = (None if actor_is_scene
                     else rng.choice(agent_candidates))

    # Scene location. Retry with different locations on placement
    # failure — actor or target may be incompatible with the
    # randomly-picked scene's containment affordances.
    if actor_is_scene:
        # Constrain scene pool to concepts that satisfy actor role.
        locations = list(agent_candidates)
    else:
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

        if actor_is_scene:
            actor_eid = scene_id
        else:
            actor_eid = agent_concept
        suffix = 0
        while not actor_is_scene and actor_eid in t.entities:
            suffix += 1
            actor_eid = (
                f"{agent_concept}_actor{suffix if suffix > 1 else ''}")
        if not actor_is_scene:
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
                    # Intersect with role.properties on the same slot
                    # (varmigi.theme=malvarma, etc.) so the chosen
                    # initial state actually lets the verb fire.
                    role_required = (
                        agent_role_spec.properties or {}).get(
                        eff.property, ())
                    if role_required:
                        candidates = [v for v in non_target
                                       if v in role_required]
                        if candidates:
                            non_target = candidates
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
            # event_fire: only bind the actor here. The planner's
            # _prespawn_for_goal fills the remaining roles via the
            # session spawner (same closure used during search), so
            # the sampler doesn't have to commit to specific theme/
            # instrument/recipient up front. Reduces no_sample bails
            # for verbs with many roles.
            drive = ("event_fire", actor_eid, verb_lemma,
                     ((actor_role_name, actor_eid),))
            return t, scene_id, drive
    return None
