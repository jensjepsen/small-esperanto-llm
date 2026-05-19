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

_PERSON_CONCEPTS_CACHE: dict[int, list[str]] = {}


def _cached_person_concepts(lex) -> list[str]:
    """Spawnable person concepts (excludes category stubs). Uses the
    per-lex ConceptIndex for the type lookup; stub-filter result is
    cached on id(lex) so the hot path is one dict lookup."""
    key = id(lex)
    cached = _PERSON_CONCEPTS_CACHE.get(key)
    if cached is None:
        cands = lex.concept_index.concepts_matching("person")
        cached = [c for c in cands
                  if not getattr(lex.concepts[c], "is_category_stub", False)]
        _PERSON_CONCEPTS_CACHE[key] = cached
    return cached


def make_spawner(
    scene_id: str,
    lex,
    rng,
    *,
    budget: int = 6,
    prefer_scene_p: float = 1.0,
    actor_eid: Optional[str] = None,
    inject_owner_p: float = 0.0,
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

    `inject_owner_p > 0` makes the spawner occasionally bring along
    an NPC who *holds* the placed item: pick a person concept, place
    them in the same container, assert havi(person, item). Populates
    scenes with owned items, unlocking peti / doni / aĉeti chains.
    Per-call override via the resolver's `inject_owner` kwarg (None
    falls back to inject_owner_p; True/False is hard).

    `actor_eid` — the seeder's chosen actor. Excluded from the NPC
    owner pool so we never reassign the actor's role.
    """
    state = {"spawned": 0}

    def resolver(role_spec, trace, lex_arg, exclude,
                 action=None, role_name=None, *,
                 prefer_scene: bool | None = None,
                 inject_owner: bool | None = None,
                 concept_filter=None):
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
        # kind="relation": value is a relation name. Pick from the
        # role's allowed_values if set, else from all of
        # lex.relations.keys(). The chosen string flows through the
        # planner's role-binding machinery the same way an eid does;
        # downstream rule patterns can match it as a literal Var.
        if getattr(role_spec, "kind", "single") == "relation":
            pool = (list(role_spec.allowed_values)
                    if getattr(role_spec, "allowed_values", None)
                    else list(lex_arg.relations.keys()))
            if not pool:
                return None
            state["spawned"] += 1
            return rng.choice(pool)
        # kind="from_precondition": look up the source relation's
        # arg_kind at the named position. Literal/slot → pick from
        # value pool. Entity → fall through to normal spawn path.
        if getattr(role_spec, "kind", "single") == "from_precondition":
            src_rel_name = getattr(
                role_spec, "from_precondition", None)
            pos = getattr(
                role_spec, "from_precondition_position", None)
            src_rel = lex_arg.relations.get(src_rel_name) \
                if src_rel_name else None
            src_kind = "entity"
            if src_rel is not None and pos is not None:
                kinds = (list(src_rel.arg_kinds) if src_rel.arg_kinds
                         else ["entity"] * src_rel.arity)
                if 0 <= pos < len(kinds):
                    src_kind = kinds[pos]
            if src_kind in ("literal", "slot"):
                pool = (list(role_spec.allowed_values)
                        if getattr(
                            role_spec, "allowed_values", None)
                        else list(lex_arg.relations.keys())
                        if src_kind == "literal"
                        else list(lex_arg.slots.keys()))
                if not pool:
                    return None
                state["spawned"] += 1
                return rng.choice(pool)
            # entity source: fall through to normal entity spawn below
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
        # Caller-supplied predicate (cross-role coordination from
        # _prespawn_for_goal: when filling role R, narrow to
        # candidates that satisfy a rule given-clause constraint
        # against an already-bound peer role). Empty after filter
        # falls back to broad — the spawner shouldn't crash on
        # degenerate caller intent.
        if concept_filter is not None:
            filtered = [c for c in candidates if concept_filter(c)]
            if filtered:
                candidates = filtered
        # Narrow to concepts that model the effect's slot when the
        # caller picked this role because the verb writes to it.
        # Otherwise the spawner cheerfully picks a cepo for glui.theme
        # and the downstream `_drive_target_unmeaningful` check rejects
        # the whole loop iteration — wasted spawn + dep-seed + h_FF
        # cycle. concept_models_slot includes part-derivation lifts
        # (pordo's lock_state from seruro), so the filter doesn't
        # exclude hosts whose state slot lands via a derivation.
        if eff is not None:
            from ..dsl.introspect import concept_models_slot
            from ..dsl.rules import runtime_derivations_for
            derivs = runtime_derivations_for(lex_arg)
            modeled = [c for c in candidates
                       if concept_models_slot(
                           lex_arg.concepts[c], eff.property,
                           lex_arg, derivs)]
            if modeled:
                candidates = modeled
            # else: keep the broad pool — degenerate goals shouldn't
            # crash the spawner, they should fail downstream and the
            # sampler loop retries.
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
        if ent is not None and role_name == "instrument":
            # A single tool is always enough — "kudri per la kvar pingloj"
            # is grammatically fine but narratively awkward. Pin count=1
            # for instruments since the verb uses one instance, not a
            # stack. _add_entity_randomized may have rolled count up
            # to 5 for countable concepts (pinglo, najlo, …).
            ent.set_property("count", "1")
        if ent is not None and action is not None and role_name is not None:
            if eff is not None:
                slot_def = lex_arg.slots.get(eff.property)
                if slot_def is not None and slot_def.vocabulary:
                    non_target = [v for v in slot_def.vocabulary
                                  if v != eff.value]
                    # If the role spec ALSO constrains this slot (e.g.
                    # varmigi.theme requires temperature=malvarma to fire),
                    # intersect — pick a value the verb can actually run
                    # on, not a random non-target. Without this, varmigi
                    # often picks varmega as the initial state, so the
                    # verb can never fire (theme isn't malvarma).
                    role_required = (role_spec.properties or {}).get(
                        eff.property, ())
                    if role_required:
                        candidates = [v for v in non_target
                                       if v in role_required]
                        if candidates:
                            non_target = candidates
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
        # Fast-path when injection is fully disabled. The bench's
        # planner-side spawner uses default `inject_owner_p=0.0` and
        # never passes `inject_owner=True`, so this short-circuit
        # skips the per-call `is_subtype` lookup that profiled at
        # ~55ms/scene parallel overhead from the planner's pre-spawn
        # + mid-search resolver loop. Seeder spawners (which set
        # `inject_owner_p=0.30`) still pay the check, but they're
        # called only a handful of times per scene.
        if inject_owner_p > 0.0 or inject_owner is True:
            # Skip for persons — we don't assign one person to "own"
            # another. `inject_owner=False` (hard skip) is set by the
            # seeder on the drive's target/recipient so the goal isn't
            # pre-satisfied or made unsolvable.
            if ent is not None and not lex_arg.types.is_subtype(
                    ent.entity_type, "person"):
                do_inject = (
                    inject_owner if inject_owner is not None
                    else rng.random() < inject_owner_p)
                if do_inject:
                    _inject_co_located_owner(
                        trace, lex_arg, rng, scene_id,
                        item_eid=eid, actor_eid=actor_eid)
        return eid

    return resolver


def _inject_co_located_owner(
    trace, lex, rng, scene_id, *, item_eid: str, actor_eid: Optional[str],
) -> None:
    """Bring an NPC owner alongside `item_eid`. First try existing
    co-located non-actor persons; if none qualify, spawn a fresh
    person into the item's container. On success, assert
    havi(person, item) and drop the item's en/sur (havi means
    in-hand, so leaving stale physical placement would contradict
    the carrying interpretation — same as scene_builder's
    distribution flow).

    Schema legality is enforced by `assert_relation`: havi's
    arg_excludes / arg_not_part / arg_compare reject locations,
    body parts, nemovebla themes, and items heavier than the
    candidate owner's lift_capacity. Illegal pairs silently no-op.
    """
    container = next(
        (r.args[1] for r in trace.relations
         if r.relation in ("en", "sur") and len(r.args) == 2
         and r.args[0] == item_eid),
        None)
    if container is None:
        return
    # Index entities-by-location once: the candidate scan and the
    # per-spawn co-location check both want O(1) lookup of "who's in
    # this container?". Without this, the candidate filter is O(E·R)
    # (entities × relations) on every injection call.
    in_container: set = {
        r.args[0] for r in trace.relations
        if r.relation == "en" and len(r.args) == 2
        and r.args[1] == container}
    # First pass: an existing person co-located in the same container.
    candidates = [
        eid for eid in in_container
        if eid != actor_eid and eid != item_eid
        and lex.types.is_subtype(
            trace.entities[eid].entity_type, "person")]
    if candidates:
        person_eid = rng.choice(candidates)
        _try_assign_havi(trace, lex, person_eid, item_eid)
        return
    # Second pass: spawn a fresh person into the container. Cached
    # concept list — see _cached_person_concepts above for why.
    person_concepts = _cached_person_concepts(lex)
    if not person_concepts:
        return
    from .seeders import _place_respecting_containment
    # Bounded random pick: rng.sample over a list copy is cheaper
    # than rng.shuffle when we only inspect the first 8.
    pool = [c for c in person_concepts if c not in trace.entities]
    if not pool:
        return
    tries = rng.sample(pool, min(8, len(pool)))
    for concept in tries:
        person_eid = _place_respecting_containment(
            trace, lex, scene_id, concept, rng,
            preferred_id=container)
        if person_eid is None:
            continue
        if _try_assign_havi(trace, lex, person_eid, item_eid):
            return


def _try_assign_havi(trace, lex, person_eid: str, item_eid: str) -> bool:
    try:
        trace.assert_relation("havi", (person_eid, item_eid), lex)
        trace.relations = [
            r for r in trace.relations
            if not (r.relation in ("en", "sur")
                    and len(r.args) == 2 and r.args[0] == item_eid)]
        return True
    except (KeyError, ValueError):
        return False
