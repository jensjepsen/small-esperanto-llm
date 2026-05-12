"""Spawn-on-demand resolver for the planner's role-binding callback.

The seeder calls `make_spawner(scene_id, away_id, lex, rng)` to get a
closure suitable for the planner's `_ENTITY_RESOLVER` ContextVar. When
the planner's `_find_role_filler` exhausts in-trace candidates, it
invokes the resolver with the unsatisfied role_spec; the resolver
materializes a fresh entity, places it in the scene, and returns its
eid. The planner then continues with the spawned entity bound to the
role.

This is the foundation of the three-layer seeder redesign — see
project_seeder_redesign memory. The seeder builds a minimal scene
(actor + scene location + drive), then hands the spawner to the
planner, which discovers what additional entities the chain needs
and asks for them.

Placement policy: spawned entities go into `away_id` by default
(forcing a locomotion subgoal). The planner's chain-richness weighting
already rewards subgoal-rich verbs; placing spawns far ensures
locomotion chains continue to surface even when the seeder didn't
explicitly set up the theme away. Falls back to `scene_id` when
no away is configured.
"""
from __future__ import annotations

from typing import Callable, Optional


def make_spawner(
    scene_id: str,
    lex,
    rng,
    *,
    budget: int = 6,
) -> Callable:
    """Return a resolver closure for the planner's _ENTITY_RESOLVER.

    Bounded by `budget` total spawns per planning session; once
    exhausted, returns None (the planner falls back to failing the
    candidate, same as today). Prevents the planner from
    arbitrarily growing scenes via greedy spawning.

    The closure mutates the trace as it goes: spawning a non-location
    entity will materialize an "away" sibling location on first need
    (apud the scene). Subsequent spawns reuse that away location, so
    the scene grows into a here/there pair organically. Locations
    spawned for `role_spec.type=="location"` go directly as apud-
    siblings of the scene.
    """
    state = {"spawned": 0, "away_id": None}

    def _spawn_concept(concept: str, role_name: str, trace) -> Optional[str]:
        """Add a fresh instance of `concept` to the trace, returning
        the new eid. Honors uniqueness, runs `_add_entity_randomized`
        so parts and randomized state come along."""
        eid = concept
        suffix = 0
        while eid in trace.entities:
            suffix += 1
            eid = f"{concept}_{role_name}{suffix if suffix > 1 else ''}"
        from ..sampler import _add_entity_randomized
        try:
            _add_entity_randomized(trace, concept, lex, rng, entity_id=eid)
        except (KeyError, ValueError):
            return None
        return eid

    def _ensure_away(trace) -> Optional[str]:
        """Find or spawn an 'away' sibling location apud the scene."""
        if state["away_id"] is not None:
            # Verify it's still in the trace (paranoia).
            if state["away_id"] in trace.entities:
                return state["away_id"]
        # Walk existing apud siblings.
        for r in trace.relations:
            if r.relation != "apud" or len(r.args) != 2:
                continue
            for cand in (r.args[0], r.args[1]):
                if cand == scene_id:
                    continue
                ent = trace.entities.get(cand)
                if ent is None:
                    continue
                if lex.types.is_subtype(ent.entity_type, "location"):
                    state["away_id"] = cand
                    return cand
        # None found: spawn one.
        locations = [
            c.lemma for c in lex.concepts.values()
            if lex.types.is_subtype(c.entity_type, "location")
            and not getattr(c, "is_category_stub", False)
            and c.lemma not in trace.entities
        ]
        if not locations:
            return None
        concept = rng.choice(locations)
        eid = _spawn_concept(concept, "away", trace)
        if eid is None:
            return None
        try:
            trace.assert_relation("apud", (eid, scene_id), lex)
        except (KeyError, ValueError):
            pass
        state["away_id"] = eid
        return eid

    def resolver(role_spec, trace, lex_arg, exclude):
        if state["spawned"] >= budget:
            return None
        # Locations get spawned as apud-siblings of the scene.
        if lex_arg.types.is_subtype("location", role_spec.type) or \
                role_spec.type == "location":
            from .seeders import _concepts_matching_role
            candidates = _concepts_matching_role(lex_arg, role_spec)
            candidates = [c for c in candidates if c not in trace.entities]
            if not candidates:
                return None
            concept = rng.choice(candidates)
            eid = _spawn_concept(concept, role_spec.name, trace)
            if eid is None:
                return None
            try:
                trace.assert_relation("apud", (eid, scene_id), lex_arg)
            except (KeyError, ValueError):
                pass
            state["spawned"] += 1
            return eid
        # Non-location: pick concept, place in (or beside) the away
        # location. Creates the away location on first need.
        from .seeders import _concepts_matching_role
        candidates = _concepts_matching_role(lex_arg, role_spec)
        if not candidates:
            return None
        concept = rng.choice(candidates)
        eid = _spawn_concept(concept, role_spec.name, trace)
        if eid is None:
            return None
        target = _ensure_away(trace) or scene_id
        from .seeders import _route_through_container
        if not _route_through_container(
                trace, eid, concept, target, lex_arg, rng):
            try:
                trace.assert_relation("en", (eid, target), lex_arg)
            except (KeyError, ValueError):
                pass
        state["spawned"] += 1
        return eid

    return resolver
