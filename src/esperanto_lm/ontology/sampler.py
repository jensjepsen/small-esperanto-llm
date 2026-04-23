"""Scenario-agnostic procedural sampler.

A Recipe is a scenario shape: role bindings + a seed function. The
sampler:
  1. Resolves containment from the scene → reachable concepts
  2. Filters recipes to those whose every role has at least one
     candidate (existing entity or reachable concept)
  3. Picks a recipe uniformly
  4. For each role, picks a candidate (uniform over existing entities
     matching the binding ∪ reachable concepts matching). Materializes
     concept-side picks into the trace via containment placement.
  5. Runs the recipe's seed_fn with the resolved bindings

There are no per-(agent, theme) recipe enumerations and no per-scene
code paths. Variety comes from the cross product of recipe shapes ×
scene reachability × role-filler choices.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from .causal import Trace, effect_changes, make_event
from .containment import (
    containment_relation_for,
    reachable_from,
    resolve_containment,
)
from .loader import Lexicon


PERSON_NAMES = ["petro", "maria", "anna", "klara", "johano",
                "elena", "pavel", "sara", "mikael", "lidia"]


@dataclass
class SceneInfo:
    seed: int
    recipe: str
    persons: list[str]
    n_objects: int
    scene_location_id: str = "kuirejo"


# ----------------------- role binding -------------------------------------

@dataclass(frozen=True)
class RoleBinding:
    """How to find a candidate entity/concept for a recipe role.

    Conjunction of all set fields. Mirrors `ContainmentPattern` shape but
    without `contains:` (no relational binding yet — recipes don't need
    it; preconditions live in seed_fn).
    """
    sense_id: Optional[str] = None
    entity_type: Optional[str] = None
    property: Optional[dict[str, str]] = None
    has_property: Optional[str] = None       # slot is set (any value)


def _binding_matches(props: dict, etype: str, lemma: str,
                     binding: RoleBinding, lex: Lexicon) -> bool:
    if binding.sense_id is not None and lemma != binding.sense_id:
        return False
    if binding.entity_type is not None:
        if not lex.types.is_subtype(etype, binding.entity_type):
            return False
    if binding.property is not None:
        for k, v in binding.property.items():
            if v not in props.get(k, []):
                return False
    if binding.has_property is not None:
        if not props.get(binding.has_property):
            return False
    return True


def _entity_matches(entity, binding: RoleBinding, lex: Lexicon) -> bool:
    return _binding_matches(entity.properties, entity.entity_type,
                            entity.concept_lemma, binding, lex)


def _concept_matches(concept, binding: RoleBinding, lex: Lexicon) -> bool:
    return _binding_matches(concept.properties, concept.entity_type,
                            concept.lemma, binding, lex)


# ----------------------- recipes ------------------------------------------

@dataclass
class Recipe:
    name: str
    role_bindings: dict[str, RoleBinding]
    # seed_fn signature: (trace, lexicon, bindings, rng) -> label string.
    # `bindings` is {role_name: entity_id} for the resolved entities.
    seed_fn: Callable[[Trace, Lexicon, dict[str, str], random.Random], str]


def _seed_uzi(t: Trace, lex: Lexicon, b: dict[str, str],
              rng: random.Random) -> str:
    agent, instrument, theme = b["agent"], b["instrument"], b["theme"]
    if not any(r.relation == "havi" and len(r.args) == 2
               and r.args == (agent, instrument) for r in t.relations):
        t.assert_relation("havi", (agent, instrument), lex)
    # Unlock pordo so the lock-event is observable.
    # set_initial_property mirrors into both `properties` (old engine)
    # and `initial_properties` (v2 engine). TODO(step 6): once the old
    # engine is removed, switch to direct initial_properties assignment.
    if t.entities[theme].concept_lemma == "pordo":
        t.entities[theme].set_property("lock_state", "unlocked")
    t.add_event(make_event("uzi", roles={
        "agent": agent, "instrument": instrument, "theme": theme}))
    return f"use_{t.entities[instrument].concept_lemma}_on_{t.entities[theme].concept_lemma}"


def _seed_eat(t: Trace, lex: Lexicon, b: dict[str, str],
              rng: random.Random) -> str:
    agent, theme = b["agent"], b["theme"]
    # Bridge: writes to both old `properties` and new `initial_properties`
    # so both engines see the hungry state. See migration TODO above.
    t.entities[agent].set_property("hunger", "hungry")
    # Persons traditionally hold their food via havi; animals just eat
    # from the scene without a possessive relation.
    if t.entities[agent].entity_type == "person":
        if not any(r.relation == "havi" and len(r.args) == 2
                   and r.args == (agent, theme) for r in t.relations):
            t.assert_relation("havi", (agent, theme), lex)
    t.add_event(make_event("manĝi", roles={"agent": agent, "theme": theme}))
    return f"{t.entities[agent].concept_lemma}_eats_{t.entities[theme].concept_lemma}"


def _seed_fali(t: Trace, lex: Lexicon, b: dict[str, str],
               rng: random.Random) -> str:
    theme = b["theme"]
    t.add_event(make_event("fali", roles={"theme": theme}))
    return f"drop_{t.entities[theme].concept_lemma}"


def _seed_akvumi(t: Trace, lex: Lexicon, b: dict[str, str],
                 rng: random.Random) -> str:
    agent, theme = b["agent"], b["theme"]
    t.add_event(make_event("akvumi", roles={"agent": agent, "theme": theme}))
    return f"water_{t.entities[theme].concept_lemma}"


def _seed_planti(t: Trace, lex: Lexicon, b: dict[str, str],
                 rng: random.Random) -> str:
    agent, theme = b["agent"], b["theme"]
    t.add_event(make_event("planti", roles={"agent": agent, "theme": theme}))
    return f"plant_{t.entities[theme].concept_lemma}"


def _seed_malfermi(t: Trace, lex: Lexicon, b: dict[str, str],
                   rng: random.Random) -> str:
    agent, theme = b["agent"], b["theme"]
    t.add_event(make_event("malfermi", roles={"agent": agent, "theme": theme}))
    return f"open_{t.entities[theme].concept_lemma}"


# Verbs that make good "scenario seeds" with their direct seed_fn.
# The seed_fn owns scenario-specific preconditions (set hunger, etc.).
# Role bindings are generated from the verb's role specs at lex-load time
# — respecting whatever type and property constraints the verb declares.
_DIRECT_SEEDERS: dict[str, Callable] = {
    "manĝi":    None,           # assigned below once _seed_* are defined
    "fali":     None,
    "akvumi":   None,
    "planti":   None,
    "malfermi": None,
    "fermi":    None,
    "kuiri":    None,
    "rompiĝi":  None,
    "bruli":    None,
}


def _seed_kuiri(t: Trace, lex: Lexicon, b: dict[str, str],
                rng: random.Random) -> str:
    agent, theme = b["agent"], b["theme"]
    t.add_event(make_event("kuiri", roles={
        "agent": agent, "theme": theme}))
    return f"kuiri_{t.entities[theme].concept_lemma}"


def _seed_fermi(t: Trace, lex: Lexicon, b: dict[str, str],
                rng: random.Random) -> str:
    agent, theme = b["agent"], b["theme"]
    t.add_event(make_event("fermi", roles={
        "agent": agent, "theme": theme}))
    return f"close_{t.entities[theme].concept_lemma}"


def _seed_rompiĝi(t: Trace, lex: Lexicon, b: dict[str, str],
                  rng: random.Random) -> str:
    """Theme just breaks — not caused by a fall. Lets
    `broken_container_releases_contents` fire in isolation."""
    theme = b["theme"]
    t.add_event(make_event("rompiĝi", roles={"theme": theme}))
    return f"break_{t.entities[theme].concept_lemma}"


def _seed_bruli(t: Trace, lex: Lexicon, b: dict[str, str],
                rng: random.Random) -> str:
    """Flammable theme catches fire. The engine's
    `fire_spreads_to_adjacent_flammables` then propagates to touching
    flammables via en/sur relations.
    """
    theme = b["theme"]
    roles = {"theme": theme}
    t.add_event(make_event(
        "bruli", roles=roles,
        property_changes=effect_changes("bruli", roles, lex)))
    return f"burn_{t.entities[theme].concept_lemma}"


_DIRECT_SEEDERS = {
    "manĝi":    _seed_eat,
    "fali":     _seed_fali,
    "akvumi":   _seed_akvumi,
    "planti":   _seed_planti,
    "malfermi": _seed_malfermi,
    "fermi":    _seed_fermi,
    "kuiri":    _seed_kuiri,
    "rompiĝi":  _seed_rompiĝi,
    "bruli":    _seed_bruli,
}


def _role_spec_to_binding(role) -> RoleBinding:
    """Convert a verb's RoleSpec (type + properties) to a RoleBinding.
    Direct mirror — the binding just inherits the verb's declared
    constraints. Empty property dicts become None for uniformity."""
    props = dict(role.properties) if role.properties else None
    # RoleSpec.properties is dict[str, list[str]] (each slot → allowed
    # values list). RoleBinding.property is dict[str, str] (single
    # required value). Flatten lists to first element — this loses some
    # expressiveness but our verb specs never use multi-value lists.
    if props is not None:
        props = {k: v[0] for k, v in props.items() if v}
        if not props:
            props = None
    return RoleBinding(entity_type=role.type, property=props)


def _build_direct_recipe(verb, seed_fn: Callable) -> Recipe:
    """Generate a recipe from a verb's role specs."""
    bindings = {role.name: _role_spec_to_binding(role)
                for role in verb.roles}
    return Recipe(name=verb.lemma, role_bindings=bindings, seed_fn=seed_fn)


def _build_use_recipes(lex: Lexicon) -> list[Recipe]:
    """One use_for_<verb> recipe per derives_instrument verb. The theme
    binding is read from THAT verb's theme role spec — so e.g.
    use_for_ŝlosi binds theme to artifact (ŝlosi's theme type), and
    use_for_purigi binds to physical (purigi's looser spec).

    This is the cross-role binding: theme depends on what the instrument's
    signature verb accepts. Resolved once at recipe-build time because
    our role bindings are declarative per-recipe."""
    recipes: list[Recipe] = []
    for verb in lex.actions.values():
        if not verb.derives_instrument:
            continue
        theme_role = next((r for r in verb.roles if r.name == "theme"), None)
        if theme_role is None:
            continue
        bindings = {
            "agent":      RoleBinding(entity_type="person"),
            "instrument": RoleBinding(
                property={"functional_signature": verb.lemma}),
            "theme":      _role_spec_to_binding(theme_role),
        }
        recipes.append(Recipe(
            name=f"use_for_{verb.lemma}",
            role_bindings=bindings,
            seed_fn=_seed_uzi,
        ))
    return recipes


_RECIPES_CACHE: dict[int, list[Recipe]] = {}


def recipes_for(lex: Lexicon) -> list[Recipe]:
    """All recipes, generated from the lexicon. Cached per lex instance
    so repeated sample_scene calls don't rebuild."""
    if id(lex) in _RECIPES_CACHE:
        return _RECIPES_CACHE[id(lex)]
    out: list[Recipe] = []
    for verb_lemma, seed_fn in _DIRECT_SEEDERS.items():
        verb = lex.actions.get(verb_lemma)
        if verb is None:
            continue
        out.append(_build_direct_recipe(verb, seed_fn))
    out.extend(_build_use_recipes(lex))
    _RECIPES_CACHE[id(lex)] = out
    return out


# ----------------------- placement helper ---------------------------------

def _ensure_placed(
    trace: Trace, lex: Lexicon, idx, scene: str,
    entity_lemma: str, rng: random.Random,
) -> None:
    """Add `entity_lemma` to the trace and assert a containment relation
    placing it under some entity already in the trace. Prefers non-scene
    containers for more specific placement."""
    if entity_lemma in trace.entities:
        return

    candidates: list[tuple[str, str]] = []
    for container_lemma in list(trace.entities.keys()):
        container_ent = trace.entity(container_lemma)
        if container_ent is None or container_ent.entity_type == "person":
            continue
        rel = containment_relation_for(container_lemma, entity_lemma, idx, lex)
        if rel is not None:
            candidates.append((container_lemma, rel))

    if not candidates:
        try:
            trace.add_entity(entity_lemma, lex, entity_id=entity_lemma)
        except (KeyError, ValueError):
            pass
        return

    non_scene = [c for c in candidates if c[0] != scene]
    pick = rng.choice(non_scene) if non_scene else rng.choice(candidates)
    trace.add_entity(entity_lemma, lex, entity_id=entity_lemma)
    trace.assert_relation(pick[1], (entity_lemma, pick[0]), lex)


# ----------------------- binding resolution -------------------------------

def _resolve_bindings(
    trace: Trace, lex: Lexicon, idx, scene: str,
    bindings_spec: dict[str, RoleBinding], reachable: set[str],
    rng: random.Random,
) -> Optional[dict[str, str]]:
    """For each role, pick uniformly across (existing entities matching)
    ∪ (reachable concepts matching, excluding ones already placed).
    Materializes concept picks via containment placement.

    Returns the resolved {role: entity_id} dict, or None if any role
    has no candidates (recipe wasn't actually eligible)."""
    resolved: dict[str, str] = {}
    for role, spec in bindings_spec.items():
        existing = [
            eid for eid, ent in trace.entities.items()
            if _entity_matches(ent, spec, lex)
        ]
        concepts_avail = [
            c for c in reachable
            if c not in trace.entities
            and _concept_matches(lex.concepts[c], spec, lex)
        ]
        # Tag each option so we know whether to materialize.
        options: list[tuple[str, str]] = (
            [("existing", e) for e in existing]
            + [("concept", c) for c in concepts_avail]
        )
        if not options:
            return None
        kind, value = rng.choice(options)
        if kind == "concept":
            _ensure_placed(trace, lex, idx, scene, value, rng)
            if value not in trace.entities:
                return None
        resolved[role] = value
    return resolved


def _recipe_eligible(recipe: Recipe, trace: Trace, lex: Lexicon,
                     reachable: set[str]) -> bool:
    """Quick precheck: does each role have at least one candidate from
    existing entities or reachable concepts?"""
    for spec in recipe.role_bindings.values():
        existing_match = any(
            _entity_matches(ent, spec, lex)
            for ent in trace.entities.values())
        if existing_match:
            continue
        concept_match = any(
            _concept_matches(lex.concepts[c], spec, lex)
            for c in reachable)
        if not concept_match:
            return False
    return True


# ----------------------- main entry point ---------------------------------

# Surface containers — sometimes pre-seed these so subsequent placements
# have something to land on (richer 'pano sur tablo' chains).
_STRUCTURAL_SURFACES = ["tablo", "breto", "korbo", "sofo"]


def sample_scene(
    lex: Lexicon, rng: random.Random, *, scene: str = "kuirejo",
) -> tuple[Trace, SceneInfo]:
    if scene not in lex.concepts:
        raise ValueError(f"unknown scene concept: {scene!r}")
    if lex.concepts[scene].entity_type != "location":
        raise ValueError(
            f"scene {scene!r} is not a location "
            f"(entity_type={lex.concepts[scene].entity_type})")

    idx = resolve_containment(lex)
    reachable = reachable_from(scene, idx, lex) - {scene}

    t = Trace()
    t.add_entity(scene, lex, entity_id=scene)

    # Persons: 1-2, with 3:1 bias toward 1.
    n_persons = rng.choices([1, 2], weights=[3, 1], k=1)[0]
    persons: list[str] = []
    for name in rng.sample(PERSON_NAMES, n_persons):
        t.add_entity("persono", lex, entity_id=name)
        t.assert_relation("en", (name, scene), lex)
        persons.append(name)

    # Optional structural surfaces for richer placement chains.
    surface_pool = [s for s in _STRUCTURAL_SURFACES if s in reachable]
    n_struct = rng.randint(0, len(surface_pool))
    for s in rng.sample(surface_pool, n_struct):
        _ensure_placed(t, lex, idx, scene, s, rng)

    # Pick eligible recipe (recipe pool is generated from the lexicon).
    eligible = [r for r in recipes_for(lex)
                if _recipe_eligible(r, t, lex, reachable)]
    if not eligible:
        raise RuntimeError(
            f"no recipes eligible in scene {scene!r}; "
            f"reachable: {sorted(reachable)}")
    recipe = rng.choice(eligible)

    # Resolve role bindings (materializes entities as needed).
    bindings = _resolve_bindings(
        t, lex, idx, scene, recipe.role_bindings, reachable, rng)
    if bindings is None:
        raise RuntimeError(
            f"recipe {recipe.name!r} eligible but resolution failed")

    # 0-2 atmospheric extras for visual richness.
    extras_pool = [
        c for c in reachable
        if c not in t.entities
        and lex.concepts[c].entity_type not in ("person", "location")
    ]
    n_extras = rng.randint(0, 2)
    for c in rng.sample(extras_pool, min(n_extras, len(extras_pool))):
        _ensure_placed(t, lex, idx, scene, c, rng)

    label = recipe.seed_fn(t, lex, bindings, rng)

    n_objects = sum(
        1 for eid, e in t.entities.items()
        if e.entity_type != "person" and eid != scene)
    return t, SceneInfo(
        seed=0,
        recipe=label,
        persons=persons,
        n_objects=n_objects,
        scene_location_id=scene,
    )


# ----------------------- prune --------------------------------------------

def prune_unused_persons(trace: Trace) -> list[str]:
    """Remove person entities who never appear in any event's role bindings,
    along with any relations referencing them."""
    used_in_events: set[str] = set()
    for ev in trace.events:
        for role_value in ev.roles.values():
            if isinstance(role_value, str):
                used_in_events.add(role_value)

    pruned: list[str] = []
    for eid, ent in list(trace.entities.items()):
        if ent.entity_type == "person" and eid not in used_in_events:
            pruned.append(eid)

    if not pruned:
        return pruned

    pruned_set = set(pruned)
    for pid in pruned:
        trace.entities.pop(pid, None)
    trace.relations = [
        r for r in trace.relations
        if not any(arg in pruned_set for arg in r.args)
    ]
    return pruned
