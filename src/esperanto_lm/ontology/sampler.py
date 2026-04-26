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


def _person_concepts(lex: Lexicon) -> list[str]:
    """Every concept whose entity_type is `person`. Pulled live from the
    lexicon so additions there flow through without code changes —
    authored (persono, knabo, infano, doktoro…) and derived
    (kuiristo, instruisto, dancisto…) alike. Persons render by their
    entity_id (a name) so visible prose is the same regardless of
    concept, but concept-coverage stats and any future concept-keyed
    rules see real variety."""
    return [c.lemma for c in lex.concepts.values()
            if c.entity_type == "person"]


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


def _make_seed_specific_verb(verb_lemma: str) -> Callable:
    """Build a sampler seeder for an instrument-deriving verb. Fires
    the verb directly with `agent + theme + instrument` bound — no
    `uzi` indirection. Sets up the same precondition state the old
    `_seed_uzi` did (havi(agent, instrument), unlock pordo themes).

    `effect_changes` materializes the verb's intrinsic property
    transitions on the seed event so downstream rules see the new
    state. The old uzi cascade did this via `.changing(...)` in the
    rule; now the seed event carries it directly."""
    def _seed(t: Trace, lex: Lexicon, b: dict[str, str],
              rng: random.Random) -> str:
        agent = b["agent"]
        instrument = b.get("instrument")
        theme = b["theme"]
        if instrument is not None and not any(
                r.relation == "havi" and len(r.args) == 2
                and r.args == (agent, instrument)
                for r in t.relations):
            t.assert_relation("havi", (agent, instrument), lex)
        if t.entities[theme].concept_lemma == "pordo":
            t.entities[theme].set_property("lock_state", "malŝlosita")
        roles = {"agent": agent, "theme": theme}
        if instrument is not None:
            roles["instrument"] = instrument
        t.add_event(make_event(
            verb_lemma, roles=roles,
            property_changes=effect_changes(verb_lemma, roles, lex)))
        return f"{verb_lemma}_{t.entities[theme].concept_lemma}"
    return _seed


def _seed_eat(t: Trace, lex: Lexicon, b: dict[str, str],
              rng: random.Random) -> str:
    agent, theme = b["agent"], b["theme"]
    # Bridge: writes to both old `properties` and new `initial_properties`
    # so both engines see the hungry state. See migration TODO above.
    t.entities[agent].set_property("hunger", "malsata")
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


def _make_generic_seeder(verb_lemma: str) -> Callable:
    """Build a no-precondition seed function for any verb. Just emits
    the event with whatever role bindings the resolver picked, baking
    in the verb's intrinsic effects via `effect_changes`. Recipe label
    embeds the theme concept (when present) so summary stats keep
    granularity, e.g. `naĝi_persono` vs `dormi_kato`.

    Verbs that need pre-state (manĝi sets hunger, instrument-using
    verbs establish havi(agent, instrument), bruli requires
    effect_changes baked) keep their bespoke seeders in
    `_DIRECT_SEEDERS`; everything else flows through here."""
    def _seeder(t: Trace, lex: Lexicon, b: dict[str, str],
                rng: random.Random) -> str:
        roles = dict(b)
        t.add_event(make_event(
            verb_lemma, roles=roles,
            property_changes=effect_changes(verb_lemma, roles, lex)))
        # Recipe label — prefer theme-based, fall back to destination,
        # then to the verb alone for intransitive/agent-only events.
        for role in ("theme", "destination"):
            if role in roles:
                concept = t.entities[roles[role]].concept_lemma
                return f"{verb_lemma}_{concept}"
        return verb_lemma
    return _seeder


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
    """Generate a recipe from a verb's role specs.

    Special handling: when a verb is `derives_instrument` AND has an
    `instrument` role, the binding is augmented with
    `functional_signature=verb.lemma` so the instrument is a tool for
    THIS verb (tranĉilo for tranĉi, ŝlosilo for ŝlosi). Without this,
    the sampler would pick any artifact, producing nonsense like
    "tranĉas serpentidon per la pordo"."""
    bindings: dict[str, RoleBinding] = {}
    for role in verb.roles:
        spec = _role_spec_to_binding(role)
        if role.name == "instrument" and verb.derives_instrument:
            sig_constraint = {"functional_signature": verb.lemma}
            merged = (
                {**spec.property, **sig_constraint} if spec.property
                else sig_constraint)
            spec = RoleBinding(
                sense_id=spec.sense_id,
                entity_type=spec.entity_type,
                property=merged,
                has_property=spec.has_property,
            )
        bindings[role.name] = spec
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
        # Inherit the agent constraint from the verb's own agent role
        # so use_for_X stays in lockstep with the direct verb recipe.
        # In practice this means tool-using verbs (which require
        # can_use_tools=yes on their agent in actions.jsonl) get apes
        # and persons; verbs without that constraint stay open.
        agent_role = next((r for r in verb.roles if r.name == "agent"), None)
        agent_binding = (_role_spec_to_binding(agent_role) if agent_role
                         else RoleBinding(entity_type="animate"))
        bindings = {
            "agent":      agent_binding,
            "instrument": RoleBinding(
                property={"functional_signature": verb.lemma}),
            "theme":      _role_spec_to_binding(theme_role),
        }
        recipes.append(Recipe(
            name=f"use_for_{verb.lemma}",
            role_bindings=bindings,
            seed_fn=_make_seed_specific_verb(verb.lemma),
        ))
    return recipes


_RECIPES_CACHE: dict[int, list[Recipe]] = {}


# Verbs intentionally excluded from the auto-generic recipe pool.
# Movement verbs (iri/veni/veturi) used to be skipped here because their
# destination role would resolve to the current scene (a no-op). That's
# now handled in `_resolve_bindings`: roles named `destination` exclude
# the scene from candidates.
_GENERIC_RECIPE_SKIP: set[str] = set()


def recipes_for(lex: Lexicon) -> list[Recipe]:
    """All recipes, generated from the lexicon. Two layers:

      1. Verbs in `_DIRECT_SEEDERS` use their bespoke seed function
         (these set up preconditions like hunger or initial havi).
      2. Every other verb in the lexicon gets a generic seeder via
         `_make_generic_seeder` — emits the event with role bindings
         resolved by the sampler, no extra setup. This auto-covers
         all new verbs added to the lexicon.

    Cached per lex instance so repeated sample_scene calls don't
    rebuild."""
    if id(lex) in _RECIPES_CACHE:
        return _RECIPES_CACHE[id(lex)]
    out: list[Recipe] = []
    custom = set(_DIRECT_SEEDERS.keys())
    for verb_lemma, seed_fn in _DIRECT_SEEDERS.items():
        verb = lex.actions.get(verb_lemma)
        if verb is None:
            continue
        out.append(_build_direct_recipe(verb, seed_fn))
    # Generic recipes for the remaining verbs.
    for verb_lemma, verb in lex.actions.items():
        if verb_lemma in custom or verb_lemma in _GENERIC_RECIPE_SKIP:
            continue
        out.append(_build_direct_recipe(
            verb, _make_generic_seeder(verb_lemma)))
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
            _add_entity_randomized(
                trace, entity_lemma, lex, rng, entity_id=entity_lemma)
        except (KeyError, ValueError):
            pass
        return

    non_scene = [c for c in candidates if c[0] != scene]
    pick = rng.choice(non_scene) if non_scene else rng.choice(candidates)
    _add_entity_randomized(
        trace, entity_lemma, lex, rng, entity_id=entity_lemma)
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

    Roles are resolved disjointly: each role's pick is excluded from
    subsequent roles' candidate pools, so transitive verbs like
    `instrui(agent, theme)` with two person slots produce two distinct
    persons rather than reflexive bindings (which the realizer would
    otherwise mis-render — "Li instruis lin" instead of "Li instruis sin").

    Role name `destination` (used by movement verbs iri/veni/veturi):
      * excludes the current scene from candidates (self-move is a no-op)
      * expands the candidate pool to all matching concepts in the
        lexicon, not just those reachable via containment from the
        scene — destinations are *targets* of motion, they don't have
        to be reachable as scene contents.
      * placed bare (no containment relation), since destinations
        aren't inside the current scene.

    Other location-typed roles (e.g. `location` for pluvi) are
    unaffected — pluvi at the current scene is exactly what we want.

    Returns the resolved {role: entity_id} dict, or None if any role
    has no candidates (recipe wasn't actually eligible)."""
    resolved: dict[str, str] = {}
    for role, spec in bindings_spec.items():
        already_used = set(resolved.values())
        excluded = set(already_used)
        is_destination = role == "destination"
        if is_destination:
            excluded.add(scene)
        existing = [
            eid for eid, ent in trace.entities.items()
            if eid not in excluded
            and _entity_matches(ent, spec, lex)
        ]
        if is_destination:
            # Whole-lexicon candidates for destinations.
            concepts_avail = [
                lemma for lemma, c in lex.concepts.items()
                if lemma not in excluded
                and lemma not in trace.entities
                and _concept_matches(c, spec, lex)
            ]
        else:
            concepts_avail = [
                c for c in reachable
                if c not in trace.entities
                and c not in excluded
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
            if is_destination:
                _add_entity_randomized(
                    trace, value, lex, rng, entity_id=value)
            else:
                _ensure_placed(trace, lex, idx, scene, value, rng)
            if value not in trace.entities:
                return None
        resolved[role] = value
    return resolved


def _recipe_eligible(recipe: Recipe, trace: Trace, lex: Lexicon,
                     reachable: set[str]) -> bool:
    """Quick precheck: does each role have at least one candidate from
    existing entities or reachable concepts?

    Destination roles draw candidates from the WHOLE lexicon (movement
    targets aren't constrained by scene reachability), so eligibility
    checks them against `lex.concepts` instead of `reachable`."""
    for role_name, spec in recipe.role_bindings.items():
        existing_match = any(
            _entity_matches(ent, spec, lex)
            for ent in trace.entities.values())
        if existing_match:
            continue
        if role_name == "destination":
            concept_match = any(
                _concept_matches(c, spec, lex)
                for c in lex.concepts.values())
        else:
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
    # Scenes don't get state-randomization — locations have fixed
    # affordances (indoor/outdoor) which are identity, not transient.
    t.add_entity(scene, lex, entity_id=scene)

    # Persons: 1-2, with 3:1 bias toward 1.
    n_persons = rng.choices([1, 2], weights=[3, 1], k=1)[0]
    persons: list[str] = []
    person_concepts = _person_concepts(lex)
    for name in rng.sample(PERSON_NAMES, n_persons):
        concept = rng.choice(person_concepts)
        _add_entity_randomized(t, concept, lex, rng, entity_id=name)
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
    # Try recipes in random order until one resolves. Disjoint role
    # resolution can fail eligibility-passing recipes (e.g. instrui
    # needs two distinct persons but only one is in the scene), so
    # fall through to the next eligible candidate rather than erroring.
    rng.shuffle(eligible)
    recipe = None
    bindings = None
    for candidate in eligible:
        resolved = _resolve_bindings(
            t, lex, idx, scene, candidate.role_bindings, reachable, rng)
        if resolved is not None:
            recipe = candidate
            bindings = resolved
            break
    if recipe is None or bindings is None:
        raise RuntimeError(
            f"no eligible recipe could resolve in scene {scene!r}; "
            f"tried {len(eligible)} candidates")

    # Atmospheric extras are off by default — they appeared in scenes
    # without doing anything narratively, leading to clutter like
    # "papagido, kaprido, kaj formikido" in unrelated recipes. Set
    # `n_extras = rng.randint(0, 1)` here if you want a small chance
    # of an extra.
    n_extras = 0
    if n_extras > 0:
        extras_pool = [
            c for c in reachable
            if c not in t.entities
            and lex.concepts[c].entity_type not in ("person", "location")
        ]
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


# ----------------------- state randomization ------------------------------

def _randomize_state(entity, lex: Lexicon, rng: random.Random) -> None:
    """Initialize transient state on a freshly-added entity.

    Single rule: for every slot the entity already has (because the
    concept authored it OR a derivation placed it during the bake)
    where the slot is marked `varies=True`, pick uniformly from the
    slot's vocabulary and replace the value.

    The concept's authored/derived value is just an *opt-in marker*
    saying "this slot is meaningful for this concept" — random varies
    the value freely. So:
      - pordo (concept authors openness=closed) → instance gets
        random open or closed
      - tablo (no openness slot) → no random openness; tablo never
        becomes a malfermi/fermi target
      - persono (animate_has_hunger derivation seeds hunger=sated) →
        every persono instance gets random hungry or sated → manĝi
        with hunger=hungry precondition fires naturally
    """
    for slot_name in list(entity.properties.keys()):
        slot = lex.slots.get(slot_name)
        if slot is None or slot.vocabulary is None or not slot.scalar:
            continue
        if not slot.varies:
            continue
        choice = rng.choice(slot.vocabulary)
        entity.set_property(slot_name, choice)


def _add_entity_randomized(
    trace: Trace, concept: str, lex: Lexicon,
    rng: random.Random, *, entity_id: str,
):
    """Wrapper around `trace.add_entity` that immediately randomizes
    transient state on the new entity AND materializes its declared
    sub-entity parts (Concept.parts). Used for ALL entity additions
    in the sampler so scenes naturally satisfy diverse preconditions
    and parts come along for the ride."""
    trace.add_entity(concept, lex, entity_id=entity_id)
    _randomize_state(trace.entities[entity_id], lex, rng)
    concept_def = lex.concepts.get(concept)
    if concept_def is None:
        return
    for part_spec in concept_def.parts:
        part_id = f"{entity_id}_{part_spec.concept}"
        if part_id in trace.entities:
            continue
        # Recurse: parts may themselves declare parts. Cycles in the
        # concept graph would loop, but our parts graph is a tree.
        _add_entity_randomized(
            trace, part_spec.concept, lex, rng, entity_id=part_id)
        trace.assert_relation(
            part_spec.relation, (entity_id, part_id), lex)


# ----------------------- recipe chaining ----------------------------------

def _recipe_can_reuse_entity(
    recipe: Recipe, touched: set[str], trace: Trace, lex: Lexicon,
) -> bool:
    """True iff at least one of `recipe`'s roles can be filled by some
    entity in `touched`. Used by `sample_chained_scene` to filter
    sequel candidates to recipes that actually connect to the prior
    event's entities."""
    for spec in recipe.role_bindings.values():
        for eid in touched:
            ent = trace.entities.get(eid)
            if ent is None:
                continue
            if _entity_matches(ent, spec, lex):
                return True
    return False


def sample_chained_scene(
    lex: Lexicon, rng: random.Random, *,
    scene: str = "kuirejo",
    rules,
    derivations,
    max_events: int = 4,
    chain_p: float = 0.5,
):
    """Sample a scene + initial event, then iteratively extend with
    sequels.

    Algorithm per chain step:
      1. Run the DSL engine on the current trace so the latest seed
         event's effects (cascades, creates, destroys, property
         changes) are materialized.
      2. Find sequel candidates: recipes where at least one role can
         be filled by an entity the most recent seed event touched
         AND whose role bindings the trace can resolve disjointly.
      3. Pick one randomly, fire its seed function, loop.

    Stops when a coin flip says no (probability 1 - chain_p), when no
    sequel candidates exist, when max_events is reached, or when role
    resolution can't satisfy any candidate.

    Engine cascades (broken→shards, rain→flako, etc.) are NOT counted
    against `max_events` — only sampler-emitted seed events are. The
    cap therefore reflects narrative beats, not total trace length.

    Returns `(trace, info, setup_relations)`. The snapshot is taken
    *after* the first seed event's pre-state relations are asserted
    (matching the historical behavior of generate_corpus's snapshot)
    but *before* the engine runs, so rule-driven relation changes
    appear in the diff as narrative changes.
    """
    from .dsl import run_dsl

    trace, info = sample_scene(lex, rng, scene=scene)
    setup_relations = trace.snapshot_relations()
    seed_event_indices: list[int] = [len(trace.events) - 1]
    run_dsl(trace, rules, derivations, lex)

    idx = resolve_containment(lex)
    reachable = reachable_from(scene, idx, lex) - {scene}
    all_recipes = recipes_for(lex)

    while len(seed_event_indices) < max_events:
        if rng.random() > chain_p:
            break
        last_seed = trace.events[seed_event_indices[-1]]
        # Touched = entities the last seed event referenced as roles
        # AND that still exist in the trace (destroyed entities can't
        # be reused — entities_at(t) would filter them out anyway).
        touched: set[str] = set()
        for v in last_seed.roles.values():
            if not isinstance(v, str):
                continue
            ent = trace.entities.get(v)
            if ent is None:
                continue
            if (ent.destroyed_at_event is not None
                    and ent.destroyed_at_event < len(trace.events)):
                continue
            touched.add(v)
        if not touched:
            break

        candidates = [
            r for r in all_recipes
            if _recipe_can_reuse_entity(r, touched, trace, lex)
        ]
        if not candidates:
            break

        rng.shuffle(candidates)
        fired = False
        for candidate in candidates:
            bindings = _resolve_bindings(
                trace, lex, idx, scene, candidate.role_bindings,
                reachable, rng)
            if bindings is None:
                continue
            new_idx = len(trace.events)
            candidate.seed_fn(trace, lex, bindings, rng)
            if len(trace.events) <= new_idx:
                continue   # seed_fn didn't add anything; try next
            seed_event_indices.append(new_idx)
            run_dsl(trace, rules, derivations, lex)
            fired = True
            break
        if not fired:
            break

    return trace, info, setup_relations


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
