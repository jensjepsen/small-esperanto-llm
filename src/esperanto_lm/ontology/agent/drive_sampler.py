"""Random scene + drive sampler.

`sample_scene` composes a random world from primitives — location,
1-2 persons, some objects, maybe-ownership — and returns it without
firing anything. `sample_drive` then picks a goal: an animate's slot
flipped to a non-preferred value (self_slot), an entity_slot mutation
the planner can write, or a derived-relation goal.

`augment_scene_for_drive` adds the supporting state a particular
drive needs (e.g. food + havi for self-hunger, key + havi for
door-unlock). The augmenter's payoff is biased into the sampler via
`_DRIVE_BOOSTS` so coverage exercises lock/food chains more often
than blind sampling would surface them.

`_build_*_writability` indices precompute which (slot, value) and
which relations the rule library can produce. Used by `sample_drive`
to filter to plannable goals.

Lives in `agent` rather than `regression` because it's the agent
loop's drive source (paired with `run_simulation`); the regression
sampler in `ontology.regression` is a different system that picks
verbs and constructs scenes around them.
"""
from __future__ import annotations

import random  # noqa: F401  (referenced by docstring examples; kept for clarity)

from ..causal import Trace
from ..containment import reachable_from, resolve_containment
from ..dsl import compute_derived_state
from ..dsl.effects import AddRelation, Change, Emit, TransferN
from ..dsl.implications import PropertyImplication, RelationImplication
from ..dsl.patterns import Var
from ..sampler import (
    PERSON_NAMES, _STRUCTURAL_SURFACES, _add_entity_randomized,
    _ensure_placed, _ensure_world, _person_concepts,
)
from .planner import (
    _container_of, _entity_property_values, _has_relation,
    _walk_for_entity_patterns_binding,
)
from .preferences import SLOT_PREFERENCES, effective_preferences



def sample_scene(lex, rng, *, max_objects=4):
    """Compose a random scene with no drive attached. Returns
    (trace, scene_id, persons). The drive sampler runs as a separate
    step so the same scene can host different drives in different
    runs."""
    # 1. Random location.
    location_concepts = [
        lemma for lemma, c in lex.concepts.items()
        if c.entity_type == "location"
    ]
    scene = rng.choice(location_concepts)
    idx = resolve_containment(lex)
    reachable = reachable_from(scene, idx, lex) - {scene}

    t = Trace()
    # Trace-wide singleton (tempo_de_tago etc.). Per-trace, picked
    # before any other state so derivations and prefs see consistent
    # world-state from t=0.
    _ensure_world(t, lex, rng)
    t.add_entity(scene, lex, entity_id=scene)

    # 2. 1-2 persons. Slight bias to 2 so altruistic / inter-agent
    # drives are possible (entity_slot drives need a non-self target).
    n_persons = rng.choices([1, 2], weights=[2, 3], k=1)[0]
    person_concepts = _person_concepts(lex)
    persons: list[str] = []
    for name in rng.sample(PERSON_NAMES, min(n_persons, len(PERSON_NAMES))):
        concept = rng.choice(person_concepts)
        _add_entity_randomized(t, concept, lex, rng, entity_id=name)
        t.assert_relation("en", (name, scene), lex)
        persons.append(name)

    # 3. 0-2 structural surfaces compatible with the scene.
    surface_pool = [s for s in _STRUCTURAL_SURFACES if s in reachable]
    rng.shuffle(surface_pool)
    for s in surface_pool[:rng.randint(0, 2)]:
        _ensure_placed(t, lex, idx, scene, s, rng)

    # 4. A few random non-person objects from the scene's reachable set.
    object_pool = [
        c for c in reachable
        if c not in t.entities
        and lex.concepts[c].entity_type not in ("person", "location")
    ]
    rng.shuffle(object_pool)
    for c in object_pool[:rng.randint(1, max_objects)]:
        _ensure_placed(t, lex, idx, scene, c, rng)

    # 4b. Plot-bias: ~25% of indoor scenes get a pordo even if it
    # didn't get picked above. Lock chains and open/close drives need
    # a door to fire; without bias pordo only appears when randomly
    # selected from a pool of dozens of artifacts (~3% of scenes).
    # Indoor heuristic: scene's indoor_outdoor=interna concept tag.
    scene_concept = lex.concepts.get(scene)
    is_indoor = (scene_concept is not None
                 and "interna" in scene_concept.properties.get(
                     "indoor_outdoor", []))
    if (is_indoor and "pordo" in reachable and "pordo" not in t.entities
            and rng.random() < 0.25):
        try:
            _ensure_placed(t, lex, idx, scene, "pordo", rng)
        except (KeyError, ValueError):
            pass

    # 5. ~40% of placed objects get a havi owner from persons. Owner
    # must be co-located with the item — otherwise scenes get "Klara
    # havis la simion" while the simio is two locations away. The
    # planner's later co_locate check would catch this for actions,
    # but the scene-setup `havi` itself is incoherent prose.
    if persons:
        for eid, ent in list(t.entities.items()):
            if eid == scene or ent.entity_type in ("person", "location"):
                continue
            item_container = _container_of(eid, t)
            valid_owners = [
                p for p in persons
                if _container_of(p, t) == item_container
            ]
            if not valid_owners:
                continue
            if rng.random() < 0.4:
                t.assert_relation(
                    "havi", (rng.choice(valid_owners), eid), lex)

    # 5b. Materialize fakto entities for the scene's `en`/`sur`/`havi`
    # relations. The vidi cascade rules also create these on demand
    # (with the same id shape, so engine dedup works), but pre-creating
    # them at scene setup gives the drive sampler a concrete pool of
    # facts to target — "X wants Y to know that Z is in W" requires
    # the fakto entity to exist when the drive is sampled.
    for r in list(t.relations):
        if r.relation in ("en", "sur", "havi") and len(r.args) == 2:
            a, b = r.args
            fakto_id = f"fakto_from_{r.relation}_{a}_{b}"
            if fakto_id in t.entities:
                continue
            t.add_entity("fakto", lex, entity_id=fakto_id)
            t.entities[fakto_id].set_property("pri_relacio", r.relation)
            t.assert_relation("subjekto", (fakto_id, a), lex)
            t.assert_relation("objekto", (fakto_id, b), lex)

    # 6. NOW add 0-2 OTHER locations as movement destinations. Adding
    # these AFTER object placement is intentional — `_ensure_placed`
    # iterates `trace.entities` for valid containers, so locations in
    # the trace at placement time become candidate containers. Items
    # would land in the wrong room ("Klara havis la simion" but simio
    # is in parko, not Klara's vilaĝo). Adjacent locations sit empty
    # so location drives have somewhere to go without polluting the
    # primary scene's content.
    other_locations_pool = [
        l for l in location_concepts if l != scene
    ]
    rng.shuffle(other_locations_pool)
    n_other_locs = rng.choices([0, 1, 2], weights=[2, 3, 1], k=1)[0]
    for loc_concept in other_locations_pool[:n_other_locs]:
        if loc_concept in t.entities:
            continue
        try:
            t.add_entity(loc_concept, lex, entity_id=loc_concept)
        except (KeyError, ValueError):
            pass

    return t, scene, persons


def augment_scene_for_drive(t, drive, lex, rng, scene_id):
    """Add props the drive's chain needs but the random scene didn't
    place. Targeted, not exhaustive — covers the cases where blind
    sampling reliably under-supplies (hunger needs food; unlock needs
    a key). Idempotent: skips augmentation when the prop is already
    present.

    Generalizing this means walking the drive's writer-verbs and
    ensuring each role's preconditions are satisfiable in-scene; for
    now the targeted cases lift fire rates significantly without that
    machinery."""
    kind = drive[0]

    # Self-hunger: needs a manĝebla theme that the actor can havi.
    if (kind == "self_slot"
            and drive[2] == "hunger" and drive[3] == "sata"):
        actor = drive[1]
        # Already a food substance the actor can grab? Check trace.
        existing_food = [
            eid for eid, e in t.entities.items()
            if "manĝebla" in e.properties.get("edibility", [])
        ]
        if not existing_food:
            food_concepts = [
                lemma for lemma, c in lex.concepts.items()
                if c.entity_type == "substance"
                and "manĝebla" in c.properties.get("edibility", [])
            ]
            if food_concepts:
                food = rng.choice(food_concepts)
                food_id = food
                if food_id not in t.entities:
                    try:
                        _add_entity_randomized(
                            t, food, lex, rng, entity_id=food_id)
                        t.assert_relation("en", (food_id, scene_id), lex)
                        existing_food = [food_id]
                    except (KeyError, ValueError):
                        pass
        # Ensure the actor has at least one of the food items, so
        # manĝi's havi precondition is reachable without a preni step.
        if existing_food:
            food_id = existing_food[0]
            already_owned = any(
                r.relation == "havi" and r.args == (actor, food_id)
                for r in t.relations)
            if not already_owned:
                # Move ownership: any prior owner gets the relation
                # removed (havi is exclusive). Simplest: remove all
                # existing havi for this food, assert actor's.
                t.relations = [
                    r for r in t.relations
                    if not (r.relation == "havi" and r.args[1] == food_id)
                ]
                t.assert_relation("havi", (actor, food_id), lex)

    # Door-unlock: entity_slot drive over openness=malfermita on an
    # artifact whose seruro is locked. Ensure a ŝlosilo (key) is in
    # the scene and the actor has it.
    if (kind == "entity_slot"
            and drive[3] == "openness" and drive[4] == "malfermita"):
        actor = drive[1]
        target = drive[2]
        target_ent = t.entities.get(target)
        if target_ent is not None:
            # Find the seruro part, if any.
            seruro_id = None
            for r in t.relations:
                if (r.relation == "havas_parton"
                        and r.args[0] == target):
                    part = t.entities.get(r.args[1])
                    if part is not None and part.concept_lemma == "seruro":
                        seruro_id = r.args[1]
                        break
            if seruro_id is not None:
                # Force the seruro to be locked so the unlock chain
                # is causally necessary (otherwise random varies init
                # half the time leaves it unlocked and the planner
                # just fires malfermi directly).
                t.entities[seruro_id].set_property("lock_state", "ŝlosita")
                # Need a key in the scene.
                key_id = None
                for eid, e in t.entities.items():
                    if e.concept_lemma == "ŝlosilo":
                        key_id = eid
                        break
                if key_id is None:
                    try:
                        key_id = "ŝlosilo"
                        _add_entity_randomized(
                            t, "ŝlosilo", lex, rng, entity_id=key_id)
                        t.assert_relation("en", (key_id, scene_id), lex)
                    except (KeyError, ValueError):
                        key_id = None
                if key_id is not None:
                    already_owned = any(
                        r.relation == "havi"
                        and r.args == (actor, key_id)
                        for r in t.relations)
                    if not already_owned:
                        t.relations = [
                            r for r in t.relations
                            if not (r.relation == "havi"
                                    and r.args[1] == key_id)
                        ]
                        t.assert_relation("havi", (actor, key_id), lex)


def _build_property_writability(lex, rules, derivations) -> dict[tuple[str, str], set[str]]:
    """For each (slot, value), the set of entity types known to be
    valid targets — extracted from verb effect role specs, rule emit
    role bindings, and (conservatively) derivation property
    implications. The drive sampler intersects (slot, value) plus the
    target entity's type against this cache to skip blind drives that
    have no writer for the entity's type.

    Lock_state on a non-seruro animal, for instance, has no writer —
    the cache says lock_state-related types are {artifact}, animal
    isn't a subtype, so the drive is skipped."""
    out: dict[tuple[str, str], set[str]] = {}

    def add(key, t):
        out.setdefault(key, set()).add(t)

    for action in lex.actions.values():
        for eff in action.effects:
            target_role = next(
                (r for r in action.roles if r.name == eff.target_role), None)
            if target_role is None:
                continue
            add((eff.property, eff.value), target_role.type)

    for rule in rules:
        effects = (rule.then if isinstance(rule.then, (list, tuple))
                   else [rule.then])
        for eff in effects:
            if isinstance(eff, Emit):
                # Look up the emitted action's role types to find what
                # the target var resolves to.
                emit_action = lex.actions.get(eff.action)
                # Map of Var → role_name for this Emit's role bindings
                var_to_emit_role: dict[int, str] = {}
                for role_name, role_var in eff.role_vars.items():
                    if isinstance(role_var, Var):
                        var_to_emit_role[id(role_var)] = role_name
                for (ent_arg, slot), val in eff.property_changes.items():
                    if isinstance(val, Var):
                        continue
                    # Resolve ent_arg to a target type via the emit's role.
                    target_type: str | None = None
                    if isinstance(ent_arg, Var):
                        emit_role = var_to_emit_role.get(id(ent_arg))
                        if emit_action is not None and emit_role is not None:
                            for r in emit_action.roles:
                                if r.name == emit_role:
                                    target_type = r.type
                                    break
                    if target_type is None:
                        # Fallback: permissive
                        target_type = "physical"
                    add((slot, val), target_type)
            elif isinstance(eff, Change):
                if not isinstance(eff.value, Var):
                    # Change's entity Var: type unknown without
                    # walking the rule's role/given patterns.
                    # Permissive fallback.
                    add((eff.slot, eff.value), "physical")

    for d in derivations or []:
        for imp in d.implies:
            if (isinstance(imp, PropertyImplication)
                    and not isinstance(imp.value, Var)):
                # Walk the derivation's when patterns to find an entity
                # pattern binding imp.entity — its `type` constraint is
                # the target type. Fall back to permissive `physical`.
                target_type = _entity_pattern_type_for_var(
                    [d.when, *d.given], imp.entity)
                add((imp.slot, imp.value), target_type or "physical")
    return out


def _entity_pattern_type_for_var(patterns, target_var):
    """Walk patterns for entity(type=X) & bind(target_var). Returns X
    or None. Used to type-bound a derivation's PropertyImplication
    target for the writability cache."""
    for pat in patterns:
        for ep in _walk_for_entity_patterns_binding(pat, target_var):
            t = ep.constraints.get("type")
            if isinstance(t, str):
                return t
    return None


def _build_relation_writability(lex, rules, derivations) -> set[str]:
    """Set of relation names reachable via at least one rule's
    AddRelation, TransferN (writes "havi"), or any derivation's
    RelationImplication."""
    out: set[str] = set()
    for rule in rules:
        effects = (rule.then if isinstance(rule.then, (list, tuple))
                   else [rule.then])
        for eff in effects:
            if isinstance(eff, AddRelation):
                out.add(eff.relation)
            elif isinstance(eff, TransferN):
                out.add("havi")
    for d in derivations or []:
        for imp in d.implies:
            if isinstance(imp, RelationImplication):
                out.add(imp.name)
    return out


def sample_drive(t, lex, rng, *, derivations=None, rules=None,
                  property_writable=None, relation_writable=None):
    """Stratified-by-kind drive sampler. Returns one of:

      ("self_slot",   actor, slot, value)
      ("entity_slot", actor, target, slot, value)
      ("location",    actor, location_eid)
      ("possession",  actor, item_eid)

    or None if no eligible drive exists. Stratification first picks a
    kind uniformly among kinds with ≥1 candidate, then samples within
    — this keeps location/possession from being drowned out by the
    combinatorially larger entity_slot pool.

    Self_slot / entity_slot drives DO NOT perturb entity state — they
    pick a goal value that differs from what the entity currently has
    (which the random `varies` init produced for free). This makes
    drive selection composable with whatever state the world arrived
    in, instead of overwriting it."""
    animates = [
        eid for eid, e in t.entities.items()
        if lex.types.is_subtype(e.entity_type, "animate")
        and e.destroyed_at_event is None
    ]
    if not animates:
        return None

    # Derived state surfaces auto-konas-of-self facts (the
    # animate_knows_self_subject/object derivations) so the
    # knowledge-drive filter doesn't sample drives like "Lidia wants
    # to know that Lidia is somewhere".
    derived = (compute_derived_state(t, derivations, lex)
               if derivations else None)

    # Writability filters — built once at coverage-run init, passed in.
    # Without them ~60% of random drives sample goals no verb/rule/
    # derivation can produce, and the planner just returns None.
    if property_writable is None and rules is not None:
        property_writable = _build_property_writability(
            lex, rules, derivations or [])
    if relation_writable is None and rules is not None:
        relation_writable = _build_relation_writability(
            lex, rules, derivations or [])

    candidates: dict[str, list] = {
        "self_slot": [], "entity_slot": [],
        "location": [], "possession": [], "knowledge": [],
    }

    # self_slot + entity_slot: actor wants some entity to have a slot
    # value that's not currently held. Restrict slots to varies-flagged
    # ones (other slots are identity, not transient state).
    for actor in animates:
        for target_eid, target_ent in t.entities.items():
            if target_ent.destroyed_at_event is not None:
                continue
            for slot_name in target_ent.properties:
                slot_def = lex.slots.get(slot_name)
                if (slot_def is None or not slot_def.vocabulary
                        or not slot_def.varies):
                    continue
                current = _entity_property_values(target_ent, slot_name)
                goal_options = [
                    v for v in slot_def.vocabulary if v not in current]
                for goal in goal_options:
                    # Writability filter: skip goals no writer can
                    # produce on an entity of this type. Saves the
                    # planner from chasing dead-end subgoals on every
                    # blind sample (lock_state on a non-seruro animal,
                    # cleanliness=pura on a person, etc.).
                    if property_writable is not None:
                        valid_types = property_writable.get(
                            (slot_name, goal))
                        if not valid_types or not any(
                                lex.types.is_subtype(
                                    target_ent.entity_type, t)
                                for t in valid_types):
                            continue
                    if target_eid == actor:
                        # Self-drive only counts when goal is the
                        # context-effective preferred value (otherwise
                        # we'd model self-harm — "Maria wants to be
                        # malsata", which is fine mechanically but
                        # noise for coverage). At nokto, sleep_state's
                        # preferred flips to dormanta, so self-drives
                        # toward sleeping become valid.
                        prefs_now = effective_preferences(t)
                        if (slot_name in prefs_now
                                and prefs_now[slot_name] == goal):
                            candidates["self_slot"].append(
                                ("self_slot", actor, slot_name, goal))
                    else:
                        candidates["entity_slot"].append((
                            "entity_slot", actor, target_eid,
                            slot_name, goal))

    # location: actor wants to be in a location they're not in.
    locations = [
        eid for eid, e in t.entities.items()
        if lex.types.is_subtype(e.entity_type, "location")
    ]
    for actor in animates:
        current_container = _container_of(actor, t)
        for loc in locations:
            if loc == current_container:
                continue
            candidates["location"].append(("location", actor, loc))

    # possession: actor wants to own an item they don't.
    # Exclude entities that are sub-entity-parts of something else
    # (body parts, locks). They aren't independently graspable.
    is_part = {
        r.args[1] for r in t.relations
        if r.relation == "havas_parton" and len(r.args) == 2
    }
    items = [
        eid for eid, e in t.entities.items()
        if lex.types.is_subtype(e.entity_type, "physical")
        and not lex.types.is_subtype(e.entity_type, "animate")
        and not lex.types.is_subtype(e.entity_type, "location")
        and eid not in is_part
    ]
    for actor in animates:
        for item in items:
            if _has_relation("havi", (actor, item), t):
                continue
            candidates["possession"].append(("possession", actor, item))

    # knowledge: drive is (actor, knower, fakto_id) — actor wants
    # knower to konas this specific fact. Faktos are pre-created in
    # scene setup for every en/sur/havi relation, so the drive pool
    # includes facts like "the libro is on the breto" or "Pavel has
    # the umbrella". When actor==knower, self-knowledge (vidi). When
    # actor!=knower and actor already knows the fact, rakonti is the
    # natural plan. When actor doesn't know either, the planner
    # recurses (actor learns first, then teaches).
    faktos = [
        eid for eid, e in t.entities.items() if e.concept_lemma == "fakto"
    ]
    for actor in animates:
        for knower in animates:
            for fakto_id in faktos:
                if _has_relation("konas", (knower, fakto_id), t,
                                 derived=derived, lex=lex):
                    continue
                candidates["knowledge"].append(
                    ("knowledge", actor, knower, fakto_id))

    nonempty = [k for k, v in candidates.items() if v]
    if not nonempty:
        return None
    kind = rng.choice(nonempty)
    pool = candidates[kind]
    weights = [_drive_weight(c) for c in pool]
    if all(w == 0 for w in weights):
        weights = [1] * len(pool)
    return rng.choices(pool, weights=weights, k=1)[0]


# Drives whose chains the augmenter knows how to support — boost them
# so coverage exercises the augmenter's payoff (lock chains, food
# chains) more often than blind sampling would. Each entry: a
# predicate over a candidate tuple → the boost multiplier.
_DRIVE_BOOSTS = [
    # Self-hunger drive — augmenter places food + havi.
    (lambda c: c[0] == "self_slot" and c[2] == "hunger" and c[3] == "sata",
     20),
    # Door-unlock drive — augmenter places key + havi. The lock chain
    # (vidi → malŝlosi → malfermi) is the highest-payoff demonstration
    # of the recent lock-as-part work, but the entity_slot pool has
    # dozens of body-part-cleanliness candidates competing per scene.
    # Heavy boost so it surfaces when conditions allow.
    (lambda c: (c[0] == "entity_slot"
                and c[3] == "openness" and c[4] == "malfermita"),
     100),
]


def _drive_weight(candidate) -> int:
    """Per-candidate boost for weighted sampling. Default weight 1
    (uniform). Augmentable goals get boosts so they sample more often
    than their share of the random pool would give them — otherwise
    rare special-cases (lock chain) almost never get picked even when
    the scene supports them."""
    weight = 1
    for predicate, boost in _DRIVE_BOOSTS:
        try:
            if predicate(candidate):
                weight *= boost
        except (IndexError, TypeError):
            pass
    return weight

