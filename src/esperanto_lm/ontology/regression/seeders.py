"""Drive-specific scene seeders. Each `regress_for_*` builds a
(trace, scene_id, drive) triple where the planner can produce a
chain ending in (or via) the named verb.

All seeders use `SceneBuilder` from `scene_builder` for placement;
they differ in which drive shape they construct and what
scaffolding they pre-place.

Knowledge / cascade / reflexive seeders are introspection-driven
(walk rules to find the verb's effect/cascade/precondition shape);
the count/buy/sell/etc. ones are still hand-coded around the count
drive shape.
"""
from __future__ import annotations

from typing import Any, Optional

from ..causal import EntityInstance, Trace, effect_changes, make_event
from ..dsl.effects import AddRelation
from ..dsl.patterns import EventPattern
from .scene_builder import SceneBuilder, scene


def _verbs_adding_konas(rules) -> set[str]:
    """Verb lemmas whose causal rules add a konas relation. Used by
    the regression sampler to surface knowledge-transfer chains
    (rakonti, vidi). The check looks at the rule's then-effects rather
    than action.effects (which is per-verb-schema property changes)
    because konas additions live in the rule layer, not the schema."""
    out: set[str] = set()
    for rule in rules:
        if not isinstance(rule.when, EventPattern):
            continue
        rule_effects = (rule.then if isinstance(rule.then, (list, tuple))
                         else [rule.then])
        for eff in rule_effects:
            if isinstance(eff, AddRelation) and eff.relation == "konas":
                out.add(rule.when.action)
                break
    return out


def _regression_verb_pool(lex) -> list[str]:
    """Verbs eligible as regression targets: any action whose first
    effect writes a `varies=True` slot. The varies check matters
    because the regressor sets the theme's effect-slot to a non-target
    initial value — a slot whose value can't vary at instance time
    has nothing to flip from."""
    out = []
    for lemma, action in lex.actions.items():
        if not action.effects:
            continue
        slot = lex.slots.get(action.effects[0].property)
        if slot is None or not slot.varies:
            continue
        out.append(lemma)
    return out


def _verb_chain_weight(action) -> float:
    """Score a verb by its subgoal-chain potential — same shape as the
    planner's `_chain_richness_weight`, lifted to seeder-side so the
    drive picker biases toward verbs that produce richer plans.

    Counts preconditions + total slots across all role.properties.
    A verb with no preconditions and bare roles scores 1; malŝlosi
    (samloke + havi + multiple role-property gates) scores ~5;
    veturi (when in pool) ~6. Surfacing rich verbs more often
    dilutes the bare-state head visible in dedup analysis without
    adding any per-verb hardcoding — the lexicon's existing
    precondition/role data is the signal."""
    score = 1 + len(action.preconditions)
    for role in action.roles:
        score += len(role.properties)
    return float(score)


def _concepts_matching_role(lex, role_spec) -> list[str]:
    """Concepts compatible with role_spec: subtype-correct AND every
    role.properties slot is meaningful for the concept.

    For immutable slots (e.g. functional_signature=ŝlosi) the concept's
    declared value must intersect the role's required set.

    For varies=True slots, the value gets randomized at instance-time —
    but only if the concept declares the slot. A concept that doesn't
    declare openness can't be a meaningful malfermi.theme even though
    the type spine allows it."""
    out = []
    for lemma, concept in lex.concepts.items():
        if not lex.types.is_subtype(concept.entity_type, role_spec.type):
            continue
        ok = True
        for slot, vals in role_spec.properties.items():
            slot_def = lex.slots.get(slot)
            if slot_def is None:
                continue
            cvals = concept.properties.get(slot, [])
            if slot_def.varies:
                # Pervasive slots (hunger, wetness, etc.) apply to every
                # concept of the slot's applies_to type via a default
                # derivation — no per-concept declaration needed.
                # Opt-in slots (openness, lock_state) require the
                # concept to declare them.
                if getattr(slot_def, "pervasive", False):
                    continue
                if not cvals:
                    ok = False
                    break
                continue
            if not (set(vals) & set(cvals)):
                ok = False
                break
        if ok:
            out.append(lemma)
    return out



# ----------------------- scene seeders ------------------------------


def regress_for_verb(verb_name, lex, rng):
    """Build a (trace, scene_id, drive) so the planner produces a chain
    ending in `verb_name`. Single-verb regression — multi-step chains
    happen via the planner subgoaling on unsatisfied role.properties.
    Returns None if no compatible concepts exist for some role."""
    from ..sampler import _add_entity_randomized
    action = lex.actions.get(verb_name)
    if action is None or not action.effects:
        return None

    locations = [l for l, c in lex.concepts.items()
                 if lex.types.is_subtype(c.entity_type, "location")]
    if not locations:
        return None
    scene_lemma = rng.choice(locations)

    t = Trace()
    try:
        _add_entity_randomized(t, scene_lemma, lex, rng,
                                entity_id=scene_lemma)
    except (KeyError, ValueError):
        return None
    scene_id = scene_lemma

    # Pick a second location for chain ingredients to live in. Forces
    # the planner to subgoal on locomotion (iri) + retrieval (preni)
    # rather than getting everything in a single room. Falls back to
    # scene_id when only one location is available.
    away_lemma = next(
        (l for l in (rng.sample(locations, len(locations)))
         if l != scene_lemma), scene_lemma)
    away_id = away_lemma
    if away_id != scene_id:
        try:
            _add_entity_randomized(t, away_lemma, lex, rng, entity_id=away_id)
            # Anchor as a sibling so the realizer's en-only chain can
            # surface it (otherwise the away location floats with no
            # spatial relation to the scene; later Tier 3 reuse would
            # find it pre-placed and skip adding the apud link).
            t.assert_relation("apud", (away_id, scene_id), lex)
        except (KeyError, ValueError):
            away_id = scene_id

    eff = action.effects[0]
    role_eids: dict[str, str] = {}
    for role in action.roles:
        candidates = _concepts_matching_role(lex, role)
        weights: list[float] | None = None
        if role.name == eff.target_role:
            # We don't filter candidates here on whether the effect
            # slot appears in `c.properties`. The role spec's
            # type+properties and the slot's `applies_to` are the
            # principled gates; an extra "concept declares the slot
            # on disk" filter would falsely reject hosts whose state
            # is lifted from a part by a derivation (pordo getting
            # lock_state from seruro) and every `varies` slot (whose
            # value lands at instance creation, never on disk).
            #
            # Soft bias toward candidates that trigger a conditional
            # precondition: gate-able candidates collectively get ~half
            # the probability mass, the rest split the other half.
            # Keeps both chain coverage (pordo for malfermi) and
            # breadth coverage (sako/botelo/...) live in one run.
            weights = _candidate_weights(candidates, role.name, action, lex)
        if not candidates:
            return None
        if weights is not None:
            concept_lemma = rng.choices(candidates, weights=weights, k=1)[0]
        else:
            concept_lemma = rng.choice(candidates)
        eid = concept_lemma
        suffix = 0
        while eid in t.entities:
            suffix += 1
            eid = f"{concept_lemma}_{role.name}{suffix if suffix > 1 else ''}"
        try:
            _add_entity_randomized(t, concept_lemma, lex, rng, entity_id=eid)
        except (KeyError, ValueError):
            return None
        role_eids[role.name] = eid

    # Hand the rest of the setup off to apply_scene_parameters: it
    # handles placement, non-target initial state, conditional gate
    # forcing, chain-dependency scaffolding, indoor lamp, and
    # priskribas scaffolding — all parameterized via SceneParameters
    # so the goal-first sampler can produce identically-shaped
    # scenes with the same setup logic.
    from .scene_params import sample_scene_parameters, apply_scene_parameters
    params = sample_scene_parameters(rng)
    if not apply_scene_parameters(
            t, scene_id, away_id, action, role_eids, lex, rng, params):
        # A role-entity has no valid containment host — abort and let
        # `sample_regression_scene` retry with a different verb/concept
        # pick.
        return None

    target_eid = role_eids.get(eff.target_role)
    agent_eid = role_eids.get("agent")
    if target_eid is None or agent_eid is None:
        return None
    drive = ("entity_slot", agent_eid, target_eid,
             eff.property, eff.value)
    return t, scene_id, drive


def _candidate_weights(
    candidates: list, role_name: str, action, lex,
) -> list[float] | None:
    """Weights for sampling a role candidate, biasing toward concepts
    that trigger a conditional precondition gating this role. Returns
    None when no bias applies (no IfPropertyPrecondition on the role,
    OR all/no candidates trigger a gate — uniform either way).

    Bias formula: gate-able candidates collectively get the same
    probability mass as non-gate-able ones, so a single gate-able
    concept (e.g. pordo among 7 openness-having candidates) lands
    roughly half the time. Keeps both chain growth and breadth in
    one run without a tunable constant."""
    from ..schemas import IfPropertyPrecondition

    gate_props = {pc.if_property for pc in action.preconditions
                  if isinstance(pc, IfPropertyPrecondition)
                  and pc.role == role_name}
    if not gate_props:
        return None
    gate_able = {c for c in candidates
                 if any(p in lex.concepts[c].properties for p in gate_props)}
    n_gate = len(gate_able)
    n_other = len(candidates) - n_gate
    if n_gate == 0 or n_other == 0:
        return None   # uniform either way
    boost = n_other / n_gate
    return [boost if c in gate_able else 1.0 for c in candidates]


def _seed_priskribas_about_theme(t, theme_eid, lex, rng) -> None:
    """Place a readable text in some trace location with priskribas →
    a fakto about the theme's `en` location. The planner can then
    route scias_lokon(actor, theme) through legi(text) when direct
    vidi isn't viable (theme in another room, scene unilluminated).

    Design:
      - Reuses an existing fakto entity for (theme, theme_loc, en) if
        one is already present (e.g. seeded by another helper);
        creates a fresh one otherwise. The pri_relacio/subjekto/objekto
        shape mirrors `vidi_learns_en` so `scias_lokon_via_en` derives
        correctly.
      - Picks a readable concept by walking the lexicon for
        readability=legebla — auto-includes any future readable
        without per-concept logic here.
      - Picks a random in-trace location for the text. Could be the
        agent's room, the theme's room, or an unrelated sibling —
        the planner figures out the locomotion. Variety in placement
        produces multi-hop chains naturally.

    No-op if the theme isn't `en` something or no readables exist."""
    from ..causal import EntityInstance
    from ..sampler import _add_entity_randomized
    if theme_eid not in t.entities:
        return
    theme_loc = None
    for r in t.relations:
        if r.relation == "en" and r.args[0] == theme_eid:
            theme_loc = r.args[1]
            break
    if theme_loc is None:
        return
    fakto_concept = lex.concepts.get("fakto")
    if fakto_concept is None:
        return
    fakto_id = f"fakto_from_en_{theme_eid}_{theme_loc}"
    if fakto_id not in t.entities:
        t.entities[fakto_id] = EntityInstance(
            id=fakto_id, concept_lemma="fakto",
            entity_type=fakto_concept.entity_type,
            properties={"pri_relacio": ["en"]},
        )
        try:
            t.assert_relation("subjekto", (fakto_id, theme_eid), lex)
            t.assert_relation("objekto", (fakto_id, theme_loc), lex)
        except (KeyError, ValueError):
            return
    readables = [c.lemma for c in lex.concepts.values()
                 if "legebla" in c.properties.get("readability", [])]
    if not readables:
        return
    text_lemma = rng.choice(readables)
    locations = [eid for eid, e in t.entities.items()
                 if lex.types.is_subtype(e.entity_type, "location")]
    if not locations:
        return
    text_loc_eid = rng.choice(locations)
    text_id = text_lemma
    sfx = 0
    while text_id in t.entities:
        sfx += 1
        text_id = f"{text_lemma}_priskribas{sfx if sfx > 1 else ''}"
    try:
        _add_entity_randomized(t, text_lemma, lex, rng, entity_id=text_id)
    except (KeyError, ValueError):
        return
    # Strict placement via containment.jsonl. Readables (libro, letero)
    # need indoor hosts; outdoor pick falls back through the placer.
    placed = _place_respecting_containment(
        t, lex, text_loc_eid, text_lemma, rng,
        preferred_id=text_loc_eid, existing_eid=text_id)
    if placed is None:
        t.entities.pop(text_id, None)
        return
    try:
        t.assert_relation("priskribas", (text_id, fakto_id), lex)
    except (KeyError, ValueError):
        return


def _seed_agent_knowledge(t, agent_eid, lex) -> None:
    """For each `en` placement currently in the trace, create a fakto
    entity (mirroring vidi_learns_en) and assert konas(agent, fakto).
    Lets the planner skip the vidi → konas → scias_lokon subgoal chain
    that's needed for preni — the regression sampler models agents as
    knowing the scene's layout, not as discovering it.

    No-op when agent_eid is None or not in the trace."""
    from ..causal import EntityInstance
    if agent_eid is None or agent_eid not in t.entities:
        return
    en_pairs = [(r.args[0], r.args[1]) for r in t.relations
                if r.relation == "en"]
    fakto_concept = lex.concepts.get("fakto")
    if fakto_concept is None:
        return
    for subj, loc in en_pairs:
        fid = f"fakto_from_en_{subj}_{loc}"
        if fid in t.entities:
            continue
        t.entities[fid] = EntityInstance(
            id=fid, concept_lemma="fakto",
            entity_type=fakto_concept.entity_type,
            properties={"pri_relacio": ["en"]},
        )
        try:
            t.assert_relation("subjekto", (fid, subj), lex)
            t.assert_relation("objekto", (fid, loc), lex)
            t.assert_relation("konas", (agent_eid, fid), lex)
        except (KeyError, ValueError):
            continue


def _force_conditional_gates(t, action, role_eids: dict, lex, rng) -> None:
    """For each IfPropertyPrecondition on the target verb, force the
    gate to fire by setting the role's if_property=if_value AND set
    then_property to a non-target value so the planner has to subgoal
    on it. Skips when the role concept doesn't declare if_property
    (gate can't fire) or when then_property has no other vocabulary."""
    from ..schemas import IfPropertyPrecondition

    for pc in action.preconditions:
        if not isinstance(pc, IfPropertyPrecondition):
            continue
        eid = role_eids.get(pc.role)
        if eid is None:
            continue
        ent = t.entities.get(eid)
        if ent is None:
            continue
        role_concept = lex.concepts.get(ent.concept_lemma)
        if role_concept is None:
            continue
        if pc.if_property not in role_concept.properties:
            continue
        ent.set_property(pc.if_property, pc.if_value)
        then_slot = lex.slots.get(pc.then_property)
        if then_slot is None or not then_slot.vocabulary:
            continue
        adverse = [v for v in then_slot.vocabulary if v != pc.then_value]
        if not adverse:
            continue
        ent.set_property(pc.then_property, rng.choice(adverse))


def _concept_satisfies_role_props(concept, role, lex) -> bool:
    """Does this concept satisfy role.properties? Mirrors
    `_concepts_matching_role`'s per-slot checks but for a single
    (concept, role) pair. Used by the chain seeder to decide whether
    an existing role binding can be reused for a producer's role."""
    for slot, vals in role.properties.items():
        slot_def = lex.slots.get(slot)
        if slot_def is None:
            continue
        cvals = concept.properties.get(slot, [])
        if slot_def.varies:
            # Pervasive slots are always meaningful via default-derivation;
            # opt-in slots require concept-level declaration.
            if getattr(slot_def, "pervasive", False):
                continue
            if not cvals:
                return False
            continue
        if not (set(vals) & set(cvals)):
            return False
    return True


def _verbs_producing(lex, slot: str, value: str) -> list:
    """Verbs whose first effect writes (slot, value). Used by chain
    seeding to find the producer for a subgoaled property."""
    out = []
    for action in lex.actions.values():
        for eff in action.effects:
            if eff.property == slot and eff.value == value:
                out.append(action)
                break
    return out


def _action_might_need_light(action) -> bool:
    """True if this action's preconditions could backchain through
    vidi — meaning the planner might subgoal `illuminated=yes` on the
    agent. havi/scias_lokon/konas preconditions all chain to vidi
    eventually. Used to gate lamp-seeding so chains that never
    involve vidi don't get a useless lampo cluttering scene-setup."""
    from ..schemas import RelationPrecondition
    for pc in action.preconditions:
        if isinstance(pc, RelationPrecondition):
            if pc.rel in ("havi", "scias_lokon", "konas"):
                return True
    return False


def _seed_indoor_lamp(t, scene_id, away_id, lex, rng) -> None:
    """If a scene location is indoor, seed a lampo en that location.
    Without it, planner-subgoaled vidi (in chains like vidi → preni →
    ...) can't derive illuminated and the chain fails or only works
    when scene happens to be outdoor. Done for both scene_id and
    away_id since chains may need light in either."""
    from ..sampler import _add_entity_randomized
    if "lampo" not in lex.concepts:
        return
    for loc_id in (scene_id, away_id):
        if loc_id is None:
            continue
        loc_ent = t.entities.get(loc_id)
        if loc_ent is None:
            continue
        loc_concept = lex.concepts.get(loc_ent.concept_lemma)
        if loc_concept is None:
            continue
        if loc_concept.properties.get("indoor_outdoor") != ["interna"]:
            continue
        # Skip if there's already a lamp en this location.
        already_has = False
        for r in t.relations:
            if r.relation != "en" or r.args[1] != loc_id:
                continue
            inner = t.entities.get(r.args[0])
            if inner is None:
                continue
            inner_concept = lex.concepts.get(inner.concept_lemma)
            if (inner_concept is not None
                    and inner_concept.properties.get("lights_when_on")
                        == ["yes"]):
                already_has = True
                break
        if already_has:
            continue
        lamp_id = "lampo"
        suffix = 0
        while lamp_id in t.entities:
            suffix += 1
            lamp_id = f"lampo_{suffix}"
        try:
            _add_entity_randomized(t, "lampo", lex, rng, entity_id=lamp_id)
            t.assert_relation("en", (lamp_id, loc_id), lex)
        except (KeyError, ValueError):
            continue


def _seed_role_property_dependencies(t, action, role_eids: dict,
                                       lex, rng, derivations) -> None:
    """For each role.property the role-entity likely won't satisfy and
    that's producible only via a derivation chain (no verb directly
    writes it), walk derivations backward to find required entities
    and seed them into the scene.

    Mirrors `_seed_chain_dependencies` but for role.properties. Without
    this, vidi.agent.illuminated=yes never gets a lampo seeded —
    illuminated has no direct verb producer, only the
    `agent_illuminated` derivation, whose `given` requires the agent
    to be in a luma location, which (indoors) requires an aktiva lamp."""
    from ..sampler import _add_entity_randomized
    seen_keys: set = set()
    for role in action.roles:
        if not role.properties:
            continue
        eid = role_eids.get(role.name)
        if eid is None:
            continue
        ent = t.entities.get(eid)
        if ent is None:
            continue
        for slot, vals in role.properties.items():
            for value in vals:
                _ensure_property_satisfiable(
                    eid, slot, value, t, lex, rng, derivations,
                    seen_keys, depth=0)


def _ensure_property_satisfiable(target_eid, slot, value, t, lex, rng,
                                   derivations, seen_keys, depth):
    """Recursive helper for `_seed_role_property_dependencies`. If the
    property is already satisfied or randomizable to satisfy, skip.
    Otherwise walk derivations producing it; for each, walk `given`
    for required entity bindings and seed missing ones."""
    if depth > 4:
        return
    key = (target_eid, slot, value)
    if key in seen_keys:
        return
    seen_keys = seen_keys | {key}

    ent = t.entities.get(target_eid)
    if ent is None:
        return
    actual = ent.properties.get(slot, [])
    if value in actual:
        return
    slot_def = lex.slots.get(slot)
    if slot_def is not None and slot_def.varies and slot in ent.properties:
        # Will randomize at instance time; could land on `value`. Skip
        # — over-seeding here would force the value, defeating the
        # planner's chain-growth on randomization.
        return

    # If a verb directly writes (slot, value), the planner can subgoal
    # via that verb at runtime; no seeding needed here (chain
    # ingredients for the producer were already seeded by
    # _seed_chain_dependencies if relevant).
    if _slot_value_producible(slot, value, lex):
        return

    # Walk derivations producing (slot, value). For each, examine the
    # `given` clauses and seed required entities.
    from ..dsl.implications import PropertyImplication
    from ..dsl.patterns import (
        AndPattern, EntityPattern, RelPattern,
    )
    for d in derivations:
        for imp in d.implies:
            if not isinstance(imp, PropertyImplication):
                continue
            if imp.slot != slot or imp.value != value:
                continue
            # Bind imp.entity → target_eid. Walk d.given for rel
            # patterns; for each, the "other" arg is a free location
            # or entity that must exist with certain slot values.
            var_bindings: dict[int, str] = {id(imp.entity): target_eid}
            patterns = list(d.given)
            # Container-walking case: `rel("en", contained=A, container=L)`
            # binds L to wherever A is. Use that, then recurse on L's
            # required slot values from sibling entity patterns.
            container_var = None
            for p in patterns:
                if not isinstance(p, RelPattern):
                    continue
                if p.relation != "en":
                    continue
                # `contained` arg = imp.entity ?
                contained_pat = p.arg_patterns.get("contained")
                container_pat = p.arg_patterns.get("container")
                if contained_pat is None or container_pat is None:
                    continue
                contained_var = _bind_var_in_pattern(contained_pat)
                container_var_local = _bind_var_in_pattern(container_pat)
                if contained_var is None or container_var_local is None:
                    continue
                if id(contained_var) != id(imp.entity):
                    continue
                # Find target_eid's container in trace.
                for r in t.relations:
                    if r.relation == "en" and r.args[0] == target_eid:
                        var_bindings[id(container_var_local)] = r.args[1]
                        container_var = container_var_local
                        break
                break
            # For every entity-pattern in given that binds a known var
            # AND has slot constraints, recurse to ensure those slots.
            for p in patterns:
                for ep, bound_var in _walk_entity_patterns_with_binds(p):
                    if bound_var is None:
                        continue
                    bound_eid = var_bindings.get(id(bound_var))
                    if bound_eid is None:
                        continue
                    for k, v in ep.constraints.items():
                        if k in ("type", "concept", "has_suffix"):
                            continue
                        if not isinstance(v, str):
                            continue
                        _ensure_property_satisfiable(
                            bound_eid, k, v, t, lex, rng, derivations,
                            seen_keys, depth + 1)
            # If a free var (the entity producer) needs to be
            # introduced: find concepts matching the entity-pattern's
            # constraints and seed one en the bound location.
            if container_var is not None:
                container_eid = var_bindings[id(container_var)]
                for p in patterns:
                    if not isinstance(p, RelPattern) or p.relation != "en":
                        continue
                    contained_pat = p.arg_patterns.get("contained")
                    contained_var = _bind_var_in_pattern(contained_pat) \
                        if contained_pat is not None else None
                    if contained_var is None:
                        continue
                    if id(contained_var) == id(imp.entity):
                        continue   # original target's en, not a producer
                    if id(contained_var) in var_bindings:
                        continue
                    # Find this var's entity-pattern constraints in `given`.
                    constraints: dict[str, str] = {}
                    for q in patterns:
                        for ep, bv in _walk_entity_patterns_with_binds(q):
                            if bv is None or id(bv) != id(contained_var):
                                continue
                            for k, v in ep.constraints.items():
                                if isinstance(v, str):
                                    constraints[k] = v
                    if not constraints:
                        continue
                    # Find a concept whose properties satisfy all
                    # non-varies constraints (varies slots like
                    # power_state randomize, so any concept declaring
                    # the slot can satisfy at instance time).
                    candidate_concepts = []
                    for lemma, concept in lex.concepts.items():
                        match = True
                        for k, v in constraints.items():
                            if k in ("type", "concept", "has_suffix"):
                                continue
                            slot_d = lex.slots.get(k)
                            cvals = concept.properties.get(k, [])
                            if slot_d is not None and slot_d.varies:
                                if getattr(slot_d, "pervasive", False):
                                    pass  # type alone suffices
                                elif not cvals:
                                    match = False
                                    break
                            else:
                                if v not in cvals:
                                    match = False
                                    break
                        if match:
                            candidate_concepts.append(lemma)
                    if not candidate_concepts:
                        continue
                    chosen = rng.choice(candidate_concepts)
                    seed_eid = chosen
                    suffix = 0
                    while seed_eid in t.entities:
                        suffix += 1
                        seed_eid = f"{chosen}_{suffix}"
                    try:
                        _add_entity_randomized(
                            t, chosen, lex, rng, entity_id=seed_eid)
                        t.assert_relation(
                            "en", (seed_eid, container_eid), lex)
                    except (KeyError, ValueError):
                        continue


def _walk_entity_patterns_with_binds(pattern):
    """Yield (EntityPattern, bound_var) pairs from a composed pattern.
    Walks AndPatterns. Returns the EntityPattern alongside the Var
    it's bound to (via `& bind(V)`), if any."""
    from ..dsl.patterns import (
        AndPattern, BindPattern, EntityPattern,
    )
    if isinstance(pattern, EntityPattern):
        yield (pattern, None)
        return
    if isinstance(pattern, AndPattern):
        # Look for the canonical `entity(...) & bind(V)` shape.
        ep = None
        bv = None
        for side in (pattern.left, pattern.right):
            if isinstance(side, EntityPattern):
                ep = side
            elif isinstance(side, BindPattern):
                bv = side.target
        if ep is not None:
            yield (ep, bv)
        else:
            yield from _walk_entity_patterns_with_binds(pattern.left)
            yield from _walk_entity_patterns_with_binds(pattern.right)


_NON_LOCATION_CONTAINERS_ATTR = "_non_location_containers_by_concept"


def _non_location_containers_for(
    concept_lemma: str, lex,
) -> list[tuple[str, str]]:
    """Cached lookup of (container_lemma, relation) pairs for non-location
    containers the concept is afforded into. Most concepts have no such
    affordance (animates, locations, plain ingredients placed directly
    in rooms) — caching the lookup prevents re-walking the containment
    graph per role-entity. Result attached to the lexicon (immutable
    after load) so workers share via copy-on-fork."""
    cache = getattr(lex, _NON_LOCATION_CONTAINERS_ATTR, None)
    if cache is None:
        from ..containment import containers_for, resolve_containment
        idx = resolve_containment(lex)
        cache = {}
        for c_lemma in lex.concepts:
            non_loc = []
            for cont_lemma, rel in containers_for(c_lemma, idx, lex):
                cont_concept = lex.concepts.get(cont_lemma)
                if cont_concept is None or cont_concept.entity_type == "location":
                    continue
                non_loc.append((cont_lemma, rel))
            if non_loc:
                cache[c_lemma] = non_loc
        try:
            setattr(lex, _NON_LOCATION_CONTAINERS_ATTR, cache)
        except (AttributeError, TypeError):
            pass
    return cache.get(concept_lemma, [])


def _route_through_container(
    t, eid: str, concept_lemma: str | None, scene_id: str,
    lex, rng,
) -> bool:
    """Route an existing role-entity through any non-location container
    the concept affords (vestaĵo → valizo/ŝranko, surfaco-able → tablo,
    etc). Materializes the intermediate container under `scene_id` via
    the recursive placer so the chain `eid (rel) container, container en
    location` is fully linked.

    Returns True iff routing succeeded (eid placed). False means no
    non-location affordance fits and the caller should fall back to
    direct placement under scene_id. Caller-friendly: never raises;
    failed asserts roll back via try/except so the trace stays clean.

    Picking among multiple affordances is uniform — both en and sur
    cascade the same way (a clothing item lands `en` valizo, a glaso
    `sur` tablo)."""
    if concept_lemma is None:
        return False
    candidates = _non_location_containers_for(concept_lemma, lex)
    if not candidates:
        return False
    candidates = list(candidates)
    rng.shuffle(candidates)
    for cont_lemma, rel in candidates:
        cont_eid = _place_respecting_containment(
            t, lex, scene_id, cont_lemma, rng, preferred_id=scene_id)
        if cont_eid is None:
            continue
        if not t.is_relation_permitted(rel, (eid, cont_eid), lex):
            continue
        try:
            t.assert_relation(rel, (eid, cont_eid), lex)
            return True
        except (KeyError, ValueError):
            continue
    return False


def _place_respecting_containment(
    t, lex, scene_id, concept_lemma, rng, *, preferred_id=None,
    existing_eid: str | None = None,
    avoid: frozenset[str] | None = None,
    _depth: int = 0,
):
    """Place a `concept_lemma` instance under a container that
    `Trace.is_relation_permitted` accepts on both axes — afforded by
    containment.jsonl AND no `required` entry violated. Mirrors
    `assert_relation`'s strict check so seeders can't smuggle in
    placements that would silently fail under raise mode.

    When `existing_eid` is None, the entity is fresh: the function
    adds it (eid = concept_lemma), attempts placement, and rolls back
    (pop) on total failure. When `existing_eid` is provided, the
    entity is already in trace (added bare by an earlier seeder step
    that owns its lifecycle); this function only attempts placement
    and returns None without rolling back. The caller decides whether
    to remove the orphan.

    Resolution (existing_eid case skips Tier 0):
      0. (fresh only) Already in trace → return that eid.
      1. `preferred_id` (typically scene_id or away_id) — if a relation
         is permitted, place there.
      2. Any other in-trace entity — first valid container wins.
      3. Walk `containers_for(concept_lemma)` and recursively place a
         valid container, then put the entity inside it. Recursion
         climbs the containment graph (forno → kuirejo → domo) until
         it bottoms out at a top-level location, which gets placed
         apud the scene via Tier 4.
      4. Top-level location with no further container → place apud
         the scene (sibling). Only fires for location concepts; non-
         location orphans are rejected so the trace stays clean.
      5. None — every option failed; caller skips this producer.

    Uses `concept_lemma` as the eid for fresh entities (no suffix) —
    the cases that need this helper are unique-per-scene appliances/
    instruments and multiple fornoj in one scene have no narrative use.

    `_depth` guards recursion (caps at 4) under cyclic containment."""
    if _depth > 4:
        return None
    from ..containment import containers_for, resolve_containment
    from ..sampler import _add_entity_randomized

    if existing_eid is not None:
        eid = existing_eid
        added_here = False
    else:
        eid = concept_lemma
        if concept_lemma in t.entities:
            return concept_lemma
        _add_entity_randomized(t, concept_lemma, lex, rng,
                                entity_id=concept_lemma)
        added_here = True

    def _try_relation(container_eid: str):
        for rel in ("en", "sur"):
            if t.is_relation_permitted(
                    rel, (eid, container_eid), lex):
                return rel
        return None

    # Tier 1: preferred container.
    if preferred_id is not None and preferred_id in t.entities:
        rel = _try_relation(preferred_id)
        if rel is not None:
            try:
                t.assert_relation(rel, (eid, preferred_id), lex)
                return eid
            except (KeyError, ValueError):
                pass

    # Tier 2: any other in-trace entity (except `avoid`-listed ones,
    # used by callers that want the concept to land in its natural
    # habitat rather than fall back to the scene as a default
    # container).
    avoid = avoid or frozenset()
    for other_eid in list(t.entities.keys()):
        if other_eid in (eid, preferred_id):
            continue
        if other_eid in avoid:
            continue
        rel = _try_relation(other_eid)
        if rel is None:
            continue
        try:
            t.assert_relation(rel, (eid, other_eid), lex)
            return eid
        except (KeyError, ValueError):
            continue

    # Tier 3: materialize a container, then place inside.
    # Pre-filter at the lemma level (containers_for is broad — it
    # returns every concept with an affordance entry, so requirement-
    # only kuirejo gets buried among 9+ generic-room siblings). Without
    # the pre-filter we'd materialize each one as a sibling before
    # discovering only one survives the required-fact check.
    #
    # The container itself is placed via a RECURSIVE call. Walks the
    # containment graph to find a valid host for it (kuirejo → domo;
    # domo → no containers, so falls to Tier 4 which places it apud
    # the scene). Generalizes one-off "sibling apud scene" to arbitrary
    # depth: forno → kuirejo → domo → apud scene.
    from ..containment import (
        containment_relations_for, required_fact_violations,
    )
    idx = resolve_containment(lex)
    for cont_lemma, _afforded_rel in containers_for(
            concept_lemma, idx, lex):
        cont_concept = lex.concepts.get(cont_lemma)
        if cont_concept is None or cont_concept.entity_type != "location":
            continue
        valid_rel: Optional[str] = None
        for candidate_rel in containment_relations_for(
                cont_lemma, concept_lemma, idx, lex):
            if not required_fact_violations(
                    cont_lemma, concept_lemma, candidate_rel, idx, lex):
                valid_rel = candidate_rel
                break
        if valid_rel is None:
            continue
        cont_eid = _place_respecting_containment(
            t, lex, scene_id, cont_lemma, rng, preferred_id=scene_id,
            _depth=_depth + 1)
        if cont_eid is None:
            continue
        try:
            t.assert_relation(valid_rel, (eid, cont_eid), lex)
            return eid
        except (KeyError, ValueError):
            continue

    # Tier 4: top-level location with no valid container of its own
    # (e.g. domo) → place as a sibling apud the scene. Only fires for
    # location concepts; non-locations without a container are
    # genuinely orphan-prone and we'd rather skip the producer than
    # smuggle them into the scene without containment.
    concept_def = lex.concepts.get(concept_lemma)
    if (concept_def is not None
            and concept_def.entity_type == "location"
            and eid != scene_id):
        try:
            t.assert_relation("apud", (eid, scene_id), lex)
            return eid
        except (KeyError, ValueError):
            pass

    # Roll back the entity ONLY if we added it here. When the caller
    # passed `existing_eid`, they own the entity's lifecycle and must
    # decide whether to remove it on placement failure.
    if added_here:
        t.entities.pop(eid, None)
    return None


def _seed_role_chain_dependencies(t, action, role_eids: dict,
                                    scene_id: str, lex, rng, *,
                                    away_id: str | None = None,
                                    seen: set | None = None) -> None:
    """For each role.property the role-entity doesn't satisfy, find
    verbs that write (slot, value) and seed their missing roles into
    the scene so the planner can subgoal the producer.

    Mirrors `_seed_chain_dependencies`, which handles
    IfPropertyPreconditions; this one handles role.properties on the
    target action — the case where dormi.agent requires posture=
    kuŝanta and the planner would subgoal kuŝi, which itself needs a
    lieable support in scene. Recursive: producer's own role.properties
    get covered too. `seen` tracks visited verbs to bound recursion.

    Reuses the actor (agent eid) for any agent-role on the producer,
    since the actor is the one trying to satisfy their own role
    requirement (kuŝi's agent IS the dormi agent)."""
    from ..sampler import _add_entity_randomized
    seen = (seen or set()) | {action.lemma}
    placement_id = away_id if (away_id and away_id != scene_id) else scene_id

    for role in action.roles:
        if not role.properties:
            continue
        eid = role_eids.get(role.name)
        if eid is None:
            continue
        ent = t.entities.get(eid)
        if ent is None:
            continue
        for slot, vals in role.properties.items():
            for value in vals:
                if value in ent.properties.get(slot, []):
                    continue
                for producer in _verbs_producing(lex, slot, value):
                    if producer.lemma in seen:
                        continue
                    prod_role_eids = dict(role_eids)
                    # When the property target is the producer's effect
                    # target, bind that role to the eid we're satisfying
                    # (kuŝi's agent IS dormi's agent).
                    for eff in producer.effects:
                        if (eff.property == slot and eff.value == value
                                and eff.target_role not in prod_role_eids):
                            prod_role_eids[eff.target_role] = eid
                    ok = True
                    for p_role in producer.roles:
                        existing = prod_role_eids.get(p_role.name)
                        if existing is not None:
                            ent_p = t.entities.get(existing)
                            role_concept = (
                                lex.concepts.get(ent_p.concept_lemma)
                                if ent_p else None)
                            if (ent_p is not None
                                    and lex.types.is_subtype(
                                        ent_p.entity_type, p_role.type)
                                    and role_concept is not None
                                    and _concept_satisfies_role_props(
                                        role_concept, p_role, lex)):
                                continue
                        cands = _concepts_matching_role(lex, p_role)
                        if not cands:
                            ok = False
                            break
                        concept_lemma = rng.choice(cands)
                        eid_new = _place_respecting_containment(
                            t, lex, scene_id, concept_lemma, rng,
                            preferred_id=placement_id)
                        if eid_new is None:
                            ok = False
                            break
                        prod_role_eids[p_role.name] = eid_new
                    if not ok:
                        continue
                    _seed_role_chain_dependencies(
                        t, producer, prod_role_eids, scene_id, lex, rng,
                        away_id=away_id, seen=seen)
                    break


def _seed_chain_dependencies(t, action, role_eids: dict, scene_id: str,
                              lex, rng, *, away_id: str | None = None,
                              seen: set | None = None) -> None:
    """For each conditional precondition on `action` whose gate could
    fire post-randomization, find verbs producing the required state
    and seed their missing roles into the scene. Recurses on producer
    verbs so multi-hop chains land. `seen` tracks visited verb lemmas
    to bound recursion under cyclic preconditions.

    `away_id` (when given) is a second location distinct from
    `scene_id`. New entities seeded here go `en` away_id so the
    planner has to subgoal on locomotion/retrieval rather than just
    binding everything in one room. Falls back to scene_id when
    away_id is None or equals scene_id.

    Currently handles IfPropertyPrecondition only — the live case for
    the lock chain. RelationPrecondition seeding (e.g. needing a
    container so havi can be established) is a future extension."""
    from ..sampler import _add_entity_randomized
    from ..schemas import IfPropertyPrecondition
    seen = (seen or set()) | {action.lemma}
    placement_id = away_id if (away_id and away_id != scene_id) else scene_id

    for pc in action.preconditions:
        if not isinstance(pc, IfPropertyPrecondition):
            continue
        gate_eid = role_eids.get(pc.role)
        if gate_eid is None:
            continue
        gate_concept = lex.concepts.get(t.entities[gate_eid].concept_lemma)
        if gate_concept is None:
            continue
        # Pessimistic firing: if the role concept declares if_property,
        # the gate could fire after randomization. Seed for it.
        if pc.if_property not in gate_concept.properties:
            continue
        for producer in _verbs_producing(
                lex, pc.then_property, pc.then_value):
            if producer.lemma in seen:
                continue
            prod_role_eids = dict(role_eids)
            for p_role in producer.roles:
                existing = prod_role_eids.get(p_role.name)
                if existing is not None:
                    ent = t.entities[existing]
                    role_concept = lex.concepts.get(ent.concept_lemma)
                    if (lex.types.is_subtype(ent.entity_type, p_role.type)
                            and role_concept is not None
                            and _concept_satisfies_role_props(
                                role_concept, p_role, lex)):
                        continue   # reusable for producer's role
                cands = _concepts_matching_role(lex, p_role)
                if not cands:
                    continue
                concept_lemma = rng.choice(cands)
                eid = _place_respecting_containment(
                    t, lex, scene_id, concept_lemma, rng,
                    preferred_id=placement_id)
                if eid is None:
                    continue
                prod_role_eids[p_role.name] = eid
            _seed_chain_dependencies(
                t, producer, prod_role_eids, scene_id, lex, rng,
                away_id=away_id, seen=seen)


def sample_regression_scene(lex, rng, *, rules=None):
    """Pick a verb uniformly from the lexicon-derived eligible pool and
    regress a scene for it. Retries up to a few times if a verb's
    regression fails; returns None if every attempt fails.

    Two pools are interleaved: (a) property-effect verbs from
    `_regression_verb_pool` (handled by `regress_for_verb`) and
    (b) verbs whose causal rules add the `konas` relation (handled
    by `regress_for_knowledge_verb`). Knowledge-transfer chains
    naturally surface from (b) — e.g. picking rakonti exercises the
    `vidi → rakonti` chain when no agent yet knows the fakto."""
    prop_pool = _regression_verb_pool(lex)
    konas_pool = (sorted(_verbs_adding_konas(rules) & set(lex.actions))
                   if rules else [])
    pool = prop_pool + konas_pool
    if not pool:
        return None
    for _ in range(8):
        # Bare-location seeders (movement, vehicle) used to fire here.
        # Removed: their drives ("AGENT wants to be in LOC") were
        # single-event walking scenes that dominated the dedup head
        # (~28% of all drives). Locomotion verbs still surface as
        # planner-discovered subgoals when the chosen drive's theme
        # lives in a different room. Vehicles and intransitive
        # movement re-enter scenes via SceneBuilder's loose-tools
        # decoration hook (see _seed_loose_instruments).
        # Cascade-driven self_slot scenes (hunger=sata, thirst=satigita,
        # …): cascade verbs (satiĝi, sensoifiĝi) have no agent role so
        # `regress_for_verb` can't seed them. `regress_for_self_slot`
        # introspects rules to find the trigger verb and walks the
        # containment graph to place its theme. Adding a new agent-
        # role cascade rule (e.g. tired_sleeps_rested) automatically
        # grows coverage.
        if rules is not None and rng.random() < 0.20:
            self_slot_pairs = _self_slot_drive_pairs(rules, lex)
            if self_slot_pairs:
                slot, target = rng.choice(self_slot_pairs)
                result = regress_for_self_slot(
                    slot, target, lex, rng, rules)
                if result is not None:
                    return result
        # Clothing chains: surmeti / demeti. Drive: actor wants to
        # be wearing a specific garment located elsewhere.
        if rng.random() < 0.08:
            result = regress_for_clothing(lex, rng)
            if result is not None:
                return result
        # Count-target chains: actor wants N units of a stack.
        # Forces preni or peti from a stack-owner.
        if rng.random() < 0.08:
            result = regress_for_count(lex, rng)
            if result is not None:
                return result
        # Generosity count chain: donor gives the recipient N units.
        # Surfaces doni with quantity-aware partial transfer.
        if rng.random() < 0.06:
            result = regress_for_give_count(lex, rng)
            if result is not None:
                return result
        # Comparative count: actor wants strictly more units than a
        # reference person has. Target resolves to ref.count + 1 at
        # plan time.
        if rng.random() < 0.06:
            result = regress_for_more_than(lex, rng)
            if result is not None:
                return result
        # Buy: buyer with money, seller with goods elsewhere. Same
        # count drive, but money in scope unlocks aĉeti.
        if rng.random() < 0.06:
            result = regress_for_buy(lex, rng)
            if result is not None:
                return result
        # Sell: seller with goods, buyer with money elsewhere. Uses
        # give_count drive to surface vendi via altruism path.
        if rng.random() < 0.06:
            result = regress_for_sell(lex, rng)
            if result is not None:
                return result
        # Weighted pick: chain-rich verbs (more preconditions / role
        # constraints) win the draw more often, so the head of the
        # drive distribution leans toward multi-step scenes instead
        # of bare single-event ones.
        weights = [_verb_chain_weight(lex.actions[v]) for v in pool]
        verb = rng.choices(pool, weights=weights, k=1)[0]
        if verb in konas_pool:
            result = regress_for_knowledge_verb(verb, lex, rng, rules)
        else:
            result = regress_for_verb(verb, lex, rng)
        if result is not None:
            return result
    return None


def regress_for_knowledge_verb(verb_name, lex, rng, rules):
    """Procedural seeder for knowledge drives. Replaces the trio of
    `regress_for_demandi` / `regress_for_legi` / `regress_for_konas_verb`.

    Introspects the verb's konas-adding rule (via `konas_adding_verbs`)
    plus its action preconditions to:
      - decide drive shape: self-learn (knower=actor) when the rule's
        konas binds the agent (legi/demandi/vidi/flari/audi), or
        altruistic-teach (knower=recipient) when the rule binds the
        recipient (rakonti/instrui/montri/respondi).
      - infer scaffolding:
          * `priskribas(text, fakto)` in rule.given → pre-place a
            readable text linked to the fakto (legi).
          * `konas(recipient, theme)` in action.preconditions → pre-
            place a knower with konas pre-asserted (demandi).
        Other preconditions (samloke) handled by placement.

    Adding a new konas-adding verb to the rule library picks up
    regression coverage automatically."""
    from ..dsl.introspect import konas_adding_verbs
    from ..dsl.patterns import RelPattern
    from ..schemas import RelationPrecondition

    if lex.actions.get(verb_name) is None:
        return None
    specs = [s for s in konas_adding_verbs(rules) if s.verb == verb_name]
    if not specs:
        return None
    spec = rng.choice(specs)

    builder = (scene(lex, rng)
        .location("here", is_scene=True)
        .location("there", different_from="here")
        .person("actor", in_="here")
        .target("item", in_="there")
        .fakto("fact", about=("en", "item", "there")))

    if spec.knower_role == "recipient":
        builder = builder.person("recipient", in_="here")
        drive_knower = "recipient"
    else:
        drive_knower = "actor"

    # Scaffolding from rule.given: priskribas pre-link for legi.
    for given_pat in spec.given_rels:
        if (isinstance(given_pat, RelPattern)
                and given_pat.relation == "priskribas"):
            text_loc = "here" if rng.random() < 0.5 else "there"
            builder = (builder
                .readable("text", in_=text_loc)
                .priskribas("text", "fact"))

    # Scaffolding from action preconditions: pre-konas a recipient
    # for demandi (recipient must already konas the fakto for the
    # extraction to make sense).
    action = lex.actions[verb_name]
    for pc in action.preconditions:
        if (isinstance(pc, RelationPrecondition)
                and pc.rel == "konas"
                and len(pc.roles) == 2
                and pc.roles[0] == "recipient"
                and pc.roles[1] == "theme"):
            builder = (builder
                .person("knower", in_="there")
                .konas("knower", "fact"))

    return (builder
        .drive("knowledge", actor="actor",
               knower=drive_knower, fakto="fact")
        .build())


def regress_for_vehicle(lex, rng):
    """Person actor + vehicle in scene, location drive. veturi never
    surfaces from the verb pool (no effects, no konas, animal-only
    movement seeder excludes persons), so this seeder closes the gap.
    Motorized vehicles bring motoro.power_state=neaktiva so the
    planner naturally chains ŝalti before veturi."""
    return (scene(lex, rng)
        .vehicle("car")
        .location("here", is_scene=True, terrain_compatible_with="car")
        .location("there", different_from="here",
                  terrain_compatible_with="car")
        .relation("en", "car", "here")
        .person("actor", in_="here")
        .drive("location", actor="actor", loc="there")
        .build())


def regress_for_movement(lex, rng):
    """Pick a non-person animate actor and drive a location goal.
    Surfaces flugi / naĝi for fliers / swimmers — verbs the konas-
    and property-effect seeders never reach because those pick
    persons (for konas) or pick the actor implicitly via a verb's
    role pool (which is usually any-animate but skewed to persons
    by sheer concept count)."""
    return (scene(lex, rng)
        .location("here", is_scene=True)
        .location("there", different_from="here")
        .animal("actor", in_="here")
        .drive("location", actor="actor", loc="there")
        .build())


def regress_for_count(lex, rng):
    """Place actor in scene with one or two stacks of a countable food
    elsewhere — sometimes a single surplus stack (5 apples for a
    target of 3), sometimes two stacks split across two owners that
    must both be acquired to reach the target. Drive: actor wants
    target_count units; planner loops over sources via preni/peti.

    Three setup variants for variety:
      A. Single source with exactly target_count.
      B. Single source with count > target_count (surplus).
      C. Two sources owned by different persons, each with a partial
         count, summing to >= target_count."""
    countable_food = lambda c: (
        "manĝebla" in c.properties.get("edibility", [])
        and "1" in c.properties.get("count", []))
    target = rng.choice([2, 3, 4, 5])
    surplus = rng.random() < 0.50    # 50% exact, 50% surplus
    stash_count = (target + rng.randint(1, 3)) if surplus else target

    builder = (scene(lex, rng)
        .location("here", is_scene=True)
        .location("there", different_from="here")
        .person("actor", in_="here")
        .person("owner", in_="there")
        .target("stash", in_="there", where=countable_food)
        .havi("owner", "stash")
        .set("stash", count=str(stash_count)))

    # Two-source split (acquire from two owners) was prototyped but
    # caused planner blowups (>60s per scene) — the second
    # acquisition's plan_to_establish_relation call from a post-first-
    # acquisition state is much more expensive than from scratch.
    # Surplus alone gives the user the "more than needed" variety
    # without the planner cost.

    return (builder
        .drive("count", actor="actor", concept="stash", target=target)
        .build())


def regress_for_give_count(lex, rng):
    """Generosity scene: donor in scene with a stash; recipient
    elsewhere. Drive: recipient should have target_count units.
    Planner picks doni — donor is the planner-actor and binds to
    agent; recipient receives. Surplus stashes (count > target)
    surface partial transfer ("Maria donis du el siaj kvin pomoj
    al Petro")."""
    countable_food = lambda c: (
        "manĝebla" in c.properties.get("edibility", [])
        and "1" in c.properties.get("count", []))
    target = rng.choice([2, 3, 4, 5])
    surplus = rng.random() < 0.50
    stash_count = (target + rng.randint(1, 3)) if surplus else target
    return (scene(lex, rng)
        .location("here", is_scene=True)
        .location("there", different_from="here")
        .person("donor", in_="here")
        .person("recipient", in_="there")
        .target("stash", in_="here", where=countable_food)
        .havi("donor", "stash")
        .set("stash", count=str(stash_count))
        .drive("give_count", donor="donor", recipient="recipient",
               concept="stash", target=target)
        .build())


def regress_for_more_than(lex, rng):
    """Comparative count: actor wants strictly more units of a
    concept than `reference` has. Setup: reference already owns a
    small stash; a third stash sits with a neutral owner elsewhere.
    Target is computed at plan time (reference.count + 1), so the
    planner adapts to whatever count the reference happens to hold."""
    countable_food = lambda c: (
        "manĝebla" in c.properties.get("edibility", [])
        and "1" in c.properties.get("count", []))
    ref_count = rng.choice([1, 2, 3])
    # Source stash must have at least ref_count + 1 to be reachable.
    source_count = ref_count + rng.choice([1, 2, 3])
    return (scene(lex, rng)
        .location("here", is_scene=True)
        .location("there", different_from="here")
        .person("actor", in_="here")
        .person("reference", in_="here")
        .person("owner", in_="there")
        .target("source_stash", in_="there", where=countable_food)
        .havi("owner", "source_stash")
        .set("source_stash", count=str(source_count))
        .target("ref_stash", in_="here",
                same_concept_as="source_stash")
        .havi("reference", "ref_stash")
        .set("ref_stash", count=str(ref_count))
        .drive("more_than", actor="actor", concept="source_stash",
               reference="reference")
        .build())


def regress_for_buy(lex, rng):
    """Buyer in scene with money; seller elsewhere with countable goods.
    Drive: buyer wants `target_count` units of the goods. Planner picks
    aĉeti naturally (chain-richness weighting prefers verbs with more
    preconditions — aĉeti needs samloke + havi(seller, goods) + havi
    (buyer, money), strictly more than preni/peti). 1:1 economy: N
    money units pay for N goods units."""
    countable_food = lambda c: (
        "manĝebla" in c.properties.get("edibility", [])
        and "1" in c.properties.get("count", []))
    target = rng.choice([2, 3, 4, 5])
    surplus = rng.random() < 0.50
    stash_count = (target + rng.randint(1, 3)) if surplus else target
    money_count = target + rng.randint(0, 3)
    return (scene(lex, rng)
        .location("here", is_scene=True)
        .location("there", different_from="here")
        .person("buyer", in_="here")
        .person("seller", in_="there")
        .target("goods", in_="there", where=countable_food)
        .havi("seller", "goods")
        .set("goods", count=str(stash_count))
        .target("money", in_="here", concept="monero")
        .havi("buyer", "money")
        .set("money", count=str(money_count))
        .drive("count", actor="buyer", concept="goods", target=target)
        .build())


def regress_for_sell(lex, rng):
    """Seller in scene with countable goods; buyer elsewhere with
    money. Drive: buyer should end up with `target_count` units of the
    goods. Planner-actor is the seller (altruism flavor: drive's
    target_owner is buyer, planner-actor is seller), so vendi —
    where the seller binds to agent — surfaces over doni/preni."""
    countable_food = lambda c: (
        "manĝebla" in c.properties.get("edibility", [])
        and "1" in c.properties.get("count", []))
    target = rng.choice([2, 3, 4, 5])
    surplus = rng.random() < 0.50
    stash_count = (target + rng.randint(1, 3)) if surplus else target
    money_count = target + rng.randint(0, 3)
    return (scene(lex, rng)
        .location("here", is_scene=True)
        .location("there", different_from="here")
        .person("seller", in_="here")
        .person("buyer", in_="there")
        .target("goods", in_="here", where=countable_food)
        .havi("seller", "goods")
        .set("goods", count=str(stash_count))
        .target("money", in_="there", concept="monero")
        .havi("buyer", "money")
        .set("money", count=str(money_count))
        .drive("give_count", donor="seller", recipient="buyer",
               concept="goods", target=target)
        .build())


def regress_for_clothing(lex, rng):
    """Place actor in scene; clothing item in another location.
    Drive: actor wants vestita(actor, garment). Planner chains
    locomotion → preni → surmeti."""
    is_clothing = lambda c: "yes" in c.properties.get("is_clothing", [])
    return (scene(lex, rng)
        .location("here", is_scene=True)
        .location("there", different_from="here")
        .person("actor", in_="here")
        .target("garment", in_="there", where=is_clothing)
        .drive("wearing", actor="actor", garment="garment")
        .build())


def _self_slot_drive_pairs(rules, lex) -> list[tuple[str, Any]]:
    """(slot, target_value) pairs eligible for `regress_for_self_slot`.

    Three sources:
      1. Cascade rules where the AGENT-role's slot flips (hungry_eats_sated:
         agent.hunger=sata via manĝi). Theme-targeting cascades (fali →
         integrity=rompita on the falling theme) excluded — no entity
         drives wanting itself broken.
      2. Direct-effect verbs marked `reflexive_ok` whose effect writes
         the THEME's slot (sekigi: theme.wetness=seka, reflexive_ok=True).
         The actor binds to both agent and theme, "Maria sekigis sin."
         Lets a wet actor naturally drive the self-drying chain when
         a tuko is reachable.
      3. Direct-effect verbs whose effect writes the AGENT's own slot
         (dormi: agent.sleep_state=dormanta; sidi: agent.posture=sidanta).
         No theme to scatter — just an actor with the opposite pre-state
         flipping themselves via the verb. Picks up displeased self-drives
         that depend on context, like sleep at night."""
    from ..dsl.introspect import self_slot_cascades
    out: set[tuple[str, Any]] = set()
    for key, specs in self_slot_cascades(rules).items():
        if any(s.agent_role == "agent" and s.pre_state is not None
               for s in specs):
            out.add(key)
    for action in lex.actions.values():
        if not getattr(action, "reflexive_ok", False):
            continue
        for eff in action.effects:
            if eff.target_role == "theme":
                out.add((eff.property, eff.value))
    # Path 3 (direct agent-slot effect: dormi/vekiĝi/sidi/stari/kuŝi)
    # is intentionally NOT included as a drive source. Those verbs
    # produce single-event scenes ("Maria vekiĝas.", "Maria dormas.")
    # that clutter the dedup head with no causal richness. The verbs
    # still surface as planner-discovered side-effects: sleep verbs
    # via cascade preconditions, posture verbs via locomotion gates
    # (iri requires staranta, etc.).
    return list(out)


def regress_for_self_slot(slot, target_value, lex, rng, rules):
    """Procedural seeder for `("self_slot", actor, slot, target_value)`.

    Three satisfaction paths:
      1. Cascade verb (hunger=sata via manĝi, thirst=satigita via
         trinki). The cascade rule's `then` flips the agent's slot;
         the seeder sets the actor's pre-state from the `when`
         entity-pattern and scatters the trigger verb's theme in
         "away" pressure so the chain includes locomotion + havi.
      2. Reflexive verb (wetness=seka via sekigi sin). The verb's
         direct effect writes the theme's slot; the seeder binds
         theme==agent==actor and scatters the required INSTRUMENT
         (tuko) in "near" pressure (sekigi can't fetch-and-return).
         Actor's pre-state comes from the theme role's properties.
      3. Direct agent-slot verb (sleep_state=dormanta via dormi,
         posture=sidanta via sidi). No theme to place — the verb's
         effect writes the agent itself; the seeder builds an actor
         with the opposite slot value and lets the planner satisfy
         the drive in one step. Forces the world into the state where
         this slot's preference matches the target (e.g. mondo.
         tempo_de_tago=nokto for sleep_state=dormanta) so context
         drive-coupling is consistent.

    Tried in order; first match wins."""
    from ..dsl.introspect import self_slot_cascades

    cascades = self_slot_cascades(rules).get((slot, target_value), [])
    cascades = [s for s in cascades if s.agent_role == "agent"
                and s.pre_state is not None]
    if cascades:
        return _seed_cascade_self_slot(
            cascades, slot, target_value, lex, rng)

    # Reflexive fallback: find a reflexive_ok verb whose theme effect
    # writes the goal slot/value. Set actor's pre-state from theme
    # role's required properties.
    refl_actions = [
        a for a in lex.actions.values()
        if getattr(a, "reflexive_ok", False)
        and any(eff.target_role == "theme"
                and eff.property == slot and eff.value == target_value
                for eff in a.effects)
    ]
    if refl_actions:
        return _seed_reflexive_self_slot(
            rng.choice(refl_actions), slot, target_value, lex, rng)

    # Agent-direct fallback: find an action whose effect writes the
    # agent's own slot to the target value (dormi → sleep_state=dormanta).
    agent_actions = [
        a for a in lex.actions.values()
        if "agent" in {r.name for r in a.roles}
        and any(eff.target_role == "agent"
                and eff.property == slot and eff.value == target_value
                for eff in a.effects)
    ]
    if agent_actions:
        return _seed_agent_self_slot(
            rng.choice(agent_actions), slot, target_value, lex, rng)
    return None


def _seed_cascade_self_slot(cascades, slot, target_value, lex, rng):
    spec = rng.choice(cascades)
    action = lex.actions.get(spec.verb)
    if action is None:
        return None
    builder = (scene(lex, rng)
        .location("here", is_scene=True)
        .person("actor", in_="here"))
    for role in action.roles:
        if role.name == spec.agent_role or role.name == "instrument":
            continue
        candidates = set(_concepts_matching_role(lex, role))
        if not candidates:
            return None
        builder = builder.scatter(
            role.name, where=lambda c, allowed=candidates: c.lemma in allowed,
            pressure="away")
    return (builder
        .set("actor", **{slot: spec.pre_state})
        .drive("self_slot", actor="actor", slot=slot, value=target_value)
        .build())


def _seed_reflexive_self_slot(action, slot, target_value, lex, rng):
    """Build a scene where actor (in pre-state) wants to flip their
    own slot via a reflexive verb. Pre-state preference order:
      1. theme role spec (sekigi.theme requires wetness=malseka — the
         verb fires only on wet things, so pre_state is constrained).
      2. slot vocabulary minus target (purigi.theme doesn't constrain
         cleanliness — any solid thing is cleanable — so we pick a
         non-target value from the slot's declared vocabulary).
    Without (2) verbs whose theme has no slot-side gate (purigi,
    future lavi) couldn't surface a self-slot drive at all.

    Instrument scattered "near" because reflexive purigi/sekigi can't
    easily fetch-and-return without breaking samloke."""
    theme_role = next((r for r in action.roles if r.name == "theme"), None)
    if theme_role is None:
        return None
    pre_values = list(theme_role.properties.get(slot, []))
    if not pre_values:
        slot_def = lex.slots.get(slot)
        if slot_def is None or not slot_def.vocabulary:
            return None
        pre_values = [v for v in slot_def.vocabulary if v != target_value]
    if not pre_values:
        return None
    pre_state = rng.choice(pre_values)

    builder = (scene(lex, rng)
        .location("here", is_scene=True)
        .person("actor", in_="here"))
    for role in action.roles:
        if role.name in ("agent", "theme"):
            continue
        # Scatter instruments and other roles in-scene so the chain
        # stays samloke-safe through the reflexive action.
        candidates = set(_concepts_matching_role(lex, role))
        if not candidates:
            return None
        builder = builder.scatter(
            role.name, where=lambda c, allowed=candidates: c.lemma in allowed,
            pressure="near")
    return (builder
        .set("actor", **{slot: pre_state})
        .drive("self_slot", actor="actor", slot=slot, value=target_value)
        .build())


def _seed_agent_self_slot(action, slot, target_value, lex, rng):
    """Build a scene where actor (in opposite pre-state) wants to flip
    their own slot via a verb whose effect writes the AGENT directly
    (dormi → sleep_state=dormanta). Pre-state is sampled from the
    slot's vocabulary minus the target.

    Calls `_nudge_world_for_pref` after build to align the trace-wide
    `mondo` state with the drive (e.g. force tempo_de_tago=nokto for
    sleep_state=dormanta) so the rendered scene reads as causally
    motivated. The drive itself doesn't depend on the nudge — the
    planner satisfies it regardless of mondo state."""
    slot_def = lex.slots.get(slot)
    if slot_def is None or not slot_def.vocabulary:
        return None
    pre_values = [v for v in slot_def.vocabulary if v != target_value]
    if not pre_values:
        return None
    pre_state = rng.choice(pre_values)

    builder = (scene(lex, rng)
        .location("here", is_scene=True)
        .person("actor", in_="here"))
    for role in action.roles:
        if role.name == "agent":
            continue
        candidates = set(_concepts_matching_role(lex, role))
        if not candidates:
            return None
        builder = builder.scatter(
            role.name, where=lambda c, allowed=candidates: c.lemma in allowed,
            pressure="near")
    result = (builder
        .set("actor", **{slot: pre_state})
        .drive("self_slot", actor="actor", slot=slot, value=target_value)
        .build())
    if result is None:
        return None
    trace, scene_id, drive = result
    actor_eid = drive[1]
    # Seed chain ingredients for any agent role.properties the actor
    # doesn't satisfy (dormi.agent.posture=kuŝanta → seed a lieable
    # support so the planner can subgoal kuŝi). Without this, plans
    # for sleep almost never fire because random scenes don't contain
    # a lito/sofo/kuseno.
    _seed_role_chain_dependencies(
        trace, action, {"agent": actor_eid}, scene_id, lex, rng)
    _nudge_world_for_pref(trace, slot, target_value, lex)
    return trace, scene_id, drive


def _nudge_world_for_pref(trace, slot, target_value, lex) -> None:
    """Mutate `mondo`'s scalar slots so that
    `effective_preferences(trace)[slot] == target_value`. No-op when
    no mondo entity, no slot/value combo achieves it, or already
    matching. Used by the agent-direct self-slot seeder to align the
    world with the drive (e.g. force nokto for sleep_state=dormanta).

    Introspection-only: no per-slot special cases. Adding a new
    preference override in `effective_preferences` automatically
    becomes a possible nudge target here."""
    from ..agent.preferences import effective_preferences
    if effective_preferences(trace).get(slot) == target_value:
        return
    mondo = trace.entities.get("mondo")
    if mondo is None:
        return
    for prop in list(mondo.properties.keys()):
        slot_def = lex.slots.get(prop)
        if (slot_def is None or not slot_def.vocabulary
                or not slot_def.scalar):
            continue
        original = list(mondo.properties.get(prop, []))
        for v in slot_def.vocabulary:
            mondo.set_property(prop, v)
            if effective_preferences(trace).get(slot) == target_value:
                return
        if original:
            mondo.properties[prop] = original

