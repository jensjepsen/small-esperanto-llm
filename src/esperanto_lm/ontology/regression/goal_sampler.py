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
from ..dsl.rules import DEFAULT_DSL_DERIVATIONS, RUNTIME_DERIVATIONS
from .goals import build_goal_index

# Both bake-time and runtime derivations need to be visible to the
# bounded precondition-closure: `illuminated` lives in
# RUNTIME_DERIVATIONS, `lit_state` in DEFAULT_DSL_DERIVATIONS.
_ALL_DERIVATIONS = list(DEFAULT_DSL_DERIVATIONS) + list(RUNTIME_DERIVATIONS)


# Cache the index per lexicon — building it walks every action and
# rule once.
_GOAL_INDEX_CACHE: dict[int, dict] = {}

# Plain relation goals we promote into drives (besides CREATED-role
# event_fire). Map relation name → drive kind understood by the
# planner's `_drive_to_goal`. The actor must be the first arg of
# the goal's role_tuple (we only emit drives where the agent is
# the subject of the relation).
_PLAIN_RELATION_DRIVES: dict[str, str] = {
    "havi": "possession",
    "en": "location",
    "vestita": "wearing",
}

# Diagnostic channel: when regress_for_goal returns None, the last
# return path tags itself here. Bench / debug scripts can read this
# after each call to classify no_sample causes.
LAST_NO_SAMPLE_REASON: str | None = None


def _bail(reason: str) -> None:
    global LAST_NO_SAMPLE_REASON
    LAST_NO_SAMPLE_REASON = reason


def _cached_goal_index(lex, rules):
    key = id(lex)
    cached = _GOAL_INDEX_CACHE.get(key)
    if cached is None:
        cached = build_goal_index(lex, rules)
        _GOAL_INDEX_CACHE[key] = cached
    return cached


def _drive_is_degenerate(target_concept_lemma: str, slot: str, lex) -> bool:
    """A drive ("write slot=value on this target") is degenerate when
    the target's concept doesn't model the slot. Mechanically the
    verb can still set the property — but "salato attachment=fiksita"
    or "vortaro presence=manĝita" reads as nonsense in narrative
    terms; the concept isn't a thing that meaningfully has the slot.

    Three sources of "models the slot":
      1. baked into `concept.properties` directly;
      2. `pervasive` slots (hunger, wetness, temperature, cleanliness)
         apply to anything matching the slot's applies_to;
      3. some runtime derivation could write the slot on this concept
         given its parts (lock_state on valizo via seruro, posture on
         onklo via animate_default_standing). The bake skips varies
         slots; this check fills the gap.

    Returns True for drives to skip; False for drives to keep."""
    target_concept = lex.concepts.get(target_concept_lemma)
    if target_concept is None:
        return True  # unknown concept — drop
    slot_def = lex.slots.get(slot)
    if slot_def is None:
        return True  # unknown slot — drop
    if getattr(slot_def, "pervasive", False):
        return False  # pervasive: applies broadly, keep
    if slot in target_concept.properties:
        return False
    from ..dsl.introspect import slot_reachable_for_concept
    from ..dsl.rules import RUNTIME_DERIVATIONS
    if slot_reachable_for_concept(
            target_concept, slot, lex, RUNTIME_DERIVATIONS):
        return False
    return True


_AGENT_GATES_CACHE: dict[int, dict[str, list[tuple[str, str]]]] = {}


def _agent_cascade_gates(verb_lemma: str, rules) -> list[tuple[str, str]]:
    """Slot-value pairs that, when set on the actor pre-plan, cause
    `verb_lemma`'s cascade rules to fire as follow-up events. Walks
    rules looking for the pattern `event(verb, agent=entity(slot=val))`
    and returns each (slot, val) found.

    Used by the construct sampler to chain a follow-up state-change on
    the actor after the primary use-drive: setting actor.hunger=malsata
    before a manĝi-use-drive makes hungry_eats_sated emit satiĝi.
    """
    key = id(rules)
    cached = _AGENT_GATES_CACHE.get(key)
    if cached is not None:
        return cached.get(verb_lemma, [])

    from ..dsl.patterns import (
        AndPattern, BindPattern, EntityPattern, EventPattern,
    )

    def _entity_constraints(patt) -> dict | None:
        if isinstance(patt, EntityPattern):
            return patt.constraints
        if isinstance(patt, AndPattern):
            left = _entity_constraints(patt.left)
            right = _entity_constraints(patt.right)
            merged: dict = {}
            for c in (left, right):
                if c:
                    merged.update(c)
            return merged or None
        return None

    out: dict[str, list[tuple[str, str]]] = {}
    for rule in rules:
        if not isinstance(rule.when, EventPattern):
            continue
        verb = rule.when.action
        agent_pat = rule.when.role_patterns.get("agent")
        if agent_pat is None:
            continue
        constraints = _entity_constraints(agent_pat) or {}
        for slot, val in constraints.items():
            if slot in ("type", "concept", "has_suffix"):
                continue
            if not isinstance(val, str):
                continue
            out.setdefault(verb, []).append((slot, val))
    _AGENT_GATES_CACHE[key] = out
    return out.get(verb_lemma, [])


def _construct_goal_scene(lex, rng: random.Random, rules,
                          theme_concept: Optional[str] = None,
                          ) -> Optional[tuple]:
    """Build a scene for a construction drive: materialize the actor +
    each part + (optional) instrument + a stub theme entity, then
    return the drive that fires `fari` with all roles pre-bound.

    `theme_concept` selects which constructable to build. When None,
    picks one randomly — kept for legacy callers; the unified
    `regress_for_goal` path passes the theme from the goal index.

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
    if theme_concept is None:
        constructables = [
            l for l, c in lex.concepts.items()
            if "yes" in c.properties.get("constructable", ())
            and c.parts]
        if not constructables:
            return None
        theme_concept = rng.choice(constructables)
    theme_def = lex.concepts.get(theme_concept)
    if theme_def is None or not theme_def.parts:
        return None
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
    # Pre-filter by the theme's containment rules: fari adds
    # en(theme, agent_loc), silently swallowed if containment.jsonl
    # forbids the pair (muro requires interna; plaĝo can't hold it).
    # Containment defaults to deny — a theme with no en rule at all
    # can't be placed anywhere via fari, so the whole construct chain
    # is unsolvable and we bail. Same for a theme whose en rules
    # admit no location concepts in the scene pool.
    from ..containment import containers_for, resolve_containment
    containment_idx = resolve_containment(lex)
    allowed_containers = {
        c for c, rel in containers_for(theme_concept, containment_idx, lex)
        if rel == "en"}
    if not allowed_containers:
        _bail(f"construct_no_en_containment:{theme_concept}")
        return None
    locations = [l for l in locations if l in allowed_containers]
    if not locations:
        _bail(f"construct_no_compatible_scene:{theme_concept}")
        return None
    goal_index = _cached_goal_index(lex, rules)
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
            # Pin count=1 on each part — _add_entity_randomized rolls
            # count to anything 1..5, so the recipe might otherwise
            # contain "three loaves of bread" as a single ingredient.
            # Recipes today encode WHICH concepts, not how many of each
            # — a single unit per part is the consistent reading.
            t.entities[placed].set_property("count", "1")
        if not ok:
            continue
        # When a part has `requires` (e.g. teo's akvo must be bolanta),
        # the planner must achieve that property via another verb (boli)
        # before fari fires. Place a tool that supports the producing
        # verb so the scene is plannable. Walks goal_index → producer
        # verb → producer's instrument signature → concept matching
        # that signature. Stays generic: a new (slot, value) requirement
        # auto-resolves if there's a producer verb in the lexicon.
        for part_spec in theme_def.parts:
            for slot, allowed in part_spec.requires.items():
                if not ok:
                    break
                for value in allowed:
                    producers = goal_index.get(
                        ("property", slot, value), ())
                    placed_tool = False
                    for v_lemma in producers:
                        action = lex.actions.get(v_lemma)
                        if action is None:
                            continue
                        instr_role = next(
                            (r for r in action.roles
                             if r.name == "instrument"), None)
                        if instr_role is None:
                            continue
                        sigs = instr_role.properties.get(
                            "functional_signature", [])
                        if not sigs:
                            continue
                        for tool_lemma, tool_def in lex.concepts.items():
                            if getattr(tool_def, "is_category_stub", False):
                                continue
                            tool_sigs = tool_def.properties.get(
                                "functional_signature", [])
                            if not any(s in tool_sigs for s in sigs):
                                continue
                            if _place_respecting_containment(
                                    t, lex, scene_id, tool_lemma, rng,
                                    preferred_id=scene_id,
                                    existing_eid=None) is not None:
                                placed_tool = True
                                break
                        if placed_tool:
                            break
                    if not placed_tool:
                        ok = False
                        break
            if not ok:
                break
        if not ok:
            continue
        # No pre-havi: each ingredient sits in the scene at its
        # natural-habitat location; the planner finds them and emits
        # vidi+preni per part before fari fires. This produces richer
        # gather-then-construct chains in SFT data than starting with
        # everything already on the agent.
        # Pre-stage the to-be-created theme as a bare entity. The
        # fari rule places it at the agent's location and attaches
        # parts; the entity exists before firing so the planner can
        # reason about it (state preconditions on theme.slot).
        stub_eid = f"{theme_concept}_planned"
        if stub_eid not in t.entities:
            try:
                _add_entity_randomized(
                    t, theme_concept, lex, rng, entity_id=stub_eid)
            except (KeyError, ValueError):
                continue
            # Drop the auto-materialized sub-parts that
            # _add_entity_randomized created from theme.parts. The fari
            # rule attaches the standalone gathered parts instead; the
            # auto-parts are unused noise that bloats the relaxed-graph
            # grounding (~25× derivation blow-up before this cleanup).
            stub_parts = [
                p_eid for p_eid in list(t.entities.keys())
                if p_eid.startswith(stub_eid + "_")]
            for p_eid in stub_parts:
                del t.entities[p_eid]
            t.relations = [
                r for r in t.relations
                if not any(a in stub_parts for a in r.args)]
            # Mark the stub as planner-only: it has no scene-init
            # location (we don't know yet where fari will fire); the
            # realizer's `created_at_event is not None` filter then
            # skips synthetic-en grounding and quality grounding,
            # treating fari as the introduction. 0 is a setup-time
            # sentinel — fari's actual event index is patched in
            # post-execution by generate_corpus, but the boolean
            # filter is what the realizer reads.
            t.entities[stub_eid].created_at_event = 0
        # Pin count=1 on the constructed entity — fari produces ONE
        # whole, not a stack. Without this, the realizer renders
        # "kvin sandviĉoj" because the stub's count rolled to 5 at
        # _add_entity_randomized time.
        t.entities[stub_eid].set_property("count", "1")
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

        # Downstream drive: agent USES the constructed thing. Walk
        # the property-goal index and keep any (slot, value) for
        # which at least one producer verb's theme role accepts the
        # constructable concept — that's exactly the same filter
        # `regress_for_goal` applies, scoped to our chosen theme.
        # Weight by producer count so common drives surface naturally.
        # Auto-extends: adding a new verb whose effect produces a
        # state change on theme.type=physical immediately becomes a
        # candidate use-verb here.
        viable_drives: list = []
        for goal_key, producer_verbs in goal_index.items():
            if goal_key[0] != "property":
                continue
            _, slot, value = goal_key
            # Skip drives whose effect slot the constructed theme
            # doesn't model — "salato attachment=fiksita" reads as
            # nonsense.
            if _drive_is_degenerate(theme_concept, slot, lex):
                continue
            for v in producer_verbs:
                pa = lex.actions.get(v)
                if pa is None:
                    continue
                # Find the producer's theme role (the one whose
                # property the effect writes to). Use any effect role
                # whose slot/value matches the goal — that's the role
                # bound to our constructable.
                target_role = None
                for eff in pa.effects:
                    if eff.property == slot and eff.value == value:
                        target_role = next(
                            (r for r in pa.roles
                             if r.name == eff.target_role), None)
                        if target_role is not None:
                            break
                if target_role is None:
                    continue
                # Same accept-check the spawner uses: subtype +
                # role.properties compatibility against the concept.
                if not lex.types.is_subtype(
                        theme_def.entity_type, target_role.type):
                    continue
                ok = True
                for s, vals in (target_role.properties or {}).items():
                    if not vals:
                        continue
                    slot_def = lex.slots.get(s)
                    cvals = theme_def.properties.get(s, [])
                    if slot_def is None:
                        continue
                    if slot_def.varies:
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
                    viable_drives.append(
                        (slot, value, v, len(producer_verbs)))
                    break
        if not viable_drives:
            _bail(f"construct_no_viable_use_drive:{theme_concept}")
            return None
        weights = [w for *_, w in viable_drives]
        slot, value, use_verb, _w = rng.choices(
            viable_drives, weights=weights, k=1)[0]
        # Sampler-side chain: if the picked use-verb has an agent-side
        # cascade gate (e.g. manĝi when malsata cascades to satiĝi,
        # trinki when soifa cascades to sensoifiĝi), force the actor
        # into that gate state so the cascade fires as a follow-up
        # event. The primary drive is unchanged — the cascade adds
        # extra events to the trace at engine-run time, chaining the
        # narrative: "fari teon → trinkis la teon → sensoifiĝis."
        cascade_gates = _agent_cascade_gates(use_verb, rules)
        for gate_slot, gate_val in cascade_gates:
            t.entities[actor_eid].set_property(gate_slot, gate_val)
        # Force the stub's current value on this slot to be NON-target
        # so the drive isn't trivially already-satisfied. The stub was
        # built via _add_entity_randomized which randomizes varies-true
        # slots, so it may have rolled the target value (e.g.
        # integrity=tranĉita matches the tranĉi drive). Mirrors the
        # verb-aware setup the regular spawner applies for effect-target
        # roles. Also intersect with the use-verb's role.properties on
        # the slot — varmigi.theme requires temperature=malvarma, so
        # picking varmega as the non-target leaves varmigi inapplicable
        # and the planner dead-ends. Same fix as regress_for_goal's
        # actor-target branch.
        slot_def = lex.slots.get(slot)
        if slot_def is not None and slot_def.vocabulary:
            non_target = [v for v in slot_def.vocabulary if v != value]
            use_action = lex.actions.get(use_verb)
            if use_action is not None:
                use_target_role = next(
                    (r for r in use_action.roles
                     for eff in use_action.effects
                     if eff.target_role == r.name
                     and eff.property == slot
                     and eff.value == value), None)
                if use_target_role is not None:
                    role_required = (use_target_role.properties
                                      or {}).get(slot, ())
                    if role_required:
                        narrowed = [v for v in non_target
                                    if v in role_required]
                        if narrowed:
                            non_target = narrowed
            if non_target:
                t.entities[stub_eid].set_property(
                    slot, rng.choice(non_target))
        _ensure_obstacle_tools(t, lex, rng, scene_id)
        drive = ("entity_slot", actor_eid, stub_eid, slot, value)
        return t, scene_id, drive
    _bail(f"construct_loop_exhausted:{theme_concept}")
    return None


def _seed_derivation_only_deps(
    t, action, role_eids: dict, scene_id: str, lex, rng,
    rules, derivations, *,
    seen: set | None = None, depth: int = 0,
) -> None:
    """Walk the verb-precondition graph from `action`, seeding only
    derivation-only role.properties (those no verb's effect writes
    directly). The planner can subgoal everything else via action
    subgoaling; seeding it would just bloat the trace without
    helping the relaxed-plan heuristic.

    Concrete cases this covers:
      - `vidi.agent.illuminated=yes` — only the `agent_illuminated`
        derivation produces it; the planner can't subgoal
        "make agent illuminated" via any action. Without seeding a
        lamp into the trace, `lit_state` stays malluma and the
        relaxed graph reports h=INF.

    The seeding itself is delegated to `_seed_role_property_dependencies`,
    which is already bounded internally to (slot, value) pairs that
    `_verbs_producing` doesn't cover. We only walk the precondition
    chain to find which producer verbs to interrogate — surmeti's
    own role.properties don't include illuminated, but its plan
    chain reaches vidi via havi → preni → scias_lokon → vidi.

    The transitive walk is the dangerous-feeling part — task #183's
    initial implementation also called `_seed_role_chain_dependencies`
    and `_seed_chain_dependencies` per visited verb, which spawned
    entities the planner could subgoal itself and slowed the bench
    ~10×. Here we ONLY call the role-property helper, which is a
    no-op for verbs without derivation-only constraints. seen-set
    dedups visits; depth cap bounds cycles."""
    from .seeders import _seed_role_property_dependencies
    from ..schemas import RelationPrecondition
    from .goals import CREATED_ROLE
    if depth > 4:
        return
    if seen is None:
        seen = set()
    if action.lemma in seen:
        return
    seen.add(action.lemma)

    # Seed derivation-only role.properties for THIS verb. The helper
    # bails internally for (slot, value) pairs already satisfied,
    # randomizable, or producible by some verb.
    _seed_role_property_dependencies(
        t, action, role_eids, lex, rng, derivations)

    # Walk relation-precondition producers to extend the chain.
    # IfProperty gates aren't followed: `_ensure_obstacle_tools`
    # handles those, and the planner can subgoal them itself.
    index = _cached_goal_index(lex, rules)
    for pc in action.preconditions:
        if not isinstance(pc, RelationPrecondition):
            continue
        for goal_key, producer_lemmas in index.items():
            if goal_key[0] != "relation" or goal_key[1] != pc.rel:
                continue
            if CREATED_ROLE in goal_key[2]:
                continue
            for verb_lemma in producer_lemmas:
                producer = lex.actions.get(verb_lemma)
                if producer is None:
                    continue
                _seed_derivation_only_deps(
                    t, producer, role_eids, scene_id, lex, rng,
                    rules, derivations,
                    seen=seen, depth=depth + 1)


def _ensure_obstacle_tools(t, lex, rng, scene_id) -> None:
    """Walk the trace for entities whose current state would fire an
    if_property gate on some action, requiring a transition that the
    planner can only achieve via an action with an unavailable
    instrument. Spawn the missing instrument so the planner can plan
    the unblock step.

    Derives from action/precondition data, not hardcoded slots:
      1. For each entity in trace, each scalar slot value.
      2. Find any action whose if_property gate matches (slot, value).
      3. Skip if the gate's then-property already holds.
      4. Find an action whose effect produces (then_property,
         then_value) and declares an instrument with a
         functional_signature.
      5. If no entity in scene matches that signature, spawn one.

    Today this fires for ŝranko/kofro with locked seruros: malfermi's
    `if lock_capable=yes then lock_state=malŝlosita` gate, satisfied
    by malŝlosi's `instrument.functional_signature=ŝlosi`, prompts a
    ŝlosilo spawn. The relaxed-graph prespawn can't reach this chain
    on its own (samloke→en→eniri→malfermi→if_property→malŝlosi
    crosses a derivation, two property gates, and an instrument
    role). Other obstacle patterns light up automatically as the
    action/gate data evolves."""
    if not t.entities:
        return
    # Index existing tool signatures so we don't double-spawn.
    present_sigs: set = set()
    for ent in t.entities.values():
        for sig in ent.properties.get("functional_signature", ()):
            present_sigs.add(sig)
    # Index if_property gates by (if_property, if_value).
    from ..schemas import IfPropertyPrecondition
    gates: dict = {}
    for action in lex.actions.values():
        for pc in action.preconditions:
            if not isinstance(pc, IfPropertyPrecondition):
                continue
            gates.setdefault(
                (pc.if_property, pc.if_value), []).append(pc)
    # Producers of (then_property, then_value).
    producers: dict = {}
    for action in lex.actions.values():
        instr_role = next(
            (r for r in action.roles if r.name == "instrument"), None)
        if instr_role is None:
            continue
        sigs = tuple(instr_role.properties.get(
            "functional_signature", ()))
        if not sigs:
            continue
        for eff in action.effects:
            producers.setdefault(
                (eff.property, eff.value), []).append((action, sigs))
    seeders_mod = None
    for ent in list(t.entities.values()):
        for slot, values in list(ent.properties.items()):
            for value in values:
                for pc in gates.get((slot, value), ()):
                    # Skip if then-property already holds.
                    then_have = ent.properties.get(pc.then_property, ())
                    if pc.then_value in then_have:
                        continue
                    # Skip when the gate transitions OUT (we'd need
                    # to UN-set the then_value, which would loop).
                    if pc.if_value == pc.then_value:
                        continue
                    for prod_action, sigs in producers.get(
                            (pc.then_property, pc.then_value), ()):
                        if any(s in present_sigs for s in sigs):
                            break  # tool already available
                        # Find a concept matching the signature, spawn.
                        spawned = False
                        for c_lemma, c_def in lex.concepts.items():
                            if getattr(c_def, "is_category_stub", False):
                                continue
                            tool_sigs = c_def.properties.get(
                                "functional_signature", ())
                            if not any(s in tool_sigs for s in sigs):
                                continue
                            if seeders_mod is None:
                                from . import seeders as seeders_mod  # noqa: F401
                                from .seeders import (
                                    _place_respecting_containment as _placer
                                )
                            else:
                                _placer = (
                                    seeders_mod._place_respecting_containment)
                            if _placer(
                                    t, lex, scene_id, c_lemma, rng,
                                    preferred_id=scene_id,
                                    existing_eid=None) is not None:
                                present_sigs.update(tool_sigs)
                                spawned = True
                                break
                        if spawned:
                            break


def regress_for_goal(lex, rng: random.Random, rules) -> Optional[tuple]:
    """Pick a goal, pick a verb from its producers, build a minimal
    trace, return (trace, scene_id, drive). The spawner — invoked by
    the planner via the entity_resolver hook — materializes everything
    else with verb-aware setup.

    Goal kinds:
      - Property goals: ("property", slot, value). Drive shape is
        `entity_slot`. The verb's effect mutates the target's slot.
      - CREATED-role relation goals: ("relation", rel, role_tuple)
        where role_tuple includes the `<created>` sentinel. Used for
        verbs whose effect is CreateEntity + AddRelation (skribi,
        vidi, aŭdi, flari, montri, krii, ludi, …). Drive shape is
        `event_fire`; the planner just has to fire the verb with the
        appropriate role bindings — the rule creates the entity.
      - Plain relation goals (havi/en/vestita) with agent-as-first-
        arg: drive shape is `possession`/`location`/`wearing`. The
        actor is the scene's actor; the target is spawned via the
        producer verb's second-arg role spec. The planner is free
        to pick any verb that achieves the relation (not necessarily
        the one we sampled).
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
            _, rel_name, role_args = g
            if CREATED_ROLE in role_args:
                # event_fire drive: rule creates the related entity.
                all_goals.append((g, verbs))
            elif (rel_name in _PLAIN_RELATION_DRIVES
                    and role_args[0] == "agent"):
                # Plain havi/en/vestita with agent-as-actor →
                # possession/location/wearing drive.
                all_goals.append((g, verbs))
        elif g[0] == "construct":
            all_goals.append((g, verbs))
    if not all_goals:
        _bail("no_goals_in_index")
        return None
    # Weight by producer count. Construct goals each have 1 producer
    # (fari) so a single constructable carries the same weight as a
    # weakly-supported property goal — fine, plenty of property goals
    # have just one or two producers too.
    weights = [len(verbs) for _, verbs in all_goals]
    chosen_goal, producers = rng.choices(
        all_goals, weights=weights, k=1)[0]
    if chosen_goal[0] == "construct":
        # Dispatch to the construct-scene builder. Goal carries the
        # theme concept; the builder picks an actor, scene, and use-
        # drive on the constructed theme.
        result = _construct_goal_scene(
            lex, rng, rules, theme_concept=chosen_goal[1])
        # No outer _bail on failure — the inner _construct_goal_scene
        # already set a specific reason (construct_no_en_containment,
        # construct_no_compatible_scene, construct_no_viable_use_drive,
        # construct_loop_exhausted). Overwriting it with
        # "construct_scene_none:<theme>" hides the actual cause.
        return result
    verb_lemma = rng.choice(producers)
    action = lex.actions.get(verb_lemma)
    if action is None:
        _bail(f"unknown_action:{verb_lemma}")
        return None

    if chosen_goal[0] == "property":
        _, slot, target_value = chosen_goal
        if not action.effects:
            _bail(f"no_effects:{verb_lemma}")
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
        _bail(f"no_agent_role:{verb_lemma}")
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
        _bail(f"no_agent_candidates:{verb_lemma}")
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
        _bail("no_locations")
        return None
    tried: set = set()
    last_loop_reason = f"no_retry_left:{verb_lemma}"
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
            last_loop_reason = f"scene_add_failed:{scene_lemma}"
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
                last_loop_reason = (
                    f"actor_placement_failed:{agent_concept}@{scene_lemma}")
                continue

        if chosen_goal[0] == "property":
            if eff.target_role == actor_role_name:
                target_eid = actor_eid
                # Same degenerate-drive check as the target branch
                # below: skip drives whose effect slot the actor's
                # concept doesn't model (e.g. kuniklido posture=
                # staranta where kuniklido never declares posture).
                if _drive_is_degenerate(
                        agent_concept, eff.property, lex):
                    last_loop_reason = (
                        f"degenerate_actor_drive:{agent_concept}.{eff.property}")
                    continue
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
                    _bail(f"missing_target_role:{verb_lemma}.{eff.target_role}")
                    return None  # schema issue, not placement
                target_eid = setup_spawner(
                    target_role_spec, t, lex,
                    set(t.entities.keys()),
                    action=action, role_name=eff.target_role)
                if target_eid is None:
                    last_loop_reason = (
                        f"target_spawn_failed:{verb_lemma}.{eff.target_role}@{scene_lemma}")
                    continue  # try another scene location
                target_ent = t.entities.get(target_eid)
                if (target_ent is not None
                        and _drive_is_degenerate(
                            target_ent.concept_lemma, eff.property, lex)):
                    last_loop_reason = (
                        f"degenerate_target_drive:"
                        f"{target_ent.concept_lemma}.{eff.property}")
                    continue  # degenerate drive — retry
            _ensure_obstacle_tools(t, lex, rng, scene_id)
            drive = ("entity_slot", actor_eid, target_eid,
                     slot, target_value)
            return t, scene_id, drive
        elif (chosen_goal[0] == "relation"
                and chosen_goal[1] in _PLAIN_RELATION_DRIVES
                and CREATED_ROLE not in chosen_goal[2]
                and chosen_goal[2][0] == "agent"):
            # possession/location/wearing drive: spawn the second-
            # arg target via the producer verb's role spec; the
            # planner is free to pick any verb achieving the
            # relation (not necessarily the one we sampled).
            _, rel_name, role_args = chosen_goal
            target_role_name = role_args[1]
            target_role_spec = next(
                (r for r in action.roles
                 if r.name == target_role_name), None)
            if target_role_spec is None:
                _bail(f"missing_target_role:"
                      f"{verb_lemma}.{target_role_name}")
                return None  # schema issue, not placement
            from .spawner import make_spawner
            setup_spawner = make_spawner(
                scene_id, lex, rng, budget=1)
            target_eid = setup_spawner(
                target_role_spec, t, lex,
                set(t.entities.keys()),
                action=action, role_name=target_role_name)
            if target_eid is None:
                last_loop_reason = (
                    f"relation_target_spawn_failed:"
                    f"{verb_lemma}.{target_role_name}@{scene_lemma}")
                continue
            # Schema-level viability: the relation's `arg_excludes`,
            # `arg_not_part`, `arg_patterns`, and `arg_compare` already
            # encode what makes a (actor, target) pairing valid — havi
            # forbids location/person themes, nemovebla themes, and
            # requires theme.maso ≤ owner.lift_capacity. The role spec
            # alone (animate, physical) accepts nonsense like
            # papagido→delfeno because both pass the type filter; the
            # relation schema is the principled gate. Reject here so
            # the planner doesn't have to try and fail.
            if not t.is_relation_permitted(
                    rel_name, (actor_eid, target_eid), lex):
                target_ent = t.entities.get(target_eid)
                last_loop_reason = (
                    f"relation_not_permitted:{rel_name}("
                    f"{agent_concept},"
                    f"{target_ent.concept_lemma if target_ent else '?'})"
                    f"@{scene_lemma}")
                continue
            _ensure_obstacle_tools(t, lex, rng, scene_id)
            _seed_derivation_only_deps(
                t, action,
                {actor_role_name: actor_eid,
                 target_role_name: target_eid}, scene_id, lex, rng,
                rules, _ALL_DERIVATIONS)
            drive_kind = _PLAIN_RELATION_DRIVES[rel_name]
            drive = (drive_kind, actor_eid, target_eid)
            return t, scene_id, drive
        else:
            _ensure_obstacle_tools(t, lex, rng, scene_id)
            # event_fire: only bind the actor here. The planner's
            # _prespawn_for_goal fills the remaining roles via the
            # session spawner (same closure used during search), so
            # the sampler doesn't have to commit to specific theme/
            # instrument/recipient up front. Reduces no_sample bails
            # for verbs with many roles.
            drive = ("event_fire", actor_eid, verb_lemma,
                     ((actor_role_name, actor_eid),))
            return t, scene_id, drive
    _bail(last_loop_reason)
    return None
