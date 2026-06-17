"""Single-agent forward sampler POC.

Pick one actor at scene start. Each tick, find every (verb, role-
binding) the actor can fire RIGHT NOW (all preconditions hold; no
subgoaling). Pick one, biased toward the actor's displeased state.
If nothing applies, try a 2-step lookahead: a verb_a that, once
fired, would unblock some verb_b. If still nothing, the actor is
stuck — stop.

This is forward sampling, not regression. The actor wanders through
plausible action with what's available; long deliberate goal-chains
(go-to-locked-room-and-fetch) just won't surface — those need the
regression planner.

Run:
    uv run python scripts/forward_one_agent_poc.py
"""
from __future__ import annotations

import random
import time

from esperanto_lm.ontology import (
    Trace, effect_changes, load_lexicon, make_event, realize_trace,
)
from esperanto_lm.ontology.dsl import compute_derived_state, run_dsl
from esperanto_lm.ontology.dsl.rules import (
    DEFAULT_DSL_RULES, RUNTIME_DERIVATIONS,
)
from esperanto_lm.ontology.schemas import (
    IfPropertyPrecondition, MatchPrecondition, RelationPrecondition,
)
from esperanto_lm.ontology.agent.preferences import SLOT_PREFERENCES
from esperanto_lm.ontology.agent.planner import (
    _effect_target_role, _entity_property_values, _role_spec_satisfied,
    _rule_writes, _trigger_event_pattern, displeased_slots,
)


# ----------------------- precondition checks --------------------------

def _relation_holds(rel_name, args, trace, derived):
    for r in trace.relations:
        if r.relation == rel_name and tuple(r.args) == tuple(args):
            return True
    if derived is not None and derived.has_relation(rel_name, tuple(args)):
        return True
    return False


def _all_preconditions_met(action, role_bindings, trace, lex, derived):
    """True iff every action.precondition holds in the current state
    with the given bindings. Pure check; no planning."""
    for pc in action.preconditions:
        if isinstance(pc, RelationPrecondition):
            args = tuple(role_bindings.get(rn) for rn in pc.roles)
            if any(a is None for a in args):
                return False
            if not _relation_holds(pc.rel, args, trace, derived):
                return False
        elif isinstance(pc, IfPropertyPrecondition):
            eid = role_bindings.get(pc.role)
            ent = trace.entities.get(eid) if eid else None
            if ent is None:
                return False
            if_vals = _entity_property_values(
                ent, pc.if_property, trace, derived)
            if pc.if_value not in if_vals:
                continue   # gate vacuously passes
            then_vals = _entity_property_values(
                ent, pc.then_property, trace, derived)
            if pc.then_value not in then_vals:
                return False
        elif isinstance(pc, MatchPrecondition):
            ea = role_bindings.get(pc.role_a)
            eb = role_bindings.get(pc.role_b)
            ent_a = trace.entities.get(ea) if ea else None
            ent_b = trace.entities.get(eb) if eb else None
            if ent_a is None or ent_b is None:
                return False
            va = _entity_property_values(ent_a, pc.slot_a, trace, derived)
            vb = _entity_property_values(ent_b, pc.slot_b, trace, derived)
            if not (va & vb):
                return False
    return True


# ----------------------- role binding enumeration ---------------------

def _enumerate_full_bindings(action, fixed, trace, lex, derived,
                              max_per_action=30):
    """Yield complete role-binding dicts. `fixed` pre-binds named
    roles (e.g. {"agent": actor_id}); other roles get filled by
    scanning compatible scene entities. Bounded to keep tick cost
    sane on heavily-populated scenes."""
    role_specs = action.roles
    role_names = [r.name for r in role_specs]

    cands = []
    for role in role_specs:
        if role.name in fixed:
            cands.append([fixed[role.name]])
            continue
        compat = []
        for eid, ent in trace.entities.items():
            if ent.destroyed_at_event is not None:
                continue
            if not _role_spec_satisfied(ent, role, lex, trace, derived):
                continue
            compat.append(eid)
        if not compat:
            return
        cands.append(compat)

    count = [0]
    def _expand(idx, current):
        if count[0] >= max_per_action:
            return
        if idx == len(role_specs):
            non_none = [v for v in current.values() if v is not None]
            if len(set(non_none)) != len(non_none):
                return  # duplicate eid in two roles
            count[0] += 1
            yield dict(current)
            return
        rname = role_names[idx]
        for eid in cands[idx]:
            current[rname] = eid
            yield from _expand(idx + 1, current)
        current.pop(rname, None)

    yield from _expand(0, {})


# ----------------------- applicable actions ---------------------------

def applicable_actions(actor_id, trace, lex, derived):
    """All (verb, role_bindings) the actor can fire right now: every
    precondition holds, every role-property constraint matches.

    Excludes cascade-only verbs (no agent role) — those are emitted
    by the engine, not chosen by an actor.

    Tries each animate role the actor's type satisfies. Most verbs
    only have one animate role (`agent`), but some have `theme=animate`
    (vekiĝi, satiĝi) — though those are filtered out by the cascade
    check above."""
    actor_ent = trace.entities.get(actor_id)
    if actor_ent is None:
        return []
    out = []
    for verb_name, action in lex.actions.items():
        if "agent" not in {r.name for r in action.roles}:
            continue
        # Find animate-typed roles the actor could bind to.
        candidate_roles = []
        for role in action.roles:
            if not lex.types.is_subtype(actor_ent.entity_type, role.type):
                continue
            if not _role_spec_satisfied(
                    actor_ent, role, lex, trace, derived):
                continue
            # Restrict to "actor" roles — the agent does the action.
            # `agent` covers most; "theme" of intransitive achievements
            # (vekiĝi-style) is also self, but those are cascade-only.
            if role.name == "agent":
                candidate_roles.append(role.name)
        for role_name in candidate_roles:
            for bindings in _enumerate_full_bindings(
                    action, {role_name: actor_id}, trace, lex, derived):
                if _all_preconditions_met(
                        action, bindings, trace, lex, derived):
                    out.append((verb_name, bindings))
    return out


# ----------------------- action scoring -------------------------------

def _verb_writes_slot(verb_name, slot, value, lex, rules):
    """True if firing this verb (directly or via a causal-rule cascade)
    would write `slot=value` to some role's bound entity. Used by
    `pick_best` to bias toward verbs that improve the actor's state."""
    action = lex.actions.get(verb_name)
    if action is not None:
        for eff in action.effects:
            if eff.property == slot and eff.value == value:
                return True
    for rule in rules:
        ev_pat = _trigger_event_pattern(rule)
        if ev_pat is None or ev_pat.action != verb_name:
            continue
        if _rule_writes(rule, slot, value):
            return True
    return False


def pick_best(cands, actor_id, trace, derived, lex, rules, rng):
    """Among applicable actions, prefer ones that satisfy a displeased
    slot of the actor (where the slot's preferred value ≠ current).
    Tie-break randomly. If no candidate addresses a displeasure, pick
    any random candidate (just keep moving)."""
    actor = trace.entities.get(actor_id)
    if actor is None:
        return None
    desired = displeased_slots(actor, trace, derived)
    if desired:
        scored = []
        for verb_name, bindings in cands:
            score = sum(
                1 for slot, val in desired
                if _verb_writes_slot(verb_name, slot, val, lex, rules))
            scored.append((score, verb_name, bindings))
        max_score = max(s[0] for s in scored)
        if max_score > 0:
            top = [(v, b) for s, v, b in scored if s == max_score]
            return rng.choice(top)
    return rng.choice(cands)


# ----------------------- 2-step lookahead -----------------------------

def bridge_actions(actor_id, trace, lex, rules, derivations,
                    derived, *, k=2):
    """K-step lookahead: find an immediately-applicable verb the actor
    can fire that starts a chain of length <= k ending in a verb that
    addresses one of the actor's drives.

    K=1: directly applicable verb satisfies a drive.
    K=2: applicable verb_a adds a relation enabling verb_b which
         satisfies a drive.
    K=3: applicable verb_a → verb_b → verb_c (drive). Bridges chains
         like vidi (sets scias_lokon) → preni (sets havi) → manĝi
         (satiates).

    Returns one (verb, bindings) or None. We don't return the full
    chain — only the FIRST applicable verb to fire. The next ticks
    will discover the rest as preconditions get satisfied."""
    if k < 1:
        return None
    actor_ent = trace.entities.get(actor_id)
    if actor_ent is None:
        return None

    desired = displeased_slots(actor_ent, trace, derived)
    if not desired:
        return None

    goal_verbs = set()
    for slot, val in desired:
        for verb_name, action in lex.actions.items():
            if "agent" not in {r.name for r in action.roles}:
                continue
            if _verb_writes_slot(verb_name, slot, val, lex, rules):
                goal_verbs.add(verb_name)

    # For each goal verb, find missing relation preconds, then BFS
    # backward through relation-adders looking for an applicable verb.
    for goal_verb_name in goal_verbs:
        action = lex.actions[goal_verb_name]
        for bindings in _enumerate_full_bindings(
                action, {"agent": actor_id}, trace, lex, derived):
            missing = []
            for pc in action.preconditions:
                if not isinstance(pc, RelationPrecondition):
                    continue
                args = tuple(bindings.get(rn) for rn in pc.roles)
                if any(a is None for a in args):
                    continue
                if not _relation_holds(pc.rel, args, trace, derived):
                    missing.append((pc.rel, args))
            for rel_name, rel_args in missing:
                cand = _bfs_unblock(
                    actor_id, rel_name, rel_args, trace, lex, rules,
                    derived, depth=k - 1, _seen=set())
                if cand is not None:
                    return cand
    return None


def _bfs_unblock(actor_id, target_rel, target_args, trace, lex, rules,
                 derived, *, depth, _seen):
    """Recursive search: find an applicable verb the actor can fire
    NOW that starts a chain ending in (target_rel, target_args) being
    established within `depth` more steps.

    Direct case (depth >= 0): some verb_x is applicable AND adds the
    target relation directly.
    Recursive (depth >= 1): some verb_x adds the target relation but
    isn't applicable; recurse on each of verb_x's missing relation
    preconditions to find a deeper applicable starter.

    `_seen` cycles guard — same (rel, args) won't be searched twice
    in one invocation."""
    key = (target_rel, target_args)
    if key in _seen:
        return None
    _seen = _seen | {key}

    # Direct case: applicable adders.
    for cand in _find_relation_adders_for_actor(
            actor_id, target_rel, target_args, trace, lex, rules,
            derived, applicable_only=True):
        return cand

    if depth <= 0:
        return None

    # Recursive: any adder (even non-applicable). For each, look at
    # its missing relation preconditions and recurse.
    for verb_b, bindings_b in _find_relation_adders_for_actor(
            actor_id, target_rel, target_args, trace, lex, rules,
            derived, applicable_only=False):
        action_b = lex.actions[verb_b]
        for pc in action_b.preconditions:
            if not isinstance(pc, RelationPrecondition):
                continue
            args_b = tuple(bindings_b.get(rn) for rn in pc.roles)
            if any(a is None for a in args_b):
                continue
            if _relation_holds(pc.rel, args_b, trace, derived):
                continue
            sub = _bfs_unblock(
                actor_id, pc.rel, args_b, trace, lex, rules, derived,
                depth=depth - 1, _seen=_seen)
            if sub is not None:
                return sub
    return None


def _find_relation_adders_for_actor(actor_id, rel_name, rel_args, trace,
                                    lex, rules, derived,
                                    *, applicable_only=True):
    """Yield (verb, bindings) where the actor is the agent and firing
    the verb would add the relation. If `applicable_only=True`,
    filter to bindings where all preconditions hold now; otherwise
    return all bindings regardless."""
    from esperanto_lm.ontology.dsl.effects import AddRelation
    from esperanto_lm.ontology.dsl.patterns import (
        BindPattern, AndPattern, EntityPattern, EventPattern, Var,
    )
    out = []
    for rule in rules:
        if not isinstance(rule.when, EventPattern):
            continue
        verb_name = rule.when.action
        action = lex.actions.get(verb_name)
        if action is None:
            continue
        if "agent" not in {r.name for r in action.roles}:
            continue
        effects = (rule.then if isinstance(rule.then, (list, tuple))
                   else [rule.then])
        for eff in effects:
            if not isinstance(eff, AddRelation):
                continue
            if eff.relation != rel_name:
                continue
            arg_role_names = []
            ok = True
            for arg in eff.args:
                if not isinstance(arg, Var):
                    ok = False; break
                role_name = None
                for rn, rp in rule.when.role_patterns.items():
                    def _walk(p):
                        if isinstance(p, BindPattern) and p.target is arg:
                            return True
                        if isinstance(p, AndPattern):
                            return any(_walk(x) for x in (p.left, p.right))
                        return False
                    if _walk(rp):
                        role_name = rn; break
                if role_name is None:
                    ok = False; break
                arg_role_names.append(role_name)
            if not ok:
                continue
            fixed = dict(zip(arg_role_names, rel_args))
            fixed["agent"] = actor_id
            for bindings in _enumerate_full_bindings(
                    action, fixed, trace, lex, derived):
                if applicable_only and not _all_preconditions_met(
                        action, bindings, trace, lex, derived):
                    continue
                out.append((verb_name, bindings))
                if len(out) >= 5:
                    return out
                break
    return out


# ----------------------- main loop ------------------------------------

def forward_one_agent(actor_id, trace, lex, rules, derivations,
                      *, max_ticks=15, lookahead=3, rng=None,
                      allow_wandering=True):
    """Run forward sampling for one actor. Returns the number of
    events fired.

    Tick priority:
      1. Drive-progressing greedy: an applicable verb whose effect
         directly satisfies a displeased slot. (Manĝi for hunger if
         applicable; verŝi for nothing currently.)
      2. Drive-progressing lookahead: an applicable verb that starts
         a K-step chain ending in drive satisfaction. Catches the
         vidi → preni → manĝi shape when manĝi isn't yet applicable.
      3. Wandering: any applicable verb. Pure exploration when no
         drive can be progressed."""
    if rng is None:
        rng = random.Random(0)
    n_fired = 0
    for _ in range(max_ticks):
        derived = compute_derived_state(trace, derivations, lex)
        cands = applicable_actions(actor_id, trace, lex, derived)

        # 1. Filter to drive-progressing applicable verbs.
        actor_ent = trace.entities.get(actor_id)
        chosen = None
        if actor_ent is not None:
            desired = displeased_slots(actor_ent, trace, derived)
            if desired and cands:
                drive_cands = [
                    (v, b) for v, b in cands
                    if any(_verb_writes_slot(v, slot, val, lex, rules)
                           for slot, val in desired)
                ]
                if drive_cands:
                    chosen = rng.choice(drive_cands)

        # 2. Lookahead: applicable verb starting a chain to drive.
        if chosen is None and lookahead > 0:
            chosen = bridge_actions(
                actor_id, trace, lex, rules, derivations, derived,
                k=lookahead)

        # 3. Wandering fallback. Skip when allow_wandering=False —
        # then the agent fires only steps that progress a drive (or
        # set up a drive-progressing chain via lookahead) and stops
        # otherwise. Cleaner narrative; some scenes produce zero
        # events when no drive is progressable.
        if chosen is None and cands and allow_wandering:
            chosen = rng.choice(cands)

        if chosen is None:
            break

        verb, roles = chosen
        ev = make_event(verb, roles=roles,
                        property_changes=effect_changes(verb, roles, lex))
        trace.events.append(ev)
        run_dsl(trace, rules, derivations, lex)
        n_fired += 1
    return n_fired


# ----------------------- demo scenes ----------------------------------

def scene_hungry_in_kitchen(lex):
    """Maria hungry in kitchen, bread on table. Greedy without
    havi: vidi → preni → manĝi requires a 3-step bridge through
    scias_lokon (derived from konas), which forward-BFS can't see.
    See scene_hungry_with_food for the workable case."""
    t = Trace()
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("pano", lex, entity_id="pano")
    t.add_entity("tablo", lex, entity_id="tablo")
    t.assert_relation("en", ("maria", "kuirejo"), lex)
    t.assert_relation("en", ("tablo", "kuirejo"), lex)
    t.assert_relation("sur", ("pano", "tablo"), lex)
    m = t.entities["maria"]
    m.set_property("hunger", "malsata")
    m.set_property("thirst", "satigita")
    m.set_property("posture", "staranta")
    m.set_property("sleep_state", "vekita")
    return t, "kuirejo", "maria"


def scene_hungry_with_food(lex):
    """Maria hungry, ALREADY holding bread. manĝi directly applicable
    (havi precondition met). One-step cascade: manĝi → satiĝi."""
    t = Trace()
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("pano", lex, entity_id="pano")
    t.assert_relation("en", ("maria", "kuirejo"), lex)
    t.assert_relation("en", ("pano", "kuirejo"), lex)
    t.assert_relation("havi", ("maria", "pano"), lex)
    m = t.entities["maria"]
    m.set_property("hunger", "malsata")
    m.set_property("thirst", "satigita")
    m.set_property("posture", "staranta")
    m.set_property("sleep_state", "vekita")
    return t, "kuirejo", "maria"


def scene_thirsty_with_water(lex):
    """Maria thirsty in kitchen with water bottle on table."""
    t = Trace()
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("akvo", lex, entity_id="akvo")
    t.add_entity("tablo", lex, entity_id="tablo")
    t.assert_relation("en", ("maria", "kuirejo"), lex)
    t.assert_relation("en", ("tablo", "kuirejo"), lex)
    t.assert_relation("sur", ("akvo", "tablo"), lex)
    m = t.entities["maria"]
    m.set_property("thirst", "soifa")
    m.set_property("hunger", "sata")
    m.set_property("posture", "staranta")
    m.set_property("sleep_state", "vekita")
    return t, "kuirejo", "maria"


def scene_dirty_with_tools(lex):
    """Maria dirty (cleanliness=malpura) with cleaning tool + dirty thing."""
    t = Trace()
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("tablo", lex, entity_id="tablo")
    t.assert_relation("en", ("maria", "kuirejo"), lex)
    t.assert_relation("en", ("tablo", "kuirejo"), lex)
    m = t.entities["maria"]
    m.set_property("cleanliness", "pura")
    m.set_property("posture", "staranta")
    m.set_property("sleep_state", "vekita")
    # Make tablo dirty
    t.entities["tablo"].set_property("cleanliness", "malpura")
    return t, "kuirejo", "maria"


def scene_animal_wander(lex):
    """A dog in a kitchen. No drives — just wanders."""
    t = Trace()
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("hundo", lex, entity_id="hundo")
    t.assert_relation("en", ("hundo", "kuirejo"), lex)
    t.entities["hundo"].set_property("hunger", "sata")
    t.entities["hundo"].set_property("posture", "staranta")
    t.entities["hundo"].set_property("sleep_state", "vekita")
    return t, "kuirejo", "hundo"


def scene_regression(lex, rng=None):
    """Reuse the regression seeder's rich scene templates as the
    starting state. Picks the drive's actor (drive[1] for every drive
    shape except give_count, where drive[1] is the donor — also fine
    as the agent the forward sampler runs)."""
    from esperanto_lm.ontology.regression import sample_regression_scene
    if rng is None:
        rng = random.Random(0)
    for _ in range(8):
        sample = sample_regression_scene(
            lex, rng, rules=DEFAULT_DSL_RULES)
        if sample is None:
            continue
        t, scene_id, drive = sample
        actor_id = drive[1] if len(drive) >= 2 else None
        if actor_id is None or actor_id not in t.entities:
            continue
        return t, scene_id, actor_id
    raise RuntimeError("regression seeder failed 8 times")


def main():
    lex = load_lexicon()
    rules = list(DEFAULT_DSL_RULES)
    derivations = list(RUNTIME_DERIVATIONS)
    rng = random.Random(0)
    scenes = [
        ("hungry maria, already has bread", scene_hungry_with_food),
        ("hungry maria in kitchen (food on table)", scene_hungry_in_kitchen),
        ("thirsty maria with water", scene_thirsty_with_water),
        ("dirty kitchen, cleaning tools", scene_dirty_with_tools),
        ("dog wandering", scene_animal_wander),
    ] + [
        (f"regression-seeded #{i}",
         lambda lex, rng=rng: scene_regression(lex, rng))
        for i in range(3)
    ]

    for label, builder in scenes:
        print(f"\n=== {label} ===")
        t, scene_id, actor_id = builder(lex)
        setup = t.snapshot_relations()
        start = time.time()
        n = forward_one_agent(
            actor_id, t, lex, rules, derivations,
            max_ticks=12, lookahead=3, rng=rng)
        elapsed = time.time() - start
        chain = " → ".join(ev.action for ev in t.events) or "<no actions>"
        print(f"  fired: {n} events in {elapsed*1000:.0f}ms")
        print(f"  chain: {chain}")
        prose = realize_trace(
            t, lex, setup_relations=setup,
            scene_location_id=scene_id)
        print(f"  prose: {prose}")


if __name__ == "__main__":
    main()
