"""Proof-of-concept: agent-driven trace generation.

Each animate entity has a set of slot-level preferences (sata,
vekita, ...). On each tick, find a displeased agent, plan an action
that moves it toward its preference using the rule library as the
action repertoire, fire the action, run the engine, and repeat.

Why this matters: traces become CAUSALLY DRIVEN. Every event happens
because an agent had a preferred state and acted to reach it. The
prose narrates causation directly:

  "Maria estis malsata. Tial Maria manĝis la panon. Maria iĝis sata."

No agent-side imperative drive code; everything reuses:
  - Existing rule library (action repertoire + cause-effect knowledge)
  - Existing engine (cascades, derivations)
  - Existing realizer (prose)

The only new thing here is the planner (~50 lines) + the simulation
loop (~30 lines).

Doesn't touch any existing module. Run as:
    uv run python scripts/agent_sim_poc.py
"""
from __future__ import annotations

import random

from esperanto_lm.ontology import (
    Trace, effect_changes, load_lexicon, make_event, realize_trace,
)
from esperanto_lm.ontology.dsl import compute_derived_state, run_dsl
from esperanto_lm.ontology.dsl.effects import (
    AddRelation, Change, CreateEntity, Emit,
)
from esperanto_lm.ontology.dsl.implications import (
    PropertyImplication, RelationImplication,
)
from esperanto_lm.ontology.dsl.patterns import (
    AndPattern, BindPattern, EntityPattern, EventPattern, NotPattern,
    RelPattern, Var,
)
from esperanto_lm.ontology.dsl.rules import (
    DEFAULT_DSL_DERIVATIONS, DEFAULT_DSL_RULES,
)


# ----------------------- precondition consumption --------------------
#
# Action-level cross-role preconditions live on `action.preconditions`
# in the lexicon (schemas.RelationPrecondition). The planner reads
# them directly. Derived relations like `samloke` are first-class —
# `plan_to_establish_relation` handles them by walking derivations
# whose `implies` produces the goal relation, then subgoaling the
# derivation's `when + given` patterns.


# ----------------------- preferences -----------------------------

# Static slot-level preferences. In a full implementation these would
# be derived from world state (e.g. "people prefer dormanta when scene
# time_of_day=nokto"). The POC uses universal defaults to keep scope
# tight; the architecture supports the conditional version unchanged.
SLOT_PREFERENCES = {
    "hunger":      "sata",
    "sleep_state": "vekita",
    "wetness":     "seka",
    "cleanliness": "pura",
}


def displeased_slots(entity, trace=None, derived=None) -> list[tuple[str, str]]:
    """Return [(slot, preferred_value)] for slots where the entity's
    CURRENT values do not include the preferred. Membership semantics
    so multi-valued slots work."""
    out = []
    for slot, pref in SLOT_PREFERENCES.items():
        values = _entity_property_values(entity, slot, trace, derived)
        if values and pref not in values:
            out.append((slot, pref))
    return out


def displeased_entities(trace, derived=None):
    """All (entity_id, slot, preferred) tuples across the trace.
    Lets animates plan altruistically (Maria notices Petro is dormanta
    and decides to wake him)."""
    out = []
    for eid, ent in trace.entities.items():
        if ent.destroyed_at_event is not None:
            continue
        for slot, pref in displeased_slots(ent, trace, derived):
            out.append((eid, slot, pref))
    return out


# ----------------------- planner ---------------------------------

def _rule_writes(rule, slot: str, value: str) -> bool:
    """True if rule.then contains an effect that writes slot=value."""
    effects = (rule.then if isinstance(rule.then, (list, tuple))
               else [rule.then])
    for eff in effects:
        if isinstance(eff, Emit):
            for (_ent, eff_slot), eff_val in eff.property_changes.items():
                if eff_slot == slot and eff_val == value:
                    return True
        if isinstance(eff, Change):
            if eff.slot == slot and eff.value == value:
                return True
    return False


def _trigger_event_pattern(rule):
    """Extract the EventPattern from rule.when (handles And-wrapped)."""
    when = rule.when
    if isinstance(when, EventPattern):
        return when
    # Patterns can be conjuncted with given clauses; check inside an And.
    while isinstance(when, AndPattern):
        if isinstance(when.left, EventPattern):
            return when.left
        if isinstance(when.right, EventPattern):
            return when.right
        when = when.left
    return None


def _entity_satisfies_pattern(entity, pattern, lex, trace=None, derived=None) -> bool:
    """Recursive check: does this entity satisfy the role pattern?
    Reads property constraints via `trace` + `derived`. Property
    constraints use set-membership so multi-valued slots match."""
    if isinstance(pattern, EntityPattern):
        for key, expected in pattern.constraints.items():
            if key == "type":
                if not lex.types.is_subtype(entity.entity_type, expected):
                    return False
            elif key == "concept":
                if entity.concept_lemma != expected:
                    return False
            elif key == "has_suffix":
                if not entity.concept_lemma.endswith(expected):
                    return False
            else:
                values = _entity_property_values(entity, key, trace, derived)
                if expected not in values:
                    return False
        return True
    if isinstance(pattern, AndPattern):
        return (_entity_satisfies_pattern(entity, pattern.left, lex, trace, derived)
                and _entity_satisfies_pattern(entity, pattern.right, lex, trace, derived))
    # BindPattern, OrPattern, NotPattern: pass through (BindPattern always
    # accepts; we under-extract for the POC).
    return True


def _find_role_filler(role_spec, trace, lex, derived=None, exclude=None):
    """Find a trace entity matching the verb's RoleSpec (type +
    property constraints). Reads current+derived state with
    set-membership semantics so an entity with locomotion=[walk,swim]
    satisfies a role spec requiring locomotion=[walk]."""
    exclude = exclude or set()
    for eid, ent in trace.entities.items():
        if eid in exclude:
            continue
        if ent.destroyed_at_event is not None:
            continue
        if not lex.types.is_subtype(ent.entity_type, role_spec.type):
            continue
        if role_spec.properties:
            ok = True
            for slot, vals in role_spec.properties.items():
                values = _entity_property_values(ent, slot, trace, derived)
                if not values & set(vals):
                    ok = False
                    break
            if not ok:
                continue
        return eid
    return None


def plan_action(agent_id, trace, lex, rules):
    """For the given agent, find the first (slot, preferred) pair
    where some rule produces preferred AND the rule's trigger event
    can fire with this agent in the agent role AND other roles can be
    filled from the scene.

    Returns (verb_lemma, {role_name: entity_id}) or None."""
    agent = trace.entities[agent_id]
    for slot, preferred in displeased_slots(agent, trace):
        for rule in rules:
            if not _rule_writes(rule, slot, preferred):
                continue
            event_pat = _trigger_event_pattern(rule)
            if event_pat is None:
                continue
            # Agent must be able to take the agent role (or theme for
            # intransitive verbs whose subject is the theme — like
            # satiĝi, dormi).
            agent_role = None
            for cand in ("agent", "theme"):
                if cand not in event_pat.role_patterns:
                    continue
                if _entity_satisfies_pattern(
                        agent, event_pat.role_patterns[cand], lex, trace):
                    agent_role = cand
                    break
            if agent_role is None:
                continue
            # Verify verb-level role spec is also satisfied for this role.
            action = lex.actions.get(event_pat.action)
            if action is None:
                continue
            agent_role_spec = next(
                (r for r in action.roles if r.name == agent_role), None)
            if agent_role_spec is None:
                continue
            if not _role_spec_satisfied(agent, agent_role_spec, lex, trace):
                continue
            # Fill remaining roles from the scene.
            roles = {agent_role: agent_id}
            ok = True
            for role_spec in action.roles:
                if role_spec.name in roles:
                    continue
                eid = _find_role_filler(
                    role_spec, trace, lex, exclude=set(roles.values()))
                if eid is None:
                    ok = False
                    break
                roles[role_spec.name] = eid
            if not ok:
                continue
            return (event_pat.action, roles)
    return None


# ----------------------- general subgoaling -------------------------

def _entity_property_values(entity, slot, trace=None, derived=None) -> set:
    """Read all CURRENT values for slot as a set — handles multi-valued
    slots (locomotion=[walk, swim], ...) so callers can do membership
    checks instead of guessing the "primary" value.

    For multi-valued slots, returns the *union* of asserted and
    derived values (a person's locomotion is asserted=swim from
    concept-bake but walk gets derived from the havas_parton+piedo
    chain — both are real). For scalar slots, asserted wins (derived
    is fallback only)."""
    out: set = set()
    asserted = None
    if trace is not None:
        asserted = trace.property_at(entity.id, slot, len(trace.events))
    if asserted is None:
        asserted = entity.properties.get(slot)
    if asserted is not None:
        if isinstance(asserted, list):
            out.update(asserted)
        else:
            out.add(asserted)
    if derived is not None:
        derived_val = derived.get(entity.id, slot)
        if derived_val is not None:
            if isinstance(derived_val, list):
                out.update(derived_val)
            else:
                out.add(derived_val)
    return out


def _entity_property(entity, slot, trace=None, derived=None):
    """Compatibility wrapper: returns a representative single value
    (any element of the values set) or None. Use
    `_entity_property_values` when you need set-membership semantics
    against a list-valued slot."""
    values = _entity_property_values(entity, slot, trace, derived)
    if not values:
        return None
    return next(iter(values))


def _entity_satisfies_props(entity, props_dict, lex, trace=None, derived=None):
    """True if entity satisfies a {slot: value} dict + type if given.
    Set-membership for multi-valued slots."""
    if not props_dict:
        return True
    for slot, expected in props_dict.items():
        if slot == "type":
            if not lex.types.is_subtype(entity.entity_type, expected):
                return False
        elif slot == "concept":
            if entity.concept_lemma != expected:
                return False
        else:
            values = _entity_property_values(entity, slot, trace, derived)
            if expected not in values:
                return False
    return True


def _action_writes(action, slot, value):
    """True if the verb's intrinsic effects write slot=value on
    SOME target role. Returns the target role name or None."""
    for eff in action.effects:
        if eff.property == slot and eff.value == value:
            return eff.target_role
    return None


def _has_relation(relation, args, trace, derived=None, lex=None) -> bool:
    """True if rel(relation, *args) currently holds — checks asserted
    trace.relations first, then derived (via DerivedState) if given.
    For symmetric relations (per lex), also checks the swapped pair."""
    target = tuple(args)
    for r in trace.relations:
        if r.relation == relation and r.args == target:
            return True
    if derived is not None and derived.has_relation(relation, target):
        return True
    is_symmetric = (
        lex is not None
        and relation in lex.relations
        and lex.relations[relation].symmetric)
    if is_symmetric and len(target) == 2 and target[0] != target[1]:
        swapped = (target[1], target[0])
        for r in trace.relations:
            if r.relation == relation and r.args == swapped:
                return True
        if derived is not None and derived.has_relation(relation, swapped):
            return True
    return False


def _knower_already_knows_about(knower, thing, trace) -> bool:
    """True if knower has any konas(_, fakto) where the fakto's
    subjekto or objekto relation points to `thing`. Coarse check used
    by the drive sampler to skip already-known targets."""
    fakto_args = {("subjekto", "fakto", "entity"),
                  ("objekto", "fakto", "entity")}
    for r in trace.relations:
        if r.relation != "konas" or r.args[0] != knower:
            continue
        fakto_id = r.args[1]
        for r2 in trace.relations:
            if r2.relation in ("subjekto", "objekto") and r2.args[0] == fakto_id:
                if r2.args[1] == thing:
                    return True
    return False


def _container_of(entity_id, trace) -> str | None:
    """Return the `en`-container of entity_id, or None if it's not in
    any. Considers `en` only — `sur` is for surface placement which
    isn't a co-location anchor for actor co-presence."""
    for r in trace.relations:
        if r.relation == "en" and r.args[0] == entity_id:
            return r.args[1]
    return None


def _build_var_to_role(event_pat) -> dict[int, str]:
    """Map id(Var) → role name for every Var bound under any role
    pattern of the event pattern. Used to translate the rule's
    effect-side Vars back to role names the planner can bind."""
    var_to_role = {}
    for role_name, role_pat in event_pat.role_patterns.items():
        for v in _extract_bind_vars(role_pat):
            var_to_role[id(v)] = role_name
    return var_to_role


def _predict_created_id(create_eff, var_bindings):
    """Compute the id `create_entity` would produce given the
    currently-known Var → eid bindings. Returns None if the prediction
    can't be made (some Var unbound or concept is itself a Var)."""
    concept = create_eff.concept
    if isinstance(concept, Var):
        v = var_bindings.get(id(concept))
        if v is None:
            return None
        concept_str = v
    else:
        concept_str = concept
    if create_eff.entity_id is not None:
        return create_eff.entity_id
    if create_eff.id_parts is not None:
        parts = []
        for p in create_eff.id_parts:
            if isinstance(p, Var):
                v = var_bindings.get(id(p))
                if v is None:
                    return None
                parts.append(str(v))
            else:
                parts.append(str(p))
        return f"{concept_str}_from_{'_'.join(parts)}"
    if create_eff.id_from is not None:
        v = var_bindings.get(id(create_eff.id_from))
        if v is None:
            return None
        return f"{concept_str}_from_{v}"
    # Auto-numbered (counter.next_id) — can't predict; treat as
    # non-matching for goal-binding purposes.
    return None


def _find_relation_adders(rules, relation):
    """Yield (rule, event_pat, arg_sources) for rules whose `then`
    includes AddRelation(relation, args). Each arg's `source` is one
    of:
      ("role", role_name) — Var bound by event_pat's role pattern
      ("created", create_entity_eff) — Var bound by a create_entity in
                                       the same `then` (e.g. fakto)
    Args that can't be sourced fail the rule entirely."""
    out = []
    for rule in rules:
        effects = (rule.then if isinstance(rule.then, (list, tuple))
                   else [rule.then])
        event_pat = _trigger_event_pattern(rule)
        if event_pat is None:
            continue
        var_to_role = _build_var_to_role(event_pat)
        # Collect create_entity outputs in this rule.
        var_to_creator: dict[int, CreateEntity] = {}
        for eff in effects:
            if isinstance(eff, CreateEntity):
                var_to_creator[id(eff.as_var)] = eff
        for eff in effects:
            if not isinstance(eff, AddRelation):
                continue
            if eff.relation != relation:
                continue
            arg_sources = []
            for arg in eff.args:
                if not isinstance(arg, Var):
                    arg_sources.append(None)
                    break
                if id(arg) in var_to_role:
                    arg_sources.append(("role", var_to_role[id(arg)]))
                elif id(arg) in var_to_creator:
                    arg_sources.append(("created", var_to_creator[id(arg)]))
                else:
                    arg_sources.append(None)
            if all(s is not None for s in arg_sources):
                out.append((rule, event_pat, tuple(arg_sources)))
    return out


def _resolve_preconditions(action, event_pat, roles, actor_id,
                           trace, lex, rules, derived,
                           max_depth, depth, seen, *, derivations=None):
    """For each role binding, recursively satisfy:
      - verb-level role property constraints,
      - rule-level role pattern entity constraints (if rule given),
      - action.preconditions (cross-role relations from the schema).
    Returns (sub_plans, ok). Sub-plans are concatenated in the order
    they're discovered (which is the order they need to fire)."""
    sub_plans = []
    for role_name, eid in list(roles.items()):
        ent = trace.entities.get(eid)
        if ent is None:
            return [], False
        role_spec = next(
            (r for r in action.roles if r.name == role_name), None)
        # Verb-level role property constraints.
        if role_spec is not None and role_spec.properties:
            for prop_slot, prop_vals in role_spec.properties.items():
                values = _entity_property_values(ent, prop_slot, trace, derived)
                if values & set(prop_vals):
                    continue
                expected = prop_vals[0] if prop_vals else None
                if expected is None:
                    return [], False
                sub = plan_to_achieve(
                    eid, prop_slot, expected, actor_id,
                    trace, lex, rules, derived=derived,
                    derivations=derivations,
                    max_depth=max_depth, _depth=depth, _seen=seen)
                if sub is None:
                    return [], False
                sub_plans.extend(sub)
        # Rule-level role pattern entity constraints.
        if event_pat is not None:
            role_pat = event_pat.role_patterns.get(role_name)
            if role_pat is not None:
                pattern_props = _extract_pattern_props(role_pat)
                for prop_slot, expected in pattern_props.items():
                    values = _entity_property_values(ent, prop_slot, trace, derived)
                    if expected in values:
                        continue
                    sub = plan_to_achieve(
                        eid, prop_slot, expected, actor_id,
                        trace, lex, rules, derived=derived,
                        derivations=derivations,
                        max_depth=max_depth, _depth=depth, _seen=seen)
                    if sub is None:
                        return [], False
                    sub_plans.extend(sub)

    # Action-level preconditions from the schema. Generic dispatch —
    # any RelationPrecondition routes through plan_to_establish_relation,
    # which itself handles derived relations via derivation walking.
    # `samloke` reaches this path with no special-case; the planner
    # walks the `shared_container_means_samloke` derivation to
    # subgoal the underlying `en` relations.
    for pc in action.preconditions:
        eids = [roles.get(rn) for rn in pc.roles]
        if any(e is None for e in eids):
            continue
        sub = plan_to_establish_relation(
            pc.rel, tuple(eids), actor_id,
            trace, lex, rules, derived=derived, derivations=derivations,
            max_depth=max_depth, _depth=depth, _seen=seen)
        if sub is None:
            return [], False
        sub_plans.extend(sub)
    return sub_plans, True


def plan_to_establish_relation(relation, target_args, actor_id,
                               trace, lex, rules, *, derived=None,
                               derivations=None,
                               max_depth=4, _depth=0, _seen=None):
    """Find a sequence of actions that asserts rel(relation, *target_args).
    Returns [] if already true, list of (verb, roles), or None.

    If no verb adds the relation directly AND `derivations` is given,
    walks derivations whose `implies` produces the relation and
    subgoals their `when + given` patterns. This is how `samloke` etc.
    are reached — no verb adds samloke, but a derivation produces it
    from shared `en` containers."""
    _seen = _seen or set()
    key = ("rel", relation, tuple(target_args))
    if key in _seen or _depth >= max_depth:
        return None
    if _has_relation(relation, target_args, trace, derived, lex):
        return []
    seen_next = _seen | {key}

    # Altruism preference: when actor isn't one of the relation's
    # participants, prefer candidates where actor will end up bound
    # to the agent role (i.e., "agent" is in the action's roles AND
    # not already consumed by AddRelation arg-binding). For
    # havi(Mikael, food) with actor=Lidia: preni binds agent=Mikael
    # (no altruism), doni binds agent=Lidia (altruism). Sort doni
    # before preni so the actor-involving plan wins.
    rel_candidates = list(_find_relation_adders(rules, relation))
    if actor_id not in tuple(target_args):
        def _priority(cand):
            _rule, _event_pat, arg_sources = cand
            action = lex.actions.get(_event_pat.action)
            if action is None:
                return 1
            action_role_names = {r.name for r in action.roles}
            role_arg_names = {
                src[1] for src in arg_sources if src[0] == "role"}
            actor_will_bind = (
                "agent" in action_role_names
                and "agent" not in role_arg_names)
            return 0 if actor_will_bind else 1
        rel_candidates.sort(key=_priority)

    for rule, event_pat, arg_sources in rel_candidates:
        action = lex.actions.get(event_pat.action)
        if action is None:
            continue
        # Bind from AddRelation arg sources. ("role", N) → role binding.
        # ("created", create_eff) → back-derive bindings for the
        # create_entity's id_parts/initial_properties Vars from the
        # target entity's properties (which already exist for
        # pre-created faktos in the scene).
        roles = {}
        # Vars bound from create_entity back-derivation. Keyed by
        # id(Var); planner consults this when filling role/given
        # constraints.
        predetermined: dict[int, str] = {}
        ok = True
        for src, target_eid in zip(arg_sources, target_args):
            if src[0] == "role":
                role_name = src[1]
                if role_name in roles and roles[role_name] != target_eid:
                    ok = False
                    break
                roles[role_name] = target_eid
            elif src[0] == "created":
                create_eff: CreateEntity = src[1]
                target_ent = trace.entities.get(target_eid)
                if target_ent is None:
                    ok = False
                    break
                # Back-derive from initial_properties (slot values
                # set at creation), then from sibling AddRelation
                # effects in the same `then` block (relations the
                # rule asserts on the new entity, e.g. subjekto/objekto
                # for fakto entities). Both sources let the planner
                # recover bindings for vars the create_entity / sibling
                # add_relation would emit, by reading the corresponding
                # asserted state on the goal entity.
                if create_eff.initial_properties:
                    for slot, val in create_eff.initial_properties.items():
                        if not isinstance(val, Var):
                            continue
                        actual = target_ent.properties.get(slot, [None])
                        actual_v = (actual[0] if isinstance(actual, list)
                                    and actual else actual)
                        if actual_v is None:
                            ok = False
                            break
                        existing = predetermined.get(id(val))
                        if existing is not None and existing != actual_v:
                            ok = False
                            break
                        predetermined[id(val)] = actual_v
                if not ok:
                    break
                # Sibling AddRelation back-derivation. For each AddRel
                # in rule.then where one arg is the create_eff's as_var
                # (now bound to target_eid), look up the asserted
                # relation in the trace and bind the OTHER arg's Var
                # to whatever the trace says.
                rule_effects = (rule.then if isinstance(
                    rule.then, (list, tuple)) else [rule.then])
                for r_eff in rule_effects:
                    if not isinstance(r_eff, AddRelation):
                        continue
                    if len(r_eff.args) != 2:
                        continue
                    create_var = create_eff.as_var
                    create_idx = None
                    for i, arg in enumerate(r_eff.args):
                        if isinstance(arg, Var) and arg is create_var:
                            create_idx = i
                            break
                    if create_idx is None:
                        continue
                    other_idx = 1 - create_idx
                    other_arg = r_eff.args[other_idx]
                    if not isinstance(other_arg, Var):
                        continue
                    found_value = None
                    for asserted in trace.relations:
                        if asserted.relation != r_eff.relation:
                            continue
                        if len(asserted.args) != 2:
                            continue
                        if asserted.args[create_idx] != target_eid:
                            continue
                        found_value = asserted.args[other_idx]
                        break
                    if found_value is None:
                        continue
                    existing = predetermined.get(id(other_arg))
                    if existing is not None and existing != found_value:
                        ok = False
                        break
                    predetermined[id(other_arg)] = found_value
                if not ok:
                    break
        if not ok:
            continue
        # For created vars, back-propagate to event_pat role bindings:
        # if a Var is bound by a role pattern AND we've also derived
        # its value from the target fakto, that role gets pre-bound.
        for role_name, role_pat in event_pat.role_patterns.items():
            for v in _extract_bind_vars(role_pat):
                if id(v) in predetermined:
                    pre_eid = predetermined[id(v)]
                    if role_name in roles and roles[role_name] != pre_eid:
                        ok = False
                        break
                    roles[role_name] = pre_eid
            if not ok:
                break
        if not ok:
            continue
        # Verify each ("created") arg's create_entity will actually
        # produce the target entity. Without this check, a rule that
        # adds rel(R, X, Y) where X is a *new* entity would get
        # accepted with target_args[0]=elena even though the new
        # entity will be flako_from_insulo (different id). Catches
        # the rain_creates_puddle false-positive for "elena en insulo"
        # location goals.
        var_bindings: dict[int, str] = dict(predetermined)
        for role_name, role_pat in event_pat.role_patterns.items():
            if role_name in roles:
                for v in _extract_bind_vars(role_pat):
                    var_bindings.setdefault(id(v), roles[role_name])
        bad_create = False
        for src, target_eid in zip(arg_sources, target_args):
            if src[0] != "created":
                continue
            create_eff = src[1]
            predicted = _predict_created_id(create_eff, var_bindings)
            if predicted is None or predicted != target_eid:
                bad_create = True
                break
        if bad_create:
            continue
        # Bind agent role to actor if not already bound. Refuse if
        # actor is already serving another role (would yield "Maria
        # metis Marian en la salono" — placing oneself).
        action_role_names = {r.name for r in action.roles}
        if "agent" in action_role_names and "agent" not in roles:
            if actor_id in roles.values():
                continue
            roles["agent"] = actor_id
        # Type-check fixed bindings against verb role specs.
        bad = False
        for role_name, eid in roles.items():
            ent = trace.entities.get(eid)
            role_spec = next(
                (r for r in action.roles if r.name == role_name), None)
            if (ent is None or role_spec is None
                    or not lex.types.is_subtype(
                        ent.entity_type, role_spec.type)):
                bad = True
                break
        if bad:
            continue
        # Reject if the same entity ends up in two different roles —
        # "Maria vidis Marian" / "kato vidis katon" / "Maria metis
        # Marian en la salono" are incoherent for the verbs we have.
        # Reflexive verbs (sin doni etc.) would need explicit opt-in.
        if len(set(roles.values())) != len(roles):
            continue
        # Fill remaining roles from scene.
        for role_spec in action.roles:
            if role_spec.name in roles:
                continue
            eid = _find_role_filler(
                role_spec, trace, lex, derived=derived,
                exclude=set(roles.values()))
            if eid is None:
                ok = False
                break
            roles[role_spec.name] = eid
        if not ok:
            continue
        # Subgoal on preconditions.
        sub_plans, ok = _resolve_preconditions(
            action, event_pat, roles, actor_id, trace, lex, rules, derived,
            max_depth, _depth + 1, seen_next, derivations=derivations)
        if not ok:
            continue
        return sub_plans + [(event_pat.action, roles)]

    # No verb plan worked. Try derivations whose `implies` produces
    # this relation — subgoal their `when + given` patterns. This is
    # how `samloke` is reachable: no verb adds samloke, but the
    # `shared_container_means_samloke` derivation says it follows from
    # both entities sharing an `en` container, so we subgoal for that.
    if derivations:
        sub = _plan_via_derivation(
            relation, target_args, actor_id, trace, lex, rules,
            derivations, derived, max_depth, _depth + 1, seen_next)
        if sub is not None:
            return sub
    return None


def _walk_for_rel_patterns(pattern):
    """All RelPatterns reachable from the (possibly And/Not-wrapped)
    pattern. NotPatterns are skipped — they're negative constraints
    not subgoaled toward. (A negated rel in a derivation's `given`
    means 'this must NOT hold'; the planner can't actively make
    something not hold via the current mechanism.)"""
    out = []
    if isinstance(pattern, RelPattern):
        out.append(pattern)
    elif isinstance(pattern, AndPattern):
        out.extend(_walk_for_rel_patterns(pattern.left))
        out.extend(_walk_for_rel_patterns(pattern.right))
    return out


def _walk_for_not_patterns(pattern):
    """Yield NotPatterns reachable from And-wrapping. Used by the
    derivation planner to check that negative preconditions don't
    currently hold for a candidate binding."""
    if isinstance(pattern, NotPattern):
        yield pattern
    elif isinstance(pattern, AndPattern):
        yield from _walk_for_not_patterns(pattern.left)
        yield from _walk_for_not_patterns(pattern.right)


def _arg_pattern_constraints(arg_pat):
    """Collect literal EntityPattern constraints (type/concept/has_suffix
    + literal property values) from a rel arg pattern. Walks through
    AndPatterns. Used to filter free-arg matches in NotPattern checks."""
    out: dict = {}
    if isinstance(arg_pat, EntityPattern):
        for k, v in arg_pat.constraints.items():
            if not isinstance(v, Var):
                out[k] = v
    elif isinstance(arg_pat, AndPattern):
        out.update(_arg_pattern_constraints(arg_pat.left))
        out.update(_arg_pattern_constraints(arg_pat.right))
    return out


def _notpattern_inner_holds(inner, var_bindings, trace, derived, lex) -> bool:
    """True if the NotPattern's inner currently holds with var_bindings
    (so the negation fails). Handles RelPattern inners — the only shape
    used inside ~rel(...) in the derivations the planner walks. For
    other shapes, returns False (be permissive: better to attempt a
    plan that fails than to skip a viable derivation)."""
    if not isinstance(inner, RelPattern):
        return False
    rel_def = lex.relations.get(inner.relation)
    if rel_def is None:
        return False
    arg_specs: list = []
    for arg_name in rel_def.arg_names:
        arg_pat = inner.arg_patterns.get(arg_name)
        if arg_pat is None:
            arg_specs.append((None, {}))
            continue
        v = _bind_var_in_pattern(arg_pat)
        if v is not None and id(v) in var_bindings:
            arg_specs.append((var_bindings[id(v)], {}))
        else:
            arg_specs.append((None, _arg_pattern_constraints(arg_pat)))

    def match(args):
        if len(args) != len(arg_specs):
            return False
        for (concrete, filt), actual in zip(arg_specs, args):
            if concrete is not None and concrete != actual:
                return False
            if filt:
                ent = trace.entities.get(actual)
                if ent is None:
                    return False
                if not _entity_matches_literal_constraints(ent, filt, lex):
                    return False
        return True

    for r in trace.relations:
        if r.relation == inner.relation and match(r.args):
            return True
    if derived is not None:
        for (name, args) in derived.relations:
            if name == inner.relation and match(args):
                return True
    return False


def _notpatterns_violated(when, given, var_bindings, trace, derived,
                          lex) -> bool:
    """True if any NotPattern in when+given currently holds with the
    given bindings — meaning the derivation can't fire even if its
    positive subgoals are achieved."""
    for pat in [when] + list(given):
        for np in _walk_for_not_patterns(pat):
            if _notpattern_inner_holds(np.inner, var_bindings,
                                        trace, derived, lex):
                return True
    return False


def _entity_matches_literal_constraints(ent, constraints, lex) -> bool:
    """Lightweight literal-only match — for filtering free-var
    candidates by entity-pattern constraints found in a derivation.
    No bindings, no derived properties; only checks asserted
    `entity.properties` plus type/concept/has_suffix. Conservative:
    Var-valued constraints would be skipped here (ignored upstream)."""
    for key, expected in constraints.items():
        if isinstance(expected, Var):
            continue  # Var-valued — caller skipped these
        if key == "type":
            if not lex.types.is_subtype(ent.entity_type, expected):
                return False
        elif key == "has_suffix":
            if not ent.concept_lemma.endswith(expected):
                return False
        elif key == "concept":
            if ent.concept_lemma != expected:
                return False
        else:
            actual = ent.properties.get(key, [])
            if isinstance(actual, list):
                if expected not in actual:
                    return False
            elif actual != expected:
                return False
    return True


def _walk_for_entity_patterns_binding(pattern, target_var):
    """Yield EntityPatterns that bind `target_var` (via an And-wrapped
    BindPattern in the same conjunction). Used to discover literal
    constraints on a free Var so candidate enumeration can pre-filter
    instead of relying on rel-pattern subgoaling to fail wrong picks."""
    if isinstance(pattern, AndPattern):
        # Check if either side is an EntityPattern AND the conjunction
        # binds target_var.
        bound = _bind_var_in_pattern(pattern)
        if bound is target_var:
            if isinstance(pattern.left, EntityPattern):
                yield pattern.left
            if isinstance(pattern.right, EntityPattern):
                yield pattern.right
        yield from _walk_for_entity_patterns_binding(pattern.left, target_var)
        yield from _walk_for_entity_patterns_binding(pattern.right, target_var)


def _bind_var_in_pattern(pattern):
    """Return the Var bound by a BindPattern (possibly inside an And).
    None if no Var is being bound."""
    if isinstance(pattern, BindPattern):
        return pattern.target
    if isinstance(pattern, AndPattern):
        v = _bind_var_in_pattern(pattern.left)
        if v is not None:
            return v
        return _bind_var_in_pattern(pattern.right)
    return None


def _plan_via_derivation(relation, target_args, actor_id, trace, lex, rules,
                          derivations, derived,
                          max_depth, depth, seen):
    """Subgoal through a derivation that implies the goal relation.

    Algorithm: find derivations whose `implies` produces `relation`;
    bind their implication args to the target args; collect rel
    patterns from when+given; identify free vars; for each free-var
    assignment from a candidate pool (current containers + all scene
    locations), subgoal each rel pattern as a sub-goal.

    Currently handles ≤1 free var per derivation — covers samloke and
    similar 'shared X' shapes. Multi-free-var derivations are rare
    enough to defer."""
    for d in derivations:
        for imp in d.implies:
            if not isinstance(imp, RelationImplication):
                continue
            if imp.name != relation or len(imp.args) != len(target_args):
                continue
            # Bind imp.args ↔ target_args. Var-args record bindings;
            # literal-args must equal the corresponding target.
            var_bindings: dict[int, str] = {}
            mismatch = False
            for arg, target in zip(imp.args, target_args):
                if isinstance(arg, Var):
                    if id(arg) in var_bindings and var_bindings[id(arg)] != target:
                        mismatch = True
                        break
                    var_bindings[id(arg)] = target
                elif arg != target:
                    mismatch = True
                    break
            if mismatch:
                continue

            # Collect rel patterns from when + given.
            patterns = [d.when] + list(d.given)
            rel_patterns = []
            for p in patterns:
                rel_patterns.extend(_walk_for_rel_patterns(p))
            if not rel_patterns:
                continue

            # Resolve each pattern: each arg is ("bound", eid),
            # ("free", var_id), or ("wild", None).
            resolved = []
            free_vars: set[int] = set()
            ok = True
            for rp in rel_patterns:
                rel_def = lex.relations.get(rp.relation)
                if rel_def is None:
                    ok = False
                    break
                arg_kinds = []
                for arg_name in rel_def.arg_names:
                    arg_pat = rp.arg_patterns.get(arg_name)
                    if arg_pat is None:
                        arg_kinds.append(("wild", None))
                        continue
                    v = _bind_var_in_pattern(arg_pat)
                    if v is None:
                        ok = False
                        break
                    if id(v) in var_bindings:
                        arg_kinds.append(("bound", var_bindings[id(v)]))
                    else:
                        arg_kinds.append(("free", id(v)))
                        free_vars.add(id(v))
                if not ok:
                    break
                resolved.append((rp.relation, arg_kinds))
            if not ok:
                continue
            if len(free_vars) > 1:
                continue  # multi-free-var derivations not handled

            # Free-var candidate enumeration. Filter by any literal
            # entity-pattern constraint that binds the free var (so the
            # planner doesn't try wrong-typed entities or wrong-relacio
            # faktos when the derivation's `given` already pins them).
            # Falls back to "containers + locations" when no constraint
            # exists (samloke-style derivations where F is a location).
            assignments: list[dict[int, str]]
            if not free_vars:
                assignments = [{}]
            else:
                fv_id = next(iter(free_vars))
                fv_var = next(
                    v for v in (
                        list(_extract_bind_vars(d.when))
                        + [v for p in d.given
                           for v in _extract_bind_vars(p)])
                    if id(v) == fv_id)
                identity, slots_to_subgoal = _split_entity_constraints(
                    d.when, d.given, fv_var)
                if identity:
                    candidates = [
                        eid for eid, ent in trace.entities.items()
                        if _entity_matches_literal_constraints(
                            ent, identity, lex)
                    ]
                else:
                    # Heuristic: containers of bound targets first,
                    # then all locations. Used when the derivation
                    # imposes no entity-level identity constraints on
                    # the free var (samloke shape).
                    candidates = []
                    seen_cand: set[str] = set()
                    for target in target_args:
                        c = _container_of(target, trace)
                        if c is not None and c not in seen_cand:
                            seen_cand.add(c)
                            candidates.append(c)
                    for eid, ent in trace.entities.items():
                        if eid in seen_cand:
                            continue
                        if lex.types.is_subtype(ent.entity_type, "location"):
                            seen_cand.add(eid)
                            candidates.append(eid)
                assignments = [{fv_id: c} for c in candidates]

            for assignment in assignments:
                combined = {**var_bindings, **assignment}
                if _notpatterns_violated(d.when, d.given, combined,
                                         trace, derived, lex):
                    continue
                sub_plans = []
                sub_ok = True
                for rel_name_pat, arg_kinds in resolved:
                    concrete = []
                    for kind, val in arg_kinds:
                        if kind == "bound":
                            concrete.append(val)
                        elif kind == "free":
                            concrete.append(assignment[val])
                        else:
                            sub_ok = False  # wildcard — can't establish
                            break
                    if not sub_ok:
                        break
                    sub = plan_to_establish_relation(
                        rel_name_pat, tuple(concrete), actor_id,
                        trace, lex, rules,
                        derived=derived, derivations=derivations,
                        max_depth=max_depth, _depth=depth, _seen=seen)
                    if sub is None:
                        sub_ok = False
                        break
                    sub_plans.extend(sub)
                if not sub_ok:
                    continue
                # Subgoal entity-pattern slot constraints (lock_state=
                # malŝlosita on the seruro, etc.). The bound entity
                # might already satisfy; otherwise plan_to_achieve.
                if free_vars:
                    for fv_var_local, slot, value in slots_to_subgoal:
                        if id(fv_var_local) != fv_id:
                            continue
                        target_eid = assignment[fv_id]
                        target_ent = trace.entities.get(target_eid)
                        if target_ent is None:
                            sub_ok = False
                            break
                        actual = _entity_property_values(
                            target_ent, slot, trace, derived)
                        if value in actual:
                            continue
                        sub = plan_to_achieve(
                            target_eid, slot, value, actor_id,
                            trace, lex, rules,
                            derived=derived, derivations=derivations,
                            max_depth=max_depth, _depth=depth, _seen=seen)
                        if sub is None:
                            sub_ok = False
                            break
                        sub_plans.extend(sub)
                if sub_ok:
                    return sub_plans
    return None


def _split_entity_constraints(when, given, target_var):
    """Walk when+given for entity patterns binding `target_var`. Split
    their constraints into:
      identity: concept/type/has_suffix — these define what KIND of
        entity it is, used to filter candidates.
      slots: ordinary property slots — these define current STATE,
        which can be subgoaled via plan_to_achieve when the candidate
        doesn't already satisfy.
    Returns (identity_dict, slot_subgoal_list).
    """
    identity: dict[str, Any] = {}
    slots: list[tuple[Var, str, Any]] = []
    for pat in [when] + list(given):
        for ep in _walk_for_entity_patterns_binding(pat, target_var):
            for k, val in ep.constraints.items():
                if isinstance(val, Var):
                    continue  # Var-valued — handled elsewhere
                if k in ("type", "concept", "has_suffix"):
                    identity[k] = val
                else:
                    slots.append((target_var, k, val))
    return identity, slots


def plan_event_firing(verb, requested_roles, actor_id,
                      trace, lex, rules, *, derived=None, derivations=None,
                      max_depth=5, _depth=0, _seen=None):
    """Plan to fire `verb` with the requested roles, resolving any
    preconditions. Used by drives that target a specific verb-event
    rather than a goal state — e.g. knowledge drives currently target
    `vidi(knower, thing)` directly because the planner can't yet
    synthesize the konas goal over a not-yet-existing fakto entity.

    `requested_roles` is a partial role binding the caller already
    knows; remaining roles get filled from the scene the same way
    plan_to_achieve does."""
    _seen = _seen or set()
    key = ("event", verb, tuple(sorted(requested_roles.items())))
    if key in _seen or _depth >= max_depth:
        return None
    seen_next = _seen | {key}

    action = lex.actions.get(verb)
    if action is None:
        return None

    roles = dict(requested_roles)
    # Type-check the requested bindings.
    for role_name, eid in roles.items():
        ent = trace.entities.get(eid)
        role_spec = next(
            (r for r in action.roles if r.name == role_name), None)
        if (ent is None or role_spec is None
                or not lex.types.is_subtype(ent.entity_type, role_spec.type)):
            return None

    # Fill remaining roles from scene.
    for role_spec in action.roles:
        if role_spec.name in roles:
            continue
        eid = _find_role_filler(
            role_spec, trace, lex, derived=derived,
            exclude=set(roles.values()))
        if eid is None:
            return None
        roles[role_spec.name] = eid

    sub_plans, ok = _resolve_preconditions(
        action, None, roles, actor_id, trace, lex, rules, derived,
        max_depth, _depth + 1, seen_next, derivations=derivations)
    if not ok:
        return None
    return sub_plans + [(verb, roles)]


def plan_to_co_locate(eid_a, eid_b, actor_id, trace, lex, rules, *,
                      derived=None, derivations=None,
                      max_depth=4, _depth=0, _seen=None):
    """Compatibility wrapper — co-location is now a `samloke` relation
    goal handled generically by `plan_to_establish_relation` walking
    the `shared_container_means_samloke` derivation. Kept so existing
    callers don't break; new code should call
    `plan_to_establish_relation("samloke", (a, b), ...)` directly."""
    return plan_to_establish_relation(
        "samloke", (eid_a, eid_b), actor_id,
        trace, lex, rules, derived=derived, derivations=derivations,
        max_depth=max_depth, _depth=_depth, _seen=_seen)


def plan_to_achieve(goal_entity_id, goal_slot, goal_value,
                    actor_id, trace, lex, rules, *, derived=None,
                    derivations=None,
                    max_depth=3, _depth=0, _seen=None):
    """General-purpose subgoal planner. Find a sequence of actions
    that, when executed, would result in `goal_entity` having
    `goal_slot=goal_value`. Returns a list of (verb, roles) — possibly
    empty if already satisfied, or None if no plan found.

    Uses ONLY the rule library — no hardcoded heuristics. The
    algorithm:
      1. If goal already met: return []
      2. For each rule whose effect writes (?, goal_slot, goal_value):
         a. Identify the trigger event + which role corresponds to
            the entity whose state will change (the EFFECT TARGET).
         b. Bind that role to goal_entity_id; bind agent role to
            actor_id.
         c. For each remaining role, find a candidate from the scene.
         d. For each role with property constraints (verb-level OR
            rule-level) the candidate doesn't satisfy: recurse with
            (candidate, slot, required_value) as the new sub-goal.
         e. If all sub-goals resolve, return [sub-plans..., this-action].
      3. If no rule works, return None.

    `_seen` tracks (entity, slot, value) tuples in the current
    recursion stack to prevent cycles.
    """
    _seen = _seen or set()
    key = ("prop", goal_entity_id, goal_slot, goal_value)
    if key in _seen or _depth >= max_depth:
        return None

    # At top-level entry, materialize the derived layer once if the
    # caller passed derivations but no precomputed cache. Recursive
    # calls just thread `derived` through.
    if derived is None and derivations is not None and _depth == 0:
        derived = compute_derived_state(trace, derivations, lex)

    goal_entity = trace.entities.get(goal_entity_id)
    if goal_entity is None:
        return None

    # Already satisfied?
    if goal_value in _entity_property_values(goal_entity, goal_slot, trace, derived):
        return []

    seen_next = _seen | {key}

    # Build candidate (action, target_role, optional_rule) tuples.
    # Cascade verbs (no agent role: satiĝi, rompiĝi, morti, ...) are
    # excluded from verb-direct candidates — they're emitted as the
    # *result* of an agent's action, not chosen by an actor. Without
    # this filter the planner picks satiĝi(maria) directly for
    # hunger=sata, bypassing the meaningful manĝi step.
    candidates = []
    for action in lex.actions.values():
        if "agent" not in {r.name for r in action.roles}:
            continue
        target_role = _action_writes(action, goal_slot, goal_value)
        if target_role is not None:
            candidates.append((action, target_role, None))
    for rule in rules:
        if not _rule_writes(rule, goal_slot, goal_value):
            continue
        event_pat = _trigger_event_pattern(rule)
        if event_pat is None:
            continue
        action = lex.actions.get(event_pat.action)
        if action is None:
            continue
        target_role = _effect_target_role(rule, event_pat, goal_slot)
        if target_role is None:
            continue
        candidates.append((action, target_role, rule))

    # Altruism preference: when the drive's actor differs from the
    # goal_entity, prefer candidates where the goal_entity is NOT in
    # the agent role — those are the candidates that let `actor` fill
    # the agent role and actually do something. Without this, "Lidia
    # wants Mikael fed" picks `hungry_eats_sated` (target_role=agent
    # → Mikael acts on himself) rather than a doni→manĝi chain that
    # involves Lidia. Self-acting candidates remain as fallback.
    if actor_id != goal_entity_id:
        candidates.sort(key=lambda c: 0 if c[1] != "agent" else 1)

    for action, target_role, rule in candidates:
        event_pat = _trigger_event_pattern(rule) if rule else None

        target_role_spec = next(
            (r for r in action.roles if r.name == target_role), None)
        if target_role_spec is None:
            continue
        if not lex.types.is_subtype(
                goal_entity.entity_type, target_role_spec.type):
            continue

        roles = {target_role: goal_entity_id}
        if "agent" in {r.name for r in action.roles} and target_role != "agent":
            if actor_id == goal_entity_id:
                continue
            actor = trace.entities.get(actor_id)
            actor_role_spec = next(
                (r for r in action.roles if r.name == "agent"), None)
            if actor_role_spec is None:
                continue
            if not _role_spec_satisfied(actor, actor_role_spec, lex, trace, derived):
                continue
            roles["agent"] = actor_id

        ok = True
        for role_spec in action.roles:
            if role_spec.name in roles:
                continue
            eid = _find_role_filler(
                role_spec, trace, lex, derived=derived,
                exclude=set(roles.values()))
            if eid is None:
                ok = False
                break
            roles[role_spec.name] = eid
        if not ok:
            continue
        # Reject duplicate-entity bindings (same entity in two roles).
        if len(set(roles.values())) != len(roles):
            continue

        sub_plans, ok = _resolve_preconditions(
            action, event_pat, roles, actor_id, trace, lex, rules, derived,
            max_depth, _depth + 1, seen_next, derivations=derivations)
        if not ok:
            continue

        verb_lemma = event_pat.action if event_pat else action.lemma
        return sub_plans + [(verb_lemma, roles)]

    # No verb/rule wrote the goal property directly. Try derivations:
    # a Derivation whose `implies` is a PropertyImplication matching
    # (slot, value) on the goal entity. Subgoal its when+given to
    # establish the conditions that make the property derivable.
    # This is what lets the planner reach door.lock_state=malŝlosita
    # via the lifted-from-seruro derivation: subgoal devolves to
    # malŝlosi the seruro.
    if derivations:
        sub = _plan_property_via_derivation(
            goal_entity_id, goal_slot, goal_value, actor_id,
            trace, lex, rules, derivations, derived,
            max_depth, _depth + 1, seen_next)
        if sub is not None:
            return sub
    return None


def _plan_property_via_derivation(entity_id, slot, value, actor_id,
                                   trace, lex, rules, derivations, derived,
                                   max_depth, depth, seen):
    """Subgoal through a derivation whose `implies` is a
    PropertyImplication producing entity_id.slot=value. Mirrors
    `_plan_via_derivation` but for property goals: bind imp.entity
    to entity_id, then walk when+given as if for a relation goal.
    Free vars get the same enumeration treatment (filter by literal
    entity-pattern constraints if any bind them; otherwise fall back
    to all-entity enumeration)."""
    for d in derivations:
        for imp in d.implies:
            if not isinstance(imp, PropertyImplication):
                continue
            if imp.slot != slot:
                continue
            # Resolve imp.value against literals (no Var case for now).
            if imp.value != value:
                continue
            if not isinstance(imp.entity, Var):
                continue
            # Bind imp.entity → entity_id.
            var_bindings: dict[int, str] = {id(imp.entity): entity_id}

            # Collect rel patterns from when + given.
            patterns = [d.when] + list(d.given)
            rel_patterns = []
            for p in patterns:
                rel_patterns.extend(_walk_for_rel_patterns(p))

            # Resolve each rel pattern's args.
            resolved = []
            free_vars: set[int] = set()
            ok = True
            for rp in rel_patterns:
                rel_def = lex.relations.get(rp.relation)
                if rel_def is None:
                    ok = False
                    break
                arg_kinds = []
                for arg_name in rel_def.arg_names:
                    arg_pat = rp.arg_patterns.get(arg_name)
                    if arg_pat is None:
                        arg_kinds.append(("wild", None))
                        continue
                    v = _bind_var_in_pattern(arg_pat)
                    if v is None:
                        ok = False
                        break
                    if id(v) in var_bindings:
                        arg_kinds.append(("bound", var_bindings[id(v)]))
                    else:
                        arg_kinds.append(("free", id(v)))
                        free_vars.add(id(v))
                if not ok:
                    break
                resolved.append((rp.relation, arg_kinds))
            if not ok:
                continue
            if len(free_vars) > 1:
                continue  # multi-free-var derivations not handled

            # Free-var candidate enumeration (same logic as
            # _plan_via_derivation: filter by literal constraints).
            assignments: list[dict[int, str]]
            if not free_vars:
                assignments = [{}]
            else:
                fv_id = next(iter(free_vars))
                fv_var = next(
                    (v for v in (
                        list(_extract_bind_vars(d.when))
                        + [v for p in d.given
                           for v in _extract_bind_vars(p)])
                     if id(v) == fv_id), None)
                if fv_var is None:
                    continue
                identity, slots_to_subgoal = _split_entity_constraints(
                    d.when, d.given, fv_var)
                if identity:
                    candidates = [
                        eid for eid, ent in trace.entities.items()
                        if _entity_matches_literal_constraints(
                            ent, identity, lex)
                    ]
                else:
                    candidates = list(trace.entities.keys())
                assignments = [{fv_id: c} for c in candidates]

            for assignment in assignments:
                combined = {**var_bindings, **assignment}
                if _notpatterns_violated(d.when, d.given, combined,
                                         trace, derived, lex):
                    continue
                sub_plans = []
                sub_ok = True
                for rel_name_pat, arg_kinds in resolved:
                    concrete = []
                    for kind, val in arg_kinds:
                        if kind == "bound":
                            concrete.append(val)
                        elif kind == "free":
                            concrete.append(assignment[val])
                        else:
                            sub_ok = False
                            break
                    if not sub_ok:
                        break
                    sub = plan_to_establish_relation(
                        rel_name_pat, tuple(concrete), actor_id,
                        trace, lex, rules,
                        derived=derived, derivations=derivations,
                        max_depth=max_depth, _depth=depth, _seen=seen)
                    if sub is None:
                        sub_ok = False
                        break
                    sub_plans.extend(sub)
                if not sub_ok:
                    continue
                # Subgoal entity-pattern slot constraints (e.g. the
                # seruro's lock_state=malŝlosita that lets the
                # host_lock_state_unlocked_from_seruro derivation
                # actually fire after we plan).
                if free_vars:
                    for fv_var_local, slot, value in slots_to_subgoal:
                        if id(fv_var_local) != fv_id:
                            continue
                        target_eid = assignment[fv_id]
                        target_ent = trace.entities.get(target_eid)
                        if target_ent is None:
                            sub_ok = False
                            break
                        actual = _entity_property_values(
                            target_ent, slot, trace, derived)
                        if value in actual:
                            continue
                        sub = plan_to_achieve(
                            target_eid, slot, value, actor_id,
                            trace, lex, rules,
                            derived=derived, derivations=derivations,
                            max_depth=max_depth, _depth=depth, _seen=seen)
                        if sub is None:
                            sub_ok = False
                            break
                        sub_plans.extend(sub)
                if sub_ok:
                    return sub_plans
    return None


def _effect_target_role(rule, event_pat, slot):
    """Identify which role variable the rule's effect mutates for the
    given slot. Walks rule.then for an Emit's property_changes or a
    Change effect, and matches the target Var back to the role name in
    event_pat.role_patterns."""
    effects = (rule.then if isinstance(rule.then, (list, tuple))
               else [rule.then])
    # Build var → role_name map from event_pat
    var_to_role = {}
    for role_name, role_pat in event_pat.role_patterns.items():
        for var in _extract_bind_vars(role_pat):
            var_to_role[id(var)] = role_name
    for eff in effects:
        if isinstance(eff, Emit):
            for (ent, eff_slot), _val in eff.property_changes.items():
                if eff_slot != slot:
                    continue
                role = var_to_role.get(id(ent))
                if role is not None:
                    return role
        if isinstance(eff, Change):
            if eff.slot != slot:
                continue
            role = var_to_role.get(id(eff.entity))
            if role is not None:
                return role
    return None


def _extract_bind_vars(pattern):
    """All Vars referenced by BindPatterns within this pattern.
    Recurses through And/Rel arg patterns so derivation patterns like
    `rel("en", contained=bind(A), container=bind(L))` yield both A
    and L."""
    from esperanto_lm.ontology.dsl.patterns import BindPattern
    out = []
    if isinstance(pattern, BindPattern):
        out.append(pattern.target)
    if isinstance(pattern, AndPattern):
        out.extend(_extract_bind_vars(pattern.left))
        out.extend(_extract_bind_vars(pattern.right))
    if isinstance(pattern, RelPattern):
        for arg_pat in pattern.arg_patterns.values():
            out.extend(_extract_bind_vars(arg_pat))
    return out


def _extract_pattern_props(pattern):
    """Extract {slot: expected_value} constraints from EntityPattern
    instances within the pattern (skipping type/concept/has_suffix)."""
    if isinstance(pattern, EntityPattern):
        return {k: v for k, v in pattern.constraints.items()
                if k not in ("type", "concept", "has_suffix")}
    if isinstance(pattern, AndPattern):
        out = dict(_extract_pattern_props(pattern.left))
        out.update(_extract_pattern_props(pattern.right))
        return out
    return {}


def plan_with_subgoals(agent_id, trace, lex, rules, *,
                       derived=None, derivations=None, max_depth=3):
    """Self-driven OR altruistic planning: scan all displeased
    entities (self first), and for each (target, slot, preferred) try
    the general subgoal planner with this agent as actor. Returns the
    first successful plan.

    Self-target gets priority so animates fix their own state before
    helping others; if the agent has no satisfiable self-goal, it
    looks across the scene for altruistic opportunities (Maria can
    wake Petro).

    Caller may pass `derived` (a precomputed DerivedState) or
    `derivations` (the engine derivations to compute it from). The
    derived layer is required for any goal whose verb references a
    derived role property (iri.agent.locomotion is the canonical case).
    """
    if derived is None and derivations is not None:
        derived = compute_derived_state(trace, derivations, lex)
    targets = displeased_entities(trace, derived)
    targets.sort(key=lambda triple: 0 if triple[0] == agent_id else 1)
    for target_id, slot, preferred in targets:
        plan = plan_to_achieve(
            target_id, slot, preferred, agent_id,
            trace, lex, rules, derived=derived, max_depth=max_depth)
        if plan:
            return plan
    return []


def _role_spec_satisfied(entity, role_spec, lex, trace=None, derived=None) -> bool:
    """Does the entity satisfy the verb's RoleSpec (type + properties)?
    Set-membership for multi-valued slots."""
    if not lex.types.is_subtype(entity.entity_type, role_spec.type):
        return False
    if role_spec.properties:
        for slot, vals in role_spec.properties.items():
            values = _entity_property_values(entity, slot, trace, derived)
            if not values & set(vals):
                return False
    return True


# ----------------------- simulation loop ---------------------------

def run_simulation(trace, lex, rules, derivations, max_ticks=8):
    """Tick loop. Each tick: find the first animate entity with a
    satisfiable plan (possibly multi-step), fire ALL steps of the plan
    in order, run the engine after each, repeat. Terminate when no
    agent can act or max_ticks reached.

    Multi-step plans count as a single tick — the agent commits to a
    sequence and executes it. (A more conservative loop would re-plan
    after each step; the current shape is fine for POC.)

    Derived state is recomputed per tick (cheap — pure function of
    trace + derivations) so role specs that gate on derived properties
    work transparently.
    """
    for tick in range(max_ticks):
        fired = False
        derived = compute_derived_state(trace, derivations, lex)
        for eid, ent in list(trace.entities.items()):
            if not lex.types.is_subtype(ent.entity_type, "animate"):
                continue
            if ent.destroyed_at_event is not None:
                continue
            plan = plan_with_subgoals(
                eid, trace, lex, rules, derived=derived)
            if not plan:
                continue
            for verb, roles in plan:
                event = make_event(
                    verb, roles=roles,
                    property_changes=effect_changes(verb, roles, lex))
                trace.events.append(event)
                run_dsl(trace, rules, derivations, lex)
            fired = True
            break
        if not fired:
            return tick
    return max_ticks


# ----------------------- demo scenes -------------------------------

def scene_hungry_maria(lex):
    """Maria is hungry; bread is on the table; she should eat it."""
    t = Trace()
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("pano", lex, entity_id="pano")
    t.add_entity("tablo", lex, entity_id="tablo")
    t.assert_relation("en", ("maria", "kuirejo"), lex)
    t.assert_relation("en", ("tablo", "kuirejo"), lex)
    t.assert_relation("sur", ("pano", "tablo"), lex)
    t.assert_relation("havi", ("maria", "pano"), lex)
    t.entities["maria"].set_property("hunger", "malsata")
    return t, "kuirejo"


def scene_asleep_petro(lex):
    """Petro is asleep; he should wake up. But he can't drive that
    himself — needs an external agent. Add Maria to wake him."""
    t = Trace()
    t.add_entity("salono", lex, entity_id="salono")
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("persono", lex, entity_id="maria")
    t.assert_relation("en", ("petro", "salono"), lex)
    t.assert_relation("en", ("maria", "salono"), lex)
    t.entities["petro"].set_property("sleep_state", "dormanta")
    t.entities["maria"].set_property("sleep_state", "vekita")
    return t, "salono"


def scene_hungry_maria_petro_has_bread(lex):
    """Multi-step demo: Maria is hungry but Petro has the bread.
    Direct plan (manĝi) is blocked by the heuristic 'should hold the
    theme'; sub-plan inserts a preni step. Expected sequence:
        preni(maria, pano)  — takes pano from petro
        manĝi(maria, pano)  — eats it
        satiĝi(maria)       — cascade
        manĝi_destroys      — pano gone
    """
    t = Trace()
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("pano", lex, entity_id="pano")
    t.assert_relation("en", ("maria", "kuirejo"), lex)
    t.assert_relation("en", ("petro", "kuirejo"), lex)
    t.assert_relation("havi", ("petro", "pano"), lex)
    t.entities["maria"].set_property("hunger", "malsata")
    t.entities["petro"].set_property("hunger", "sata")
    return t, "kuirejo"


# ----------------------- random scene + drive sampler --------------
#
# Composes a scene from primitives — location + 1-2 persons + some
# objects + maybe-ownership — then picks a drive (an animate's slot
# flipped to a non-preferred value). Returns to the caller without
# firing anything; the agent sim runs on top.
#
# The point: stop hand-crafting scenes. The sampler explores the
# space of possible (entities, relations, initial state) combos
# uniformly, then the planner figures out whether an interesting
# action chain exists. Most scenes are degenerate ("Maria has the
# bread, Maria is hungry" → 1-step manĝi); the runner filters by
# event count to find the coverage edges.

from esperanto_lm.ontology.containment import (
    reachable_from, resolve_containment,
)
from esperanto_lm.ontology.sampler import (
    PERSON_NAMES, _STRUCTURAL_SURFACES, _add_entity_randomized,
    _ensure_placed, _person_concepts,
)


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
    AddRelation or any derivation's RelationImplication."""
    out: set[str] = set()
    for rule in rules:
        effects = (rule.then if isinstance(rule.then, (list, tuple))
                   else [rule.then])
        for eff in effects:
            if isinstance(eff, AddRelation):
                out.add(eff.relation)
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
                        # SLOT_PREFERENCES preferred value (otherwise
                        # we'd model self-harm — "Maria wants to be
                        # malsata", which is fine mechanically but
                        # noise for coverage).
                        if (slot_name in SLOT_PREFERENCES
                                and SLOT_PREFERENCES[slot_name] == goal):
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


def plan_for_drive(drive, t, lex, rules, derivations, *, max_depth=5):
    """Dispatch one drive to the right planner entry. Returns the
    plan or None. Computes derived state once. Doesn't fire — caller
    is responsible for executing the plan against the trace."""
    derived = compute_derived_state(t, derivations, lex)
    kind = drive[0]
    if kind == "self_slot":
        _, actor, slot, value = drive
        return plan_to_achieve(
            actor, slot, value, actor, t, lex, rules,
            derived=derived, max_depth=max_depth)
    if kind == "entity_slot":
        _, actor, target, slot, value = drive
        return plan_to_achieve(
            target, slot, value, actor, t, lex, rules,
            derived=derived, max_depth=max_depth)
    if kind == "location":
        _, actor, loc = drive
        return plan_to_establish_relation(
            "en", (actor, loc), actor, t, lex, rules,
            derived=derived, derivations=derivations, max_depth=max_depth)
    if kind == "possession":
        _, actor, item = drive
        return plan_to_establish_relation(
            "havi", (actor, item), actor, t, lex, rules,
            derived=derived, derivations=derivations, max_depth=max_depth)
    if kind == "knowledge":
        _, actor, knower, fakto_id = drive
        return plan_to_establish_relation(
            "konas", (knower, fakto_id), actor, t, lex, rules,
            derived=derived, derivations=derivations, max_depth=max_depth)
    return None


def _drive_summary(drive):
    """Short human-readable form for the report."""
    kind = drive[0]
    if kind == "self_slot":
        return f"{drive[1]}.{drive[2]}={drive[3]}"
    if kind == "entity_slot":
        return f"{drive[1]} wants {drive[2]}.{drive[3]}={drive[4]}"
    if kind == "location":
        return f"{drive[1]} wants to be in {drive[2]}"
    if kind == "knowledge":
        actor, knower, fakto_id = drive[1], drive[2], drive[3]
        if actor == knower:
            return f"{actor} wants to know fact {fakto_id}"
        return f"{actor} wants {knower} to know fact {fakto_id}"
    if kind == "possession":
        return f"{drive[1]} wants to have {drive[2]}"
    return repr(drive)


# ---------------------- goal-regression sampler ----------------------
#
# Backward-from-goal scene construction (companion to the drive sampler).
# Pick a verb you want fired, instantiate its roles into a fresh scene,
# set the theme's initial state to the inverse of the verb's effect (so
# the verb has work to do), and emit a drive that asks the planner to
# achieve the effect. Multi-step chains fall out naturally: leaving a
# role-property precondition unsatisfied (e.g. theme.lock_state=malŝlosita
# when initial is ŝlosita) makes the planner subgoal on it via existing
# derivations + chained verbs — no augmenter, no boost weights.

# Verbs the regressor targets, with a per-verb relative weight. Equal
# weights = balanced across chains. Verbs not listed are skipped.
REGRESSION_VERBS = [
    ("malfermi", 1),
    ("fermi", 1),
    ("malŝlosi", 1),
    ("ŝlosi", 1),
    ("manĝi", 1),
    ("preni", 1),
    ("doni", 1),
    ("veki", 1),
    ("dormi", 1),
    ("iri", 1),
    ("rakonti", 1),
    ("vidi", 1),
]


def _concepts_matching_role(lex, role_spec) -> list[str]:
    """Concepts compatible with role_spec: subtype-correct AND any
    immutable role.properties already present on the concept (e.g.
    functional_signature=ŝlosi narrows instrument candidates to
    ŝlosilo). Mutable slots (varies=True) are not filtered — those are
    set at scene-init."""
    out = []
    for lemma, concept in lex.concepts.items():
        if not lex.types.is_subtype(concept.entity_type, role_spec.type):
            continue
        ok = True
        for slot, vals in role_spec.properties.items():
            slot_def = lex.slots.get(slot)
            if slot_def is None:
                continue
            if slot_def.varies:
                continue
            cvals = concept.properties.get(slot, [])
            if not (set(vals) & set(cvals)):
                ok = False
                break
        if ok:
            out.append(lemma)
    return out


def _concept_chain_potential(lex, concept_lemma, slot) -> int:
    """Heuristic weight: how much does this concept extend the chain
    if used as `slot`-producing target? Concepts with parts whose
    state lifts to the host (pordo+seruro→lock_state) get a bonus
    because the planner subgoals through host_lock_state derivations,
    yielding a 2-step chain. Plain artifacts return 1."""
    concept = lex.concepts.get(concept_lemma)
    if concept is None:
        return 1
    if slot == "openness":
        for part_spec in concept.parts:
            part = lex.concepts.get(part_spec.concept)
            if part and "lock_state" in part.properties:
                return 5  # pordo (has seruro) preferred over kuseno
    return 1


def regress_for_verb(verb_name, lex, rng):
    """Build a (trace, scene_id, drive) so the planner produces a chain
    ending in `verb_name`. Single-verb regression — multi-step chains
    happen via the planner subgoaling on unsatisfied role.properties.
    Returns None if no compatible concepts exist for some role."""
    from esperanto_lm.ontology.sampler import _add_entity_randomized
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

    role_eids: dict[str, str] = {}
    for role in action.roles:
        candidates = _concepts_matching_role(lex, role)
        if not candidates:
            return None
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

    for role_name, eid in role_eids.items():
        try:
            t.assert_relation("en", (eid, scene_id), lex)
        except (KeyError, ValueError):
            return None

    # Set theme's effect-slot to a non-target value so the verb has
    # work to do. Other role.properties are deliberately NOT preset —
    # the planner subgoals on them, growing the chain.
    eff = action.effects[0]
    target_eid = role_eids.get(eff.target_role)
    if target_eid is None:
        return None
    slot_def = lex.slots.get(eff.property)
    if slot_def is not None and slot_def.vocabulary:
        non_target = [v for v in slot_def.vocabulary if v != eff.value]
        if non_target:
            t.entities[target_eid].set_property(
                eff.property, rng.choice(non_target))

    # Instrument needs to be held by the agent for tool-using verbs.
    agent_eid = role_eids.get("agent")
    instrument_eid = role_eids.get("instrument")
    if agent_eid and instrument_eid:
        try:
            t.assert_relation("havi", (agent_eid, instrument_eid), lex)
        except (KeyError, ValueError):
            pass

    if agent_eid is None:
        return None
    drive = ("entity_slot", agent_eid, target_eid,
             eff.property, eff.value)
    return t, scene_id, drive


def sample_regression_scene(lex, rng):
    """Pick a verb (weighted) and regress a scene for it. Retries up to
    a few times if a verb's regression fails; returns None if every
    attempt fails."""
    pool = [(v, w) for v, w in REGRESSION_VERBS if v in lex.actions]
    if not pool:
        return None
    verbs = [v for v, _ in pool]
    weights = [w for _, w in pool]
    for _ in range(8):
        verb = rng.choices(verbs, weights=weights, k=1)[0]
        result = regress_for_verb(verb, lex, rng)
        if result is not None:
            return result
    return None


def run_coverage_regression(lex, rules, derivations, *, n_scenes=50, seed=0,
                             verbose_samples=8, save_jsonl=None):
    """Companion to run_coverage that uses goal-regression to build
    scenes. Same plan/fire/realize pipeline; different scene source."""
    rng = random.Random(seed)
    fired_records = []
    idle = 0
    failed = 0
    verb_counts: dict[str, int] = {}
    chain_counts: dict[str, int] = {}
    drive_kind_counts = {"sampled": {}, "fired": {}}
    jsonl_records = []

    for _ in range(n_scenes):
        sample = sample_regression_scene(lex, rng)
        if sample is None:
            failed += 1
            continue
        t, scene_id, drive = sample
        kind = drive[0]
        drive_kind_counts["sampled"][kind] = (
            drive_kind_counts["sampled"].get(kind, 0) + 1)
        setup = t.snapshot_relations()
        try:
            plan = plan_for_drive(drive, t, lex, rules, derivations)
        except Exception:
            failed += 1
            continue
        if not plan:
            idle += 1
            continue
        try:
            for verb, roles in plan:
                event = make_event(
                    verb, roles=roles,
                    property_changes=effect_changes(verb, roles, lex))
                t.events.append(event)
                run_dsl(t, rules, derivations, lex)
        except Exception:
            failed += 1
            continue
        try:
            prose = realize_trace(
                t, lex, setup_relations=setup,
                scene_location_id=scene_id)
        except Exception:
            prose = "<render failed>"
        chain = " → ".join(ev.action for ev in t.events)
        chain_counts[chain] = chain_counts.get(chain, 0) + 1
        for ev in t.events:
            verb_counts[ev.action] = verb_counts.get(ev.action, 0) + 1
        drive_kind_counts["fired"][kind] = (
            drive_kind_counts["fired"].get(kind, 0) + 1)
        fired_records.append((drive, chain, prose))
        jsonl_records.append({
            "scene": scene_id,
            "drive": {"kind": kind, "spec": list(drive[1:])},
            "drive_summary": _drive_summary(drive),
            "chain": chain,
            "n_events": len(t.events),
            "prose": prose,
        })

    print(f"\n========== regression coverage ({n_scenes} scenes) ==========")
    print(f"  fired: {len(fired_records)}   "
          f"idle: {idle}   failed: {failed}")
    print(f"\n  Verb counts (over all events fired):")
    for verb in sorted(verb_counts, key=lambda v: -verb_counts[v]):
        print(f"    {verb:<15} {verb_counts[verb]:>3}")
    print(f"\n  Plan-chain distribution:")
    for chain, n in sorted(chain_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"    {n:>3}×  {chain}")
    print(f"\n  Sample prose ({min(verbose_samples, len(fired_records))} "
          f"of {len(fired_records)}):")
    for drive, chain, prose in rng.sample(
            fired_records, min(verbose_samples, len(fired_records))):
        print(f"    [{chain}]  drive: {_drive_summary(drive)}")
        print(f"      prose: {prose}")
    if save_jsonl:
        import json
        from pathlib import Path
        Path(save_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(save_jsonl, "w") as f:
            for rec in jsonl_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n  Saved {len(jsonl_records)} records to {save_jsonl}")


def run_coverage(lex, rules, derivations, *, n_scenes=50, seed=0,
                 verbose_samples=8, save_jsonl=None):
    """Sample N scenes, sample one drive per scene, dispatch the drive
    to its planner entry, fire the resulting plan if any. Direct
    dispatch (not run_simulation) means we resolve ONLY the sampled
    drive — no opportunistic resolution of whatever else random init
    left unsatisfied.

    Filtering rule: keep scenes where the planner returned a non-empty
    plan AND firing it produced ≥1 trace event. Idle = no plan found
    for the sampled drive (can happen when scene lacks the props the
    drive's chain needs)."""
    rng = random.Random(seed)
    fired_records = []  # (drive, chain, prose)
    idle = 0
    failed = 0
    verb_counts: dict[str, int] = {}
    chain_counts: dict[str, int] = {}
    drive_kind_counts: dict[str, int] = {"sampled": {}, "fired": {}}
    drive_kind_counts["sampled"] = {}
    drive_kind_counts["fired"] = {}
    jsonl_records = []

    # Build writability caches once — they're a pure function of the
    # lexicon + rules + derivations, so reused across all scenes.
    property_writable = _build_property_writability(lex, rules, derivations)
    relation_writable = _build_relation_writability(lex, rules, derivations)

    for _ in range(n_scenes):
        try:
            t, scene_id, persons = sample_scene(lex, rng)
        except Exception:
            failed += 1
            continue
        drive = sample_drive(
            t, lex, rng, derivations=derivations, rules=rules,
            property_writable=property_writable,
            relation_writable=relation_writable)
        if drive is None:
            idle += 1
            continue
        # Goal-aware augmentation: place props the drive's chain
        # needs but blind sampling didn't put in scene.
        try:
            augment_scene_for_drive(t, drive, lex, rng, scene_id)
        except Exception:
            pass  # augmentation is best-effort; never let it block
        kind = drive[0]
        drive_kind_counts["sampled"][kind] = (
            drive_kind_counts["sampled"].get(kind, 0) + 1)
        setup = t.snapshot_relations()
        try:
            plan = plan_for_drive(drive, t, lex, rules, derivations)
        except Exception:
            failed += 1
            continue
        if not plan:
            idle += 1
            continue
        try:
            for verb, roles in plan:
                event = make_event(
                    verb, roles=roles,
                    property_changes=effect_changes(verb, roles, lex))
                t.events.append(event)
                run_dsl(t, rules, derivations, lex)
        except Exception:
            failed += 1
            continue
        try:
            prose = realize_trace(
                t, lex, setup_relations=setup,
                scene_location_id=scene_id)
        except Exception:
            prose = "<render failed>"
        chain = " → ".join(ev.action for ev in t.events)
        chain_counts[chain] = chain_counts.get(chain, 0) + 1
        for ev in t.events:
            verb_counts[ev.action] = verb_counts.get(ev.action, 0) + 1
        drive_kind_counts["fired"][kind] = (
            drive_kind_counts["fired"].get(kind, 0) + 1)
        fired_records.append((drive, chain, prose))
        jsonl_records.append({
            "scene": scene_id,
            "drive": {"kind": kind, "spec": list(drive[1:])},
            "drive_summary": _drive_summary(drive),
            "chain": chain,
            "n_events": len(t.events),
            "prose": prose,
        })

    print(f"\n========== coverage run ({n_scenes} scenes) ==========")
    print(f"  fired: {len(fired_records)}   "
          f"idle: {idle}   failed: {failed}")
    print(f"\n  Drive kind: sampled vs fired (success rate):")
    all_kinds = sorted(set(drive_kind_counts["sampled"]) |
                       set(drive_kind_counts["fired"]))
    for kind in all_kinds:
        s = drive_kind_counts["sampled"].get(kind, 0)
        f = drive_kind_counts["fired"].get(kind, 0)
        rate = f"{100*f/s:.0f}%" if s else "—"
        print(f"    {kind:15s} sampled={s:3d}  fired={f:3d}  ({rate})")
    print(f"\n  Verb counts (over all events fired):")
    for v, c in sorted(verb_counts.items(), key=lambda x: -x[1]):
        print(f"    {v:15s} {c}")
    print(f"\n  Plan-chain distribution:")
    for ch, c in sorted(chain_counts.items(), key=lambda x: -x[1]):
        print(f"    {c:3d}×  {ch}")
    print(f"\n  Sample prose ({verbose_samples} of {len(fired_records)}):")
    rng2 = random.Random(seed + 1)
    samples = rng2.sample(
        fired_records, min(verbose_samples, len(fired_records)))
    for drive, chain, prose in samples:
        print(f"    [{chain}]  drive: {_drive_summary(drive)}")
        print(f"      prose: {prose}")
    if save_jsonl:
        import json
        with open(save_jsonl, "w") as f:
            for rec in jsonl_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n  Saved {len(jsonl_records)} records to {save_jsonl}")


def scene_locked_door(lex):
    """Maria wants the door open. SLOT_PREFERENCES doesn't cover door
    openness (no entity has a "I want this door open" drive in the
    universal-preferences model), so we hand the goal in explicitly via
    the third return — the runner dispatches direct rather than via
    the agent-sim loop's self-/altruistic-drive scan."""
    t = Trace()
    t.add_entity("oficejo", lex, entity_id="oficejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("pordo", lex, entity_id="pordo")
    t.add_entity("ŝlosilo", lex, entity_id="ŝlosilo")
    t.assert_relation("en", ("maria", "oficejo"), lex)
    t.assert_relation("en", ("pordo", "oficejo"), lex)
    t.assert_relation("en", ("ŝlosilo", "oficejo"), lex)
    t.assert_relation("havi", ("maria", "ŝlosilo"), lex)
    t.entities["pordo"].set_property("openness", "fermita")
    t.entities["pordo"].set_property("lock_state", "ŝlosita")
    return t, "oficejo", ("explicit", "pordo", "openness", "malfermita", "maria")


def scene_bread_with_petro(lex):
    """Maria is hungry; Petro holds the bread; both in kuirejo. With
    EXTRA_PRECONDITIONS in effect (manĝi requires havi(agent,theme)),
    Maria's self-drive forces a preni→manĝi chain."""
    t = Trace()
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("pano", lex, entity_id="pano")
    t.assert_relation("en", ("maria", "kuirejo"), lex)
    t.assert_relation("en", ("petro", "kuirejo"), lex)
    t.assert_relation("en", ("pano", "kuirejo"), lex)
    t.assert_relation("havi", ("petro", "pano"), lex)
    t.entities["maria"].set_property("hunger", "malsata")
    t.entities["petro"].set_property("hunger", "sata")
    return t, "kuirejo", "maria.hunger=sata (self-drive)"


def scene_petro_in_other_room(lex):
    """Petro asleep in salono; Maria awake in kuirejo. With co-location
    on veki, Maria must iri to salono before vekiing Petro. Altruistic
    drive (Maria has no own displeased slot)."""
    t = Trace()
    t.add_entity("salono", lex, entity_id="salono")
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("persono", lex, entity_id="petro")
    t.assert_relation("en", ("maria", "kuirejo"), lex)
    t.assert_relation("en", ("petro", "salono"), lex)
    t.entities["petro"].set_property("sleep_state", "dormanta")
    t.entities["maria"].set_property("sleep_state", "vekita")
    return t, "salono", "petro.sleep_state=vekita (altruistic)"


def scene_dirty_door(lex):
    """A locked door. Maria has a key. She should unlock it.
    (Lock-state isn't in our prefs, so this won't fire as a drive —
    included for contrast: shows agent loop terminates idle.)"""
    t = Trace()
    t.add_entity("oficejo", lex, entity_id="oficejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("pordo", lex, entity_id="pordo")
    t.add_entity("ŝlosilo", lex, entity_id="ŝlosilo")
    t.assert_relation("en", ("maria", "oficejo"), lex)
    t.assert_relation("en", ("pordo", "oficejo"), lex)
    t.assert_relation("havi", ("maria", "ŝlosilo"), lex)
    return t, "oficejo"


def main():
    lex = load_lexicon()
    rules = list(DEFAULT_DSL_RULES)
    derivations = list(DEFAULT_DSL_DERIVATIONS)

    scenes = [
        ("hungry maria (1-step direct)", scene_hungry_maria),
        ("asleep petro", scene_asleep_petro),
        ("locked door (no relevant drive)", scene_dirty_door),
    ]

    for label, builder in scenes:
        print(f"\n=== {label} ===")
        t, scene_id = builder(lex)
        setup = t.snapshot_relations()

        for eid, ent in t.entities.items():
            if lex.types.is_subtype(ent.entity_type, "animate"):
                pref_state = {s: ent.properties.get(s, [None])[0]
                              for s in SLOT_PREFERENCES}
                print(f"  {eid}: {pref_state}")

        ticks = run_simulation(t, lex, rules, derivations, max_ticks=5)
        print(f"  -> ran {ticks} ticks, {len(t.events)} total events")
        prose = realize_trace(
            t, lex, setup_relations=setup, scene_location_id=scene_id)
        print(f"  prose: {prose}")

    # ---- general-subgoaling proofs ----
    print("\n\n========== general subgoaling proofs ==========")
    print("(Direct goal-passing; no self-preferences involved.)")
    print("Scope: planner subgoals on (1) entity-property preconditions,")
    print("(2) relation preconditions (havi(agent,theme) etc.), and")
    print("(3) co-location (same `en` container for two roles). The")
    print("relation/co-location preconditions live in EXTRA_PRECONDITIONS")
    print("until the action schema grows a `requires_relations` field.")
    print()

    # Proof 1: goal already met → empty plan
    print("=== proof 1: goal already met ===")
    t = Trace()
    t.add_entity("oficejo", lex, entity_id="oficejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("pordo", lex, entity_id="pordo")
    t.assert_relation("en", ("maria", "oficejo"), lex)
    t.assert_relation("en", ("pordo", "oficejo"), lex)
    print(f"  pordo.openness = {t.entities['pordo'].properties.get('openness')}")
    plan = plan_to_achieve(
        "pordo", "openness", "fermita", "maria", t, lex, rules,
        derivations=derivations)
    print(f"  goal: pordo.openness=fermita")
    print(f"  plan: {plan}  (already satisfied, no actions needed)")

    # Proof 2: 1-step plan to a non-self goal
    print("\n=== proof 2: 1-step plan (open the closed door) ===")
    t = Trace()
    t.add_entity("oficejo", lex, entity_id="oficejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("pordo", lex, entity_id="pordo")
    t.assert_relation("en", ("maria", "oficejo"), lex)
    t.assert_relation("en", ("pordo", "oficejo"), lex)
    # Door starts unlocked so the new malfermi precondition is already
    # satisfied (the locked-door chain has its own proof, see #6).
    t.entities["pordo"].set_property("lock_state", "malŝlosita")
    print(f"  pordo.openness = {t.entities['pordo'].properties.get('openness')}")
    plan = plan_to_achieve(
        "pordo", "openness", "malfermita", "maria", t, lex, rules,
        derivations=derivations)
    print(f"  goal: pordo.openness=malfermita")
    print(f"  plan: {plan}")

    # Proof 3: planner picks the right inverse verb based on goal.
    # Same scene as proof 2 but mirrored: door starts open, goal is
    # closed. Planner picks `fermi`, not `malfermi`, because fermi's
    # effect writes openness=fermita.
    print("\n=== proof 3: inverse goal, inverse verb (close the open door) ===")
    t = Trace()
    t.add_entity("oficejo", lex, entity_id="oficejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("pordo", lex, entity_id="pordo")
    t.assert_relation("en", ("maria", "oficejo"), lex)
    t.assert_relation("en", ("pordo", "oficejo"), lex)
    t.entities["pordo"].set_property("openness", "malfermita")
    print(f"  pordo.openness = {t.entities['pordo'].properties.get('openness')}")
    plan = plan_to_achieve(
        "pordo", "openness", "fermita", "maria", t, lex, rules,
        derivations=derivations)
    print(f"  goal: pordo.openness=fermita")
    print(f"  plan: {plan}")

    # Proof 4: agent ≠ target enforcement. Petro asks for door to
    # open (and Petro is animate, so he'd be a valid agent), but
    # plan_to_achieve was called with actor=petro AND target=petro for
    # an intransitive case demonstration. Here we use a transitive
    # case where actor and target differ to confirm the new check
    # doesn't break the legitimate non-self plan.
    print("\n=== proof 4: legitimate cross-entity plan (actor != target) ===")
    print("    (Petro plans to open a door; door is the target, Petro is")
    print("     the actor — not self-acting, so plan succeeds.)")
    t = Trace()
    t.add_entity("oficejo", lex, entity_id="oficejo")
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("pordo", lex, entity_id="pordo")
    t.assert_relation("en", ("petro", "oficejo"), lex)
    t.assert_relation("en", ("pordo", "oficejo"), lex)
    t.entities["pordo"].set_property("lock_state", "malŝlosita")
    plan = plan_to_achieve(
        "pordo", "openness", "malfermita", "petro", t, lex, rules,
        derivations=derivations)
    print(f"  plan: {plan}")

    # Proof 5: agent == target rejection for non-self-target verbs.
    # Petro plans to wake himself via `veki` (transitive theme=Petro,
    # agent=Petro). This was the bug: previously gave 5x "Petro vekis
    # Petron". Now plan_to_achieve refuses to bind agent=target when
    # target_role != "agent", so the plan fails (None) — correctly,
    # because Petro can't wake himself.
    print("\n=== proof 5: agent==target rejected for transitive verbs ===")
    print("    (Petro can't wake himself — plan_to_achieve returns None")
    print("     because veki's target is theme, and we now refuse to bind")
    print("     agent and theme to the same entity.)")
    t = Trace()
    t.add_entity("salono", lex, entity_id="salono")
    t.add_entity("persono", lex, entity_id="petro")
    t.assert_relation("en", ("petro", "salono"), lex)
    t.entities["petro"].set_property("sleep_state", "dormanta")
    plan = plan_to_achieve(
        "petro", "sleep_state", "vekita", "petro", t, lex, rules,
        derivations=derivations)
    print(f"  goal: petro.sleep_state=vekita (actor=petro, target=petro)")
    print(f"  plan: {plan}  (None ⇒ no self-wake; needs altruistic helper)")

    # Proof 6: locked door requires unlock-then-open chain. The new
    # malfermi.theme.lock_state=malŝlosita constraint forces the
    # planner to subgoal on lock_state, find malŝlosi, and chain.
    print("\n=== proof 6: locked door 2-step (malŝlosi → malfermi) ===")
    print("    (door is fermita+ŝlosita; goal is malfermita. malfermi now")
    print("     requires lock_state=malŝlosita, so planner inserts malŝlosi.)")
    t = Trace()
    t.add_entity("oficejo", lex, entity_id="oficejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("pordo", lex, entity_id="pordo")
    t.add_entity("ŝlosilo", lex, entity_id="ŝlosilo")
    t.assert_relation("en", ("maria", "oficejo"), lex)
    t.assert_relation("en", ("pordo", "oficejo"), lex)
    t.assert_relation("en", ("ŝlosilo", "oficejo"), lex)
    t.assert_relation("havi", ("maria", "ŝlosilo"), lex)
    t.entities["pordo"].set_property("openness", "fermita")
    t.entities["pordo"].set_property("lock_state", "ŝlosita")
    plan = plan_to_achieve(
        "pordo", "openness", "malfermita", "maria", t, lex, rules,
        derivations=derivations, max_depth=4)
    print(f"  goal: pordo.openness=malfermita")
    print(f"  plan: {plan}")

    # Proof 7: hungry Maria, Petro has the bread. EXTRA_PRECONDITIONS
    # makes manĝi require havi(agent, theme), so the planner subgoals
    # to ESTABLISH a relation: maria must have pano. preni transfers
    # ownership from current owner — which is Petro. Two-step plan.
    print("\n=== proof 7: relation precondition (preni → manĝi) ===")
    print("    (Maria is hungry, Petro holds the bread. manĝi now requires")
    print("     havi(agent,theme); planner subgoals to establish that")
    print("     relation via preni_transfers_ownership.)")
    t = Trace()
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("pano", lex, entity_id="pano")
    t.assert_relation("en", ("maria", "kuirejo"), lex)
    t.assert_relation("en", ("petro", "kuirejo"), lex)
    t.assert_relation("en", ("pano", "kuirejo"), lex)
    t.assert_relation("havi", ("petro", "pano"), lex)
    t.entities["maria"].set_property("hunger", "malsata")
    plan = plan_to_achieve(
        "maria", "hunger", "sata", "maria", t, lex, rules,
        derivations=derivations, max_depth=4)
    print(f"  goal: maria.hunger=sata")
    print(f"  plan: {plan}")

    # Proof 8: co-location. Petro asleep in salono, Maria in kuirejo.
    # veki now requires co_locate(agent, theme); planner subgoals via
    # plan_to_co_locate, which finds a rule that adds en(maria, salono)
    # — iri_moves_agent. Two-step plan: iri then veki.
    print("\n=== proof 8: co-location (iri → veki) ===")
    print("    (Petro asleep in salono, Maria in kuirejo. veki now requires")
    print("     same-location; planner inserts iri to bring Maria over.)")
    t = Trace()
    t.add_entity("salono", lex, entity_id="salono")
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("persono", lex, entity_id="petro")
    t.assert_relation("en", ("maria", "kuirejo"), lex)
    t.assert_relation("en", ("petro", "salono"), lex)
    t.entities["petro"].set_property("sleep_state", "dormanta")
    plan = plan_to_achieve(
        "petro", "sleep_state", "vekita", "maria", t, lex, rules,
        derivations=derivations, max_depth=4)
    print(f"  goal: petro.sleep_state=vekita (actor=maria)")
    print(f"  plan: {plan}")

    # ---- multi-step simulation scenes ----
    print("\n\n========== multi-step simulation scenes ==========")
    print()

    multi_scenes = [
        ("locked door (malŝlosi → malfermi)", scene_locked_door),
        ("Petro has bread (preni → manĝi)", scene_bread_with_petro),
        ("Petro asleep elsewhere (iri → veki)", scene_petro_in_other_room),
    ]
    for label, builder in multi_scenes:
        print(f"\n=== {label} ===")
        t, scene_id, drive = builder(lex)
        setup = t.snapshot_relations()
        print(f"  drive: {drive}")
        if isinstance(drive, tuple) and drive and drive[0] == "explicit":
            # Direct goal dispatch — no SLOT_PREFERENCES drive matches.
            _, target, slot, value, actor = drive
            plan = plan_to_achieve(
                target, slot, value, actor, t, lex, rules,
                derivations=derivations, max_depth=4)
            if plan is None:
                print("  -> no plan found")
            else:
                for verb, roles in plan:
                    event = make_event(
                        verb, roles=roles,
                        property_changes=effect_changes(verb, roles, lex))
                    t.events.append(event)
                    run_dsl(t, rules, derivations, lex)
                print(f"  -> {len(t.events)} events from explicit plan")
        else:
            ticks = run_simulation(t, lex, rules, derivations, max_ticks=6)
            print(f"  -> ran {ticks} ticks, {len(t.events)} total events")
        prose = realize_trace(
            t, lex, setup_relations=setup, scene_location_id=scene_id)
        print(f"  prose: {prose}")

    # ---- random scene + drive coverage ----
    run_coverage(lex, rules, derivations, n_scenes=200, seed=0,
                 save_jsonl="runs/agent_sim_coverage.jsonl")

    # ---- goal-regression coverage (companion run) ----
    run_coverage_regression(
        lex, rules, derivations, n_scenes=200, seed=0,
        save_jsonl="runs/agent_sim_regression.jsonl")


if __name__ == "__main__":
    main()
