"""Backward-chaining planner for the agent.

Given a drive (a tuple like `("self_slot", actor, "thirst", "satigita")`
or `("possession", actor, item)`) the planner returns a sequence of
actions whose execution would satisfy it. The planner is purely
declarative — every decision is driven by introspecting the rule
library and the action schemas; nothing is hardcoded per-verb.

Layout (single file because the helpers and the public planners are
tightly interconnected; cross-module circular imports were the main
risk of splitting):
  - displeased_* utilities (used by the loop / drive sampler too)
  - rule / action introspection helpers (`_rule_writes`, `_action_writes`,
    `_trigger_event_pattern`, etc.)
  - entity / relation lookup helpers (`_entity_property_values`,
    `_has_relation`, `_container_*`, `_bindings_ok_with_reflexive`)
  - simulation primitives (`_simulate_from_scratch`, `_SimulationBudget`,
    `_BudgetExceeded`)
  - precondition resolution (`_resolve_preconditions`,
    `_resolve_preconditions_in_order`, `_candidate_breaks_preserved`)
  - public planners (`plan_to_achieve`, `plan_to_establish_relation`,
    `plan_to_reach_count`, `plan_event_firing`, `plan_to_co_locate`,
    `plan_action`, `plan_with_subgoals`)
  - derivation walkers (`_plan_via_derivation`, `_plan_via_synthesis`,
    `_plan_property_via_derivation`, plus the pattern walks they need)

Module-level state:
  - `_SIM_CACHE` / `_DERIVED_CACHE`: per-plan simulation memoization
  - `_PLANNER_RNG`, `_PRESERVE_CONSTRAINTS`, `_RP_DEPTH`, `_SIM_BUDGET`:
    contextvars used for thread-safe per-call configuration
"""
from __future__ import annotations

import contextvars
import random

from ..causal import (
    EntityInstance, Event, RelationAssertion, Trace, effect_changes,
    make_event,
)
from ..dsl import compute_derived_state, run_dsl
from ..dsl.effects import (
    AddRelation, Change, ConsumeOne, CreateEntity, DestroyEntity, Effect,
    Emit, RemoveRelation, TransferN,
)
from ..dsl.implications import PropertyImplication, RelationImplication
from ..dsl.patterns import (
    AndPattern, BindPattern, EntityPattern, EventPattern, NotPattern,
    OrPattern, RelPattern, Var,
)
from ..dsl.rules import (
    DEFAULT_DSL_DERIVATIONS, DEFAULT_DSL_RULES, RUNTIME_DERIVATIONS,
)
from .preferences import SLOT_PREFERENCES, effective_preferences


def displeased_slots(entity, trace=None, derived=None) -> list[tuple[str, str]]:
    """Return [(slot, preferred_value)] for slots where the entity's
    CURRENT values do not include the preferred. Membership semantics
    so multi-valued slots work.

    Reads the *effective* preferences (trace-context-aware), so e.g.
    a sleep_state of vekita at night counts as displeased relative
    to the night-time preference of dormanta."""
    out = []
    prefs = effective_preferences(trace)
    for slot, pref in prefs.items():
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


# Cache of (slot, value) pairs reachable via SOME action effect, rule
# effect (Change/Emit.property_changes), or derivation implication.
# Computed once per (rules, derivations, lex) triple by id-keying;
# the planner is the only consumer and treats this as immutable.
#
# A pair NOT in this set is intrinsic — no path achieves it. Subgoals
# asking for an absent (slot, value) are unsatisfiable, so the
# planner short-circuits them before recursing into a guaranteed-
# failing plan_to_achieve. This catches the "verŝi a location"
# class of dead-end paths, where a verb's role-property check would
# subgoal e.g. state_of_matter=likva on a balcony — an intrinsic
# property no rule writes.
_WRITABLE_CACHE: dict = {}


def _writable_slot_values(rules, actions, derivations) -> set:
    """Return a set of (slot, value) pairs that could possibly be
    written by some rule/action/derivation. Cached by id of the
    inputs — they're treated as immutable during a plan_for_drive
    call. Includes only literal (non-Var) values; varies-true slots
    that get randomized at instance time are picked up via their
    declared vocabulary in the slot-spec, not here."""
    key = (id(rules), id(actions), id(derivations))
    cached = _WRITABLE_CACHE.get(key)
    if cached is not None:
        return cached
    out: set = set()
    for action in actions.values():
        for eff in action.effects:
            if (isinstance(eff.property, str)
                    and isinstance(eff.value, str)):
                out.add((eff.property, eff.value))
    for rule in rules:
        effects = (rule.then if isinstance(rule.then, (list, tuple))
                   else [rule.then])
        for eff in effects:
            if isinstance(eff, Emit):
                for (_ent, slot), val in eff.property_changes.items():
                    if isinstance(slot, str) and isinstance(val, str):
                        out.add((slot, val))
            elif isinstance(eff, Change):
                if (isinstance(eff.slot, str)
                        and isinstance(eff.value, str)):
                    out.add((eff.slot, eff.value))
    for d in (derivations or []):
        for imp in d.implies:
            if isinstance(imp, PropertyImplication):
                if (isinstance(imp.slot, str)
                        and isinstance(imp.value, str)):
                    out.add((imp.slot, imp.value))
    _WRITABLE_CACHE[key] = out
    return out


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
    satisfies a role spec requiring locomotion=[walk].

    When no in-trace entity matches AND the `_ENTITY_RESOLVER`
    ContextVar holds a callable, the planner asks the resolver to
    spawn one. Resolver returns a freshly-added eid or None. This is
    the spawn-on-demand hook the seeder uses to materialize entities
    the planner discovers it needs."""
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
    resolver = _ENTITY_RESOLVER.get()
    if resolver is not None:
        return resolver(role_spec, trace, lex, exclude)
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


def _relation_args_admissible(relation, concrete, trace, lex) -> bool:
    """True if `relation(*concrete)` would satisfy the relation
    schema's arg_types and arg_excludes — i.e., asserting it would
    not be rejected by `Trace.assert_relation` at runtime. Used to
    pre-skip derivation candidates whose subgoaled patterns would be
    schema-impossible: prevents the planner from backchaining
    havi_implies_samloke into havi(actor, location), which is rejected
    upstream and would only contribute noise to the failure reason."""
    rel_def = lex.relations.get(relation)
    if rel_def is None or len(concrete) != rel_def.arity:
        return True
    for i, arg_eid in enumerate(concrete):
        ent = trace.entities.get(arg_eid)
        if ent is None:
            continue
        if not lex.types.is_subtype(ent.entity_type, rel_def.arg_types[i]):
            return False
        if i < len(rel_def.arg_excludes):
            for forbidden in rel_def.arg_excludes[i]:
                if lex.types.is_subtype(ent.entity_type, forbidden):
                    return False
    return True


# ---------- havas_parton is static: subgoaling it is futile -----------
#
# `havas_parton` is materialized at scene-build time from each entity's
# concept-level `parts` declaration and never changes thereafter — no
# rule or verb adds it at runtime, and no derivation implies it. So
# any planner subgoal of `havas_parton(D, S)` where the relation
# doesn't already hold is fundamentally unsatisfiable: there's no
# action sequence that establishes it.
#
# Derivations like `host_lock_state_unlocked_from_seruro` reference
# `havas_parton(D, S)` in their `given` clause to LOOK UP a host's
# part (not to establish it). When the planner backchains through
# such derivations and the bound `(D, S)` pair doesn't have an
# existing parts-relationship, it should immediately give up on that
# derivation candidate and try another, instead of recursing into a
# `plan_to_establish_relation("havas_parton", ...)` that's guaranteed
# to fail.
#
# Without this short-circuit, the planner wasted budget on impossible
# parts-subgoals and (with scoped failure-recording) surfaced their
# deepest leaves as the reported failure reason, drowning out actual
# blockers like closed doors and missing keys.


# ---------- post-subplan rebind: handle TransferN split etc. -----------
#
# When a sub-plan establishes `havi(actor, X)`, the engine's `TransferN`
# may split a stacked source: the original entity stays with the prior
# owner (count decremented), and a NEW entity of the requested quantity
# is created and havi'd to the actor. After this, the original goal
# `havi(actor, X)` is *not* satisfied — the actor instead has a fresh
# entity of the same concept.
#
# Without intervention the precondition retry loop sees the literal
# goal still unmet, tries to re-plan it, and fails (the candidate that
# previously worked now violates the preserve constraint that was
# pushed when the sub-plan committed). This blocks every drive that
# acquires from a stack > 1: fruits-on-trees, NPC-owned multi-stacks,
# etc. — empirically the largest single failure cluster in the
# regression sampler.
#
# Fix: after `_refresh()` brings the simulated post-subplan state into
# `cur_trace`, look for a new entity that satisfies the goal "morally"
# (same content for fakto, same concept for havi-target) and rebind
# the parent action's role dict to use the new entity. The rest of the
# precondition loop and the eventually-returned plan step both pick up
# the rewritten role.
#
# Equivalence is relation-keyed: the right "find replacement" check
# differs by relation (havi: same concept_lemma + actor havi's; konas:
# same fakto content + knower konas; etc.). Resolvers are registered
# in `_REBIND_RESOLVERS` so future cases drop in without touching the
# pc-loop site.

def _havi_split_replacement(role_eids, pre_event_count, cur_trace, lex):
    """For a havi(actor, X) goal where X may have been split: find a
    new entity created at-or-after `pre_event_count`, sharing concept
    with X, that the actor now `havi`s. Returns the new eid or None.

    None covers two cases the caller already handles correctly:
      - no split happened (qty >= source.count → wholesale transfer,
        original X transferred, no rebind needed),
      - the goal genuinely still can't be satisfied (no new entity +
        original havi false → caller's None-return path)."""
    if len(role_eids) != 2:
        return None
    actor_eid, original_eid = role_eids
    original_ent = cur_trace.entities.get(original_eid)
    if original_ent is None:
        return None
    target_concept = original_ent.concept_lemma
    havi_set = {
        tuple(r.args) for r in cur_trace.relations
        if r.relation == "havi"
    }
    for eid, ent in cur_trace.entities.items():
        if ent.concept_lemma != target_concept:
            continue
        if ent.created_at_event is None:
            continue
        if ent.created_at_event < pre_event_count:
            continue
        if (actor_eid, eid) in havi_set:
            return eid
    return None


_REBIND_RESOLVERS: dict = {
    "havi": _havi_split_replacement,
    # Future: "konas" rebind once knowledge drives show the same
    # fakto-recreation symptom in the sampler. Equivalence check there
    # is content-based: same pri_relacio + subjekto + objekto.
}


def _find_post_subplan_replacement(rel_name, role_eids, pre_event_count,
                                    cur_trace, lex):
    """Dispatch to a relation-specific resolver. Returns a (role_index,
    new_eid) pair indicating which arg of `role_eids` should be replaced,
    or None if no rebind is appropriate.

    Currently every resolver replaces the second arg (the theme), since
    that's the role TransferN can mutate — agent stays put. If future
    relations need first-arg rebinds, the resolver can return its own
    index."""
    resolver = _REBIND_RESOLVERS.get(rel_name)
    if resolver is None:
        return None
    new_eid = resolver(role_eids, pre_event_count, cur_trace, lex)
    if new_eid is None or new_eid == role_eids[1]:
        return None
    return (1, new_eid)


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


def _bindings_ok_with_reflexive(roles: dict, action) -> bool:
    """Validate role-binding uniqueness with the reflexive exception.

    Default rule: every role binds to a distinct entity. Verbs marked
    `reflexive_ok` (sekigi, future lavi/vesti) permit agent==theme,
    since "Maria sekigis sin" is well-formed and a wet actor needs
    the option to dry themselves with a tuko. Other duplicates
    (agent==instrument, theme==instrument) remain rejected — those
    aren't semantically natural for any verb in the lexicon."""
    values = list(roles.values())
    if len(set(values)) == len(values):
        return True
    if not getattr(action, "reflexive_ok", False):
        return False
    # The only allowed duplicate is agent==theme for a reflexive_ok
    # verb. Confirm no OTHER pair coincides.
    seen: dict[str, str] = {}
    agent = roles.get("agent")
    theme = roles.get("theme")
    for role_name, eid in roles.items():
        prior = seen.get(eid)
        if prior is None:
            seen[eid] = role_name
            continue
        # A duplicate exists. Allow only if the pair is (agent, theme).
        pair = {prior, role_name}
        if pair != {"agent", "theme"} or eid != agent or eid != theme:
            return False
    return True


def _container_of(entity_id, trace) -> str | None:
    """Return the `en`-container of entity_id, or None if it's not in
    any. Considers `en` only — `sur` is for surface placement which
    isn't a co-location anchor for actor co-presence."""
    for r in trace.relations:
        if r.relation == "en" and r.args[0] == entity_id:
            return r.args[1]
    return None


def _container_chain_of(entity_id, trace) -> list[str]:
    """Walk transitively up `en`/`sur` containment edges from entity_id,
    returning every container above it. Used by the relation-derivation
    planner to enumerate candidate "meeting places" for two entities
    that may be deeply nested — `_container_of` only sees the immediate
    container, so for `lakto en glaso, glaso en laborejo` it returns
    `[glaso]` and the planner never tries `laborejo` as the venue.

    `sur` is included because surfaces transit samloke under our chain
    derivations (botelo sur tablo en kuirejo → samloke(botelo, kuirejo)).
    Cycle-safe via a visited set — physically impossible but cheap."""
    out: list[str] = []
    visited = {entity_id}
    cur = entity_id
    while True:
        nxt: str | None = None
        for r in trace.relations:
            if r.relation in ("en", "sur") and r.args[0] == cur:
                nxt = r.args[1]
                break
        if nxt is None or nxt in visited:
            return out
        visited.add(nxt)
        out.append(nxt)
        cur = nxt


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
      ("given", given_pat, arg_name) — Var bound by a `given` rel
                                       pattern's named arg (e.g. legi
                                       extracts a fakto via priskribas)
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
        # Vars bound by `given` rel patterns (not also event-role-bound).
        # Lets rules like legi_extracts_fakto — where the konas fakto
        # is sourced from a priskribas given clause rather than an
        # event role — surface as konas-adders to the planner.
        var_to_given: dict[int, tuple] = {}
        for g_pat in (rule.given or ()):
            if not isinstance(g_pat, RelPattern):
                continue
            for arg_name, arg_pat in g_pat.arg_patterns.items():
                v = _bind_var_in_pattern(arg_pat)
                if v is None:
                    continue
                if id(v) in var_to_role or id(v) in var_to_creator:
                    continue
                var_to_given.setdefault(id(v), (g_pat, arg_name))
        for eff in effects:
            # TransferN(source=S, target=T) is equivalent to
            # AddRelation("havi", T, S) for planning purposes — the
            # engine resolves full-vs-partial transfer at firing time.
            if isinstance(eff, TransferN) and relation == "havi":
                args: tuple = (eff.target, eff.source)
            elif isinstance(eff, AddRelation) and eff.relation == relation:
                args = eff.args
            else:
                continue
            arg_sources = []
            for arg in args:
                if not isinstance(arg, Var):
                    arg_sources.append(None)
                    break
                if id(arg) in var_to_role:
                    arg_sources.append(("role", var_to_role[id(arg)]))
                elif id(arg) in var_to_creator:
                    arg_sources.append(("created", var_to_creator[id(arg)]))
                elif id(arg) in var_to_given:
                    g_pat, arg_name = var_to_given[id(arg)]
                    arg_sources.append(("given", g_pat, arg_name))
                else:
                    arg_sources.append(None)
            if all(s is not None for s in arg_sources):
                out.append((rule, event_pat, tuple(arg_sources)))
    return out


# Per-plan-for-drive cache for simulation results. _simulate_from_scratch
# is called dozens of times per scene (refresh after each precondition
# subgoal + synthesis exploration), often with the same (base, plan)
# inputs. Caching here cuts wallclock dramatically. Cleared at
# plan_for_drive entry so cross-call staleness is impossible. Callers
# treat results as read-only — they query state, never mutate.
_SIM_CACHE: dict[tuple, tuple] = {}

# Sibling cache for `compute_derived_state` calls inside the planner.
# Same lifetime as _SIM_CACHE. The trace is read-only during planning,
# so id-based keying is safe. Hit rate is high because the planner
# repeatedly asks for derived state on the same trace fork while
# resolving multiple preconditions.
# Content-addressable derived-state cache. Keyed by a digest of trace
# relations + event ids + entity-key set. Two forks with identical
# state share the cached derived layer — important at large entity
# counts where samloke alone is O(N²) per derive pass and the planner
# spawns hundreds of forks during a single plan_for_drive call.
_DERIVED_CACHE: dict[tuple, "DerivedState"] = {}


def _trace_state_signature(trace):
    """Content digest of everything that affects compute_derived_state:
      - asserted relations (sorted to canonicalize)
      - event ids in order (order matters for property_at; ids encode
        action+roles+property_changes per the engine's hash invariant)
      - entity-key set (events can create entities; new entities are
        only in the dict of forks that ran the creating event)"""
    rels = tuple(sorted((r.relation, r.args) for r in trace.relations))
    events = tuple(e.id for e in trace.events)
    entities = frozenset(trace.entities.keys())
    return (rels, events, entities)


def _cached_compute_derived_state(trace, derivations, lex):
    """Memoizing wrapper around compute_derived_state for planner use.
    Keyed by `_trace_state_signature` so forks with identical state
    share results — improves hit rate dramatically at large scenes.
    Cleared per plan_for_drive entry; signature build is O(N_relations
    + N_events), much cheaper than re-running the engine."""
    from ..dsl.engine import compute_derived_state
    key = _trace_state_signature(trace)
    cached = _DERIVED_CACHE.get(key)
    if cached is not None:
        return cached
    result = compute_derived_state(trace, derivations or [], lex)
    _DERIVED_CACHE[key] = result
    return result


_QUANTIFIABLE_TRANSFER_CACHE: dict[int, frozenset[str]] = {}


def _quantifiable_transfer_verbs(rules) -> frozenset[str]:
    """Verbs whose rule fires `transfer_n` — the count-drive planner
    stamps explicit quantity onto these so partial-stack transfers
    work. Cached by `id(rules)` since the rule set is stable for the
    lifetime of a planning session."""
    key = id(rules)
    cached = _QUANTIFIABLE_TRANSFER_CACHE.get(key)
    if cached is not None:
        return cached
    from ..dsl.introspect import transfer_verbs
    cached = transfer_verbs(rules)
    _QUANTIFIABLE_TRANSFER_CACHE[key] = cached
    return cached


def _unpack_step(step):
    """Plan items are (verb, roles) by default; (verb, roles, quantity)
    when the planner has set an explicit count for partial-stack
    transfer or count-drive consumption. Returns (verb, roles, qty)
    with qty=1 for the 2-tuple form."""
    if len(step) == 3:
        verb, roles, qty = step
        return verb, roles, qty
    verb, roles = step
    return verb, roles, 1


def _step_to_event(step, lex):
    """Materialize a plan step as an Event, threading quantity onto
    the event when the step carries one."""
    verb, roles, qty = _unpack_step(step)
    return make_event(
        verb, roles=roles,
        property_changes=effect_changes(verb, roles, lex),
        quantity=qty)


def _plan_cache_key(plan):
    """Hashable key for a plan: tuple of (verb, sorted-role-pairs, qty)."""
    return tuple(
        (verb, tuple(sorted(roles.items())), qty)
        for verb, roles, qty in (_unpack_step(s) for s in plan)
    )


_SLOT_PRODUCERS_CACHE: dict[int, dict[tuple[str, str], bool]] = {}


def _chain_richness_weight(candidate, lex) -> float:
    """Score a relation-adder candidate by how many subgoals its
    selection would create — proxy for chain length. Counts
    action.preconditions + total slots across all role.properties,
    then doubles when the action has an `instrument` role.

    The instrument bonus is the cheap, generic version of "prefer
    verbs that recruit tools when one is available" — veturi over
    iri for samloke goals when a vehicle's reachable, lavi-with-
    lavilo over lavi-bare-handed when a lavilo's around. The planner
    already vetted the candidate (instrument-bearing verbs only pass
    the candidate filter when a satisfiable instrument exists), so
    doubling here just shifts the weighted-shuffle's probability mass
    toward the richer plan without adding any per-verb flag."""
    _rule, event_pat, _arg_sources = candidate
    action = lex.actions.get(event_pat.action)
    if action is None:
        return 1.0
    score = 1 + len(action.preconditions)
    for role in action.roles:
        score += len(role.properties)
    if any(r.name == "instrument" for r in action.roles):
        score *= 3
    return float(score)


def _count_satisfied_preconds(candidate, target_args, actor_id,
                              trace, derived, lex) -> int:
    """Count how many of the candidate verb's preconditions ALREADY
    hold given the partial role bindings the planner can determine
    upfront (agent=actor + roles bound from arg_sources). Uncertain
    preconditions (whose roles aren't yet bound) don't count.

    Used as a sort key in `plan_to_establish_relation` so candidates
    with more pre-satisfied preconditions are tried first — the planner
    needs fewer subgoals on those, so they're cheaper to verify and
    typically produce shorter plans. Pure ordering hint; semantics
    unchanged. Profiled fix for hard knowledge-drive scenes where the
    planner stumbled through 6,000+ simulations to find a 6-step plan
    that lived in the candidate list all along."""
    _rule, event_pat, arg_sources = candidate
    action = lex.actions.get(event_pat.action)
    if action is None:
        return 0
    # Build partial role bindings: arg_sources first, then agent=actor
    # if action takes an agent and we haven't bound it.
    partial: dict[str, str] = {}
    for src, target_eid in zip(arg_sources, target_args):
        if src[0] == "role":
            partial[src[1]] = target_eid
    action_role_names = {r.name for r in action.roles}
    if "agent" in action_role_names and "agent" not in partial:
        partial["agent"] = actor_id
    satisfied = 0
    for pc in action.preconditions:
        kind = getattr(pc, "kind", None)
        if kind == "relation":
            if not all(role in partial for role in pc.roles):
                continue
            args = tuple(partial[r] for r in pc.roles)
            if _has_relation(pc.rel, args, trace, derived, lex):
                satisfied += 1
        elif kind == "if_property":
            if pc.role not in partial:
                continue
            ent = trace.entities.get(partial[pc.role])
            if ent is None:
                continue
            # Vacuous-true if the gate condition (if_property=if_value)
            # doesn't hold; the precondition is "satisfied" in that
            # case because there's nothing to subgoal.
            gate_vals = ent.properties.get(pc.if_property, [])
            if isinstance(gate_vals, list):
                gate_holds = pc.if_value in gate_vals
            else:
                gate_holds = pc.if_value == gate_vals
            if not gate_holds:
                satisfied += 1
                continue
            then_vals = ent.properties.get(pc.then_property, [])
            if isinstance(then_vals, list):
                if pc.then_value in then_vals:
                    satisfied += 1
            elif pc.then_value == then_vals:
                satisfied += 1
        elif kind == "match":
            if pc.role_a not in partial or pc.role_b not in partial:
                continue
            ent_a = trace.entities.get(partial[pc.role_a])
            ent_b = trace.entities.get(partial[pc.role_b])
            if ent_a is None or ent_b is None:
                continue
            vals_a = set(ent_a.properties.get(pc.slot_a, []))
            vals_b = set(ent_b.properties.get(pc.slot_b, []))
            if vals_a & vals_b:
                satisfied += 1
    return satisfied


def _weighted_shuffle(items, weights, rng):
    """Return items in a weighted-random order: higher weight → more
    likely to appear earlier. Uses iterative weighted sampling without
    replacement. Falls back to uniform shuffle when all weights are
    equal."""
    if len(items) <= 1:
        return list(items)
    out = []
    pool = list(zip(items, weights))
    while pool:
        ws = [w for _, w in pool]
        idx = rng.choices(range(len(pool)), weights=ws, k=1)[0]
        out.append(pool.pop(idx)[0])
    return out


def _slot_value_producible(slot: str, value: str, lex) -> bool:
    """True if (slot, value) can be produced — either by a verb's
    direct effect OR by a derivation's implication. Cached per-lexicon
    so repeated planner candidate filtering doesn't re-scan.

    Including derivations matters: lit_state=luma is never written by
    a verb directly; it's derived from indoor_lit_by_active_lamp etc.
    Without derivation reachability the filter drops every candidate
    that could only satisfy the slot via derivation, killing the
    `ŝalti → vidi` lighting chain."""
    cache = _SLOT_PRODUCERS_CACHE.setdefault(id(lex), {})
    key = (slot, value)
    cached = cache.get(key)
    if cached is not None:
        return cached
    found = False
    for action in lex.actions.values():
        for eff in action.effects:
            if eff.property == slot and eff.value == value:
                found = True
                break
        if found:
            break
    if not found:
        from ..dsl.implications import (
            PropertyImplication,
        )
        from ..dsl.rules import (
            DEFAULT_DSL_DERIVATIONS,
        )
        for d in DEFAULT_DSL_DERIVATIONS:
            for imp in d.implies:
                if (isinstance(imp, PropertyImplication)
                        and imp.slot == slot and imp.value == value):
                    found = True
                    break
            if found:
                break
    cache[key] = found
    return found


def _filter_candidates_by_slots(candidates, fv_id, slots_to_subgoal,
                                  trace, lex):
    """Drop candidates whose slot constraints can never be satisfied.
    Two filters per slot:

      (a) Type-applicability: the slot must apply to the candidate's
          entity_type (per `slot.applies_to`). lit_state.applies_to=
          ["location"] means non-locations can never have it, even
          though it's derivation-producible. Without this, every
          entity becomes a candidate for L in agent_illuminated and
          the planner explodes recursively.
      (b) Reachability: if the value isn't currently set, it must be
          producible by some verb's effect OR by a derivation. Static
          markers like `lights_when_on=yes` are only set on lampo;
          no-one can change them, so candidates without it are
          dropped."""
    out = []
    for cand in candidates:
        ent = trace.entities.get(cand)
        if ent is None:
            continue
        ok = True
        for entry in slots_to_subgoal:
            v_local, slot, value, sign = entry
            if id(v_local) != fv_id:
                continue
            slot_def = lex.slots.get(slot)
            if slot_def is not None:
                applies = any(
                    lex.types.is_subtype(ent.entity_type, t)
                    for t in slot_def.applies_to)
                if not applies:
                    # Slot doesn't apply to this entity type. For
                    # positive constraints, the candidate is dead.
                    # For negative ("not fermita"), candidates with
                    # no openness slot at all vacuously satisfy the
                    # negation — keep them.
                    if sign == "pos":
                        ok = False
                        break
                    continue
            actual = ent.properties.get(slot, [])
            if sign == "pos":
                if value in actual:
                    continue
                if not _slot_value_producible(slot, value, lex):
                    ok = False
                    break
            else:  # sign == "neg"
                if value not in actual:
                    continue
                # Need to flip away from `value`. Viable iff some
                # alternate value in slot.vocabulary is producible.
                alts = [v for v in (slot_def.vocabulary if slot_def else [])
                        if v != value]
                if not any(_slot_value_producible(slot, alt, lex)
                           for alt in alts):
                    ok = False
                    break
        if ok:
            out.append(cand)
    return out


def _plan_slot_subgoal(target_eid, slot, value, sign, actual,
                        actor_id, trace, lex, rules, derived,
                        derivations, max_depth, depth, seen):
    """Subgoal a slot constraint with sign. Returns a list of plan
    steps (possibly empty if already satisfied) or None if unreachable.

    sign="pos": want slot=value. Already satisfied iff value in actual.
    Otherwise plan_to_achieve(target_eid, slot, value).

    sign="neg": want slot != value. Already satisfied iff value not
    in actual. Otherwise iterate slot.vocabulary, try plan_to_achieve
    for each non-forbidden alternate; first one that resolves wins.
    Used by entity NotPattern subgoaling — `~entity(openness="fermita")`
    on a closed valizo subgoals plan_to_achieve(valizo, openness,
    malfermita), which finds malfermi."""
    if sign == "pos":
        if value in actual:
            return []
        return plan_to_achieve(
            target_eid, slot, value, actor_id, trace, lex, rules,
            derived=derived, derivations=derivations,
            max_depth=max_depth, _depth=depth, _seen=seen)
    # sign == "neg"
    if value not in actual:
        return []
    slot_def = lex.slots.get(slot)
    if slot_def is None or not slot_def.vocabulary:
        return None
    for alt in slot_def.vocabulary:
        if alt == value:
            continue
        sub = plan_to_achieve(
            target_eid, slot, alt, actor_id, trace, lex, rules,
            derived=derived, derivations=derivations,
            max_depth=max_depth, _depth=depth, _seen=seen)
        if sub is not None:
            return sub
    return None


def _simulate_from_scratch(base_trace, plan, lex, rules, derivations):
    """Deepcopy `base_trace`, append events for each (verb, roles) in
    `plan`, run the engine once. Returns the forked trace + the
    derived state on top of it. Used by the planner to compute the
    hypothetical state the next precondition check should see, after
    prior subgoals have run.

    The committed-plan-list pattern (vs. mutating in place + run_dsl
    incrementally) is necessary because run_dsl recreates its
    fired-bindings memo each call and would re-fire causal rules on
    prior events — iri_moves_agent in particular re-fires its
    remove_relation against the post-iri state and clobbers the only
    `en` it just added. Running once on the full event list from a
    fresh base avoids that.

    Memoized per-plan-for-drive — see `_SIM_CACHE`. Counts against
    the plan_for_drive simulation budget; raises `_BudgetExceeded`
    when exhausted (caught at plan_for_drive entry, returns None plan).
    Cache hits do NOT consume budget — they're free reads."""
    import copy
    from ..causal import make_event, effect_changes
    from ..dsl.engine import (
        compute_derived_state, run_dsl,
    )
    from ..causal import Event
    derivs = derivations or []
    cache_key = (id(base_trace), _plan_cache_key(plan))
    cached = _SIM_CACHE.get(cache_key)
    if cached is not None:
        return cached
    budget = _SIM_BUDGET.get()
    if budget is not None:
        budget.consume()
    if not plan:
        # Empty-plan short-circuit: skip the deepcopy. Compute
        # derived once for the base; callers only read.
        derived_state = _cached_compute_derived_state(
            base_trace, derivs, lex)
        result = (base_trace, derived_state)
        _SIM_CACHE[cache_key] = result
        return result
    t = base_trace.fork()
    for step in plan:
        ev = _step_to_event(step, lex)
        t.events.append(ev)
        t._event_ids.add(ev.id)
    run_dsl(t, rules, derivs, lex)
    # Engine quirk: entities with `created_at_event=k` are visible
    # only at `t>k` (see `Trace.property_at` liveness check).
    # Without a follow-up event, derivations can't see freshly
    # created entities like a fakto from `vidi`. Append a phantom
    # event so `len(trace.events)` is one past every created
    # entity's `created_at_event`, making them visible to the
    # post-simulation derivation pass we use for what-if checks.
    phantom = Event(
        id=f"_settle_phantom_{len(t.events)}",
        action="_settle_phantom",
        roles={})
    t.events.append(phantom)
    t._event_ids.add(phantom.id)
    derived_state = _cached_compute_derived_state(t, derivs, lex)
    result = (t, derived_state)
    _SIM_CACHE[cache_key] = result
    return result


def _resolve_preconditions(action, event_pat, roles, actor_id,
                           trace, lex, rules, derived,
                           max_depth, depth, seen, *, derivations=None):
    """Try the default precondition order first; on failure, try
    permutations. Wraps `_resolve_preconditions_in_order`.

    The declared order works for most actions (preconditions are
    typically chosen so the easiest gates come first). When it
    doesn't — e.g. malŝlosi's `[samloke, havi]` commits an iri that
    moves the agent away from the key needed for havi — a different
    order solves it. With N PCs the search is N! orderings; for our
    actions N ≤ 3, so worst case is 6 attempts. Early-exits on the
    first ordering that succeeds, so the typical cost is one attempt.
    """
    import itertools
    rp_depth = _RP_DEPTH.get() + 1
    rp_token = _RP_DEPTH.set(rp_depth)
    try:
        pcs = list(action.preconditions)
        if len(pcs) <= 1:
            return _resolve_preconditions_in_order(
                action, event_pat, roles, actor_id, trace, lex, rules,
                derived, max_depth, depth, seen,
                pcs_order=pcs, derivations=derivations)
        # Try the declared order first.
        result = _resolve_preconditions_in_order(
            action, event_pat, roles, actor_id, trace, lex, rules,
            derived, max_depth, depth, seen,
            pcs_order=pcs, derivations=derivations)
        if result[1]:
            return result
        # Permute only when shallow enough that the cost stays
        # bounded. At deeper levels, declared order has to suffice.
        # The locked-door cascade hits permutation around rp_depth=3,
        # so the cap accommodates it.
        if rp_depth > _RP_PERMUTE_MAX_DEPTH:
            return result
        # Try other permutations. Take the first working one.
        declared_tuple = tuple(pcs)
        for perm in itertools.permutations(pcs):
            if perm == declared_tuple:
                continue
            result = _resolve_preconditions_in_order(
                action, event_pat, roles, actor_id, trace, lex, rules,
                derived, max_depth, depth, seen,
                pcs_order=list(perm), derivations=derivations)
            if result[1]:
                return result
        return [], False
    finally:
        _RP_DEPTH.reset(rp_token)


def _resolve_preconditions_in_order(action, event_pat, roles, actor_id,
                                     trace, lex, rules, derived,
                                     max_depth, depth, seen, *,
                                     pcs_order, derivations=None):
    """For each role binding, recursively satisfy:
      - verb-level role property constraints,
      - rule-level role pattern entity constraints (if rule given),
      - action.preconditions (cross-role relations from the schema)
        in the order specified by `pcs_order`.
    Returns (sub_plans, ok). Sub-plans are concatenated in the order
    they're discovered (which is the order they need to fire).

    Forward simulation: each subgoal's plan is added to a `committed`
    list; before checking the next precondition, we resimulate from
    the original trace + committed plan to get a fresh state. Loops
    until all preconditions hold simultaneously (or max iters). This
    catches the case where one subgoal invalidates a previously-
    satisfied precondition — e.g. fetching a key in another room
    breaks samloke(agent, theme) for malŝlosi, and a return-iri
    needs to be inserted."""
    from ..schemas import (
        IfPropertyPrecondition, MatchPrecondition, RelationPrecondition,
    )
    committed: list = []
    cur_trace, cur_derived = trace, derived
    # Track constraints we've certified as satisfied; sibling subgoals
    # (and recursive descendants) must not invalidate them. See the
    # _PRESERVE_CONSTRAINTS docstring for context.
    accumulated_preserve = list(_PRESERVE_CONSTRAINTS.get())
    _preserve_token = _PRESERVE_CONSTRAINTS.set(tuple(accumulated_preserve))

    def _push_preserve(entry):
        nonlocal _preserve_token
        if entry in accumulated_preserve:
            return
        accumulated_preserve.append(entry)
        _PRESERVE_CONSTRAINTS.reset(_preserve_token)
        _preserve_token = _PRESERVE_CONSTRAINTS.set(
            tuple(accumulated_preserve))

    def _refresh():
        nonlocal cur_trace, cur_derived
        if committed:
            cur_trace, cur_derived = _simulate_from_scratch(
                trace, committed, lex, rules, derivations)

    def _ent(eid):
        ent = cur_trace.entities.get(eid)
        return ent

    def _ret(value):
        # Guarantee the ContextVar token is reset on every exit.
        _PRESERVE_CONSTRAINTS.reset(_preserve_token)
        return value

    for _ in range(6):
        progress = False
        unresolved = False

        # Verb-level role property constraints.
        # Statically-unachievable property values are pre-filtered:
        # if no rule/action/derivation writes (slot, value), the
        # candidate verb can never satisfy this role-property and
        # we reject it without recording a slot-leaf failure. This
        # keeps the failure attribution at the parent relation/
        # property level instead of bottoming out at e.g.
        # state_of_matter=likva on a balcony.
        writable = _writable_slot_values(rules, lex.actions, derivations)
        for role_name, eid in list(roles.items()):
            ent = _ent(eid)
            if ent is None:
                return _ret(([], False))
            role_spec = next(
                (r for r in action.roles if r.name == role_name), None)
            if role_spec is not None and role_spec.properties:
                for prop_slot, prop_vals in role_spec.properties.items():
                    values = _entity_property_values(
                        ent, prop_slot, cur_trace, cur_derived)
                    if values & set(prop_vals):
                        continue
                    unresolved = True
                    expected = prop_vals[0] if prop_vals else None
                    if expected is None:
                        return _ret(([], False))
                    if (prop_slot, expected) not in writable:
                        return _ret(([], False))
                    sub = plan_to_achieve(
                        eid, prop_slot, expected, actor_id,
                        cur_trace, lex, rules, derived=cur_derived,
                        derivations=derivations,
                        max_depth=max_depth, _depth=depth, _seen=seen)
                    if sub is None:
                        return _ret(([], False))
                    if sub:
                        committed.extend(sub)
                        progress = True
                        _refresh()
                        ent = _ent(eid)
                        if ent is None:
                            return _ret(([], False))
            # Rule-level role pattern entity constraints.
            if event_pat is not None:
                role_pat = event_pat.role_patterns.get(role_name)
                if role_pat is not None:
                    pattern_props = _extract_pattern_props(role_pat)
                    for prop_slot, expected in pattern_props.items():
                        values = _entity_property_values(
                            ent, prop_slot, cur_trace, cur_derived)
                        if expected in values:
                            continue
                        unresolved = True
                        if (prop_slot, expected) not in writable:
                            return _ret(([], False))
                        sub = plan_to_achieve(
                            eid, prop_slot, expected, actor_id,
                            cur_trace, lex, rules, derived=cur_derived,
                            derivations=derivations,
                            max_depth=max_depth, _depth=depth, _seen=seen)
                        if sub is None:
                            return [], False
                        if sub:
                            committed.extend(sub)
                            progress = True
                            _refresh()
                            ent = _ent(eid)
                            if ent is None:
                                return [], False

        # Action-level preconditions, in the order picked by the
        # outer permutation wrapper.
        for pc in pcs_order:
            if isinstance(pc, RelationPrecondition):
                eids = [roles.get(rn) for rn in pc.roles]
                if any(e is None for e in eids):
                    continue
                if _has_relation(
                        pc.rel, tuple(eids), cur_trace, cur_derived, lex):
                    _push_preserve(("rel", pc.rel, tuple(eids)))
                    continue
                unresolved = True
                pre_event_count = len(cur_trace.events)
                sub = plan_to_establish_relation(
                    pc.rel, tuple(eids), actor_id,
                    cur_trace, lex, rules, derived=cur_derived,
                    derivations=derivations,
                    max_depth=max_depth, _depth=depth, _seen=seen)
                if sub is None:
                    return _ret(([], False))
                if sub:
                    committed.extend(sub)
                    progress = True
                    _refresh()
                    # Post-refresh: the sub-plan may have created a new
                    # entity (TransferN split, etc.) that morally
                    # satisfies the goal even though the literal eids
                    # don't. Rebind the parent action's role to the
                    # new entity so subsequent pc checks and the final
                    # plan step pick it up. See `_REBIND_RESOLVERS`.
                    if not _has_relation(
                            pc.rel, tuple(eids),
                            cur_trace, cur_derived, lex):
                        rebind = _find_post_subplan_replacement(
                            pc.rel, eids, pre_event_count, cur_trace, lex)
                        if rebind is not None:
                            arg_idx, new_eid = rebind
                            old_eid = eids[arg_idx]
                            for role_name, eid in list(roles.items()):
                                if eid == old_eid:
                                    roles[role_name] = new_eid
                            eids[arg_idx] = new_eid
                _push_preserve(("rel", pc.rel, tuple(eids)))
            elif isinstance(pc, IfPropertyPrecondition):
                eid = roles.get(pc.role)
                if eid is None:
                    continue
                ent = _ent(eid)
                if ent is None:
                    return _ret(([], False))
                gate = _entity_property_values(
                    ent, pc.if_property, cur_trace, cur_derived)
                if pc.if_value not in gate:
                    continue
                current = _entity_property_values(
                    ent, pc.then_property, cur_trace, cur_derived)
                if pc.then_value in current:
                    continue
                unresolved = True
                sub = plan_to_achieve(
                    eid, pc.then_property, pc.then_value, actor_id,
                    cur_trace, lex, rules, derived=cur_derived,
                    derivations=derivations,
                    max_depth=max_depth, _depth=depth, _seen=seen)
                if sub is None:
                    return _ret(([], False))
                if sub:
                    committed.extend(sub)
                    progress = True
                    _refresh()
            elif isinstance(pc, MatchPrecondition):
                # Pure rejection: no subgoaling. The slots involved
                # (terrain, material, ...) are intrinsic — no verb
                # writes them — so an empty intersection means the
                # candidate role-fill is incompatible. Caller tries
                # the next candidate in the enumeration.
                eid_a = roles.get(pc.role_a)
                eid_b = roles.get(pc.role_b)
                if eid_a is None or eid_b is None:
                    continue
                ent_a = _ent(eid_a)
                ent_b = _ent(eid_b)
                if ent_a is None or ent_b is None:
                    return _ret(([], False))
                values_a = _entity_property_values(
                    ent_a, pc.slot_a, cur_trace, cur_derived)
                values_b = _entity_property_values(
                    ent_b, pc.slot_b, cur_trace, cur_derived)
                if not values_a & values_b:
                    return _ret(([], False))

        if not unresolved:
            return _ret((committed, True))
        if not progress:
            return _ret(([], False))
    return _ret(([], False))


_PLANNER_RNG: contextvars.ContextVar = contextvars.ContextVar(
    "_PLANNER_RNG", default=None)


# Optional callback the seeder installs to materialize a missing
# role-binding entity on demand. When `_find_role_filler` exhausts
# in-trace candidates AND this resolver is set, the planner invokes
# it with the role_spec (and the in-flight trace + exclude set) and
# the resolver returns a freshly-spawned eid or None. Lets the
# seeder start from a minimal scene and let the planner request
# entities as it discovers it needs them — see project_seeder_redesign
# memory for the three-layer design.
#
# Signature: resolver(role_spec, trace, lex, exclude) -> eid | None.
# Default None preserves current behavior (no spawning; missing
# entities cause the candidate plan to fail).
_ENTITY_RESOLVER: contextvars.ContextVar = contextvars.ContextVar(
    "_ENTITY_RESOLVER", default=None)

# Constraints that any candidate sub-plan must preserve, accumulated
# by `_resolve_preconditions` as each precondition resolves. Inner
# planners (plan_to_establish_relation, _plan_via_derivation, ...)
# read this and reject candidates whose simulation would invalidate
# a prior PC. Without it, sibling subgoals oscillate — e.g.
# samloke(klara, mantelo) commits "klara → manĝejo", then
# samloke(klara, pavel) commits "klara → laborejo", which breaks
# the first samloke; the loop then re-resolves the first, and so on.
#
# Each entry is ("rel", relation_name, args_tuple) — only relation
# constraints for now; property constraints can be added if the
# pattern surfaces there too.
_PRESERVE_CONSTRAINTS: contextvars.ContextVar = contextvars.ContextVar(
    "_PRESERVE_CONSTRAINTS", default=())

# Tracks how deeply nested the current `_resolve_preconditions` call
# is. The PC-permutation wrapper uses this to bound where alternate
# orderings are tried — without a cap the cost compounds at every
# multi-PC action down a chain (N!^D). Cap chosen so the locked-
# door-key cascade (which needs malŝlosi-PC reordering at depth ~3)
# still works while shallower actions don't multiplicatively explore.
_RP_DEPTH: contextvars.ContextVar = contextvars.ContextVar(
    "_RP_DEPTH", default=0)
_RP_PERMUTE_MAX_DEPTH = 0


# Per-`plan_for_drive` simulation budget. Counts each
# `_simulate_from_scratch` call; when the budget is exhausted, raises
# `_BudgetExceeded`, which unwinds to plan_for_drive and converts to a
# None plan. Bounds wall-clock latency on pathological searches
# (deep multi-room knowledge goals that the planner can't satisfy
# within max_depth) without affecting plans that have a reachable
# solution — typical regression scenes use ~200 simulations, the
# largest successful multi-room cases ~600. Default budget is
# generous (5000) so it only bites genuine runaways.
class _BudgetExceeded(Exception):
    """Internal signal: simulation budget consumed for this plan."""


class _SimulationBudget:
    __slots__ = ("remaining",)
    def __init__(self, initial: int):
        self.remaining = initial
    def consume(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise _BudgetExceeded()


_SIM_BUDGET: contextvars.ContextVar = contextvars.ContextVar(
    "_SIM_BUDGET", default=None)


# Last unmet sub-goal recorded by the planner during its DFS. Each
# leaf-failure call site (`plan_to_establish_relation`, `plan_to_achieve`,
# `plan_to_reach_count`) writes here when it bails with None, so the
# dispatcher can surface a structured failure reason after planning.
#
# Storage shape: None, or (depth, reason). Depth is the planner's DFS
# recursion depth at the failure point. Deepest-wins: the recorded
# reason is the most-deeply-nested leaf the planner ever reached
# across ANY explored DFS branch — i.e., the path that came closest
# to a solution before giving up. First-write-wins on ties so the
# behavior is deterministic.
#
# Why deepest, not first: an earlier (now-removed) first-write-wins
# strategy reported the deepest leaf of the FIRST DFS branch. But the
# planner enumerates many derivations per goal; the FIRST one explored
# is often a one-step dead-end (e.g. en_implies_samloke trying to put
# the actor inside a tree), while the actual blocker lives deep in a
# later, longer branch. First-write-wins pinned the report on the
# shallow noise. Deepest-wins picks the leaf that actually represents
# planner exhaustion: the place a successful derivation chain finally
# couldn't bridge.
#
# Reason shape:
#   ("relation", relation_name, args_tuple)
#   ("property", entity_id, slot, value)
#   ("count", actor_id, concept_lemma, target_count)
#   ("budget",)  — set with sentinel depth by the dispatcher so it
#                  always wins (budget exhaustion is the real cause
#                  whenever it fires).
_FAILURE_REASON: contextvars.ContextVar = contextvars.ContextVar(
    "_FAILURE_REASON", default=None)

# Sentinel "infinite" depth for terminal reasons (budget exhaustion)
# that should always overwrite any subgoal-leaf reason.
_TERMINAL_DEPTH = 1 << 30


def _record_failure(reason, depth: int = 0) -> None:
    """Record a failed sub-goal. Deepest-wins: keeps the most-deeply-
    nested failure recorded across all DFS branches. Ties keep the
    first one written (deterministic). Best-effort: outside a planner
    context the write is harmless."""
    try:
        cur = _FAILURE_REASON.get()
        if cur is not None and cur[0] >= depth:
            return
        _FAILURE_REASON.set((depth, reason))
    except LookupError:
        pass


def get_planner_failure_reason():
    """Read the failure reason recorded by the planner during the
    current `plan_for_drive` call. Returns the bare reason tuple, or
    None if no failure was recorded."""
    cur = _FAILURE_REASON.get()
    if cur is None:
        return None
    return cur[1]


def _candidate_breaks_preserved(sub_plan, trace, lex, rules, derivations):
    """True if simulating `trace + sub_plan` would invalidate any
    constraint in the current `_PRESERVE_CONSTRAINTS`. Empty plans
    (no events to fire) trivially preserve, since they don't change
    state.

    Known limitation: this strict check rejects any candidate that
    breaks a preserved constraint, even when the break is temporary
    (e.g. `iri` away to fetch a tool breaks `samloke(agent, theme)`
    momentarily; a return trip would re-establish it). Trying to
    relax the check — either by removing it or by probing whether
    the broken constraint is re-establishable — caused either
    long-tail planner slowdowns or infinite recursion via the inner
    plan-call. The proper fix is a STRIPS-with-re-establishment
    planner that simulates the plan step-by-step and checks each
    action's preconditions at firing time, then auto-inserts
    restoration steps. That's a larger rewrite. For now, the
    reflexive-self-slot seeder (sekigi/purigi) scatters the
    required instrument with `pressure="near"` so no fetch-trip is
    needed."""
    preserve = _PRESERVE_CONSTRAINTS.get()
    if not preserve or not sub_plan:
        return False
    sim_trace, sim_derived = _simulate_from_scratch(
        trace, sub_plan, lex, rules, derivations)
    for entry in preserve:
        if entry[0] != "rel":
            continue
        _, rel_name, args = entry
        if not _has_relation(rel_name, args, sim_trace, sim_derived, lex):
            return True
    return False


def _count_owned(actor_id: str, concept_lemma: str, trace) -> int:
    """Sum the `count` slot across all stacks of `concept_lemma` owned
    by `actor_id`. Used by `plan_to_reach_count`'s progress check.
    Stacks where count is unset default to 1."""
    total = 0
    for r in trace.relations:
        if r.relation != "havi":
            continue
        if r.args[0] != actor_id:
            continue
        ent = trace.entities.get(r.args[1])
        if ent is None or ent.concept_lemma != concept_lemma:
            continue
        if ent.destroyed_at_event is not None:
            continue
        try:
            total += int(ent.properties.get("count", ["1"])[0])
        except (TypeError, ValueError):
            total += 1
    return total


def plan_to_reach_count(target_owner_id: str, concept_lemma: str,
                        target_count: int, trace, lex, rules,
                        *, planner_actor_id: Optional[str] = None,
                        derived=None, derivations=None,
                        max_depth=8):
    """Public entry: scope failure recording (see plan_to_establish_relation
    for rationale)."""
    _saved = _FAILURE_REASON.get()
    result = _plan_to_reach_count_impl(
        target_owner_id, concept_lemma, target_count, trace, lex, rules,
        planner_actor_id=planner_actor_id,
        derived=derived, derivations=derivations,
        max_depth=max_depth)
    if result is not None:
        _FAILURE_REASON.set(_saved)
    return result


def _plan_to_reach_count_impl(target_owner_id: str, concept_lemma: str,
                              target_count: int, trace, lex, rules,
                              *, planner_actor_id: Optional[str] = None,
                              derived=None, derivations=None,
                              max_depth=8):
    """Plan for `target_owner_id` to own at least `target_count` units
    of `concept_lemma`, summed across all stacks. Greedy across sources:
    pick the largest unowned stack first, plan to acquire it via
    `plan_to_establish_relation('havi', ...)`, simulate, repeat until
    the target is reached or no more sources are available.

    `planner_actor_id` is who's doing the planning (the entity bound
    to `agent` in the chosen verb). Defaults to `target_owner_id` for
    self-acquire; differs from it for altruism (donor planning to give
    a recipient N units → planner_actor_id=donor, target_owner_id=
    recipient — the planner picks doni since it's the verb where the
    planner-actor binds to agent).

    Returns a flat plan (concatenation of acquisition sub-plans) or
    None if unreachable. Empty list if already satisfied."""
    if planner_actor_id is None:
        planner_actor_id = target_owner_id
    current = _count_owned(target_owner_id, concept_lemma, trace)
    if current >= target_count:
        return []

    # Find candidate source stacks: live entities of the right concept
    # not currently owned by the target owner.
    candidates = []
    for eid, ent in trace.entities.items():
        if ent.concept_lemma != concept_lemma:
            continue
        if ent.destroyed_at_event is not None:
            continue
        owned_by_target = any(
            r.relation == "havi" and tuple(r.args) == (target_owner_id, eid)
            for r in trace.relations)
        if owned_by_target:
            continue
        candidates.append(eid)
    if not candidates:
        return None

    # Largest stacks first — minimizes plan length.
    def _stack_count(eid):
        try:
            return int(trace.entities[eid].properties.get("count", ["1"])[0])
        except (TypeError, ValueError):
            return 1
    candidates.sort(key=_stack_count, reverse=True)

    plan: list = []
    cur_trace = trace
    cur_derived = derived
    # Cap candidates considered. Each acquisition's
    # plan_to_establish_relation call is expensive (each spawns its own
    # backward-chaining tree) and after the first acquisition the trace
    # state is more complex (actor relocated, owns a stack), making
    # subsequent calls even slower. Two acquisitions covers the
    # split-source case; more would mostly be diminishing returns.
    for cand in candidates[:2]:
        # Use a shorter depth on the inner call. With max_depth=8 the
        # planner explores deep alternatives that rarely help here;
        # 5 is enough for "iri → eniri → preni"-style chains and
        # caps explosion when the second acquisition kicks in from
        # a complex post-first-acquisition state.
        sub = plan_to_establish_relation(
            "havi", (target_owner_id, cand), planner_actor_id,
            cur_trace, lex, rules,
            derived=cur_derived, derivations=derivations,
            max_depth=min(5, max_depth))
        if sub is None:
            continue
        # Stamp the acquisition step with explicit quantity = remaining
        # need, capped by the source stack's count. TransferN reads
        # `event.quantity` and either does a full ownership swap (when
        # qty >= source.count) or a partial split (qty < source.count),
        # leaving a smaller stack with the prior owner.
        need = max(1, target_count - current)
        cand_count = _stack_count(cand)
        qty = min(need, cand_count)
        if qty > 1 and sub:
            last = sub[-1]
            verb, roles, _ = _unpack_step(last)
            if verb in _quantifiable_transfer_verbs(rules):
                sub = sub[:-1] + [(verb, roles, qty)]
        plan = plan + sub
        cur_trace, cur_derived = _simulate_from_scratch(
            trace, plan, lex, rules, derivations)
        current = _count_owned(target_owner_id, concept_lemma, cur_trace)
        if current >= target_count:
            return plan
    _record_failure(("count", target_owner_id, concept_lemma, target_count))
    return None


def plan_to_establish_relation(relation, target_args, actor_id,
                               trace, lex, rules, *, derived=None,
                               derivations=None,
                               max_depth=8, _depth=0, _seen=None):
    """Public entry: scope failure recording. Save the current
    failure_reason on entry; restore it if this call returns a working
    plan. Failures captured inside a successful call are noise from
    rejected sibling DFS branches — discard them. Failures inside a
    call that returns None survive (the call's caller may later treat
    it as the actual blocker)."""
    _saved = _FAILURE_REASON.get()
    result = _plan_to_establish_relation_impl(
        relation, target_args, actor_id, trace, lex, rules,
        derived=derived, derivations=derivations,
        max_depth=max_depth, _depth=_depth, _seen=_seen)
    if result is not None:
        _FAILURE_REASON.set(_saved)
    return result


def _plan_to_establish_relation_impl(
        relation, target_args, actor_id,
        trace, lex, rules, *, derived=None,
        derivations=None,
        max_depth=8, _depth=0, _seen=None):
    """Find a sequence of actions that asserts rel(relation, *target_args).
    Returns [] if already true, list of (verb, roles), or None.

    If no verb adds the relation directly AND `derivations` is given,
    walks derivations whose `implies` produces the relation and
    subgoals their `when + given` patterns. This is how `samloke` etc.
    are reached — no verb adds samloke, but a derivation produces it
    from shared `en` containers."""
    _seen = _seen or set()
    # At top-level entry, materialize the derived layer once if the
    # caller passed derivations but no precomputed cache. Without
    # this, downstream candidate filters and synthesis checks see
    # an empty derived layer and silently fail to find paths that
    # depend on derivation-only properties (e.g. illuminated, terrain).
    # Mirrors plan_to_achieve's same-shape guard.
    if derived is None and derivations is not None and _depth == 0:
        derived = _cached_compute_derived_state(trace, derivations, lex)
    key = ("rel", relation, tuple(target_args))
    if key in _seen or _depth >= max_depth:
        return None
    if _has_relation(relation, target_args, trace, derived, lex):
        return []
    # Schema-level impossibility filter: if the relation's
    # `arg_excludes` would forbid this entity at this position, no
    # action sequence can establish it — `assert_relation` would
    # reject. Catches the planner's "havi(actor, location)" exploration
    # (for malŝlosi → ŝlosilo paths bottoming out as "actor wants to
    # acquire the room") and any future relation-arg-type mismatch
    # without per-relation hardcoding. Type-spine subtype checks too:
    # `havi.theme=physical` forbids non-physical themes regardless of
    # arg_excludes.
    rel_def = lex.relations.get(relation)
    if rel_def is not None and len(target_args) == rel_def.arity:
        for i, arg_eid in enumerate(target_args):
            ent = trace.entities.get(arg_eid)
            if ent is None:
                continue
            expected_type = rel_def.arg_types[i]
            if not lex.types.is_subtype(ent.entity_type, expected_type):
                _record_failure(
                    ("relation", relation, tuple(target_args)),
                    depth=_depth)
                return None
            if i < len(rel_def.arg_excludes):
                forbidden = rel_def.arg_excludes[i]
                if any(lex.types.is_subtype(ent.entity_type, f)
                       for f in forbidden):
                    _record_failure(
                        ("relation", relation, tuple(target_args)),
                        depth=_depth)
                    return None
    # Movement-verb impossibility: apud(X, Y) is unreachable from
    # current state if X is already `en Y`. Every movement verb
    # (iri/veni/kuri/flugi/...) carries a NotPattern `~rel("en", IA,
    # ID)` that refuses to move the agent into a location it's
    # already in — so no verb can produce apud(X, Y) here. Without
    # this short-circuit, derivations like `shared_apud_means_samloke`
    # backchain samloke(actor, X) into apud(actor, current_loc) +
    # apud(X, current_loc), and the actor-already-in-current case
    # bottoms out on the deepest leaf and surfaces as a confusing
    # "actor cannot reach own location" failure.
    #
    # Returns None silently — does NOT record this as a failure.
    # The outer goal (samloke etc.) will record its own failure if
    # all its derivation candidates fail. Suppressing this leaf lets
    # the actual blocker surface.
    if relation == "apud" and len(target_args) == 2:
        x_eid, y_eid = target_args
        for r in trace.relations:
            if (r.relation == "en"
                    and len(r.args) == 2
                    and r.args[0] == x_eid
                    and r.args[1] == y_eid):
                return None
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
    # Weighted shuffle of equally-priority candidates so chain-rich
    # verbs (more preconditions / role-properties → more subgoaling
    # potential) are picked more often, surfacing longer chains
    # naturally. Without the weight, the shuffle was uniform and the
    # planner reliably picked the verb with fewest preconditions
    # (`iri` over `veturi`), making rich chains rare. Weight is just
    # a count of action.preconditions + total role.properties — pure
    # data, no per-verb hardcoding.
    # Weighted sampling: combine chain-richness and pre-satisfaction
    # into a single weight, then shuffle. Pre-satisfaction is a strong
    # P(success) proxy — candidates whose preconditions already hold
    # need fewer subgoals and rarely dead-end — so we amplify it
    # exponentially while letting richness break ties / explore
    # subgoal-deep variants. The previous shape (weighted shuffle
    # then stable sort by satisfaction) was deterministic per goal
    # given trace state; sampling makes same-goal-same-trace produce
    # plan-shape variety across seeds, useful for training-data
    # diversity. No additional compute since high-P candidates still
    # win the draw most of the time.
    _shuffle_rng = _PLANNER_RNG.get()
    if _shuffle_rng is not None:
        weights = []
        for c in rel_candidates:
            richness = _chain_richness_weight(c, lex)
            satisfied = _count_satisfied_preconds(
                c, target_args, actor_id, trace, derived, lex)
            weights.append(richness * (2 ** satisfied))
        rel_candidates = _weighted_shuffle(
            rel_candidates, weights, _shuffle_rng)
    else:
        # No rng (tests / deterministic mode): fall back to ordering
        # by satisfaction count alone.
        rel_candidates.sort(
            key=lambda c: -_count_satisfied_preconds(
                c, target_args, actor_id, trace, derived, lex))

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
            elif src[0] == "given":
                _tag, g_pat, fixed_arg_name = src
                rel_def = lex.relations.get(g_pat.relation)
                if rel_def is None:
                    ok = False
                    break
                arg_names = list(rel_def.arg_names)
                if fixed_arg_name not in arg_names:
                    ok = False
                    break
                fixed_idx = arg_names.index(fixed_arg_name)
                # Find a trace relation matching g_pat with fixed
                # arg = target_eid; bind the other-arg Vars from the
                # trace's actual values via `predetermined`. Mirrors
                # the sibling-AddRelation back-derivation used by the
                # ("created", ...) branch above.
                match_args = None
                for asserted in trace.relations:
                    if asserted.relation != g_pat.relation:
                        continue
                    if len(asserted.args) != len(arg_names):
                        continue
                    if asserted.args[fixed_idx] != target_eid:
                        continue
                    match_args = asserted.args
                    break
                if match_args is None:
                    ok = False
                    break
                for other_name, other_pat in g_pat.arg_patterns.items():
                    if other_name == fixed_arg_name:
                        continue
                    other_var = _bind_var_in_pattern(other_pat)
                    if other_var is None:
                        continue
                    if other_name not in arg_names:
                        ok = False
                        break
                    other_idx = arg_names.index(other_name)
                    other_value = match_args[other_idx]
                    existing = predetermined.get(id(other_var))
                    if existing is not None and existing != other_value:
                        ok = False
                        break
                    predetermined[id(other_var)] = other_value
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
        # Marian en la salono" are incoherent for most verbs. Verbs
        # marked `reflexive_ok` (sekigi/lavi/vesti) opt in to allowing
        # agent==theme — see `_bindings_ok_with_reflexive`.
        if not _bindings_ok_with_reflexive(roles, action):
            continue
        # Fill remaining roles from scene.
        for role_spec in action.roles:
            if role_spec.name in roles:
                continue
            exclude = set(roles.values())
            # Reflexive_ok actions: when filling theme, ALLOW the agent
            # as a candidate. Without this exclusion-skip, the planner
            # for sekigi(agent=actor, theme=?) excludes actor, so it
            # picks a different wet thing and never plans self-drying.
            if (action.reflexive_ok and role_spec.name == "theme"
                    and "agent" in roles):
                exclude = exclude - {roles["agent"]}
            eid = _find_role_filler(
                role_spec, trace, lex, derived=derived,
                exclude=exclude)
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
        candidate_plan = sub_plans + [(event_pat.action, roles)]
        if _candidate_breaks_preserved(
                candidate_plan, trace, lex, rules, derivations):
            continue
        return candidate_plan

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
    _record_failure(
        ("relation", relation, tuple(target_args)), depth=_depth)
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
    positive subgoals are achieved.

    Two flavors of NotPattern:
      (a) `~rel(...)`: inner is a RelPattern. Check via existing
          `_notpattern_inner_holds`.
      (b) `~entity(type=..., concept=..., slot=...)` inside an And
          that binds a var. Identity constraints (type/concept/has_suffix)
          are immutable, so a violation here means the rule can't fire
          under this binding — reject. Slot constraints are flippable
          and handled separately via slots_to_subgoal; we leave those
          to the planner's subgoal loop and don't reject here.
    """
    for pat in [when] + list(given):
        for np in _walk_for_not_patterns(pat):
            if _notpattern_inner_holds(np.inner, var_bindings,
                                        trace, derived, lex):
                return True
        for var, ep in _walk_negated_entity_patterns_with_vars(pat):
            if id(var) not in var_bindings:
                continue
            ent = trace.entities.get(var_bindings[id(var)])
            if ent is None:
                continue
            identity = {k: v for k, v in ep.constraints.items()
                        if k in ("type", "concept", "has_suffix")
                        and not isinstance(v, Var)}
            if identity and _entity_matches_literal_constraints(
                    ent, identity, lex):
                return True
    return False


def _walk_negated_entity_patterns_with_vars(pattern):
    """Yield (var, EntityPattern) pairs for each NotPattern(EntityPattern)
    inside an AndPattern that binds `var`. Pairs the inner identity
    constraints with their target binding so `_notpatterns_violated`
    can evaluate them. Recurses into RelPattern arg_patterns so
    inline rel-arg negations (samloke_chains_through_en's container)
    are picked up."""
    if isinstance(pattern, AndPattern):
        bound = _bind_var_in_pattern(pattern)
        if bound is not None:
            for ep in _yield_negated_entity_patterns(pattern):
                yield (bound, ep)
        yield from _walk_negated_entity_patterns_with_vars(pattern.left)
        yield from _walk_negated_entity_patterns_with_vars(pattern.right)
    elif isinstance(pattern, RelPattern):
        for arg_pat in pattern.arg_patterns.values():
            yield from _walk_negated_entity_patterns_with_vars(arg_pat)


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
    """Yield EntityPatterns that constrain `target_var` (anywhere inside
    an AndPattern subtree whose conjunction binds it). Used to discover
    literal constraints on a free Var so candidate enumeration can
    pre-filter instead of relying on rel-pattern subgoaling to fail
    wrong picks.

    Recurses into RelPattern arg_patterns: a constraint can sit
    inline as `rel("en", container=(entity(...) & bind(B)))` rather
    than at top level. Also recurses through nested AndPatterns once
    the binding-And is found, so multi-conjunct shapes like
    `(c1 & c2 & bind(B))` yield both c1 and c2."""
    if isinstance(pattern, AndPattern):
        if _bind_var_in_pattern(pattern) is target_var:
            yield from _yield_positive_entity_patterns(pattern)
            return
        yield from _walk_for_entity_patterns_binding(pattern.left, target_var)
        yield from _walk_for_entity_patterns_binding(pattern.right, target_var)
    elif isinstance(pattern, RelPattern):
        for arg_pat in pattern.arg_patterns.values():
            yield from _walk_for_entity_patterns_binding(
                arg_pat, target_var)


def _walk_for_negated_entity_patterns_binding(pattern, target_var):
    """Yield EntityPatterns nested inside NotPatterns within any And
    subtree that binds `target_var`. Mirrors the positive walker but
    for negated constraints — `container = ~entity(openness="fermita")
    & bind(B)` yields the inner `entity(openness="fermita")`. The
    planner inverts these: instead of "make slot=value hold," subgoal
    "make slot != value" (typically by flipping the slot to another
    vocabulary value)."""
    if isinstance(pattern, AndPattern):
        if _bind_var_in_pattern(pattern) is target_var:
            yield from _yield_negated_entity_patterns(pattern)
            return
        yield from _walk_for_negated_entity_patterns_binding(
            pattern.left, target_var)
        yield from _walk_for_negated_entity_patterns_binding(
            pattern.right, target_var)
    elif isinstance(pattern, RelPattern):
        for arg_pat in pattern.arg_patterns.values():
            yield from _walk_for_negated_entity_patterns_binding(
                arg_pat, target_var)


def _yield_positive_entity_patterns(pattern):
    """Recursively yield EntityPatterns from inside an AndPattern
    subtree (skipping NotPatterns)."""
    if isinstance(pattern, EntityPattern):
        yield pattern
    elif isinstance(pattern, AndPattern):
        yield from _yield_positive_entity_patterns(pattern.left)
        yield from _yield_positive_entity_patterns(pattern.right)


def _yield_negated_entity_patterns(pattern):
    """Recursively yield EntityPatterns nested inside NotPatterns
    from within an AndPattern subtree."""
    if isinstance(pattern, NotPattern):
        if isinstance(pattern.inner, EntityPattern):
            yield pattern.inner
    elif isinstance(pattern, AndPattern):
        yield from _yield_negated_entity_patterns(pattern.left)
        yield from _yield_negated_entity_patterns(pattern.right)


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
    # Symmetric relations (samloke, apud) zip with target_args in both
    # orders — picking one direction silently skips derivations that
    # match the other. e.g. samloke_chains_through_en's `en(A, B) ∧
    # samloke(B, C) → samloke(A, C)` only resolves cleanly with A bound
    # to whichever target has an `en` container chain. Iterating both
    # orders lets the planner find the working assignment without
    # needing per-derivation symmetry.
    is_symmetric = (
        relation in lex.relations
        and lex.relations[relation].symmetric
        and len(target_args) == 2
        and target_args[0] != target_args[1])
    binding_orders = [tuple(target_args)]
    if is_symmetric:
        binding_orders.append((target_args[1], target_args[0]))

    for cur_targets in binding_orders:
      for d in derivations:
        for imp in d.implies:
            if not isinstance(imp, RelationImplication):
                continue
            if imp.name != relation or len(imp.args) != len(cur_targets):
                continue
            # Bind imp.args ↔ cur_targets. Var-args record bindings;
            # literal-args must equal the corresponding target.
            var_bindings: dict[int, str] = {}
            mismatch = False
            for arg, target in zip(imp.args, cur_targets):
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
                    # Candidates: only the bound targets' own containers.
                    # Used when the derivation imposes no entity-level
                    # identity constraints on the free var (samloke shape).
                    #
                    # When the actor is one of the targets, prefer the
                    # NON-actor target's container so the actor does the
                    # moving. Without this the planner satisfies
                    # samloke(agent_in_lib, recipient_in_kitchen) by
                    # moving the recipient to the library, which is
                    # narratively backwards for "actor goes back to
                    # tell" chains.
                    #
                    # We deliberately do NOT enumerate other locations
                    # as a "third meeting place". For samloke(A, B) the
                    # only ways to satisfy the goal are: A moves to B's
                    # container, or B moves to A's container, or both
                    # move to a third place. The third option is
                    # strictly worse (an extra movement event) than the
                    # first two unless something forces it, which never
                    # happens in our verb roster. Enumerating extra
                    # locations multiplied the recursive subgoal cost
                    # by N_locations, blowing up multi-room planning
                    # without ever producing a winning plan.
                    ordered_targets = list(target_args)
                    if actor_id in ordered_targets:
                        ordered_targets.sort(
                            key=lambda x: 0 if x != actor_id else 1)
                    # Transitive container chain: for `lakto en glaso,
                    # glaso en laborejo`, both `glaso` (immediate) and
                    # `laborejo` (the room) are valid "meeting places"
                    # under the en/sur samloke chain rules. Without the
                    # chain walk, the planner only sees the immediate
                    # container — and can't plan en(actor, glaso) since
                    # glaso isn't a location.
                    candidates = []
                    seen_cand: set[str] = set()
                    for target in ordered_targets:
                        for c in _container_chain_of(target, trace):
                            if c not in seen_cand:
                                seen_cand.add(c)
                                candidates.append(c)
                # Drop candidates whose slot constraints are unreachable
                # from any verb's effects (e.g. lights_when_on=yes only
                # holds on lampo; no verb writes it). Without this the
                # planner explores every entity for every free var.
                candidates = _filter_candidates_by_slots(
                    candidates, fv_id, slots_to_subgoal, trace, lex)
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
                    # havas_parton is static — never added at runtime.
                    # Any subgoal whose pair doesn't already hold can't
                    # be satisfied. Short-circuit instead of recursing.
                    if (rel_name_pat == "havas_parton"
                            and len(concrete) == 2
                            and not _has_relation(
                                "havas_parton", tuple(concrete),
                                trace, derived, lex)):
                        sub_ok = False
                        break
                    # Schema-impossibility: if the derivation's bound
                    # subgoal would violate the relation's arg_excludes
                    # or arg_types, skip the candidate without recording
                    # a leaf for it. Otherwise the derivation backchain
                    # produces noise like havi(actor, location) when
                    # exploring havi_implies_samloke, drowning out the
                    # actual outer goal that couldn't be satisfied.
                    if not _relation_args_admissible(
                            rel_name_pat, concrete, trace, lex):
                        sub_ok = False
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
                # Sign="pos" → make slot=value; sign="neg" → flip
                # away from value (try each non-forbidden vocab value
                # until one resolves).
                if free_vars:
                    for entry in slots_to_subgoal:
                        if not sub_ok:
                            break
                        fv_var_local, slot, value, sign = entry
                        if id(fv_var_local) != fv_id:
                            continue
                        target_eid = assignment[fv_id]
                        target_ent = trace.entities.get(target_eid)
                        if target_ent is None:
                            sub_ok = False
                            break
                        actual = _entity_property_values(
                            target_ent, slot, trace, derived)
                        sub = _plan_slot_subgoal(
                            target_eid, slot, value, sign, actual,
                            actor_id, trace, lex, rules, derived,
                            derivations, max_depth, depth, seen)
                        if sub is None:
                            sub_ok = False
                            break
                        sub_plans.extend(sub)
                if sub_ok:
                    if _candidate_breaks_preserved(
                            sub_plans, trace, lex, rules, derivations):
                        continue
                    return sub_plans
    # Last resort: the free var refers to an entity that doesn't exist
    # in the trace yet. Look for a causal rule whose `create_entity`
    # would produce a matching entity AND whose `add_relation` effects
    # satisfy the derivation's rel patterns about it. Plan that rule's
    # trigger verb. For scias_lokon, this finds vidi: vidi creates a
    # fakto and adds konas + subjekto + objekto in one shot, so firing
    # vidi(actor, target) satisfies the entire derivation.
    sub = _plan_via_synthesis(
        relation, target_args, actor_id, trace, lex, rules,
        derivations, derived, max_depth, depth, seen)
    if sub is not None:
        return sub
    return None


def _plan_via_synthesis(relation, target_args, actor_id, trace, lex,
                        rules, derivations, derived,
                        max_depth, depth, seen):
    """Plan a verb whose causal rule synthesizes the entities/relations
    that satisfy the goal — possibly via one derivation step.

    Strategy: try firing each create-entity verb (with role bindings
    seeded from the goal's arg targets) on a forked trace; if the
    goal relation then holds, plan that verb. This catches the
    create-then-relate pattern (vidi creates a fakto whose subjekto
    binding + konas relation, after derivation closure, satisfy
    scias_lokon) without needing to reason symbolically about what
    each rule's effects mean for the goal."""
    # Pre-filter: only worth attempting synthesis when the goal
    # relation is reachable from some create-entity rule's add_relation
    # closure (directly or via one-step derivation). Avoids forking
    # the trace for goals like `en` or `havi` where existing planner
    # paths already handle everything.
    if not _is_synthesis_candidate_relation(relation, rules, derivations):
        return None

    candidate_rules = []
    for rule in rules:
        if not any(isinstance(e, CreateEntity) for e in rule.then):
            continue
        verb_lemma = rule.when.action
        action = lex.actions.get(verb_lemma)
        if action is None:
            continue
        candidate_rules.append((rule, verb_lemma, action))
    _shuffle_rng = _PLANNER_RNG.get()
    if _shuffle_rng is not None:
        _shuffle_rng.shuffle(candidate_rules)

    for rule, verb_lemma, action in candidate_rules:
        for role_bindings in _enumerate_seeded_role_bindings(
                action, target_args, actor_id, trace, lex):
            t_fork, d_fork = _simulate_from_scratch(
                trace, [(verb_lemma, role_bindings)], lex, rules,
                derivations)
            if not _has_relation(
                    relation, target_args, t_fork, d_fork, lex):
                continue
            plan = _plan_specific_action(
                action, role_bindings, actor_id, trace, lex, rules,
                derivations, derived, max_depth, depth, seen,
                trigger_event_pat=rule.when)
            if plan is not None:
                return plan
    return None


_SYNTHESIS_CANDIDATE_CACHE: dict[int, set[str]] = {}


def _is_synthesis_candidate_relation(
    relation: str, rules: list, derivations: list,
) -> bool:
    """Static-precompute (and cache by `id(rules)`) the set of relations
    reachable from any create-entity rule's add_relation effects, plus
    one step of derivation closure. Goals outside this set can't
    benefit from synthesis — skip the expensive forward simulation."""
    cache_key = id(rules)
    cached = _SYNTHESIS_CANDIDATE_CACHE.get(cache_key)
    if cached is None:
        # Direct: relations added by create-entity rules.
        direct: set[str] = set()
        for rule in rules:
            if not any(isinstance(e, CreateEntity) for e in rule.then):
                continue
            for eff in rule.then:
                if isinstance(eff, AddRelation):
                    direct.add(eff.relation)
        # One-step derivation closure: derived relations whose `when`
        # mentions a directly-synthesizable relation.
        reachable = set(direct)
        for d in derivations:
            for imp in d.implies:
                if not isinstance(imp, RelationImplication):
                    continue
                whens = _walk_for_rel_patterns(d.when)
                for g in d.given:
                    whens.extend(_walk_for_rel_patterns(g))
                if any(rp.relation in direct for rp in whens):
                    reachable.add(imp.name)
        cached = reachable
        _SYNTHESIS_CANDIDATE_CACHE[cache_key] = cached
    return relation in cached


def _enumerate_seeded_role_bindings(action, target_args, actor_id,
                                     trace, lex):
    """Yield candidate role-binding dicts for `action`, seeded from
    `target_args` (concrete entity ids the goal wants to involve).

    Heuristic: for each subset of action's roles, try assigning each
    target arg + the actor to a role whose type they're compatible
    with. Other roles get filled at plan-execute time. Bounded by
    O(roles! × len(targets)+1), small for our verb roster."""
    role_specs = action.roles
    candidates_per_role: list[list[str]] = []
    seed_pool = list(set(list(target_args) + [actor_id]))
    for role in role_specs:
        compat = []
        for eid in seed_pool:
            ent = trace.entities.get(eid)
            if ent is None:
                continue
            if not lex.types.is_subtype(ent.entity_type, role.type):
                continue
            compat.append(eid)
        compat.append(None)   # placeholder: leave for _find_role_filler
        candidates_per_role.append(compat)

    def _expand(idx, current):
        if idx == len(role_specs):
            if not current:
                return
            non_none = [v for v in current.values() if v is not None]
            if len(set(non_none)) != len(non_none):
                return   # duplicate eid in two roles
            yield {k: v for k, v in current.items() if v is not None}
            return
        role_name = role_specs[idx].name
        for eid in candidates_per_role[idx]:
            current[role_name] = eid
            yield from _expand(idx + 1, current)
        current.pop(role_name, None)

    seen_combos: set = set()
    for combo in _expand(0, {}):
        key = tuple(sorted(combo.items()))
        if key in seen_combos:
            continue
        seen_combos.add(key)
        yield combo


def _role_name_for_var(rule, var) -> str | None:
    """Find which role of `rule.when` binds `var`. Returns None if
    `var` isn't bound by any role in the trigger event pattern."""
    if not isinstance(rule.when, EventPattern):
        return None
    for role_name, role_pat in rule.when.role_patterns.items():
        for v in _extract_bind_vars(role_pat):
            if v is var:
                return role_name
    return None


def _role_name_for_var(rule, var) -> str | None:
    """Find which role of `rule.when` binds `var`. Returns None if
    `var` isn't bound by any role in the trigger event pattern."""
    if not isinstance(rule.when, EventPattern):
        return None
    for role_name, role_pat in rule.when.role_patterns.items():
        for v in _extract_bind_vars(role_pat):
            if v is var:
                return role_name
    return None


def _plan_specific_action(action, role_bindings, actor_id, trace, lex,
                          rules, derivations, derived,
                          max_depth, depth, seen, *, trigger_event_pat=None):
    """Plan a specific action with given role bindings. Fills any
    missing roles via the standard scene scan, resolves preconditions,
    returns sub_plans + [(verb, roles)] or None."""
    roles = dict(role_bindings)
    action_role_names = {r.name for r in action.roles}
    # Bind agent to actor if available and not already set.
    if "agent" in action_role_names and "agent" not in roles:
        if actor_id not in roles.values():
            roles["agent"] = actor_id
    # Type-check fixed bindings.
    for role_name, eid in roles.items():
        ent = trace.entities.get(eid)
        role_spec = next(
            (r for r in action.roles if r.name == role_name), None)
        if (ent is None or role_spec is None
                or not lex.types.is_subtype(
                    ent.entity_type, role_spec.type)):
            return None
    # Reject duplicate-entity bindings (allow agent==theme on
    # reflexive_ok verbs — see `_bindings_ok_with_reflexive`).
    if not _bindings_ok_with_reflexive(roles, action):
        return None
    # Fill remaining roles from scene.
    for role_spec in action.roles:
        if role_spec.name in roles:
            continue
        exclude = set(roles.values())
        if (action.reflexive_ok and role_spec.name == "theme"
                and "agent" in roles):
            exclude = exclude - {roles["agent"]}
        eid = _find_role_filler(
            role_spec, trace, lex, derived=derived,
            exclude=exclude)
        if eid is None:
            return None
        roles[role_spec.name] = eid
    # Resolve preconditions.
    sub_plans, ok = _resolve_preconditions(
        action, trigger_event_pat, roles, actor_id, trace, lex, rules,
        derived, max_depth, depth + 1, seen, derivations=derivations)
    if not ok:
        return None
    return sub_plans + [(action.lemma, roles)]


def _split_entity_constraints(when, given, target_var):
    """Walk when+given for entity patterns binding `target_var`. Split
    their constraints into:
      identity: concept/type/has_suffix — these define what KIND of
        entity it is, used to filter candidates.
      slots: ordinary property slots — these define current STATE.
        Each entry is (var, slot, value, sign) where sign is "pos"
        (must equal value) or "neg" (must NOT equal value).
        Positive slots come from `entity(...)` patterns; negative
        slots come from `~entity(...)` patterns inside the same
        conjunction. Both can be subgoaled via plan_to_achieve when
        the candidate currently violates them — the negative case
        picks any non-forbidden value from slot.vocabulary.

    Returns (identity_dict, slot_subgoal_list).
    """
    identity: dict[str, Any] = {}
    slots: list[tuple[Var, str, Any, str]] = []
    for pat in [when] + list(given):
        for ep in _walk_for_entity_patterns_binding(pat, target_var):
            for k, val in ep.constraints.items():
                if isinstance(val, Var):
                    continue  # Var-valued — handled elsewhere
                if k in ("type", "concept", "has_suffix"):
                    identity[k] = val
                else:
                    slots.append((target_var, k, val, "pos"))
        for ep in _walk_for_negated_entity_patterns_binding(
                pat, target_var):
            for k, val in ep.constraints.items():
                if isinstance(val, Var):
                    continue
                if k in ("type", "concept", "has_suffix"):
                    # Negative identity ("not a location") — no
                    # standard channel to subgoal a type change, so
                    # let candidate enumeration handle it elsewhere.
                    continue
                slots.append((target_var, k, val, "neg"))
    return identity, slots


def plan_event_firing(verb, requested_roles, actor_id,
                      trace, lex, rules, *, derived=None, derivations=None,
                      max_depth=8, _depth=0, _seen=None):
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
                      max_depth=8, _depth=0, _seen=None):
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
                    max_depth=8, _depth=0, _seen=None):
    """Public entry: scope failure recording (see plan_to_establish_relation
    for rationale)."""
    _saved = _FAILURE_REASON.get()
    result = _plan_to_achieve_impl(
        goal_entity_id, goal_slot, goal_value,
        actor_id, trace, lex, rules,
        derived=derived, derivations=derivations,
        max_depth=max_depth, _depth=_depth, _seen=_seen)
    if result is not None:
        _FAILURE_REASON.set(_saved)
    return result


def _plan_to_achieve_impl(goal_entity_id, goal_slot, goal_value,
                          actor_id, trace, lex, rules, *, derived=None,
                          derivations=None,
                          max_depth=8, _depth=0, _seen=None):
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
        derived = _cached_compute_derived_state(trace, derivations, lex)

    goal_entity = trace.entities.get(goal_entity_id)
    if goal_entity is None:
        return None

    # Already satisfied?
    if goal_value in _entity_property_values(goal_entity, goal_slot, trace, derived):
        return []

    # Statically unachievable? If no rule, action, or derivation in the
    # registry produces (slot, value), the goal can never be reached.
    # Short-circuit and record the failure here, instead of letting
    # the candidate-loop walk every verb/derivation only to find no
    # writer. Catches verŝi-on-balcony chains that try to flip
    # state_of_matter on a location, lock_state on a non-lockable
    # artifact, etc. — the planner formerly recursed deeply into
    # these dead ends and surfaced their state_of_matter/locomotion/
    # etc. leaves as misleading failure reasons.
    writable = _writable_slot_values(rules, lex.actions, derivations)
    if (goal_slot, goal_value) not in writable:
        _record_failure(
            ("property", goal_entity_id, goal_slot, goal_value),
            depth=_depth)
        return None

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
    _shuffle_rng = _PLANNER_RNG.get()
    if _shuffle_rng is not None:
        _shuffle_rng.shuffle(candidates)

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
            # Reflexive verbs let actor==goal_entity (Maria wants
            # Maria.wetness=seka → sekigi(agent=Maria, theme=Maria)).
            # Non-reflexive verbs reject this binding — there's no
            # way for the actor to bring about another's state via a
            # verb where they'd be both agent and patient.
            if (actor_id == goal_entity_id
                    and not getattr(action, "reflexive_ok", False)):
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
        # Exception: actions marked `reflexive_ok` permit agent==theme
        # — sekigi/lavi/vesti, where doing the action to oneself is
        # semantically natural ("Maria sekigis sin"). Other duplicates
        # (agent==instrument, etc.) remain rejected.
        if not _bindings_ok_with_reflexive(roles, action):
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
    _record_failure(
        ("property", goal_entity_id, goal_slot, goal_value),
        depth=_depth)
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
                candidates = _filter_candidates_by_slots(
                    candidates, fv_id, slots_to_subgoal, trace, lex)
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
                    # havas_parton is static — short-circuit the
                    # subgoal when it doesn't already hold. See
                    # `_plan_via_derivation` for rationale.
                    if (rel_name_pat == "havas_parton"
                            and len(concrete) == 2
                            and not _has_relation(
                                "havas_parton", tuple(concrete),
                                trace, derived, lex)):
                        sub_ok = False
                        break
                    # Schema-impossibility: skip derivation candidates
                    # whose subgoaled args would violate arg_types /
                    # arg_excludes. See `_plan_via_derivation`.
                    if not _relation_args_admissible(
                            rel_name_pat, concrete, trace, lex):
                        sub_ok = False
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
                # Subgoal entity-pattern slot constraints. Two scopes:
                # (a) constraints on imp.entity itself — e.g. when
                #     `outdoor_is_luma` is bound to an indoor location,
                #     its `indoor_outdoor=ekstera` constraint must be
                #     satisfied (it can't be) so this derivation must
                #     NOT claim the goal is achievable.
                # (b) constraints on the free var (lock_state on the
                #     seruro, power_state on the lamp).
                # Both are subgoaled via plan_to_achieve.
                # NB: rebind to es_/fv_ names so we don't shadow the
                # function's `slot` / `value` params — earlier shadowing
                # caused the next derivation in the outer loop to be
                # skipped because `imp.slot != slot` saw the leaked
                # inner loop value.
                _, imp_entity_slots = _split_entity_constraints(
                    d.when, d.given, imp.entity)
                imp_ent = trace.entities.get(entity_id)
                for entry in imp_entity_slots:
                    if not sub_ok:
                        break
                    es_var, es_slot, es_value, es_sign = entry
                    if es_var is not imp.entity:
                        continue
                    if imp_ent is None:
                        sub_ok = False
                        break
                    actual = _entity_property_values(
                        imp_ent, es_slot, trace, derived)
                    sub = _plan_slot_subgoal(
                        entity_id, es_slot, es_value, es_sign, actual,
                        actor_id, trace, lex, rules, derived,
                        derivations, max_depth, depth, seen)
                    if sub is None:
                        sub_ok = False
                        break
                    sub_plans.extend(sub)
                if sub_ok and free_vars:
                    for entry in slots_to_subgoal:
                        if not sub_ok:
                            break
                        fv_var_local, fv_slot, fv_value, fv_sign = entry
                        if id(fv_var_local) != fv_id:
                            continue
                        target_eid = assignment[fv_id]
                        target_ent = trace.entities.get(target_eid)
                        if target_ent is None:
                            sub_ok = False
                            break
                        actual = _entity_property_values(
                            target_ent, fv_slot, trace, derived)
                        sub = _plan_slot_subgoal(
                            target_eid, fv_slot, fv_value, fv_sign, actual,
                            actor_id, trace, lex, rules, derived,
                            derivations, max_depth, depth, seen)
                        if sub is None:
                            sub_ok = False
                            break
                        sub_plans.extend(sub)
                if sub_ok:
                    # Guard against shadowed-default derivations.
                    # A trivially-conditioned derivation (e.g.
                    # `animate_has_thirst` whose only requirement is
                    # `entity(type="animate")`) yields `sub_plans=[]`
                    # — but if the derivation actually fired in the
                    # current derived state, the goal would already
                    # hold. When `value not in derived.get(...)`, an
                    # asserted-value (or a stronger override) is
                    # shadowing this derivation and the planner can't
                    # un-assert it. Returning [] would falsely claim
                    # the goal is met. Skip and try the next path.
                    if not sub_plans:
                        derived_vals = (derived.get(entity_id, slot)
                                        if derived is not None else None)
                        if derived_vals is None or value not in derived_vals:
                            continue
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
    from ..dsl.patterns import BindPattern
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

