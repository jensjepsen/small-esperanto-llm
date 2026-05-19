"""Forward state-space planner POC.

Alternative to `planner.py`'s backward chainer. Search goes from the
initial state outward: enumerate actions whose preconditions hold
NOW, apply each, recurse. Goal-directedness comes from a heuristic
`h(state)` that estimates remaining work; greedy best-first picks
the state with lowest h.

Why this exists alongside the backward chainer:
  - Backward chaining fights our event-calculus model (state advances
    via fired events; deriving "what state did action X need to fire"
    requires reasoning backward through derivations and effects).
  - Backward chaining accumulates "preserve constraint" complexity
    (do A, do B; if B breaks A, reject and try other order). The
    STRIPS-with-re-establishment limitation. Forward search has no
    such issue: we only ever fire actions whose preconditions hold
    in the current simulated state.
  - Failure semantics: when forward search runs out of applicable
    progress-making actions, the heuristic plateau is a clean signal.
    Backward chaining reports a misleading "deepest leaf" reason.

POC scope (first iteration):
  - Greedy best-first search (no A*, no admissibility worry — we
    don't care about plan optimality for training data; existence
    suffices).
  - h_add over a relaxed planning graph that treats derived facts
    (samloke, scias_lokon, ...) as first-class via the runtime
    derivation engine on each state.
  - Drive shapes: entity_slot and self_slot only for now.
  - Reuses Trace, _simulate_from_scratch, _find_role_filler from
    the existing planner — no domain code duplicated.
"""
from __future__ import annotations

import itertools
from typing import Optional

from ..causal import Trace
from ..fact_table import FactTable, fact_table_for, iter_bits
from .planner import (
    _cached_compute_derived_state, _entity_has_asserted_scalar,
    _entity_property_values, _find_role_filler, _has_relation,
    _simulate_from_scratch, _step_to_event,
)


# ---------- applicability ----------

def _applicable_actions(
    trace: Trace, lex, rules, derivations, derived,
    *, max_per_action: int = 4,
    relevant_entities: set | None = None,
    helpful: set | None = None,
):
    """Enumerate (verb, role_bindings) tuples whose preconditions
    hold in the current state. Yields one binding per role-fillable
    candidate, up to `max_per_action` per action to bound branching.

    A role binding is built by binding each of the action's roles to
    an in-trace entity matching the role spec (type + properties).
    Preconditions (relation, if_property, match) are checked against
    the resulting binding before yielding.
    """
    from ..schemas import (
        IfPropertyPrecondition, MatchPrecondition, RelationPrecondition,
    )

    for action in lex.actions.values():
        if not action.roles:
            continue
        # Per-role candidates. Each role's candidates are the
        # in-trace entities satisfying role_spec.type + properties.
        per_role: list[list[str]] = []
        for role_spec in action.roles:
            candidates: list[str] = []
            for eid, ent in trace.entities.items():
                if not lex.types.is_subtype(
                        ent.entity_type, role_spec.type):
                    continue
                ok = True
                for slot, vals in (role_spec.properties or {}).items():
                    values = _entity_property_values(
                        ent, slot, trace, derived)
                    if not values & set(vals):
                        ok = False
                        break
                if ok:
                    candidates.append(eid)
            if not candidates:
                # Some role can't be bound to any in-trace entity —
                # the whole action is inapplicable here.
                per_role = []
                break
            per_role.append(candidates)
        if not per_role:
            continue

        # Enumerate cross-products. Sort combos so helpful bindings
        # come first (so the max_per_action cap doesn't drop them).
        combos = []
        for combo in itertools.product(*per_role):
            if len(set(combo)) != len(combo):
                continue
            if relevant_entities is not None:
                if not any(e in relevant_entities for e in combo):
                    continue
            combos.append(combo)
        if helpful is not None:
            def _helpful_score(combo):
                roles = {r.name: e for r, e in zip(action.roles, combo)}
                return 0 if (action.lemma,
                              frozenset(roles.items())) in helpful else 1
            combos.sort(key=_helpful_score)
        emitted = 0
        for combo in combos:
            roles = {r.name: e for r, e in zip(action.roles, combo)}
            if not _preconditions_hold(
                    action, roles, trace, derived, lex):
                continue
            yield (action.lemma, roles)
            emitted += 1
            if emitted >= max_per_action:
                break


def _preconditions_hold(action, roles, trace, derived, lex) -> bool:
    """All action.preconditions satisfied by current state +
    derived facts. Mirrors `_resolve_preconditions_in_order`'s
    CHECK side, without the subgoaling branch.

    `scias_lokon` and `scias` ARE enforced; perception verbs (vidi/
    aŭdi/flari/montri) emit them directly so the fact-set search
    can plan through preni/kapti/veki's perception preconditions."""
    for pc in action.preconditions:
        if not _pc_holds(pc, roles, trace, derived, lex):
            return False
    return True


def _pc_holds(pc, roles, trace, derived, lex) -> bool:
    """Evaluate a single precondition. Recursive over OrPrecondition.
    Returns True iff the precondition is satisfied for the given role
    bindings."""
    from ..schemas import (
        HasPropertyPrecondition, IfPropertyPrecondition, MatchPrecondition,
        OrPrecondition, RelationPrecondition,
    )
    if isinstance(pc, RelationPrecondition):
        eids = tuple(roles.get(r) for r in pc.roles)
        if any(e is None for e in eids):
            # Unbound role — vacuous (matches IfPropertyPrecondition
            # at line 154 and the grounding-template's skip-on-None
            # at _ground_facts_from_template). Lets optional roles
            # carry preconditions that only fire when bound (e.g.
            # manĝi's havi(agent, instrument) when cutlery is used).
            return True
        return _has_relation(pc.rel, eids, trace, derived, lex)
    if isinstance(pc, IfPropertyPrecondition):
        eid = roles.get(pc.role)
        if eid is None:
            return True  # missing role — pc vacuously holds
        ent = trace.entities.get(eid)
        if ent is None:
            return False
        gate = _entity_property_values(
            ent, pc.if_property, trace, derived)
        if pc.if_value not in gate:
            return True  # Gate not active — pc vacuously holds.
        then_vals = _entity_property_values(
            ent, pc.then_property, trace, derived)
        return pc.then_value in then_vals
    if isinstance(pc, MatchPrecondition):
        ea = trace.entities.get(roles.get(pc.role_a))
        eb = trace.entities.get(roles.get(pc.role_b))
        if ea is None or eb is None:
            return False
        va = set(ea.properties.get(pc.slot_a, []))
        vb = set(eb.properties.get(pc.slot_b, []))
        return bool(va & vb)
    if isinstance(pc, HasPropertyPrecondition):
        eid = roles.get(pc.role)
        if eid is None:
            return False
        ent = trace.entities.get(eid)
        if ent is None:
            return False
        return pc.value in _entity_property_values(
            ent, pc.property, trace, derived)
    if isinstance(pc, OrPrecondition):
        return any(_pc_holds(alt, roles, trace, derived, lex)
                   for alt in pc.alternatives)
    return True  # unknown kind — vacuous (forward-compat)


# ---------- goal test ----------

def _goal_satisfied(goal, trace, derived, lex) -> bool:
    """Goal shapes:
      ("property", eid, slot, value)
      ("relation", relation, args_tuple)
    """
    kind = goal[0]
    if kind == "property":
        _, eid, slot, value = goal
        ent = trace.entities.get(eid)
        if ent is None:
            return False
        return value in _entity_property_values(
            ent, slot, trace, derived, lex)
    if kind == "relation":
        _, relation, args = goal
        return _has_relation(relation, args, trace, derived, lex)
    return False


# ---------- heuristic (h_add over relaxed planning graph) ----------
#
# Standard relaxed-planning-graph layered expansion. Each layer is a
# pair (fact set, action set):
#   layer 0 facts = state's facts ∪ derived facts.
#   layer k actions = grounded actions whose precondition facts ⊆
#                     layer k-1 facts.
#   layer k facts = layer k-1 facts ∪ effects of layer k actions.
# Iterate to fixed point or cap.
#
# Cost(fact) = first layer at which the fact appears.
# h_add(goal) = sum over goal literals of cost(literal). Inadmissible
# but informative, well-suited to greedy best-first.
#
# Encoding:
#   Facts are tuples — ("prop", eid, slot, value) for property
#   assertions, ("rel", relation, args) for relation assertions.
#
# Limitations (acceptable for POC, can be refined):
#   - Action grounding uses entities currently in the trace; the
#     relaxed graph doesn't model entity creation. Actions whose
#     effect target needs a not-yet-spawned entity get cost INF.
#   - if_property preconditions: gate-true cases treated as adding
#     the gate's then-fact as a precondition; gate-false cases are
#     treated as vacuous.
#   - match preconditions: treated as vacuous in the relaxation
#     (terrain compat is structural, doesn't bottleneck).
#   - Derivations: re-run after each layer so derived facts surface
#     naturally with their producer-fact's cost.

_HEURISTIC_INF = 10_000

# Cache the rule-effects index across heuristic calls in the same
# planning session. Keyed by id(lex) — the rules are loaded once and
# never change during a benchmark, so this is effectively a global
# memoization with safe invalidation on a fresh lexicon load.
_RULE_EFFECTS_CACHE: dict = {}

# Static (relation, arg_idx) → {slot: frozenset[forbidden_values]} index
# lifted from Relation.arg_patterns NotPattern shapes. Used to prune
# action groundings whose effects would produce a forbidden relation
# (havi(person, fajro) where fajro is nemovebla=yes) without
# constructing the trace. Same gate is enforced at runtime by
# Trace.validate_relation — single source of truth.
_REL_ARG_EXCLUDES_CACHE: dict = {}


def _filter_forbidden_effs(effs, trace, lex):
    """Return `effs` with any ("rel", name, args) fact removed when it
    would produce a relation forbidden by `relation_arg_excludes`
    (Relation.arg_patterns NotPattern shapes). Mirrors the runtime
    engine's behavior at engine.py:778, which swallows ValueError on
    AddRelation — that effect silently fails at runtime, so the
    planner should plan as if it weren't there. Other effects on the
    same action are preserved, so e.g. `fari(muro)` retains its
    havas_parton attachments while dropping the silently-failing
    havi(agent, muro). When the filtered effs is empty the caller's
    existing "no-effect actions are skipped" guard handles it.

    Cheap O(|effs| * |arg-positions-with-forbids|) lookup; typical
    effs sets are < 10."""
    excludes = _REL_ARG_EXCLUDES_CACHE.get(id(lex))
    if excludes is None:
        from ..dsl.introspect import relation_arg_excludes
        excludes = relation_arg_excludes(lex)
        _REL_ARG_EXCLUDES_CACHE[id(lex)] = excludes
    out = set()
    from ..dsl.patterns import numeric_args_compare
    for fact in effs:
        if fact[0] != "rel":
            out.add(fact)
            continue
        _, rname, args = fact
        forbidden_hit = False
        # arg_patterns set-membership (nemovebla etc.)
        if excludes:
            for i, eid in enumerate(args):
                slot_map = excludes.get((rname, i))
                if not slot_map:
                    continue
                ent = trace.entities.get(eid)
                if ent is None:
                    continue
                for slot, forbidden in slot_map.items():
                    vals = ent.properties.get(slot, ())
                    if any(v in forbidden for v in vals):
                        forbidden_hit = True
                        break
                if forbidden_hit:
                    break
        # arg_compare numeric checks (carry capacity etc.)
        if not forbidden_hit:
            rel = lex.relations.get(rname)
            if rel is not None and rel.arg_compare:
                ents = tuple(trace.entities.get(e) for e in args)
                if all(e is not None for e in ents):
                    for spec in rel.arg_compare:
                        if not numeric_args_compare(ents, spec):
                            forbidden_hit = True
                            break
        if not forbidden_hit:
            out.add(fact)
    return out

# Per-grounded_derivations cache for the pres-fact inverted index
# used by _apply_delta's worklist re-derivation. Keyed by
# id(derivation_pseudos) — caller is plan_for_goal which builds
# grounded_derivs once per plan and reuses for ~hundreds of
# _apply_delta calls during search.
_DERIV_PRES_IDX_CACHE: dict = {}


def _compile_action_template(action, rule_effects, sym, forbid_rels=None):
    """Decompose `action`'s structural pres/effs once so the per-combo
    `_ground_action_facts` call avoids isinstance dispatch on every
    invocation. Returns a `_CompiledAction` carrying:
      * `prop_pres` — list of (role_name, slot, value0) for the first
        property in each role.properties entry (matches the legacy
        `vals[0]` behavior).
      * `rel_pres_simple` — list of (rel, role_names_tuple): pres-rels
        whose roles all bind to scalar entities.
      * `rel_pres_list` — list of (rel, roles_tuple, list_role_index):
        pres-rels with one role bound to a list (fari.parts).
      * `eff_props` — list of (target_role, property, value).
      * `rule_adds_simple` — list of (relation, role_arg_names_tuple)
        for rule-effect adds with NO markers; the per-combo path can
        substitute directly without calling `_expand_list_role_args`.
      * `rule_adds_marker` — list of (relation, role_arg_names) for
        adds that contain `<list>`, `<lookup>`, or `<literal>` markers
        and must go through `_expand_list_role_args`.
      * `sym` — symmetric-relation frozenset, passed through so the
        per-combo path doesn't redo the lookup."""
    from ..schemas import (
        IfPropertyPrecondition, MatchPrecondition, RelationPrecondition,
    )
    prop_pres: list = []
    rel_pres_simple: list = []
    rel_pres_list: list = []
    for role_spec in action.roles:
        rk = getattr(role_spec, "kind", "single")
        if rk in ("list", "relation"):
            continue
        if role_spec.properties:
            for slot, vals in role_spec.properties.items():
                if vals:
                    prop_pres.append((role_spec.name, slot, vals[0]))
    list_role_names: set = {
        r.name for r in action.roles
        if getattr(r, "kind", "single") == "list"}
    for pc in action.preconditions:
        if isinstance(pc, RelationPrecondition):
            pc_roles = tuple(pc.roles)
            # Find list-role position (at most one per pc).
            list_pos = None
            for i, rn in enumerate(pc_roles):
                if rn in list_role_names:
                    list_pos = i
                    break
            if list_pos is not None:
                rel_pres_list.append((pc.rel, pc_roles, list_pos))
            else:
                rel_pres_simple.append((pc.rel, pc_roles))
        # IfPropertyPrecondition / MatchPrecondition skipped in relaxed encoding.
    eff_props: list = [
        (eff.target_role, eff.property, eff.value)
        for eff in action.effects]
    rule_adds_simple: list = []
    rule_adds_marker: list = []
    entry = rule_effects.get(action.lemma)
    if entry is not None:
        for relation, role_arg_names in entry["adds"]:
            has_markers = any(isinstance(a, tuple) for a in role_arg_names)
            if has_markers:
                rule_adds_marker.append((relation, role_arg_names))
            else:
                rule_adds_simple.append((relation, tuple(role_arg_names)))
    # Forbid-skip flag: if none of the rule-add relations can ever
    # appear in `excludes`/`arg_compare`, the per-combo
    # `_filter_forbidden_effs` call is pure overhead and can be
    # bypassed. (Schema-level eff_props produce only "prop" facts,
    # which the filter passes through unchanged.)
    skip_filter = True
    if forbid_rels:
        for relation, _ in rule_adds_simple:
            if relation in forbid_rels:
                skip_filter = False
                break
        if skip_filter:
            for relation, _ in rule_adds_marker:
                if relation in forbid_rels:
                    skip_filter = False
                    break
    entry = rule_effects.get(action.lemma) or {}
    concept_constraints = tuple(entry.get("concept_constraints", ()))
    return _CompiledAction(
        prop_pres=prop_pres,
        rel_pres_simple=rel_pres_simple,
        rel_pres_list=rel_pres_list,
        eff_props=eff_props,
        rule_adds_simple=rule_adds_simple,
        rule_adds_marker=rule_adds_marker,
        sym=sym,
        skip_filter=skip_filter,
        concept_constraints=concept_constraints,
    )


class _CompiledAction:
    """Slotted holder for `_compile_action_template`'s output. Pure
    data, no methods — the substitution loop lives in the fast
    `_ground_facts_from_template` helper."""
    __slots__ = (
        "prop_pres", "rel_pres_simple", "rel_pres_list",
        "eff_props", "rule_adds_simple", "rule_adds_marker", "sym",
        "skip_filter", "concept_constraints")

    def __init__(self, prop_pres, rel_pres_simple, rel_pres_list,
                 eff_props, rule_adds_simple, rule_adds_marker, sym,
                 skip_filter=False, concept_constraints=()):
        self.prop_pres = prop_pres
        self.rel_pres_simple = rel_pres_simple
        self.rel_pres_list = rel_pres_list
        self.eff_props = eff_props
        self.rule_adds_simple = rule_adds_simple
        self.rule_adds_marker = rule_adds_marker
        self.sym = sym
        self.skip_filter = skip_filter
        # List of (entity_role, source_role, field_name) tuples lifted
        # from rule given clauses: per-combo, entity's concept must
        # model the slot named by source.concept.properties[field][0].
        # Empty for actions whose rules don't use the
        # has_concept_field + concept_models_slot_check pattern.
        self.concept_constraints = concept_constraints


def _ground_facts_from_template(tmpl, roles, facts):
    """Per-combo fast path: apply `_CompiledAction`'s prebaked
    pres/effs templates against `roles`. Avoids isinstance dispatch
    on action.preconditions / action.effects / role.properties — those
    were classified once at template-compile time."""
    pres: set = set()
    effs: set = set()
    sym = tmpl.sym
    roles_get = roles.get
    # Role property pres.
    for role_name, slot, value in tmpl.prop_pres:
        eid = roles_get(role_name)
        if eid is not None:
            pres.add(("prop", eid, slot, value))
    # Simple rel pres (no list-role expansion).
    for rel, pc_roles in tmpl.rel_pres_simple:
        eids = tuple(roles_get(r) for r in pc_roles)
        if not any(e is None for e in eids):
            pres.add(("rel", rel, _canon_rel(rel, eids, sym)))
    # List-role rel pres (fari.parts).
    for rel, pc_roles, list_pos in tmpl.rel_pres_list:
        lrole = pc_roles[list_pos]
        lv = roles_get(lrole)
        if not isinstance(lv, (list, tuple)):
            continue
        for item in lv:
            eids = tuple(
                item if i == list_pos else roles_get(r)
                for i, r in enumerate(pc_roles))
            if not any(e is None for e in eids):
                pres.add(("rel", rel, _canon_rel(rel, eids, sym)))
    # Schema-level effects.
    for target_role, prop, value in tmpl.eff_props:
        eid = roles_get(target_role)
        if eid is not None:
            effs.add(("prop", eid, prop, value))
    # Rule-add effects: simple (no markers) → direct substitution.
    for relation, role_arg_names in tmpl.rule_adds_simple:
        eids = tuple(roles_get(r) for r in role_arg_names)
        if not any(e is None for e in eids):
            effs.add(("rel", relation, _canon_rel(relation, eids, sym)))
    # Rule-add effects: marker-bearing → use full expander.
    for relation, role_arg_names in tmpl.rule_adds_marker:
        for eids in _expand_list_role_args(
                role_arg_names, roles, facts=facts):
            if not any(e is None for e in eids):
                effs.add(
                    ("rel", relation, _canon_rel(relation, eids, sym)))
    return pres, effs


def _ground_action_facts(action, roles, lex, rule_effects, facts=None):
    """Return (precondition_facts, effect_facts) for a grounded
    action — both as sets of fact tuples. `rule_effects` is the
    `{verb: [(relation, role_arg_names)]}` index of rule-added
    relations (preni adds havi, vidi adds scias, etc.) — without
    these the relaxed graph misses key transitions and the
    heuristic returns INF for any chain that depends on them.

    `facts` (optional) is the initial state fact set used to resolve
    `<lookup>` markers in adds (given-bound vars like fari's agent
    location). Without it, lookup-marked adds are skipped."""
    from ..schemas import (
        IfPropertyPrecondition, MatchPrecondition, RelationPrecondition,
    )
    pres: set = set()
    effs: set = set()
    sym = _symmetric_relations(lex)
    # Role property requirements: pre `("prop", eid, slot, value)`.
    # Skip list-kind roles — their property requirements would need to
    # apply to each list element, which the spawner enforces at scene
    # time (each ingredient already matches its role.properties).
    for role_spec in action.roles:
        if getattr(role_spec, "kind", "single") in ("list", "relation"):
            continue
        eid = roles.get(role_spec.name)
        if eid is None:
            continue
        for slot, vals in (role_spec.properties or {}).items():
            if vals:
                pres.add(("prop", eid, slot, vals[0]))
    # Action-level preconditions.
    for pc in action.preconditions:
        if isinstance(pc, RelationPrecondition):
            # List-role expansion: if any precondition role name maps
            # to a list-valued role binding (fari.parts), emit one
            # precondition per element. Optional roles (instrument
            # when crafted_with is empty) drop out via None binding.
            list_roles = [r for r in pc.roles
                           if isinstance(roles.get(r), (list, tuple))]
            if list_roles:
                # Only support one list role per precondition for now
                # — fari's preconditions have at most parts as a list.
                lrole = list_roles[0]
                list_pos = pc.roles.index(lrole)
                for item in roles[lrole]:
                    eids = tuple(
                        item if i == list_pos else roles.get(r)
                        for i, r in enumerate(pc.roles))
                    if any(e is None for e in eids):
                        continue
                    pres.add(("rel", pc.rel, _canon_rel(pc.rel, eids, sym)))
            else:
                eids = tuple(roles.get(r) for r in pc.roles)
                if any(e is None for e in eids):
                    continue
                pres.add(("rel", pc.rel, _canon_rel(pc.rel, eids, sym)))
        elif isinstance(pc, IfPropertyPrecondition):
            # Skipped in relaxed encoding. The gate is conditional
            # — fires only when `if_property=if_value` holds on the
            # bound entity — and the entity may not even carry the
            # gate's slot (eniri's openness gate on ĝardeno, which
            # has no openness; or sekigi's reflexive gate). Modeling
            # this conditionally in the relaxed graph requires
            # gate-state tracking; for the POC the planner's
            # _preconditions_hold check at applicability time still
            # enforces the real gate semantics, so plans firing
            # closed doors get caught there.
            pass
        # MatchPrecondition skipped.
    # Schema-level effects (property writes).
    for eff in action.effects:
        eid = roles.get(eff.target_role)
        if eid is None:
            continue
        effs.add(("prop", eid, eff.property, eff.value))
    # Rule-level effects: only 'adds' for the relaxed graph (delete
    # relaxation). The fact-set incremental simulator uses the same
    # index but reads 'dels' too.
    entry = rule_effects.get(action.lemma)
    if entry is not None:
        for relation, role_arg_names in entry["adds"]:
            for eids in _expand_list_role_args(
                    role_arg_names, roles, facts=facts):
                if any(e is None for e in eids):
                    continue
                effs.add(("rel", relation, _canon_rel(relation, eids, sym)))
    return pres, effs


def _resolve_lookup(rel_name, template, roles, facts):
    """Find facts matching `("rel", rel_name, args)` where args agree
    with template at every non-None position (resolved via roles),
    and return the values at the None position. Mirrors the dels-
    wildcard resolution in _action_delta_mask."""
    extract_idx = template.index(None)
    pattern = tuple(roles.get(t) if t is not None else None
                    for t in template)
    out: list = []
    for f in facts:
        if (f[0] == "rel" and f[1] == rel_name
                and len(f[2]) == len(pattern)
                and all(p is None or p == a
                        for p, a in zip(pattern, f[2]))):
            out.append(f[2][extract_idx])
    return out


def _expand_list_role_args(role_arg_names, roles, facts=None):
    """Yield one eid tuple per resolved combination of marker args:

    - bare role name → single value from `roles`
    - `("<list>", role)` → element-wise zip across list positions
      (fari's variadic havas_parton / havi-detach effects, one fact
      per part).
    - `("<lookup>", relation, args_template)` → cross-product across
      lookup positions (given-bound vars in adds; resolved against
      `facts`). Yields nothing if `facts is None` or no fact matches
      — the rule's `given` doesn't hold, so the effect is dropped.

    If no list/lookup markers are present, yields a single tuple."""
    # Fast path: no markers at all (the overwhelmingly common case).
    if not any(isinstance(a, tuple) for a in role_arg_names):
        yield tuple(roles.get(r) for r in role_arg_names)
        return
    n = len(role_arg_names)
    per_pos: list = [None] * n
    is_list = [False] * n
    is_lookup = [False] * n
    for i, arg in enumerate(role_arg_names):
        if isinstance(arg, tuple) and arg:
            kind = arg[0]
            if kind == "<list>":
                lv = roles.get(arg[1])
                if lv is None or not isinstance(lv, (list, tuple)):
                    return
                per_pos[i] = list(lv)
                is_list[i] = True
                continue
            if kind == "<lookup>":
                if facts is None:
                    return
                vals = _resolve_lookup(arg[1], arg[2], roles, facts)
                if not vals:
                    return
                per_pos[i] = list(vals)
                is_lookup[i] = True
                continue
            if kind == "<literal>":
                # Literal-string arg from a rule's AddRelation (e.g.
                # the "en"/"sur"/"havi" rel_type in scias). No
                # substitution needed — the value is the marker's
                # payload itself.
                per_pos[i] = [arg[1]]
                continue
        per_pos[i] = [roles.get(arg)]
    list_len = max(
        (len(per_pos[i]) for i in range(n) if is_list[i]),
        default=1)
    lookup_indices = [i for i in range(n) if is_lookup[i]]
    if lookup_indices:
        from itertools import product
        lookup_combos = list(product(
            *(per_pos[i] for i in lookup_indices)))
    else:
        lookup_combos = [()]
    for k in range(list_len):
        for combo in lookup_combos:
            out: list = []
            for i in range(n):
                if is_list[i]:
                    out.append(
                        per_pos[i][k] if k < len(per_pos[i]) else None)
                elif is_lookup[i]:
                    out.append(combo[lookup_indices.index(i)])
                else:
                    out.append(per_pos[i][0])
            yield tuple(out)


def _build_initial_evidence_pools(
    trace, lex, rule_effects, relevant_entities,
) -> dict:
    """Per-rel evidence pool: set of canonical arg-tuples that could
    satisfy a `rel(R, ...)` pattern given the current state +
    rule-effects synthesis. Sources:

      1. Real tuples in `trace.relations` (canonicalized for symmetric
         rels). These are concrete evidence.
      2. Producible tuples from `rule_effects.adds` — each verb whose
         effect adds R contributes one tuple per type-compatible role
         binding. Mirrors `_fp_tuple_pool`'s rule-effects half.

    Derivation outputs (e.g. samloke from `en_implies_…`) are NOT
    seeded here — they accumulate during the two-pass grounding loop
    in `_ground_derivations` (seeds first, chains second).

    Constrained to `relevant_entities` when supplied. Used by
    `_evidence_join_combos` to drive evidence-based grounding for
    multi-rel derivations, replacing the cubic var-domain cartesian
    that dominated `_ground_derivations` cost for samloke chains."""
    sym = _symmetric_relations(lex)
    pools: dict[str, set] = {}

    def _ent_ok(eid):
        if eid == "mondo":
            return False
        ent = trace.entities.get(eid)
        if ent is None or ent.entity_type == "inanimate":
            return False
        if relevant_entities is not None and eid not in relevant_entities:
            return False
        return True

    def _add(rel_name, tup):
        for a in tup:
            if not _ent_ok(a):
                return
        pools.setdefault(rel_name, set()).add(
            _canon_rel(rel_name, tup, sym))

    for r in trace.relations:
        _add(r.relation, tuple(r.args))

    from ..entity_index import entity_index_for
    entity_idx = entity_index_for(trace, lex)
    for verb, entry in rule_effects.items():
        action_obj = lex.actions.get(verb)
        if action_obj is None:
            continue
        role_cands: dict = {}
        for rs in action_obj.roles:
            rk = getattr(rs, "kind", "single")
            if rk in ("list", "relation", "from_precondition"):
                continue
            cands = [eid for eid in entity_idx.entities_matching(rs.type)
                     if _ent_ok(eid)]
            role_cands[rs.name] = cands
        for rel, role_args in entry.get("adds", []):
            per_pos: list = []
            ok = True
            for arg in role_args:
                if isinstance(arg, str) and arg in role_cands:
                    pool = role_cands[arg]
                    if not pool:
                        ok = False
                        break
                    per_pos.append(pool)
                else:
                    # Skip <lookup>/<literal>/list markers — out of
                    # scope for the prototype; the cartesian
                    # fallback handles derivations with such args.
                    ok = False
                    break
            if not ok:
                continue
            for tup in itertools.product(*per_pos):
                if len(set(tup)) != len(tup):
                    continue
                _add(rel, tup)
    return pools


def _pool_position_index(pool, rel_name, sym) -> list:
    """Per-position index of `pool`: list of dicts mapping `eid` →
    list of tuples that have `eid` at that position. Used by
    `_evidence_join_combos` to look up "tuples consistent with a
    bound var" in O(matches), not O(pool).

    For symmetric arity-2 relations, args are stored canonically
    (sorted). To allow lookup by either argument, both orderings
    are indexed; the join consumer treats this transparently."""
    if not pool:
        return []
    arity = len(next(iter(pool)))
    idx: list = [dict() for _ in range(arity)]
    for tup in pool:
        for pos, eid in enumerate(tup):
            idx[pos].setdefault(eid, []).append(tup)
        if rel_name in sym and arity == 2 and tup[0] != tup[1]:
            # Index reverse ordering so lookup-by-either-arg works.
            rev = (tup[1], tup[0])
            idx[0].setdefault(rev[0], []).append(rev)
            idx[1].setdefault(rev[1], []).append(rev)
    return idx


def _evidence_join_combos(
    rel_patterns, rel_arg_vids, pool_indices, pools_all_tuples,
    var_constraints, var_exclusions,
    trace, lex, _entity_matches_constraints,
):
    """Generator over bindings (dict[vid → eid]) satisfying ALL
    `rel_patterns` via a left-deep relational join. Replaces the
    cartesian-over-var-domains enumeration when a derivation has
    multiple rel-patterns and all of their relations have non-empty
    evidence pools.

    Each step picks a rel_pattern; if any of its arg_vids is already
    bound, look up only tuples consistent with the bound eid (O(k)
    via `pool_indices`); otherwise iterate the rel's full pool. Per-
    var type/concept/NotPattern constraints are enforced as each var
    is bound — mirrors the per-entity filter in the cartesian path.

    Order: smallest pool first to maximize early pruning."""
    if not rel_patterns:
        return
    sizes = [(len(pools_all_tuples[i]), i) for i in range(len(rel_patterns))]
    sizes.sort()
    order = [i for _, i in sizes]

    def _recurse(step, binding):
        if step == len(order):
            yield dict(binding)
            return
        rp_idx = order[step]
        rp = rel_patterns[rp_idx]
        arg_vids = rel_arg_vids[rp_idx]
        idx = pool_indices.get(rp.relation, [])
        # Pick most-constrained lookup: any arg_vid already bound
        # narrows the candidate list to O(matches) via the index.
        candidates = None
        for pos, av in enumerate(arg_vids):
            anchor_eid = None
            if isinstance(av, tuple) and av and av[0] == "<lit>":
                anchor_eid = av[1]
            elif not isinstance(av, tuple) and av in binding:
                anchor_eid = binding[av]
            if anchor_eid is None:
                continue
            if idx and pos < len(idx):
                candidates = idx[pos].get(anchor_eid, ())
                break
        if candidates is None:
            candidates = pools_all_tuples[rp_idx]
        for tup in candidates:
            new_binding = dict(binding)
            ok = True
            for av, eid in zip(arg_vids, tup):
                if isinstance(av, tuple):
                    if av and av[0] == "<lit>" and av[1] != eid:
                        ok = False
                        break
                    continue
                if av in new_binding:
                    if new_binding[av] != eid:
                        ok = False
                        break
                    continue
                type_, concept_, _props = var_constraints.get(
                    av, (None, None, ()))
                ent = trace.entities.get(eid)
                if ent is None:
                    ok = False
                    break
                if type_ is not None and not lex.types.is_subtype(
                        ent.entity_type, type_):
                    ok = False
                    break
                if concept_ is not None and ent.concept_lemma != concept_:
                    ok = False
                    break
                excluded = False
                for excl in var_exclusions.get(av, ()):
                    if _entity_matches_constraints(ent, excl, lex):
                        excluded = True
                        break
                if excluded:
                    ok = False
                    break
                new_binding[av] = eid
            if not ok:
                continue
            yield from _recurse(step + 1, new_binding)

    yield from _recurse(0, {})


def _ground_derivations(
    derivations, trace, lex,
    relevant_entities: set | None = None,
    rule_effects: dict | None = None,
) -> list:
    """Compile derivations into grounded pseudo-actions for the
    relaxed graph. Each binding of a derivation's variables to
    entities becomes one pseudo-action:
      pres = facts from rel patterns in when+given (with synthesized
             vars for inline EntityPattern args) + property constraints
             from EntityPattern constraints
      effs = facts from implies clause

    `relevant_entities` (optional) restricts var domains to a subset of
    trace.entities. Used by the heuristic to cut the cubic samloke-
    chain blow-up: with 19 derivation-relevant entities the four
    samloke derivations alone generate 23k bindings (19³ each);
    restricting to action-mentioned entities (typically 6-10) cuts
    that 5-10×.

    Pseudo-actions are cost-0 — derived facts get the same layer as
    their producing fact, modeling "if X holds, samloke(X, Y) also
    holds, instantly". Skips NotPatterns (relaxation drops negation).

    Reuses existing planner helpers (`_walk_for_rel_patterns`,
    `_extract_bind_vars`) so the pattern walking matches the live
    engine's interpretation."""
    from ..dsl.implications import (
        PropertyImplication, RelationImplication,
    )
    from ..dsl.patterns import (
        AndPattern, BindPattern, EntityPattern, Var,
    )
    from .planner import (
        _walk_for_rel_patterns, _extract_bind_vars,
    )
    sym_local = _symmetric_relations(lex)

    def _bind_target_var(arg_pat):
        """If arg_pat is a Var, returns it. If it's a BindPattern,
        returns its target. Otherwise None — including for inline
        EntityPatterns with no Var binding."""
        if isinstance(arg_pat, Var):
            return arg_pat
        if isinstance(arg_pat, BindPattern):
            return arg_pat.target
        if isinstance(arg_pat, AndPattern):
            # Common shape: EntityPattern(...) & bind(Var)
            for side in (arg_pat.left, arg_pat.right):
                if isinstance(side, BindPattern):
                    return side.target
                if isinstance(side, Var):
                    return side
        return None

    def _entity_pattern_for(arg_pat):
        """Find the EntityPattern inside arg_pat, if any."""
        if isinstance(arg_pat, EntityPattern):
            return arg_pat
        if isinstance(arg_pat, AndPattern):
            for side in (arg_pat.left, arg_pat.right):
                ep = _entity_pattern_for(side)
                if ep is not None:
                    return ep
        if isinstance(arg_pat, BindPattern):
            return _entity_pattern_for(arg_pat.pattern)
        return None

    def _ep_for_var(target_var, when, given):
        """Find an EntityPattern co-bound with target_var anywhere
        under when+given. Returns the EntityPattern or None."""
        for pat in [when] + list(given):
            ep = _walk_for_ep_binding_local(pat, target_var)
            if ep is not None:
                return ep
        return None

    def _walk_for_ep_binding_local(pattern, target_var):
        if isinstance(pattern, AndPattern):
            # EntityPattern(...) & bind(Var)
            if (isinstance(pattern.left, EntityPattern)
                    and isinstance(pattern.right, BindPattern)
                    and pattern.right.target is target_var):
                return pattern.left
            if (isinstance(pattern.right, EntityPattern)
                    and isinstance(pattern.left, BindPattern)
                    and pattern.left.target is target_var):
                return pattern.right
            for side in (pattern.left, pattern.right):
                ep = _walk_for_ep_binding_local(side, target_var)
                if ep is not None:
                    return ep
        # Descend into RelPattern arg_patterns — Vars co-bound with
        # EntityPatterns inside `rel("en", container=entity(...) & bind(B))`
        # would otherwise be invisible to the walker.
        from ..dsl.patterns import RelPattern
        if isinstance(pattern, RelPattern):
            for arg_pat in pattern.arg_patterns.values():
                ep = _walk_for_ep_binding_local(arg_pat, target_var)
                if ep is not None:
                    return ep
        return None

    def _flatten_conj(pat) -> list:
        """Flatten an AndPattern tree into a list of non-And leaf
        patterns. `a & b & c` → [a, b, c]. Used to find sibling
        constraints of a `bind(v)` within one rel-arg conjunction."""
        if isinstance(pat, AndPattern):
            return _flatten_conj(pat.left) + _flatten_conj(pat.right)
        return [pat]

    def _exclusions_for_var(target_var, when, given) -> list[dict]:
        """Collect NotPattern(EntityPattern(...)) conjuncts that sit
        alongside `bind(target_var)` within the same rel-arg
        conjunction. Lets the samloke chain's `~entity(type="location")`
        actually narrow the bridge var.

        Only structural fields (type, concept, category, suffix) get
        collected — those are immutable for the life of an entity, so
        excluding them at grounding time is sound. Property-based
        NotPatterns (`~entity(openness="fermita")`) reference mutable
        slots and applying them at grounding wrongly bakes the
        INITIAL value into the relaxed graph: a fermita ŝranko then
        never participates in any samloke-chain pseudo-action, even
        when the plan eventually fires malfermi on it. The relaxed
        graph drops negative preconditions per standard relaxation
        anyway, so over-estimating reachability through mutable
        NotPatterns is what we want; the actual planner enforces
        them at action time."""
        from ..dsl.patterns import NotPattern, RelPattern
        structural_keys = ("type", "concept", "category", "suffix")
        out: list[dict] = []
        for pat in [when] + list(given):
            for rp in _walk_for_rel_patterns(pat):
                for arg_pat in rp.arg_patterns.values():
                    conjuncts = _flatten_conj(arg_pat)
                    has_target_bind = any(
                        isinstance(c, BindPattern)
                        and c.target is target_var
                        for c in conjuncts)
                    if not has_target_bind:
                        continue
                    for c in conjuncts:
                        if not (isinstance(c, NotPattern)
                                and isinstance(c.inner, EntityPattern)):
                            continue
                        structural_only = {
                            k: v for k, v in c.inner.constraints.items()
                            if k in structural_keys
                        }
                        if structural_only:
                            out.append(structural_only)
        return out

    def _entity_matches_constraints(ent, constraints, lex) -> bool:
        """Does ent satisfy ALL literal type/concept/slot constraints
        in the dict? Mirrors the matcher in dsl/patterns.py but without
        Var or has_suffix support (negated exclusions use literals)."""
        for k, v in constraints.items():
            if isinstance(v, Var):
                continue
            if k == "type":
                if not lex.types.is_subtype(ent.entity_type, v):
                    return False
            elif k == "concept":
                if ent.concept_lemma != v:
                    return False
            elif k in lex.slots:
                vals = ent.properties.get(k, [])
                if isinstance(v, (list, tuple)):
                    if not any(x in vals for x in v):
                        return False
                else:
                    if v not in vals:
                        return False
        return True

    def _constraints_from_ep(ep, lex):
        """Pull literal type/concept/property constraints from an
        EntityPattern.constraints dict. Returns
        (type_, concept_, prop_pairs)."""
        type_ = None
        concept_ = None
        props: list = []  # (slot, value)
        if ep is None:
            return type_, concept_, props
        for k, v in ep.constraints.items():
            if isinstance(v, Var):
                continue
            if k == "type":
                type_ = v
            elif k == "concept":
                concept_ = v
            elif k in lex.slots:
                # Property constraint
                if isinstance(v, (list, tuple)):
                    if v:
                        # Multi-value — relaxation: take the first.
                        props.append((k, v[0]))
                else:
                    props.append((k, v))
        return type_, concept_, props

    out = []
    next_synth_id = [-1_000_000]

    def _new_synth_var(constraints):
        next_synth_id[0] -= 1
        return next_synth_id[0]

    # Evidence pools: real trace facts + rule_effects-producible
    # tuples, accumulated with derivation outputs as we ground in
    # dependency order. Multi-rel derivations (samloke chains,
    # shared_*-samloke) use these to drive a left-deep relational
    # join instead of the var-domain cartesian. Falls back to
    # cartesian if rule_effects isn't supplied (legacy callers).
    pools: dict | None = None
    pool_indices: dict = {}
    if rule_effects is not None:
        pools = _build_initial_evidence_pools(
            trace, lex, rule_effects, relevant_entities)
        for rname, p in pools.items():
            pool_indices[rname] = _pool_position_index(p, rname, sym_local)

    def _initial_pool_for(d) -> bool:
        """True if every rel-pattern in d's when+given refers to a
        relation already present in the initial pool. Used to order
        derivations so chains run AFTER their seed-derivation inputs
        have populated the pool."""
        if pools is None:
            return True
        for pat in [d.when] + list(d.given):
            for rp in _walk_for_rel_patterns(pat):
                if rp.relation not in pools or not pools[rp.relation]:
                    return False
        return True

    if pools is not None:
        # Stable two-pass: seeds first, then chains.
        derivations = (
            [d for d in derivations if _initial_pool_for(d)]
            + [d for d in derivations if not _initial_pool_for(d)])

    def _augment_pools_with_eff(eff_rel, eff_args):
        """Add a derivation-produced rel tuple to pools/index so
        downstream chain derivations can use it as evidence."""
        if pools is None:
            return
        canon = _canon_rel(eff_rel, eff_args, sym_local)
        pool = pools.setdefault(eff_rel, set())
        if canon in pool:
            return
        pool.add(canon)
        idx = pool_indices.get(eff_rel)
        if idx is None or (canon and len(idx) != len(canon)):
            pool_indices[eff_rel] = _pool_position_index(
                pool, eff_rel, sym_local)
            return
        for pos, eid in enumerate(canon):
            idx[pos].setdefault(eid, []).append(canon)
        if (eff_rel in sym_local and len(canon) == 2
                and canon[0] != canon[1]):
            rev = (canon[1], canon[0])
            idx[0].setdefault(rev[0], []).append(rev)
            idx[1].setdefault(rev[1], []).append(rev)

    for d in derivations:
        # All explicit Vars used in when+given+implies.
        all_vars: dict = {}  # id(Var) -> Var (or synth-int)
        var_constraints: dict = {}  # id -> (type, concept, [(slot, val)])

        for v in _extract_bind_vars(d.when):
            all_vars[id(v)] = v
        for g in d.given:
            for v in _extract_bind_vars(g):
                all_vars[id(v)] = v
        for imp in d.implies:
            if isinstance(imp, PropertyImplication):
                if isinstance(imp.entity, Var):
                    all_vars[id(imp.entity)] = imp.entity
            elif isinstance(imp, RelationImplication):
                for a in imp.args:
                    if isinstance(a, Var):
                        all_vars[id(a)] = a

        # Gather constraints for each explicit Var.
        var_exclusions: dict = {}
        for vid, v in all_vars.items():
            ep = _ep_for_var(v, d.when, d.given)
            var_constraints[vid] = _constraints_from_ep(ep, lex)
            # NotPattern siblings of bind(v) — entities matching these
            # are excluded from v's domain. Critical for the samloke
            # chains' `~entity(type="location")` to actually narrow B.
            var_exclusions[vid] = _exclusions_for_var(v, d.when, d.given)

        # RelPatterns in when+given. For each rel-arg that's an
        # inline EntityPattern (no Var), synthesize a free Var so we
        # ground over its constraints too.
        rel_patterns: list = []
        # For each rel pattern, keep arg → vid mapping (synth or real).
        rel_arg_vids: list = []
        for pat in [d.when] + list(d.given):
            for rp in _walk_for_rel_patterns(pat):
                rel_def = lex.relations.get(rp.relation)
                if rel_def is None:
                    continue
                from ..dsl.patterns import LiteralValuePattern
                arg_vids: list = []
                ok = True
                for arg_name in rel_def.arg_names:
                    arg_pat = rp.arg_patterns.get(arg_name)
                    if arg_pat is None:
                        ok = False
                        break
                    # Literal-value constraint (e.g.
                    # rel("scias", rel_type="en", ...)) — encode as
                    # ("<lit>", value) so the pres-construction loop
                    # substitutes the literal directly without trying
                    # to look it up as an entity binding.
                    if isinstance(arg_pat, LiteralValuePattern):
                        arg_vids.append(("<lit>", arg_pat.value))
                        continue
                    tgt = _bind_target_var(arg_pat)
                    if tgt is not None:
                        arg_vids.append(id(tgt))
                        if id(tgt) not in all_vars:
                            # Var referenced but not previously bound;
                            # treat as free.
                            all_vars[id(tgt)] = tgt
                            ep_inline = _entity_pattern_for(arg_pat)
                            var_constraints[id(tgt)] = (
                                _constraints_from_ep(ep_inline, lex))
                    else:
                        # Inline EntityPattern with no Var binding.
                        ep_inline = _entity_pattern_for(arg_pat)
                        synth_id = _new_synth_var(None)
                        all_vars[synth_id] = None  # synth marker
                        var_constraints[synth_id] = (
                            _constraints_from_ep(ep_inline, lex))
                        arg_vids.append(synth_id)
                if not ok:
                    continue
                rel_patterns.append(rp)
                rel_arg_vids.append(arg_vids)

        # Implies → effects (only Var-bound entities).
        impl_specs: list = []
        for imp in d.implies:
            if isinstance(imp, PropertyImplication):
                if not isinstance(imp.entity, Var):
                    continue
                if isinstance(imp.value, Var):
                    continue
                impl_specs.append(("prop", imp.entity, imp.slot, imp.value))
            elif isinstance(imp, RelationImplication):
                impl_specs.append(("rel", imp.name, tuple(imp.args)))
        if not impl_specs:
            continue

        # Ground each Var to entities matching its constraints.
        # Body parts (inanimate sub-entities like virino_piedo) and
        # the mondo singleton aren't goal-relevant for plan search;
        # excluding them collapses the grounding count by ~10× on
        # typical scenes (was 42k samloke groundings → ~4k).
        domains: list = []
        var_ids = list(all_vars.keys())
        for vid in var_ids:
            type_, concept_, _props = var_constraints[vid]
            exclusions = var_exclusions.get(vid, [])
            cands = []
            for eid, ent in trace.entities.items():
                if eid == "mondo":
                    continue
                if ent.entity_type == "inanimate":
                    continue  # body parts
                if (relevant_entities is not None
                        and eid not in relevant_entities):
                    continue
                if type_ is not None:
                    if not lex.types.is_subtype(
                            ent.entity_type, type_):
                        continue
                if concept_ is not None:
                    if ent.concept_lemma != concept_:
                        continue
                # NotPattern exclusion: if the entity matches ALL of any
                # excluded EntityPattern's literal constraints, skip.
                excluded = False
                for excl in exclusions:
                    if _entity_matches_constraints(ent, excl, lex):
                        excluded = True
                        break
                if excluded:
                    continue
                cands.append(eid)
            domains.append(cands)

        if not all(domains):
            continue

        # Chain-derivation filter: samloke_chains_through_{en,sur} have
        # 3 free vars (A=contained, B=bridge, C=other-endpoint) and
        # generate cubic bindings. Almost all goal-relevant samloke
        # facts have one endpoint that's the agent (or some animate).
        # Restricting bindings to those where A or C is animate cuts
        # the per-chain count ~6×, with no observable yield loss (the
        # missing chains were all object×object×object which the
        # planner rarely needs).
        chain_filter_vids: list = []
        if d.name in (
                "samloke_chains_through_en", "samloke_chains_through_sur",
                "shared_container_means_samloke",
                "shared_apud_means_samloke"):
            for spec in impl_specs:
                if spec[0] == "rel" and spec[1] == "samloke":
                    from ..dsl.patterns import Var as _Var
                    for a in spec[2]:
                        if isinstance(a, _Var) and id(a) in all_vars:
                            chain_filter_vids.append(id(a))

        # Evidence-driven join eligibility: multi-rel derivation AND
        # every var appears in some rel-pattern AND every rel has a
        # non-empty pool. Otherwise fall back to cartesian.
        use_join = False
        if pools is not None and len(rel_patterns) >= 2:
            covered: set = set()
            for arg_vids in rel_arg_vids:
                for av in arg_vids:
                    if not isinstance(av, tuple):
                        covered.add(av)
            if (set(var_ids) <= covered
                    and all(rp.relation in pools and pools[rp.relation]
                            for rp in rel_patterns)):
                use_join = True

        def _emit_binding(binding):
            binding_vals = [binding[vid] for vid in var_ids]
            if len(set(binding_vals)) != len(binding_vals):
                return
            if chain_filter_vids:
                has_animate = False
                for vid in chain_filter_vids:
                    eid = binding.get(vid)
                    if eid is None:
                        continue
                    ent = trace.entities.get(eid)
                    if ent is not None and lex.types.is_subtype(
                            ent.entity_type, "animate"):
                        has_animate = True
                        break
                if not has_animate:
                    return
            pres: set = set()
            ok = True
            for rp, arg_vids in zip(rel_patterns, rel_arg_vids):
                args = []
                for av in arg_vids:
                    if isinstance(av, tuple) and av and av[0] == "<lit>":
                        args.append(av[1])
                        continue
                    eid = binding.get(av)
                    if eid is None:
                        ok = False
                        break
                    args.append(eid)
                if not ok:
                    break
                pres.add(("rel", rp.relation,
                          _canon_rel(rp.relation, tuple(args), sym_local)))
            if not ok:
                return
            for vid in var_ids:
                _, _, props = var_constraints[vid]
                eid = binding[vid]
                for slot, val in props:
                    pres.add(("prop", eid, slot, val))
            effs: set = set()
            for spec in impl_specs:
                if spec[0] == "prop":
                    _, e_var, slot, val = spec
                    eid = binding.get(id(e_var))
                    if eid is None:
                        continue
                    ent = trace.entities.get(eid)
                    if (ent is not None
                            and _entity_has_asserted_scalar(
                                ent, slot, lex)):
                        continue
                    effs.add(("prop", eid, slot, val))
                else:
                    _, name, arg_vars = spec
                    args = []
                    ok2 = True
                    for a in arg_vars:
                        if isinstance(a, Var):
                            eid = binding.get(id(a))
                            if eid is None:
                                ok2 = False
                                break
                            args.append(eid)
                        else:
                            args.append(a)
                    if not ok2:
                        continue
                    canon_args = _canon_rel(name, tuple(args), sym_local)
                    effs.add(("rel", name, canon_args))
                    _augment_pools_with_eff(name, canon_args)
            if effs:
                out.append((None, binding, pres, effs))

        if use_join:
            for binding in _evidence_join_combos(
                    rel_patterns, rel_arg_vids, pool_indices,
                    [list(pools[rp.relation]) for rp in rel_patterns],
                    var_constraints, var_exclusions,
                    trace, lex, _entity_matches_constraints):
                _emit_binding(binding)
            continue

        for combo in itertools.product(*domains):
            binding = {vid: combo[i] for i, vid in enumerate(var_ids)}
            _emit_binding(binding)
    return out


_ENTITIES_FOR_DELTA_CACHE: dict = {}


def _action_delta_mask(action, roles, rule_effects, lex,
                       table: FactTable, state_mask: int):
    """Compute the (add_mask, del_mask) bitmap delta for firing
    this grounded action. Replaces the search loop's tuple-mode
    `(adds, dels) → mask_for` two-step with direct `id_for | (1
    << id)` accumulation, which was the bulk of the per-successor
    cost on the old path.

    Sources of facts:
      - schema-level Effects (property writes on the target role)
      - rule-level effects: AddRelation, TransferN-as-havi,
        per-element list emissions
      - rule-level dels: RemoveRelation, wildcard-position dels
        resolved against the live state via `rel_name_mask` (only
        inspect bits for the matching relation name)

    Wildcard-pattern matching of dels is the only path that touches
    individual facts in state: AND `state_mask` with
    `table.rel_name_mask[relation]` and iterate just those bits,
    instead of scanning every fact.
    """
    sym = _symmetric_relations(lex)
    add_mask = 0
    del_mask = 0
    id_for = table.id_for
    rel_name_mask = table.rel_name_mask

    # Schema-level effects.
    for eff in action.effects:
        eid = roles.get(eff.target_role)
        if eid is None:
            continue
        add_mask |= 1 << id_for(
            ("prop", eid, eff.property, eff.value))

    # Rule-level effects, per-rule (cascade rules with wildcard pres
    # don't gate the main rule's effects).
    entry = rule_effects.get(action.lemma)
    if entry is not None:
        for rule_entry in entry.get("rules", ()):
            this_del = 0
            fires = True
            for relation, role_arg_names in rule_entry["dels"]:
                has_list = any(
                    isinstance(a, tuple) and a and a[0] == "<list>"
                    for a in role_arg_names)
                if has_list:
                    for eids in _expand_list_role_args(
                            role_arg_names, roles):
                        if any(e is None for e in eids):
                            continue
                        this_del |= 1 << id_for(
                            ("rel", relation,
                             _canon_rel(relation, eids, sym)))
                    continue
                if None in role_arg_names:
                    pattern = tuple(roles.get(r) if r else None
                                    for r in role_arg_names)
                    matched = 0
                    rel_bits = (state_mask
                                & rel_name_mask.get(relation, 0))
                    for fid in iter_bits(rel_bits):
                        f = table.tuple_for(fid)
                        args = f[2]
                        if (len(args) == len(pattern)
                                and all(p is None or p == a
                                        for p, a in zip(pattern, args))):
                            this_del |= 1 << fid
                            matched += 1
                    if matched == 0:
                        fires = False
                        break
                else:
                    eids = tuple(roles.get(r) for r in role_arg_names)
                    if any(e is None for e in eids):
                        continue
                    this_del |= 1 << id_for(
                        ("rel", relation,
                         _canon_rel(relation, eids, sym)))
            if not fires:
                continue
            # Lookup-add markers need a tuple view of state; build
            # once per rule-entry only if such a marker is present.
            facts_for_lookup = None
            for relation, role_arg_names in rule_entry["adds"]:
                if (facts_for_lookup is None
                        and any(isinstance(a, tuple) and a
                                and a[0] == "<lookup>"
                                for a in role_arg_names)):
                    facts_for_lookup = {
                        table.tuple_for(fid)
                        for fid in iter_bits(state_mask)}
                for eids in _expand_list_role_args(
                        role_arg_names, roles, facts=facts_for_lookup):
                    if any(e is None for e in eids):
                        continue
                    add_mask |= 1 << id_for(
                        ("rel", relation,
                         _canon_rel(relation, eids, sym)))
            del_mask |= this_del

    # validate_relation excludes filter: drop adds whose
    # `("rel", rname, args)` would be silently rejected by the engine
    # because some arg's entity has a forbidden value on a related
    # slot. Mirrors engine.py:778.
    excludes = _REL_ARG_EXCLUDES_CACHE.get(id(lex))
    if excludes is None:
        from ..dsl.introspect import relation_arg_excludes
        excludes = relation_arg_excludes(lex)
        _REL_ARG_EXCLUDES_CACHE[id(lex)] = excludes
    if excludes and add_mask:
        ent_map = _ENTITIES_FOR_DELTA_CACHE.get(id(lex), {})
        filtered = 0
        for fid in iter_bits(add_mask):
            f = table.tuple_for(fid)
            if f[0] != "rel":
                filtered |= 1 << fid
                continue
            _, rname, args = f
            hit = False
            for i, eid in enumerate(args):
                slot_map = excludes.get((rname, i))
                if not slot_map:
                    continue
                vals_by_slot = ent_map.get(eid, {})
                for slot, forbidden in slot_map.items():
                    vals = vals_by_slot.get(slot, ())
                    if any(v in forbidden for v in vals):
                        hit = True
                        break
                if hit:
                    break
            if not hit:
                filtered |= 1 << fid
        add_mask = filtered

    return add_mask, del_mask


def _seed_entity_props_for_delta(trace, lex):
    """Populate `_ENTITIES_FOR_DELTA_CACHE[id(lex)]` with each entity's
    relevant static slot values, so `_action_delta_mask` can filter
    forbidden adds without a trace handle. Run once per plan_for_goal
    invocation (entities don't change during search)."""
    rel_idx = _REL_ARG_EXCLUDES_CACHE.get(id(lex))
    if rel_idx is None:
        from ..dsl.introspect import relation_arg_excludes
        rel_idx = relation_arg_excludes(lex)
        _REL_ARG_EXCLUDES_CACHE[id(lex)] = rel_idx
    relevant_slots: set = set()
    for slot_map in rel_idx.values():
        relevant_slots.update(slot_map.keys())
    ent_map: dict = {}
    for eid, ent in trace.entities.items():
        slots = {s: tuple(ent.properties.get(s, ()))
                 for s in relevant_slots}
        ent_map[eid] = slots
    _ENTITIES_FOR_DELTA_CACHE[id(lex)] = ent_map


_DERIV_RELDEP_CACHE: dict = {}


def _derived_strip_set(
    derivation_pseudos, dels: set, adds: set,
) -> frozenset:
    """Given a delta (dels+adds) and the derivation list, return the
    set of relation names whose facts should be stripped before
    re-derivation. A relation R is stripped iff some derivation has
    R in its effs AND has some seed_rel in its pres that the delta
    changes. Cached per `id(derivation_pseudos)`.

    Closure-based: a derivation may produce R1 which feeds another
    derivation producing R2 — both R1 and R2 get stripped when any
    seed of R1 changes."""
    # Identity-verified cache: see _apply_delta for the rationale.
    key = id(derivation_pseudos)
    cached = _DERIV_RELDEP_CACHE.get(key)
    if cached is not None and cached[0] is derivation_pseudos:
        closure = cached[1]
    else:
        closure = None
    if closure is None:
        # rel_to_producers[r] = set of relations the derivation also
        # produces in its effs (so changing r might invalidate them
        # transitively).
        direct: dict[str, set[str]] = {}
        for _info, _binding, pres, effs in derivation_pseudos:
            in_rels = {p[1] for p in pres if p[0] == "rel"}
            out_rels = {e[1] for e in effs if e[0] == "rel"}
            for r in in_rels:
                direct.setdefault(r, set()).update(out_rels)
        # Transitive closure: if R1 → R2 and R2 → R3, also R1 → R3.
        closure = {r: set(outs) for r, outs in direct.items()}
        changed = True
        while changed:
            changed = False
            for r, outs in closure.items():
                new_outs = set(outs)
                for o in outs:
                    new_outs.update(direct.get(o, ()))
                if new_outs != outs:
                    closure[r] = new_outs
                    changed = True
        closure = {r: frozenset(outs) for r, outs in closure.items()}
        _DERIV_RELDEP_CACHE[key] = (derivation_pseudos, closure)
    changed_rels = {
        f[1] for f in dels if f[0] == "rel"} | {
        f[1] for f in adds if f[0] == "rel"}
    result: set[str] = set()
    for r in changed_rels:
        result.update(closure.get(r, ()))
    return frozenset(result)


def _apply_delta(facts: frozenset, adds: set, dels: set,
                  derivation_pseudos: list, slot_vocab: dict) -> frozenset:
    """Apply an event's delta to a fact set and re-derive.

    Scalar slot semantics: when an `adds` fact is `("prop", eid,
    slot, value)` for a scalar slot, all OTHER ("prop", eid, slot,
    *) facts are removed (only one value per scalar slot per entity
    at a time). Non-scalar slots accumulate values.

    Re-derivation: after the base delta is applied, run grounded
    derivation pseudo-actions to fixed point. Each derivation adds
    its `effs` if its `pres` are all in the fact set.
    """
    new_facts = set(facts)
    new_facts -= dels
    # Strip stale derived relations: when the delta changes a relation
    # that some derivation's pres depend on, the derivation's output
    # relation must be recomputed (otherwise stale facts from earlier
    # states latch — e.g. samloke(agent, akvo) derived when agent was
    # in salono persists after iri+eniri'd kuirejo). The strip-set is
    # computed once per derivation list via _derived_strip_set: for
    # each input relation, the set of relations any derivation chain
    # would invalidate.
    strip_targets = _derived_strip_set(
        derivation_pseudos, dels, adds)
    if strip_targets:
        new_facts = {
            f for f in new_facts
            if not (f[0] == "rel" and f[1] in strip_targets)}
    # Scalar slot displacement: replace other values on same slot.
    for f in adds:
        if f[0] == "prop":
            _, eid, slot, _val = f
            if slot_vocab.get(slot, {}).get("scalar", True):
                new_facts = {
                    g for g in new_facts
                    if not (g[0] == "prop"
                            and g[1] == eid and g[2] == slot)
                }
    new_facts |= adds
    # (eid, slot) -> set of values, for O(1) scalar shadow checks.
    prop_index: dict = {}
    for g in new_facts:
        if g[0] == "prop":
            prop_index.setdefault((g[1], g[2]), set()).add(g[3])
    # Worklist-based re-derivation. fact -> [pseudo_idx] inverted
    # index is built lazily by id(derivation_pseudos) and stashed in
    # a module-level cache — building this is the dominant cost if
    # we let it run per call. Pseudos with empty pres are always-
    # firable; collected separately so the initial pass triggers
    # them once.
    # Identity-verified cache: id() alone is unsafe because Python
    # reuses ids after GC, and a stale entry's indices may overshoot
    # a new, shorter list. Storing the list reference forces a
    # recompute on collision.
    cache_key = id(derivation_pseudos)
    cached = _DERIV_PRES_IDX_CACHE.get(cache_key)
    if cached is not None and cached[0] is derivation_pseudos:
        pres_idx = cached[1]
    else:
        fact_to: dict = {}
        no_pres_pseudos: list = []
        for i, (_info, _binding, pres, _effs) in enumerate(
                derivation_pseudos):
            if not pres:
                no_pres_pseudos.append(i)
                continue
            for p in pres:
                fact_to.setdefault(p, []).append(i)
        pres_idx = (fact_to, no_pres_pseudos)
        _DERIV_PRES_IDX_CACHE[cache_key] = (derivation_pseudos, pres_idx)
    fact_to, no_pres_pseudos = pres_idx

    fired: set = set()

    def _try_fire(pi, worklist):
        if pi in fired:
            return
        _info, _binding, pres, effs = derivation_pseudos[pi]
        for p in pres:
            if p not in new_facts:
                return
        fired.add(pi)
        for e in effs:
            if e in new_facts:
                continue
            if e[0] == "prop":
                _, e_eid, e_slot, _ = e
                if slot_vocab.get(e_slot, {}).get("scalar", True):
                    if prop_index.get((e_eid, e_slot)):
                        continue  # scalar-shadowed
                    prop_index.setdefault(
                        (e_eid, e_slot), set()).add(e[3])
            new_facts.add(e)
            worklist.append(e)

    # Seed worklist: every current fact. The fact_to index then only
    # surfaces pseudos that actually need a fact we have. Pseudos
    # with empty pres get a separate one-shot pass — they fire iff
    # not shadowed.
    from collections import deque
    worklist: deque = deque(new_facts)
    for pi in no_pres_pseudos:
        _try_fire(pi, worklist)
    while worklist:
        f = worklist.popleft()
        for pi in fact_to.get(f, ()):
            _try_fire(pi, worklist)
    return frozenset(new_facts)


_PSEUDOS_MASK_CACHE: dict = {}


def _compile_pseudos_for_mask(pseudos, table: FactTable, slot_vocab: dict):
    """Cached mask-form of the derivation pseudo list. Returns
    `(pres_masks, eff_list_per_pi, fact_to_pi, no_pres_pis,
    consumer_facts_mask)`:
      - `pres_masks[pi]`: pres bitmap; subset test is one AND.
      - `eff_list_per_pi[pi]`: list of `(fact_id, shadow_mask)` —
        `shadow_mask = scalar_slot_mask[(eid, slot)]` for scalar
        prop effects (intersect with state to detect shadowing),
        else 0.
      - `fact_to_pi[fact_id]`: pseudos whose pres includes that fact.
      - `no_pres_pis`: pseudos with empty pres (always-firable).
      - `consumer_facts_mask`: OR of fact IDs that appear in some
        pres, used by `_apply_delta_mask` to seed the worklist
        only with facts that actually trigger consumers.

    Identity-verified cache: Python may reuse `id()` after GC, so
    the stored pseudos reference is checked before reuse.
    """
    key = id(pseudos)
    cached = _PSEUDOS_MASK_CACHE.get(key)
    if cached is not None and cached[0] is pseudos:
        return cached[1]
    pres_masks: list = []
    eff_list_per_pi: list = []
    fact_to_pi: dict = {}
    no_pres_pis: list = []
    consumer_facts_mask = 0
    id_for = table.id_for
    for i, (_info, _binding, pres, effs) in enumerate(pseudos):
        pres_mask = 0
        for p in pres:
            pid = id_for(p)
            pres_mask |= 1 << pid
            consumer_facts_mask |= 1 << pid
            fact_to_pi.setdefault(pid, []).append(i)
        pres_masks.append(pres_mask)
        eff_list: list = []
        for e in effs:
            fid = id_for(e)
            # shadow_key = (eid, slot) for scalar prop effects;
            # looked up live in `scalar_slot_mask` at fire time so
            # new prop bits interned later get checked.
            if (e[0] == "prop"
                    and slot_vocab.get(e[2], {}).get("scalar", True)):
                eff_list.append((fid, (e[1], e[2])))
            else:
                eff_list.append((fid, None))
        eff_list_per_pi.append(eff_list)
        if not pres:
            no_pres_pis.append(i)
    result = (pres_masks, eff_list_per_pi, fact_to_pi,
              no_pres_pis, consumer_facts_mask)
    _PSEUDOS_MASK_CACHE[key] = (pseudos, result)
    return result


def _apply_delta_mask(state: int, add_mask: int, del_mask: int,
                     derivation_pseudos, slot_vocab: dict,
                     table: FactTable) -> int:
    """Bitmap variant of `_apply_delta`. Same semantics:
    apply dels, strip stale derived relations, apply scalar
    displacement on adds, apply adds, then re-derive to fixed point.

    The frozenset reconstruction the tuple-mode version does on every
    call is the bottleneck _apply_delta was profiling at — here that
    cost becomes a couple of int ops, and the derivation worklist
    iterates fact IDs instead of hashing tuple keys.
    """
    new_state = state & ~del_mask

    # Decode the delta to feed _derived_strip_set (which inspects
    # relation names). Only the bits actually in the delta are
    # touched, so this is O(|adds|+|dels|), not O(|state|).
    add_tuples = [table.tuple_for(fid) for fid in iter_bits(add_mask)]
    del_tuples = [table.tuple_for(fid) for fid in iter_bits(del_mask)]
    strip_targets = _derived_strip_set(
        derivation_pseudos, set(del_tuples), set(add_tuples))
    if strip_targets:
        new_state &= ~table.rel_mask(strip_targets)

    # Scalar slot displacement: clear all known values for any scalar
    # (eid, slot) we're writing, then OR in the adds.
    for f in add_tuples:
        if f[0] != "prop":
            continue
        _, eid, slot, _ = f
        if slot_vocab.get(slot, {}).get("scalar", True):
            new_state &= ~table.scalar_slot_mask.get((eid, slot), 0)
    new_state |= add_mask

    # Re-derivation worklist on mask form.
    (pres_masks, eff_list_per_pi, fact_to_pi, no_pres_pis,
        consumer_facts_mask) = _compile_pseudos_for_mask(
            derivation_pseudos, table, slot_vocab)

    # Scalar shadowing: live lookup against `scalar_slot_mask`
    # (which can grow after compile time as new prop bits are
    # interned by `_action_delta_mask`). An effect with
    # shadow_key=(eid, slot) is blocked iff `new_state` already has
    # some bit for that slot — we just checked `bit` isn't set, so
    # any remaining bit in `slot_mask` is a competing scalar value.
    fired: set = set()
    scalar_slot_mask = table.scalar_slot_mask
    from collections import deque
    # Seed only with facts some pseudo cares about — most state
    # facts have no consumers in the derivation graph.
    worklist: deque = deque(iter_bits(new_state & consumer_facts_mask))

    def try_fire(pi):
        nonlocal new_state
        if pi in fired:
            return
        pres_mask = pres_masks[pi]
        if (new_state & pres_mask) != pres_mask:
            return
        fired.add(pi)
        for efid, shadow_key in eff_list_per_pi[pi]:
            bit = 1 << efid
            if new_state & bit:
                continue
            if shadow_key is not None:
                slot_mask = scalar_slot_mask.get(shadow_key, 0)
                if new_state & slot_mask:
                    continue
            new_state |= bit
            worklist.append(efid)

    for pi in no_pres_pis:
        try_fire(pi)
    while worklist:
        fid = worklist.popleft()
        for pi in fact_to_pi.get(fid, ()):
            try_fire(pi)
    return new_state


def _build_rule_effects_index(rules, lex=None) -> dict:
    """Index rules by their trigger verb, collecting AddRelation and
    RemoveRelation effects with role-arg-name mappings. Maps
    verb_lemma → {'adds': [(relation, (role_name, ...))], 'dels':
    [...]}. Both lists are used: 'adds' for the relaxed graph
    (delete-relaxation drops dels there), 'dels' for the
    fact-set incremental simulator (which DOES apply dels).

    Given-bound vars appearing in AddRelation args become lookup
    specs `("<lookup>", relation, args_template)`. `args_template`
    is positional; each entry is either an event-role name or None
    at the position we extract. The simulator resolves the lookup
    against the current fact set, mirroring how RemoveRelation
    wildcards work."""
    from ..dsl.effects import (
        AddRelation, CreateEntity, ForEach, RemoveRelation, TransferN,
    )
    from ..dsl.patterns import (
        AndPattern, BindPattern, EntityPattern, EventPattern,
        RelPattern, Var,
    )
    out: dict = {}

    def _find_event_pattern(p):
        if isinstance(p, EventPattern):
            return p
        if isinstance(p, AndPattern):
            return (_find_event_pattern(p.left)
                    or _find_event_pattern(p.right))
        return None

    def _bind_var_of(p):
        """Extract the Var bound by a role pattern.
        BindPattern.target carries the Var. AndPattern wraps a
        BindPattern alongside an EntityPattern (e.g.
        entity(type="animate") & bind(A))."""
        if isinstance(p, BindPattern):
            return p.target
        if isinstance(p, AndPattern):
            return _bind_var_of(p.left) or _bind_var_of(p.right)
        if isinstance(p, Var):
            return p
        return None

    def _lookup_for_var(target_var, given_patterns, var_to_role):
        """Find a RelPattern in `given` that binds `target_var` and
        return ("<lookup>", relation, args_template). args_template
        is positional (by lex.relations[rel].arg_names); None marks
        the position holding target_var, other positions are role
        names from `var_to_role`. Returns None if not found or any
        other arg in the lookup relation isn't event-role-bound."""
        if lex is None:
            return None
        for g in given_patterns:
            if not isinstance(g, RelPattern):
                continue
            rel_def = lex.relations.get(g.relation)
            if rel_def is None:
                continue
            extract_idx = None
            template = []
            ok = True
            for i, arg_name in enumerate(rel_def.arg_names):
                patt = g.arg_patterns.get(arg_name)
                v = _bind_var_of(patt) if patt is not None else None
                if isinstance(v, Var) and v is target_var:
                    if extract_idx is not None:
                        ok = False
                        break
                    extract_idx = i
                    template.append(None)
                elif isinstance(v, Var):
                    role = var_to_role.get(id(v))
                    if role is None:
                        ok = False
                        break
                    template.append(role)
                else:
                    ok = False
                    break
            if ok and extract_idx is not None:
                return ("<lookup>", g.relation, tuple(template))
        return None

    from ..dsl.patterns import (
        ConceptModelsSlotPattern, HasConceptFieldPattern,
    )

    def _extract_concept_constraints(rule, var_to_role):
        """Walk given clauses for `concept_models_slot_check(entity,
        slot)` whose `slot` is bound by an earlier
        `has_concept_field(source, field, slot)`. Each pair becomes a
        per-combo grounding constraint: at firing time, entity's
        concept must model whatever slot source's concept declares at
        `field`. Returns list of (entity_role, source_role, field_name)
        triples. Generic — fires for any rule using these patterns,
        no hardcoded verb names."""
        if not rule.given:
            return []
        # slot_var_id → (source_role, field_name)
        slot_origin: dict = {}
        for g in rule.given:
            if not isinstance(g, HasConceptFieldPattern):
                continue
            source_role = var_to_role.get(id(g.entity_var))
            if source_role is None:
                continue
            slot_origin[id(g.bind_target)] = (source_role, g.field_name)
        out: list = []
        for g in rule.given:
            if not isinstance(g, ConceptModelsSlotPattern):
                continue
            entity_role = var_to_role.get(id(g.entity_var))
            origin = slot_origin.get(id(g.slot_var))
            if entity_role is None or origin is None:
                continue
            source_role, field_name = origin
            out.append((entity_role, source_role, field_name))
        return out

    for rule in rules:
        ep = _find_event_pattern(rule.when)
        if ep is None:
            continue
        verb = ep.action
        var_to_role: dict = {}
        for role_name, role_pat in ep.role_patterns.items():
            v = _bind_var_of(role_pat)
            if isinstance(v, Var):
                var_to_role[id(v)] = role_name
        # Collect concept-models-slot constraints from this rule's
        # given clauses (compositional with has_concept_field).
        constraints = _extract_concept_constraints(rule, var_to_role)
        if constraints:
            entry_for_constraints = out.setdefault(
                verb, {"adds": [], "dels": [], "rules": []})
            entry_for_constraints.setdefault(
                "concept_constraints", []).extend(constraints)
        # Walk rule.then for AddRelation / RemoveRelation / TransferN.
        effects = (rule.then if isinstance(rule.then, (list, tuple))
                   else [rule.then])
        entry = out.setdefault(
            verb, {"adds": [], "dels": [], "rules": []})
        rule_adds: list = []
        rule_dels: list = []
        # Flatten ForEach effects: each inner effect's args use the
        # for-loop's item_var, which the planner resolves at grounding
        # time by reading the event's list-valued role binding. Encode
        # the substitution as a `("<list>", role_name)` marker so the
        # downstream grounder / _action_delta_mask can expand the variadic
        # effects per element.
        flat_effects: list = []
        for eff in effects:
            if isinstance(eff, ForEach):
                list_role = var_to_role.get(id(eff.list_var))
                if list_role is None:
                    # for_each over a list_var not bound by any event
                    # role pattern — skip (rule misconfigured or
                    # using a var bound elsewhere we don't model).
                    continue
                list_marker = ("<list>", list_role)
                # Snapshot var_to_role with the item_var pinned to the
                # list marker so the inner-effect walk substitutes it.
                inner_var_to_role = dict(var_to_role)
                inner_var_to_role[id(eff.item_var)] = list_marker
                for inner in eff.effects:
                    flat_effects.append((inner, inner_var_to_role))
            else:
                flat_effects.append((eff, var_to_role))
        given_patterns = rule.given if rule.given else []
        for eff, eff_var_to_role in flat_effects:
            if isinstance(eff, AddRelation):
                role_args = []
                ok = True
                for arg in eff.args:
                    role_name = eff_var_to_role.get(id(arg))
                    if role_name is not None:
                        role_args.append(role_name)
                        continue
                    # Literal string arg (e.g. scias's rel_type
                    # "en"/"sur"/"havi" in
                    # add_relation("scias", agent, "en", theme, loc)).
                    # Tag positionally with the literal value so the
                    # grounder substitutes it verbatim instead of
                    # treating the position as a role binding.
                    if isinstance(arg, str):
                        role_args.append(("<literal>", arg))
                        continue
                    # Not event-role-bound — try given-clause lookup.
                    if isinstance(arg, Var):
                        spec = _lookup_for_var(
                            arg, given_patterns, eff_var_to_role)
                        if spec is not None:
                            role_args.append(spec)
                            continue
                    ok = False
                    break
                if ok:
                    rule_adds.append((eff.relation, tuple(role_args)))
                    entry["adds"].append(
                        (eff.relation, tuple(role_args)))
            elif isinstance(eff, RemoveRelation):
                # Args may include given-bound vars (e.g. iri's "from"
                # container O, matched against rel("en", agent, var(O)) in
                # `given`). Those positions can't be resolved until
                # simulation time. Encode them as None; _action_delta_mask
                # expands wildcards by querying the current fact set.
                role_args = []
                for arg in eff.args:
                    role_name = eff_var_to_role.get(id(arg))
                    role_args.append(role_name)  # None = wildcard
                rule_dels.append((eff.relation, tuple(role_args)))
                entry["dels"].append(
                    (eff.relation, tuple(role_args)))
            elif isinstance(eff, TransferN):
                src_role = eff_var_to_role.get(id(eff.source))
                tgt_role = eff_var_to_role.get(id(eff.target))
                if src_role and tgt_role:
                    rule_adds.append(("havi", (tgt_role, src_role)))
                    entry["adds"].append(
                        ("havi", (tgt_role, src_role)))
        if rule_adds or rule_dels:
            entry["rules"].append(
                {"adds": rule_adds, "dels": rule_dels})
    _synthesize_implications_from_derivations(out, lex)
    return out


def _synthesize_implications_from_derivations(rule_effects: dict, lex) -> dict:
    """For each DSL derivation of the form
        derive(when=rel(R_in, arg_a=Var, arg_b=literal, arg_c=Var, arg_d=Var),
               implies=relation(R_out, *some_subset_of_args))
    propagate the implication into per-verb rule_effects: any rule that
    adds R_in(...) with the same literal at arg_b synthesizes R_out(...)
    at the same arg positions used by the implies side.

    Why this is needed even though the engine already runs the
    derivation at trace time: the planner's relaxed graph grounds
    derivation pseudo-actions over concrete (K, X, L) entity tuples
    drawn from `relevant_entities`. The 'L' position above is
    existential in the derivation (its value is irrelevant to the
    implication), but the grounder must bind it to a concrete entity
    that's in the relevant set — and the actual scias L (the
    container) often isn't. The synthesis bypasses that by attaching
    the implication directly to the producing rule's effects, which
    the per-rule simulator emits unconditionally on firing.

    Source of truth: the derivations list. Adding another
    locative-style derivation auto-extends the synthesis."""
    from ..dsl.patterns import LiteralValuePattern, BindPattern, RelPattern, Var
    from ..dsl.engine import RelationImplication
    from ..dsl.rules import runtime_derivations_for
    for d in runtime_derivations_for(lex):
        when = d.when
        if not isinstance(when, RelPattern):
            continue
        rel_in = when.relation
        rel_def = lex.relations.get(rel_in)
        if rel_def is None:
            continue
        # Map arg_name → BindPattern target Var (for entity args) or
        # LiteralValuePattern (for literal-constrained positions).
        arg_kinds: list = []
        var_to_arg_idx: dict = {}
        for i, arg_name in enumerate(rel_def.arg_names):
            ap = when.arg_patterns.get(arg_name)
            if isinstance(ap, LiteralValuePattern):
                arg_kinds.append(("lit", ap.value))
            elif isinstance(ap, BindPattern):
                arg_kinds.append(("var", ap.target))
                var_to_arg_idx[id(ap.target)] = i
            elif isinstance(ap, Var):
                arg_kinds.append(("var", ap))
                var_to_arg_idx[id(ap)] = i
            else:
                arg_kinds = []
                break
        if not arg_kinds:
            continue
        # Need at least one literal-constrained position — without it
        # the derivation already grounds fine via the relaxed path.
        if not any(k == "lit" for k, _ in arg_kinds):
            continue
        for imp in d.implies:
            if not isinstance(imp, RelationImplication):
                continue
            rel_out = imp.name
            # Each implies arg must be a Var bound by `when` — map it
            # back to its position in rel_in.
            imp_positions: list = []
            ok = True
            for a in imp.args:
                idx = var_to_arg_idx.get(id(a))
                if idx is None:
                    ok = False
                    break
                imp_positions.append(idx)
            if not ok:
                continue
            # Walk rule_effects entries: any rule adding rel_in with
            # the same literal at the literal positions yields a
            # synthesized rel_out add at imp_positions.
            for verb_lemma, entry in rule_effects.items():
                synth_adds: list = []
                seen: set = set()
                for relation, role_args in entry["adds"]:
                    if relation != rel_in:
                        continue
                    if len(role_args) != len(arg_kinds):
                        continue
                    # Literal positions must match.
                    if not all(
                            (isinstance(role_args[i], tuple)
                             and len(role_args[i]) == 2
                             and role_args[i][0] == "<literal>"
                             and role_args[i][1] == val)
                            for i, (k, val) in enumerate(arg_kinds)
                            if k == "lit"):
                        continue
                    new_args = tuple(role_args[i] for i in imp_positions)
                    key = (rel_out, new_args)
                    if key in seen:
                        continue
                    seen.add(key)
                    synth_adds.append((rel_out, new_args))
                if synth_adds:
                    entry["adds"].extend(synth_adds)
                    # Mirror inside a rule entry so the per-rule
                    # _action_delta_mask loop emits the synth on fire too.
                    entry["rules"].append(
                        {"adds": list(synth_adds), "dels": []})
    return rule_effects


def _symmetric_relations(lex) -> frozenset:
    """Cache symmetric-arity-2 relation names for canonicalization
    in the fact-tuple representation."""
    cache = getattr(lex, "_fwd_planner_sym_cache", None)
    if cache is not None:
        return cache
    result = frozenset(
        name for name, rel_def in lex.relations.items()
        if getattr(rel_def, "symmetric", False)
        and rel_def.arity == 2)
    try:
        object.__setattr__(lex, "_fwd_planner_sym_cache", result)
    except Exception:
        pass
    return result


def _canon_rel(rel_name: str, args: tuple, sym: frozenset) -> tuple:
    """Canonicalize a symmetric arity-2 relation's args (alphabetical
    order). Live engine matches symmetric relations by trying both
    orderings; the forward planner uses fact-tuple equality, so we
    canonicalize once at emission and reuse the canonical form
    everywhere.

    Skips canonicalization when an arg is a tuple — list-valued role
    bindings (fari.parts = (p1,p2,p3)) reach here via the producer-
    recursion in _infra_prespawn, and `string > tuple` would raise.
    A tuple-vs-string pair never represents the same symmetric fact
    under arg-swap, so returning args unchanged is safe."""
    if (rel_name not in sym or len(args) != 2
            or isinstance(args[0], tuple) or isinstance(args[1], tuple)):
        return args
    if args[0] > args[1]:
        return (args[1], args[0])
    return args


def _state_facts(trace, derived, lex=None) -> set:
    """Snapshot of state facts (property + relation) for the
    relaxed graph's layer 0. Symmetric arity-2 relation tuples are
    canonicalized to a single ordering — every emission point uses
    `_canon_rel`, so fact-tuple equality finds matches regardless
    of the order in which a derivation or pre wrote the relation.

    For scalar slots, asserted wins: a derived `physical_has_wetness
    = seka` default doesn't add a fact when the entity already has
    an asserted `wetness=malseka`. Otherwise the relaxed graph thinks
    the goal `wetness=seka` is trivially satisfied and the search
    short-circuits before any drying action gets tried.

    Cached on (len(trace.entities), len(trace.relations),
    len(trace.events), id(derived)). The trace is append-only for
    entities/relations and entity.properties is immutable after
    creation (CLAUDE.md), so these counts uniquely identify the
    fact set under the given derived view. `_heuristic_and_helpful`
    calls _state_facts per h_FF expansion (~thousands per plan),
    each with identical inputs — the rebuild was 230µs and dominated
    the planner profile."""
    cache = getattr(trace, "_fwd_state_facts_cache", None)
    key = (len(trace.entities), len(trace.relations),
           len(trace.events), id(derived))
    if cache is not None and cache[0] == key:
        return cache[1]
    facts: set = set()
    sym = _symmetric_relations(lex) if lex is not None else frozenset()
    asserted_scalar_keys: set = set()
    for eid, ent in trace.entities.items():
        for slot, values in ent.properties.items():
            for v in values:
                facts.add(("prop", eid, slot, v))
            if _entity_has_asserted_scalar(ent, slot, lex):
                asserted_scalar_keys.add((eid, slot))
    for r in trace.relations:
        facts.add(("rel", r.relation,
                   _canon_rel(r.relation, tuple(r.args), sym)))
    if derived is not None:
        for rel_tuple in derived.relations:
            args = tuple(rel_tuple[1])
            facts.add(("rel", rel_tuple[0],
                       _canon_rel(rel_tuple[0], args, sym)))
        for (eid, slot), val in (
                getattr(derived, "properties", {}) or {}).items():
            if (eid, slot) in asserted_scalar_keys:
                continue
            if isinstance(val, list):
                for v in val:
                    facts.add(("prop", eid, slot, v))
            else:
                facts.add(("prop", eid, slot, val))
    try:
        object.__setattr__(trace, "_fwd_state_facts_cache", (key, facts))
    except Exception:
        pass
    return facts


def _state_facts_mask(trace, derived, lex=None) -> int:
    """Bitmap form of `_state_facts`. Allocates fact IDs in the
    trace's `FactTable` on first sight, returns the OR of bits for
    every fact in the state.

    Cached on the same `(entities, relations, events, derived_id)`
    key as `_state_facts` but in a separate slot — the masks depend
    on the table's assignment order, so a callsite that hasn't yet
    primed the table can still hit the tuple-form cache without
    contaminating the mask cache.
    """
    cache = getattr(trace, "_fwd_state_facts_mask_cache", None)
    table = fact_table_for(trace)
    key = (len(trace.entities), len(trace.relations),
           len(trace.events), id(derived), id(table))
    if cache is not None and cache[0] == key:
        return cache[1]
    facts = _state_facts(trace, derived, lex)
    mask = table.mask_for(facts)
    try:
        object.__setattr__(trace, "_fwd_state_facts_mask_cache", (key, mask))
    except Exception:
        pass
    return mask


def _fp_tuple_pool(source_rel: str, trace, lex, rule_effects) -> list:
    """Enumerate (positional-args-tuple) for `source_rel` from:
      (a) real instances in `trace.relations` matching the name
      (b) hypothetical instances producible by rule_effects against
          the current trace facts — for each rule that adds
          source_rel, resolve its <literal>/role-name/<lookup>
          arg-template against in-trace en/havi/sur tuples and the
          verb's role-type space.

    Used by _ground_all_actions for kind="from_precondition" roles:
    instead of `R × E^N` cartesian over literal pool × entities,
    enumerate from `O(|en| + |sur| + |havi| + |source_rel|)`
    tuples grounded in actual scene state — typically ~5-20 per
    scene vs ~10k+ cartesian.

    Returns deduplicated list of tuples; each tuple has arity
    equal to source_rel.arity, args in positional order."""
    out: set = set()
    # (a) Real instances.
    for r in trace.relations:
        if r.relation == source_rel and len(r.args) == lex.relations[
                source_rel].arity:
            out.add(tuple(r.args))
    # (b) Hypothetical from producers.
    src_rel = lex.relations.get(source_rel)
    arity = src_rel.arity if src_rel else None
    if arity is None:
        return list(out)
    # Index trace.relations by name for <lookup> resolution.
    by_rel: dict = {}
    for r in trace.relations:
        by_rel.setdefault(r.relation, []).append(tuple(r.args))
    for verb, entry in rule_effects.items():
        action_obj = lex.actions.get(verb)
        if action_obj is None:
            continue
        # Per-verb-role enumeration: pre-bind role names to candidate
        # entities by type. Used to fill role-name positions in the
        # add template.
        from ..entity_index import entity_index_for
        entity_idx = entity_index_for(trace, lex)
        role_pools: dict = {}
        for rs in action_obj.roles:
            rk = getattr(rs, "kind", "single")
            if rk in ("list", "relation", "from_precondition"):
                continue
            role_pools[rs.name] = list(entity_idx.entities_matching(rs.type))
        for relation, role_args in entry.get("adds", []):
            if relation != source_rel or len(role_args) != arity:
                continue
            # Resolve each positional arg into a candidate list.
            per_pos: list = []
            ok = True
            for arg in role_args:
                if isinstance(arg, tuple) and arg:
                    kind = arg[0]
                    if kind == "<literal>":
                        per_pos.append([arg[1]])
                        continue
                    if kind == "<lookup>":
                        # arg = ("<lookup>", relation, args_template).
                        # Walk by_rel for matches; extract the
                        # unbound (None) position.
                        lookup_rel = arg[1]
                        tmpl = arg[2]
                        extract_idx = None
                        for i, t in enumerate(tmpl):
                            if t is None:
                                extract_idx = i
                                break
                        if extract_idx is None:
                            ok = False
                            break
                        vals = []
                        for fact_args in by_rel.get(lookup_rel, ()):
                            if len(fact_args) != len(tmpl):
                                continue
                            # Other-position templates that name a
                            # role must match against role_pools when
                            # we later pick a verb-role binding;
                            # accept any value for now and re-check
                            # via cross-product below.
                            vals.append(fact_args[extract_idx])
                        if not vals:
                            ok = False
                            break
                        per_pos.append(vals)
                        continue
                    if kind == "<list>":
                        # List role from verb event — skip
                        ok = False
                        break
                    ok = False
                    break
                # Plain role-name positional arg.
                cands = role_pools.get(arg)
                if not cands:
                    ok = False
                    break
                per_pos.append(cands)
            if not ok:
                continue
            for combo in itertools.product(*per_pos):
                out.add(tuple(combo))
    return list(out)


def _ground_all_actions(trace, lex, derived, rule_effects) -> list:
    """All (action, roles, pres, effs) for actions whose roles can
    be bound to current entities. Pure enumeration — preconditions
    not checked here (the relaxed graph layer decides applicability).
    Skips bindings where the same entity fills two roles, matches the
    behavior of `_applicable_actions`. `rule_effects` is the
    rule-effect index built by `_build_rule_effects_index`.

    Also enforces *effect meaningfulness*: an action whose effect
    writes slot S on target T is skipped when T's concept doesn't
    declare S (and S isn't pervasive, and S isn't derivable for T's
    type). This reads BOTH halves of the schema's intent — Action's
    role.properties (who can fill the role) AND Concept.properties
    (which slots are tracked for this concept) — so the planner's
    reachability matches the data author's narrative intent. Without
    it, `kuiri(actor, kafo)` would be a valid grounding (kafo is
    edible substance; passes role.type/properties) even though kafo
    doesn't declare cooking_state and the data author signals it
    isn't a tracked dimension for coffee."""
    out = []
    # State facts for lookup resolution in given-bound add args
    # (fari's en(theme, agent_location)).
    state_facts = _state_facts(trace, derived, lex)
    # Per-type entity pools come from the trace-attached EntityIndex
    # (lazy-built, keyed on len(entities); shared across grounding
    # calls within a scene). list() materializes for itertools.product
    # downstream.
    from ..entity_index import entity_index_for
    entity_idx = entity_index_for(trace, lex)
    type_pool: dict = {}
    def cands_for_type(type_name):
        pool = type_pool.get(type_name)
        if pool is None:
            pool = list(entity_idx.entities_matching(type_name))
            type_pool[type_name] = pool
        return pool
    # Effect-meaningfulness lookup: the per-slot bitmap on
    # `lex.concept_index.concepts_modeling_slot(eff.property)` gives
    # the set of concept lemmas where the slot is meaningfully
    # tracked; combo loop just does a `concept_lemma in frozenset`
    # check. Hoist concept_models_slot deps out of the combo loop —
    # same lex, same derivations across every combo.
    from ..dsl.introspect import concept_models_slot
    from ..dsl.rules import runtime_derivations_for
    _runtime_derivs = runtime_derivations_for(lex)
    # Per-action precondition/effect templates, compiled once per lex.
    sym = _symmetric_relations(lex)
    tmpl_cache = getattr(lex, "_fwd_planner_action_tmpls", None)
    if tmpl_cache is None or tmpl_cache.get("_rule_effects_id") != id(rule_effects):
        # Collect the set of relations that COULD be forbidden by
        # arg_patterns excludes or arg_compare. Templates whose
        # rule_adds touch none of these can skip _filter_forbidden_effs.
        excludes = _REL_ARG_EXCLUDES_CACHE.get(id(lex))
        if excludes is None:
            from ..dsl.introspect import relation_arg_excludes
            excludes = relation_arg_excludes(lex)
            _REL_ARG_EXCLUDES_CACHE[id(lex)] = excludes
        forbid_rels: set = set()
        for (rname, _i) in (excludes or {}):
            forbid_rels.add(rname)
        for rname, rel_def in lex.relations.items():
            if getattr(rel_def, "arg_compare", None):
                forbid_rels.add(rname)
        tmpl_cache = {"_rule_effects_id": id(rule_effects)}
        for a in lex.actions.values():
            tmpl_cache[a.lemma] = _compile_action_template(
                a, rule_effects, sym, forbid_rels=forbid_rels)
        try:
            object.__setattr__(lex, "_fwd_planner_action_tmpls", tmpl_cache)
        except Exception:
            pass
    for action in lex.actions.values():
        if not action.roles:
            continue
        # Actions with list-kind roles (fari.parts) can't be enumerated
        # combinatorially — the role binds to a TUPLE of entities, not
        # to one entity per slot. `_ground_constructable_actions`
        # handles them separately by walking the constructable concept
        # graph for valid recipes.
        if any(getattr(r, "kind", "single") == "list" for r in action.roles):
            continue
        # Group from_precondition roles by source relation; each group
        # enumerates from a shared pool of tuples (real scias from
        # trace + hypothetical producible by rule_effects), so the
        # group's role values are coupled at the same tuple's
        # positions rather than independent cartesian.
        fp_groups: dict = {}
        for r in action.roles:
            if getattr(r, "kind", "single") == "from_precondition":
                src = getattr(r, "from_precondition", None)
                if src:
                    fp_groups.setdefault(src, []).append(r)
        # Couple-all-precondition-roles optimization: when an action has
        # a RelationPrecondition on `src` AND src is the source for one
        # or more from_precondition roles, ALL roles in that
        # precondition's role-tuple get bound from the same scias
        # tuple — not just the from_precondition-marked ones. Without
        # this, rakonti/demandi/respondi/instrui enumerate the cartesian
        # `|agents| × |recipients| × |themes| × |scias_tuples|`
        # because agent/recipient/theme are scalar roles, blowing the
        # per-scene grounding count from ~hundreds to ~hundreds of
        # thousands. With it: the action's grounding count is bounded
        # by `|scias_tuples| × |free_roles|`. role_pos_map[src] maps
        # role_name → position in the precondition's role-tuple.
        role_pos_map: dict = {}
        if fp_groups:
            from ..schemas import RelationPrecondition
            for pc in action.preconditions:
                if (isinstance(pc, RelationPrecondition)
                        and pc.rel in fp_groups):
                    pos_map = role_pos_map.setdefault(pc.rel, {})
                    for i, rname in enumerate(pc.roles):
                        if rname not in pos_map:
                            pos_map[rname] = i
        coupled_role_names: set = set()
        for src, pos_map in role_pos_map.items():
            coupled_role_names.update(pos_map.keys())
        fp_pools: dict = {}
        for src in fp_groups:
            fp_pools[src] = _fp_tuple_pool(src, trace, lex, rule_effects)
        per_role: list = []
        # role_origin[i]: ("role", role_spec) or ("fp", src, [(role, pos), ...])
        role_origin: list = []
        ok = True
        seen_fp: set = set()
        for role_spec in action.roles:
            # Coupled scalar role — handled as part of an fp group.
            if (role_spec.name in coupled_role_names
                    and getattr(role_spec, "kind", "single") != "from_precondition"):
                continue  # picked up by the fp_groups iteration below
            if getattr(role_spec, "kind", "single") == "from_precondition":
                src = getattr(role_spec, "from_precondition", None)
                if src in seen_fp:
                    continue  # group handled below
                seen_fp.add(src)
                tuples = fp_pools.get(src, [])
                if not tuples:
                    ok = False
                    break
                per_role.append(tuples)
                # Group all roles tied to this scias tuple: marked
                # from_precondition AND scalars sharing the precondition.
                role_pos_list = [
                    (r.name,
                     getattr(r, "from_precondition_position", None))
                    for r in fp_groups[src]]
                pos_map = role_pos_map.get(src, {})
                seen_role_names = {n for n, _ in role_pos_list}
                for rname, pos in pos_map.items():
                    if rname in seen_role_names:
                        continue
                    role_pos_list.append((rname, pos))
                role_origin.append(("fp", src, role_pos_list))
                continue
            # kind="relation": value is a relation name (string), not
            # an entity. Enumerate from allowed_values or all
            # registered relations. The rule's `given` clauses filter
            # to plannable ones.
            if getattr(role_spec, "kind", "single") == "relation":
                pool = (list(role_spec.allowed_values)
                        if getattr(role_spec, "allowed_values", None)
                        else list(lex.relations.keys()))
                if not pool:
                    ok = False
                    break
                per_role.append(pool)
                role_origin.append(("role", role_spec))
                continue
            cand = cands_for_type(role_spec.type)
            if not cand:
                ok = False
                break
            per_role.append(cand)
            role_origin.append(("role", role_spec))
        if not ok:
            continue
        # Build per-effect target-role concept-validity sets once per
        # action — combo loop does O(1) lookup rather than dispatching
        # to concept_models_slot per combo.
        eff_target_roles: list = []
        eff_target_valid: list = []
        if action.effects:
            for eff in action.effects:
                valid = lex.concept_index.concepts_modeling_slot(eff.property)
                if valid:
                    eff_target_roles.append(eff.target_role)
                    eff_target_valid.append(valid)
        for combo in itertools.product(*per_role):
            # Unpack: regular role slots take their value; fp slots
            # carry a tuple to be exploded across multiple role names.
            roles = {}
            scalar_seen = []
            for item, origin in zip(combo, role_origin):
                if origin[0] == "role":
                    roles[origin[1].name] = item
                    scalar_seen.append(item)
                else:  # fp group
                    _, _src, role_pos_list = origin
                    for role_name, pos in role_pos_list:
                        if pos is None or pos >= len(item):
                            roles[role_name] = None
                        else:
                            roles[role_name] = item[pos]
                            scalar_seen.append(item[pos])
            if len(set(scalar_seen)) != len(scalar_seen):
                continue
            # Effect-meaningfulness via precomputed per-concept set.
            skip = False
            for tr, valid in zip(eff_target_roles, eff_target_valid):
                target_eid = roles.get(tr)
                if target_eid is None:
                    continue
                ent = trace.entities.get(target_eid)
                if ent is None:
                    continue
                if ent.concept_lemma not in valid:
                    skip = True
                    break
            if skip:
                continue
            tmpl = tmpl_cache.get(action.lemma)
            # Concept-constraints from rule given clauses: per
            # (entity_role, source_role, field_name), check that
            # entity's concept models source's concept_field value
            # as a slot. Mirrors mezuri's rule guard at planning time
            # so the planner doesn't waste effort on combinations
            # whose rule body would no-op.
            if tmpl is not None and tmpl.concept_constraints:
                constr_skip = False
                for entity_role, source_role, field_name \
                        in tmpl.concept_constraints:
                    source_eid = roles.get(source_role)
                    entity_eid = roles.get(entity_role)
                    if source_eid is None or entity_eid is None:
                        continue
                    src_ent = trace.entities.get(source_eid)
                    tgt_ent = trace.entities.get(entity_eid)
                    if src_ent is None or tgt_ent is None:
                        continue
                    src_concept = lex.concepts.get(src_ent.concept_lemma)
                    if src_concept is None:
                        continue
                    vals = src_concept.properties.get(field_name)
                    if not vals:
                        constr_skip = True
                        break
                    slot_name = vals[0]
                    tgt_concept = lex.concepts.get(tgt_ent.concept_lemma)
                    if tgt_concept is None:
                        constr_skip = True
                        break
                    if not concept_models_slot(
                            tgt_concept, slot_name, lex,
                            _runtime_derivs):
                        constr_skip = True
                        break
                if constr_skip:
                    continue
            if tmpl is not None:
                pres, effs = _ground_facts_from_template(
                    tmpl, roles, state_facts)
                if not effs:
                    continue  # No-effect actions don't add facts.
                if not tmpl.skip_filter:
                    effs = _filter_forbidden_effs(effs, trace, lex)
                    if not effs:
                        continue  # All effects forbidden — no-op grounding.
            else:
                pres, effs = _ground_action_facts(
                    action, roles, lex, rule_effects, facts=state_facts)
                if not effs:
                    continue
                effs = _filter_forbidden_effs(effs, trace, lex)
                if not effs:
                    continue
            out.append((action, roles, pres, effs))
    return out



def _action_effects_meaningful(action, roles, trace, lex) -> bool:
    """True iff every effect's slot is meaningful for its target
    entity's concept. Delegates to `concept_models_slot` so the
    planner and the sampler enforce the same predicate. Rejects
    `kuiri(actor, kafo)` because kafo doesn't declare cooking_state
    and no derivation produces it for substances — schema's two
    halves (Action.role.properties for pre-state, Concept.properties
    for slot relevance) read jointly determine reachability."""
    if not action.effects:
        return True
    from ..dsl.introspect import concept_models_slot
    from ..dsl.rules import runtime_derivations_for
    derivs = runtime_derivations_for(lex)
    for eff in action.effects:
        target_eid = roles.get(eff.target_role)
        if target_eid is None:
            continue
        target_ent = trace.entities.get(target_eid)
        if target_ent is None:
            continue
        target_concept = lex.concepts.get(target_ent.concept_lemma)
        if not concept_models_slot(
                target_concept, eff.property, lex, derivs):
            return False
    return True


def _ground_constructable_actions(
    trace, lex, rule_effects, derived=None,
) -> list:
    """Emit grounded fari (and any future construct-style) actions
    by walking the constructable concepts whose recipe components are
    materialized in the trace.

    For each in-trace entity whose concept declares `constructable=yes`
    AND has a `parts` recipe, try to bind one in-trace entity per
    declared part concept. Combined with each animate agent and (if
    `crafted_with` is set) each available tool concept, this becomes
    a regular grounded-action tuple that the search loop treats no
    differently from preni / kuiri / iri.

    Without this, fari can only fire when explicitly nominated by an
    event_fire drive — the goal-driven path (manĝi a constructable,
    doni a constructable, …) wouldn't discover it as a producer of
    havi facts."""
    fari = lex.actions.get("fari")
    if fari is None:
        return []
    state_facts = _state_facts(trace, derived, lex)
    # Index entities by concept_lemma once for the inner per-part
    # matching loop.
    by_concept: dict[str, list[str]] = {}
    for eid, ent in trace.entities.items():
        by_concept.setdefault(ent.concept_lemma, []).append(eid)
    # Agent candidates: any in-trace animate.
    animate_eids = [
        eid for eid, ent in trace.entities.items()
        if lex.types.is_subtype(ent.entity_type, "animate")]
    if not animate_eids:
        return []
    out: list = []
    for theme_eid, theme_ent in trace.entities.items():
        concept = lex.concepts.get(theme_ent.concept_lemma)
        if concept is None:
            continue
        if "yes" not in concept.properties.get("constructable", ()):
            continue
        if not concept.parts:
            continue
        # Match one entity per part concept. We don't enumerate all
        # permutations — pick the first matching eid per part, skipping
        # the theme itself. Enumerating all combinations would explode
        # for multi-part recipes; deterministic-first is sufficient
        # because each part concept usually has one or two instances.
        part_eids: list[str] = []
        for part_spec in concept.parts:
            cands = by_concept.get(part_spec.concept, ())
            chosen = next(
                (eid for eid in cands
                 if eid != theme_eid and eid not in part_eids),
                None)
            if chosen is None:
                break
            part_eids.append(chosen)
        if len(part_eids) != len(concept.parts):
            continue
        # Tool candidates: first eid matching any concept in
        # crafted_with. Skip when crafted_with is empty.
        instrument_eid: Optional[str] = None
        if concept.crafted_with:
            for tool_concept in concept.crafted_with:
                cand = next(
                    (eid for eid in by_concept.get(tool_concept, ())
                     if eid != theme_eid and eid not in part_eids),
                    None)
                if cand is not None:
                    instrument_eid = cand
                    break
        for agent_eid in animate_eids:
            if agent_eid == theme_eid or agent_eid in part_eids:
                continue
            if agent_eid == instrument_eid:
                continue
            roles = {
                "agent": agent_eid,
                "theme": theme_eid,
                "parts": tuple(part_eids),
            }
            if instrument_eid is not None:
                roles["instrument"] = instrument_eid
            pres, effs = _ground_action_facts(
                fari, roles, lex, rule_effects, facts=state_facts)
            if not effs:
                continue
            effs = _filter_forbidden_effs(effs, trace, lex)
            if not effs:
                continue
            # Per-part state requirements (e.g. teo's akvo must be
            # bolanta). The planner has to achieve these property
            # values before fari can fire — typically via a separate
            # verb that sets the slot (boli for temperature=bolanta).
            # Fact-tuple key is "prop" (relaxed-graph convention), not
            # "property" (goal tuple convention).
            for part_spec, part_eid in zip(concept.parts, part_eids):
                for slot, allowed in part_spec.requires.items():
                    if not allowed:
                        continue
                    pres = pres | {
                        ("prop", part_eid, slot, v) for v in allowed}
            out.append((fari, roles, pres, effs))
    return out


def _build_consumer_index(actions, derivations, table: FactTable):
    """Inverted index: fact_id → list of (base_cost, pres_fids,
    effs_fids, cid) consumers that need this fact in their pres.
    Actions have base_cost=1, derivations have base_cost=0. Used
    by the worklist h_add relaxation so each fact addition only
    re-checks the consumers that actually care about it.

    Pres/effs are translated through `table.id_for` once at build
    time; the h_add loop and back-walk then operate on `int` fact
    IDs all the way through (no tuple hashing on the hot path).

    Returns (fact_to_consumers, all_consumers, cid_to_info,
    pres_len) — the latter two are per-plan invariants reused
    across the ~100 h_add calls in one search:
      - cid_to_info: list indexed by cid → (verb_lemma_or_None,
        roles, pres_fids, effs_fids). Used by h_add's helpful-
        action backtrack — pres_fids feeds the back-walk stack.
      - pres_len: list indexed by cid → len(pres). Used to seed
        unsat[cid] each h_add call without re-iterating all_consumers.
    """
    fact_to: dict[int, list] = {}
    all_consumers: list = []
    cid_to_info: list = []
    pres_len: list = []
    cid = 0
    id_for = table.id_for
    for action, roles, pres, effs in actions:
        pres_fids = tuple({id_for(p) for p in pres})
        effs_fids = tuple({id_for(e) for e in effs})
        consumer = (1, pres_fids, effs_fids, cid)
        all_consumers.append(consumer)
        cid_to_info.append((action.lemma, roles, pres_fids, effs_fids))
        pres_len.append(len(pres_fids))
        for pid in pres_fids:
            fact_to.setdefault(pid, []).append(consumer)
        cid += 1
    for _, _, pres, effs in derivations:
        pres_fids = tuple({id_for(p) for p in pres})
        effs_fids = tuple({id_for(e) for e in effs})
        consumer = (0, pres_fids, effs_fids, cid)
        all_consumers.append(consumer)
        cid_to_info.append((None, None, pres_fids, effs_fids))
        pres_len.append(len(pres_fids))
        for pid in pres_fids:
            fact_to.setdefault(pid, []).append(consumer)
        cid += 1
    return fact_to, all_consumers, cid_to_info, pres_len


def _heuristic_and_helpful(
    goal_fact_id: int, facts_mask: int, consumer_index,
) -> tuple[int, set]:
    """h_FF over the relaxed planning graph using a worklist with
    an inverted fact_id→consumers index. Also returns the "helpful
    actions" set — verbs whose grounded effects appear on the
    cheapest path back from the goal in the relaxed plan.

    Operates entirely on fact IDs / bitmaps — `consumer_index` is
    built mask-native by `_build_consumer_index(_, _, table)`,
    `facts_mask` is the bitmap state, `goal_fact_id` is the
    single goal literal's interned ID. The h_FF graph and the
    helpful-action back-walk both run on integer keys: no tuple
    hashing on the hot path.

    Initial: cost(initial_facts) = 0. For each consumer c (action
    or derivation) whose pres are all known: tentative_cost =
    c.base + sum(cost[p] for p in c.pres). For each effect e: if
    tentative_cost < cost[e], update cost[e] and remember the
    producer. Terminates when the worklist is empty.

    Helpful action extraction: walk back from the goal through
    cheapest producers; collect (verb, roles) pairs of action
    producers (skipping derivation pseudo-actions). The forward
    search only expands these — typically 3-8 out of ~80
    applicable.

    Returns (h, helpful_actions) — h is 0 if goal holds, INF if
    unreached, else the count of unique action cids on the
    relaxed plan back from the goal.
    """
    if facts_mask & (1 << goal_fact_id):
        return 0, set()

    fact_to, all_consumers, cid_to_info, pres_len = consumer_index

    # Layer 0: state facts at cost 0.
    cost: dict = {fid: 0 for fid in iter_bits(facts_mask)}
    # producer[fid] = cid of the consumer that achieved cost[fid].
    producer: dict = {}
    # producers_cheapest[fid] = consumer cids that achieved the
    # current cheapest cost for that fact. On strict cost
    # improvement, reset to [new_cid]; on tie, append. Used for
    # both h_FF count and the helpful set.
    producers_cheapest: dict = {}

    from collections import deque
    # Counter-based h_add propagation. Maintain per-consumer:
    #   unsat[cid] = number of pres still at INF
    #   pre_sum[cid] = sum of cost[p] for known pres
    # List-indexed-by-cid (sequential from _build_consumer_index)
    # — faster than dict because the consumer set is the same
    # across all h_add calls in one plan; only values reset.
    n_consumers = len(all_consumers)
    pre_sum = [0] * n_consumers
    unsat = pres_len[:]
    ready_consumers: deque = deque(
        cid_ for cid_, u in enumerate(unsat) if u == 0)

    # Seed: each cost-0 fact decrements unsat for its consumers.
    # pre_sum stays 0 since cost is 0.
    for fid in cost:
        for _base, _pres, _effs, cid_ in fact_to.get(fid, ()):
            unsat[cid_] -= 1
            if unsat[cid_] == 0:
                ready_consumers.append(cid_)

    iter_cap = 200_000
    iters = 0
    INF = _HEURISTIC_INF
    while ready_consumers and iters < iter_cap:
        iters += 1
        cid_ = ready_consumers.popleft()
        base, _pres, effs, _ = all_consumers[cid_]
        new_cost = base + pre_sum[cid_]
        for e in effs:
            prev = cost.get(e, INF)
            if new_cost < prev:
                cost[e] = new_cost
                producer[e] = cid_
                producers_cheapest[e] = [cid_]
                if prev >= INF:
                    # First time e is reached.
                    for _b2, _p2, _e2, cid2_ in fact_to.get(e, ()):
                        unsat[cid2_] -= 1
                        pre_sum[cid2_] += new_cost
                        if unsat[cid2_] == 0:
                            ready_consumers.append(cid2_)
                else:
                    # Cost improvement; pres already counted.
                    delta = prev - new_cost
                    for _b2, _p2, _e2, cid2_ in fact_to.get(e, ()):
                        pre_sum[cid2_] -= delta
                        if unsat[cid2_] == 0:
                            ready_consumers.append(cid2_)
            elif new_cost == prev:
                producers_cheapest.setdefault(e, []).append(cid_)

    if cost.get(goal_fact_id, INF) >= INF:
        return INF, set()

    # h_FF: walk back from the goal through CHEAPEST producers,
    # collecting unique action cids. h = |unique actions|.
    relaxed_action_cids: set = set()
    visited_cheapest: set = set()
    stack = [goal_fact_id]
    while stack:
        fid = stack.pop()
        if fid in visited_cheapest:
            continue
        visited_cheapest.add(fid)
        for cid_ in producers_cheapest.get(fid, ()):
            if cid_ < 0 or cid_ >= len(cid_to_info):
                continue
            verb, _roles, pres, _effs = cid_to_info[cid_]
            if verb is not None:
                relaxed_action_cids.add(cid_)
            for p in pres:
                if p not in visited_cheapest:
                    stack.append(p)
            break  # one producer per fact for h_FF
    h_ff = len(relaxed_action_cids)

    # Helpful set: walk back through cheapest producers, collect
    # verbs. (Earlier versions also walked sub-cheapest producers
    # via a separate `producers_all` map for diversity; profiling
    # showed that map's setdefault was a hot loop and the broader
    # set didn't materially change search outcomes.)
    helpful: set = set()
    visited_all: set = set()
    stack = [goal_fact_id]
    while stack:
        fid = stack.pop()
        if fid in visited_all:
            continue
        visited_all.add(fid)
        # Take ONE cheapest producer per fact (mirroring h_FF).
        # Adding every alternative to helpful inflates branching
        # factor without changing reachability.
        for cid_ in producers_cheapest.get(fid, ()):
            if cid_ < 0 or cid_ >= len(cid_to_info):
                continue
            verb, roles, pres, _effs = cid_to_info[cid_]
            if verb is not None:
                helpful.add((verb, frozenset(
                    (k, tuple(v) if isinstance(v, list) else v)
                    for k, v in roles.items())))
            for p in pres:
                if p not in visited_all:
                    stack.append(p)
            break

    return h_ff, helpful


# ---------- search ----------

def _trace_signature(trace: Trace) -> tuple:
    """Hashable fingerprint of a trace for visited-state dedup. Uses
    the sorted (eid, frozenset(items)) tuples for entities + a sorted
    tuple of (relation, args) for relations + the event id list."""
    ents = tuple(sorted(
        (eid, tuple(sorted(
            (k, tuple(v) if isinstance(v, list) else v)
            for k, v in ent.properties.items())))
        for eid, ent in trace.entities.items()))
    rels = tuple(sorted(
        (r.relation, tuple(r.args)) for r in trace.relations))
    evs = tuple(ev.id for ev in trace.events)
    return (ents, rels, evs)


def _infra_prespawn(action, role_map, trace, lex, derivations, resolver,
                     rule_effects, derivable_slots, depth):
    """Recursive infrastructure pre-spawn: ensure `action` is plannable
    by walking its preconditions and role.properties, spawning enabling
    entities for derivable requirements.

    Only recurses on RelationPreconditions whose producer verbs have
    role-properties on a derivable slot — i.e., chains that need
    infrastructure (perception → illumination → lampo). Skips normal
    action chains the planner can subgoal directly (havi via preni,
    samloke via iri)."""
    if depth <= 0:
        return
    from ..schemas import RelationPrecondition
    from ..dsl.implications import PropertyImplication
    derived = _cached_compute_derived_state(trace, derivations, lex)
    facts = _state_facts(trace, derived, lex)

    # (a) Role.property requirements: derivable + not already true.
    for role_spec in action.roles:
        eid = role_map.get(role_spec.name)
        if eid is None:
            continue
        # List-kind roles bind to a list of eids; the property
        # requirements apply to each. We don't recurse for derivation
        # planting on list roles — the parts already exist as scene
        # entities with their declared properties.
        if isinstance(eid, (list, tuple)):
            continue
        ent_type = trace.entities[eid].entity_type
        for slot, vals in (role_spec.properties or {}).items():
            if not vals:
                continue
            target_val = vals[0]
            if ("prop", eid, slot, target_val) in facts:
                continue
            applicable = derivable_slots.get(slot, ())
            if not any(lex.types.is_subtype(ent_type, t)
                       for t in applicable):
                continue
            for d in derivations:
                if not any(isinstance(imp, PropertyImplication)
                           and imp.slot == slot
                           and imp.value == target_val
                           for imp in d.implies):
                    continue
                _spawn_for_derivation(
                    d, eid, trace, lex, derivations, resolver,
                    rule_effects, derivable_slots, depth)
                break

    # (b) RelationPreconditions whose synthesized producer needs infra.
    sym = _symmetric_relations(lex)
    for pc in action.preconditions:
        if not isinstance(pc, RelationPrecondition):
            continue
        eids = tuple(role_map.get(r) for r in pc.roles)
        if any(e is None for e in eids):
            continue
        if ("rel", pc.rel, _canon_rel(pc.rel, eids, sym)) in facts:
            continue
        # Find a producer verb whose synthesized adds include this relation.
        for prod_verb, entry in rule_effects.items():
            chose = None
            for relation, role_args in entry.get("adds", []):
                if relation != pc.rel:
                    continue
                prod_action = lex.actions.get(prod_verb)
                if prod_action is None:
                    continue
                chose = (prod_action, role_args)
                break
            if chose is None:
                continue
            prod_action, role_args = chose
            # Map producer's role args to goal's eids positionally.
            prod_role_map: dict = {}
            for goal_role, prod_role in zip(pc.roles, role_args):
                if goal_role in role_map:
                    prod_role_map[prod_role] = role_map[goal_role]
            # Spawn producer's missing roles.
            derived_now = _cached_compute_derived_state(
                trace, derivations, lex)
            exclude = set(prod_role_map.values())
            for rs in prod_action.roles:
                if rs.name in prod_role_map:
                    continue
                filler = _find_role_filler(
                    rs, trace, lex, derived=derived_now,
                    exclude=exclude, action=prod_action,
                    role_name=rs.name)
                if filler is None:
                    filler = resolver(
                        rs, trace, lex, exclude,
                        action=prod_action, role_name=rs.name)
                if filler is not None:
                    prod_role_map[rs.name] = filler
                    exclude.add(filler)
            # Recurse: ensure producer's own preconds + role-props.
            _infra_prespawn(
                prod_action, prod_role_map, trace, lex, derivations,
                resolver, rule_effects, derivable_slots, depth - 1)
            break  # one producer per precondition


def _spawn_for_derivation(d, eid, trace, lex, derivations, resolver,
                            rule_effects, derivable_slots, depth):
    """For derivation `d` (which produces a property on eid), walk its
    given clauses and spawn supporting entities for any EntityPattern
    & bind() that binds a NON-target var (i.e. the supporting entity)."""
    if depth <= 0:
        return
    from ..dsl.patterns import (
        AndPattern, BindPattern, EntityPattern, Var,
    )
    from ..dsl.implications import PropertyImplication

    target_var = None
    for imp in d.implies:
        if isinstance(imp, PropertyImplication) and isinstance(imp.entity, Var):
            target_var = imp.entity
            break
    if target_var is None:
        return

    from ..dsl.patterns import RelPattern

    def _walk_eps(pat):
        if isinstance(pat, AndPattern):
            if (isinstance(pat.left, EntityPattern)
                    and isinstance(pat.right, BindPattern)):
                yield (pat.left, pat.right.target)
            elif (isinstance(pat.right, EntityPattern)
                    and isinstance(pat.left, BindPattern)):
                yield (pat.right, pat.left.target)
            else:
                yield from _walk_eps(pat.left)
                yield from _walk_eps(pat.right)
        # Descend into RelPattern arg_patterns. The lieable bind for
        # animate_lying_when_on_lieable lives inside the sur rel's
        # container arg, which the AndPattern-only walker missed —
        # so the planner couldn't pre-spawn a lit/sofo for sleep
        # drives.
        if isinstance(pat, RelPattern):
            for arg_pat in pat.arg_patterns.values():
                yield from _walk_eps(arg_pat)

    class _SynthRole:
        def __init__(self, type_, properties):
            self.type = type_
            self.properties = properties
            self.name = "<infra>"

    for given in d.given:
        for ep, bv in _walk_eps(given):
            if bv is target_var:
                continue
            constraints = ep.constraints
            type_ = constraints.get("type", "physical")
            props: dict = {}
            for k, v in constraints.items():
                if k in ("type", "concept", "has_suffix"):
                    continue
                if isinstance(v, str):
                    props[k] = [v]
            # Skip if a matching entity exists. Derived properties
            # don't show in concept.properties directly; for now
            # treat any matching-type entity as fulfilling the
            # static-property part — if the entity needs further
            # derived state, we'll spawn another via recursion.
            matched = False
            for other_eid, other_ent in trace.entities.items():
                if not lex.types.is_subtype(other_ent.entity_type, type_):
                    continue
                ok = True
                for slot, vals in props.items():
                    if slot in derivable_slots:
                        continue  # check derivation separately
                    cvals = other_ent.properties.get(slot, [])
                    if not (set(vals) & set(cvals)):
                        ok = False
                        break
                if ok:
                    matched = True
                    # If derived properties needed, recurse on this
                    # existing entity to ensure they become reachable.
                    for slot, vals in props.items():
                        if slot in derivable_slots and vals:
                            _ensure_prop_reachable(
                                other_eid, slot, vals[0], trace, lex,
                                derivations, resolver, rule_effects,
                                derivable_slots, depth - 1)
                    break
            if matched:
                continue
            # Need a fresh entity.
            synth = _SynthRole(type_, props)
            filler = resolver(
                synth, trace, lex, set(),
                action=None, role_name=None)
            if filler is None:
                continue
            # The spawner randomizes varies-slot values; force the
            # property values our derivation needs (e.g. power_state=
            # aktiva on a freshly-spawned lampo). Without this the
            # planner has to subgoal ŝalti, which it can do — but in
            # practice the deeper chain blows past the search cap.
            # No-op for slots that aren't varies (the spawner just
            # left the concept's declared value).
            ent = trace.entities.get(filler)
            if ent is not None:
                for slot, vals in props.items():
                    if not vals:
                        continue
                    slot_def = lex.slots.get(slot)
                    if slot_def is not None and slot_def.varies:
                        ent.set_property(slot, vals[0])
            # Recurse on derivable properties of the new entity.
            for slot, vals in props.items():
                if slot in derivable_slots and vals:
                    _ensure_prop_reachable(
                        filler, slot, vals[0], trace, lex, derivations,
                        resolver, rule_effects, derivable_slots, depth - 1)


def _ensure_prop_reachable(eid, slot, value, trace, lex, derivations,
                              resolver, rule_effects, derivable_slots, depth):
    """Ensure ("prop", eid, slot, value) becomes derivable in trace.
    Tries every derivation producing (slot, value), not just the
    first — different derivations apply to different entity contexts
    (outdoor_luma_during_day vs location_lit_by_active_lamp both
    produce lit_state=luma but for different entity shapes)."""
    if depth <= 0:
        return
    from ..dsl.implications import PropertyImplication
    derived = _cached_compute_derived_state(trace, derivations, lex)
    if ("prop", eid, slot, value) in _state_facts(trace, derived, lex):
        return
    for d in derivations:
        if not any(isinstance(imp, PropertyImplication)
                   and imp.slot == slot and imp.value == value
                   for imp in d.implies):
            continue
        _spawn_for_derivation(
            d, eid, trace, lex, derivations, resolver,
            rule_effects, derivable_slots, depth)
        # Re-check; stop once the property becomes reachable.
        derived = _cached_compute_derived_state(trace, derivations, lex)
        if ("prop", eid, slot, value) in _state_facts(trace, derived, lex):
            return


def _prespawn_for_goal(goal, trace, lex, rules, derivations, resolver):
    """For each action whose effects could produce `goal`, ensure all
    role types have at least one matching entity. Missing roles are
    filled by calling `resolver`. Mirrors what the backward planner's
    `_find_role_filler` does lazily during search.

    Then runs `_infra_prespawn` recursively to handle
    derivation-chain infrastructure (e.g., spawn a lampo when the
    chain requires illuminated agents)."""
    rule_effects = _RULE_EFFECTS_CACHE.get(id(lex))
    if rule_effects is None:
        rule_effects = _build_rule_effects_index(rules, lex)
        _RULE_EFFECTS_CACHE[id(lex)] = rule_effects
    # Pre-computed at lex-load time by the concept index — same
    # value `fully_derivable_slots(lex, RUNTIME_DERIVATIONS)` returns.
    derivable_slots = lex.concept_index.derivable or {}

    if goal[0] == "event_fire":
        # Specific verb already chosen; just fill any unbound roles.
        _, gv_verb, gv_bindings_frozen = goal
        gv_bindings = dict(gv_bindings_frozen)
        action = lex.actions.get(gv_verb)
        if action is None:
            return
        derived_now = _cached_compute_derived_state(
            trace, derivations, lex)
        # Flatten list-valued bindings (fari.parts) when building the
        # exclude set — set members must be hashable.
        exclude: set = set()
        for v in gv_bindings.values():
            if isinstance(v, (list, tuple)):
                exclude.update(v)
            else:
                exclude.add(v)
        role_map = dict(gv_bindings)
        # Concept-constraints from rule given clauses (e.g. mezuri's
        # theme must model the slot named by instrument's mezuras).
        # The constraint is symmetric: when filling the entity_role,
        # filter candidates that model source.concept.field's value;
        # when filling the source_role, filter candidates whose
        # `field` value is a slot entity.concept models. We can
        # apply the filter only when the PEER role is already bound.
        constraints = (rule_effects.get(gv_verb, {})
                       .get("concept_constraints", ()))
        from ..dsl.introspect import concept_models_slot
        from ..dsl.rules import runtime_derivations_for
        for role_spec in action.roles:
            if role_spec.name in gv_bindings:
                continue
            # Optional roles (fari.instrument when crafted_with is
            # empty) are intentionally left unbound — don't burn the
            # spawner budget hunting for a random artifact.
            if getattr(role_spec, "optional", False):
                continue
            # List-kind roles must be pre-bound by the goal sampler;
            # the planner doesn't synthesize variadic lists.
            if getattr(role_spec, "kind", "single") == "list":
                continue
            # Build per-role concept_filter from any constraint whose
            # peer role is already bound. Filter chain (multiple
            # constraints on same role): candidate must satisfy ALL.
            current = role_spec.name
            filter_clauses: list = []
            for entity_role, source_role, field_name in constraints:
                if current == entity_role and source_role in role_map:
                    src_ent = trace.entities.get(role_map[source_role])
                    if src_ent is None:
                        continue
                    src_concept = lex.concepts.get(src_ent.concept_lemma)
                    if src_concept is None:
                        continue
                    vals = src_concept.properties.get(field_name)
                    if not vals:
                        continue
                    slot_name = vals[0]
                    filter_clauses.append(
                        ("entity_must_model", slot_name))
                elif current == source_role and entity_role in role_map:
                    ent = trace.entities.get(role_map[entity_role])
                    if ent is None:
                        continue
                    tgt_concept = lex.concepts.get(ent.concept_lemma)
                    if tgt_concept is None:
                        continue
                    filter_clauses.append(
                        ("source_field_modeled_by_entity",
                         field_name, tgt_concept))
            concept_filter = None
            if filter_clauses:
                derivs = runtime_derivations_for(lex)
                def _filter(c, _cl=filter_clauses, _ds=derivs):
                    c_def = lex.concepts.get(c)
                    if c_def is None:
                        return False
                    for clause in _cl:
                        if clause[0] == "entity_must_model":
                            if not concept_models_slot(
                                    c_def, clause[1], lex, _ds):
                                return False
                        else:
                            _, field_name, ent_concept = clause
                            vals = c_def.properties.get(field_name)
                            if not vals:
                                return False
                            if not concept_models_slot(
                                    ent_concept, vals[0], lex, _ds):
                                return False
                    return True
                concept_filter = _filter
            filler = _find_role_filler(
                role_spec, trace, lex, derived=derived_now,
                exclude=exclude, action=action,
                role_name=role_spec.name)
            if filler is None:
                filler = resolver(role_spec, trace, lex, exclude,
                                  action=action,
                                  role_name=role_spec.name,
                                  concept_filter=concept_filter)
            if filler is not None:
                role_map[role_spec.name] = filler
                exclude.add(filler)
        _infra_prespawn(
            action, role_map, trace, lex, derivations, resolver,
            rule_effects, derivable_slots, depth=6)
        return
    if goal[0] != "property":
        return
    _, target_eid, slot, value = goal
    target_ent = trace.entities.get(target_eid)
    if target_ent is None:
        return
    derived_now = _cached_compute_derived_state(trace, derivations, lex)
    for action in lex.actions.values():
        # Does this action's effect spec match the goal?
        produces = False
        target_role_name = None
        for eff in action.effects:
            if eff.property == slot and eff.value == value:
                produces = True
                target_role_name = eff.target_role
                break
        if not produces:
            continue
        # Check target role's type matches our target entity.
        target_role_spec = next(
            (r for r in action.roles if r.name == target_role_name), None)
        if target_role_spec is None:
            continue
        if not lex.types.is_subtype(
                target_ent.entity_type, target_role_spec.type):
            continue
        # For each non-target role, check whether an in-trace entity
        # already fills it; if not, ask the resolver to spawn one.
        exclude = {target_eid}
        role_map = {target_role_name: target_eid}
        for role_spec in action.roles:
            if role_spec.name == target_role_name:
                continue
            filler = _find_role_filler(
                role_spec, trace, lex, derived=derived_now,
                exclude=exclude, action=action,
                role_name=role_spec.name)
            if filler is None:
                # Let the spawner closure's prefer_scene_p decide
                # whether to scene-prefer or go to natural habitat.
                # Default (1.0) reproduces BP behavior; <1.0 mixes in
                # fetch chains.
                filler = resolver(role_spec, trace, lex, exclude,
                                  action=action, role_name=role_spec.name)
            if filler is not None:
                role_map[role_spec.name] = filler
                exclude.add(filler)
        _infra_prespawn(
            action, role_map, trace, lex, derivations, resolver,
            rule_effects, derivable_slots, depth=6)


def plan_for_goal(
    drive, initial_trace: Trace, lex, rules, derivations,
    *, max_states: int = 200, max_plan_length: int = 12,
    entity_resolver=None, rng=None, exclude_verbs=None,
) -> Optional[list]:
    """Greedy best-first forward search. Returns a plan (list of
    (verb, roles) steps) or None.

    `max_states` bounds total state expansions (latency cap).
    `max_plan_length` bounds plan depth (prevents infinite chains
    when goal is unreachable but heuristic plateaus).
    `entity_resolver`: optional callable matching the backward
    planner's `_ENTITY_RESOLVER` signature. When provided, pre-spawns
    missing role-fillers for goal-producing actions so the
    forward search sees the same scene as backward search would.
    `rng`: optional `random.Random`. When supplied, each transition's
    cost gets a small uniform[0, 0.5] perturbation so equal-and
    near-equal-length plans are shuffled in the open list. Identical
    runs without rng remain deterministic and pick the shortest plan;
    runs with rng get narrative diversity across seeds (a bicycle vs.
    a walk for the same destination, etc.) without breaking weighted
    A*'s correctness — only optimality.

    `exclude_verbs`: optional set of action lemmas to filter out of
    grounding. Used by the seeder to force a specific producer for
    the goal-fact: when the seeder commits to `veturi` for a
    same-place drive, it passes the alternative producers (`iri`,
    `kuri`, `rajdi`, `flugi`, `veni`) here so the planner has to
    construct the eniri→veturi chain instead of substituting a
    cheaper direct action. Support verbs (preni, eniri, etc.) for
    preconditions remain available — only same-goal alternatives
    get filtered.

    `max_states=0` is the reachability-only mode: do the initial
    setup (prespawn, ground actions+derivations, compute h on the
    relaxed graph) and return `[]` iff h is finite (goal reachable),
    `None` iff h is INF (unreachable). No search runs. Sampler-side
    callers use this as a viability gate — see the goal_sampler's
    relation-drive branch."""
    goal = _drive_to_goal(drive, initial_trace, lex)
    if goal is None:
        return None

    # Augment the passed derivations with lex-derived dynamic ones
    # (scias_lokon-per-locative-rel). Callers don't carry the lex
    # at module-load time, so they pass the static base list; the
    # planner extends it here. Idempotent — skip names already
    # present so a caller that already includes them isn't doubled.
    from ..dsl.rules import make_scias_lokon_derivations
    _extra_derivs = make_scias_lokon_derivations(lex)
    if _extra_derivs:
        _existing_names = {getattr(d, "name", None) for d in derivations}
        _new = [d for d in _extra_derivs
                if getattr(d, "name", None) not in _existing_names]
        if _new:
            derivations = list(derivations) + _new

    initial_derived = _cached_compute_derived_state(
        initial_trace, derivations, lex)
    if _goal_satisfied(goal, initial_trace, initial_derived, lex):
        return []

    if entity_resolver is not None:
        _prespawn_for_goal(
            goal, initial_trace, lex, rules, derivations, entity_resolver)
        # Re-derive after spawns may have added entities.
        initial_derived = _cached_compute_derived_state(
            initial_trace, derivations, lex)
    # Seed the per-entity slot snapshot used by `_action_delta_mask` to
    # filter forbidden AddRelation effects without a trace handle.
    # Captures static properties only (varies-slot randomization is
    # already baked into entity.properties at this point).
    _seed_entity_props_for_delta(initial_trace, lex)

    # Ground actions once for the initial entity set; reuse across
    # all heuristic + applicability calls. Event firing rarely
    # creates new entities in our domain, so the grounding stays
    # valid for the search. Invalidation: not needed for the POC.
    rule_effects = _RULE_EFFECTS_CACHE.get(id(lex))
    if rule_effects is None:
        rule_effects = _build_rule_effects_index(rules, lex)
        _RULE_EFFECTS_CACHE[id(lex)] = rule_effects
    # Per-trace grounding cache: regress_for_goal often calls
    # plan_for_goal twice on the same trace state (once via
    # _drive_is_h_reachable as max_states=0, once via execute_drive).
    # `_ground_all_actions` + `_ground_constructable_actions` is
    # ~400ms per call in the profile; caching by a fingerprint of
    # entities + relations (which fully determines grounding)
    # eliminates the second pass at a few μs of fingerprint cost.
    cache = getattr(initial_trace, "_fwd_planner_grounding_cache", None)
    fp = (id(lex), id(rule_effects), len(initial_trace.entities),
          frozenset((eid, ent.entity_type)
                    for eid, ent in initial_trace.entities.items()),
          frozenset((r.relation, r.args) for r in initial_trace.relations))
    if cache is not None and cache[0] == fp:
        grounded = list(cache[1])
    else:
        grounded = _ground_all_actions(
            initial_trace, lex, initial_derived, rule_effects)
        grounded.extend(_ground_constructable_actions(
            initial_trace, lex, rule_effects, derived=initial_derived))
        try:
            object.__setattr__(
                initial_trace, "_fwd_planner_grounding_cache",
                (fp, tuple(grounded)))
        except Exception:
            pass
    if exclude_verbs:
        grounded = [g for g in grounded if g[0].lemma not in exclude_verbs]
    # Goal-aware action + derivation pruning. Walk backward from the
    # goal through action effects, rule-effects adds, preconditions,
    # AND derivation implications to find the verbs + derivations
    # that could plausibly contribute. Drop the rest from grounding
    # so the search heap doesn't waste slots on actions whose
    # effects can never land on the goal property, and the relaxed
    # graph doesn't ground derivations that produce facts no goal-
    # relevant action consumes.
    _walk_result = _goal_reachable_walk(
        goal, lex, rule_effects, derivations)
    if _walk_result is not None:
        relevant_verbs, relevant_derivs = _walk_result
        grounded = [g for g in grounded if g[0].lemma in relevant_verbs]
    else:
        relevant_derivs = None
    # Restrict derivation var domains to entities that actually
    # participate in some action's pres/effs. This is the cubic
    # samloke-chain bound: a scene with 19 non-inanimate entities
    # yields 19³ samloke bindings per chain derivation; restricting
    # to action-mentioned eids (typically 6-10) cuts that ~5-10×.
    relevant_entities: set = set()
    for _action, _roles, pres, effs in grounded:
        for f in pres:
            if f[0] == "rel":
                relevant_entities.update(f[2])
            elif f[0] == "prop":
                relevant_entities.add(f[1])
        for f in effs:
            if f[0] == "rel":
                relevant_entities.update(f[2])
            elif f[0] == "prop":
                relevant_entities.add(f[1])
    # Include goal-target entity even if no action mentions it
    # (uncommon but possible for synthetic goals).
    if goal[0] == "property":
        relevant_entities.add(goal[1])
    elif goal[0] == "relation":
        relevant_entities.update(goal[2])
    # Skip cubic samloke chain derivations in the heuristic — they
    # alone contribute 19³ ≈ 6000 bindings each on a 19-entity scene
    # (4 × 6000 = 24K of 30K total), and the heuristic doesn't need
    # exact samloke chain reachability: the direct 2-var variants
    # (en_implies_samloke_with_container, havi_implies_samloke_…,
    # sur_implies_samloke_…) are sufficient for the relaxed plan
    # since iri/eniri/preni still establish the en/havi facts.
    _SKIP_IN_HEURISTIC = {
        # samloke_propagates_through_artifact_parts stays skipped —
        # 3K bindings for negligible yield gain. The en/sur chains
        # are re-enabled but with an animate-endpoint filter in
        # _ground_derivations that drops object×object×object combos
        # (~6× fewer bindings per chain), making them cheap enough.
        "samloke_propagates_through_artifact_parts",
    }
    # Derivations whose `when` is a literal-constrained relation
    # pattern (currently: scias_lokon-via-scias-<locative>) are
    # covered by `_synthesize_implications_from_derivations` in
    # `_build_rule_effects_index`. Grounding them here too would
    # waste cycles AND miss combinations where the existential arg
    # (the locative-container) isn't in `relevant_entities`. Auto-
    # skip them by detecting the literal-constrained shape, so adding
    # a new derivation of this shape is automatically handled.
    from ..dsl.patterns import LiteralValuePattern, RelPattern
    for d in derivations:
        when = getattr(d, "when", None)
        if isinstance(when, RelPattern) and any(
                isinstance(ap, LiteralValuePattern)
                for ap in when.arg_patterns.values()):
            _SKIP_IN_HEURISTIC = _SKIP_IN_HEURISTIC | {d.name}
    heuristic_derivations = [
        d for d in derivations
        if getattr(d, "name", None) not in _SKIP_IN_HEURISTIC]
    # Goal-aware derivation pruning: keep only derivations whose
    # `implies` produces a fact reachable backward from the goal.
    # Derivations outside this set fire freely but their outputs are
    # consumed by nothing on the goal's chain, so they cannot lower
    # the relaxed plan's cost. Skip the filter when the walk
    # declined to analyze the goal shape.
    if relevant_derivs is not None:
        heuristic_derivations = [
            d for d in heuristic_derivations if d.name in relevant_derivs]
    grounded_derivs = _ground_derivations(
        heuristic_derivations, initial_trace, lex,
        relevant_entities=relevant_entities,
        rule_effects=rule_effects)

    # Event-fire goals: the goal is "fire verb V with these role
    # bindings". Lacking direct property/relation effects on most
    # speech-act / perception verbs, we synthesize an `event_fired`
    # pseudo-fact that the goal verb's grounding produces. The
    # relaxed graph then has something to aim for; the search
    # satisfies when the fact lands in the state.
    if goal[0] == "event_fire":
        _, gv_verb, gv_bindings_frozen = goal
        gv_bindings = dict(gv_bindings_frozen)
        event_fired_fact = ("event_fired", gv_verb, gv_bindings_frozen)
        action_obj = lex.actions.get(gv_verb)
        if action_obj is None:
            return None
        # Enumerate role bindings for gv_verb. Constrained roles are
        # forced to the drive's eid; unconstrained roles enumerate
        # all type-compatible entities in trace.
        #
        # List-kind roles (fari.parts) carry a tuple of eids in the
        # drive; we keep the list intact (not enumerable as a single
        # entity) and downstream grounding/expansion handles it.
        # Optional unbound roles get a None placeholder so the combo
        # enumeration emits a single grounding where the role is None.
        per_role: list = []
        ok = True
        for role_spec in action_obj.roles:
            forced = gv_bindings.get(role_spec.name)
            if isinstance(forced, (list, tuple)):
                # List role; keep as a single combinatorial element.
                per_role.append([list(forced)])
                continue
            if forced is not None:
                ent = initial_trace.entities.get(forced)
                if ent is None or not lex.types.is_subtype(
                        ent.entity_type, role_spec.type):
                    ok = False
                    break
                per_role.append([forced])
                continue
            if getattr(role_spec, "optional", False):
                # Unbound optional role: emit one grounding with this
                # role absent from the binding dict.
                per_role.append([None])
                continue
            # kind="relation" / "from_precondition": pull values from
            # a string pool, same dispatch as _ground_all_actions.
            kind = getattr(role_spec, "kind", "single")
            if kind == "relation":
                pool = (list(role_spec.allowed_values)
                        if getattr(
                            role_spec, "allowed_values", None)
                        else list(lex.relations.keys()))
                if not pool:
                    ok = False
                    break
                per_role.append(pool)
                continue
            if kind == "from_precondition":
                src_rel_name = getattr(
                    role_spec, "from_precondition", None)
                pos = getattr(
                    role_spec, "from_precondition_position", None)
                src_rel = lex.relations.get(src_rel_name) \
                    if src_rel_name else None
                src_kind = "entity"
                if src_rel is not None and pos is not None:
                    kinds = (list(src_rel.arg_kinds)
                             if src_rel.arg_kinds
                             else ["entity"] * src_rel.arity)
                    if 0 <= pos < len(kinds):
                        src_kind = kinds[pos]
                if src_kind in ("literal", "slot"):
                    pool = (list(role_spec.allowed_values)
                            if getattr(
                                role_spec, "allowed_values", None)
                            else list(lex.relations.keys())
                            if src_kind == "literal"
                            else list(lex.slots.keys()))
                    if not pool:
                        ok = False
                        break
                    per_role.append(pool)
                    continue
                # entity source: fall through to entity enumeration
            cand = []
            for eid, ent in initial_trace.entities.items():
                if not lex.types.is_subtype(
                        ent.entity_type, role_spec.type):
                    continue
                cand.append(eid)
            if not cand:
                ok = False
                break
            per_role.append(cand)
        if ok:
            for combo in itertools.product(*per_role):
                # Hashable dedup: skip combos where two scalar roles
                # bind the same entity. List/None elements don't
                # participate in this check.
                scalar = [c for c in combo
                           if not isinstance(c, list) and c is not None]
                if len(set(scalar)) != len(scalar):
                    continue
                roles = {}
                for r, e in zip(action_obj.roles, combo):
                    if e is None:
                        continue
                    roles[r.name] = e
                pres, effs = _ground_action_facts(
                    action_obj, roles, lex, rule_effects,
                    facts=_state_facts(initial_trace, initial_derived, lex))
                effs = set(effs) | {event_fired_fact}
                grounded.append((action_obj, roles, pres, effs))

    # Goal-aware filter: shrink the consumer set to only those that
    # could reach the goal in the relaxed graph. Walk back from the
    # goal facts collecting any consumer whose effects intersect
    # with the reachable-fact frontier; recurse via their pres.
    # Reduces 1300+5000 → typically a few hundred consumers,
    # cutting per-heuristic-call cost ~5-10×.
    sym_goal = _symmetric_relations(lex)
    if goal[0] == "property":
        _, eid, slot, value = goal
        seed_goal = {("prop", eid, slot, value)}
    elif goal[0] == "relation":
        _, relation, args = goal
        seed_goal = {("rel", relation,
                      _canon_rel(relation, tuple(args), sym_goal))}
    elif goal[0] == "event_fire":
        seed_goal = {event_fired_fact}
    else:
        seed_goal = set()

    # Build effect→consumers index for the back-walk.
    effs_to: dict = {}
    for i, (_, _, _pres, effs) in enumerate(grounded):
        for e in effs:
            effs_to.setdefault(e, []).append(("action", i))
    for i, (_, _, _pres, effs) in enumerate(grounded_derivs):
        for e in effs:
            effs_to.setdefault(e, []).append(("deriv", i))

    relevant_act: set = set()
    relevant_deriv: set = set()
    visited_facts: set = set()
    stack = list(seed_goal)
    walk_iter = 0
    while stack and walk_iter < 50_000:
        walk_iter += 1
        f = stack.pop()
        if f in visited_facts:
            continue
        visited_facts.add(f)
        for kind, i in effs_to.get(f, ()):
            if kind == "action":
                if i in relevant_act:
                    continue
                relevant_act.add(i)
                _, _, pres, _ = grounded[i]
            else:
                if i in relevant_deriv:
                    continue
                relevant_deriv.add(i)
                _, _, pres, _ = grounded_derivs[i]
            for p in pres:
                if p not in visited_facts:
                    stack.append(p)

    grounded = [grounded[i] for i in sorted(relevant_act)]
    grounded_derivs = [grounded_derivs[i] for i in sorted(relevant_deriv)]

    # Slot scalar info for the fact-set incremental simulator —
    # needed to correctly displace prior values on scalar slots.
    slot_vocab: dict = {
        name: {"scalar": getattr(s, "scalar", True)}
        for name, s in lex.slots.items()
    }

    # Goal as fact for direct fact-set check.
    if goal[0] == "property":
        _, eid, slot, value = goal
        goal_fact = ("prop", eid, slot, value)
    elif goal[0] == "relation":
        _, relation, args = goal
        goal_fact = ("rel", relation,
                     _canon_rel(relation, tuple(args), sym_goal))
    elif goal[0] == "event_fire":
        goal_fact = event_fired_fact
    else:
        return None

    # Bitmap-state representation. The FactTable interns each fact
    # tuple into a stable bit position; the search then carries
    # state as a single `int`, subset checks become a 2-op AND/eq,
    # `_apply_delta_mask` rebuilds the next state without allocating
    # a frozenset. See ../fact_table.py for the table.
    table = fact_table_for(initial_trace)
    consumer_index = _build_consumer_index(
        grounded, grounded_derivs, table)
    grounded_pres_masks = [
        table.mask_for(pres) for _a, _r, pres, _e in grounded]
    grounded_effs_masks = [
        table.mask_for(effs) for _a, _r, _p, effs in grounded]
    goal_fact_id = table.id_for(goal_fact)
    goal_bit = 1 << goal_fact_id

    def heuristic_and_helpful(state_mask: int):
        # Fully bitmap-native: the consumer index is fact-id keyed,
        # cost/producer dicts use int keys, the back-walk walks
        # `pres_fids` lists. No tuple hashing on the hot path.
        return _heuristic_and_helpful(
            goal_fact_id, state_mask, consumer_index)

    # Fact-set incremental search on bitmap state. Each successor:
    # `(state & pres_mask) == pres_mask` for applicability, mask-form
    # delta via `_action_delta_mask`, apply via `_apply_delta_mask`
    # (which re-derives in-place on the int). Far faster than
    # `_simulate_from_scratch` per successor (~1ms vs ~9ms) at the
    # cost of skipping cascades / entity creation.
    H_WEIGHT = 2

    initial_mask = _state_facts_mask(
        initial_trace, initial_derived, lex)
    # Apply derivations to layer-0 to get full derived layer.
    initial_mask = _apply_delta_mask(
        initial_mask, 0, 0, grounded_derivs, slot_vocab, table)

    if initial_mask & goal_bit:
        return []

    initial_h, initial_helpful = heuristic_and_helpful(initial_mask)
    if initial_h >= _HEURISTIC_INF:
        return None
    if max_states == 0:
        # Reachability gate: h is finite, goal is reachable in the
        # relaxed graph. Caller wants the viability bit, not a plan.
        return []

    import heapq
    from collections import deque

    # Phase 1: Enforced Hill Climbing (FF-style). From the anchor
    # state, BFS through helpful actions only until a strictly-
    # h-improving state is found, then restart from that state.
    # Tunnels through plateaus that confound weighted A* (every
    # equal-f sibling gets explored before the anchor advances).
    # Falls back to weighted A* when EHC dead-ends (helpful set
    # incomplete) or hits its slice of the search budget.
    ehc_budget = max_states // 2
    anchor_mask = initial_mask
    anchor_h = initial_h
    anchor_helpful = initial_helpful
    anchor_plan: list = []
    ehc_visited: set = {initial_mask}
    expansions = 0
    while expansions < ehc_budget and anchor_h > 0:
        bfs: deque = deque(
            [(anchor_mask, anchor_plan, anchor_helpful)])
        found = None
        while bfs and expansions < ehc_budget:
            cur_mask, cur_plan, cur_helpful = bfs.popleft()
            expansions += 1
            if len(cur_plan) >= max_plan_length:
                continue
            for i, (action, roles, _pres, _effs) in enumerate(grounded):
                pres_mask = grounded_pres_masks[i]
                if (cur_mask & pres_mask) != pres_mask:
                    continue
                key = (action.lemma, frozenset(
                    (k, tuple(v) if isinstance(v, list) else v)
                    for k, v in roles.items()))
                if cur_helpful and key not in cur_helpful:
                    continue
                add_mask, del_mask = _action_delta_mask(
                    action, roles, rule_effects, lex, table, cur_mask)
                # Event-fire goal injection: the synthetic
                # event_fired fact lives in the grounded effs; the
                # native delta of perception/speech verbs may be
                # empty, so inject the goal bit when this action's
                # effs would have produced it.
                if (goal[0] == "event_fire"
                        and (grounded_effs_masks[i] & goal_bit)):
                    add_mask |= goal_bit
                if add_mask == 0 and del_mask == 0:
                    continue
                new_mask = _apply_delta_mask(
                    cur_mask, add_mask, del_mask,
                    grounded_derivs, slot_vocab, table)
                if new_mask == cur_mask or new_mask in ehc_visited:
                    continue
                ehc_visited.add(new_mask)
                new_plan = cur_plan + [(action.lemma, roles)]
                if new_mask & goal_bit:
                    return new_plan
                new_h, new_helpful = heuristic_and_helpful(new_mask)
                if new_h >= _HEURISTIC_INF:
                    continue
                if new_h < anchor_h:
                    found = (new_mask, new_h, new_helpful, new_plan)
                    break
                bfs.append((new_mask, new_plan, new_helpful))
            if found:
                break
        if not found:
            break  # EHC plateau — drop to A*.
        anchor_mask, anchor_h, anchor_helpful, anchor_plan = found
    if anchor_h == 0:
        return anchor_plan

    # Phase 2: weighted A* from the deepest EHC anchor reached. The
    # anchor's mask/h/helpful/plan replace the initial seed so we
    # don't redo the EHC progress.
    open_list: list = []
    # (f, helpful_priority, tiebreak, g, h, plan, state_mask, helpful)
    heapq.heappush(open_list, (
        len(anchor_plan) + H_WEIGHT * anchor_h, 0, 0,
        len(anchor_plan), anchor_h,
        anchor_plan, anchor_mask, anchor_helpful))
    visited: dict = {anchor_mask: len(anchor_plan)}
    tiebreak = 1

    while open_list and expansions < max_states:
        (f, _hp, _tie, g, h_cur, plan, cur_mask, helpful) = (
            heapq.heappop(open_list))
        if h_cur >= _HEURISTIC_INF:
            continue
        # Lazy h: helpful entries are pushed with real h; non-helpful
        # are pushed with the parent's h as a placeholder. When a
        # non-helpful entry is popped, recompute h and re-push if it
        # turned out worse than the placeholder. `helpful` is reused
        # as the "is this entry fully evaluated" flag: None means
        # stale, a set means fresh.
        if helpful is None:
            real_h, real_helpful = heuristic_and_helpful(cur_mask)
            if real_h >= _HEURISTIC_INF:
                continue
            if real_h > h_cur:
                heapq.heappush(open_list, (
                    g + H_WEIGHT * real_h,
                    1, tiebreak, g, real_h, plan, cur_mask, real_helpful))
                tiebreak += 1
                continue
            helpful = real_helpful
            h_cur = real_h
        if cur_mask & goal_bit:
            return plan
        expansions += 1
        if g >= max_plan_length:
            continue

        # Iterate grounded actions: pres-check is a single AND+eq
        # on int (was an O(|pres|) frozenset membership loop).
        # Helpful actions get real h computed immediately; non-
        # helpful are pushed with parent's h as a placeholder (lazy
        # h) and re-evaluated only when popped — saves ~90% of
        # heuristic calls.
        for i, (action, roles, _pres, _effs) in enumerate(grounded):
            pres_mask = grounded_pres_masks[i]
            if (cur_mask & pres_mask) != pres_mask:
                continue
            add_mask, del_mask = _action_delta_mask(
                action, roles, rule_effects, lex, table, cur_mask)
            if (goal[0] == "event_fire"
                    and (grounded_effs_masks[i] & goal_bit)):
                add_mask |= goal_bit
            if add_mask == 0 and del_mask == 0:
                continue
            new_mask = _apply_delta_mask(
                cur_mask, add_mask, del_mask,
                grounded_derivs, slot_vocab, table)
            if new_mask == cur_mask:
                continue  # no-op event
            new_g = g + 1
            prev_g = visited.get(new_mask)
            if prev_g is not None and prev_g <= new_g:
                continue
            visited[new_mask] = new_g
            if new_mask & goal_bit:
                return plan + [(action.lemma, roles)]
            key = (action.lemma, frozenset(
                (k, tuple(v) if isinstance(v, list) else v)
                for k, v in roles.items()))
            is_helpful = key in helpful
            # Diversity noise: small uniform perturbation on the open-list
            # priority shuffles equal-and-near-equal paths so different
            # rng seeds explore different sub-optimal plans. new_g stays
            # integer for visited bookkeeping; only the heap sees noise.
            noise = (rng.random() * 0.5 if rng is not None else 0.0)
            if is_helpful:
                new_h, new_helpful = heuristic_and_helpful(new_mask)
                if new_h >= _HEURISTIC_INF:
                    continue
                heapq.heappush(open_list, (
                    new_g + H_WEIGHT * new_h + noise,
                    0, tiebreak,
                    new_g, new_h,
                    plan + [(action.lemma, roles)],
                    new_mask, new_helpful))
            else:
                heapq.heappush(open_list, (
                    new_g + H_WEIGHT * h_cur + noise,
                    1, tiebreak,
                    new_g, h_cur,
                    plan + [(action.lemma, roles)],
                    new_mask, None))
            tiebreak += 1
    return None


_RELEVANT_VERBS_CACHE: dict = {}


def _goal_reachable_verbs(goal, lex, rule_effects, derivations) -> set | None:
    """Thin wrapper around `_goal_reachable_walk` that returns just
    the verb set — kept for callers that don't need the derivation
    set."""
    result = _goal_reachable_walk(goal, lex, rule_effects, derivations)
    if result is None:
        return None
    verbs, _ = result
    return verbs


def _goal_reachable_walk(
    goal, lex, rule_effects, derivations,
) -> tuple[set, set] | None:
    """Backward-walk from goal facts through action effects, rule-
    effect adds, action preconditions, AND derivation implications,
    returning `(relevant_verbs, relevant_derivation_names)` — or
    None for goal shapes we don't analyze.

    Two backward graphs are walked together:
      - Action graph: an action's effects/role-properties/preconditions
        determine which goal facts it can produce and which sub-goals
        it imposes.
      - Derivation graph: a derivation's `implies` clauses identify
        which facts it can synthesize; its `when` + `given` patterns
        identify the input facts it consumes. Walking back through a
        derivation surfaces verbs that produce the *inputs* — so
        `samloke` (purely derived) leads to `en`, which leads to
        iri/veni/eniri/fari (rule_effects producers of en).

    No hardcoded transit list. Movement/perception/state-prep verbs
    fall out naturally from precondition + derivation chasing on the
    goal-producing verb's needs. The derivation set is sound for
    h_FF pruning: a derivation outside it produces facts no goal-
    relevant action consumes (directly or via chained derivations),
    so it cannot lower the relaxed plan's cost to the goal.

    Cached per `(id(lex), goal)`."""
    cache_key = (id(lex), goal)
    cached = _RELEVANT_VERBS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    from collections import deque
    from ..dsl.implications import PropertyImplication, RelationImplication
    from ..dsl.patterns import (
        AndPattern, BindPattern, EntityPattern, NotPattern, RelPattern, Var,
    )

    # Index derivations by what they produce, both for relations
    # (synthesized via RelationImplication) and properties.
    rel_to_derivs: dict = {}
    prop_to_derivs: dict = {}
    for d in derivations:
        for imp in d.implies:
            if isinstance(imp, RelationImplication):
                rel_to_derivs.setdefault(imp.name, []).append(d)
            elif isinstance(imp, PropertyImplication):
                prop_to_derivs.setdefault(
                    (imp.slot, imp.value), []).append(d)

    def _walk_pattern_for_inputs(pattern, rel_inputs, prop_inputs):
        """Walk a Pattern tree, collecting (relation_name) from
        RelPatterns and (slot, value) constraints from EntityPatterns
        into the given output sets. Skips NotPatterns — relaxation
        drops negation, so the absence of a fact isn't a producer-
        requirement we need to chase."""
        if pattern is None:
            return
        if isinstance(pattern, NotPattern):
            return
        if isinstance(pattern, RelPattern):
            rel_inputs.add(pattern.relation)
            for arg in pattern.arg_patterns.values():
                _walk_pattern_for_inputs(arg, rel_inputs, prop_inputs)
            return
        if isinstance(pattern, EntityPattern):
            for slot, vals in (pattern.constraints or {}).items():
                # Property constraints with a fixed value are
                # subgoals; range/list constraints we conservatively
                # ignore.
                if isinstance(vals, (str, int, float)):
                    prop_inputs.add((slot, vals))
                elif isinstance(vals, (list, tuple)) and len(vals) == 1:
                    prop_inputs.add((slot, vals[0]))
            return
        if isinstance(pattern, AndPattern):
            _walk_pattern_for_inputs(pattern.left, rel_inputs, prop_inputs)
            _walk_pattern_for_inputs(pattern.right, rel_inputs, prop_inputs)
            return
        if isinstance(pattern, BindPattern):
            return  # Bind alone has no constraint.
        # Other patterns (Var, literal) carry no input fact.

    rel_visited: set = set()
    prop_visited: set = set()
    relevant: set = set()
    relevant_derivs: set = set()
    queue: deque = deque()

    def _add_verb(verb):
        if verb and verb in lex.actions and verb not in relevant:
            relevant.add(verb)
            queue.append(verb)

    def _seed_relation(rel_name):
        if rel_name in rel_visited:
            return
        rel_visited.add(rel_name)
        # Direct action producers.
        for v2, entry in rule_effects.items():
            for rel, _ in entry.get("adds", []):
                if rel == rel_name:
                    _add_verb(v2)
                    break
        # Derivation producers — walk back to their inputs.
        for d in rel_to_derivs.get(rel_name, ()):
            relevant_derivs.add(d.name)
            rel_inputs: set = set()
            prop_inputs: set = set()
            _walk_pattern_for_inputs(
                getattr(d, "when", None), rel_inputs, prop_inputs)
            for g in getattr(d, "given", ()) or ():
                _walk_pattern_for_inputs(g, rel_inputs, prop_inputs)
            for r2 in rel_inputs:
                _seed_relation(r2)
            for slot2, val2 in prop_inputs:
                _seed_property(slot2, val2)

    def _seed_property(slot, value):
        if (slot, value) in prop_visited:
            return
        prop_visited.add((slot, value))
        # Direct action producers.
        for v2, a2 in lex.actions.items():
            for eff in a2.effects:
                if eff.property == slot and eff.value == value:
                    _add_verb(v2)
                    break
        # Derivation producers — walk back.
        for d in prop_to_derivs.get((slot, value), ()):
            relevant_derivs.add(d.name)
            rel_inputs: set = set()
            prop_inputs: set = set()
            _walk_pattern_for_inputs(
                getattr(d, "when", None), rel_inputs, prop_inputs)
            for g in getattr(d, "given", ()) or ():
                _walk_pattern_for_inputs(g, rel_inputs, prop_inputs)
            for r2 in rel_inputs:
                _seed_relation(r2)
            for slot2, val2 in prop_inputs:
                _seed_property(slot2, val2)

    # Seed from the goal.
    kind = goal[0]
    if kind == "property":
        _, _eid, slot, value = goal
        _seed_property(slot, value)
    elif kind == "relation":
        _, rel_name, _args = goal
        _seed_relation(rel_name)
    elif kind == "event_fire":
        _, verb_lemma, _bindings = goal
        _add_verb(verb_lemma)
    else:
        return None

    # BFS over verbs' preconditions and role requirements.
    while queue:
        verb = queue.popleft()
        action = lex.actions.get(verb)
        if action is None:
            continue
        for role in action.roles:
            for slot, values in (role.properties or {}).items():
                if not values:
                    continue
                _seed_property(slot, values[0])
            # Parts-binding roles (fari.parts, kind="list",
            # from_field="parts") draw their bindings from a
            # constructable concept's `parts` list, where each part
            # spec may carry its own `requires` map (akvo's
            # temperature must be bolanta when it's used as part of
            # teo/kafo). Treat those requires as implicit role
            # requirements: each constructable contributes property
            # subgoals that the planner must satisfy before fari
            # fires.
            if (getattr(role, "kind", "single") == "list"
                    and getattr(role, "from_field", None) == "parts"):
                for c_lemma in lex.concept_index.concepts_matching(
                        properties={"constructable": ["yes"]}):
                    c_def = lex.concepts[c_lemma]
                    for part_spec in getattr(c_def, "parts", ()) or ():
                        for r_slot, r_vals in (
                                getattr(part_spec, "requires", None)
                                or {}).items():
                            if not r_vals:
                                continue
                            _seed_property(r_slot, r_vals[0])
        for pc in action.preconditions:
            kind2 = getattr(pc, "kind", None)
            if kind2 == "relation":
                _seed_relation(pc.rel)
            elif kind2 == "if_property":
                _seed_property(pc.then_property, pc.then_value)
    result = (relevant, relevant_derivs)
    _RELEVANT_VERBS_CACHE[cache_key] = result
    return result


def _drive_to_goal(drive, trace, lex):
    """Translate the dispatcher's drive shape into a forward-planner
    goal. Currently supports entity_slot, self_slot, location,
    possession, wearing, event_fire. Returns the goal tuple or None
    for unsupported shapes.

    event_fire: ("event_fire", actor_eid, verb_lemma, role_bindings)
    where role_bindings is an iterable of (role_name, eid) pairs.
    Goal becomes "the plan must fire `verb_lemma` with these roles
    bound" — used for verbs whose postcondition is CreateEntity +
    AddRelation with a `<created>` sentinel (legi, skribi, vidi,
    flari, aŭdi), which the property-shape goal index can't express.
    """
    kind = drive[0]
    if kind == "entity_slot":
        _, _actor, target, slot, value = drive
        return ("property", target, slot, value)
    if kind == "self_slot":
        _, actor, slot, value = drive
        return ("property", actor, slot, value)
    if kind == "location":
        _, actor, loc = drive
        return ("relation", "en", (actor, loc))
    if kind == "proximity":
        _, actor, loc = drive
        return ("relation", "apud", (actor, loc))
    if kind == "possession":
        _, actor, item = drive
        return ("relation", "havi", (actor, item))
    if kind == "wearing":
        _, actor, garment = drive
        return ("relation", "vestita", (actor, garment))
    # Generic relation-drive shapes: the seeder carries the
    # relation name in the drive itself instead of going through
    # a per-relation kind label. Three forms, one per role-position
    # the seeder dispatches on. Replaces the "possession"/"location"/
    # "wearing"/"proximity" + altruistic-* fixed labels — the schema's
    # relations.jsonl is now the single source of truth for what's
    # drive-able, with no per-relation code paths. Checked BEFORE the
    # legacy `altruistic_*` prefix match so the new
    # `altruistic_relation_drive` shape (5-tuple) doesn't fall into
    # the 4-tuple unpack.
    if kind == "relation_drive":
        # ("relation_drive", rel_name, actor, target) — actor IS the
        # beneficiary (role_args[0]=="agent"), so goal targets actor.
        _, rel, actor, target = drive
        return ("relation", rel, (actor, target))
    if kind == "altruistic_relation_drive":
        # (kind, rel_name, actor, beneficiary, target) — actor performs
        # the producer verb, beneficiary ends up in the relation.
        _, rel, _actor, beneficiary, target = drive
        return ("relation", rel, (beneficiary, target))
    if kind == "place_drive":
        # (kind, rel_name, actor, obj, location) — actor manipulates
        # obj to be related to location (meti/verŝi/planti style).
        # Neither actor nor a beneficiary appears in the goal — obj
        # and location do.
        _, rel, _actor, obj, location = drive
        return ("relation", rel, (obj, location))
    if kind.startswith("altruistic_"):
        # Legacy altruistic_* labels (altruistic_possession etc.) from
        # the pre-generic-shape era. Kept for backward compat with any
        # remaining caller that uses them.
        _, _actor, beneficiary, target = drive
        rel = {
            "altruistic_possession": "havi",
            "altruistic_location":   "en",
            "altruistic_wearing":    "vestita",
            "altruistic_proximity":  "apud",
        }.get(kind)
        if rel is None:
            return None
        return ("relation", rel, (beneficiary, target))
    if kind == "event_fire":
        _, _actor, verb_lemma, role_bindings = drive
        return ("event_fire", verb_lemma, frozenset(role_bindings))
    return None
