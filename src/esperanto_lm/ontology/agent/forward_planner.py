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
from .planner import (
    _cached_compute_derived_state, _entity_property_values,
    _find_role_filler, _has_relation, _simulate_from_scratch,
    _step_to_event,
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

    `scias_lokon`/`konas` ARE enforced; the relaxed encoding
    synthesizes them as direct effects of perception verbs (vidi/
    aŭdi/flari/montri) via `_build_rule_effects_index`, so the
    fact-set search can plan through preni/kapti/veki's perception
    preconditions without modeling fakto-entity creation."""
    from ..schemas import (
        IfPropertyPrecondition, MatchPrecondition, RelationPrecondition,
    )
    for pc in action.preconditions:
        if isinstance(pc, RelationPrecondition):
            eids = tuple(roles.get(r) for r in pc.roles)
            if any(e is None for e in eids):
                return False
            if not _has_relation(pc.rel, eids, trace, derived, lex):
                return False
        elif isinstance(pc, IfPropertyPrecondition):
            eid = roles.get(pc.role)
            if eid is None:
                continue
            ent = trace.entities.get(eid)
            if ent is None:
                return False
            gate = _entity_property_values(
                ent, pc.if_property, trace, derived)
            if pc.if_value not in gate:
                continue  # Gate not active — pc vacuously holds.
            then_vals = _entity_property_values(
                ent, pc.then_property, trace, derived)
            if pc.then_value not in then_vals:
                return False
        elif isinstance(pc, MatchPrecondition):
            ea = trace.entities.get(roles.get(pc.role_a))
            eb = trace.entities.get(roles.get(pc.role_b))
            if ea is None or eb is None:
                return False
            va = set(ea.properties.get(pc.slot_a, []))
            vb = set(eb.properties.get(pc.slot_b, []))
            if not (va & vb):
                return False
    return True


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
        return value in _entity_property_values(ent, slot, trace, derived)
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

# Per-grounded_derivations cache for the pres-fact inverted index
# used by _apply_delta's worklist re-derivation. Keyed by
# id(derivation_pseudos) — caller is plan_for_goal which builds
# grounded_derivs once per plan and reuses for ~hundreds of
# _apply_delta calls during search.
_DERIV_PRES_IDX_CACHE: dict = {}


def _ground_action_facts(action, roles, lex, rule_effects):
    """Return (precondition_facts, effect_facts) for a grounded
    action — both as sets of fact tuples. `rule_effects` is the
    `{verb: [(relation, role_arg_names)]}` index of rule-added
    relations (preni adds havi, vidi adds konas, etc.) — without
    these the relaxed graph misses key transitions and the
    heuristic returns INF for any chain that depends on them."""
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
        if getattr(role_spec, "kind", "single") == "list":
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
            for eids in _expand_list_role_args(role_arg_names, roles):
                if any(e is None for e in eids):
                    continue
                effs.add(("rel", relation, _canon_rel(relation, eids, sym)))
    return pres, effs


def _expand_list_role_args(role_arg_names, roles):
    """Yield one tuple of eids per element of a list-marked arg
    position. If no arg is a `("<list>", role_name)` marker, yield
    a single tuple (the trivial expansion). Used by both the
    relaxed-graph grounding and the fact-set delta computation so
    fari's variadic havas_parton / havi-detach effects emit one
    fact per gathered part."""
    # Find list-marker positions and resolve to their list role values.
    list_positions: list = []
    for i, arg in enumerate(role_arg_names):
        if (isinstance(arg, tuple) and len(arg) == 2
                and arg[0] == "<list>"):
            list_role = arg[1]
            list_val = roles.get(list_role)
            if list_val is None or not isinstance(list_val, (list, tuple)):
                # List role unbound (e.g. optional instrument with no
                # crafted_with) — drop this effect by yielding nothing.
                return
            list_positions.append((i, list_val))
    if not list_positions:
        yield tuple(roles.get(r) for r in role_arg_names)
        return
    # For our actual fari usage there's only one list arg per effect;
    # support multiple by element-wise zipping the list values.
    n = max(len(vals) for _, vals in list_positions)
    for k in range(n):
        out: list = []
        for i, arg in enumerate(role_arg_names):
            list_match = next((vals for pos, vals in list_positions
                               if pos == i), None)
            if list_match is not None:
                out.append(list_match[k] if k < len(list_match) else None)
            else:
                out.append(roles.get(arg))
        yield tuple(out)


def _ground_derivations(
    derivations, trace, lex,
    relevant_entities: set | None = None,
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
        actually narrow the bridge var."""
        from ..dsl.patterns import NotPattern, RelPattern
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
                        if (isinstance(c, NotPattern)
                                and isinstance(c.inner, EntityPattern)):
                            out.append(dict(c.inner.constraints))
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
                arg_vids: list = []
                ok = True
                for arg_name in rel_def.arg_names:
                    arg_pat = rp.arg_patterns.get(arg_name)
                    if arg_pat is None:
                        ok = False
                        break
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

        for combo in itertools.product(*domains):
            if len(set(combo)) != len(combo):
                continue
            binding = {vid: combo[i] for i, vid in enumerate(var_ids)}
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
                    continue

            # pres: each rel pattern grounded with binding, plus
            # property constraints on each Var.
            pres: set = set()
            ok = True
            for rp, arg_vids in zip(rel_patterns, rel_arg_vids):
                args = []
                for av in arg_vids:
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
                continue
            # Property constraints from each Var → pres.
            for vid in var_ids:
                _, _, props = var_constraints[vid]
                eid = binding[vid]
                for slot, val in props:
                    pres.add(("prop", eid, slot, val))

            # effs: from impl_specs.
            effs: set = set()
            for spec in impl_specs:
                if spec[0] == "prop":
                    _, e_var, slot, val = spec
                    eid = binding.get(id(e_var))
                    if eid is None:
                        continue
                    effs.add(("prop", eid, slot, val))
                else:
                    _, name, arg_vars = spec
                    args = []
                    ok = True
                    for a in arg_vars:
                        if isinstance(a, Var):
                            eid = binding.get(id(a))
                            if eid is None:
                                ok = False
                                break
                            args.append(eid)
                        else:
                            args.append(a)
                    if not ok:
                        continue
                    effs.add(("rel", name,
                              _canon_rel(name, tuple(args), sym_local)))

            if effs:
                out.append((None, binding, pres, effs))
    return out


def _action_delta(action, roles, rule_effects, lex, facts=None):
    """Compute (adds, dels) fact deltas for firing this grounded
    action. Used by the fact-set incremental simulator. Far cheaper
    than `_simulate_from_scratch` (no Trace fork, no engine rerun)
    at the cost of skipping cascades and entity creation. For our
    common chains (preni, iri, eniri, sekigi, malŝalti) the action+
    rule effects are sufficient.

    adds:
      - action.effects (property writes on the target role)
      - rule effects: AddRelation, TransferN-as-havi
    dels:
      - rule effects: RemoveRelation
      - For scalar slot writes, the previous value on that slot is
        implicitly displaced (caller treats slot=value as scalar).

    When `facts` is supplied, dels with wildcard positions (None in
    the role-arg tuple, indicating a given-bound var like iri's "from"
    container) are expanded by querying the fact set for matching
    relations. Without this, iri/veni/flugi would leave their old
    en(agent, origin) fact in place and downstream movement actions
    could fire spuriously from the stale origin.
    """
    adds: set = set()
    dels: set = set()
    sym = _symmetric_relations(lex)
    # Schema-level effects (property writes).
    for eff in action.effects:
        eid = roles.get(eff.target_role)
        if eid is None:
            continue
        adds.add(("prop", eid, eff.property, eff.value))
    # Rule-level effects: apply per-rule so cascade rules with
    # wildcard pres (porti_drop_when_carrier_falls) don't gate the
    # main rule's effects. A rule "fires" iff its wildcard dels
    # match — same semantics as the engine's `given` patterns.
    entry = rule_effects.get(action.lemma)
    if entry is not None:
        for rule_entry in entry.get("rules", ()):
            this_dels: list = []
            fires = True
            for relation, role_arg_names in rule_entry["dels"]:
                # `<list>` markers (variadic dels like fari's
                # per-part havi-detach) expand to one del per element.
                has_list = any(
                    isinstance(a, tuple) and a and a[0] == "<list>"
                    for a in role_arg_names)
                if has_list:
                    for eids in _expand_list_role_args(
                            role_arg_names, roles):
                        if any(e is None for e in eids):
                            continue
                        this_dels.append(
                            ("rel", relation,
                             _canon_rel(relation, eids, sym)))
                    continue
                if None in role_arg_names:
                    if facts is None:
                        # Caller doesn't need accurate dels (relaxed
                        # graph). Skip — adds still flow through.
                        continue
                    pattern = tuple(roles.get(r) if r else None
                                    for r in role_arg_names)
                    matched = 0
                    for f in facts:
                        if (f[0] == "rel" and f[1] == relation
                                and len(f[2]) == len(pattern)
                                and all(p is None or p == a
                                        for p, a in zip(pattern, f[2]))):
                            this_dels.append(f)
                            matched += 1
                    if matched == 0:
                        # Rule's `given` doesn't hold — engine wouldn't
                        # fire this rule, so neither its adds nor dels
                        # apply. The action itself may still progress
                        # (event_fire injection, schema effects); the
                        # search loop's `not adds and not dels` check
                        # skips pure no-ops.
                        fires = False
                        break
                else:
                    eids = tuple(roles.get(r) for r in role_arg_names)
                    if any(e is None for e in eids):
                        continue
                    this_dels.append(
                        ("rel", relation,
                         _canon_rel(relation, eids, sym)))
            if not fires:
                continue
            for relation, role_arg_names in rule_entry["adds"]:
                for eids in _expand_list_role_args(role_arg_names, roles):
                    if any(e is None for e in eids):
                        continue
                    adds.add(
                        ("rel", relation,
                         _canon_rel(relation, eids, sym)))
            for d in this_dels:
                dels.add(d)
    return adds, dels


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
    key = id(derivation_pseudos)
    closure = _DERIV_RELDEP_CACHE.get(key)
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
        _DERIV_RELDEP_CACHE[key] = closure
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
    pres_idx = _DERIV_PRES_IDX_CACHE.get(id(derivation_pseudos))
    if pres_idx is None:
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
        _DERIV_PRES_IDX_CACHE[id(derivation_pseudos)] = pres_idx
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


def _build_rule_effects_index(rules) -> dict:
    """Index rules by their trigger verb, collecting AddRelation and
    RemoveRelation effects with role-arg-name mappings. Maps
    verb_lemma → {'adds': [(relation, (role_name, ...))], 'dels':
    [...]}. Both lists are used: 'adds' for the relaxed graph
    (delete-relaxation drops dels there), 'dels' for the
    fact-set incremental simulator (which DOES apply dels).

    Synthesizes `scias_lokon` adds for perception verbs (vidi, aŭdi,
    flari, montri) by detecting the canonical pattern:
       create_entity(fakto, as_var=F)
       add_relation(subjekto, F, X)
       add_relation(konas, K, F)
    The runtime engine derives `scias_lokon(K, X)` from these via
    the konas+subjekto rule chain; we collapse that chain into a
    direct relation effect so the fact-set search can plan through
    perception preconditions on preni/kapti/veki/etc. without
    modeling fakto-entity creation."""
    from ..dsl.effects import (
        AddRelation, CreateEntity, ForEach, RemoveRelation, TransferN,
    )
    from ..dsl.patterns import (
        AndPattern, BindPattern, EntityPattern, EventPattern, Var,
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
        # downstream grounder / _action_delta can expand the variadic
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
        for eff, eff_var_to_role in flat_effects:
            if isinstance(eff, AddRelation):
                role_args = []
                ok = True
                for arg in eff.args:
                    role_name = eff_var_to_role.get(id(arg))
                    if role_name is None:
                        ok = False
                        break
                    role_args.append(role_name)
                if ok:
                    rule_adds.append((eff.relation, tuple(role_args)))
                    entry["adds"].append(
                        (eff.relation, tuple(role_args)))
            elif isinstance(eff, RemoveRelation):
                # Args may include given-bound vars (e.g. iri's "from"
                # container O, matched against rel("en", agent, var(O)) in
                # `given`). Those positions can't be resolved until
                # simulation time. Encode them as None; _action_delta
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
        # Synthesize scias_lokon from the create-fakto+konas+subjekto
        # pattern. We need: a CreateEntity(fakto, as_var=F), an
        # AddRelation(subjekto, F, X) where X maps to a role, and an
        # AddRelation(konas, K, F) where K maps to a role. Then add
        # ("scias_lokon", (K_role, X_role)).
        fakto_vars: set = set()
        for eff in effects:
            if isinstance(eff, CreateEntity):
                concept = getattr(eff, "concept", None)
                if concept != "fakto":
                    continue
                av = getattr(eff, "as_var", None)
                if isinstance(av, Var):
                    fakto_vars.add(id(av))
        if not fakto_vars:
            continue
        for fv in fakto_vars:
            subj_role = None
            knower_role = None
            for eff in effects:
                if not isinstance(eff, AddRelation):
                    continue
                if eff.relation == "subjekto" and len(eff.args) == 2:
                    if isinstance(eff.args[0], Var) and id(eff.args[0]) == fv:
                        target = eff.args[1]
                        if isinstance(target, Var):
                            subj_role = var_to_role.get(id(target))
                elif eff.relation == "konas" and len(eff.args) == 2:
                    if isinstance(eff.args[1], Var) and id(eff.args[1]) == fv:
                        knower = eff.args[0]
                        if isinstance(knower, Var):
                            knower_role = var_to_role.get(id(knower))
            if subj_role and knower_role:
                synth = ("scias_lokon", (knower_role, subj_role))
                if synth not in entry["adds"]:
                    entry["adds"].append(synth)
                    # Mirror in a synthetic rule so the per-rule
                    # _action_delta loop still emits it. Without this
                    # the fact-set simulator drops scias_lokon and
                    # downstream preni/kapti can't fire.
                    entry["rules"].append(
                        {"adds": [synth], "dels": []})
    return out


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
    everywhere."""
    if rel_name in sym and len(args) == 2 and args[0] > args[1]:
        return (args[1], args[0])
    return args


def _state_facts(trace, derived, lex=None) -> set:
    """Snapshot of state facts (property + relation) for the
    relaxed graph's layer 0. Symmetric arity-2 relation tuples are
    canonicalized to a single ordering — every emission point uses
    `_canon_rel`, so fact-tuple equality finds matches regardless
    of the order in which a derivation or pre wrote the relation."""
    facts: set = set()
    sym = _symmetric_relations(lex) if lex is not None else frozenset()
    for eid, ent in trace.entities.items():
        for slot, values in ent.properties.items():
            for v in values:
                facts.add(("prop", eid, slot, v))
    for r in trace.relations:
        facts.add(("rel", r.relation,
                   _canon_rel(r.relation, tuple(r.args), sym)))
    # Derived state — relations + properties.
    if derived is not None:
        for rel_tuple in derived.relations:
            args = tuple(rel_tuple[1])
            facts.add(("rel", rel_tuple[0],
                       _canon_rel(rel_tuple[0], args, sym)))
        for (eid, slot), val in (
                getattr(derived, "properties", {}) or {}).items():
            if isinstance(val, list):
                for v in val:
                    facts.add(("prop", eid, slot, v))
            else:
                facts.add(("prop", eid, slot, val))
    return facts


def _ground_all_actions(trace, lex, derived, rule_effects) -> list:
    """All (action, roles, pres, effs) for actions whose roles can
    be bound to current entities. Pure enumeration — preconditions
    not checked here (the relaxed graph layer decides applicability).
    Skips bindings where the same entity fills two roles, matches the
    behavior of `_applicable_actions`. `rule_effects` is the
    rule-effect index built by `_build_rule_effects_index`."""
    out = []
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
        per_role: list[list[str]] = []
        ok = True
        for role_spec in action.roles:
            cand = []
            for eid, ent in trace.entities.items():
                if not lex.types.is_subtype(
                        ent.entity_type, role_spec.type):
                    continue
                cand.append(eid)
            if not cand:
                ok = False
                break
            per_role.append(cand)
        if not ok:
            continue
        for combo in itertools.product(*per_role):
            if len(set(combo)) != len(combo):
                continue
            roles = {r.name: e for r, e in zip(action.roles, combo)}
            pres, effs = _ground_action_facts(
                action, roles, lex, rule_effects)
            if not effs:
                continue  # No-effect actions don't add facts.
            out.append((action, roles, pres, effs))
    return out


def _ground_constructable_actions(
    trace, lex, rule_effects,
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
                fari, roles, lex, rule_effects)
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


def _build_consumer_index(actions, derivations):
    """Inverted index: fact → list of (base_cost, pres_set, effs_set,
    pres_tuple) consumers that need this fact in their pres. Actions
    have base_cost=1, derivations have base_cost=0. Used by the
    worklist h_add relaxation so each fact addition only re-checks
    the consumers that actually care about it, instead of iterating
    all ~5000 grounded actions+derivations per layer.

    Returns (fact_to_consumers, all_consumers, cid_to_info,
    pres_len) — the latter two are per-plan invariants precomputed
    once and reused across the ~100 h_add calls in one search:
      - cid_to_info: list indexed by cid → (verb_lemma_or_None,
        roles, pres, effs). Used by h_add's helpful-action backtrack.
      - pres_len: list indexed by cid → len(pres). Used to seed
        unsat[cid] each h_add call without re-iterating all_consumers.
    """
    fact_to: dict[tuple, list] = {}
    all_consumers: list = []
    cid_to_info: list = []
    pres_len: list = []
    cid = 0
    for action, roles, pres, effs in actions:
        pres_set = frozenset(pres)
        consumer = (1, pres_set, frozenset(effs), cid)
        all_consumers.append(consumer)
        cid_to_info.append((action.lemma, roles, pres, effs))
        pres_len.append(len(pres_set))
        for p in pres_set:
            fact_to.setdefault(p, []).append(consumer)
        cid += 1
    for _, _, pres, effs in derivations:
        pres_set = frozenset(pres)
        consumer = (0, pres_set, frozenset(effs), cid)
        all_consumers.append(consumer)
        cid_to_info.append((None, None, pres, effs))
        pres_len.append(len(pres_set))
        for p in pres_set:
            fact_to.setdefault(p, []).append(consumer)
        cid += 1
    return fact_to, all_consumers, cid_to_info, pres_len


def _heuristic_and_helpful(
    goal, trace, derived, lex, grounded_actions=None,
    grounded_derivations=None, consumer_index=None,
    facts: set | None = None,
) -> tuple[int, set]:
    """h_add over the relaxed planning graph using a worklist with
    an inverted fact→consumers index. Also returns the "helpful
    actions" set — verbs whose grounded effects appear on the
    cheapest path back from the goal in the relaxed plan.

    Initial: cost(initial_facts) = 0. For each consumer c (action or
    derivation) whose pres are all known: tentative_cost = c.base +
    sum(cost[p] for p in c.pres). For each effect e: if
    tentative_cost < cost[e], update cost[e] and remember the
    producer. Terminates when the worklist is empty.

    Helpful action extraction: walk back from goal facts through
    cheapest producers; collect the (verb, roles) pairs of action
    producers (skipping derivation pseudo-actions). Forward search
    only expands these — typically 3-8 out of ~80 applicable, so
    the branching factor collapses to something tractable.

    Returns (h, helpful_actions) — h is 0 if goal holds, INF if
    unreached, else sum of costs over goal literals.
    """
    # If facts provided, skip trace-based goal check (used in
    # fact-set search where there's no real Trace per state).
    if facts is None:
        if _goal_satisfied(goal, trace, derived, lex):
            return 0, set()
    else:
        # Goal as fact-set check.
        if goal[0] == "property":
            _, eid, slot, value = goal
            if ("prop", eid, slot, value) in facts:
                return 0, set()
        elif goal[0] == "relation":
            _, relation, args = goal
            _sym_h = _symmetric_relations(lex)
            if ("rel", relation,
                    _canon_rel(relation, tuple(args), _sym_h)) in facts:
                return 0, set()
        elif goal[0] == "event_fire":
            _, verb_lemma, bindings_frozen = goal
            if ("event_fired", verb_lemma, bindings_frozen) in facts:
                return 0, set()

    rule_effects = _RULE_EFFECTS_CACHE.get(id(lex))
    if rule_effects is None:
        from ..dsl.rules import DEFAULT_DSL_RULES
        rule_effects = _build_rule_effects_index(DEFAULT_DSL_RULES)
        _RULE_EFFECTS_CACHE[id(lex)] = rule_effects

    actions = (grounded_actions if grounded_actions is not None
                else _ground_all_actions(trace, lex, derived, rule_effects))
    if grounded_derivations is not None:
        derivation_pseudos = grounded_derivations
    else:
        from ..dsl.rules import RUNTIME_DERIVATIONS
        derivation_pseudos = _ground_derivations(
            RUNTIME_DERIVATIONS, trace, lex)
    if consumer_index is None:
        fact_to, all_consumers, cid_to_info, pres_len = (
            _build_consumer_index(actions, derivation_pseudos))
    else:
        fact_to, all_consumers, cid_to_info, pres_len = consumer_index

    # Layer 0: state facts at cost 0. Either from a fact set
    # (incremental sim) or extracted from the trace+derived.
    if facts is not None:
        cost: dict = {f: 0 for f in facts}
    else:
        cost: dict = {f: 0 for f in _state_facts(trace, derived, lex)}
    # producer[fact] = cid of the consumer that achieved cost[fact].
    producer: dict = {}

    from collections import deque
    # producers_cheapest[fact] = consumer cids that achieved the
    # current cheapest cost for that fact. On strict cost
    # improvement, reset to [new_cid]; on tie, append. Used for
    # both h_FF count and the helpful set.
    producers_cheapest: dict[tuple, list] = {}

    # Counter-based h_add propagation. Maintain per-consumer:
    #   unsat[cid] = number of pres still at INF (consumer fires
    #                when this hits 0)
    #   pre_sum[cid] = sum of cost[p] for known pres
    # List-indexed-by-cid (cids are sequential from _build_consumer_index)
    # — faster init and access than dict because the consumer set is
    # the same across all h_add calls in one plan, only values reset.
    n_consumers = len(all_consumers)
    pre_sum = [0] * n_consumers
    unsat = pres_len[:]  # copy of the precomputed pres lengths
    ready_consumers: deque = deque(
        cid_ for cid_, u in enumerate(unsat) if u == 0)

    # Seed: each cost-0 fact decrements unsat for its consumers.
    # pre_sum stays 0 since cost is 0.
    for f in cost:
        for _base, _pres, _effs, cid_ in fact_to.get(f, ()):
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
                # Emit fact-cost event for consumers of e.
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

    if goal[0] == "property":
        _, eid, slot, value = goal
        goal_facts = [("prop", eid, slot, value)]
    elif goal[0] == "relation":
        _, relation, args = goal
        goal_facts = [("rel", relation,
                       _canon_rel(relation, tuple(args),
                                   _symmetric_relations(lex)))]
    elif goal[0] == "event_fire":
        _, verb_lemma, bindings_frozen = goal
        goal_facts = [("event_fired", verb_lemma, bindings_frozen)]
    else:
        return _HEURISTIC_INF, set()

    # Check goal reachability.
    for g in goal_facts:
        if cost.get(g, _HEURISTIC_INF) >= _HEURISTIC_INF:
            return _HEURISTIC_INF, set()

    # producers_cheapest / producers_all built incrementally during
    # the worklist above — no second pass needed.
    #
    # h_FF: walk back from goal through CHEAPEST producers,
    # collecting unique action cids. h = |unique actions|.
    relaxed_action_cids: set = set()
    visited_cheapest: set = set()
    stack = list(goal_facts)
    while stack:
        f = stack.pop()
        if f in visited_cheapest:
            continue
        visited_cheapest.add(f)
        for cid_ in producers_cheapest.get(f, ()):
            if cid_ < 0 or cid_ >= len(cid_to_info):
                continue
            info = cid_to_info[cid_]
            verb, _roles, pres, _effs = info
            if verb is not None:
                relaxed_action_cids.add(cid_)
            for p in pres:
                if p not in visited_cheapest:
                    stack.append(p)
            break  # one producer per fact for h_FF
    h_ff = len(relaxed_action_cids)

    # Helpful set: walk back through cheapest producers, collect
    # verbs. (Earlier versions also walked sub-cheapest producers via
    # a separate producers_all map for diversity; profiling showed
    # that map's setdefault was a hot loop and the broader set
    # didn't materially change search outcomes.)
    helpful: set = set()
    visited_all: set = set()
    stack = list(goal_facts)
    while stack:
        f = stack.pop()
        if f in visited_all:
            continue
        visited_all.add(f)
        for cid_ in producers_cheapest.get(f, ()):
            if cid_ < 0 or cid_ >= len(cid_to_info):
                continue
            info = cid_to_info[cid_]
            verb, roles, pres, _effs = info
            if verb is not None:
                # List-valued role bindings (fari.parts) are unhashable
                # in the frozenset key; tuple-coerce them.
                helpful.add((verb, frozenset(
                    (k, tuple(v) if isinstance(v, list) else v)
                    for k, v in roles.items())))
            for p in pres:
                if p not in visited_all:
                    stack.append(p)

    return h_ff, helpful


def _heuristic(goal, trace, derived, lex, grounded_actions=None,
                grounded_derivations=None, consumer_index=None) -> int:
    """Backward-compatible: just the h value, without helpful set."""
    h, _ = _heuristic_and_helpful(
        goal, trace, derived, lex, grounded_actions,
        grounded_derivations, consumer_index)
    return h


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
    facts = set(_state_facts(trace, derived, lex))

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
    if ("prop", eid, slot, value) in set(_state_facts(trace, derived, lex)):
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
        if ("prop", eid, slot, value) in set(_state_facts(trace, derived, lex)):
            return


def _prespawn_for_goal(goal, trace, lex, rules, derivations, resolver):
    """For each action whose effects could produce `goal`, ensure all
    role types have at least one matching entity. Missing roles are
    filled by calling `resolver`. Mirrors what the backward planner's
    `_find_role_filler` does lazily during search.

    Then runs `_infra_prespawn` recursively to handle
    derivation-chain infrastructure (e.g., spawn a lampo when the
    chain requires illuminated agents)."""
    from ..regression.seeders import _fully_derivable_slots
    rule_effects = _RULE_EFFECTS_CACHE.get(id(lex))
    if rule_effects is None:
        rule_effects = _build_rule_effects_index(rules)
        _RULE_EFFECTS_CACHE[id(lex)] = rule_effects
    derivable_slots = _fully_derivable_slots(lex, derivations)

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
            filler = _find_role_filler(
                role_spec, trace, lex, derived=derived_now,
                exclude=exclude, action=action,
                role_name=role_spec.name)
            if filler is None:
                filler = resolver(role_spec, trace, lex, exclude,
                                  action=action,
                                  role_name=role_spec.name)
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
    entity_resolver=None,
) -> Optional[list]:
    """Greedy best-first forward search. Returns a plan (list of
    (verb, roles) steps) or None.

    `max_states` bounds total state expansions (latency cap).
    `max_plan_length` bounds plan depth (prevents infinite chains
    when goal is unreachable but heuristic plateaus).
    `entity_resolver`: optional callable matching the backward
    planner's `_ENTITY_RESOLVER` signature. When provided, pre-spawns
    missing role-fillers for goal-producing actions so the
    forward search sees the same scene as backward search would."""
    goal = _drive_to_goal(drive, initial_trace, lex)
    if goal is None:
        return None

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

    # Ground actions once for the initial entity set; reuse across
    # all heuristic + applicability calls. Event firing rarely
    # creates new entities in our domain, so the grounding stays
    # valid for the search. Invalidation: not needed for the POC.
    rule_effects = _RULE_EFFECTS_CACHE.get(id(lex))
    if rule_effects is None:
        rule_effects = _build_rule_effects_index(rules)
        _RULE_EFFECTS_CACHE[id(lex)] = rule_effects
    grounded = _ground_all_actions(
        initial_trace, lex, initial_derived, rule_effects)
    grounded.extend(_ground_constructable_actions(
        initial_trace, lex, rule_effects))
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
    heuristic_derivations = [
        d for d in derivations
        if getattr(d, "name", None) not in _SKIP_IN_HEURISTIC]
    grounded_derivs = _ground_derivations(
        heuristic_derivations, initial_trace, lex,
        relevant_entities=relevant_entities)

    # Event-fire goals: the goal is "fire verb V with these role
    # bindings". Lacking direct property/relation effects (legi/
    # skribi/vidi/flari/aŭdi all create a fakto + konas relation
    # via the rule body, which the goal-index can't trace through
    # role variables), we synthesize an `event_fired` pseudo-fact
    # that the goal verb's grounding produces. The relaxed graph
    # then has something to aim for; the search satisfies when the
    # fact lands in the state.
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
                    action_obj, roles, lex, rule_effects)
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
    consumer_index = _build_consumer_index(grounded, grounded_derivs)

    # Slot scalar info for the fact-set incremental simulator —
    # needed to correctly displace prior values on scalar slots.
    slot_vocab: dict = {
        name: {"scalar": getattr(s, "scalar", True)}
        for name, s in lex.slots.items()
    }

    def heuristic_and_helpful(facts):
        return _heuristic_and_helpful(
            goal, None, None, lex,
            grounded_actions=grounded,
            grounded_derivations=grounded_derivs,
            consumer_index=consumer_index, facts=facts)

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

    # Fact-set incremental search. State is frozenset(facts). Each
    # successor: check pres ⊆ state, compute delta via
    # `_action_delta`, apply via `_apply_delta` (which re-derives).
    # Far faster than `_simulate_from_scratch` per successor (~1ms
    # vs ~9ms) at the cost of skipping cascades / entity creation.
    H_WEIGHT = 2

    initial_facts = frozenset(
        _state_facts(initial_trace, initial_derived, lex))
    # Apply derivations to layer-0 to get full derived layer.
    initial_facts = _apply_delta(
        initial_facts, set(), set(),
        grounded_derivs, slot_vocab)

    if goal_fact in initial_facts:
        return []

    initial_h, initial_helpful = heuristic_and_helpful(initial_facts)
    if initial_h >= _HEURISTIC_INF:
        return None

    import heapq
    open_list: list = []
    # (f, helpful_priority, tiebreak, g, h, plan, facts, helpful)
    heapq.heappush(open_list, (
        H_WEIGHT * initial_h, 0, 0, 0, initial_h,
        [], initial_facts, initial_helpful))
    visited: dict = {initial_facts: 0}
    expansions = 0
    tiebreak = 1

    while open_list and expansions < max_states:
        (f, _hp, _tie, g, h_cur, plan, facts, helpful) = (
            heapq.heappop(open_list))
        if h_cur >= _HEURISTIC_INF:
            continue
        if goal_fact in facts:
            return plan
        expansions += 1
        if g >= max_plan_length:
            continue

        # Iterate grounded actions: pres-check is just subset on
        # frozenset (O(|pres|), ~5 facts per action).
        for action, roles, pres, _effs in grounded:
            if not all(p in facts for p in pres):
                continue
            adds, dels = _action_delta(
                action, roles, rule_effects, lex, facts=facts)
            # For event-fire goal entries, the synthetic event_fired
            # fact lives in the grounded `effs` set; inject it so the
            # action can fire even when its native delta is empty
            # (legi/skribi have no property changes, only rule-mediated
            # CreateEntity that the fact-set search can't represent).
            if goal[0] == "event_fire" and goal_fact in _effs:
                adds = set(adds) | {goal_fact}
            if not adds and not dels:
                continue
            new_facts = _apply_delta(
                facts, adds, dels, grounded_derivs, slot_vocab)
            if new_facts == facts:
                continue  # no-op event
            new_g = g + 1
            prev_g = visited.get(new_facts)
            if prev_g is not None and prev_g <= new_g:
                continue
            visited[new_facts] = new_g
            if goal_fact in new_facts:
                return plan + [(action.lemma, roles)]
            new_h, new_helpful = heuristic_and_helpful(new_facts)
            if new_h >= _HEURISTIC_INF:
                continue
            key = (action.lemma, frozenset(roles.items()))
            is_helpful = key in helpful
            heapq.heappush(open_list, (
                new_g + H_WEIGHT * new_h,
                0 if is_helpful else 1, tiebreak,
                new_g, new_h, plan + [(action.lemma, roles)],
                new_facts, new_helpful))
            tiebreak += 1
    return None


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
    if kind == "possession":
        _, actor, item = drive
        return ("relation", "havi", (actor, item))
    if kind == "wearing":
        _, actor, garment = drive
        return ("relation", "vestita", (actor, garment))
    if kind == "event_fire":
        _, _actor, verb_lemma, role_bindings = drive
        return ("event_fire", verb_lemma, frozenset(role_bindings))
    return None
