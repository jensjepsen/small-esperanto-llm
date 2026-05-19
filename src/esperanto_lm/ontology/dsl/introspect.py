"""Introspect causal rules to recover verb-level classification used
by the planner and realizer. Avoids hand-curated action-name lists
that drift as new verbs (aĉeti, vendi, …) get added.

Each helper takes a list of `Rule` and returns a frozenset of action
lemmas with the property in question. Computed once per rule set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .effects import (
    AddRelation, Change, ConsumeOne, CreateEntity, DestroyEntity, Emit,
    RemoveRelation, TransferN,
)
from .engine import Derivation, Rule
from .implications import PropertyImplication
from .patterns import (
    AndPattern, BindPattern, EntityPattern, EventPattern, NotPattern,
    OrPattern, RelPattern, Var,
)


def _bind_var_in_pattern(patt) -> Var | None:
    """First Var bound by a BindPattern reachable from `patt`. Mirrors
    the planner's helper of the same name; pulled out to keep this
    module self-contained."""
    if isinstance(patt, BindPattern):
        return patt.target
    if isinstance(patt, AndPattern):
        return (_bind_var_in_pattern(patt.left)
                or _bind_var_in_pattern(patt.right))
    if isinstance(patt, (OrPattern, NotPattern)):
        return None
    return None


def _role_vars(event_pat: EventPattern) -> dict[str, Var]:
    """{role_name: Var} for each role pattern that binds a Var."""
    out: dict[str, Var] = {}
    for name, patt in event_pat.role_patterns.items():
        v = _bind_var_in_pattern(patt)
        if v is not None:
            out[name] = v
    return out


def _effects(rule: Rule):
    """Normalize rule.then to an iterable."""
    return (rule.then if isinstance(rule.then, (list, tuple))
            else [rule.then])


def consumption_verbs(rules: list[Rule]) -> frozenset[str]:
    """Verbs whose rule fires `consume_one` — the realizer always
    quantity-overrides the theme for these (manĝi, trinki et al.)."""
    out: set[str] = set()
    for rule in rules:
        if not isinstance(rule.when, EventPattern):
            continue
        for eff in _effects(rule):
            if isinstance(eff, ConsumeOne):
                out.add(rule.when.action)
                break
    return frozenset(out)


def transfer_verbs(rules: list[Rule]) -> frozenset[str]:
    """Verbs whose rule fires `transfer_n` — possession changes are
    inherent to the verb (suppress redundant ownership narration;
    quantity-override theme when ev.quantity > 1)."""
    out: set[str] = set()
    for rule in rules:
        if not isinstance(rule.when, EventPattern):
            continue
        for eff in _effects(rule):
            if isinstance(eff, TransferN):
                out.add(rule.when.action)
                break
    return frozenset(out)


def acquisition_verbs(rules: list[Rule]) -> frozenset[str]:
    """Verbs whose theme-transfer ends at the agent (preni, kapti,
    aĉeti). The realizer attributes "de PRIOR_OWNER" to these via
    relation-change source tracking."""
    out: set[str] = set()
    for rule in rules:
        if not isinstance(rule.when, EventPattern):
            continue
        rvars = _role_vars(rule.when)
        agent_v = rvars.get("agent")
        theme_v = rvars.get("theme")
        if agent_v is None or theme_v is None:
            continue
        for eff in _effects(rule):
            if not isinstance(eff, TransferN):
                continue
            src, tgt = eff.source, eff.target
            if (isinstance(src, Var) and id(src) == id(theme_v)
                    and isinstance(tgt, Var) and id(tgt) == id(agent_v)):
                out.add(rule.when.action)
                break
    return frozenset(out)


def instrument_quantified_verbs(rules: list[Rule]) -> frozenset[str]:
    """Verbs whose rule transfers the instrument as one of the moved
    stacks (aĉeti pays with money, vendi receives money). The realizer
    matches the instrument's count to ev.quantity for these."""
    out: set[str] = set()
    for rule in rules:
        if not isinstance(rule.when, EventPattern):
            continue
        rvars = _role_vars(rule.when)
        instr_v = rvars.get("instrument")
        if instr_v is None:
            continue
        for eff in _effects(rule):
            if not isinstance(eff, TransferN):
                continue
            if isinstance(eff.source, Var) and id(eff.source) == id(instr_v):
                out.add(rule.when.action)
                break
    return frozenset(out)


def state_modifying_verbs(rules: list[Rule], lex) -> frozenset[str]:
    """Verbs whose firing writes any state — either via the action's
    intrinsic `effects` (baked into the seed event by `effect_changes`)
    or via a rule keyed on the verb that asserts/changes/transfers/
    creates/destroys. The complement is the no-op set: verbs whose
    event has no downstream consequence (kanti, danci, plori, ami,
    timi, ridi, helpi, ludi, vivi, labori, ripozi …). Vocal verbs
    (bleki, boji, miaŭi, krii, flustri) are state-modifying — their
    announce rules add scias entries — so they stay in.
    Computed once per (rules, lex); cheap fixpoint over Emit edges."""
    out: set[str] = {a.lemma for a in lex.actions.values() if a.effects}

    direct = (AddRelation, Change, ConsumeOne, CreateEntity, DestroyEntity,
              RemoveRelation, TransferN)
    # verb -> set of action names it Emits (without baked changes), used
    # for the fixpoint: emitting a state-modifier transitively counts.
    emit_edges: dict[str, set[str]] = {}
    for rule in rules:
        if not isinstance(rule.when, EventPattern):
            continue
        verb = rule.when.action
        for eff in _effects(rule):
            if isinstance(eff, direct):
                out.add(verb)
                continue
            if isinstance(eff, Emit):
                if eff.property_changes:
                    out.add(verb)
                else:
                    emit_edges.setdefault(verb, set()).add(eff.action)

    changed = True
    while changed:
        changed = False
        for verb, emitted in emit_edges.items():
            if verb in out:
                continue
            if any(target in out for target in emitted):
                out.add(verb)
                changed = True
    return frozenset(out)


@dataclass(frozen=True)
class BackgroundSatisfier:
    """One way to make a scene location have a target property value
    via the derivation graph. Naming a concept (`lampo`) plus the
    `varies=true` properties to set on the materialized instance
    (`{power_state: aktiva}`). The static side of the entity-pattern
    constraints is what concept selection already filtered on.

    Used by `SceneBuilder.apply_scene_preferences` to pre-place
    ambient entities — the table says "indoor rooms usually
    `lit_state=luma`", introspection finds `indoor_lit_by_active_lamp`,
    and this returns `(lampo, {power_state: aktiva})`. The scatter
    primitive places the lamp; `set` flips power_state."""
    concept: str
    set_properties: dict


def background_satisfiers(
    slot: str, target_value, scene_concept,
    derivations: list[Derivation], lex,
) -> list[BackgroundSatisfier]:
    """Enumerate ways to make `scene_concept` (a `Concept`) acquire
    property (slot, target_value) by adding ONE entity to the scene.

    Algorithm: walk derivations whose `implies` is
    `property(?, slot, target_value)` and whose `when` admits
    `scene_concept` (entity-pattern constraints satisfied by the
    concept's static properties). For each such derivation, find an
    entity-pattern in `given` not bound to the scene var; enumerate
    concepts whose static properties satisfy the entity-pattern.
    Yield (concept, varies-properties-to-set) pairs.

    "Static" means non-`varies` slots — those are pinned by the
    concept declaration. `varies=true` slots like `power_state` get
    a value at instance time; the satisfier carries the target value
    (`aktiva`) so the materializer knows what to set.

    Excludes cascade-emerged concepts (flako, skribaĵo) — pre-placing
    them preempts the cascade that would otherwise introduce them."""
    transient = cascade_emerged_concepts_for(derivations, lex)
    out: list[BackgroundSatisfier] = []
    for d in derivations:
        for imp in d.implies:
            if not (isinstance(imp, PropertyImplication)
                    and imp.slot == slot
                    and imp.value == target_value
                    and isinstance(imp.entity, Var)):
                continue
            scene_var = imp.entity
            when_ep = _entity_pattern_for_var(d.when, scene_var)
            if when_ep is not None and not _concept_satisfies_constraints(
                    scene_concept, when_ep.constraints, lex):
                continue
            for given_pat in d.given:
                ent_patt = _entity_pattern_in(given_pat)
                bv = _bind_var_in_pattern(given_pat)
                if ent_patt is None:
                    continue
                if bv is not None and bv is scene_var:
                    continue
                static, varies = _split_constraints_by_varies(
                    ent_patt.constraints, lex)
                for lemma, concept in lex.concepts.items():
                    if lemma in transient:
                        continue
                    if _concept_satisfies_constraints(concept, static, lex):
                        out.append(BackgroundSatisfier(lemma, varies))
    return out


def _entity_pattern_for_var(pattern, target_var) -> EntityPattern | None:
    """Find an EntityPattern in `pattern` that's bound (via And/Bind
    composition) to `target_var`. Returns None if `target_var` isn't
    bound by any reachable EntityPattern. Used to recover the static
    constraints on a particular var — `outdoor_is_luma`'s `when`
    binds L to `entity(type=location, indoor_outdoor=ekstera)`, so
    asking for var L returns those constraints."""
    if isinstance(pattern, AndPattern):
        bv_left = _bind_var_in_pattern(pattern.left)
        bv_right = _bind_var_in_pattern(pattern.right)
        ep_left = (pattern.left if isinstance(pattern.left, EntityPattern)
                   else None)
        ep_right = (pattern.right if isinstance(pattern.right, EntityPattern)
                    else None)
        if bv_left is target_var and ep_right is not None:
            return ep_right
        if bv_right is target_var and ep_left is not None:
            return ep_left
        for sub in (pattern.left, pattern.right):
            r = _entity_pattern_for_var(sub, target_var)
            if r is not None:
                return r
    return None


def _split_constraints_by_varies(constraints: dict, lex) -> tuple[dict, dict]:
    """Partition entity-pattern constraints into (static, varies). A
    constraint key is `varies` if its slot definition declares
    `varies=True`. Static keys (`type`, `concept`, `category`,
    `has_suffix`, plus non-varies slots) gate concept selection;
    varies keys get applied via `entity.set_property` after
    materialization."""
    static: dict = {}
    varies: dict = {}
    for k, v in constraints.items():
        if k in ("type", "concept", "category", "has_suffix"):
            static[k] = v
            continue
        slot_def = lex.slots.get(k)
        if slot_def is not None and getattr(slot_def, "varies", False):
            varies[k] = v
        else:
            static[k] = v
    return static, varies


def _concept_satisfies_constraints(concept, constraints: dict, lex) -> bool:
    """True iff `concept` satisfies every static EntityPattern
    constraint. Mirrors the runtime matcher in `patterns._entity_matches`
    but at the concept (lex) level — no trace, no derived state."""
    from ..containment import _concept_in_category
    for key, expected in constraints.items():
        if key == "type":
            if not lex.types.is_subtype(concept.entity_type, expected):
                return False
        elif key == "concept":
            if concept.lemma != expected:
                return False
        elif key == "category":
            if not _concept_in_category(concept, expected, lex):
                return False
        elif key == "has_suffix":
            if not concept.lemma.endswith(expected):
                return False
        else:
            vals = concept.properties.get(key, [])
            if expected not in vals:
                return False
    return True


def cascade_emerged_concepts_for(
    derivations: list[Derivation], lex,
) -> frozenset[str]:
    """Backwards-compat shim: callers that already had a Rule list
    use `cascade_emerged_concepts(rules)`. Background-satisfier callers
    only have derivations on hand, so we re-walk the canonical rule
    set via the lexicon's reach. Returns the same set when the
    canonical DEFAULT_DSL_RULES is what's loaded."""
    # Re-use the rule-walking version by importing rules lazily.
    # background_satisfiers is called from scene-build paths that
    # already have the canonical rule set in scope; this avoids a
    # circular import at module load.
    from .rules import DEFAULT_DSL_RULES
    return cascade_emerged_concepts(DEFAULT_DSL_RULES)


def cascade_emerged_concepts(rules: list[Rule]) -> frozenset[str]:
    """Concepts introduced by a `create_entity` effect in some rule's
    `then`. These are "transient" — they appear via cascade (flako from
    rain/spill, vitropecetoj from breakage) rather than being authored
    into a scene.

    Regression seeders (and lazy materialization) should skip these
    when picking a container or placing a target — pre-placing a flako
    in a scene preempts the cascade that would otherwise introduce it,
    and "actor walks to a fresh puddle of beer" is narratively
    incoherent. Only literal `concept=<lemma>` writes count; Var-valued
    concepts (e.g. `concept=K_spill` that resolves at firing time) are
    skipped here since they're driven by the bound concept slot.

    Pure introspection — no curated list, so a new cascade rule (e.g.
    `tornado_creates_debris`) automatically excludes its emitted
    concept from pre-placement."""
    out: set[str] = set()
    for rule in rules:
        for eff in _effects(rule):
            if isinstance(eff, CreateEntity) and isinstance(eff.concept, str):
                out.add(eff.concept)
    return frozenset(out)


@dataclass(frozen=True)
class CascadeSpec:
    """One way to procedurally satisfy a `("self_slot", actor, slot,
    target_value)` drive. The cascade rule fires when `actor` (bound
    to `agent_role` of `verb`) has `slot=pre_state` AND does `verb`;
    its `.changing(actor, slot, target_value)` flips the slot.

    Seeders consume these to build "actor needs X → planner finds verb"
    scenes without hardcoding the verb. See
    `esperanto_lm.ontology.regression.seeders.regress_for_self_slot`."""
    verb: str
    agent_role: str
    pre_state: Any


def _entity_pattern_in(patt) -> EntityPattern | None:
    """First EntityPattern reachable from `patt`. Mirrors
    `_bind_var_in_pattern` shape — used to recover the slot constraints
    on a role pattern of the form `entity(slot=val) & bind(VAR)`."""
    if isinstance(patt, EntityPattern):
        return patt
    if isinstance(patt, AndPattern):
        return (_entity_pattern_in(patt.left)
                or _entity_pattern_in(patt.right))
    return None


def self_slot_cascades(rules: list[Rule]) -> dict[tuple[str, Any], list[CascadeSpec]]:
    """Index of cascade rules keyed by their (slot, target_value) outcome.

    A cascade rule has the shape::

        when=event(VERB, ROLE=entity(SLOT=PRE) & bind(VAR))
        then=emit(...).changing(VAR, SLOT, POST)   # or change(VAR, SLOT, POST)

    The output maps `(SLOT, POST)` → list of `CascadeSpec(VERB, ROLE,
    PRE)`. Used by procedural seeders to build a scene satisfying a
    `self_slot` drive — pick a cascade, place the actor in PRE state,
    let the planner chain into VERB."""
    out: dict[tuple[str, Any], list[CascadeSpec]] = {}
    for rule in rules:
        if not isinstance(rule.when, EventPattern):
            continue
        verb = rule.when.action

        # Recover {var → role_name} so we can attribute a Change/Emit's
        # entity-var back to the role that bound it.
        var_to_role: dict[int, str] = {}
        for role_name, role_patt in rule.when.role_patterns.items():
            v = _bind_var_in_pattern(role_patt)
            if v is not None:
                var_to_role[id(v)] = role_name

        # Walk effects: collect Change(VAR, slot, val) plus Emit's
        # property_changes which encode the same shape.
        changes: list[tuple[Var, str, Any]] = []
        for eff in _effects(rule):
            if isinstance(eff, Change):
                if isinstance(eff.entity, Var):
                    changes.append((eff.entity, eff.slot, eff.value))
            elif isinstance(eff, Emit):
                for (ent, slot), val in eff.property_changes.items():
                    if isinstance(ent, Var):
                        changes.append((ent, slot, val))

        for var, slot, val in changes:
            role_name = var_to_role.get(id(var))
            if role_name is None:
                continue
            ent_patt = _entity_pattern_in(rule.when.role_patterns[role_name])
            pre_state = None
            if ent_patt is not None:
                pre = ent_patt.constraints.get(slot)
                if pre is not None and not isinstance(pre, Var):
                    pre_state = pre
            spec = CascadeSpec(verb, role_name, pre_state)
            out.setdefault((slot, val), []).append(spec)
    return out


def verb_relation_kinds(rules: list[Rule]) -> dict[str, frozenset[str]]:
    """Per-verb set of relation names the rule mutates — used by the
    realizer's relation-change attribution to gate which havi/en
    diff-entries an event can claim."""
    out: dict[str, set[str]] = {}
    for rule in rules:
        if not isinstance(rule.when, EventPattern):
            continue
        action = rule.when.action
        from .effects import AddRelation, RemoveRelation
        for eff in _effects(rule):
            if isinstance(eff, TransferN):
                out.setdefault(action, set()).add("havi")
            elif isinstance(eff, (AddRelation, RemoveRelation)):
                out.setdefault(action, set()).add(eff.relation)
    return {k: frozenset(v) for k, v in out.items()}


def slot_reachable_for_concept(
    concept, slot: str, lex, derivations: list[Derivation],
) -> bool:
    """True if some derivation could write to `slot` on `concept` —
    given the concept's static properties and parts, with no extra
    entities added.

    Used by drive-validity checks to recognize that varies/derived
    slots (`lock_state` on `valizo`, `posture` on `onklo`) are
    meaningful even though the bake skips them: a derivation
    (`host_lock_state_locked_from_seruro`, `animate_default_standing`)
    will populate them at runtime.

    Algorithm: walk derivations whose `implies` writes to `slot` with
    the implication's host bound to a Var. For each, check:
      1. the host-var's EntityPattern constraints (in `when`) admit
         `concept` — via `_concept_satisfies_constraints`;
      2. every `given` pattern is satisfiable from the concept's parts.
         RelPatterns of `havas_parton` mean a part must exist whose
         concept satisfies the part-var's EntityPattern (static
         constraints only — `varies` slots like `lock_state=ŝlosita`
         on the seruro part get a value at instance time).
         Other given-patterns (non-part relations) are conservatively
         accepted; the host-when admittance is the load-bearing
         check for derivations like `animate_default_standing` which
         have no `given`."""
    for d in derivations:
        for imp in d.implies:
            if not (isinstance(imp, PropertyImplication)
                    and imp.slot == slot
                    and isinstance(imp.entity, Var)):
                continue
            host_var = imp.entity
            when_ep = _entity_pattern_for_var(d.when, host_var)
            if when_ep is not None:
                if not _concept_satisfies_constraints(
                        concept, when_ep.constraints, lex):
                    continue
            if _given_satisfied_by_parts(d.given, host_var, concept, lex):
                return True
    return False


# Relations whose participants are fixed at concept-time (asserted
# from concept.parts when the entity is instantiated). A derivation
# whose `given` references one of these on its implied entity can't
# be subgoaled by the planner — banano can't be made to have a
# seruro part. Add others as new asserted-only relations appear.
_STATIC_RELATIONS = frozenset({"havas_parton"})


def fully_derivable_slots(lex, derivations) -> dict:
    """Returns {slot_name: [type_name, ...]} — slots whose value is
    produced by a derivation whose `when`/`given` constrain the
    implied entity only via dynamic (runtime-mutable) state. The
    planner can subgoal any dynamic constraint, so any concept of
    one of the listed types is eligible to satisfy a role-property
    requirement on that slot.

    A derivation like `agent_illuminated` (when=animate, given=samloke
    + lit_state=luma location) qualifies: samloke is runtime-mutable.
    A derivation like `host_lock_state_from_seruro` (given havas_parton
    on the host) doesn't: havas_parton is concept-time. Banano can't
    be subgoaled into having a seruro part.

    Called once per lex at load time; the result is cached on
    `lex.concept_index.derivable`. Callers should read from there
    instead of re-invoking this function."""

    def _entity_patterns_binding(pat, target_var):
        if isinstance(pat, AndPattern):
            if (isinstance(pat.left, EntityPattern)
                    and isinstance(pat.right, BindPattern)
                    and pat.right.target is target_var):
                yield pat.left
            elif (isinstance(pat.right, EntityPattern)
                    and isinstance(pat.left, BindPattern)
                    and pat.left.target is target_var):
                yield pat.right
            else:
                yield from _entity_patterns_binding(pat.left, target_var)
                yield from _entity_patterns_binding(pat.right, target_var)

    def _rel_patterns(pat):
        if isinstance(pat, RelPattern):
            yield pat
        elif isinstance(pat, AndPattern):
            yield from _rel_patterns(pat.left)
            yield from _rel_patterns(pat.right)

    def _arg_uses(arg_pat, var):
        if arg_pat is var:
            return True
        if isinstance(arg_pat, BindPattern) and arg_pat.target is var:
            return True
        if isinstance(arg_pat, AndPattern):
            return (_arg_uses(arg_pat.left, var)
                    or _arg_uses(arg_pat.right, var))
        return False

    out: dict = {}
    for d in derivations:
        for imp in d.implies:
            if not isinstance(imp, PropertyImplication):
                continue
            if not isinstance(imp.entity, Var):
                continue
            target = imp.entity
            type_constraint = None
            for ep in _entity_patterns_binding(d.when, target):
                t = ep.constraints.get("type")
                if isinstance(t, str):
                    type_constraint = t
            if type_constraint is None:
                for clause in d.given:
                    for ep in _entity_patterns_binding(clause, target):
                        t = ep.constraints.get("type")
                        if isinstance(t, str):
                            type_constraint = t
                            break
                    if type_constraint is not None:
                        break
            if type_constraint is None:
                continue
            has_static = False
            for clause in d.given:
                for rp in _rel_patterns(clause):
                    if rp.relation not in _STATIC_RELATIONS:
                        continue
                    for arg in rp.arg_patterns.values():
                        if _arg_uses(arg, target):
                            has_static = True
                            break
                    if has_static:
                        break
                if has_static:
                    break
            if has_static:
                continue
            out.setdefault(imp.slot, []).append(type_constraint)

    return out


def concept_models_slot(
    concept, slot: str, lex, derivations: list,
) -> bool:
    """True iff `slot` is meaningfully tracked for `concept` per the
    schema. Three sources of meaningfulness:

      1. `concept.properties` declares `slot` directly — the data
         author has flagged it as relevant (ovo declares
         cooking_state, so kuiri(ovo) is meaningful).
      2. The slot is `pervasive` — applies broadly to its
         `applies_to` types via a default derivation (hunger,
         wetness, temperature, cleanliness).
      3. A derivation could populate `slot` on this concept given
         its parts (lock_state on valizo via seruro; posture on
         onklo via animate_default_standing). Catches varies slots
         the bake intentionally skips.

    Returns False when none of these hold — the concept doesn't
    model the slot, and writing it would be a narratively-empty
    state change ("salato attachment=fiksita", "vortaro
    presence=manĝita"). Used by:

      - planner action grounding to skip effects targeting
        irrelevant concepts (relaxed graph then reports goal as
        unreachable → h_FF gate catches at sample time);
      - sampler's cheap pre-filter to bail before the costly
        spawn + grounding cycle.

    The lex argument is the Lexicon; derivations is the runtime
    derivation list (RUNTIME_DERIVATIONS) — passed in rather than
    imported to avoid a circular dependency with dsl/rules.py."""
    if concept is None:
        return False
    slot_def = lex.slots.get(slot)
    if slot_def is None:
        return False
    if slot in concept.properties:
        return True
    if getattr(slot_def, "pervasive", False):
        return True
    return slot_reachable_for_concept(concept, slot, lex, derivations)


def _given_satisfied_by_parts(
    given: list, host_var: Var, concept, lex,
) -> bool:
    """True if every `havas_parton(host_var, <part-var>)` clause in
    `given` is matched by some part of `concept`, AND every non-part
    given is conservatively accepted. Used by
    `slot_reachable_for_concept`."""
    # Collect part-var → static-constraints (skip varies).
    part_var_constraints: dict[int, dict] = {}
    for given_pat in given:
        if not isinstance(given_pat, RelPattern):
            continue
        if given_pat.relation != "havas_parton":
            continue
        # Identify the part-var: the non-host bound var among args.
        part_var = None
        for arg in given_pat.arg_patterns.values():
            bv = _bind_var_in_pattern(arg)
            if bv is None or bv is host_var:
                continue
            part_var = bv
            break
        if part_var is not None:
            part_var_constraints.setdefault(id(part_var), {})
    # Second pass: collect EntityPattern constraints attached to each
    # part-var via separate `entity(...) & bind(VAR)` given-clauses.
    for given_pat in given:
        if isinstance(given_pat, RelPattern):
            continue
        ep = _entity_pattern_in(given_pat)
        bv = _bind_var_in_pattern(given_pat)
        if ep is None or bv is None:
            continue
        if id(bv) in part_var_constraints:
            static, _varies = _split_constraints_by_varies(
                ep.constraints, lex)
            part_var_constraints[id(bv)] = static
    # Each part-var's static constraints must be satisfied by some
    # part of `concept`.
    if part_var_constraints:
        for static in part_var_constraints.values():
            ok = False
            for part in (concept.parts or []):
                part_concept = lex.concepts.get(part.concept)
                if part_concept is None:
                    continue
                if _concept_satisfies_constraints(
                        part_concept, static, lex):
                    ok = True
                    break
            if not ok:
                return False
    return True


def relation_arg_excludes(lex) -> dict:
    """Static index of property values forbidden at each (relation, arg)
    position. Walks `lex.relations[*].arg_patterns` and lifts simple
    NotPattern(EntityPattern(slot=val)) shapes to a lookup table.

    Returns `{(rel_name, arg_idx): {slot: frozenset[forbidden_values]}}`.
    Used by the forward planner to prune action groundings before
    search: any grounding whose role binding would produce a relation
    matching a forbidden entry is discarded statically, without
    constructing the trace.

    Compound shapes (And/Or/non-EntityPattern NotPattern) are skipped —
    they still gate at runtime via `validate_relation`, but don't
    contribute to the static cheap-pruning index. This is the same
    pattern as cascade introspection: narrow shape supported by the
    index, full DSL behind runtime checks."""
    out: dict = {}
    for rel_name, rel in lex.relations.items():
        if not rel.arg_patterns:
            continue
        for i, pat in enumerate(rel.arg_patterns):
            if pat is None:
                continue
            forbidden = _extract_forbidden_from_not(pat)
            if not forbidden:
                continue
            key = (rel_name, i)
            slot_map = out.setdefault(key, {})
            for slot, vals in forbidden.items():
                existing = slot_map.get(slot, frozenset())
                slot_map[slot] = existing | frozenset(vals)
    return out


def _extract_forbidden_from_not(pattern) -> dict:
    """If `pattern` is `NotPattern(EntityPattern(slot=val, ...))`, return
    `{slot: [val], ...}` — the slot/value pairs whose presence on the
    entity makes the NotPattern fail. Returns {} for other shapes."""
    if not isinstance(pattern, NotPattern):
        return {}
    inner = pattern.inner
    if not isinstance(inner, EntityPattern):
        return {}
    out: dict = {}
    for slot, val in inner.constraints.items():
        if slot in ("type", "concept", "category", "has_suffix"):
            continue  # not a slot-level forbid
        if isinstance(val, list):
            out[slot] = list(val)
        else:
            out[slot] = [val]
    return out
