"""Pattern AST for the declarative matching DSL.

Patterns are a small algebraic tree — `event`, `entity`, `rel`, `closure`,
`has_concept_field` — composed via `&`, `|`, `~`. Each pattern kind
plays one of two matching roles:

  search mode:       the pattern is a top-level clause (e.g. a
                     derivation's `when`, a causal rule's `given`).
                     The matcher enumerates worlds that satisfy it and
                     yields extended bindings.
  value-filter mode: the pattern is embedded in a role slot
                     (`theme=entity(...) & bind(T)`) or inside a
                     relation arg (`rel("en", container=C)`). The
                     matcher applies it against a specific value
                     (entity id) and yields bindings or nothing.

Patterns answer both via `.search()` and `.apply_to_value()` — which one
the engine calls depends on where the pattern appears.

Variables are declared via `var()` and used inline with the walrus
operator: `bind(T := var())` at first use, bare `T` thereafter. Vars
are compared by identity (not name), so re-use within a rule means the
same Python object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


# ---------------------------- variables ----------------------------

class Var:
    """A match variable. Identity-compared; name is for error messages.

    Vars are used as `bindings` dict keys (millions of lookups per
    planning session) and never mutate or get copied — each `var()`
    call constructs a fresh instance with module-lifetime identity.
    The hash is cached at construction so __hash__ avoids the per-call
    `id()` builtin frame setup; safe because identity is by-design
    immutable for Vars."""
    __slots__ = ("name", "_hash")

    def __init__(self, name: str = "_anon"):
        self.name = name
        self._hash = id(self)

    def __repr__(self) -> str:
        return f"${self.name}"

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return self is other


def var(name: str = "_anon") -> Var:
    """Construct a fresh match variable. Typical use: `bind(T := var("T"))`."""
    return Var(name)


class VarProp:
    """Reference to a bound entity's property value, for use inside a
    `Compare` constraint. Resolved at match time by looking up
    `bindings[var]` to get the entity id, then reading that entity's
    `prop` from the trace via the matcher's context.

    Example: `Compare("<", VarProp(WATER, "denseco"))` reads as
    "less than the denseco of the entity currently bound to WATER".
    """
    __slots__ = ("var_", "prop")

    def __init__(self, var_: Var, prop: str):
        self.var_ = var_
        self.prop = prop

    def __repr__(self) -> str:
        return f"${self.var_.name}.{self.prop}"


class Compare:
    """Numeric comparison constraint on a slot value inside an
    `EntityPattern`. The constraint key names the slot on the matching
    entity (LHS); `rhs` is a literal number, numeric string, or
    `VarProp` resolved from current bindings.

    Operators: '<', '<=', '>', '>=', '=='.

    Vacuous-on-missing-data: if either side has no numeric value,
    the comparison FAILS the match (strict). This is the opposite of
    Relation.arg_compare's vacuous-pass semantics — derivation
    pattern matching is selective by design (we want the rule to
    fire only when the comparison is meaningfully true), while
    relation gates default open so existing scenes aren't broken.
    """
    __slots__ = ("op", "rhs")

    def __init__(self, op: str, rhs):
        if op not in {"<", "<=", ">", ">=", "=="}:
            raise ValueError(f"Compare op {op!r} not in <, <=, >, >=, ==")
        self.op = op
        self.rhs = rhs

    def __repr__(self) -> str:
        return f"Compare({self.op!r}, {self.rhs!r})"


class VarList(Var):
    """A match variable whose binding is a list of entities, not a single
    one. Used for variadic event roles like `fari.parts` where the
    binding holds one entity per declared part of the theme concept.

    Identity-compared exactly like Var; subclass purely so downstream
    code (engine effect application, rule effects index, realizer)
    can dispatch on `isinstance(v, VarList)` to iterate the binding."""
    __slots__ = ()


def var_list(name: str = "_anon") -> VarList:
    """Construct a fresh list-valued match variable. Pair with
    `bind(Ps := var_list("P"))` on a list-kind event role; pass to
    `for_each(Ps, item_var, *effects)` in the rule's then-clause."""
    return VarList(name)


Bindings = dict[Var, Any]


# ---------------------------- base --------------------------------

class Pattern:
    """Base class for all patterns. Subclasses implement one or both of
    `search` and `apply_to_value` depending on how they're used."""

    def variables(self) -> set[Var]:
        """All Vars referenced transitively in this pattern."""
        raise NotImplementedError

    def search(self, ctx, bindings: Bindings) -> Iterator[Bindings]:
        """Top-level search. Yield extended bindings for each world that
        satisfies this pattern under the given context."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support top-level search")

    def apply_to_value(
        self, value: Any, ctx, bindings: Bindings,
    ) -> Iterator[Bindings]:
        """Value-filter mode. Apply this pattern to a specific value
        (usually an entity id from a role slot). Yield extended bindings
        or nothing if the pattern fails for this value."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support value-filter mode")

    def __and__(self, other: "Pattern | Var") -> "AndPattern":
        return AndPattern(self, _coerce(other))

    def __or__(self, other: "Pattern | Var") -> "OrPattern":
        return OrPattern(self, _coerce(other))

    def __invert__(self) -> "NotPattern":
        return NotPattern(self)

    def __rand__(self, other: "Pattern | Var") -> "AndPattern":
        return AndPattern(_coerce(other), self)


def _coerce(x: "Pattern | Var") -> Pattern:
    """Auto-wrap a bare Var as a BindPattern so `T & entity(...)` works.
    Scalar literals (str/int/float/bool) coerce to LiteralValuePattern
    so callers can write `rel("scias", rel_type="en", ...)` to
    pin a positional arg to an exact value (used by the scias_lokon
    derivations to filter on rel_type without needing OR over
    multiple bound vars)."""
    if isinstance(x, Pattern):
        return x
    if isinstance(x, Var):
        return BindPattern(x)
    if isinstance(x, (str, int, float, bool)):
        return LiteralValuePattern(x)
    raise TypeError(f"expected Pattern or Var, got {type(x).__name__}: {x!r}")


@dataclass(eq=False)
class LiteralValuePattern(Pattern):
    """Matches iff the value-under-test equals `value`. Used as a
    rel-arg constraint, e.g. `rel("scias", rel_type="en", ...)`
    pins the rel_type position to literal "en"."""
    value: Any

    def variables(self) -> set[Var]:
        return set()

    def apply_to_value(self, value, ctx, bindings):
        if value == self.value:
            yield bindings

    def search(self, ctx, bindings):
        return
        yield  # pragma: no cover - generator marker


# ---------------------------- bind --------------------------------

@dataclass(eq=False)
class BindPattern(Pattern):
    target: Var

    def variables(self) -> set[Var]:
        return {self.target}

    def apply_to_value(self, value, ctx, bindings):
        # Mutate-and-restore: write target into the SHARED bindings
        # dict, yield it, then delete the key on the way out. Saves
        # millions of `{**bindings, k: v}` allocations per planning
        # session.
        #
        # Safety contract: callers may consume the yielded `bindings`
        # only during the for-iteration. To capture across iterations
        # (e.g. `list(...)` materialization), copy explicitly with
        # `dict(b)`. The engine's causal phase does this at line 450
        # of engine.py; the derivation phase consumes immediately.
        if self.target in bindings:
            if bindings[self.target] == value:
                yield bindings
            return
        bindings[self.target] = value
        try:
            yield bindings
        finally:
            del bindings[self.target]

    def search(self, ctx, bindings):
        # Bind at top-level: only makes sense if the var is already bound
        # (then we "satisfy" by passing through).
        if self.target in bindings:
            yield bindings


def bind(target: Var | BindPattern) -> BindPattern:
    """Mark a role slot or entity value as bound to `target`. Identity
    check on Var; repeated use within a rule refers to the same slot.
    """
    if isinstance(target, BindPattern):
        return target
    if not isinstance(target, Var):
        raise TypeError(
            f"bind() expects Var, got {type(target).__name__}: {target!r}")
    return BindPattern(target)


def bind_list(target: VarList) -> BindPattern:
    """Bind a list-valued event role to a VarList. The matcher writes
    the role's value (a list of entity ids) into bindings[target] as-is.
    Downstream effect application (for_each) iterates the bound list.

    Validates the target is actually a VarList; ordinary `bind()` would
    silently accept it but downstream loops expect list semantics."""
    if not isinstance(target, VarList):
        raise TypeError(
            f"bind_list() expects VarList, got {type(target).__name__}: "
            f"{target!r}")
    return BindPattern(target)


# ---------------------------- entity ------------------------------

@dataclass(eq=False)
class EntityPattern(Pattern):
    """Matches an entity given property/type constraints.

    Constraint keys:
      `type` — required entity_type (subtype-checked via Lexicon).
      `concept` — exact concept-lemma match.
      `has_suffix` — concept lemma must end with this suffix.
      anything else — treated as a slot name; value must hold (asserted
                      or derived) on the entity at match time.

    Unknown slot names raise at rule-construction; see validation.
    """
    constraints: dict[str, Any]

    def variables(self) -> set[Var]:
        return set()

    def apply_to_value(self, value, ctx, bindings):
        ent = ctx.trace.entities.get(value)
        if ent is None:
            return
        if _entity_matches(ent, self.constraints, ctx, bindings):
            yield bindings

    def search(self, ctx, bindings):
        # Indexed fast paths: pick the most selective constraint
        # available. Concept is more selective than type
        # (each concept has fewer entities than its type does),
        # so prefer concept when present. Falls back to type, then
        # full scan.
        concept_constraint = self.constraints.get("concept")
        if isinstance(concept_constraint, str):
            for eid, ent in ctx.entities_of_concept(concept_constraint):
                if _entity_matches(ent, self.constraints, ctx, bindings):
                    yield bindings
            return
        type_constraint = self.constraints.get("type")
        if isinstance(type_constraint, str):
            for eid, ent in ctx.entities_of_type(type_constraint):
                if _entity_matches(ent, self.constraints, ctx, bindings):
                    yield bindings
            return
        for eid, ent in ctx.trace.entities.items():
            if _entity_matches(ent, self.constraints, ctx, bindings):
                yield bindings


def _entity_matches(ent, constraints: dict[str, Any], ctx, bindings) -> bool:
    for key, expected in constraints.items():
        # Compare-valued constraints resolve numerically against this
        # entity's slot (LHS) and a literal or VarProp (RHS).
        if isinstance(expected, Compare):
            if not _compare_entity_slot(ent, key, expected, ctx, bindings):
                return False
            continue
        # Var-valued constraints resolve from current bindings — lets
        # patterns like `entity(owner=A)` test "this entity's owner
        # equals the entity bound to A". An unbound Var fails the
        # match (no candidate value to compare against yet).
        if isinstance(expected, Var):
            resolved = bindings.get(expected)
            if resolved is None:
                return False
            expected = resolved
        if key == "type":
            if not ctx.lexicon.types.is_subtype(ent.entity_type, expected):
                return False
        elif key == "has_suffix":
            if not ent.concept_lemma.endswith(expected):
                return False
        elif key == "concept":
            # Exact concept-lemma match — the entity instantiates this
            # specific concept. Used by rules that key on a concept's
            # identity (e.g. viŝi_destroys_skribaĵo matches themes
            # whose concept is exactly "skribaĵo", not just any
            # physical thing).
            if ent.concept_lemma != expected:
                return False
        elif key == "category":
            # Transitive `concept.category` chain match. Walks
            # supertypes via `category` field — "is_a" classification.
            # e.g. `entity(category="surfaco")` matches tablo, breto,
            # sofo (all categorized as surfaco). Mirrors the
            # `containment.jsonl` `category` pattern field.
            from ..containment import _concept_in_category
            concept = ctx.lexicon.concepts.get(ent.concept_lemma)
            if concept is None:
                return False
            if not _concept_in_category(concept, expected, ctx.lexicon):
                return False
        else:
            # Slot lookup: check effective property (asserted | derived).
            actual = ctx.effective_property(ent.id, key)
            if expected is Ellipsis:
                # Wildcard: succeed iff the slot is set (any value).
                # Spelled `entity(slot=...)` at the call site.
                if actual is None or actual == [] or actual == "":
                    return False
            elif not _value_matches(actual, expected):
                return False
    return True


def _value_matches(actual, expected) -> bool:
    """Property values may be scalars or single-element lists. Accept both."""
    if actual is None:
        return False
    if isinstance(actual, list):
        return expected in actual
    return actual == expected


def _to_float(v):
    """Coerce a slot value (list or scalar) to a single float.
    Returns None when missing or non-numeric."""
    if v is None or v == [] or v == "":
        return None
    if isinstance(v, list):
        v = v[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _compare_entity_slot(ent, slot: str, cmp: "Compare", ctx, bindings) -> bool:
    """Evaluate a Compare constraint: this entity's `slot` (LHS) op
    `cmp.rhs` (literal or VarProp). Both sides must resolve to floats
    or the match fails (strict — see Compare docstring for why)."""
    lhs = _to_float(ctx.effective_property(ent.id, slot))
    if lhs is None:
        return False
    rhs_spec = cmp.rhs
    if isinstance(rhs_spec, VarProp):
        rhs_eid = bindings.get(rhs_spec.var_)
        if rhs_eid is None:
            return False
        rhs = _to_float(ctx.effective_property(rhs_eid, rhs_spec.prop))
    else:
        rhs = _to_float(rhs_spec)
    if rhs is None:
        return False
    op = cmp.op
    if op == "<":  return lhs < rhs
    if op == "<=": return lhs <= rhs
    if op == ">":  return lhs > rhs
    if op == ">=": return lhs >= rhs
    if op == "==": return lhs == rhs
    return False


def entity(**constraints) -> EntityPattern:
    """Match an entity by type, concept, and/or properties. Keys:
    `type`, `concept`, `has_suffix`, or any slot name.

        entity(type="artifact", fragility="fragila")
        entity(concept="skribaĵo")
    """
    return EntityPattern(dict(constraints))


# ---------------------------- rel ---------------------------------

@dataclass(eq=False)
class RelPattern(Pattern):
    """Match a relation instance by name and arg patterns.

    `arg_patterns` is keyed by positional slot name (e.g. "container",
    "contained" for "en"; "owner", "theme" for "havi"). Resolved
    at match time against the relation's declared arg order.
    """
    relation: str
    arg_patterns: dict[str, Pattern]

    def variables(self) -> set[Var]:
        out: set[Var] = set()
        for p in self.arg_patterns.values():
            out |= p.variables()
        return out

    def search(self, ctx, bindings):
        # Arg names come straight from the Relation schema — the
        # lexicon is the single source of truth.
        rel_def = ctx.lexicon.relations.get(self.relation)
        if rel_def is None:
            return
        # Pre-compute filter constraints from bound vars: if an arg
        # pattern is `BindPattern(V)` and V is already in `bindings`,
        # the corresponding tuple position must equal bindings[V].
        # Filtering candidates here avoids recursing into _apply_args
        # for tuples that can't possibly match — a major win when
        # the planner subgoals into a derivation with most vars
        # already bound by the parent.
        arg_filters: list[tuple[int, str]] = []
        for arg_idx, name in enumerate(rel_def.arg_names):
            patt = self.arg_patterns.get(name)
            if patt is None:
                continue
            v = _bound_var_value(patt, bindings)
            if v is not None:
                arg_filters.append((arg_idx, v))

        # Use the relation-by-name index instead of scanning all
        # asserted + derived relations per pattern call. Symmetric
        # relations yield matches for both arg orderings —
        # `samloke(A, B)` should also satisfy a query for
        # `rel("samloke", a=B, b=A)`. We expand swaps BEFORE filtering
        # by arg_filters: otherwise an asserted `apud(koridoro, kuirejo)`
        # would be dropped when querying for `neighbor=koridoro`, since
        # only the swap matches the filter.
        seen: set[tuple[str, ...]] = set()
        candidates: list[tuple[str, ...]] = []
        for args in ctx.relations_of(self.relation):
            if len(args) != len(rel_def.arg_names):
                continue
            for cand in (
                    (args, (args[1], args[0]))
                    if rel_def.symmetric and len(args) == 2
                    and args[0] != args[1]
                    else (args,)):
                if cand in seen:
                    continue
                if arg_filters and not all(
                        cand[i] == v for i, v in arg_filters):
                    continue
                seen.add(cand)
                candidates.append(cand)

        for args in candidates:
            yield from _apply_args(
                self.arg_patterns, rel_def.arg_names, args, ctx, bindings)


    def apply_to_value(self, value, ctx, bindings):
        # Rel is a relational existence check — no inherent "value"
        # position. Treat as search with whatever bindings are in scope.
        yield from self.search(ctx, bindings)


def _bound_var_value(pattern, bindings):
    """If `pattern` is a BindPattern (or And-wrapping one) whose Var
    is already in `bindings`, return the bound value. LiteralValue
    patterns return their literal directly so the rel-search loop
    pre-filters candidates instead of iterating the full relation
    index. Otherwise None — caller treats as "no constraint to check"."""
    if isinstance(pattern, LiteralValuePattern):
        return pattern.value
    if isinstance(pattern, BindPattern):
        if pattern.target in bindings:
            return bindings[pattern.target]
        return None
    if isinstance(pattern, AndPattern):
        v = _bound_var_value(pattern.left, bindings)
        if v is not None:
            return v
        return _bound_var_value(pattern.right, bindings)
    return None


def _apply_args(arg_patterns, arg_names, arg_values, ctx, bindings):
    """Apply a dict of role-name → pattern to a positional arg tuple.
    Unmentioned role names are wildcards (no constraint)."""
    def recur(i: int, b: Bindings) -> Iterator[Bindings]:
        if i == len(arg_names):
            yield b
            return
        name = arg_names[i]
        val = arg_values[i]
        patt = arg_patterns.get(name)
        if patt is None:
            yield from recur(i + 1, b)
            return
        for b2 in patt.apply_to_value(val, ctx, b):
            yield from recur(i + 1, b2)
    yield from recur(0, bindings)


def rel(relation: str, **arg_patterns) -> RelPattern:
    """Match a relation instance by name and keyword args per-role.

        rel("en", container=C, contained=bind(I))
    """
    return RelPattern(relation, {k: _coerce(v) for k, v in arg_patterns.items()})


# ---------------------------- event -------------------------------

@dataclass(eq=False)
class EventPattern(Pattern):
    """Matches the focus event (the one a causal rule fires on).

    Role keys in `role_patterns` correspond to the event's `roles` dict.
    The matcher resolves each role's entity id and applies the
    corresponding pattern in value-filter mode.
    """
    action: str
    role_patterns: dict[str, Pattern]

    def variables(self) -> set[Var]:
        out: set[Var] = set()
        for p in self.role_patterns.values():
            out |= p.variables()
        return out

    def search(self, ctx, bindings):
        ev = ctx.focus_event
        if ev is None or ev.action != self.action:
            return
        yield from _apply_event_roles(self.role_patterns, ev, ctx, bindings)


def _apply_event_roles(role_patterns, ev, ctx, bindings):
    """Apply each role pattern as a value-filter on the event's role
    entity id. Shared by EventPattern / PastEventPattern / CausedByPattern."""
    role_items = list(role_patterns.items())

    def recur(i: int, b: Bindings) -> Iterator[Bindings]:
        if i == len(role_items):
            yield b
            return
        role_name, patt = role_items[i]
        if role_name not in ev.roles:
            return
        val = ev.roles[role_name]
        for b2 in patt.apply_to_value(val, ctx, b):
            yield from recur(i + 1, b2)

    yield from recur(0, bindings)


def event(action: str, **role_patterns) -> EventPattern:
    """Match a causal rule's triggering event.

        event("fali", theme=entity(fragility="fragila") & bind(T))
    """
    return EventPattern(action, {k: _coerce(v) for k, v in role_patterns.items()})


# ------------------------ past event / caused_by ------------------------

@dataclass(eq=False)
class PastEventPattern(Pattern):
    """Match any event in the trace's history (not the focus event).

    Used in `given` clauses as a guard ("has this already fired?") or
    for cross-event reasoning. Negate with `~past_event(...)` to assert
    absence.
    """
    action: str
    role_patterns: dict[str, Pattern]

    def variables(self) -> set[Var]:
        out: set[Var] = set()
        for p in self.role_patterns.values():
            out |= p.variables()
        return out

    def search(self, ctx, bindings):
        for ev in ctx.trace.events:
            if ev.action != self.action:
                continue
            yield from _apply_event_roles(
                self.role_patterns, ev, ctx, bindings)


def past_event(action: str, **role_patterns) -> PastEventPattern:
    """Match against any event in the trace's history. Negate for
    "hasn't already fired" guards:

        given=[~past_event("fali", theme=I)]
    """
    return PastEventPattern(
        action, {k: _coerce(v) for k, v in role_patterns.items()})


@dataclass(eq=False)
class CausedByPattern(Pattern):
    """Match an event in the focus event's `caused_by` chain (direct
    causes only, not transitive). Used to reach back to the triggering
    event's context — e.g. `person_walks_on_hazard_falls` needs the
    origin entity from the rompiĝi/fali that created the hazard.
    """
    action: str
    role_patterns: dict[str, Pattern]

    def variables(self) -> set[Var]:
        out: set[Var] = set()
        for p in self.role_patterns.values():
            out |= p.variables()
        return out

    def search(self, ctx, bindings):
        focus = ctx.focus_event
        if focus is None:
            return
        cause_ids = set(focus.caused_by)
        if not cause_ids:
            return
        for ev in ctx.trace.events:
            if ev.id not in cause_ids:
                continue
            if ev.action != self.action:
                continue
            yield from _apply_event_roles(
                self.role_patterns, ev, ctx, bindings)


def caused_by(action: str, **role_patterns) -> CausedByPattern:
    """Match an event in the focus event's `caused_by` set. Direct
    causes only (not transitive).

        given=[caused_by("rompiĝi", theme=bind(O))]
    """
    return CausedByPattern(
        action, {k: _coerce(v) for k, v in role_patterns.items()})


# ---------------------------- closure -----------------------------

@dataclass(eq=False)
class ClosurePattern(Pattern):
    """Transitive closure over a set of relations.

    Walks relation edges starting from `from_` (must be already bound).
    For each reachable entity, applies `to_` as a value-filter AND
    `where` (if given) as an additional value-filter. Cycles detected
    by visited set.

    `relations` is interpreted as an undirected union over the named
    relations — an edge `(a, b)` of a named relation contributes both
    a→b and b→a. Matches spatial-adjacency intuition.

    `max_steps` bounds the BFS depth. `max_steps=1` yields only direct
    neighbors (one-hop adjacency), useful for layer-by-layer cascades
    where the engine's outer loop drives further propagation. None
    (default) means unlimited.
    """
    relations: frozenset[str]
    from_: Var
    to_: Pattern
    where: Optional[Pattern]
    max_steps: Optional[int] = None

    def variables(self) -> set[Var]:
        out = {self.from_} | self.to_.variables()
        if self.where is not None:
            out |= self.where.variables()
        return out

    def search(self, ctx, bindings):
        start = bindings.get(self.from_)
        if start is None:
            return
        visited = {start}
        frontier = [start]
        step = 0
        # Pre-fetch the relation buckets once per top-level call:
        # ClosurePattern walks every node × every relation tuple, and
        # the unifier's `relations_of` is already cached per-trace —
        # one lookup beats one-scan-per-node.
        rel_buckets = [
            (rname, ctx.relations_of(rname))
            for rname in self.relations
        ]
        while frontier and (self.max_steps is None or step < self.max_steps):
            next_frontier: list[str] = []
            for node in frontier:
                for _rname, tuples in rel_buckets:
                    for args in tuples:
                        if len(args) != 2:
                            continue
                        if args[0] == node and args[1] not in visited:
                            next_frontier.append(args[1])
                        elif args[1] == node and args[0] not in visited:
                            next_frontier.append(args[0])
            for nb in next_frontier:
                visited.add(nb)
            for nb in next_frontier:
                for b1 in self.to_.apply_to_value(nb, ctx, bindings):
                    if self.where is None:
                        yield b1
                    else:
                        yield from self.where.apply_to_value(nb, ctx, b1)
            frontier = next_frontier
            step += 1


def closure(
    relations: set[str] | list[str] | frozenset[str],
    *, from_: Var, to_: Var | Pattern,
    where: Optional[Pattern] = None,
    max_steps: Optional[int] = None,
) -> ClosurePattern:
    """Transitive closure over the named relations. `from_` must be a
    pre-bound Var; `to_` is typically `bind(X)` optionally AND-composed
    with an entity filter. Undirected union semantics. `max_steps=1`
    restricts to direct neighbors.
    """
    if not isinstance(from_, Var):
        raise TypeError(
            f"closure(from_=...) expects Var, got {type(from_).__name__}")
    return ClosurePattern(
        frozenset(relations), from_, _coerce(to_), where, max_steps)


# ------------------------ concept field ---------------------------

@dataclass(eq=False)
class HasConceptFieldPattern(Pattern):
    """Read a concept-level property (e.g. `transforms_on_break`) and
    bind it to a variable. Useful for rules that branch on lexicon
    metadata without traversing the full entity/property model.
    """
    entity_var: Var
    field_name: str
    bind_target: Var

    def variables(self) -> set[Var]:
        return {self.entity_var, self.bind_target}

    def search(self, ctx, bindings):
        eid = bindings.get(self.entity_var)
        if eid is None:
            return
        ent = ctx.trace.entities.get(eid)
        if ent is None:
            return
        concept = ctx.lexicon.concepts.get(ent.concept_lemma)
        if concept is None:
            return
        values = concept.properties.get(self.field_name)
        if not values:
            return
        for v in values:
            if self.bind_target in bindings:
                if bindings[self.bind_target] == v:
                    yield bindings
            else:
                yield {**bindings, self.bind_target: v}


def has_concept_field(
    entity_var: Var, field_name: str, bind_target: Var,
) -> HasConceptFieldPattern:
    """Bind the named concept-level field of the entity referenced by
    `entity_var` to `bind_target`."""
    if not isinstance(entity_var, Var):
        raise TypeError("has_concept_field: entity_var must be a Var")
    if not isinstance(bind_target, Var):
        raise TypeError("has_concept_field: bind_target must be a Var")
    return HasConceptFieldPattern(entity_var, field_name, bind_target)


@dataclass(eq=False)
class ConceptModelsSlotPattern(Pattern):
    """Matches iff `entity_var`'s concept models the slot whose name
    is bound to `slot_var`. "Models" includes declared properties,
    pervasive slots, AND derivation-lifted slots — same predicate the
    planner's _action_effects_meaningful enforces. Used by mezuri's
    rule to require the theme to model whatever slot the instrument
    measures (termometro→temperature, pesilo→maso, ...) without
    hardcoding a per-tool whitelist."""
    entity_var: Var
    slot_var: Var

    def variables(self) -> set[Var]:
        return {self.entity_var, self.slot_var}

    def search(self, ctx, bindings):
        eid = bindings.get(self.entity_var)
        slot = bindings.get(self.slot_var)
        if eid is None or slot is None:
            return
        ent = ctx.trace.entities.get(eid)
        if ent is None:
            return
        concept = ctx.lexicon.concepts.get(ent.concept_lemma)
        if concept is None:
            return
        # Lazy import — patterns is a low-level module and importing
        # introspect/rules at module load would risk cycles.
        from .introspect import concept_models_slot
        from .rules import RUNTIME_DERIVATIONS
        if concept_models_slot(
                concept, slot, ctx.lexicon, RUNTIME_DERIVATIONS):
            yield bindings


def concept_models_slot_check(
    entity_var: Var, slot_var: Var,
) -> ConceptModelsSlotPattern:
    """Given-clause check: `entity_var`'s concept must model the slot
    named by the binding of `slot_var`. Both vars must be bound by
    earlier `when`/`given` clauses."""
    if not isinstance(entity_var, Var):
        raise TypeError(
            "concept_models_slot_check: entity_var must be a Var")
    if not isinstance(slot_var, Var):
        raise TypeError(
            "concept_models_slot_check: slot_var must be a Var")
    return ConceptModelsSlotPattern(entity_var, slot_var)


# ---------------------------- logic -------------------------------

@dataclass(eq=False)
class AndPattern(Pattern):
    left: Pattern
    right: Pattern

    def variables(self) -> set[Var]:
        return self.left.variables() | self.right.variables()

    def apply_to_value(self, value, ctx, bindings):
        for b1 in self.left.apply_to_value(value, ctx, bindings):
            yield from self.right.apply_to_value(value, ctx, b1)

    def search(self, ctx, bindings):
        # Common derivation/given idiom: `entity(...) & bind(T)` — the
        # left enumerates entity candidates, the right applies as a
        # value-filter on each. Detect this by checking whether either
        # side iterates entities; if so, drive iteration from there
        # and apply the other as value-filter (or fall back to search).
        from_entity = _find_entity_source(self.left)
        if from_entity is not None:
            for eid in _iter_entity_candidates(from_entity, ctx, bindings):
                yield from _value_or_search(self.right, eid, ctx, bindings)
            return
        from_entity = _find_entity_source(self.right)
        if from_entity is not None:
            for eid in _iter_entity_candidates(from_entity, ctx, bindings):
                yield from _value_or_search(self.left, eid, ctx, bindings)
            return
        # Neither side enumerates entities — pure search composition.
        for b1 in self.left.search(ctx, bindings):
            yield from self.right.search(ctx, b1)


def _find_entity_source(pattern: Pattern) -> Optional[Pattern]:
    """Return a pattern that should drive candidate iteration — either
    an EntityPattern or an OrPattern whose branches are all
    entity-sources. Recurses through And/Or so e.g.
    `(entity(made_of="wood") | entity(made_of="paper")) & bind(T)` and
    `(entity(...) & bind(T)) & bind(U)` both work."""
    if isinstance(pattern, EntityPattern):
        return pattern
    if isinstance(pattern, OrPattern):
        # Both branches must produce entity candidates; otherwise we
        # can't enumerate the union.
        if (_find_entity_source(pattern.left) is not None
                and _find_entity_source(pattern.right) is not None):
            return pattern
        return None
    if isinstance(pattern, AndPattern):
        return (_find_entity_source(pattern.left)
                or _find_entity_source(pattern.right))
    return None


def _iter_entity_candidates(pattern: Pattern, ctx, bindings):
    """Enumerate distinct entity_ids matching an EntityPattern or an
    Or-union of them. Dedupes across branches so an entity matching
    multiple branches yields once. `bindings` lets EntityPattern
    constraints with Var values resolve against the current bindings
    (e.g. `entity(owner=A)` checks ent.owner == bindings[A]).

    Type-indexed fast path mirrors `EntityPattern.search` — when
    the pattern constrains `type=X`, hit `ctx.entities_of_type(X)`
    instead of scanning every entity. Frequent enough in derivation
    cascades that this shows up as a hot loop without the index."""
    if isinstance(pattern, EntityPattern):
        concept_constraint = pattern.constraints.get("concept")
        if isinstance(concept_constraint, str):
            for eid, ent in ctx.entities_of_concept(concept_constraint):
                if _entity_matches(ent, pattern.constraints, ctx, bindings):
                    yield eid
            return
        type_constraint = pattern.constraints.get("type")
        if isinstance(type_constraint, str):
            for eid, ent in ctx.entities_of_type(type_constraint):
                if _entity_matches(ent, pattern.constraints, ctx, bindings):
                    yield eid
            return
        for eid, ent in ctx.trace.entities.items():
            if _entity_matches(ent, pattern.constraints, ctx, bindings):
                yield eid
    elif isinstance(pattern, OrPattern):
        seen: set[str] = set()
        for branch in (pattern.left, pattern.right):
            for eid in _iter_entity_candidates(branch, ctx, bindings):
                if eid not in seen:
                    seen.add(eid)
                    yield eid


def _value_or_search(pattern: Pattern, value, ctx, bindings):
    """Apply pattern as value-filter on value; fall back to search if
    the pattern doesn't have a sensible value-filter semantics."""
    try:
        yield from pattern.apply_to_value(value, ctx, bindings)
    except NotImplementedError:
        yield from pattern.search(ctx, bindings)


@dataclass(eq=False)
class OrPattern(Pattern):
    left: Pattern
    right: Pattern

    def variables(self) -> set[Var]:
        return self.left.variables() | self.right.variables()

    def apply_to_value(self, value, ctx, bindings):
        yield from self.left.apply_to_value(value, ctx, bindings)
        yield from self.right.apply_to_value(value, ctx, bindings)

    def search(self, ctx, bindings):
        yield from self.left.search(ctx, bindings)
        yield from self.right.search(ctx, bindings)


@dataclass(eq=False)
class NotPattern(Pattern):
    inner: Pattern

    def variables(self) -> set[Var]:
        return self.inner.variables()

    def apply_to_value(self, value, ctx, bindings):
        # Closed-world: succeed iff inner has no match.
        for _ in self.inner.apply_to_value(value, ctx, bindings):
            return
        yield bindings

    def search(self, ctx, bindings):
        for _ in self.inner.search(ctx, bindings):
            return
        yield bindings


# ---------------------------- static evaluation -------------------
# A trace-free evaluator + JSON compiler used by relation arg patterns
# (Relation.arg_patterns) and the corresponding static introspection
# index (introspect.relation_arg_excludes). Lets `validate_relation`
# and the forward planner share one source of truth for schema-level
# entity gates without needing a full DSL context.


def entity_matches_static(entity, pattern, lex) -> bool:
    """Evaluate `pattern` against an EntityInstance using only its
    static (asserted) properties + lexicon types. Supports the
    pattern shapes meaningful as a relation arg gate: EntityPattern,
    NotPattern, AndPattern, OrPattern.

    No trace, no derived state, no var bindings — invariants must hold
    on the entity's own static attributes, not on transient derived
    facts. (A varies-slot value is technically derived per-instance,
    but it lives in entity.properties after randomization, so this
    still sees it.)"""
    if isinstance(pattern, EntityPattern):
        return _entity_static_matches(entity, pattern.constraints, lex)
    if isinstance(pattern, NotPattern):
        return not entity_matches_static(entity, pattern.inner, lex)
    if isinstance(pattern, AndPattern):
        return (entity_matches_static(entity, pattern.left, lex)
                and entity_matches_static(entity, pattern.right, lex))
    if isinstance(pattern, OrPattern):
        return (entity_matches_static(entity, pattern.left, lex)
                or entity_matches_static(entity, pattern.right, lex))
    raise TypeError(
        f"entity_matches_static: unsupported pattern "
        f"{type(pattern).__name__}")


def _entity_static_matches(entity, constraints: dict, lex) -> bool:
    from ..containment import _concept_in_category
    for key, expected in constraints.items():
        if key == "type":
            if not lex.types.is_subtype(entity.entity_type, expected):
                return False
        elif key == "has_suffix":
            if not entity.concept_lemma.endswith(expected):
                return False
        elif key == "concept":
            if entity.concept_lemma != expected:
                return False
        elif key == "category":
            concept = lex.concepts.get(entity.concept_lemma)
            if concept is None:
                return False
            if not _concept_in_category(concept, expected, lex):
                return False
        else:
            vals = entity.properties.get(key, [])
            if expected is Ellipsis:
                if not vals:
                    return False
            elif not _value_matches(vals, expected):
                return False
    return True


def compile_arg_pattern(d):
    """Parse a JSON-encoded arg pattern into a Pattern object. Supported
    shapes:
      None                              → None (no constraint)
      {"entity": {slot: val, ...}}      → EntityPattern(constraints)
      {"not": <pattern>}                → NotPattern(inner)
      {"and": [<pat>, <pat>]}           → AndPattern(left, right)
      {"or":  [<pat>, <pat>]}           → OrPattern(left, right)
    Used by the loader to compile `Relation.arg_patterns` entries from
    relations.jsonl."""
    if d is None:
        return None
    if not isinstance(d, dict) or len(d) != 1:
        raise ValueError(
            f"arg pattern must be a single-key dict, got {d!r}")
    key, body = next(iter(d.items()))
    if key == "entity":
        if not isinstance(body, dict):
            raise ValueError(f"entity pattern body must be dict: {body!r}")
        return EntityPattern(dict(body))
    if key == "not":
        return NotPattern(compile_arg_pattern(body))
    if key == "and":
        if not isinstance(body, list) or len(body) != 2:
            raise ValueError(f"and pattern needs 2-element list: {body!r}")
        return AndPattern(compile_arg_pattern(body[0]),
                          compile_arg_pattern(body[1]))
    if key == "or":
        if not isinstance(body, list) or len(body) != 2:
            raise ValueError(f"or pattern needs 2-element list: {body!r}")
        return OrPattern(compile_arg_pattern(body[0]),
                         compile_arg_pattern(body[1]))
    raise ValueError(f"unknown arg pattern key: {key!r}")


def numeric_args_compare(entities, compare_spec: dict) -> bool:
    """Evaluate a Relation.arg_compare spec against the (entity_a,
    entity_b, …) tuple at the relation's arg positions. Returns True
    when the comparison holds OR either property is missing (vacuous).

    spec keys: left_arg, left_property, op, right_arg, right_property.
    Numeric values parsed via float(); non-numeric strings → vacuous
    success. Used both by `Trace.validate_relation` and the forward
    planner's grounding filter — one evaluator, two consumers."""
    try:
        left_ent = entities[compare_spec["left_arg"]]
        right_ent = entities[compare_spec["right_arg"]]
    except (IndexError, KeyError):
        return True
    lvals = left_ent.properties.get(compare_spec["left_property"], [])
    rvals = right_ent.properties.get(compare_spec["right_property"], [])
    if not lvals or not rvals:
        return True
    try:
        lhs = float(lvals[0])
        rhs = float(rvals[0])
    except (TypeError, ValueError):
        return True
    op = compare_spec["op"]
    if op == "<":  return lhs < rhs
    if op == "<=": return lhs <= rhs
    if op == ">":  return lhs > rhs
    if op == ">=": return lhs >= rhs
    if op == "==": return lhs == rhs
    return True
