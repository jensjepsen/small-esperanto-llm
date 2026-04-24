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
    """A match variable. Identity-compared; name is for error messages."""
    __slots__ = ("name",)

    def __init__(self, name: str = "_anon"):
        self.name = name

    def __repr__(self) -> str:
        return f"${self.name}"

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other) -> bool:
        return self is other


def var(name: str = "_anon") -> Var:
    """Construct a fresh match variable. Typical use: `bind(T := var("T"))`."""
    return Var(name)


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
    """Auto-wrap a bare Var as a BindPattern so `T & entity(...)` works."""
    if isinstance(x, Pattern):
        return x
    if isinstance(x, Var):
        return BindPattern(x)
    raise TypeError(f"expected Pattern or Var, got {type(x).__name__}: {x!r}")


# ---------------------------- bind --------------------------------

@dataclass(eq=False)
class BindPattern(Pattern):
    target: Var

    def variables(self) -> set[Var]:
        return {self.target}

    def apply_to_value(self, value, ctx, bindings):
        if self.target in bindings:
            if bindings[self.target] == value:
                yield bindings
            return
        yield {**bindings, self.target: value}

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


# ---------------------------- entity ------------------------------

@dataclass(eq=False)
class EntityPattern(Pattern):
    """Matches an entity given property/type constraints.

    Constraint keys:
      `type` — required entity_type (subtype-checked via Lexicon).
      `has_suffix` — lemma must end with this suffix.
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
        if _entity_matches(ent, self.constraints, ctx):
            yield bindings

    def search(self, ctx, bindings):
        for eid, ent in ctx.trace.entities.items():
            if _entity_matches(ent, self.constraints, ctx):
                yield bindings


def _entity_matches(ent, constraints: dict[str, Any], ctx) -> bool:
    for key, expected in constraints.items():
        if key == "type":
            if not ctx.lexicon.types.is_subtype(ent.entity_type, expected):
                return False
        elif key == "has_suffix":
            if not ent.concept_lemma.endswith(expected):
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


def entity(**constraints) -> EntityPattern:
    """Match an entity by type and/or properties. Keys: `type`,
    `has_suffix`, or any slot name.

        entity(type="artifact", fragility="fragile")
    """
    return EntityPattern(dict(constraints))


# ---------------------------- rel ---------------------------------

@dataclass(eq=False)
class RelPattern(Pattern):
    """Match a relation instance by name and arg patterns.

    `arg_patterns` is keyed by positional slot name (e.g. "container",
    "contained" for "en"; "agent", "theme" for "uzi"-roles). Resolved
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
        # Relation arg names come from the declared Relation schema.
        rel_def = ctx.lexicon.relations.get(self.relation)
        if rel_def is None:
            return
        arg_names = _relation_arg_names(rel_def, self.relation)
        for r in ctx.trace.relations:
            if r.relation != self.relation:
                continue
            if len(r.args) != len(arg_names):
                continue
            # Apply each arg pattern in turn.
            yield from _apply_args(
                self.arg_patterns, arg_names, r.args, ctx, bindings)

    def apply_to_value(self, value, ctx, bindings):
        # Rel is a relational existence check — no inherent "value"
        # position. Treat as search with whatever bindings are in scope.
        yield from self.search(ctx, bindings)


def _relation_arg_names(rel_def, name: str) -> list[str]:
    """Positional arg names for a relation. Uses the Relation schema's
    `arg_types`-derived role names when available, with canonical
    fallbacks for the built-in relations."""
    # Canonical fallbacks — the schema stores arg_types but not arg
    # names, so we hard-code the standard names.
    canonical = {
        "en":   ["contained", "container"],
        "sur":  ["contained", "container"],
        "havi": ["owner", "theme"],
    }
    return canonical.get(name, [f"arg{i}" for i in range(len(rel_def.arg_types))])


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

        event("fali", theme=entity(fragility="fragile") & bind(T))
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
        while frontier and (self.max_steps is None or step < self.max_steps):
            next_frontier: list[str] = []
            for node in frontier:
                for r in ctx.trace.relations:
                    if r.relation not in self.relations or len(r.args) != 2:
                        continue
                    if r.args[0] == node and r.args[1] not in visited:
                        next_frontier.append(r.args[1])
                    elif r.args[1] == node and r.args[0] not in visited:
                        next_frontier.append(r.args[0])
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
            for eid in _iter_entity_candidates(from_entity, ctx):
                yield from _value_or_search(self.right, eid, ctx, bindings)
            return
        from_entity = _find_entity_source(self.right)
        if from_entity is not None:
            for eid in _iter_entity_candidates(from_entity, ctx):
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


def _iter_entity_candidates(pattern: Pattern, ctx):
    """Enumerate distinct entity_ids matching an EntityPattern or an
    Or-union of them. Dedupes across branches so an entity matching
    multiple branches yields once."""
    if isinstance(pattern, EntityPattern):
        for eid, ent in ctx.trace.entities.items():
            if _entity_matches(ent, pattern.constraints, ctx):
                yield eid
    elif isinstance(pattern, OrPattern):
        seen: set[str] = set()
        for branch in (pattern.left, pattern.right):
            for eid in _iter_entity_candidates(branch, ctx):
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
