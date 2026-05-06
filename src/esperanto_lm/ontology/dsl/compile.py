"""Per-rule pattern matcher codegen.

`compile_rule(rule)` walks `rule.when` and `rule.given`, generates a
Python generator function tailored to that rule, and `exec`s it in a
namespace prepared with the Var objects + runtime helpers. The result
is a callable `enum(event, ctx) -> Iterator[bindings]` that yields each
satisfying binding dict — same protocol as
`enumerate_bindings(when, given, ctx)` but ~5× faster on the hot loop
because pattern dispatch and slot lookups are inlined.

What's inlined:
  - `EventPattern` action filter and role binding.
  - `EntityPattern & BindPattern` (the dominant given shape) — direct
    `entities_of_type` / `entities_of_concept` iteration + slot checks.
  - Bare `BindPattern` on event roles.
  - Simple `RelPattern` lookups via `ctx.relations_of(name)`.

What falls back to the interpreter (`_run_pattern`):
  - `OrPattern`, `NotPattern`, `ClosurePattern`, `CausedByPattern`,
    `PastEventPattern`, `HasConceptFieldPattern`, anything nested in
    ways the inliner doesn't recognize.

Effect application (rule.then) is NOT compiled — `_apply_effects` from
the engine handles it, kept untouched to bound the codegen surface.

Set `ESPLLM_DSL_COMPILE=0` in the environment to disable codegen and
fall back to the interpreted path for parity testing.
"""
from __future__ import annotations

from typing import Callable, Iterator, Optional

from .effects import Effect
from .patterns import (
    AndPattern, BindPattern, EntityPattern, EventPattern, NotPattern,
    OrPattern, Pattern, RelPattern, Var, _iter_entity_candidates,
)


class _Unsupported(Exception):
    """Raised when the compiler encounters a pattern shape it doesn't
    inline. Caller wraps the offending clause in a runtime fallback
    instead of failing the whole rule."""


def _run_pattern(pattern: Pattern, ctx, bindings: dict) -> Iterator[dict]:
    """Runtime fallback — invoke the interpreter for one pattern.
    Yields binding dicts. Used for clauses the codegen doesn't inline.
    Mutate-and-restore semantics inside the iteration are honored by
    the interpreter; callers must `dict(b)` to keep across iterations."""
    yield from pattern.search(ctx, bindings)


# -------------------- compiler state ------------------------------

class _Compiler:
    def __init__(self, rule):
        self.rule = rule
        # Source lines (indent stripped — applied at emit time).
        self._lines: list[str] = []
        # Prelude lines (always at indent=1, run once per enum call).
        # Used for loop-invariant lookups: lexicon.relations[name],
        # arg_names.index(role), etc. Inlined below the function
        # signature in `compile()` / `compile_derivation()`.
        self._prelude: list[str] = []
        # rel_name -> (argnames_local, {arg_name: idx_local})
        # Memoize so each rel(name) referenced multiple times reuses
        # the same prelude lookup.
        self._rel_argnames: dict[str, tuple[str, dict[str, str]]] = {}
        self._indent = 1   # body of `def enum(event, ctx):`
        # Var -> local Python identifier holding the resolved entity_id.
        self.var_to_local: dict[Var, str] = {}
        # Var -> python identifier in the closure namespace (so the Var
        # object itself can be referenced when constructing bindings).
        self.var_to_obj: dict[Var, str] = {}
        # Names exposed in the namespace at exec time.
        self.namespace: dict[str, object] = {
            "_run_pattern": _run_pattern,
            "_iter_entity_candidates": _iter_entity_candidates,
        }
        self._next_id = 0

    # ---- emit helpers ----

    def emit(self, line: str = "") -> None:
        if line:
            self._lines.append("    " * self._indent + line)
        else:
            self._lines.append("")

    def emit_prelude(self, line: str) -> None:
        """Emit a line at function-prelude scope (indent=1, before any
        clause body). Use for loop-invariant work that should run once
        per enum() call rather than per inner-loop iteration."""
        self._prelude.append("    " + line)

    def _ensure_rel_argnames(self, rel_name: str) -> tuple[str, str]:
        """Return (argnames_local, abort_check_local) for the given rel.
        Lazily emits a prelude lookup the first time the rel is referenced;
        subsequent rel clauses with the same name reuse the same locals.
        Also emits a symmetric-flag local (`_sym_<rel>`) so generated
        loops can yield both arg orderings for symmetric relations."""
        cached = self._rel_argnames.get(rel_name)
        if cached is not None:
            return cached[0], cached[1].setdefault(
                "__abort__", cached[0])
        argnames_local = self.fresh("argnames")
        sym_local = self.fresh(f"sym_{rel_name}")
        self.emit_prelude(
            f"_rel_def = ctx.lexicon.relations.get({rel_name!r})")
        self.emit_prelude(
            f"{argnames_local} = (_rel_def.arg_names "
            f"if _rel_def is not None else None)")
        self.emit_prelude(
            f"{sym_local} = (_rel_def.symmetric "
            f"if _rel_def is not None else False)")
        # Stash sym_local under a sentinel key so _compile_rel_clause
        # can retrieve it without changing the public return shape.
        self._rel_argnames[rel_name] = (argnames_local, {"__sym__": sym_local})
        return argnames_local, argnames_local

    def _sym_local_for(self, rel_name: str) -> str:
        """Symmetric-flag local emitted by _ensure_rel_argnames."""
        return self._rel_argnames[rel_name][1]["__sym__"]

    def _arg_index_local(self, rel_name: str, arg_name: str) -> str:
        """Return a prelude-resolved local for `argnames.index(arg_name)`.
        Memoized per (rel_name, arg_name)."""
        argnames_local, idx_map = self._rel_argnames[rel_name]
        if arg_name in idx_map:
            return idx_map[arg_name]
        idx_local = self.fresh(f"idx_{rel_name}_{arg_name}")
        self.emit_prelude(
            f"{idx_local} = ({argnames_local}.index({arg_name!r}) "
            f"if {argnames_local} is not None else -1)")
        idx_map[arg_name] = idx_local
        return idx_local

    def _push(self) -> None:
        self._indent += 1

    def _pop(self) -> None:
        self._indent -= 1

    def fresh(self, prefix: str) -> str:
        n = self._next_id
        self._next_id += 1
        return f"_{prefix}_{n}"

    def var_obj_name(self, v: Var) -> str:
        if v in self.var_to_obj:
            return self.var_to_obj[v]
        name = f"_VAR_{v.name}_{len(self.var_to_obj)}"
        self.var_to_obj[v] = name
        self.namespace[name] = v
        return name

    def expose(self, name: str, value) -> str:
        """Stash a runtime object in the closure namespace and return
        the python identifier that references it. Used for pattern
        objects passed to the runtime fallback."""
        ident = self.fresh(name)
        self.namespace[ident] = value
        return ident

    # ---- top-level ----

    def compile(self) -> str:
        rule = self.rule
        when = rule.when
        if not isinstance(when, EventPattern):
            raise _Unsupported(f"rule {rule.name}: when is not EventPattern")

        # Function signature.
        self._lines.append(f"def enum(event, ctx):")
        # Action filter.
        self.emit(f"if event.action != {when.action!r}:")
        self._push(); self.emit("return"); self._pop()
        # Bind context locals once.
        self.emit("trace = ctx.trace")

        # Compile role patterns (sorted for determinism).
        for role_name, patt in when.role_patterns.items():
            self._compile_role_pattern(role_name, patt)

        # Compile given clauses in order; each adds a layer of nesting.
        for clause in rule.given:
            self._compile_given_clause(clause)

        # Emit the yield. Build the bindings dict literal from
        # var_to_local at this point.
        items = ", ".join(
            f"{self.var_obj_name(v)}: {local}"
            for v, local in self.var_to_local.items()
        )
        self.emit(f"yield {{{items}}}")
        return self._assemble_source()

    def compile_derivation(self) -> str:
        """Same as compile() but for Derivation: no event arg, the
        `when` clause is treated as the first state-clause in a chain
        (just like a `given` clause). Generates `def enum(ctx):`.

        Three optimization passes, in order:

        1. **Self-join indexing** — when `deriv.when` and the only
           `deriv.given` clause are both `rel(R, k=bind(A), j=bind(L))`
           with one shared join key (L), bucket by L once and emit
           pairs from each bucket. Cuts O(N²) iteration to
           O(N + sum(bucket²)) — much smaller when entities are
           distributed across containers. Big win for
           shared_container_means_samloke, shared_apud_means_samloke.

        2. **Constrained-when reorder** — if `deriv.when` is
           `entity(<constraints>) & bind(V)` AND some `given` clause
           already binds V, skip the outer entity scan and apply the
           entity constraints (if any) as a direct-lookup check on V
           after V gets bound by the given clause. Cuts the O(N_ent ×
           N_facts) outer-product to O(N_facts) for many derivations
           (host_openness_*, samloke_propagates_*, entity_in_water_*,
           agent_illuminated, etc.).

        3. Generic clause-chain compilation."""
        deriv = self.rule  # we reuse `self.rule` to hold a Derivation
        self._lines.append(f"def enum(ctx):")
        self.emit("trace = ctx.trace")

        if self._try_compile_self_join(deriv):
            # Self-join took over and emitted nested loops with
            # var_to_local populated. Fall through to the final yield.
            pass
        else:
            clauses = list((deriv.when, *deriv.given))
            deferred = self._extract_deferred_when_constraints(
                deriv.when, deriv.given)
            if deferred is not None:
                deferred_var, deferred_constraints = deferred
                clauses = list(deriv.given)
            for clause in clauses:
                self._compile_given_clause(clause)
            # If we deferred constraints, apply them now that V is bound.
            if deferred is not None and deferred_constraints:
                v_local = self.var_to_local.get(deferred_var)
                if v_local is not None:
                    self._emit_deferred_entity_constraints(
                        deferred_var, deferred_constraints)

        items = ", ".join(
            f"{self.var_obj_name(v)}: {local}"
            for v, local in self.var_to_local.items()
        )
        self.emit(f"yield {{{items}}}")
        return self._assemble_source()

    @staticmethod
    def _extract_deferred_when_constraints(when, given):
        """If `when` is `entity(<constraints>) & bind(V)` AND some
        clause in `given` POSITIVELY binds V, return (V, constraints).
        Caller skips the outer scan and applies the constraints later
        as a direct-lookup filter on V (now bound by the given clause).
        Returns None if not swap-eligible.

        Negation safety: skip the optimization if any given clause is
        (or contains) a `NotPattern`. NotPattern's correctness depends
        on its referenced vars being bound before evaluation — with V
        unbound, `~rel(..., container=V)` checks "no such relation
        anywhere" instead of "no such relation for this specific V",
        which is strictly stronger and yields no matches. Conservative:
        bail on the whole rule rather than try to detect which clauses
        are safe."""
        if not isinstance(when, AndPattern):
            return None
        ent_pat = bind_pat = None
        if (isinstance(when.left, EntityPattern)
                and isinstance(when.right, BindPattern)):
            ent_pat, bind_pat = when.left, when.right
        elif (isinstance(when.right, EntityPattern)
              and isinstance(when.left, BindPattern)):
            ent_pat, bind_pat = when.right, when.left
        if ent_pat is None:
            return None
        target = bind_pat.target
        # Bail if any given clause uses negation — skipping when would
        # leave NotPattern checking too broadly.
        if any(_Compiler._contains_not_pattern(c) for c in given):
            return None
        for clause in given:
            if target in clause.variables():
                return (target, ent_pat.constraints)
        return None

    @staticmethod
    def _contains_not_pattern(pattern: Pattern) -> bool:
        """True if `pattern` is or contains a NotPattern. Recurses
        through And/Or composition so nested negations are caught too."""
        if isinstance(pattern, NotPattern):
            return True
        if isinstance(pattern, (AndPattern, OrPattern)):
            return (_Compiler._contains_not_pattern(pattern.left)
                    or _Compiler._contains_not_pattern(pattern.right))
        return False

    def _emit_deferred_entity_constraints(self, target_var, constraints):
        """Emit entity-constraint checks on a Var that's already bound
        from a given clause. Uses the same single-iteration loop trick
        as `_compile_entity_direct_lookup` so subsequent code (the
        final yield) nests inside a `continue`-skippable scope."""
        if not constraints:
            return
        eid_local = self.var_to_local[target_var]
        ent_local = self.fresh("dlu_ent")
        self.emit(f"{ent_local} = trace.entities.get({eid_local})")
        sentinel = self.fresh("dlu_skip")
        self.emit(f"for {sentinel} in ((None,) if {ent_local} is not None "
                  f"else ()):")
        self._push()
        for key, expected in constraints.items():
            self._emit_loop_constraint_check(
                ent_local, eid_local, key, expected)

    def _try_compile_self_join(self, deriv) -> bool:
        """Detect and compile the self-join derivation shape:

            when=rel(R, k=bind(L), x=bind(A))
            given=[rel(R, k=L, x=bind(B))]      # same R, shared L

        where L is the join key (shared) and A/B are the outer keys
        (distinct vars bound from the two arg positions). Generates
        bucketed code: index `relations_of(R)` by the join key in one
        pass, then emit pair combinations per bucket.

        Returns True if it took over codegen; False to let the caller
        use the general path. Conservative: bails on any deviation
        (extra given clauses, non-bind arg patterns, more than one
        join key or outer key)."""
        when = deriv.when
        given = deriv.given
        if not isinstance(when, RelPattern):
            return False
        if len(given) != 1:
            return False
        g = given[0]
        if not isinstance(g, RelPattern):
            return False
        if g.relation != when.relation:
            return False

        # Extract bind targets from each arg pattern. Only BindPatterns
        # qualify — anything else (entity check, complex pattern) means
        # we can't safely bucket without losing constraints.
        def bind_target(p):
            return p.target if isinstance(p, BindPattern) else None

        when_args = {name: bind_target(p)
                     for name, p in when.arg_patterns.items()}
        given_args = {name: bind_target(p)
                      for name, p in g.arg_patterns.items()}
        if any(v is None for v in when_args.values()):
            return False
        if any(v is None for v in given_args.values()):
            return False

        # Categorize arg names: shared-Var = join key, different-Var = outer.
        join_keys: list[str] = []
        outer_pairs: list[tuple[str, Var, Var]] = []
        common_names = set(when_args) & set(given_args)
        if common_names != set(when_args) or common_names != set(given_args):
            # One side has args the other doesn't — non-rectangular shape.
            return False
        for name in common_names:
            if when_args[name] is given_args[name]:
                join_keys.append(name)
            else:
                outer_pairs.append(
                    (name, when_args[name], given_args[name]))
        if len(join_keys) != 1 or len(outer_pairs) != 1:
            return False

        rel_name = when.relation
        join_arg = join_keys[0]
        outer_arg, when_outer_var, given_outer_var = outer_pairs[0]
        join_var = when_args[join_arg]

        # Emit prelude lookups (cached per-rel via the existing helper).
        argnames_local, _ = self._ensure_rel_argnames(rel_name)
        sym_local = self._sym_local_for(rel_name)
        join_idx = self._arg_index_local(rel_name, join_arg)
        outer_idx = self._arg_index_local(rel_name, outer_arg)

        # Bucket pass: bucket[join_value] -> [outer_value, ...]. For
        # symmetric relations (apud, samloke), each asserted tuple
        # also contributes its swap — without this, the join key on
        # one side gets bucketed but not the other, missing half the
        # closure (e.g. shared_apud_means_samloke would yield
        # samloke(kuirejo, kuirejo) but not samloke(vendejo, vendejo)
        # for asserted apud(kuirejo, vendejo)).
        bucket = self.fresh("bucket")
        self.emit(f"{bucket} = {{}}")
        outer_args = self.fresh("rawargs")
        self.emit(f"for {outer_args} in ctx.relations_of({rel_name!r}):")
        self._push()
        self.emit(f"if {argnames_local} is None or "
                  f"len({outer_args}) != len({argnames_local}):")
        self._push(); self.emit("continue"); self._pop()
        args_l = self.fresh("args")
        self.emit(f"for {args_l} in (({outer_args}, ({outer_args}[1], "
                  f"{outer_args}[0])) if {sym_local} and "
                  f"len({outer_args}) == 2 and {outer_args}[0] != "
                  f"{outer_args}[1] else ({outer_args},)):")
        self._push()
        join_v = self.fresh("jv")
        outer_v = self.fresh("ov")
        self.emit(f"{join_v} = {args_l}[{join_idx}]")
        self.emit(f"{outer_v} = {args_l}[{outer_idx}]")
        self.emit(f"{bucket}.setdefault({join_v}, []).append({outer_v})")
        self._pop()
        self._pop()

        # Pair-emission pass: per bucket, all (a, b) cross-product yields.
        join_key_l = self.fresh("jkey")
        members_l = self.fresh("members")
        self.emit(f"for {join_key_l}, {members_l} in {bucket}.items():")
        self._push()
        a_l = self.fresh("a")
        self.emit(f"for {a_l} in {members_l}:")
        self._push()
        b_l = self.fresh("b")
        self.emit(f"for {b_l} in {members_l}:")
        self._push()

        # Bind the rule's vars to locals so the final yield can build
        # the bindings dict.
        self.var_to_local[join_var] = join_key_l
        self.var_to_local[when_outer_var] = a_l
        self.var_to_local[given_outer_var] = b_l
        return True

    def _assemble_source(self) -> str:
        """Splice the prelude (loop-invariant lookups) between the
        function signature and the body."""
        if not self._prelude:
            return "\n".join(self._lines) + "\n"
        # _lines[0] is the `def enum(...)` line; everything after is
        # the indented body that the prelude must precede.
        out = [self._lines[0]] + self._prelude + self._lines[1:]
        return "\n".join(out) + "\n"

    @staticmethod
    def _unconstrained_when_var_bound_in_given(when, given) -> bool:
        """True iff `when` is `entity() & bind(V)` (no constraints) AND
        some clause in `given` binds the same V — meaning the entity
        scan is wasted work (V will be bound by the relation/clause
        anyway, and the empty entity() check is implied by relation
        arg-type contracts). Conservative: only fires on the empty-
        entity smell, leaves constrained whens alone."""
        if not isinstance(when, AndPattern):
            return False
        # Find the EntityPattern + BindPattern within.
        ent_pat = bind_pat = None
        if (isinstance(when.left, EntityPattern)
                and isinstance(when.right, BindPattern)):
            ent_pat, bind_pat = when.left, when.right
        elif (isinstance(when.right, EntityPattern)
              and isinstance(when.left, BindPattern)):
            ent_pat, bind_pat = when.right, when.left
        if ent_pat is None or ent_pat.constraints:
            return False    # constrained — don't skip
        target = bind_pat.target
        for clause in given:
            if target in clause.variables():
                return True
        return False

    # ---- role patterns (event roles) ----

    def _compile_role_pattern(self, role_name: str, patt: Pattern) -> None:
        """Each role pattern resolves the event's role to an entity_id
        and (optionally) constrains it via entity(...) / binds via
        bind(V) / both via entity(...) & bind(V)."""
        eid_local = self.fresh(f"role_{role_name}")
        self.emit(f"{eid_local} = event.roles.get({role_name!r})")
        self.emit(f"if {eid_local} is None:")
        self._push(); self.emit("return"); self._pop()

        # Apply value-pattern: bind / entity / and-of-both.
        self._apply_value_pattern(patt, eid_local)

    def _apply_value_pattern(self, patt: Pattern, eid_local: str) -> None:
        """Apply a pattern as a value-filter on a known entity_id local.
        Handles bind / entity / (entity & bind) / (bind & entity).
        Falls back to runtime for the rest."""
        if isinstance(patt, BindPattern):
            self._do_bind(patt.target, eid_local)
            return
        if isinstance(patt, EntityPattern):
            self._do_entity_check(patt.constraints, eid_local)
            return
        if isinstance(patt, AndPattern):
            self._apply_value_pattern(patt.left, eid_local)
            self._apply_value_pattern(patt.right, eid_local)
            return
        # Unknown shape — fall back to runtime as a value-filter.
        self._fallback_value(patt, eid_local)

    def _do_bind(self, var: Var, eid_local: str) -> None:
        if var in self.var_to_local:
            existing = self.var_to_local[var]
            # Skip the equality check when both sides are literally the
            # same local — `if X != X` is a tautology and millions of
            # those add up. Happens after direct-lookup picks the same
            # local that the bind targets.
            if existing == eid_local:
                return
            self.emit(f"if {existing} != {eid_local}:")
            self._push(); self.emit("return"); self._pop()
        else:
            self.var_to_local[var] = eid_local

    def _do_entity_check(self, constraints: dict, eid_local: str) -> None:
        ent_local = self.fresh("ent")
        self.emit(f"{ent_local} = trace.entities.get({eid_local})")
        self.emit(f"if {ent_local} is None:")
        self._push(); self.emit("return"); self._pop()
        for key, expected in constraints.items():
            self._emit_constraint_check(ent_local, eid_local, key, expected)

    def _emit_constraint_check(
        self, ent_local: str, eid_local: str, key: str, expected,
    ) -> None:
        """Emit lines that `return` if the entity fails this constraint.
        Mirrors `_entity_matches` from patterns.py."""
        if isinstance(expected, Var):
            # Var-valued constraint resolves from current bindings.
            if expected not in self.var_to_local:
                # Unbound — fail (matches _entity_matches behavior).
                self.emit(f"return  # unbound var constraint")
                return
            resolved_local = self.var_to_local[expected]
            self._emit_value_constraint(ent_local, eid_local, key,
                                        resolved_local, is_local=True)
            return
        self._emit_value_constraint(
            ent_local, eid_local, key, repr(expected), is_local=False)

    def _emit_value_constraint(
        self, ent_local: str, eid_local: str, key: str,
        value_repr: str, *, is_local: bool,
    ) -> None:
        if key == "type":
            # Subtype-aware via lex.types.is_subtype.
            self.emit(
                f"if not ctx.lexicon.types.is_subtype("
                f"{ent_local}.entity_type, {value_repr}):")
            self._push(); self.emit("return"); self._pop()
        elif key == "concept":
            self.emit(f"if {ent_local}.concept_lemma != {value_repr}:")
            self._push(); self.emit("return"); self._pop()
        elif key == "has_suffix":
            self.emit(f"if not {ent_local}.concept_lemma.endswith("
                      f"{value_repr}):")
            self._push(); self.emit("return"); self._pop()
        else:
            actual = self.fresh("v")
            self.emit(f"{actual} = ctx.effective_property("
                      f"{eid_local}, {key!r})")
            # Wildcard: expected is Ellipsis (literal `...`).
            if value_repr == "Ellipsis":
                self.emit(f"if {actual} is None or {actual} == [] "
                          f"or {actual} == '':")
                self._push(); self.emit("return"); self._pop()
                return
            # Scalar-or-list match (mirrors _value_matches).
            self.emit(f"if {actual} is None:")
            self._push(); self.emit("return"); self._pop()
            self.emit(f"if isinstance({actual}, list):")
            self._push()
            self.emit(f"if {value_repr} not in {actual}:")
            self._push(); self.emit("return"); self._pop()
            self._pop()
            self.emit("else:")
            self._push()
            self.emit(f"if {actual} != {value_repr}:")
            self._push(); self.emit("return"); self._pop()
            self._pop()

    def _fallback_value(self, patt: Pattern, eid_local: str) -> None:
        """Runtime fallback for value-position patterns we don't inline.
        Apply the pattern as a value-filter on `eid_local` via the
        interpreter."""
        patt_ident = self.expose("patt", patt)
        # Build a partial bindings dict from what's bound so far.
        items = ", ".join(
            f"{self.var_obj_name(v)}: {local}"
            for v, local in self.var_to_local.items()
        )
        bdict = self.fresh("b")
        self.emit(f"{bdict} = {{{items}}}")
        flag = self.fresh("ok")
        self.emit(f"{flag} = False")
        self.emit(f"for _ in {patt_ident}.apply_to_value("
                  f"{eid_local}, ctx, {bdict}):")
        self._push()
        self.emit(f"{flag} = True; break")
        self._pop()
        self.emit(f"if not {flag}:")
        self._push(); self.emit("return"); self._pop()

    # ---- given clauses ----

    def _compile_given_clause(self, clause: Pattern) -> None:
        """Each clause adds nesting: entity scans become `for` loops,
        relation lookups become `for args in ctx.relations_of(...)`,
        unsupported clauses become a runtime-driven `for b in
        _run_pattern(...)` that re-enters with extended bindings.

        Optimization: when an `entity(...) & bind(V)` clause has V
        already bound from a prior clause, skip the entity scan
        entirely and do a direct `trace.entities.get(V)` lookup. Common
        in propagation derivations (samloke_propagates_through_*,
        host_openness_*) where the entity to constrain is already
        named by an upstream rel(...) match."""
        if isinstance(clause, AndPattern):
            # Pattern: <entity-source> & <filter> in either order.
            for entity_pos, filter_pos in (
                    (clause.left, clause.right),
                    (clause.right, clause.left)):
                if isinstance(entity_pos, EntityPattern):
                    pre_bound = self._already_bound_local(filter_pos)
                    if pre_bound is not None:
                        self._compile_entity_direct_lookup(
                            entity_pos, filter_pos, pre_bound)
                    else:
                        self._compile_entity_scan_and_filter(
                            entity_pos, filter_pos)
                    return
                if (isinstance(entity_pos, OrPattern)
                        and self._or_is_entity_union(entity_pos)):
                    self._compile_or_entity_scan_and_filter(
                        entity_pos, filter_pos)
                    return
            # AndPattern with entity source nested deeper — fall back.
            self._fallback_clause(clause)
            return
        if isinstance(clause, EntityPattern):
            self._compile_entity_scan_and_filter(clause, None)
            return
        if isinstance(clause, RelPattern):
            self._compile_rel_clause(clause)
            return
        # Unknown — runtime fallback.
        self._fallback_clause(clause)

    @staticmethod
    def _or_is_entity_union(pattern: OrPattern) -> bool:
        """True iff every leaf of an OrPattern tree is an EntityPattern.
        Lets us drive iteration via _iter_entity_candidates which
        already dedupes across branches."""
        def walk(p):
            if isinstance(p, EntityPattern):
                return True
            if isinstance(p, OrPattern):
                return walk(p.left) and walk(p.right)
            return False
        return walk(pattern)

    def _already_bound_local(self, pattern: Pattern):
        """Walk a filter-position pattern and return the python local
        name for its bind target IF that target is already bound from
        a prior clause. Otherwise None.

        Recognizes `bind(V)` directly and inside `AndPattern`. Returns
        the local string so the caller can emit a direct lookup
        instead of scanning."""
        if isinstance(pattern, BindPattern):
            return self.var_to_local.get(pattern.target)
        if isinstance(pattern, AndPattern):
            return (self._already_bound_local(pattern.left)
                    or self._already_bound_local(pattern.right))
        return None

    def _compile_entity_direct_lookup(
        self, entity_pat: EntityPattern, filter_pat: Pattern,
        eid_local: str,
    ) -> None:
        """Emit a direct `trace.entities.get(V)` lookup with the entity
        constraints inlined as `if ...: continue` filters. Wrapped in a
        single-iteration for-loop so subsequent clauses + the final
        yield can nest inside the same `continue`-skippable scope —
        works correctly whether we're at top function scope or inside
        an outer loop from a prior clause."""
        ent_local = self.fresh("dlu_ent")
        self.emit(f"{ent_local} = trace.entities.get({eid_local})")
        # 1-iteration scope: single tuple element when ent is found,
        # empty when not. `continue` inside the body skips remaining
        # constraints/yield without affecting any outer loop.
        sentinel = self.fresh("dlu_skip")
        self.emit(f"for {sentinel} in ((None,) if {ent_local} is not None "
                  f"else ()):")
        self._push()
        # Apply entity_pat's constraints inline (continue on failure).
        for key, expected in entity_pat.constraints.items():
            self._emit_loop_constraint_check(
                ent_local, eid_local, key, expected)
        # Apply the filter pattern in case it carries extra constraints
        # or vars to bind beyond the pre-bound one. The bind that's
        # already bound becomes a no-op equality check (which is
        # vacuously true since eid_local IS that var's local).
        self._apply_value_pattern_in_loop(filter_pat, eid_local)

    def _compile_or_entity_scan_and_filter(
        self, or_pat: OrPattern, filter_pat: Pattern,
    ) -> None:
        """Drive iteration via the runtime helper
        `_iter_entity_candidates` (which handles OR-union dedup), then
        apply filter_pat as a value-filter on each yielded eid.

        Cheaper than re-implementing OR-union dedup in codegen, and the
        helper already short-circuits on type/concept indexes per
        branch."""
        or_ident = self.expose("or_pat", or_pat)
        items = ", ".join(
            f"{self.var_obj_name(v)}: {local}"
            for v, local in self.var_to_local.items()
        )
        bdict = self.fresh("b")
        self.emit(f"{bdict} = {{{items}}}")
        eid_local = self.fresh("eid")
        self.emit(f"for {eid_local} in _iter_entity_candidates("
                  f"{or_ident}, ctx, {bdict}):")
        self._push()
        if filter_pat is not None:
            self._apply_value_pattern_in_loop(filter_pat, eid_local)

    def _compile_entity_scan_and_filter(
        self, entity_pat: EntityPattern, filter_pat: Optional[Pattern],
    ) -> None:
        """Generate `for eid, ent in <iter>:` over candidates matching
        entity_pat, then apply filter_pat (typically `bind(V)`) as a
        value-filter."""
        if not isinstance(entity_pat, EntityPattern):
            self._fallback_clause(entity_pat)
            return

        # Pick the most selective iteration: concept > type > all.
        constraints = entity_pat.constraints
        eid_local = self.fresh("eid")
        ent_local = self.fresh("ent")
        concept_c = constraints.get("concept")
        type_c = constraints.get("type")
        if isinstance(concept_c, str):
            self.emit(f"for {eid_local}, {ent_local} in "
                      f"ctx.entities_of_concept({concept_c!r}):")
        elif isinstance(type_c, str):
            self.emit(f"for {eid_local}, {ent_local} in "
                      f"ctx.entities_of_type({type_c!r}):")
        else:
            self.emit(f"for {eid_local}, {ent_local} in "
                      f"trace.entities.items():")
        self._push()

        # Inline remaining constraints as `if ...: continue` filters.
        skip_keys = set()
        if isinstance(concept_c, str):
            skip_keys.add("concept")
        elif isinstance(type_c, str):
            skip_keys.add("type")
        for key, expected in constraints.items():
            if key in skip_keys:
                continue
            self._emit_loop_constraint_check(
                ent_local, eid_local, key, expected)

        # Apply filter (usually bind(V)).
        if filter_pat is not None:
            self._apply_value_pattern_in_loop(filter_pat, eid_local)

        # Loop body intentionally left open — the next clause / final
        # yield emits inside this scope. Compiler stays at the indented
        # level; subsequent emits land here.

    def _apply_value_pattern_in_loop(
        self, patt: Pattern, eid_local: str,
    ) -> None:
        """Like _apply_value_pattern but uses `continue` instead of
        `return` since we're inside a for-loop."""
        if isinstance(patt, BindPattern):
            if patt.target in self.var_to_local:
                existing = self.var_to_local[patt.target]
                if existing == eid_local:
                    return     # tautology: same local on both sides
                self.emit(f"if {existing} != {eid_local}:")
                self._push(); self.emit("continue"); self._pop()
            else:
                self.var_to_local[patt.target] = eid_local
            return
        if isinstance(patt, EntityPattern):
            for key, expected in patt.constraints.items():
                self._emit_loop_constraint_check(
                    f"trace.entities[{eid_local}]", eid_local, key, expected)
            return
        if isinstance(patt, AndPattern):
            self._apply_value_pattern_in_loop(patt.left, eid_local)
            self._apply_value_pattern_in_loop(patt.right, eid_local)
            return
        # Unknown — runtime fallback inside the loop.
        patt_ident = self.expose("patt", patt)
        items = ", ".join(
            f"{self.var_obj_name(v)}: {local}"
            for v, local in self.var_to_local.items()
        )
        bdict = self.fresh("b")
        self.emit(f"{bdict} = {{{items}}}")
        flag = self.fresh("ok")
        self.emit(f"{flag} = False")
        self.emit(f"for _ in {patt_ident}.apply_to_value("
                  f"{eid_local}, ctx, {bdict}):")
        self._push(); self.emit(f"{flag} = True; break"); self._pop()
        self.emit(f"if not {flag}:")
        self._push(); self.emit("continue"); self._pop()

    def _emit_loop_constraint_check(
        self, ent_local: str, eid_local: str, key: str, expected,
    ) -> None:
        """In-loop variant: `continue` instead of `return` on failure."""
        is_local = False
        if isinstance(expected, Var):
            if expected not in self.var_to_local:
                self.emit("continue  # unbound var constraint")
                return
            value_repr = self.var_to_local[expected]
            is_local = True
        else:
            value_repr = repr(expected)

        if key == "type":
            self.emit(
                f"if not ctx.lexicon.types.is_subtype("
                f"{ent_local}.entity_type, {value_repr}):")
            self._push(); self.emit("continue"); self._pop()
        elif key == "concept":
            self.emit(f"if {ent_local}.concept_lemma != {value_repr}:")
            self._push(); self.emit("continue"); self._pop()
        elif key == "has_suffix":
            self.emit(f"if not {ent_local}.concept_lemma.endswith("
                      f"{value_repr}):")
            self._push(); self.emit("continue"); self._pop()
        else:
            actual = self.fresh("v")
            self.emit(f"{actual} = ctx.effective_property("
                      f"{eid_local}, {key!r})")
            if value_repr == "Ellipsis":
                self.emit(f"if {actual} is None or {actual} == [] "
                          f"or {actual} == '':")
                self._push(); self.emit("continue"); self._pop()
                return
            self.emit(f"if {actual} is None:")
            self._push(); self.emit("continue"); self._pop()
            self.emit(f"if isinstance({actual}, list):")
            self._push()
            self.emit(f"if {value_repr} not in {actual}:")
            self._push(); self.emit("continue"); self._pop()
            self._pop()
            self.emit("else:")
            self._push()
            self.emit(f"if {actual} != {value_repr}:")
            self._push(); self.emit("continue"); self._pop()
            self._pop()

    def _compile_rel_clause(self, clause: RelPattern) -> None:
        """Generate a `for args in ctx.relations_of(name)` loop with
        positional binding extraction. Pre-bound vars become equality
        filters via `if args[i] != local: continue`. Loop-invariant
        lookups (`lexicon.relations[name]`, `arg_names.index(role)`)
        are hoisted to the function prelude — see `_ensure_rel_argnames`
        and `_arg_index_local`.

        Symmetric-relation handling: for relations declared `symmetric`
        in the lexicon (apud, samloke), the interpreter yields BOTH arg
        orderings. We mirror that by generating an inner loop that
        feeds each tuple twice — once as-asserted, once swapped — when
        the relation is symmetric and arity is 2.

        Indexed-lookup optimization: if any arg's BindPattern targets
        a Var already in `var_to_local`, that arg is effectively a
        pre-known value. Use `ctx.relations_with_arg(name, idx, val)`
        to fetch only matching relations, instead of iterating every
        relation and post-filtering. Cuts O(N_rels) → O(matching).
        Skipped for symmetric relations (the swap-yielding inner loop
        complicates which arg-index actually carries the join key).
        """
        rel_name = clause.relation
        argnames_local, _ = self._ensure_rel_argnames(rel_name)
        sym_local = self._sym_local_for(rel_name)
        outer_args = self.fresh("rawargs")
        self.emit(f"for {outer_args} in ctx.relations_of({rel_name!r}):")
        self._push()
        self.emit(f"if {argnames_local} is None or "
                  f"len({outer_args}) != len({argnames_local}):")
        self._push(); self.emit("continue"); self._pop()
        # Yield the asserted ordering, plus the swap when symmetric +
        # arity 2 + non-reflexive. Materializing as a tuple is cheap;
        # we'd otherwise need to emit a duplicated body.
        args_local = self.fresh("args")
        self.emit(f"for {args_local} in ((({outer_args},) + "
                  f"((({outer_args}[1], {outer_args}[0]),) "
                  f"if {sym_local} and len({outer_args}) == 2 "
                  f"and {outer_args}[0] != {outer_args}[1] else ())) "
                  f"if {sym_local} else ({outer_args},)):")
        self._push()

        for arg_name, arg_pat in clause.arg_patterns.items():
            idx_local = self._arg_index_local(rel_name, arg_name)
            arg_val_local = self.fresh(f"arg_{arg_name}")
            self.emit(f"{arg_val_local} = {args_local}[{idx_local}]")
            self._apply_value_pattern_in_loop(arg_pat, arg_val_local)

    def _fallback_clause(self, clause: Pattern) -> None:
        """Runtime fallback for a given clause: invoke `.search(ctx, b)`
        with the current bindings dict, iterate over results.

        Negation (`NotPattern`) is special: it yields the input bindings
        unchanged when the inner has no match — it never binds new vars.
        Treating its inner's `.variables()` as introducing locals would
        try to read keys that aren't there and skip the iteration.
        Other fallback patterns (Or, Closure, HasConceptField) DO bind
        their inner vars, so we extract those into locals if present."""
        patt_ident = self.expose("patt", clause)
        items = ", ".join(
            f"{self.var_obj_name(v)}: {local}"
            for v, local in self.var_to_local.items()
        )
        bdict = self.fresh("b")
        self.emit(f"{bdict} = {{{items}}}")
        b_iter = self.fresh("bnew")
        self.emit(f"for {b_iter} in {patt_ident}.search(ctx, {bdict}):")
        self._push()
        if isinstance(clause, NotPattern):
            # Negation succeeded — no new vars to extract. Fall through
            # so subsequent clauses / final yield run inside this loop.
            return
        # Bind any new vars the runtime pattern produced into locals
        # so subsequent codegen + the final yield can reference them.
        new_vars = clause.variables() - set(self.var_to_local.keys())
        for v in new_vars:
            new_local = self.fresh(f"v_{v.name}")
            obj = self.var_obj_name(v)
            self.emit(f"{new_local} = {b_iter}.get({obj})")
            self.emit(f"if {new_local} is None:")
            self._push(); self.emit("continue"); self._pop()
            self.var_to_local[v] = new_local


# ---------- helpers (mirror patterns._find_entity_source) ----------

def _find_entity_source(pattern: Pattern) -> Optional[Pattern]:
    """Same logic as patterns._find_entity_source but returns the
    EntityPattern itself if found at the top of an AndPattern's
    immediate children (we don't recurse deeper to keep codegen sane)."""
    if isinstance(pattern, AndPattern):
        if isinstance(pattern.left, EntityPattern):
            return pattern.left
        if isinstance(pattern.right, EntityPattern):
            return pattern.right
    return None


# -------------------- public API --------------------------------

def compile_rule(rule) -> Optional[Callable]:
    """Compile a rule's match phase to a generator. Returns None if the
    compiler can't handle the rule (caller falls back to interpreter)."""
    try:
        c = _Compiler(rule)
        src = c.compile()
    except _Unsupported:
        return None
    return _exec_enum(src, c, rule.name, "rule")


def compile_derivation(deriv) -> Optional[Callable]:
    """Compile a derivation's match phase. Returns `enum(ctx)` instead
    of `enum(event, ctx)` since derivations have no focus event. Same
    failure mode: returns None if codegen bails out."""
    try:
        c = _Compiler(deriv)
        src = c.compile_derivation()
    except _Unsupported:
        return None
    return _exec_enum(src, c, deriv.name, "deriv")


def _exec_enum(src, compiler, name, kind) -> Callable:
    ns = dict(compiler.namespace)
    try:
        exec(compile(src, f"<dsl_compile:{kind}:{name}>", "exec"), ns)
    except SyntaxError as e:
        raise RuntimeError(
            f"compiled {kind} {name!r} has SyntaxError: {e}\n"
            f"--- generated source ---\n{src}\n") from e
    fn = ns["enum"]
    fn.__source__ = src
    return fn


# Sentinel: distinguishes "not yet compiled" from "compiled to None
# (codegen bailed out, use interpreter)". Stored as the default for
# the per-rule cache attribute.
_UNCOMPILED = object()


def get_compiled_enum(rule) -> Optional[Callable]:
    """Cached compile_rule. Stores the result on the rule object itself
    (`rule._compiled_enum`) so the cache lifetime tracks the rule's
    lifetime — avoids the id() reuse bug where a freshly-allocated
    rule could land at a dead rule's memory address and inherit a
    stale compiled function from a module-level dict cache.
    Returns None when codegen bailed out — caller uses the interpreter."""
    cached = getattr(rule, "_compiled_enum", _UNCOMPILED)
    if cached is _UNCOMPILED:
        cached = compile_rule(rule)
        rule._compiled_enum = cached
    return cached


def get_compiled_deriv_enum(deriv) -> Optional[Callable]:
    """Cached compile_derivation. Same per-instance attribute caching
    as `get_compiled_enum` — see that function's docstring for the
    rationale."""
    cached = getattr(deriv, "_compiled_enum", _UNCOMPILED)
    if cached is _UNCOMPILED:
        cached = compile_derivation(deriv)
        deriv._compiled_enum = cached
    return cached
