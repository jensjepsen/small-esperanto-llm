"""Parity + sanity tests for the DSL rule compiler.

The compiler in `esperanto_lm.ontology.dsl.compile` generates a per-rule
`enum(event, ctx)` generator that yields binding dicts equivalent to the
interpreter's `enumerate_bindings(when, given, ctx)`. These tests assert
that equivalence on a representative set of rules and trace shapes —
catching codegen drift before it changes generated traces.
"""
from __future__ import annotations

import random

import pytest

from esperanto_lm.ontology import load_lexicon
from esperanto_lm.ontology.dsl.compile import (
    compile_derivation, compile_rule, get_compiled_deriv_enum,
    get_compiled_enum,
)
from esperanto_lm.ontology.dsl.engine import Rule
from esperanto_lm.ontology.dsl.rules import (
    DEFAULT_DSL_DERIVATIONS, DEFAULT_DSL_RULES, RUNTIME_DERIVATIONS,
)
from esperanto_lm.ontology.dsl.unifier import (
    DerivedState, MatchContext, enumerate_bindings,
)
from esperanto_lm.ontology.sampler import sample_chained_scene


@pytest.fixture(scope="module")
def lex():
    return load_lexicon()


@pytest.fixture(scope="module")
def rules():
    return list(DEFAULT_DSL_RULES)


@pytest.fixture(scope="module")
def derivs():
    return list(RUNTIME_DERIVATIONS)


def _bindings_set(iterator):
    """Materialize an iterator of binding dicts into a frozenset of
    frozensets so order doesn't matter for comparison."""
    return frozenset(
        frozenset(b.items()) for b in [dict(x) for x in iterator]
    )


def _all_rules_compile():
    """Every shipping rule must compile (or explicitly fall back).
    Surfaces compiler regressions on new pattern shapes immediately."""
    bailed = []
    for r in DEFAULT_DSL_RULES:
        if compile_rule(r) is None:
            bailed.append(r.name)
    return bailed


def test_every_default_rule_compiles_or_falls_back():
    """Codegen must produce SOMETHING for every rule — either a
    compiled generator or an explicit fallback marker. No SyntaxErrors,
    no crashes during generation. (Falling back is OK; silent breakage
    is not.)"""
    bailed = _all_rules_compile()
    # We don't assert bailed is empty — fallback is by design for
    # rules with NotPattern/OrPattern/Closure/etc. Just assert no
    # exceptions were raised during compile_rule (the test reaches
    # this point only if all 64 calls returned cleanly).
    assert isinstance(bailed, list)


def _build_traces(lex, rules, derivs, n=10):
    """Sample a small batch of chained scenes, return [(trace, ctx)]
    pairs covering a variety of verbs and binding shapes."""
    out = []
    for seed in range(n):
        rng = random.Random(seed + 7777)
        try:
            t, info, _setup = sample_chained_scene(
                lex, rng, rules=rules, derivations=derivs,
                max_events=12, chain_p=0.9)
        except Exception:
            continue
        ctx = MatchContext(
            trace=t, lexicon=lex, derived=DerivedState(), focus_event=None)
        out.append((t, ctx))
    return out


def test_compiled_enum_matches_interpreted_on_real_traces(lex, rules, derivs):
    """For each (rule, event) pair across a batch of sampled traces,
    the compiled enum must yield exactly the same set of bindings as
    the interpreter. Catches subtle slot-lookup or constraint-check
    drift the compiler might introduce."""
    pairs = _build_traces(lex, rules, derivs, n=10)
    assert pairs, "sampler produced no traces; test cannot run"

    mismatches = []
    checked = 0
    for trace, ctx in pairs:
        for r in rules:
            enum_fn = get_compiled_enum(r)
            if enum_fn is None:
                continue
            for ev in list(trace.events):
                if ev.action != r.when.action:
                    continue
                ctx.focus_event = ev
                interp = _bindings_set(
                    enumerate_bindings(r.when, r.given, ctx))
                compiled = _bindings_set(enum_fn(ev, ctx))
                checked += 1
                if interp != compiled:
                    mismatches.append((
                        r.name, ev.action, ev.id[:8],
                        sorted(interp), sorted(compiled)))
    if mismatches:
        # Show first few for diagnostic.
        sample = "\n".join(
            f"  {name} / {action} / {eid}: "
            f"interp={i!r}\n     compiled={c!r}"
            for name, action, eid, i, c in mismatches[:5])
        pytest.fail(
            f"{len(mismatches)} mismatches over {checked} (rule, event) "
            f"pairs:\n{sample}")
    assert checked > 0, "no rule/event matches found in sampled traces"


def test_compiled_enum_handles_role_with_no_constraints(lex):
    """A rule like manĝi_consumes_theme uses bare bind(T) on theme —
    no entity(...) constraint. Compiler should still produce a working
    enum that just binds the role's entity_id."""
    from esperanto_lm.ontology.dsl.rules import manĝi_consumes_theme
    fn = compile_rule(manĝi_consumes_theme)
    assert fn is not None
    src = fn.__source__
    # Sanity on shape — no entity_matches loop, just role lookup + yield.
    assert "event.roles.get('theme')" in src
    assert "yield {" in src


def test_compiled_enum_inlines_concept_constraint(lex):
    """When entity(concept=X) is present, compiled code should hit the
    concept-indexed iteration path (entities_of_concept) rather than
    a full scan. Verified by inspecting generated source."""
    # Find a rule with concept= constraint in given.
    target = None
    for r in DEFAULT_DSL_RULES:
        for clause in r.given:
            if _contains_concept_constraint(clause):
                target = r
                break
        if target is not None:
            break
    if target is None:
        pytest.skip("no shipping rule uses entity(concept=...)")
    fn = compile_rule(target)
    assert fn is not None
    assert "entities_of_concept" in fn.__source__


def _contains_concept_constraint(pattern):
    from esperanto_lm.ontology.dsl.patterns import (
        AndPattern, EntityPattern,
    )
    if isinstance(pattern, EntityPattern):
        return "concept" in pattern.constraints
    if isinstance(pattern, AndPattern):
        return (_contains_concept_constraint(pattern.left)
                or _contains_concept_constraint(pattern.right))
    return False


def test_compiled_enum_returns_fresh_dicts(lex, rules, derivs):
    """Compiled yields must produce distinct dicts (not share a mutated
    one like the interpreter does). Otherwise the engine's
    `[dict(b) for b in ...]` materialization wouldn't be redundant —
    it would be load-bearing — and a future caller that omits the
    copy would silently break."""
    pairs = _build_traces(lex, rules, derivs, n=5)
    for trace, ctx in pairs:
        for r in rules:
            enum_fn = get_compiled_enum(r)
            if enum_fn is None:
                continue
            for ev in list(trace.events):
                if ev.action != r.when.action:
                    continue
                ctx.focus_event = ev
                results = list(enum_fn(ev, ctx))
                if len(results) < 2:
                    continue
                # Assert all dicts are distinct objects.
                ids = {id(d) for d in results}
                assert len(ids) == len(results), (
                    f"{r.name}: compiled enum returned shared dict refs")
                return  # one positive case is enough


def test_every_derivation_compiles_or_falls_back():
    """Same guard as the rule version, for derivations."""
    bailed = []
    for d in DEFAULT_DSL_DERIVATIONS + RUNTIME_DERIVATIONS:
        if compile_derivation(d) is None:
            bailed.append(d.name)
    assert isinstance(bailed, list)


def test_compiled_deriv_enum_matches_interpreted(lex, rules, derivs):
    """For each derivation, verify the compiled enum yields the same
    bindings as `enumerate_bindings(d.when, d.given, ctx)` on a real
    sampled trace. Catches drift on derivation-specific patterns
    (OR-of-entity, RelPattern with multiple binds, etc.)."""
    pairs = _build_traces(lex, rules, derivs, n=8)
    assert pairs

    all_derivs = list(DEFAULT_DSL_DERIVATIONS) + list(RUNTIME_DERIVATIONS)
    mismatches = []
    checked = 0
    for trace, ctx in pairs:
        for d in all_derivs:
            enum_fn = get_compiled_deriv_enum(d)
            if enum_fn is None:
                continue
            interp = _bindings_set(
                enumerate_bindings(d.when, d.given, ctx))
            compiled = _bindings_set(enum_fn(ctx))
            checked += 1
            if interp != compiled:
                mismatches.append((d.name, sorted(interp)[:3],
                                   sorted(compiled)[:3]))
    if mismatches:
        sample = "\n".join(
            f"  {name}: interp={i}\n    compiled={c}"
            for name, i, c in mismatches[:5])
        pytest.fail(
            f"{len(mismatches)} derivation mismatches over {checked} "
            f"(deriv, trace) pairs:\n{sample}")
    assert checked > 0


def test_negation_in_given_compiles_correctly(lex):
    """`given=[~rel(...)]` must fire when the negation succeeds.
    Earlier bug: NotPattern fallback tried to extract its inner
    pattern's vars as if they'd been bound, so the rule never reached
    the yield. Regression guard."""
    from esperanto_lm.ontology.dsl import (
        bind, entity, event, rel, rule, var, emit,
    )
    from esperanto_lm.ontology.causal import Trace, make_event
    from esperanto_lm.ontology.dsl.engine import run_dsl

    T = var("T")
    orphan_breaks = rule(
        when=event("fali", theme=entity(fragility="fragila") & bind(T)),
        given=[~rel("en", container=bind(var("anyC")), contained=T)],
        then=emit("rompiĝi", theme=T),
        name="_orphan_breaks_compiled",
    )
    fn = compile_rule(orphan_breaks)
    assert fn is not None, "orphan_breaks should compile (falls back internally)"

    # Glaso NOT en anything → rompiĝi must fire.
    t = Trace()
    t.add_entity("glaso", lex, entity_id="glaso")
    t.events.append(make_event("fali", roles={"theme": "glaso"}))
    run_dsl(t, [orphan_breaks], [], lex)
    assert any(e.action == "rompiĝi" for e in t.events), (
        "compiled rule failed to fire when negation should pass; "
        f"events={[e.action for e in t.events]}")


def test_constrained_when_reorder_skips_negation_clauses(lex):
    """The constrained-when reorder must NOT defer past a NotPattern
    in given. NotPattern semantics depend on its referenced vars
    being bound — with V unbound, `~rel(..., k=V)` checks "no such
    relation anywhere" instead of "no such relation for this V",
    yielding nothing. Earlier bug: indoor_dark_without_active_lamp
    silently produced empty bindings post-reorder.

    Using indoor_dark_without_active_lamp directly: it has the exact
    shape (when=entity(constraints) & bind(L); given=[~rel(..., k=L)])
    that the reorder is tempted to swap."""
    from esperanto_lm.ontology.dsl.rules import (
        indoor_dark_without_active_lamp,
    )
    fn = compile_derivation(indoor_dark_without_active_lamp)
    src = fn.__source__
    # The compiled enum must NOT skip the when's outer entity scan —
    # confirmed by checking the source includes an entities_of_type
    # iteration (the original when's driving loop).
    assert "entities_of_type('location')" in src, (
        f"reorder fired despite NotPattern in given; src=\n{src}")


def test_compile_falls_back_cleanly_on_unknown_pattern(lex):
    """A hand-built rule using a not-yet-supported pattern shape should
    either compile (if the codegen handles it) or return None — never
    raise. Confirms the bail-out path is wired."""
    from esperanto_lm.ontology.dsl import (
        bind, entity, event, rel, rule, var, emit,
    )
    from esperanto_lm.ontology.dsl.patterns import OrPattern, NotPattern
    A = var("A")
    T = var("T")
    # Construct manually to bypass _validate_rule's stricter checks
    # if needed. event() + given with NotPattern of past_event is a
    # known stress test — our compiler currently falls back via runtime.
    r = rule(
        when=event("manĝi", agent=bind(A), theme=bind(T)),
        given=[
            # OrPattern with two entity branches — codegen falls back.
            OrPattern(
                entity(type="animate") & bind(A),
                entity(type="person") & bind(A),
            ),
        ],
        then=emit("satiĝi", theme=A),
        name="_test_or_branch",
    )
    fn = compile_rule(r)
    # Should not raise. Either compiles (if our inliner handles Or in
    # value position) or returns None.
    assert fn is None or callable(fn)
