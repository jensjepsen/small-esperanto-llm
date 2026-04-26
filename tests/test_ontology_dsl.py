"""Phase 1 acceptance tests for the DSL substrate.

Nine must-have behaviors per the spec. Each test uses an in-memory
mini-lexicon built just for the scenario — isolates the DSL from the
real ontology so tests can exercise slot names (`made_of`, `wet`,
`flat`, `slippery`, ...) that the migration will eventually add.
"""
from __future__ import annotations

import pytest

from esperanto_lm.ontology.causal import Event, Trace, make_event
from esperanto_lm.ontology.loader import Lexicon
from esperanto_lm.ontology.schemas import (
    Action, Concept, PropertySlot, Relation, RoleSpec,
)
from esperanto_lm.ontology.types import TypeSpine

from esperanto_lm.ontology.dsl import (
    add_relation, bind, change, closure, create_entity, derive, emit,
    entity, event, has_concept_field, property, rel, rule, run_dsl, var,
)


# ------------------------- test lexicon fixture --------------------------

def _build_lex() -> Lexicon:
    """In-memory lexicon with the slot / type / concept / action /
    relation vocabulary the tests need. Isolated from the real
    ontology on disk."""
    types = TypeSpine({
        "physical": None, "animate": "physical", "inanimate": "physical",
        "location": "physical", "person": "animate", "animal": "animate",
        "artifact": "inanimate", "substance": "inanimate",
        "natural_object": "inanimate",
    })
    slots = {
        "made_of": PropertySlot(
            name="made_of", vocabulary=None, applies_to=["inanimate"]),
        "flammability": PropertySlot(
            name="flammability", vocabulary=["flammable", "fireproof"],
            applies_to=["inanimate"]),
        "ignitable": PropertySlot(
            name="ignitable", vocabulary=["yes", "no"],
            applies_to=["inanimate"]),
        "flat": PropertySlot(
            name="flat", vocabulary=None, applies_to=["inanimate"]),
        "wet": PropertySlot(
            name="wet", vocabulary=None, applies_to=["physical"]),
        "slippery": PropertySlot(
            name="slippery", vocabulary=None, applies_to=["physical"]),
        "fragility": PropertySlot(
            name="fragility", vocabulary=["fragile", "sturdy"],
            applies_to=["inanimate"]),
        "integrity": PropertySlot(
            name="integrity", vocabulary=["intact", "broken", "severed"],
            applies_to=["physical"]),
        "transforms_on_break": PropertySlot(
            name="transforms_on_break", vocabulary=None,
            applies_to=["inanimate"]),
    }
    concepts = {
        "breto": Concept(lemma="breto", entity_type="artifact", properties={}),
        "tablo": Concept(lemma="tablo", entity_type="artifact", properties={}),
        "glaso": Concept(
            lemma="glaso", entity_type="artifact",
            properties={"fragility": ["fragila"], "integrity": ["tuta"],
                        "transforms_on_break": ["vitropecetoj"]}),
        "vitropecetoj": Concept(
            lemma="vitropecetoj", entity_type="inanimate",
            properties={"fragility": ["fortika"]}),
        "persono": Concept(
            lemma="persono", entity_type="person", properties={}),
        "floro": Concept(
            lemma="floro", entity_type="natural_object", properties={}),
    }
    actions = {
        "fali": Action(
            lemma="fali", transitivity="intransitive", aspect="achievement",
            roles=[RoleSpec(name="theme", type="inanimate")],
            effects=[]),
        "rompiĝi": Action(
            lemma="rompiĝi", transitivity="intransitive",
            aspect="achievement",
            roles=[RoleSpec(name="theme", type="inanimate")],
            effects=[]),
        "aperi": Action(
            lemma="aperi", transitivity="intransitive",
            aspect="achievement",
            roles=[RoleSpec(name="theme", type="inanimate")],
            effects=[]),
        "bruli": Action(
            lemma="bruli", transitivity="intransitive", aspect="activity",
            roles=[RoleSpec(name="theme", type="inanimate")],
            effects=[]),
        "paŝi_sur": Action(
            lemma="paŝi_sur", transitivity="transitive", aspect="activity",
            roles=[RoleSpec(name="agent", type="person"),
                   RoleSpec(name="theme", type="physical")],
            effects=[]),
        "gliti": Action(
            lemma="gliti", transitivity="intransitive",
            aspect="achievement",
            roles=[RoleSpec(name="agent", type="person")],
            effects=[]),
    }
    relations = {
        "en": Relation(name="en", arity=2,
                       arg_types=["physical", "physical"],
                       arg_names=["contained", "container"]),
        "sur": Relation(name="sur", arity=2,
                        arg_types=["physical", "physical"],
                        arg_names=["contained", "container"]),
        "havi": Relation(name="havi", arity=2,
                         arg_types=["animate", "physical"],
                         arg_names=["owner", "theme"]),
    }
    return Lexicon(
        types=types, slots=slots, concepts=concepts,
        relations=relations, actions=actions, affixes={})


@pytest.fixture
def lex():
    return _build_lex()


def _ent(trace: Trace, concept_lemma: str, lex: Lexicon, eid: str):
    """Shortcut to add an entity by concept."""
    return trace.add_entity(concept_lemma, lex, entity_id=eid)


# =========================================================================
# Test 1 — pattern reuse across rules
# =========================================================================

def test_1_pattern_reuse_across_rules(lex):
    """An entity pattern assigned to a module-level constant works in
    multiple rules (both causal and derivation)."""
    T = var("T")
    WOODEN = entity(made_of="wood") & bind(T)

    # Reused in a derivation.
    wooden_flammable = derive(
        when=WOODEN,
        implies=property(T, "flammability", "brulebla"),
        name="wooden_flammable")

    # Reused in a causal rule (same pattern, different context).
    wooden_falls_burns = rule(
        when=event("fali", theme=WOODEN),
        then=emit("bruli", theme=T),
        name="wooden_falls_burns")

    t = Trace()
    _ent(t, "breto", lex, "breto")
    t.entities["breto"].set_property("made_of", "wood")
    t.events.append(make_event("fali", roles={"theme": "breto"}))

    run_dsl(t, [wooden_falls_burns], [wooden_flammable], lex)

    actions = [e.action for e in t.events]
    assert "bruli" in actions


# =========================================================================
# Test 2 — closure predicates bind-transparent
# =========================================================================

def test_2_closure_bind_transparent(lex):
    """Variables bound outside a closure are accessible inside the
    where predicate. OWNER is bound in a given clause that runs before
    the closure; the closure's `where` references OWNER to filter
    candidates to those the same OWNER possesses."""
    T = var("T")
    OWNER = var("OWNER")
    X = var("X")

    # Emit `aperi` (a different action) so the rule doesn't self-trigger
    # — the test is about bind-transparency, not contagion convergence.
    mark_owner_possessions = rule(
        when=event("bruli", theme=bind(T)),
        given=[
            rel("havi", owner=bind(OWNER), theme=T),
            closure({"en"}, from_=T, to_=bind(X),
                    where=rel("havi", owner=OWNER, theme=X)),
        ],
        then=emit("aperi", theme=X),
        name="mark_owner_possessions")

    t = Trace()
    _ent(t, "persono", lex, "petro")
    _ent(t, "breto", lex, "breto1")
    _ent(t, "breto", lex, "breto2")
    _ent(t, "tablo", lex, "tablo")
    t.assert_relation("havi", ("petro", "breto1"), lex)
    t.assert_relation("havi", ("petro", "breto2"), lex)
    # tablo is connected via en but NOT owned by petro.
    t.assert_relation("en", ("breto1", "breto2"), lex)
    t.assert_relation("en", ("breto2", "tablo"), lex)

    t.events.append(make_event("bruli", roles={"theme": "breto1"}))
    run_dsl(t, [mark_owner_possessions], [], lex)

    appeared = {e.roles.get("theme") for e in t.events if e.action == "aperi"}
    assert "breto2" in appeared          # owned + connected via en
    assert "tablo" not in appeared       # connected but not owned


# =========================================================================
# Test 3 — creation-to-reference flow within a firing
# =========================================================================

def test_3_create_entity_then_emit_resolves(lex):
    """create_entity(as_var=S) followed by emit(..., theme=S) in the
    same then-block resolves: the second effect sees the binding from
    the first."""
    T = var("T")
    S = var("S")
    K = var("K")

    shatter_creates_shards = rule(
        when=event("rompiĝi", theme=bind(T)),
        given=[has_concept_field(T, "transforms_on_break", K)],
        then=[
            create_entity(concept=K, as_var=S),
            emit("aperi", theme=S),
        ],
        name="shatter_creates_shards")

    t = Trace()
    _ent(t, "glaso", lex, "glaso")
    t.events.append(make_event("rompiĝi", roles={"theme": "glaso"}))

    run_dsl(t, [shatter_creates_shards], [], lex)

    aperi = [e for e in t.events if e.action == "aperi"]
    assert len(aperi) == 1
    shards_id = aperi[0].roles["theme"]
    assert shards_id in t.entities
    assert t.entities[shards_id].concept_lemma == "vitropecetoj"


# =========================================================================
# Test 4 — negation inside given
# =========================================================================

def test_4_negation_in_given(lex):
    """~rel(...) composes as a conjunctive clause."""
    T = var("T")
    orphan_fragile_breaks = rule(
        when=event("fali", theme=entity(fragility="fragila") & bind(T)),
        given=[~rel("en", container=bind(var("anyC")), contained=T)],
        then=emit("rompiĝi", theme=T),
        name="orphan_fragile_breaks")

    # Case A: glaso NOT en anything → rompiĝi fires.
    t = Trace()
    _ent(t, "glaso", lex, "glaso")
    t.events.append(make_event("fali", roles={"theme": "glaso"}))
    run_dsl(t, [orphan_fragile_breaks], [], lex)
    assert any(e.action == "rompiĝi" for e in t.events)

    # Case B: glaso en a tablo → negation fails → rompiĝi does not fire.
    t2 = Trace()
    _ent(t2, "glaso", lex, "glaso")
    _ent(t2, "tablo", lex, "tablo")
    t2.assert_relation("en", ("glaso", "tablo"), lex)
    t2.events.append(make_event("fali", roles={"theme": "glaso"}))
    run_dsl(t2, [orphan_fragile_breaks], [], lex)
    assert not any(e.action == "rompiĝi" for e in t2.events)


# =========================================================================
# Test 5 — simple derivation
# =========================================================================

def test_5_simple_derivation(lex):
    """entity(made_of='wood') → property(flammability, 'flammable')."""
    T = var("T")
    wooden_flammable = derive(
        when=entity(made_of="wood") & bind(T),
        implies=property(T, "flammability", "brulebla"),
        name="wooden_flammable")

    t = Trace()
    _ent(t, "breto", lex, "breto")
    t.entities["breto"].set_property("made_of", "wood")

    run_dsl(t, [], [wooden_flammable], lex)

    from esperanto_lm.ontology.dsl import DerivedState
    # The engine's DerivedState is internal to the run; re-run with the
    # observable: a causal rule that matches on the derived property.
    T2 = var("T2")
    bruli_if_flammable = rule(
        when=event("fali", theme=entity(flammability="brulebla") & bind(T2)),
        then=emit("bruli", theme=T2),
        name="bruli_if_flammable")

    t.events.append(make_event("fali", roles={"theme": "breto"}))
    run_dsl(t, [bruli_if_flammable], [wooden_flammable], lex)

    assert any(e.action == "bruli" and e.roles.get("theme") == "breto"
               for e in t.events)


# =========================================================================
# Test 6 — chained derivation
# =========================================================================

def test_6_chained_derivation(lex):
    """Derivation A produces P; derivation B matches on P and produces Q.
    Chain resolves within one derivation phase (fixed point)."""
    T = var("T")
    T2 = var("T2")

    wooden_flammable = derive(
        when=entity(made_of="wood") & bind(T),
        implies=property(T, "flammability", "brulebla"),
        name="wooden_flammable")

    flammable_ignitable = derive(
        when=entity(flammability="brulebla") & bind(T2),
        implies=property(T2, "ignitable", "yes"),
        name="flammable_ignitable")

    t = Trace()
    _ent(t, "breto", lex, "breto")
    t.entities["breto"].set_property("made_of", "wood")

    # Observe both via causal rules.
    T3 = var("T3")
    fires_on_ignitable = rule(
        when=event("fali", theme=entity(ignitable="yes") & bind(T3)),
        then=emit("bruli", theme=T3),
        name="fires_on_ignitable")

    t.events.append(make_event("fali", roles={"theme": "breto"}))
    run_dsl(t, [fires_on_ignitable],
            [wooden_flammable, flammable_ignitable], lex)

    assert any(e.action == "bruli" for e in t.events), (
        "chained derivation should make breto ignitable, "
        "which the causal rule should see")


# =========================================================================
# Test 7 — asserted overrides derived
# =========================================================================

def test_7_asserted_overrides_derived(lex):
    """Explicitly-asserted flammability=fireproof wins over the
    wooden→flammable derivation."""
    T = var("T")
    wooden_flammable = derive(
        when=entity(made_of="wood") & bind(T),
        implies=property(T, "flammability", "brulebla"),
        name="wooden_flammable")

    t = Trace()
    _ent(t, "breto", lex, "breto")
    t.entities["breto"].set_property("made_of", "wood")
    # Explicit asserted override.
    t.entities["breto"].set_property("flammability", "fireproof")

    T2 = var("T2")
    fires_if_flammable = rule(
        when=event("fali", theme=entity(flammability="brulebla") & bind(T2)),
        then=emit("bruli", theme=T2),
        name="fires_if_flammable")

    t.events.append(make_event("fali", roles={"theme": "breto"}))
    run_dsl(t, [fires_if_flammable], [wooden_flammable], lex)

    # Since asserted=fireproof wins, the derivation should not have
    # set flammability=flammable, so the causal rule doesn't fire.
    assert not any(e.action == "bruli" for e in t.events)


# =========================================================================
# Test 8 — reversal on condition failure
# =========================================================================

def test_8_reversal_on_condition_failure(lex):
    """When the derivation's precondition no longer holds, the derived
    property disappears — subsequent causal rules don't match."""
    T = var("T")
    wet_flat_slippery = derive(
        when=entity(flat="yes", wet="yes") & bind(T),
        implies=property(T, "slippery", "yes"),
        name="wet_flat_slippery")

    T2 = var("T2")
    A = var("A")
    slip_on_slippery = rule(
        when=event("paŝi_sur",
                   theme=entity(slippery="yes") & bind(T2),
                   agent=bind(A)),
        then=emit("gliti", agent=A),
        name="slip_on_slippery")

    t = Trace()
    _ent(t, "persono", lex, "petro")
    _ent(t, "tablo", lex, "tablo")
    t.entities["tablo"].set_property("flat", "yes")
    t.entities["tablo"].set_property("wet", "yes")

    # First firing: slippery should be derived; stepping on it → slip.
    t.events.append(make_event(
        "paŝi_sur", roles={"agent": "petro", "theme": "tablo"}))
    run_dsl(t, [slip_on_slippery], [wet_flat_slippery], lex)
    n_slips_first = sum(1 for e in t.events if e.action == "gliti")
    assert n_slips_first == 1

    # Now "dry" the tablo by re-setting wet to something else. In a real
    # pipeline this'd come from a causal change() effect; here we mutate
    # directly to exercise the reversal path cleanly.
    t.entities["tablo"].set_property("wet", "no")

    # Step again. Derivation should no longer fire (tablo isn't wet any
    # more), so slippery shouldn't hold, so slip_on_slippery shouldn't
    # match the new paŝi_sur event.
    t.events.append(make_event(
        "paŝi_sur", roles={"agent": "petro", "theme": "tablo"}))
    run_dsl(t, [slip_on_slippery], [wet_flat_slippery], lex)
    n_slips_second = sum(1 for e in t.events if e.action == "gliti")
    assert n_slips_second == n_slips_first, (
        f"after drying, no new gliti should fire — got {n_slips_second}")


# =========================================================================
# Test 9 — causal rule matches derived property
# =========================================================================

def test_9_causal_rule_matches_derived_property(lex):
    """A derivation produces flammability='brulebla' on a wooden thing.
    A causal rule matching entity(flammability='brulebla') fires on
    that wooden thing. Derivations are transparent to the causal
    matcher."""
    T = var("T")
    wooden_flammable = derive(
        when=entity(made_of="wood") & bind(T),
        implies=property(T, "flammability", "brulebla"),
        name="wooden_flammable")

    T2 = var("T2")
    falling_flammable_burns = rule(
        when=event("fali", theme=entity(flammability="brulebla") & bind(T2)),
        then=emit("bruli", theme=T2),
        name="falling_flammable_burns")

    t = Trace()
    _ent(t, "breto", lex, "breto")
    t.entities["breto"].set_property("made_of", "wood")  # asserted
    # Note: no direct flammability assertion. The derivation sets it.
    t.events.append(make_event("fali", roles={"theme": "breto"}))

    run_dsl(t, [falling_flammable_burns], [wooden_flammable], lex)

    actions = [e.action for e in t.events]
    assert "bruli" in actions, actions
