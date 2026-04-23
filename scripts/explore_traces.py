"""Print traces from a handful of scenarios over the current ontology.

Used to eyeball what kinds of structured causal sequences fall out of the
existing 11-verb / 5-concept world before we invest in verbalization.

Each scenario:
  - sets up a Trace with some entities and relations
  - fires a seed event
  - runs the rule engine to fixed point
  - prints entities, relations, events (with causal links), and the
    property changes that effects produced

Some scenarios are "negative cases" where the rule should *not* fire — those
are useful to keep in the output to see the type-check guarding the engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from esperanto_lm.ontology import (
    DEFAULT_RULES,
    Trace,
    load_lexicon,
    make_event,
    make_use_instrument,
    run_to_fixed_point,
)
from esperanto_lm.ontology.loader import Lexicon


@dataclass
class Scenario:
    title: str
    setup: Callable[[Lexicon], Trace]
    seed: Callable[[Lexicon, Trace], None]
    note: str = ""


def _scenario_cut_bread(lex: Lexicon) -> Trace:
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("tablo", lex, entity_id="tablo")
    t.add_entity("pano", lex, entity_id="pano")
    t.add_entity("tranĉilo", lex, entity_id="tranĉilo")
    t.assert_relation("en", ("petro", "kuirejo"), lex)
    t.assert_relation("havi", ("petro", "tranĉilo"), lex)
    t.assert_relation("sur", ("pano", "tablo"), lex)
    return t

def _seed_cut_bread(lex, t):
    t.add_event(make_event("uzi", roles={
        "agent": "petro", "instrument": "tranĉilo", "theme": "pano"}))


def _scenario_lock_door(lex: Lexicon) -> Trace:
    t = Trace()
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("pordo", lex, entity_id="pordo")
    t.add_entity("ŝlosilo", lex, entity_id="ŝlosilo")
    t.assert_relation("havi", ("maria", "ŝlosilo"), lex)
    # Pordo concept defaults to locked; unlock so we can watch the lock fire.
    t.entities["pordo"].set_property("lock_state", "unlocked")
    return t

def _seed_lock_door(lex, t):
    t.add_event(make_event("uzi", roles={
        "agent": "maria", "instrument": "ŝlosilo", "theme": "pordo"}))


def _scenario_cut_table_edge_case(lex: Lexicon) -> Trace:
    """Type-passes but semantically odd: tranĉi accepts theme=physical, and
    tablo is an artifact (which inherits physical). The engine fires."""
    t = Trace()
    t.add_entity("persono", lex, entity_id="anna")
    t.add_entity("tablo", lex, entity_id="tablo")
    t.add_entity("tranĉilo", lex, entity_id="tranĉilo")
    t.assert_relation("havi", ("anna", "tranĉilo"), lex)
    return t

def _seed_cut_table(lex, t):
    t.add_event(make_event("uzi", roles={
        "agent": "anna", "instrument": "tranĉilo", "theme": "tablo"}))


def _scenario_lock_bread_negative(lex: Lexicon) -> Trace:
    """ŝlosi.theme = artifact; pano is substance (inanimate→physical, NOT
    artifact). Type check should reject; no ŝlosi event synthesized."""
    t = Trace()
    t.add_entity("persono", lex, entity_id="klaras")
    t.add_entity("pano", lex, entity_id="pano")
    t.add_entity("ŝlosilo", lex, entity_id="ŝlosilo")
    t.assert_relation("havi", ("klaras", "ŝlosilo"), lex)
    return t

def _seed_lock_bread(lex, t):
    t.add_event(make_event("uzi", roles={
        "agent": "klaras", "instrument": "ŝlosilo", "theme": "pano"}))


def _scenario_two_uses(lex: Lexicon) -> Trace:
    """Two seed events on the same trace — demonstrates engine handling
    multiple roots in the causal DAG."""
    t = Trace()
    t.add_entity("persono", lex, entity_id="petro")
    t.add_entity("persono", lex, entity_id="maria")
    t.add_entity("kuirejo", lex, entity_id="kuirejo")
    t.add_entity("pano", lex, entity_id="pano")
    t.add_entity("tablo", lex, entity_id="tablo")
    t.add_entity("pordo", lex, entity_id="pordo")
    t.add_entity("tranĉilo", lex, entity_id="tranĉilo")
    t.add_entity("ŝlosilo", lex, entity_id="ŝlosilo")
    t.assert_relation("en", ("petro", "kuirejo"), lex)
    t.assert_relation("en", ("maria", "kuirejo"), lex)
    t.assert_relation("havi", ("petro", "tranĉilo"), lex)
    t.assert_relation("havi", ("maria", "ŝlosilo"), lex)
    t.assert_relation("sur", ("pano", "tablo"), lex)
    t.entities["pordo"].set_property("lock_state", "unlocked")
    return t

def _seed_two_uses(lex, t):
    t.add_event(make_event("uzi", roles={
        "agent": "petro", "instrument": "tranĉilo", "theme": "pano"}))
    t.add_event(make_event("uzi", roles={
        "agent": "maria", "instrument": "ŝlosilo", "theme": "pordo"}))


SCENARIOS = [
    Scenario("cut bread (canonical)", _scenario_cut_bread, _seed_cut_bread),
    Scenario("lock door", _scenario_lock_door, _seed_lock_door),
    Scenario("cut table (type-passes; semantically odd)",
             _scenario_cut_table_edge_case, _seed_cut_table,
             note="Engine fires; the type spine has no notion of 'artifacts shouldn't be cut'."),
    Scenario("lock bread (NEGATIVE — type mismatch)",
             _scenario_lock_bread_negative, _seed_lock_bread,
             note="ŝlosi.theme=artifact; pano is substance. No ŝlosi synthesized."),
    Scenario("two parallel uses (multi-root DAG)",
             _scenario_two_uses, _seed_two_uses),
]


def print_trace(t: Trace, lex: Lexicon) -> None:
    print("  entities:")
    for ent in t.entities.values():
        c = lex.concepts.get(ent.concept_lemma)
        tag = "  [derived]" if (c and c.derived) else ""
        props = ""
        if ent.properties:
            props = "  " + ", ".join(f"{k}={v[0] if len(v)==1 else v}"
                                     for k, v in ent.properties.items())
        print(f"    {ent.id:<10} {ent.concept_lemma:<10} ({ent.entity_type}){tag}{props}")

    if t.relations:
        print("  relations:")
        for r in t.relations:
            print(f"    {r.relation}({', '.join(r.args)})")

    print("  events:")
    if not t.events:
        print("    (none)")
    for ev in t.events:
        causes = ""
        if ev.caused_by:
            causes = "  ← " + ", ".join(ev.caused_by)
        roles = ", ".join(f"{k}={v}" for k, v in ev.roles.items())
        print(f"    {ev.id}  {ev.action}({roles}){causes}")


def main() -> None:
    lex = load_lexicon()
    for sc in SCENARIOS:
        print(f"\n══ {sc.title} ══")
        if sc.note:
            print(f"  note: {sc.note}")
        t = sc.setup(lex)
        sc.seed(lex, t)
        iters = run_to_fixed_point(t, DEFAULT_RULES + [make_use_instrument(lex)])
        print(f"  (fixed point in {iters} iter)")
        print_trace(t, lex)


if __name__ == "__main__":
    main()
