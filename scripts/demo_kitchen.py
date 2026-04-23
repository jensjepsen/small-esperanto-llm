"""End-to-end demo of the ontology + causal engine.

Scene: a person is in the kitchen, holding tranĉilo (derived at lexicon
load time from tranĉi + -il-, NOT authored). Bread is on the table.
The person uses the knife on the bread. The generic instrument-use rule
fires the cut, which mutates the bread's integrity.
"""
from __future__ import annotations

from esperanto_lm.ontology import (
    DEFAULT_RULES,
    Trace,
    load_lexicon,
    make_event,
    make_use_instrument,
    run_to_fixed_point,
)
from esperanto_lm.ontology.loader import resolve_signature


def main() -> None:
    lex = load_lexicon()

    # Sanity: tranĉilo must be derived, not authored.
    knife = lex.concepts["tranĉilo"]
    assert knife.derived, "tranĉilo should have been compositionally derived"
    sig_verb = resolve_signature(lex, knife)
    assert sig_verb is not None and sig_verb.lemma == "tranĉi"
    print(f"derived: {knife.lemma}  (entity_type={knife.entity_type}, "
          f"functional_signature -> {sig_verb.lemma})")

    # Build a trace.
    trace = Trace()
    person = trace.add_entity("persono", lex, entity_id="petro")
    kitchen = trace.add_entity("kuirejo", lex, entity_id="kuirejo")
    table = trace.add_entity("tablo", lex, entity_id="tablo")
    bread = trace.add_entity("pano", lex, entity_id="pano")
    knife_inst = trace.add_entity("tranĉilo", lex, entity_id="tranĉilo")

    trace.assert_relation("en", (person.id, kitchen.id), lex)
    trace.assert_relation("havi", (person.id, knife_inst.id), lex)
    trace.assert_relation("sur", (bread.id, table.id), lex)

    # Seed event: person uses the knife on the bread.
    seed = make_event(
        action="uzi",
        roles={
            "agent": person.id,
            "instrument": knife_inst.id,
            "theme": bread.id,
        },
    )
    trace.add_event(seed)

    iters = run_to_fixed_point(trace, DEFAULT_RULES + [make_use_instrument(lex)])

    # ---- print the trace ----
    print(f"\nfixed point reached in {iters} iteration(s).\n")
    print("entities:")
    for ent in trace.entities.values():
        derived_tag = ""
        c = lex.concepts.get(ent.concept_lemma)
        if c is not None and c.derived:
            derived_tag = "  [derived]"
        print(f"  {ent.id:<10} {ent.concept_lemma:<10} "
              f"({ent.entity_type}){derived_tag}")
        if ent.properties:
            for k, v in ent.properties.items():
                print(f"    {k} = {v}")

    print("\nrelations:")
    for r in trace.relations:
        print(f"  {r.relation}({', '.join(r.args)})")

    print("\nevents (with causal links):")
    for ev in trace.events:
        causes = (" <- " + ", ".join(ev.caused_by)) if ev.caused_by else ""
        roles = ", ".join(f"{k}={v}" for k, v in ev.roles.items())
        print(f"  {ev.id}  {ev.action}({roles}){causes}")


if __name__ == "__main__":
    main()
