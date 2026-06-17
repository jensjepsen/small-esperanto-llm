"""Reproduce a goal_sampler scene EXACTLY and trace what the planner
does, so we can find what's different vs. a clean hand-built version.

Setup: pick the seed where goal_sampler produces a known failing
scene (virino wants herbo.wetness=seka, from earlier diagnostic),
then run plan_for_drive with the same machinery the bench uses.

The goal here is forensic: identify why this particular scene
yields a None plan despite the chain being short in principle.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from esperanto_lm.ontology import load_lexicon
from esperanto_lm.ontology.dsl.rules import (
    DEFAULT_DSL_RULES, RUNTIME_DERIVATIONS,
)
from esperanto_lm.ontology.regression.goal_sampler import regress_for_goal
from esperanto_lm.ontology.regression.spawner import make_spawner
from esperanto_lm.ontology.agent.dispatcher import plan_for_drive
from esperanto_lm.ontology.agent import planner as _p


def main() -> None:
    lex = load_lexicon()

    # Hand-build the same scene goal_sampler produces (virino +
    # herbo.wetness=seka) — minus the runtime-spawned azenido —
    # with the tuko pre-placed so we isolate whether the planner can
    # find the iri+preni+iri+eniri+sekigi chain when there's
    # nothing exotic to subgoal.
    rng = random.Random(0)
    from esperanto_lm.ontology.causal import Trace
    from esperanto_lm.ontology.sampler import (
        _add_entity_randomized, _ensure_world,
    )
    t = Trace()
    _ensure_world(t, lex, rng)
    t.entities["mondo"].set_property("tempo_de_tago", "tago")
    _add_entity_randomized(t, "domo", lex, rng, entity_id="domo")
    _add_entity_randomized(t, "ĝardeno", lex, rng, entity_id="ĝardeno")
    t.assert_relation("apud", ("ĝardeno", "domo"), lex)
    _add_entity_randomized(t, "virino", lex, rng, entity_id="virino")
    # Place virino in domo (with the tuko). The chain should be
    # preni(tuko) → iri(ĝardeno) → eniri(ĝardeno) → sekigi(herbo).
    # havi(virino, tuko) derives samloke after preni, so the
    # iri doesn't break the instrument-samloke check.
    t.assert_relation("en", ("virino", "domo"), lex)
    _add_entity_randomized(t, "herbo", lex, rng, entity_id="herbo")
    t.entities["herbo"].set_property("wetness", "malseka")
    t.assert_relation("en", ("herbo", "ĝardeno"), lex)
    _add_entity_randomized(t, "tuko", lex, rng, entity_id="tuko")
    t.assert_relation("en", ("tuko", "domo"), lex)
    t.entities["domo_pordo"].set_property("openness", "malfermita")
    # Add active lamp so virino can vidi (preni needs scias_lokon).
    _add_entity_randomized(t, "lampo", lex, rng, entity_id="lampo")
    t.entities["lampo"].set_property("power_state", "aktiva")
    t.assert_relation("en", ("lampo", "domo"), lex)
    scene_id = "domo"
    drive = ("entity_slot", "virino", "herbo", "wetness", "seka")
    print(f"=== Scene (scene_id={scene_id}) ===")
    for eid, ent in t.entities.items():
        print(f"  {eid} ({ent.concept_lemma}, {ent.entity_type}) "
              f"props={dict(ent.properties)}")
    print("=== Relations ===")
    for r in t.relations:
        print(f"  {r.relation}{r.args}")
    print(f"\n=== Drive: {drive} ===\n")

    # Instrument
    log: list[tuple[int, tuple]] = []
    orig = _p._record_failure
    def trace_record(reason, depth=0):
        log.append((depth, reason))
        return orig(reason, depth)
    _p._record_failure = trace_record

    # Confirm derived state shows samloke(virino, tuko).
    from esperanto_lm.ontology.dsl.engine import compute_derived_state
    derived = compute_derived_state(t, RUNTIME_DERIVATIONS, lex)
    samloke_rels = [r for r in derived.relations if r[0] == 'samloke']
    print(f"\n=== Derived samloke relations ({len(samloke_rels)}) ===")
    for r in samloke_rels[:20]:
        print(f"  {r}")
    print()

    spawner = make_spawner(scene_id, lex, rng)
    plan = plan_for_drive(
        drive, t, lex, DEFAULT_DSL_RULES, RUNTIME_DERIVATIONS,
        rng=rng, entity_resolver=spawner)
    print("=== Plan ===")
    if plan is None:
        print("  None")
    else:
        for verb, roles in plan:
            print(f"  {verb}{roles}")

    print("\n=== All subgoal recordings (dedup, deepest-first) ===")
    seen = set()
    uniq = []
    for depth, reason in log:
        key = (depth, reason)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((depth, reason))
    for depth, reason in sorted(uniq, key=lambda x: -x[0])[:40]:
        print(f"  d{depth}: {reason}")


if __name__ == "__main__":
    main()
