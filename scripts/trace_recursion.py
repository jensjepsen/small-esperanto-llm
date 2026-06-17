"""Trace each plan_to_X call the planner makes for a failing scene.
Goal: see the actual recursion tree, not just the deepest leaf.

Scene: virino in domo, tuko in domo (with lamp aktiva), herbo in
ĝardeno apud domo. Drive: virino wants herbo.wetness=seka.

The expected chain is preni(tuko) → iri(ĝardeno) → eniri(ĝardeno) →
sekigi. Currently fails. We dump every recursion entry/exit to see
where the planner gives up.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from esperanto_lm.ontology import load_lexicon
from esperanto_lm.ontology.causal import Trace
from esperanto_lm.ontology.dsl.rules import (
    DEFAULT_DSL_RULES, RUNTIME_DERIVATIONS,
)
from esperanto_lm.ontology.sampler import (
    _add_entity_randomized, _ensure_world,
)
from esperanto_lm.ontology.agent.dispatcher import plan_for_drive
from esperanto_lm.ontology.agent import planner as _p


def main() -> None:
    lex = load_lexicon()
    rng = random.Random(0)

    t = Trace()
    _ensure_world(t, lex, rng)
    t.entities["mondo"].set_property("tempo_de_tago", "tago")
    _add_entity_randomized(t, "domo", lex, rng, entity_id="domo")
    _add_entity_randomized(t, "ĝardeno", lex, rng, entity_id="ĝardeno")
    t.assert_relation("apud", ("ĝardeno", "domo"), lex)
    _add_entity_randomized(t, "virino", lex, rng, entity_id="virino")
    t.assert_relation("en", ("virino", "domo"), lex)
    _add_entity_randomized(t, "herbo", lex, rng, entity_id="herbo")
    t.entities["herbo"].set_property("wetness", "malseka")
    t.assert_relation("en", ("herbo", "ĝardeno"), lex)
    _add_entity_randomized(t, "tuko", lex, rng, entity_id="tuko")
    t.assert_relation("en", ("tuko", "domo"), lex)
    t.entities["domo_pordo"].set_property("openness", "malfermita")
    _add_entity_randomized(t, "lampo", lex, rng, entity_id="lampo")
    t.entities["lampo"].set_property("power_state", "aktiva")
    t.assert_relation("en", ("lampo", "domo"), lex)
    drive = ("entity_slot", "virino", "herbo", "wetness", "seka")

    # Wrap the three public planner entries with depth-aware logging.
    # Keep output short by truncating very long traces.
    indent = [0]
    log: list[str] = []
    MAX_LINES = 500

    def wrap(fn, name):
        def wrapped(*args, **kwargs):
            d = indent[0]
            if len(log) < MAX_LINES:
                key_args = args[:3]
                log.append(f"{'  '*d}-> {name}{key_args}")
            indent[0] += 1
            try:
                result = fn(*args, **kwargs)
            finally:
                indent[0] -= 1
            if len(log) < MAX_LINES:
                status = (f"plan({len(result)})" if result is not None
                          else "FAIL")
                log.append(f"{'  '*d}<- {name} {status}")
            return result
        return wrapped

    _p.plan_to_achieve = wrap(_p.plan_to_achieve, "achieve")
    _p.plan_to_establish_relation = wrap(
        _p.plan_to_establish_relation, "establish")
    _p.plan_event_firing = wrap(_p.plan_event_firing, "fire")

    plan = plan_for_drive(
        drive, t, lex, DEFAULT_DSL_RULES, RUNTIME_DERIVATIONS, rng=rng)

    for line in log:
        print(line)
    print(f"\n=== Plan: {plan} ===")


if __name__ == "__main__":
    main()
