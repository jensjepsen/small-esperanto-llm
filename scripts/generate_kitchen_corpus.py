"""Generate N kitchen-scene (trace, prose) pairs and write JSONL.

Each line: {trace, prose, metadata} where metadata captures the per-sample
seed, recipe label, number of events, and which rules fired (by inferring
from synthesized vs seeded events).

This is the read-and-decide artifact for the slice. Eyeball the output;
ask whether the variety is real.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from esperanto_lm.ontology import (
    DEFAULT_RULES,
    Trace,
    load_lexicon,
    make_use_instrument,
    prune_unused_persons,
    realize_trace,
    run_to_fixed_point,
    sample_scene,
)


def trace_to_dict(trace: Trace) -> dict:
    return {
        "entities": [
            {
                "id": e.id,
                "concept": e.concept_lemma,
                "entity_type": e.entity_type,
                "properties": e.properties,
            }
            for e in trace.entities.values()
        ],
        "relations": [
            {"relation": r.relation, "args": list(r.args)}
            for r in trace.relations
        ],
        "events": [
            {
                "id": ev.id,
                "action": ev.action,
                "roles": ev.roles,
                "caused_by": list(ev.caused_by),
            }
            for ev in trace.events
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20,
                    help="Number of scenes to sample (default: 20).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("data/causal_corpus/kitchen.jsonl"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    lex = load_lexicon()
    base_rng = random.Random(args.seed)

    written = 0
    with open(args.out, "w") as f:
        for i in range(args.n):
            # Per-sample seed so individual scenes are independently reproducible.
            sample_seed = base_rng.randrange(1 << 31)
            rng = random.Random(sample_seed)
            try:
                trace, info = sample_scene(lex, rng)
            except (ValueError, KeyError) as e:
                # Validation failure during sampling — log & skip rather than
                # silently producing invalid data.
                print(f"  [{i}] skipped (sampler invalid: {e})")
                continue

            seed_event_ids = {ev.id for ev in trace.events}
            iters = run_to_fixed_point(trace, DEFAULT_RULES + [make_use_instrument(lex)])

            synth_actions = [
                ev.action for ev in trace.events
                if ev.id not in seed_event_ids
            ]

            pruned_persons = prune_unused_persons(trace)

            prose = realize_trace(
                trace, lex, scene_location_id=info.scene_location_id)

            record = {
                "trace": trace_to_dict(trace),
                "prose": prose,
                "metadata": {
                    "sample_seed": sample_seed,
                    "recipe": info.recipe,
                    "persons": info.persons,
                    "n_objects": info.n_objects,
                    "n_events": len(trace.events),
                    "n_synthesized": len(synth_actions),
                    "synthesized_actions": synth_actions,
                    "pruned_persons": pruned_persons,
                    "iterations_to_fixed_point": iters,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written}/{args.n} samples → {args.out}")


if __name__ == "__main__":
    main()
