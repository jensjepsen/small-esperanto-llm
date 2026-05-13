"""Side-by-side: verb-first vs goal-first sampler yield + speed.

Runs N scenes through each sampler across W workers and reports yield
+ elapsed. Skips realization to focus on planner success rate.

Usage:
    uv run --python pypy3.11 --with pydantic --no-project python \\
        scripts/bench_samplers.py --scenes 500 --workers 4

The verb-first path is `sample_regression_scene` (current production
pool: regress_for_verb + specialized seeders). The goal-first path
is `regress_for_goal` (the unified-pipeline POC).
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

_LEX = None
_RULES = None
_DERIVATIONS = None


def _init_worker():
    here = Path(__file__).parent
    sys.path.insert(0, str(here.parent / "src"))
    from esperanto_lm.ontology import load_lexicon
    from esperanto_lm.ontology.dsl.rules import (
        DEFAULT_DSL_RULES, RUNTIME_DERIVATIONS,
    )
    global _LEX, _RULES, _DERIVATIONS
    _LEX = load_lexicon()
    _RULES = list(DEFAULT_DSL_RULES)
    _DERIVATIONS = list(RUNTIME_DERIVATIONS)


def _drive_summary(drive, t):
    """Compact dict describing the drive in concept terms."""
    if not drive:
        return {}
    kind = drive[0]
    if kind == "entity_slot":
        _, actor, target, slot, val = drive
        return {
            "kind": kind,
            "actor": t.entities[actor].concept_lemma,
            "target": t.entities[target].concept_lemma,
            "slot": slot, "value": val}
    if kind == "event_fire":
        _, actor, verb, _ = drive
        return {"kind": kind, "actor": t.entities[actor].concept_lemma,
                "verb": verb}
    return {"kind": kind, "raw": str(drive)}


def _plan_summary(plan, t):
    """Plan as list of (lemma, {role: concept_lemma}) for logging.

    List-valued role bindings (fari.parts) render as a comma-joined
    string of concept lemmas so the summary stays JSON-serializable."""
    def _render(e):
        if isinstance(e, (list, tuple)):
            return [t.entities[x].concept_lemma if x in t.entities else x
                    for x in e]
        return t.entities[e].concept_lemma if e in t.entities else e
    out = []
    for lemma, roles in plan:
        out.append({
            "verb": lemma,
            "roles": {rn: _render(e) for rn, e in roles.items()}})
    return out


def _entities_summary(t):
    """List in-scene concept lemmas (skipping body parts)."""
    return sorted(
        {e.concept_lemma for eid, e in t.entities.items()
         if e.entity_type != "inanimate" and eid != "mondo"})


def _run_batch(args):
    """Process one batch. args = (seed, n, sampler_name, max_depth,
    capture_samples). Returns (fired, failed, no_sample, failures,
    samples) where samples is a list of {kind, scene, drive, plan?,
    entities} dicts (cap at ~3 per batch each for fired/failed)."""
    seed, n, sampler_name, max_depth, capture = args
    import os
    use_forward = os.environ.get("USE_FORWARD") == "1"
    from esperanto_lm.ontology.agent.dispatcher import plan_for_drive
    from esperanto_lm.ontology.agent.planner import get_planner_failure_reason
    from esperanto_lm.ontology.regression import sample_regression_scene
    from esperanto_lm.ontology.regression.goal_sampler import regress_for_goal
    from esperanto_lm.ontology.regression.spawner import make_spawner
    from esperanto_lm.ontology.agent.forward_planner import plan_for_goal

    import os
    spawn_budget = int(os.environ.get("SPAWN_BUDGET", "6"))
    rng = random.Random(seed)
    fired = 0
    failed = 0
    no_sample = 0
    failures: list[tuple] = []
    samples: list = []
    SAMPLES_PER_KIND = 3
    fired_seen = 0
    failed_seen = 0
    for _ in range(n):
        if sampler_name == "verb":
            sample = sample_regression_scene(_LEX, rng, rules=_RULES)
        else:
            sample = regress_for_goal(_LEX, rng, _RULES)
        if sample is None:
            no_sample += 1
            continue
        t, scene_id, drive = sample
        spawner = make_spawner(
            scene_id, _LEX, rng, budget=spawn_budget,
            prefer_scene_p=float(os.environ.get("PREFER_SCENE_P", "1.0")))
        try:
            if use_forward:
                plan = plan_for_goal(
                    drive, t, _LEX, _RULES, _DERIVATIONS,
                    max_states=int(os.environ.get("MAX_STATES", "300")),
                    entity_resolver=spawner)
            else:
                plan = plan_for_drive(
                    drive, t, _LEX, _RULES, _DERIVATIONS, rng=rng,
                    entity_resolver=spawner, max_depth=max_depth,
                    simulation_budget=20000)
        except Exception:
            plan = None
        if plan:
            fired += 1
            if capture and fired_seen < SAMPLES_PER_KIND:
                samples.append({
                    "kind": "fired",
                    "scene": scene_id,
                    "drive": _drive_summary(drive, t),
                    "entities": _entities_summary(t),
                    "plan": _plan_summary(plan, t)})
                fired_seen += 1
        else:
            failed += 1
            leaf = get_planner_failure_reason()
            # Coarse-grain the leaf: drop the entity ids, keep
            # (kind, slot/name, types).
            if leaf is None:
                key = ("<no-leaf>",)
            elif leaf[0] == "property":
                _, eid, slot, val = leaf
                ent = t.entities.get(eid)
                et = ent.entity_type if ent else "?"
                cl = ent.concept_lemma if ent else "?"
                key = ("property", et, cl, slot, val)
            elif leaf[0] == "relation":
                _, name, args = leaf
                types = tuple(
                    (t.entities.get(a).entity_type if t.entities.get(a) else "?")
                    for a in args)
                key = ("relation", name, types)
            else:
                key = leaf[:2]
            failures.append(key)
            if capture and failed_seen < SAMPLES_PER_KIND:
                samples.append({
                    "kind": "failed",
                    "scene": scene_id,
                    "drive": _drive_summary(drive, t),
                    "entities": _entities_summary(t),
                    "leaf": str(leaf) if leaf is not None else None})
                failed_seen += 1
    return fired, failed, no_sample, failures, samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", type=int, default=500)
    p.add_argument("--workers", type=int, default=cpu_count())
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampler", choices=("verb", "goal", "both"),
                   default="both",
                   help="Which sampler(s) to bench")
    p.add_argument("--max-depth", type=int, default=8,
                   help="Planner DFS depth budget")
    p.add_argument("--samples-out", type=str,
                   default="runs/bench_samples.jsonl",
                   help="Write captured fired+failed plan samples as "
                        "JSONL to this path (one entry per scene). "
                        "Pass empty string to disable.")
    p.add_argument("--show-samples", type=int, default=0,
                   help="Print N fired and N failed samples to stdout")
    args = p.parse_args()

    import json
    from collections import Counter
    n_tasks = (args.scenes + args.batch - 1) // args.batch
    samplers = ("verb", "goal") if args.sampler == "both" else (args.sampler,)
    capture = bool(args.samples_out or args.show_samples)
    for sampler in samplers:
        tasks = [(args.seed + i, args.batch, sampler,
                  args.max_depth, capture)
                 for i in range(n_tasks)]
        start = time.time()
        fired = failed = no_sample = 0
        all_failures: Counter = Counter()
        all_samples: list = []
        with Pool(processes=args.workers,
                   initializer=_init_worker) as pool:
            for f, fa, ns, failures, samples in (
                    pool.imap_unordered(_run_batch, tasks)):
                fired += f
                failed += fa
                no_sample += ns
                for k in failures:
                    all_failures[k] += 1
                all_samples.extend(samples)
        elapsed = time.time() - start
        total = fired + failed
        yld = fired / total if total > 0 else 0.0
        print(f"{sampler:>5}: fired={fired}, failed={failed}, "
              f"no_sample={no_sample}, yield={yld:.1%}, "
              f"elapsed={elapsed:.1f}s "
              f"({elapsed*1000/max(total,1):.0f}ms/scene)")
        print(f"    top failures:")
        for k, v in all_failures.most_common(10):
            print(f"      {v}x {k}")

        if args.samples_out:
            path = args.samples_out
            if len(samplers) > 1:
                base, _, ext = path.rpartition(".")
                path = f"{base}.{sampler}.{ext}" if ext else f"{path}.{sampler}"
            with open(path, "w") as f:
                for s in all_samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"    wrote {len(all_samples)} samples to {path}")

        if args.show_samples:
            for kind in ("fired", "failed"):
                shown = 0
                print(f"    --- {kind} samples ---")
                for s in all_samples:
                    if s["kind"] != kind:
                        continue
                    print(f"      scene={s['scene']} drive={s['drive']}")
                    if kind == "fired":
                        for step in s["plan"]:
                            print(f"        > {step['verb']}({step['roles']})")
                    else:
                        print(f"        leaf={s.get('leaf')}")
                        print(f"        entities={s['entities']}")
                    shown += 1
                    if shown >= args.show_samples:
                        break


if __name__ == "__main__":
    main()
