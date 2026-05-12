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


def _run_batch(args):
    """Process one batch. args = (seed, n, sampler_name, max_depth)."""
    seed, n, sampler_name, max_depth = args
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
                    max_states=300, entity_resolver=spawner)
            else:
                plan = plan_for_drive(
                    drive, t, _LEX, _RULES, _DERIVATIONS, rng=rng,
                    entity_resolver=spawner, max_depth=max_depth,
                    simulation_budget=20000)
        except Exception:
            plan = None
        if plan:
            fired += 1
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
    return fired, failed, no_sample, failures


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
    args = p.parse_args()

    from collections import Counter
    n_tasks = (args.scenes + args.batch - 1) // args.batch
    samplers = ("verb", "goal") if args.sampler == "both" else (args.sampler,)
    for sampler in samplers:
        tasks = [(args.seed + i, args.batch, sampler, args.max_depth)
                 for i in range(n_tasks)]
        start = time.time()
        fired = failed = no_sample = 0
        all_failures: Counter = Counter()
        with Pool(processes=args.workers,
                   initializer=_init_worker) as pool:
            for f, fa, ns, failures in pool.imap_unordered(_run_batch, tasks):
                fired += f
                failed += fa
                no_sample += ns
                for k in failures:
                    all_failures[k] += 1
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


if __name__ == "__main__":
    main()
