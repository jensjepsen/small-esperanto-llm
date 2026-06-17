"""For each failed scene from the goal sampler, categorize the
failure cause:

  re-establishment: at least one candidate plan was rejected by
                    `_candidate_breaks_preserved`. Permutation alone
                    couldn't find a non-breaking order.
  unproducible:     deepest leaf was a property whose (slot, value)
                    has no writer (verb effect, rule effect, or
                    derivation). Real schema gap.
  search-exhausted: no `_candidate_breaks_preserved` hit AND no
                    unproducible leaf; just ran out of options. May
                    be a depth/budget limit or a missing producer
                    branch.
  spawn-budget:     the entity resolver returned None during the
                    failed plan, suggesting the search needed more
                    distinct entities than the budget allowed.

Reports counts + a few sample drives per category. Runs N scenes
single-threaded so we can intercept the planner's internals.
"""
from __future__ import annotations

import random
import sys
from collections import Counter
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
    n_scenes = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    lex = load_lexicon()
    rng = random.Random(0)

    writable = _p._writable_slot_values(
        DEFAULT_DSL_RULES, lex.actions, RUNTIME_DERIVATIONS)

    # Intercept the breaks_preserved gate and the spawner.
    _state: dict = {"breaks_hits": 0, "spawn_nones": 0}
    orig_breaks = _p._candidate_breaks_preserved

    def trace_breaks(*args, **kwargs):
        result = orig_breaks(*args, **kwargs)
        if result:
            _state["breaks_hits"] += 1
        return result
    _p._candidate_breaks_preserved = trace_breaks

    counts: Counter = Counter()
    samples: dict[str, list] = {
        k: [] for k in (
            "re-establishment", "unproducible", "search-exhausted",
            "spawn-budget")
    }
    failed = total = 0
    for _ in range(n_scenes):
        sample = regress_for_goal(lex, rng, DEFAULT_DSL_RULES)
        if sample is None:
            continue
        total += 1
        t, scene_id, drive = sample
        _state["breaks_hits"] = 0
        _state["spawn_nones"] = 0

        # Wrap the spawner to count None returns.
        base_spawner = make_spawner(scene_id, lex, rng)

        def wrapped_spawner(*a, **kw):
            r = base_spawner(*a, **kw)
            if r is None:
                _state["spawn_nones"] += 1
            return r

        plan = plan_for_drive(
            drive, t, lex, DEFAULT_DSL_RULES, RUNTIME_DERIVATIONS,
            rng=rng, entity_resolver=wrapped_spawner)
        if plan is not None:
            continue
        failed += 1
        leaf = _p.get_planner_failure_reason()

        # Classify
        category = "search-exhausted"
        if _state["breaks_hits"] > 0:
            category = "re-establishment"
        elif (leaf is not None and leaf[0] == "property"
                and (leaf[2], leaf[3]) not in writable):
            category = "unproducible"
        elif _state["spawn_nones"] > 5:
            category = "spawn-budget"
        counts[category] += 1
        if len(samples[category]) < 5:
            samples[category].append({
                "drive": drive,
                "leaf": leaf,
                "breaks_hits": _state["breaks_hits"],
                "spawn_nones": _state["spawn_nones"],
            })

    print(f"scenes={total}, failed={failed}, "
          f"yield={(total-failed)/max(total,1):.1%}\n")
    for cat, n in counts.most_common():
        print(f"  {n}× {cat}")
    print()
    for cat, sams in samples.items():
        if not sams:
            continue
        print(f"=== {cat} samples ===")
        for s in sams:
            print(f"  drive={s['drive']}, breaks_hits={s['breaks_hits']}, "
                  f"spawn_nones={s['spawn_nones']}, leaf={s['leaf']}")


if __name__ == "__main__":
    main()
