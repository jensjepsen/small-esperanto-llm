"""Run regress_for_goal N times, tally the no-sample reasons set by
the `_bail()` channel in goal_sampler."""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from esperanto_lm.ontology import load_lexicon
from esperanto_lm.ontology.dsl.rules import DEFAULT_DSL_RULES
from esperanto_lm.ontology.regression import goal_sampler as gs


def main():
    lex = load_lexicon()
    rules = list(DEFAULT_DSL_RULES)
    rng = random.Random(0)
    n = 2000
    none_count = 0
    reasons: Counter = Counter()
    for _ in range(n):
        gs.LAST_NO_SAMPLE_REASON = None
        out = gs.regress_for_goal(lex, rng, rules)
        if out is None:
            none_count += 1
            reasons[gs.LAST_NO_SAMPLE_REASON or "<unset>"] += 1
    print(f"total={n} none={none_count} ({none_count/n:.1%})")
    print("breakdown:")
    for k, v in reasons.most_common(30):
        print(f"  {v:4d} {k}")


if __name__ == "__main__":
    main()
