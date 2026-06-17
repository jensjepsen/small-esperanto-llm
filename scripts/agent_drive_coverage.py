"""Agent-drive coverage runner.

Generates training-corpus traces by running the agent planner across
sampled scenes. Two complementary samplers run in sequence, each
producing a JSONL file under `runs/`:

  - `run_coverage`            forward sampler: build a random scene,
                              pick a drive its content supports,
                              augment scene if needed, plan, fire.
  - `run_coverage_regression` regression sampler: pick a verb you
                              want fired, instantiate its roles into
                              a fresh scene, drive the planner toward
                              it. Multi-step chains fall out via
                              precondition subgoaling.

The library this script orchestrates lives in
`esperanto_lm.ontology.agent` (planner / dispatcher / sampler /
loop / coverage) and `esperanto_lm.ontology.regression`. This file
is intentionally thin — its only job is wiring the lexicon + rule
set into the two coverage runners.

Run as:
    uv run python scripts/agent_drive_coverage.py
"""
from __future__ import annotations

from esperanto_lm.ontology import load_lexicon
from esperanto_lm.ontology.agent import (
    run_coverage, run_coverage_regression,
)
from esperanto_lm.ontology.dsl.rules import (
    DEFAULT_DSL_RULES, RUNTIME_DERIVATIONS,
)


def main():
    lex = load_lexicon()
    rules = list(DEFAULT_DSL_RULES)
    # RUNTIME_DERIVATIONS only — concept-static derivations are
    # baked into entity.properties at lexicon load time, so re-firing
    # them at runtime is wasted work. ~2× speedup on coverage runs
    # with no loss of derived state at the consumer level.
    derivations = list(RUNTIME_DERIVATIONS)

    run_coverage(
        lex, rules, derivations, n_scenes=200, seed=0,
        save_jsonl="runs/agent_sim_coverage.jsonl")

    run_coverage_regression(
        lex, rules, derivations, n_scenes=200, seed=0,
        save_jsonl="runs/agent_sim_regression.jsonl")


if __name__ == "__main__":
    main()
