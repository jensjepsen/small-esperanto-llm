"""Agent planner: drives, simulation, plan_for_drive dispatcher.

Public API re-exports the most-used names so callers can `from
esperanto_lm.ontology.agent import plan_for_drive` without chasing
submodules. Internals live in `planner`, `drive_sampler`,
`dispatcher`, `loop`, `coverage`.
"""
from .coverage import run_coverage, run_coverage_regression
from .dispatcher import plan_for_drive
from .drive_sampler import sample_drive, sample_scene
from .loop import run_simulation
from .planner import (
    plan_action, plan_event_firing, plan_to_achieve, plan_to_co_locate,
    plan_to_establish_relation, plan_to_reach_count, plan_with_subgoals,
)
from .preferences import SLOT_PREFERENCES

__all__ = [
    "SLOT_PREFERENCES",
    "plan_action",
    "plan_event_firing",
    "plan_for_drive",
    "plan_to_achieve",
    "plan_to_co_locate",
    "plan_to_establish_relation",
    "plan_to_reach_count",
    "plan_with_subgoals",
    "run_coverage",
    "run_coverage_regression",
    "run_simulation",
    "sample_drive",
    "sample_scene",
]
