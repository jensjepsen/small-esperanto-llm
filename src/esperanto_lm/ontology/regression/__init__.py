"""Regression-scene sampling: SceneBuilder DSL, scene preferences,
seeders, and the dispatcher.

Public API re-exports the most-used names for callers that don't want
to chase submodules.
"""
from .scene_builder import (
    SCENE_PREFERENCES,
    SceneBuilder,
    scene,
)
from .seeders import (
    regress_for_buy,
    regress_for_clothing,
    regress_for_count,
    regress_for_give_count,
    regress_for_more_than,
    regress_for_movement,
    regress_for_self_slot,
    regress_for_sell,
    regress_for_vehicle,
    regress_for_verb,
    sample_regression_scene,
)

__all__ = [
    "SCENE_PREFERENCES",
    "SceneBuilder",
    "scene",
    "regress_for_buy",
    "regress_for_clothing",
    "regress_for_count",
    "regress_for_give_count",
    "regress_for_more_than",
    "regress_for_movement",
    "regress_for_self_slot",
    "regress_for_sell",
    "regress_for_vehicle",
    "regress_for_verb",
    "sample_regression_scene",
]
