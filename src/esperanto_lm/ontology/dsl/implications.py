"""Derivation-rule implications.

In this slice, derivations can only imply properties (no relations, no
entities). A `property(entity_var, slot, value)` implication means:
for every binding produced by the derivation's `when + given`, assert
that entity has slot=value in the derived layer.

Derived properties are erased and re-materialized every cycle of the
engine's fixed-point loop, so a derivation whose precondition stops
holding automatically "un-derives" its output — this is what lets
'drying a wet surface makes it not slippery' just work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .patterns import Var


class Implication:
    def reads(self) -> set[Var]:
        return set()


@dataclass
class PropertyImplication(Implication):
    entity: Var
    slot: str
    value: Any                      # literal or Var

    def reads(self) -> set[Var]:
        out = {self.entity}
        if isinstance(self.value, Var):
            out.add(self.value)
        return out


def property(entity: Var, slot: str, value: Any) -> PropertyImplication:
    """Imply that the entity bound to `entity` has the named slot equal
    to `value` (a literal or a Var that will be resolved at firing)."""
    if not isinstance(entity, Var):
        raise TypeError("property(): entity must be a Var")
    return PropertyImplication(entity, slot, value)
