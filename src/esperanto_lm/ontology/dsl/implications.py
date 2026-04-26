"""Derivation-rule implications.

Derivations can imply two kinds of facts:

  property(entity_var, slot, value)
    Assert entity.slot=value in the derived-property layer for every
    binding produced by the derivation's `when + given`.

  relation(name, *args)
    Assert rel(name, *args) in the derived-relation layer. Args are
    Vars (resolved per binding) or literal entity ids. Used for
    relations whose existence follows from other state — e.g.
    `samloke(A, B)` follows from `A and B share an en container`.

Derived facts are erased and re-materialized every cycle of the
engine's fixed-point loop, so a derivation whose precondition stops
holding automatically "un-derives" its output — this is what lets
'drying a wet surface makes it not slippery' just work, and what lets
samloke un-derive when an actor walks away.
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


@dataclass
class RelationImplication(Implication):
    """`relation(name, *args)` — derived relation assertion. Each arg
    is a Var (resolved per binding) or a literal entity id string."""
    name: str
    args: tuple[Any, ...]

    def reads(self) -> set[Var]:
        return {a for a in self.args if isinstance(a, Var)}


@dataclass
class PartImplication(Implication):
    """`part(host, part_concept, relation)` — append a ConceptPart to the
    matched host concept's `parts` list.

    Bake-time only (parts are static structural metadata, not runtime
    state). Lets rules express "all persons have these parts" without
    repeating the parts list on every authored OR derived person concept.
    The runtime derivation engine skips this implication kind."""
    entity: Var
    part_concept: str
    relation: str = "havas_parton"

    def reads(self) -> set[Var]:
        return {self.entity}


def property(entity: Var, slot: str, value: Any) -> PropertyImplication:
    """Imply that the entity bound to `entity` has the named slot equal
    to `value` (a literal or a Var that will be resolved at firing)."""
    if not isinstance(entity, Var):
        raise TypeError("property(): entity must be a Var")
    return PropertyImplication(entity, slot, value)


def relation(name: str, *args: Any) -> RelationImplication:
    """Imply that rel(name, *args) holds in the derived-relation layer.
    Args may be Vars (resolved per binding) or literal entity ids."""
    return RelationImplication(name, tuple(args))


def part(entity: Var, part_concept: str,
         relation_name: str = "havas_parton") -> PartImplication:
    """Imply that the matched host concept has a part referencing
    `part_concept` via `relation_name`. Materialized at bake time;
    silent at runtime."""
    if not isinstance(entity, Var):
        raise TypeError("part(): entity must be a Var")
    return PartImplication(entity, part_concept, relation_name)
