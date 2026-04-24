"""Causal-rule effects.

Effects are the RHS of a causal rule — what happens when `when + given`
matches. Each effect carries references to variables (bound in `when`
or `given`) that the engine resolves at firing time.

Effects are applied sequentially within a single rule firing:
`create_entity(as_var=S) → emit(theme=S)` resolves because the second
effect sees bindings extended by the first.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

from .patterns import Var


class Effect:
    """Base class. Each effect knows which variables it reads; the
    engine checks these are bound at firing time."""

    def reads(self) -> set[Var]:
        return set()

    def writes(self) -> set[Var]:
        """Vars this effect binds (e.g. create_entity's as_var)."""
        return set()


# ---------------------------- emit ---------------------------------

@dataclass
class Emit(Effect):
    action: str
    role_vars: dict[str, Var | str]                     # role_name -> Var or literal
    # Property changes baked into the emitted event itself (not a
    # separate _change event). Mirrors the old engine's
    # `make_event(..., property_changes=...)` shape so traces match.
    property_changes: dict[tuple[Var | str, str], Any] = field(default_factory=dict)

    def reads(self) -> set[Var]:
        out = {v for v in self.role_vars.values() if isinstance(v, Var)}
        for (ent, _slot), val in self.property_changes.items():
            if isinstance(ent, Var):
                out.add(ent)
            if isinstance(val, Var):
                out.add(val)
        return out

    def changing(self, entity: Var | str, slot: str, value: Any) -> "Emit":
        """Return a copy of this emit with one more property_change
        attached to the emitted event. Chainable."""
        pc = dict(self.property_changes)
        pc[(entity, slot)] = value
        return dataclasses.replace(self, property_changes=pc)


def emit(action: str, **role_vars) -> Emit:
    """Emit an event. Role values may be Vars (bound earlier) or literal
    entity ids (strings). Chain `.changing(entity, slot, value)` to
    attach property changes to the emitted event itself.

        emit("rompiĝi", theme=T)
        emit("satiĝi", theme=A).changing(A, "hunger", "sated")
    """
    return Emit(action, dict(role_vars))


# ------------------------- create_entity ---------------------------

@dataclass
class CreateEntity(Effect):
    concept: Var | str              # concept lemma (literal or bound var)
    as_var: Var                     # variable that will hold the new id
    entity_id: Optional[str] = None  # explicit id (literal)
    # If set, the new entity's id is `f"{concept}_from_{from_value}"`.
    # Used by cascade rules (broken_fragile_creates_shards et al.) to
    # produce deterministic ids of the same shape the old engine used,
    # so trace parity holds across the migration.
    id_from: Optional[Var] = None

    def reads(self) -> set[Var]:
        out: set[Var] = set()
        if isinstance(self.concept, Var):
            out.add(self.concept)
        if self.id_from is not None:
            out.add(self.id_from)
        return out

    def writes(self) -> set[Var]:
        return {self.as_var}


def create_entity(
    concept: Var | str, as_var: Var, *,
    entity_id: Optional[str] = None,
    from_: Optional[Var] = None,
) -> CreateEntity:
    """Bring a new entity into the trace. `as_var` is bound to the new
    entity id so subsequent effects can reference it.

        create_entity(concept=K, as_var=S)
        create_entity(concept=K, as_var=S, from_=T)  # id = "{K}_from_{T}"
    """
    if not isinstance(as_var, Var):
        raise TypeError("create_entity: as_var must be a Var")
    return CreateEntity(concept, as_var, entity_id, from_)


# ------------------------- destroy_entity --------------------------

@dataclass
class DestroyEntity(Effect):
    """Mark an entity as no longer existing from this event onward.
    Sets `destroyed_at_event` on the EntityInstance; `Trace.entities_at(t)`
    stops returning it for `t > destroyed_at_event`. The entity stays
    in `trace.entities` for historical lookups (so the realizer can
    still read its concept/name), but downstream rules that check
    presence or iterate `entities_at` see it as gone."""
    target: Var | str

    def reads(self) -> set[Var]:
        return {self.target} if isinstance(self.target, Var) else set()


def destroy_entity(target: Var | str) -> DestroyEntity:
    """Mark the bound entity as destroyed at the current trace position.

        destroy_entity(T)
    """
    return DestroyEntity(target)


# ---------------------------- change -------------------------------

@dataclass
class Change(Effect):
    entity: Var | str
    slot: str
    value: Any

    def reads(self) -> set[Var]:
        return {self.entity} if isinstance(self.entity, Var) else set()


def change(entity: Var | str, slot: str, value: Any) -> Change:
    """Assert a property change on an entity (as part of an event's
    `property_changes`). Treated as asserted, not derived."""
    return Change(entity, slot, value)


# --------------------- add / remove relation -----------------------

@dataclass
class AddRelation(Effect):
    relation: str
    args: tuple[Var | str, ...]

    def reads(self) -> set[Var]:
        return {a for a in self.args if isinstance(a, Var)}


def add_relation(relation: str, *args: Var | str) -> AddRelation:
    """Assert a relation. Arguments may be Vars or literal entity ids."""
    return AddRelation(relation, tuple(args))


@dataclass
class RemoveRelation(Effect):
    relation: str
    args: tuple[Var | str, ...]

    def reads(self) -> set[Var]:
        return {a for a in self.args if isinstance(a, Var)}


def remove_relation(relation: str, *args: Var | str) -> RemoveRelation:
    """Retract a previously-asserted relation."""
    return RemoveRelation(relation, tuple(args))
