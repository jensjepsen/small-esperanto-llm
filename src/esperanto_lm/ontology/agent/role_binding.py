"""Helper for `kind="from_precondition"` role bindings.

A few schema verbs (rakonti, demandi, respondi, instrui) have roles
whose values come from positions in an existing relation tuple, not
from naive type-match enumeration over scene entities. The canonical
case is rakonti.rel_type: the actor can only tell about a relation
they *know*, so rel_type, theme, and objekto are bound by reading
positions 1, 2, and 3 of an existing `scias(agent, rel, theme, obj)`
fact.

The forward planner has elaborate from-precondition handling
(`_fp_tuple_pool` + `_ground_all_actions` around line 2755) that
also enumerates *hypothetical* producer tuples (rules whose effects
could add the scias fact). The forward sampler doesn't plan ahead —
it picks verbs whose preconditions hold *right now* — so it only
needs the real-tuple subset. This module exposes that simpler
helper so the planner and sampler share the same coupling logic for
scalar roles in the same precondition.
"""
from __future__ import annotations

from itertools import product
from typing import Optional


def from_precondition_role_bindings(action, trace, lex) -> list[dict]:
    """Enumerate valid role-binding dicts for an action's
    `kind=from_precondition` roles, sourced from real relation
    tuples in `trace.relations`.

    Return value:
      - `[{}]` when the action has NO from-precondition roles —
        caller should proceed with full naive role binding.
      - `[]` when the action HAS from-precondition roles but no
        matching tuples exist in state — action is ungroundable
        right now.
      - Otherwise, list of dicts. Each dict binds the fp roles AND
        any scalar role that appears in the same precondition's
        role-tuple (coupled-binding, see line 2756-2767 of
        forward_planner.py — without coupling, the cartesian over
        |agents|×|themes|×|scias_tuples| blows up the candidate
        count by orders of magnitude).

    Caller responsibility: roles NOT in the returned binding need
    naive type-matched filling. The coupled-scalar-role bindings
    in the dict take precedence over naive matches.
    """
    from ..schemas import RelationPrecondition

    fp_roles = [r for r in action.roles
                if getattr(r, "kind", "single") == "from_precondition"]
    if not fp_roles:
        return [{}]

    # Group fp roles by their source relation name.
    fp_groups: dict = {}
    for r in fp_roles:
        src = getattr(r, "from_precondition", None)
        if src is None:
            continue
        fp_groups.setdefault(src, []).append(r)
    if not fp_groups:
        return [{}]

    # Position map: source rel → {role_name → position} for ALL
    # roles named in the precondition matching that source. Drives
    # the scalar-coupling: when a fp role + a non-fp scalar role
    # both appear in `scias(agent, rel_type, theme, objekto)`,
    # binding from one fixes the other.
    role_pos_map: dict = {}
    for pc in action.preconditions:
        if not isinstance(pc, RelationPrecondition):
            continue
        if pc.rel not in fp_groups:
            continue
        pos_map = role_pos_map.setdefault(pc.rel, {})
        for i, rname in enumerate(pc.roles):
            pos_map.setdefault(rname, i)

    # Enumerate real tuples for each source.
    fp_tuples: dict = {}
    for src in fp_groups:
        rel_def = lex.relations.get(src)
        arity = rel_def.arity if rel_def is not None else None
        tuples: list = []
        seen: set = set()
        for r in trace.relations:
            if r.relation != src:
                continue
            args = tuple(r.args)
            if arity is not None and len(args) != arity:
                continue
            if args in seen:
                continue
            seen.add(args)
            tuples.append(args)
        fp_tuples[src] = tuples

    # Any source with zero tuples → action ungroundable.
    if any(not tuples for tuples in fp_tuples.values()):
        return []

    # Cartesian product over sources, build per-combination binding
    # dicts. Each binding fixes both the marked fp roles AND the
    # scalar roles coupled via shared precondition.
    sources = list(fp_tuples.keys())
    bindings: list[dict] = []
    for combo in product(*[fp_tuples[s] for s in sources]):
        binding: dict = {}
        ok = True
        for src, tup in zip(sources, combo):
            # Marked fp roles use from_precondition_position.
            for r in fp_groups[src]:
                pos = getattr(r, "from_precondition_position", None)
                if pos is None or pos >= len(tup):
                    ok = False
                    break
                if r.name in binding and binding[r.name] != tup[pos]:
                    ok = False
                    break
                binding[r.name] = tup[pos]
            if not ok:
                break
            # Coupled scalar roles via the precondition's position.
            pos_map = role_pos_map.get(src, {})
            for role_name, pos in pos_map.items():
                if pos >= len(tup):
                    continue
                val = tup[pos]
                if role_name in binding and binding[role_name] != val:
                    ok = False
                    break
                binding[role_name] = val
            if not ok:
                break
        if ok:
            bindings.append(binding)
    return bindings


__all__ = ["from_precondition_role_bindings"]
