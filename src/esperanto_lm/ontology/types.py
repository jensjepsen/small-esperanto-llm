"""Thin type spine.

Stored on disk as `data/ontology/types.json`: a flat dict
`{type_name: parent_type_name | None}`.

Used solely for argument-type constraints — no property inheritance.
"""
from __future__ import annotations

import json
from pathlib import Path


class TypeSpine:
    """Read-only type lattice.

    Construction validates that every parent referenced is also a key in the
    spine, so a typo like `"animat"` instead of `"animate"` fails loudly.
    """

    def __init__(self, parents: dict[str, str | None]):
        for child, parent in parents.items():
            if parent is not None and parent not in parents:
                raise ValueError(
                    f"type spine: '{child}' has unknown parent '{parent}'"
                )
        self._parents = dict(parents)
        # Memoize ancestor walks; the spine is small enough that we can
        # eagerly compute closures.
        self._ancestors: dict[str, frozenset[str]] = {}
        for t in self._parents:
            chain: list[str] = []
            cur: str | None = t
            while cur is not None:
                chain.append(cur)
                cur = self._parents[cur]
            self._ancestors[t] = frozenset(chain)

    @classmethod
    def from_json(cls, path: Path) -> "TypeSpine":
        with open(path) as f:
            return cls(json.load(f))

    def known(self, t: str) -> bool:
        return t in self._parents

    def is_subtype(self, t: str, ancestor: str) -> bool:
        """True if `t` is `ancestor` or a descendant of it.

        Hot-path simplification: skip the up-front validity checks on
        `t` and `ancestor`. Bad inputs now return False rather than
        raising KeyError. The engine validates entity_types and rule
        constraint types at construction time (`_validate_rule`,
        TypeSpine `__init__`), so by the time we're in the hot loop
        every (t, ancestor) pair is well-formed. Profile showed 1.1M
        calls / 0.36s self before — the validation lookups dominated."""
        ancs = self._ancestors.get(t)
        return ancs is not None and ancestor in ancs

    def all_types(self) -> list[str]:
        return list(self._parents.keys())
