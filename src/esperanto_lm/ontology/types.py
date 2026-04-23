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

        Unknown types raise; we want load-time errors, not silent False.
        """
        if t not in self._ancestors:
            raise KeyError(f"unknown type {t!r}")
        if ancestor not in self._parents:
            raise KeyError(f"unknown ancestor type {ancestor!r}")
        return ancestor in self._ancestors[t]

    def all_types(self) -> list[str]:
        return list(self._parents.keys())
