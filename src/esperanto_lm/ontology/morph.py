"""Morphology adapter.

The ontology only needs a thin slice of the morphological parser:
  - the bare root of a word
  - the affixes attached (prefixes, suffixes)
  - the final ending (-i, -o, -as, ...) if any

We define a `MorphParser` Protocol so the rest of the ontology never imports
the underlying decomposer directly. This makes the lexicon testable without
loading the full root dictionary, and lets us swap in a stub for a tiny
fixture set if the real parser misbehaves on a specific lemma.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class MorphParse:
    root: str
    prefixes: tuple[str, ...] = ()
    suffixes: tuple[str, ...] = ()
    ending: str | None = None
    # Original word, kept for debugging.
    word: str = ""


class MorphParser(Protocol):
    def parse(self, word: str) -> MorphParse: ...


class DefaultMorphParser:
    """Wraps `esperanto_lm.morphology.decompose_tagged`.

    Returns the *first* root encountered. Compound words yield the first root
    only — the rest is discarded for now since this slice doesn't need
    compound-aware composition.
    """

    def parse(self, word: str) -> MorphParse:
        # Local import: avoid loading the root dictionary at package import.
        from esperanto_lm.morphology import decompose_tagged
        tagged = decompose_tagged(word)
        prefixes: list[str] = []
        suffixes: list[str] = []
        ending: str | None = None
        root: str | None = None
        for m, t in tagged:
            if t == "prefix":
                prefixes.append(m)
            elif t == "root":
                if root is None:
                    root = m
                # discard further roots (compounds) for this slice
            elif t == "suffix":
                suffixes.append(m)
            elif t == "ending":
                ending = m
            elif t == "particle":
                # treat as a bare root for invariant words
                if root is None:
                    root = m
        return MorphParse(
            root=root or word,
            prefixes=tuple(prefixes),
            suffixes=tuple(suffixes),
            ending=ending,
            word=word,
        )


@dataclass
class StubMorphParser:
    """Lookup-table fallback for tests / for cases where the default parser
    misclassifies a known lemma. Pass a dict of {word: MorphParse}; words
    not in the table fall through to a default parser if provided, else
    raise.
    """
    table: dict[str, MorphParse] = field(default_factory=dict)
    fallback: MorphParser | None = None

    def parse(self, word: str) -> MorphParse:
        if word in self.table:
            return self.table[word]
        if self.fallback is not None:
            return self.fallback.parse(word)
        raise KeyError(f"StubMorphParser: no entry for {word!r}")
