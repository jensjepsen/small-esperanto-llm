"""Facts disclosed in rendered prose.

A `Fact` is a structured record of something the prose tells the reader.
Each render function returns its facts alongside the text it emits, so
downstream consumers (Q/A generators, eval) can know exactly which
propositions appeared in the narrative.

Fact kinds:
  intro       — entity first introduced ("En la kuirejo estas pano.")
  event       — event narrated ("Petro fermis la pordon.")
  state       — slot value disclosed predicatively ("La pordo estas
                fermita.") or attributively ("la fermita pordo")
  relation    — relation rendered as setup/grouped ("Sur la tablo estas
                fromaĝo.") or as a specifier ("la libro de Petro")
  definition  — definitional sentence ("Petro estas kuracisto.")
  appearance  — entity creation announced ("Aperis vitropecetoj.")
  destruction — entity destruction ("La glaso malaperis.")
  relation_removed — relation no longer holds ("La pordo malfermiĝis.")
  relation_added   — new relation asserted in narrative
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Fact:
    kind: str
    payload: tuple[tuple[str, Any], ...]

    @classmethod
    def make(cls, kind: str, **payload) -> "Fact":
        return cls(kind=kind, payload=tuple(sorted(payload.items())))

    def get(self, key: str, default: Any = None) -> Any:
        for k, v in self.payload:
            if k == key:
                return v
        return default

    def as_dict(self) -> dict:
        return {"kind": self.kind, **dict(self.payload)}


# A render result is the prose substring plus the facts that rendering
# this substring disclosed to the reader.
RenderResult = tuple[str, list[Fact]]
