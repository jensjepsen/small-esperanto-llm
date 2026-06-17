"""Data shapes for the Wikidata KB subgraph.

Read-only after load. Index dicts are populated by `load.py` from the
extracted JSON. All lookups O(1); the indices are sized to the kept
subgraph (~10–20k entities), well under 100 MB RAM.
"""
from __future__ import annotations

from dataclasses import dataclass, field

QID = str  # Wikidata Q-id, e.g. "Q183" for Germanio


@dataclass(frozen=True, slots=True)
class EntityRec:
    """A single KB entity. `types` is the set of broad bucket labels
    (lando, urbo, persono, kontinento, ...). `eo_tags` is the set of
    Esperanto labels of EVERY rdf:type the entity has — Marie Curie's
    eo_tags include "fizikisto", "kemiisto", "universitata instruisto".
    This is what `KB.by_eo_tag` indexes against, letting the sampler
    ground "kuiristo" to an entity actually tagged kuiristo. `facts`
    maps each relation name (property label) to the tuple of target
    values — QIDs for entity-valued facts, bare strings for dates/
    numbers. Both directions of relations are precomputed at load
    time via `KB.reverse`."""
    qid: QID
    label: str                          # Esperanto label, e.g. "Germanio"
    alt:   tuple[str, ...]              # Alternate EO labels
    comment: str                        # One-line EO description (may be empty)
    types: tuple[str, ...]              # Broad buckets: ("lando", "loko")
    eo_tags: tuple[str, ...]            # Type labels: ("fizikisto", "kemiisto")
    facts: dict[str, tuple[str, ...]]   # {"yago:capital": ("yago:Paris",), ...}


@dataclass
class KB:
    """Query-fast subgraph of Wikidata. All indices are O(1) dict
    lookups."""
    by_id:    dict[QID, EntityRec]                    = field(default_factory=dict)
    # Label → set of candidate QIDs. Labels collide ("Jupitero" → both
    # the planet and a Florida town); callers disambiguate by type or
    # by fact count. See `by_name()` / `qids_for_label()`.
    by_label:   dict[str, frozenset[QID]]             = field(default_factory=dict)
    by_type:    dict[str, frozenset[QID]]             = field(default_factory=dict)
    # EO-type-label → entities carrying that tag. Drives concept-aware
    # grounding: `by_eo_tag["kuiristo"]` is all KB persons whose
    # rdf:types include yago:Cook (or any subclass labelled "kuiristo").
    by_eo_tag:  dict[str, frozenset[QID]]             = field(default_factory=dict)
    # QID → EO label for entities that didn't land in any bucket but are
    # referenced from kept facts (e.g. `yago:German_language` for the
    # `parolas_lingvon` fact). Consulted by the resolver as a fallback
    # when `by_id[qid]` has no record.
    extra_labels: dict[QID, str]                      = field(default_factory=dict)
    # forward[(qid, prop)] = entity's facts[prop]; redundant with
    # `by_id[qid].facts[prop]` but kept for symmetry with reverse.
    forward:  dict[tuple[QID, str], tuple[QID, ...]] = field(default_factory=dict)
    reverse:  dict[tuple[QID, str], tuple[QID, ...]] = field(default_factory=dict)

    def get(self, qid: QID) -> EntityRec | None:
        return self.by_id.get(qid)

    def qids_for_label(self, label: str) -> frozenset[QID]:
        """All entities sharing this exact EO label. Empty frozenset
        when unknown. Use this when you intend to disambiguate."""
        return self.by_label.get(label, frozenset())

    def by_name(self, label: str, *, type_filter: str | None = None) -> EntityRec | None:
        """Look up an entity by EO label. If `type_filter` is given,
        return the first candidate whose `types` includes it (lets
        callers say `by_name("Jupitero", type_filter="planedo")`).
        Otherwise picks the candidate with the most facts (rough
        notability proxy). Returns None when no candidate matches."""
        qids = self.by_label.get(label)
        if not qids:
            return None
        recs = [self.by_id[q] for q in qids if q in self.by_id]
        if type_filter:
            recs = [r for r in recs if type_filter in r.types]
        if not recs:
            return None
        return max(recs, key=lambda r: sum(len(v) for v in r.facts.values()))

    def of_type(self, type_label: str) -> frozenset[QID]:
        """All QIDs of the given type. Returns empty frozenset when
        the type is unknown."""
        return self.by_type.get(type_label, frozenset())

    def neighbors(self, qid: QID, prop: str) -> tuple[QID, ...]:
        """Outgoing edges from qid via prop. Empty tuple when none."""
        return self.forward.get((qid, prop), ())

    def reverse_neighbors(self, qid: QID, prop: str) -> tuple[QID, ...]:
        """Incoming edges into qid via prop — e.g.
        reverse_neighbors(Q183, 'lando') returns all entities whose
        `lando` fact points at Germanio (its cities, born-there persons,
        etc.). Empty tuple when none."""
        return self.reverse.get((qid, prop), ())
