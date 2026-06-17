"""Load the extracted KB JSON into a queryable `KB` object.

One-shot load at startup; all lookups are O(1) dict access after.
Computes the reverse-relation index alongside the forward one so
"who has X as their country?" queries are direct lookups.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .schema import KB, EntityRec, QID


def load_kb(path: Path | str) -> KB:
    """Read the JSON written by `extract.py` and build all indices."""
    with open(path) as f:
        doc = json.load(f)
    raw = doc["entities"]
    by_id: dict[QID, EntityRec] = {}
    forward: dict[tuple[QID, str], tuple[QID, ...]] = {}
    reverse_acc: dict[tuple[QID, str], list[QID]] = defaultdict(list)
    type_acc: dict[str, list[QID]] = defaultdict(list)
    # EO-tag inverted index. For each EO label of an rdf:type, the set
    # of entities carrying it. Drives concept-name grounding via
    # `KB.by_eo_tag["kuiristo"]` etc.
    eo_tag_acc: dict[str, list[QID]] = defaultdict(list)
    for qid, entry in raw.items():
        facts = {
            prop: tuple(targets)
            for prop, targets in entry.get("facts", {}).items()
        }
        for t in entry.get("types", ()):
            type_acc[t].append(qid)
        for tag in entry.get("eo_tags", ()):
            eo_tag_acc[tag].append(qid)
        for prop, targets in facts.items():
            forward[(qid, prop)] = targets
            for tgt in targets:
                reverse_acc[(tgt, prop)].append(qid)
        by_id[qid] = EntityRec(
            qid=qid,
            label=entry.get("label", ""),
            alt=tuple(entry.get("alt", ())),
            comment=entry.get("comment", ""),
            types=tuple(entry.get("types", ())),
            eo_tags=tuple(entry.get("eo_tags", ())),
            facts=facts,
        )
    # `labels` is expected to be {label: [qid, ...]} in YAGO output.
    # Legacy wikidata5m output had {label: qid}; wrap single strings
    # in a singleton frozenset for backwards compatibility.
    raw_labels = doc.get("labels", {})
    by_label: dict[str, frozenset[str]] = {}
    for label, value in raw_labels.items():
        if isinstance(value, str):
            by_label[label] = frozenset({value})
        else:
            by_label[label] = frozenset(value)
    return KB(
        by_id=by_id,
        by_label=by_label,
        by_type={k: frozenset(v) for k, v in type_acc.items()},
        by_eo_tag={k: frozenset(v) for k, v in eo_tag_acc.items()},
        extra_labels=dict(doc.get("extra_labels", {})),
        forward=forward,
        reverse={k: tuple(v) for k, v in reverse_acc.items()},
    )
