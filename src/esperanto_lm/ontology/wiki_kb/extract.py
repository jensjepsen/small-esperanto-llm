"""Extract a small queryable subgraph from the raw wikidata factoid
dump (`eo_factoids.jsonl`). Two-pass streaming algorithm:

  Pass 1: scan every record. Identify entities whose `estas` (P31)
          value_id lands in `TYPE_QIDS`. Keep these in the working set
          and stash their facts where the value_id wasn't filtered.

  Pass 2: re-scan once to resolve value_id → label for any reference
          whose target wasn't kept in pass 1 (rare — most type-of and
          location-of targets are themselves "kept" types so they're
          available from pass 1 already).

Output: a single JSON file consumed by `load.py` at runtime.

Configurable via TYPE_QIDS / KEEP_PROPS at the top. Add more Q-ids
to the type filter to widen the subgraph.

Run via `scripts/extract_wiki_kb.py`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


# Q-IDs of entity types we want in the KB. The `estas` (instance-of)
# fact's value_id is checked against this set; entities with at least
# one matching type are kept. Add more here to widen coverage.
TYPE_QIDS: dict[str, str] = {
    # Geography
    "Q6256":    "lando",        # country
    "Q5107":    "kontinento",   # continent
    "Q515":     "urbo",         # city
    "Q5119":    "ĉefurbo",      # capital
    "Q4022":    "rivero",       # river
    "Q23397":   "lago",         # lake
    "Q8502":    "monto",        # mountain
    "Q39594":   "fjordo",       # fjord
    "Q1145276": "oceano",       # ocean
    # Persons (single Q5 captures all humans — large but bounded by
    # the EO-label filter from the source dump).
    "Q5":       "persono",      # human
    # Notable things
    "Q11424":   "filmo",        # film
    "Q571":     "libro",        # book
    "Q34770":   "lingvo",       # language
}


# Property IDs (P-ids) to retain on kept entities. The Esperanto
# property name from the dump is preserved as-is for the relation
# label in the KB.
KEEP_PROPS: frozenset[str] = frozenset({
    # Geo: country/continent/location structure
    "P30",   # kontinento (continent)
    "P17",   # lando (country)
    "P36",   # ĉefurbo (capital)
    "P361",  # parto de (part of)
    "P131",  # situas en (admin location)
    "P276",  # situas (located in)
    # Person facts
    "P19",   # naskiĝloko (birthplace)
    "P20",   # mortejo (place of death)
    "P27",   # ŝtataneco (citizenship)
    "P106",  # okupo (occupation)
    "P40",   # infano (child)
    "P22",   # patro
    "P25",   # patrino
    "P800",  # grava verko
    # Naming / etymology
    "P138",  # nomita laŭ (named after)
    # Type — kept always (used for filtering)
    "P31",   # estas (instance of)
})


def _types_of(facts: list[dict]) -> list[str]:
    """Return the human-readable types this entity claims via P31."""
    out = []
    for f in facts:
        if f.get("property_id") != "P31":
            continue
        type_qid = f.get("value_id")
        if type_qid in TYPE_QIDS:
            out.append(TYPE_QIDS[type_qid])
    return out


def _kept_facts(
    facts: list[dict], kept_qids: set[str] | None = None,
) -> dict[str, list[str]]:
    """Filter facts to `KEEP_PROPS` and only those whose value_id is in
    `kept_qids` (when given). The `estas` fact is special-cased: its
    value (a type label) is stored under the synthetic key "estas" so
    types are recoverable from facts too.

    Returns relation_label → list[value_qid]. Entries with no
    surviving value_id are dropped to keep the KB graph-shaped."""
    out: dict[str, list[str]] = defaultdict(list)
    for f in facts:
        pid = f.get("property_id")
        if pid not in KEEP_PROPS:
            continue
        vid = f.get("value_id")
        if not vid:
            continue
        if kept_qids is not None and vid not in kept_qids:
            continue
        prop = f.get("property") or pid
        out[prop].append(vid)
    return dict(out)


def extract(
    source: Path, out_path: Path, *,
    type_qids: dict[str, str] | None = None,
    keep_props: frozenset[str] | None = None,
    limit: int | None = None,
) -> dict:
    """Two-pass streaming extraction. Writes a JSON document to
    `out_path` shaped:

      {"entities": {qid: {label, types, facts}, ...},
       "labels":   {label: qid, ...},
       "meta":     {n_entities, types_kept, props_kept, source}}

    Idempotent; same input + config produces byte-identical output.
    """
    if type_qids is None:
        type_qids = TYPE_QIDS
    if keep_props is None:
        keep_props = KEEP_PROPS

    # Pass 1: collect entities that match the type filter; stash their
    # raw facts to resolve in pass 2.
    print(f"[wiki_kb] Pass 1: scanning {source} for typed entities...")
    kept: dict[str, dict] = {}  # qid → {label, types, raw_facts}
    n_seen = 0
    with open(source) as fin:
        for line in fin:
            n_seen += 1
            if limit and n_seen > limit:
                break
            rec = json.loads(line)
            qid = rec.get("id")
            if not qid:
                continue
            facts = rec.get("facts", [])
            types = _types_of(facts)
            if not types:
                continue
            kept[qid] = {
                "label": rec.get("label", ""),
                "types": types,
                "raw_facts": facts,
            }
            if len(kept) % 5000 == 0:
                print(f"  kept {len(kept)} of {n_seen} scanned")
    print(f"[wiki_kb] Pass 1 done: {len(kept)} entities of "
          f"{n_seen} total")

    # Pass 2 (here: just filter facts to point at kept value_ids only).
    # No second file scan needed because every kept entity's facts
    # were captured in pass 1 — we just drop value_ids whose entity
    # we didn't keep.
    print("[wiki_kb] Pass 2: filtering facts to in-KB targets...")
    kept_set = set(kept.keys())
    entities: dict[str, dict] = {}
    for qid, entry in kept.items():
        filtered = _kept_facts(entry["raw_facts"], kept_qids=kept_set)
        entities[qid] = {
            "label": entry["label"],
            "types": entry["types"],
            "facts": filtered,
        }

    # Label index — first occurrence wins on collision (rare; most
    # Esperanto labels are unique).
    labels: dict[str, str] = {}
    for qid, e in entities.items():
        if e["label"] and e["label"] not in labels:
            labels[e["label"]] = qid

    doc = {
        "meta": {
            "n_entities": len(entities),
            "types_kept": list(type_qids.values()),
            "props_kept": sorted(keep_props),
            "source": str(source),
        },
        "entities": entities,
        "labels": labels,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        json.dump(doc, fout, ensure_ascii=False, indent=None)
    print(f"[wiki_kb] Wrote {len(entities)} entities to {out_path}")
    return doc
