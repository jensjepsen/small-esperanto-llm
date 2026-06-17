"""Extract a query-fast KB subgraph from YAGO 4.5 (Esperanto-only).

YAGO ships triples in tab-separated Turtle. This extractor uses
`yago-taxonomy.ttl` (subclass-of edges) as the source of truth for
classification — no heuristics. An entity is "person" iff *any* of
its rdf:types transitively descends from `schema:Person`; same for
city / country / place. Then a single allowlist of relations is kept.

Three streaming passes:

  Pass 0: load taxonomy. Build subclass-of graph and compute the
          set of types that descend from each target class.

  Pass 1: scan `yago-facts.ttl` once, collecting per-entity rdf:types
          only. After scan, classify each entity by intersecting its
          type set with the target-descendant sets.

  Pass 2: scan `yago-facts.ttl` again, keep facts for classified
          entities filtered to `KEEP_RELATIONS`. Capture EO labels,
          comments, and alternate names along the way.

Output: JSON consumed by `load.py` (same shape as `extract.py`'s
wikidata5m output), so all downstream code is source-agnostic.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator


# Target classes from schema.org / YAGO. An entity is classified into
# bucket B iff at least one of its rdf:types descends transitively
# from `_TARGETS[B]` in the taxonomy. Classifications are ADDITIVE:
# a country is both `lando` and `loko` (since schema:Country descends
# from schema:Place via AdministrativeArea). This keeps `by_type` queries
# faithful to the type hierarchy — "pick any place" returns countries,
# cities, continents, and planets alike.
_TARGETS: dict[str, str] = {
    # People
    "persono":     "schema:Person",

    # Geography — specific subcategories (a country is also `loko`,
    # a planet is also `loko`, etc.)
    "lando":       "schema:Country",
    "urbo":        "schema:City",
    "kontinento":  "schema:Continent",
    "monto":       "yago:Mountain",
    "rivero":      "yago:River",
    "lago":        "yago:Lake",
    "oceano":      "yago:Ocean",
    "maro":        "yago:Sea",
    "insulo":      "yago:Island",
    "planedo":     "yago:Planet",
    "stelo":       "yago:Star",
    # Catch-all
    "loko":        "schema:Place",

    # Organizations
    "kompanio":    "schema:Corporation",
    "lernejo":     "schema:EducationalOrganization",
    "organizo":    "schema:Organization",

    # Creative works
    "libro":       "schema:Book",
    "filmo":       "schema:Movie",
    "muziko":      "schema:MusicComposition",
    "tvserio":     "schema:TVSeries",
    "verko":       "schema:CreativeWork",

    # Intangibles (named things referenced from person/place facts —
    # languages, genders, awards, belief systems). Without these the
    # resolver can't translate fact values like yago:English_language.
    "lingvo":      "schema:Language",
    "sekso":       "yago:Gender",
    "premio":      "yago:Award",
    "kredsistemo": "yago:BeliefSystem",
}


# Relation allowlist. Anything outside is dropped on Pass 2 to keep
# the KB tight. The property name is preserved verbatim (YAGO uses
# `yago:capital`, `schema:birthPlace`, etc.) so cross-source code can
# tell them apart if needed.
_KEEP_RELATIONS: frozenset[str] = frozenset({
    # Geography
    "yago:capital",
    "yago:neighbors",
    "yago:populationNumber",
    "yago:area",
    "yago:demonym",
    "schema:location",
    "schema:containedInPlace",
    "yago:flowsInto",
    "yago:elevation",
    # Person
    "schema:birthDate",
    "schema:birthPlace",
    "schema:deathDate",
    "schema:deathPlace",
    "schema:nationality",
    "schema:gender",
    "schema:knowsLanguage",
    "schema:memberOf",
    "schema:alumniOf",
    "schema:award",
    "schema:worksFor",
    "schema:spouse",
    "schema:children",
    "yago:notableWork",
    # Creative works
    "schema:author",
    "schema:director",
    "schema:musicBy",
    "schema:actor",
    "schema:productionCompany",
    "schema:publisher",
    "schema:inLanguage",
    "schema:dateCreated",
    "schema:material",
    # Organizations
    "schema:founder",
    "schema:foundingDate",
    "schema:foundingLocation",
    "schema:numberOfEmployees",
    "yago:ownedBy",
    # Country/org → language facts (Belgium's official languages, etc.)
    "yago:officialLanguage",
})


def _stream_triples(path: Path, limit: int | None = None) -> Iterator[tuple[str, str, str]]:
    """Yield (s, p, o) tab-tuples from a YAGO ttl file. Skips header
    @prefix lines and blank lines. The trailing ` .` line terminator
    is stripped from the object position."""
    with open(path) as f:
        for i, raw in enumerate(f):
            if limit and i >= limit:
                return
            line = raw.rstrip("\n")
            if not line or line.startswith("@") or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            s, p, o = parts[0], parts[1], parts[2]
            yield s, p, o


def _load_taxonomy(path: Path) -> dict[str, set[str]]:
    """Read all `rdfs:subClassOf` triples and return a parent index
    `parents[type] = {direct parent types}`. Inverse direction
    (children) is computed in `_descendants`."""
    parents: dict[str, set[str]] = defaultdict(set)
    for s, p, o in _stream_triples(path):
        if p == "rdfs:subClassOf":
            parents[s].add(o)
    return parents


def _descendants(root: str, parents: dict[str, set[str]]) -> frozenset[str]:
    """BFS-invert: given `parents[type]→{supers}`, return the set of
    all types whose ancestor chain includes `root` (i.e. all types
    that ARE descendants of `root`, including `root` itself)."""
    children: dict[str, set[str]] = defaultdict(set)
    for child, supers in parents.items():
        for sup in supers:
            children[sup].add(child)
    seen: set[str] = {root}
    frontier: list[str] = [root]
    while frontier:
        nxt: list[str] = []
        for t in frontier:
            for c in children.get(t, ()):
                if c not in seen:
                    seen.add(c)
                    nxt.append(c)
        frontier = nxt
    return frozenset(seen)


def _strip_eo_literal(obj: str) -> str | None:
    """If `obj` is a triple-object of the form `"<text>"@eo`, return
    `<text>` (with surrounding quotes removed). Otherwise return None
    so the caller can skip non-EO literals."""
    if not obj.endswith('"@eo'):
        return None
    # obj is `"<text>"@eo` — strip leading `"` and trailing `"@eo`.
    if len(obj) < 5 or obj[0] != '"':
        return None
    return obj[1:-4]


def _parse_iso_date(obj: str) -> str | None:
    """YAGO dates are `"YYYY-MM-DDT00:00:00Z"^^xsd:dateTime`. Return
    the bare YYYY-MM-DD, or None when the value isn't a parseable
    date (some dates are negative or have month-precision modifiers)."""
    if obj.startswith('"') and "T" in obj:
        try:
            return obj[1:].split("T", 1)[0]
        except (IndexError, ValueError):
            return None
    return None


def _parse_decimal(obj: str) -> str | None:
    """YAGO numbers: `"+11584008"^^xsd:decimal` → return `11584008`."""
    if obj.startswith('"+') or obj.startswith('"-') or (obj.startswith('"') and obj[1:2].isdigit()):
        try:
            return obj.split('"')[1].lstrip("+")
        except IndexError:
            return None
    return None


def extract_yago(
    facts_path: Path,
    taxonomy_path: Path,
    out_path: Path,
    *,
    keep_relations: frozenset[str] = _KEEP_RELATIONS,
    limit: int | None = None,
) -> dict:
    """Run the three-pass extraction; write JSON to `out_path`.

    Output shape (consumed by `load.py`):
      {
        "meta":     {n_entities, types_kept, props_kept, source},
        "entities": {qid: {label, types, facts}, ...},
        "labels":   {label: qid, ...},
      }
    """
    print(f"[yago_kb] Pass 0: loading taxonomy from {taxonomy_path}", file=sys.stderr)
    parents = _load_taxonomy(taxonomy_path)
    # type → human-readable bucket label. An entity's bucket is the
    # FIRST target whose descendant set the type lands in. Order
    # matters only for entities classified into multiple buckets
    # (e.g. a city that's also typed as a polity / country) — those
    # get both labels since `types` is a tuple.
    descendants_of: dict[str, frozenset[str]] = {}
    for label, root in _TARGETS.items():
        descendants_of[label] = _descendants(root, parents)
        print(f"  {label}: {len(descendants_of[label]):,} type descendants of {root}",
              file=sys.stderr)

    print(f"[yago_kb] Pass 1: collecting rdf:types + EO label index from "
          f"{facts_path}", file=sys.stderr)
    entity_types: dict[str, list[str]] = defaultdict(list)
    # Subject → first EO label seen. Indexed for ALL subjects (entities
    # AND type-classes) so we can later look up a YAGO type's Esperanto
    # name (e.g. yago:Cook → "kuiristo") to annotate person occupations.
    eo_label_index: dict[str, str] = {}
    n_seen = 0
    for s, p, o in _stream_triples(facts_path, limit=limit):
        n_seen += 1
        if p == "rdf:type":
            entity_types[s].append(o)
        elif p == "rdfs:label":
            lit = _strip_eo_literal(o)
            if lit and s not in eo_label_index:
                eo_label_index[s] = lit
        if n_seen % 20_000_000 == 0:
            print(f"  scanned {n_seen:,}, {len(entity_types):,} typed, "
                  f"{len(eo_label_index):,} EO-labeled",
                  file=sys.stderr)
    print(f"[yago_kb] Pass 1 done: {len(entity_types):,} typed entities of "
          f"{n_seen:,} triples scanned; {len(eo_label_index):,} EO-labeled subjects",
          file=sys.stderr)

    print("[yago_kb] Classifying entities via taxonomy ancestors...", file=sys.stderr)
    entity_buckets: dict[str, list[str]] = {}  # qid → ["persono", "loko", ...]
    # qid → ordered tuple of EO labels of its rdf:types. Each label is
    # an "eo_tag" — Marie Curie's tags include "fizikisto", "kemiisto",
    # "universitata instruisto". Lets the sampler ground "kuiristo" to
    # an entity actually tagged `kuiristo` (via yago:Cook) instead of
    # any random person.
    entity_eo_tags: dict[str, tuple[str, ...]] = {}
    for qid, types in entity_types.items():
        buckets: list[str] = []
        ts = set(types)
        for bucket, descs in descendants_of.items():
            if ts & descs:
                buckets.append(bucket)
        if buckets:
            entity_buckets[qid] = buckets
            # Collect EO labels for each type; drop dupes, preserve order.
            tags: list[str] = []
            seen: set[str] = set()
            for t in types:
                lab = eo_label_index.get(t)
                if lab and lab not in seen:
                    seen.add(lab)
                    tags.append(lab)
            if tags:
                entity_eo_tags[qid] = tuple(tags)
    bucket_counts: dict[str, int] = defaultdict(int)
    for buckets in entity_buckets.values():
        for b in buckets:
            bucket_counts[b] += 1
    print(f"  classified {len(entity_buckets):,} entities, "
          f"{len(entity_eo_tags):,} with eo_tags", file=sys.stderr)
    for b, c in sorted(bucket_counts.items()):
        print(f"  {b}: {c:,}", file=sys.stderr)

    # Free pass-1 memory before pass 2 — raw types are baked into eo_tags.
    # Keep `eo_label_index`: Pass 2 cross-references it to surface
    # EO-labelled fact-target entities (rdfs:Class definitions, etc.)
    # that didn't land in any of our buckets.
    del entity_types

    print(f"[yago_kb] Pass 2: collecting facts for classified entities",
          file=sys.stderr)
    # Per-entity accumulator. Labels lists are de-duplicated at write
    # time. `facts` keeps lists for ordering parity with insertion;
    # final form is tuple-of-strings.
    Entry = lambda: {  # noqa: E731
        "label": "",
        "alt": [],
        "comment": "",
        "facts": defaultdict(list),
    }
    entities: dict[str, dict] = {}
    # QIDs that show up as fact targets. After Pass 2 we cross-reference
    # against eo_label_index to surface EO-labelled entities that didn't
    # land in any bucket but ARE referenced — typically YAGO `rdfs:Class`
    # definitions like `yago:German_language` that are labelled "germana
    # lingvo" but only appear in facts via their `_generic_instance`
    # singleton.
    referenced_qids: set[str] = set()
    n_seen = 0
    for s, p, o in _stream_triples(facts_path, limit=limit):
        n_seen += 1
        if n_seen % 30_000_000 == 0:
            print(f"  scanned {n_seen:,}, accumulated {len(entities):,} entities",
                  file=sys.stderr)
        if s not in entity_buckets:
            continue
        e = entities.get(s)
        if e is None:
            e = Entry()
            entities[s] = e
        if p == "rdfs:label":
            lit = _strip_eo_literal(o)
            if lit and not e["label"]:
                e["label"] = lit
            elif lit and lit != e["label"]:
                e["alt"].append(lit)
        elif p == "schema:alternateName":
            lit = _strip_eo_literal(o)
            if lit and lit != e["label"]:
                e["alt"].append(lit)
        elif p == "rdfs:comment":
            lit = _strip_eo_literal(o)
            if lit and not e["comment"]:
                e["comment"] = lit
        elif p in keep_relations:
            # For date / decimal literals, extract the bare value; for
            # entity references (start with `yago:` or `wd:`), keep as-is
            # AND record so we can later emit fallback labels for
            # entities not classified into any bucket.
            if o.startswith("yago:") or o.startswith("wd:"):
                e["facts"][p].append(o)
                referenced_qids.add(o)
            else:
                dt = _parse_iso_date(o)
                if dt is not None:
                    e["facts"][p].append(dt)
                    continue
                dec = _parse_decimal(o)
                if dec is not None:
                    e["facts"][p].append(dec)
                    continue
                lit = _strip_eo_literal(o) if o.endswith('"@eo') else None
                if lit:
                    e["facts"][p].append(lit)
    print(f"[yago_kb] Pass 2 done: {n_seen:,} triples, {len(entities):,} "
          f"entities have data", file=sys.stderr)

    print("[yago_kb] Writing JSON...", file=sys.stderr)
    out_entities: dict[str, dict] = {}
    # Multi-label index: every label (canonical + alt) maps to the SET
    # of all entities carrying it. Caller disambiguates by type or by
    # fact-count. Serialized as a sorted list for JSON; loaded back as
    # frozenset.
    labels_index: dict[str, set[str]] = defaultdict(set)
    for qid, e in entities.items():
        if not e["label"]:
            continue  # no EO label → can't surface this entity
        # Dedupe alt labels against the canonical label.
        alt = []
        seen_labels = {e["label"]}
        for a in e["alt"]:
            if a not in seen_labels:
                alt.append(a)
                seen_labels.add(a)
        types = entity_buckets[qid]
        # Convert fact lists to deduped tuples (preserve order).
        facts: dict[str, list[str]] = {}
        for prop, vals in e["facts"].items():
            uniq: list[str] = []
            seen: set[str] = set()
            for v in vals:
                if v not in seen:
                    uniq.append(v)
                    seen.add(v)
            facts[prop] = uniq
        out_entities[qid] = {
            "label": e["label"],
            "alt":   alt,
            "comment": e["comment"],
            "types": types,
            "eo_tags": list(entity_eo_tags.get(qid, ())),
            "facts": facts,
        }
        # Index canonical + every alt — all paths to this entity.
        labels_index[e["label"]].add(qid)
        for a in alt:
            labels_index[a].add(qid)

    # Fallback labels for QIDs referenced from kept facts but not
    # themselves in `out_entities` (typically YAGO `rdfs:Class`
    # definitions: `yago:German_language` carries the EO label "germana
    # lingvo" while only `yago:German_language_generic_instance` shows
    # up as a fact value). Loader merges this into `KB.by_id`-style
    # label resolution so the resolver can translate fact values to
    # EO strings even when the target entity wasn't classified.
    extra_labels: dict[str, str] = {}
    in_entities = set(out_entities)
    for qid in referenced_qids:
        if qid in in_entities:
            continue
        lab = eo_label_index.get(qid)
        if lab is None and qid.endswith("_generic_instance"):
            # The singleton variant rarely has its own EO label; the
            # labelled sibling is the bare-suffix class.
            lab = eo_label_index.get(qid[: -len("_generic_instance")])
        if lab:
            extra_labels[qid] = lab
    print(f"[yago_kb] {len(extra_labels):,} fallback labels for "
          f"unbucketed referenced QIDs", file=sys.stderr)

    doc = {
        "meta": {
            "n_entities": len(out_entities),
            "types_kept": list(_TARGETS.keys()),
            "props_kept": sorted(keep_relations),
            "source":     str(facts_path),
            "taxonomy":   str(taxonomy_path),
        },
        "entities": out_entities,
        # Sorted lists for deterministic JSON output; loader converts
        # back to frozenset.
        "labels":   {k: sorted(v) for k, v in labels_index.items()},
        "extra_labels": extra_labels,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        json.dump(doc, fout, ensure_ascii=False)
    print(f"[yago_kb] Wrote {len(out_entities):,} entities to {out_path}",
          file=sys.stderr)
    return doc
