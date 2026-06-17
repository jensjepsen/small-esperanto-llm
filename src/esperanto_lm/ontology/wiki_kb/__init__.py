"""Wikidata-derived factual knowledge base.

A small, query-fast subgraph of Wikidata in Esperanto, intended to seed
named-entity locations (countries, continents, cities), notable people,
and the relations between them during scene construction and ICL
generation. Distinct from `Lexicon` (rules + grammar) — this is
factual knowledge.

Sources: `/mnt/data2/wikidata5m/eo_factoids/eo_factoids.jsonl` —
the raw extraction with value_ids preserved (not the rendered
factoid_text.jsonl which drops the graph).

See `extract.py` for the extraction pipeline (run via
`scripts/extract_wiki_kb.py`) and `schema.py` for the data shape.
`load.py` reads the resulting JSON into the queryable `KB` object."""
from .schema import KB, EntityRec, QID
from .load import load_kb

__all__ = ["KB", "EntityRec", "QID", "load_kb"]
