"""Build the wiki KB subgraph from the raw EO factoid dump.

Reads `eo_factoids.jsonl` (raw form with value_ids preserved) and
emits a small JSON of the configured type/property subgraph.

Default config (in src/esperanto_lm/ontology/wiki_kb/extract.py)
keeps countries / continents / cities / persons / rivers / lakes /
mountains plus geography & biography relations. Tune TYPE_QIDS /
KEEP_PROPS there to widen or narrow coverage.

Usage:
  python scripts/extract_wiki_kb.py \\
    --source /mnt/data2/wikidata5m/eo_factoids/eo_factoids.jsonl \\
    --out src/esperanto_lm/ontology/wiki_kb/data/kb.json
"""
import argparse
from pathlib import Path

from esperanto_lm.ontology.wiki_kb.extract import extract


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source", type=Path,
        default=Path("/mnt/data2/wikidata5m/eo_factoids/eo_factoids.jsonl"),
        help="Raw EO factoid jsonl (with value_ids).")
    p.add_argument(
        "--out", type=Path,
        default=Path("src/esperanto_lm/ontology/wiki_kb/data/kb.json"),
        help="Output JSON path.")
    p.add_argument(
        "--limit", type=int, default=0,
        help="Cap source records scanned (0 = all). For dev runs.")
    args = p.parse_args()
    extract(
        source=args.source, out_path=args.out,
        limit=args.limit or None,
    )


if __name__ == "__main__":
    main()
