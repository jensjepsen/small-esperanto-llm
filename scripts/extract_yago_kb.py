"""Extract YAGO 4.5 → query-fast EO KB (countries, cities, persons).

Uses the taxonomy (`yago-taxonomy.ttl`) as the source of truth for
classification — no heuristics. See `extract_yago.extract_yago`
docstring for the three-pass algorithm.

Usage:
  python scripts/extract_yago_kb.py \\
    --facts /mnt/data2/yago4.5/extracted/yago_eo_facts.ttl \\
    --taxonomy /mnt/data2/yago4.5/extracted/yago-taxonomy.ttl \\
    --out src/esperanto_lm/ontology/wiki_kb/data/yago_kb.json
"""
import argparse
from pathlib import Path

from esperanto_lm.ontology.wiki_kb.extract_yago import extract_yago


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--facts", type=Path,
        default=Path("/mnt/data2/yago4.5/extracted/yago_eo_facts.ttl"),
        help="EO-filtered YAGO facts (pre-extracted via unzip|grep|filter).")
    p.add_argument(
        "--taxonomy", type=Path,
        default=Path("/mnt/data2/yago4.5/extracted/yago-taxonomy.ttl"),
        help="YAGO subclass-of taxonomy (small, 13MB).")
    p.add_argument(
        "--out", type=Path,
        default=Path("src/esperanto_lm/ontology/wiki_kb/data/yago_kb.json"),
        help="Output JSON path.")
    p.add_argument(
        "--limit", type=int, default=0,
        help="Cap fact lines scanned per pass (0 = all). For dev runs.")
    args = p.parse_args()
    extract_yago(
        facts_path=args.facts,
        taxonomy_path=args.taxonomy,
        out_path=args.out,
        limit=args.limit or None,
    )


if __name__ == "__main__":
    main()
