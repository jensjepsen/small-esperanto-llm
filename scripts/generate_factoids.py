"""Generate Esperanto factoid training text from extracted Wikidata entities."""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from esperanto_lm.factoids import (
    find_comparable_pairs,
    generate_comparison,
    generate_variants,
)

console = Console()

DEFAULT_INPUT = Path("/mnt/data2/wikidata5m/eo_factoids/eo_factoids.jsonl")
DEFAULT_OUTPUT = Path("/mnt/data2/wikidata5m/eo_factoids/factoid_text.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Generate Esperanto factoid text")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", type=int, default=3,
                        help="Number of paragraph variants per entity")
    parser.add_argument("--comparison-samples", type=int, default=10000,
                        help="Number of comparison pairs to sample")
    parser.add_argument("--comparison-variants", type=int, default=2,
                        help="Number of variants per comparison pair")
    args = parser.parse_args()

    console.print(f"[bold green]Reading entities from {args.input}")

    entities = []
    with open(args.input) as f:
        for line in f:
            entities.append(json.loads(line))

    console.print(f"[bold]Entities loaded:[/] {len(entities):,}")

    single_count = 0
    comparison_count = 0

    with open(args.output, "w") as out:
        # --- Single-entity factoids ---
        console.print("[bold green]Generating single-entity factoids...")
        with Progress() as progress:
            task = progress.add_task("Single-entity...", total=len(entities))

            for entity in entities:
                paragraphs = generate_variants(
                    entity["label"], entity["facts"],
                    n_variants=args.variants,
                )
                for text in paragraphs:
                    record = {"text": text, "source": "wikidata_factoid",
                              "entity_id": entity["id"]}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    single_count += 1

                progress.advance(task)

        console.print(f"[bold]Single-entity paragraphs:[/] {single_count:,}")

        # --- Cross-entity comparisons ---
        console.print("[bold green]Finding comparable entity pairs...")
        pairs = find_comparable_pairs(entities)
        console.print(f"[bold]Comparable pairs found:[/] {len(pairs):,}")

        console.print("[bold green]Generating comparisons...")
        with Progress() as progress:
            task = progress.add_task("Comparisons...", total=len(pairs))

            for entity_a, entity_b, shared_props in pairs:
                for _ in range(args.comparison_variants):
                    text = generate_comparison(entity_a, entity_b, shared_props)
                    if text:
                        record = {
                            "text": text,
                            "source": "wikidata_comparison",
                            "entity_ids": [entity_a["id"], entity_b["id"]],
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        comparison_count += 1

                progress.advance(task)

    console.print()
    console.print(f"[bold]Single-entity paragraphs:[/] {single_count:,}")
    console.print(f"[bold]Comparison paragraphs:[/] {comparison_count:,}")
    console.print(f"[bold]Total:[/] {single_count + comparison_count:,}")
    console.print(f"[bold green]Saved to {args.output}")


if __name__ == "__main__":
    main()
