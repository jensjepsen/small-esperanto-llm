"""Generate Esperanto factoid training text from extracted Wikidata entities."""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from esperanto_lm.factoids import (
    find_comparable_pairs,
    generate_comparison,
    generate_few_shot_lists,
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
    parser.add_argument("--comparison-variants", type=int, default=5,
                        help="Number of variants per comparison pair")
    parser.add_argument("--few-shot-lists", type=int, default=0,
                        help="Number of few-shot lists (0 = auto-match 25%% of single-entity count)")
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

        # --- Few-shot lists ---
        n_few_shot = args.few_shot_lists if args.few_shot_lists > 0 else single_count // 2
        console.print(f"[bold green]Generating {n_few_shot:,} few-shot lists...")
        lists = generate_few_shot_lists(entities, n_lists=n_few_shot)
        list_count = 0
        for text in lists:
            record = {"text": text, "source": "wikidata_few_shot"}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            list_count += 1

        console.print(f"[bold]Few-shot lists:[/] {list_count:,}")

    console.print()
    console.print(f"[bold]Single-entity paragraphs:[/] {single_count:,}")
    console.print(f"[bold]Comparison paragraphs:[/] {comparison_count:,}")
    console.print(f"[bold]Few-shot lists:[/] {list_count:,}")
    console.print(f"[bold]Total:[/] {single_count + comparison_count + list_count:,}")
    console.print(f"[bold green]Saved to {args.output}")


if __name__ == "__main__":
    main()
