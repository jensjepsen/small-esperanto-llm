"""Extract Esperanto-labelled factoids from philippesaade/wikidata."""

import json
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress

console = Console()

OUTPUT_DIR = Path("/mnt/data2/wikidata5m/eo_factoids")


def extract_eo_claims(claims: dict | str) -> list[dict]:
    """Extract claims where both property and value have Esperanto labels."""
    if isinstance(claims, str):
        claims = json.loads(claims)

    results = []
    for prop_id, claim_list in claims.items():
        for claim in claim_list:
            ms = claim["mainsnak"]
            prop_labels = ms.get("property-labels", {})
            if "eo" not in prop_labels:
                continue

            prop_eo = prop_labels["eo"]
            dv = ms.get("datavalue", "")

            # Resolve object value
            if isinstance(dv, dict):
                obj_labels = dv.get("labels", {})
                if isinstance(obj_labels, dict):
                    if "eo" in obj_labels:
                        obj_eo = obj_labels["eo"]
                    elif "en" in obj_labels:
                        obj_eo = obj_labels["en"]
                    else:
                        continue
                elif isinstance(obj_labels, str):
                    obj_eo = obj_labels
                else:
                    continue
                obj_id = dv.get("id")
            elif isinstance(dv, str) and dv:
                obj_eo = dv
                obj_id = None
            else:
                continue

            results.append({
                "property": prop_eo,
                "property_id": prop_id,
                "value": obj_eo,
                "value_id": obj_id,
            })

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "eo_factoids.jsonl"

    console.print("[bold green]Loading philippesaade/wikidata (streaming)...")
    ds = load_dataset("philippesaade/wikidata", streaming=True, split="train")

    total_entities = 0
    eo_entities = 0
    total_facts = 0

    with open(output_path, "w") as out:
        with Progress() as progress:
            task = progress.add_task("Extracting...", total=None)

            for ex in ds:
                total_entities += 1

                labels = ex["labels"]
                if isinstance(labels, str):
                    labels = json.loads(labels)

                if "eo" not in labels:
                    if total_entities % 100000 == 0:
                        progress.update(task, description=f"Scanned {total_entities:,} | EO: {eo_entities:,} | Facts: {total_facts:,}")
                    continue

                eo_label = labels["eo"]["value"]
                facts = extract_eo_claims(ex["claims"])

                if not facts:
                    continue

                eo_entities += 1
                total_facts += len(facts)

                record = {
                    "id": ex["id"],
                    "label": eo_label,
                    "facts": facts,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

                if total_entities % 100000 == 0:
                    progress.update(task, description=f"Scanned {total_entities:,} | EO: {eo_entities:,} | Facts: {total_facts:,}")

    console.print(f"[bold]Total entities scanned:[/] {total_entities:,}")
    console.print(f"[bold]Entities with EO label + facts:[/] {eo_entities:,}")
    console.print(f"[bold]Total facts extracted:[/] {total_facts:,}")
    console.print(f"[bold green]Saved to {output_path}")


if __name__ == "__main__":
    main()
