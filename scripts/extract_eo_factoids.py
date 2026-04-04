"""Extract Esperanto-labelled factoids from philippesaade/wikidata."""

import json
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress

console = Console()

OUTPUT_DIR = Path("/mnt/data2/wikidata5m/eo_factoids_v2")


def _parse_time(val: str) -> str | None:
    """Parse '+1879-03-14T00:00:00Z' to '1879-03-14'."""
    if not isinstance(val, str):
        return None
    val = val.lstrip("+")
    if "T" in val:
        val = val.split("T")[0]
    if val.endswith("-00-00"):
        val = val[:-6]
    elif val.endswith("-00"):
        val = val[:-3]
    return val if val else None


def _parse_quantity(dv: dict) -> str | None:
    """Parse a quantity dict to a readable string."""
    amount = dv.get("amount", "")
    if isinstance(amount, str):
        amount = amount.lstrip("+")
    unit = dv.get("unit", "")
    if unit and unit != "1":
        return f"{amount} {unit}"
    return str(amount) if amount else None


def extract_eo_claims(claims: dict | str) -> list[dict]:
    """Extract claims where property has Esperanto label.
    Handles entity references, dates, and quantities."""
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
            datatype = ms.get("datatype", "")

            obj_eo = None
            obj_id = None

            if isinstance(dv, dict):
                # Entity reference — has labels
                obj_labels = dv.get("labels", {})
                if isinstance(obj_labels, dict) and obj_labels:
                    if "eo" in obj_labels:
                        obj_eo = obj_labels["eo"]
                    elif "en" in obj_labels:
                        obj_eo = obj_labels["en"]
                    obj_id = dv.get("id")
                elif isinstance(obj_labels, str) and obj_labels:
                    obj_eo = obj_labels
                    obj_id = dv.get("id")

                # Quantity value
                if obj_eo is None and "amount" in dv:
                    obj_eo = _parse_quantity(dv)

                # Time value
                if obj_eo is None and "time" in dv:
                    obj_eo = _parse_time(dv.get("time", ""))

            elif isinstance(dv, str) and dv:
                if "T00:00:00" in dv:
                    obj_eo = _parse_time(dv)
                else:
                    obj_eo = dv

            if obj_eo is None:
                continue

            results.append({
                "property": prop_eo,
                "property_id": prop_id,
                "value": obj_eo,
                "value_id": obj_id,
                "datatype": datatype,
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

                # Extract Esperanto description if available
                descriptions = ex.get("descriptions", {})
                if isinstance(descriptions, str):
                    descriptions = json.loads(descriptions)
                eo_desc = None
                if isinstance(descriptions, dict) and "eo" in descriptions:
                    d = descriptions["eo"]
                    eo_desc = d["value"] if isinstance(d, dict) else d

                eo_entities += 1
                total_facts += len(facts)

                record = {
                    "id": ex["id"],
                    "label": eo_label,
                    "description": eo_desc,
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
