"""Translate GSM8K math problems to Esperanto using Gemini."""

import argparse
import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress

console = Console()

DEFAULT_OUTPUT_DIR = Path("data/sft/gsm8k")


def translate_batch(client, items: list[dict]) -> list[dict]:
    """Translate a batch of GSM8K Q&A pairs to Esperanto."""
    request = f"""Traduku la sekvajn matematikajn problemojn kaj respondojn al Esperanto.

Reguloj:
- Traduku NUR al Esperanto
- Uzu ĝustajn supersignojn (ĉ, ĝ, ĥ, ĵ, ŝ, ŭ)
- Konservu la nomojn de personoj (ne traduku nomojn)
- Konservu la kalkulojn kaj nombrojn ekzakte kiel ili estas
- Konservu la formaton: ĉiu paŝo en nova linio, fina respondo post ####
- Uzu ĝustajn akuzativojn (-n) kaj verbformojn

Respondu kiel JSON-listo:
{json.dumps(items, ensure_ascii=False)}

Plenigu la "question_eo" kaj "answer_eo" kampojn. Respondu NUR kun la JSON, sen alia teksto."""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=request,
    )
    text = response.text.strip()

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            if part.startswith("json"):
                text = part[4:]
                break
            elif part.strip().startswith("["):
                text = part
                break

    try:
        results = json.loads(text.strip())
        translated = []
        for r in results:
            q_eo = r.get("question_eo", "")
            a_eo = r.get("answer_eo", "")
            if q_eo and a_eo and len(q_eo) > 10:
                translated.append({
                    "messages": [
                        {"role": "user", "content": q_eo},
                        {"role": "assistant", "content": a_eo},
                    ]
                })
        return translated
    except (json.JSONDecodeError, KeyError) as e:
        console.print(f"[red]Parse error: {e}")
        return []


def translate_split(client, items: list, output_path: Path, batch_size: int):
    """Translate a dataset split, resuming from existing progress."""
    existing = 0
    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for _ in f)
        if existing > 0:
            console.print(f"[bold]  Resuming from:[/] {existing:,} already done")

    skip_items = existing * batch_size
    total_translated = existing

    with open(output_path, "a") as out:
        with Progress() as progress:
            task = progress.add_task("Translating...", total=len(items))
            progress.advance(task, min(skip_items, len(items)))

            for batch_start in range(skip_items, len(items), batch_size):
                batch = items[batch_start:batch_start + batch_size]

                batch_input = [{
                    "question": item["question"],
                    "answer": item["answer"],
                    "question_eo": "...",
                    "answer_eo": "...",
                } for item in batch]

                for attempt in range(5):
                    try:
                        translated = translate_batch(client, batch_input)
                        for pair in translated:
                            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                            total_translated += 1
                        break
                    except Exception as e:
                        wait = 2 ** attempt
                        console.print(f"[red]Batch at {batch_start} attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                        time.sleep(wait)

                progress.advance(task, len(batch))

    return total_translated


def main():
    parser = argparse.ArgumentParser(description="Translate GSM8K to Esperanto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=None,
                        help="Max items per split (default: all)")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    from google import genai
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print("[red]Set GOOGLE_API_KEY or pass --api-key")
        return
    client = genai.Client(api_key=api_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        console.print(f"\n[bold green]Loading GSM8K {split} split...")
        ds = load_dataset("openai/gsm8k", "main", split=split)
        items = list(ds)
        if args.max_items:
            items = items[:args.max_items]
        console.print(f"[bold]  Items:[/] {len(items):,}")

        output_path = args.output_dir / f"{split}.jsonl"
        count = translate_split(client, items, output_path, args.batch_size)
        console.print(f"[bold]  Translated:[/] {count:,} → {output_path}")

    # Show samples
    console.print("\n[bold]Samples from train:")
    import random
    with open(args.output_dir / "train.jsonl") as f:
        lines = f.readlines()
        for line in random.sample(lines, min(3, len(lines))):
            pair = json.loads(line)
            console.print(f"  Q: {pair['messages'][0]['content']}")
            console.print(f"  A: {pair['messages'][1]['content'][:200]}...")
            console.print()


if __name__ == "__main__":
    main()
