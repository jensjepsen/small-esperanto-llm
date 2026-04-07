"""Generate creative/explanatory SFT pairs using Gemini.

Topics are sampled from Wikidata factoid entities for diversity.
Also generates continuation pairs by splitting responses:
first sentence → prompt, rest → response.
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

console = Console()

DEFAULT_FACTOIDS = Path("/mnt/data2/wikidata5m/eo_factoids_v2/eo_factoids.jsonl")
DEFAULT_OUTPUT = Path("data/sft/sft_creative.jsonl")

# Varied prompt templates
CREATIVE_PROMPTS = [
    "Skribu mallongan rakonton pri {topic}.",
    "Rakontu fabelon pri {topic}.",
    "Skribu poemon pri {topic}.",
    "Priskribu {topic}n en kelkaj frazoj.",
    "Imagu ke vi estas {topic}. Kion vi pensas?",
    "Skribu leteron al {topic}.",
    "Skribu scenon kie {topic} ludas gravan rolon.",
    "Inventi mallongan fabelon pri {topic} por infanoj.",
    "Priskribu {topic}n kvazaŭ vi vidus ĝin la unuan fojon.",
    "Rakontu kiel {topic} ŝanĝis la vivon de iu homo.",
    "Priskribu belan memoron rilatan al {topic}.",
    "Skribu mallongan misteran rakonton pri {topic}.",
    "Kreu ĉarman rakonton pri {topic} kaj amikeco.",
    "Priskribu kiel aspektus la mondo sen {topic}.",
    "Imagu {topic}n en la estonteco. Kiel ĝi aspektos?",
    "Rakontu legendon pri {topic}.",
    "Skribu humuran rakonteton pri {topic}.",
    "Skribu tagan notaĵon de iu kiu tre amas {topic}n.",
    "Skribu konversacion inter infano kaj {topic}.",
    "Kreu dialogon pri {topic}.",
]

EXPLANATION_PROMPTS = [
    "Kio estas {topic}?",
    "Klarigu al infano kio estas {topic}.",
    "Kial {topic} estas grava?",
    "Kiel funkcias {topic}?",
    "Rakontu al mi pri {topic}.",
    "Kion vi scias pri {topic}?",
    "Klarigu {topic}n per simplaj vortoj.",
    "Kio estas la historio de {topic}?",
    "Kial homoj ŝatas {topic}n?",
    "Kio estas interesa pri {topic}?",
    "Kiel oni uzas {topic}n?",
    "Kio okazus sen {topic}?",
    "Priskribu la plej gravajn trajtojn de {topic}.",
    "Kiel {topic} influas nian vivon?",
    "Donu tri interesajn faktojn pri {topic}.",
    "Klarigu kial {topic} estas unika.",
    "Kio surprizas homojn pri {topic}?",
    "Kiel oni malkovris {topic}n?",
    "Komparu {topic}n kun io simila.",
    "Kio estas la diferenco inter {topic} kaj similaj aferoj?",
]

OPINION_PROMPTS = [
    "Ĉu vi ŝatas {topic}n? Kial?",
    "Kio estas via opinio pri {topic}?",
    "Kio estas la plej bela afero pri {topic}?",
    "Rekomandu {topic}n al iu. Kial?",
    "Se vi povus ŝanĝi ion pri {topic}, kion vi ŝanĝus?",
]

HOWTO_PROMPTS = [
    "Kiel oni lernas pri {topic}?",
    "Donu konsilojn pri {topic}.",
    "Kiel komencanto devus alproksimiĝi al {topic}?",
    "Kio estas la plej bona maniero ĝui {topic}n?",
]

ALL_PROMPTS = CREATIVE_PROMPTS + EXPLANATION_PROMPTS + OPINION_PROMPTS + HOWTO_PROMPTS


def load_topics(factoids_path: Path, max_topics: int = 5000) -> list[str]:
    """Load diverse topic names from factoid entities."""
    topics = []
    skip = {"taksono", "vikimedia apartigilo", "vikimedia kategorio",
            "familia nomo", "jaro", "taxonomy template",
            "wikimedia human name disambiguation page",
            "asteroido", "galaksio", "stelo", "supernovao"}

    with open(factoids_path) as f:
        for line in f:
            entity = json.loads(line)
            estas = [fact["value"].lower() for fact in entity["facts"]
                     if fact["property"] == "estas"]
            if any(e in skip for e in estas):
                continue
            label = entity["label"]
            if len(label) > 3 and not label.startswith("Q"):
                topics.append(label)
            if len(topics) >= max_topics * 3:
                break

    random.shuffle(topics)
    return topics[:max_topics]


def generate_batch(client, prompts: list[dict]) -> list[dict]:
    """Generate a batch of SFT pairs using Gemini."""
    request = f"""Generu respondojn en Esperanto por la sekvaj instrukcioj.

Reguloj:
- Skribu NUR en Esperanto
- Uzu ĝustajn supersignojn (ĉ, ĝ, ĥ, ĵ, ŝ, ŭ)
- Uzu ĝustajn akuzativojn (-n), verbformojn (-as, -is, -os), kaj afiksojn
- Respondoj estu 2-5 frazoj longaj
- Estu kreema, interesa kaj natura

Respondu kiel JSON-listo:
{json.dumps(prompts, ensure_ascii=False)}

Plenigu la "assistant" kampojn. Respondu NUR kun la JSON, sen alia teksto."""

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
        pairs = json.loads(text.strip())
        results = []
        for p in pairs:
            if p.get("assistant") and p["assistant"] != "..." and len(p["assistant"]) > 10:
                results.append({
                    "messages": [
                        {"role": "user", "content": p["user"]},
                        {"role": "assistant", "content": p["assistant"]},
                    ]
                })
        return results
    except (json.JSONDecodeError, KeyError) as e:
        console.print(f"[red]Parse error: {e}")
        return []


def split_continuation(pair: dict) -> dict | None:
    """Split a creative response into a continuation prompt."""
    text = pair["messages"][1]["content"]

    match = re.search(r'[.!?]\s+', text)
    if not match or match.end() > len(text) * 0.7:
        return None

    first = text[:match.end()].strip()
    rest = text[match.end():].strip()

    if len(rest) < 20:
        return None

    return {
        "messages": [
            {"role": "user", "content": f"Daŭrigu la tekston: {first}"},
            {"role": "assistant", "content": rest},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Generate creative SFT data via Gemini")
    parser.add_argument("--factoids", type=Path, default=DEFAULT_FACTOIDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    from google import genai
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print("[red]Set GOOGLE_API_KEY or pass --api-key")
        return
    client = genai.Client(api_key=api_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]Loading topics from factoids...")
    topics = load_topics(args.factoids)
    console.print(f"[bold]Topics loaded:[/] {len(topics):,}")

    total_creative = 0
    total_continuation = 0

    with open(args.output, "w") as out:
        with Progress() as progress:
            task = progress.add_task("Generating...", total=args.num_batches)

            for batch_idx in range(args.num_batches):
                prompts = []
                for _ in range(args.batch_size):
                    topic = random.choice(topics)
                    tmpl = random.choice(ALL_PROMPTS)
                    prompts.append({
                        "user": tmpl.format(topic=topic),
                        "assistant": "...",
                    })

                try:
                    pairs = generate_batch(client, prompts)
                    for pair in pairs:
                        out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                        total_creative += 1

                        cont = split_continuation(pair)
                        if cont:
                            out.write(json.dumps(cont, ensure_ascii=False) + "\n")
                            total_continuation += 1
                except Exception as e:
                    console.print(f"[red]Batch {batch_idx} failed: {e}")

                progress.advance(task)
                time.sleep(0.5)

    console.print()
    console.print(f"[bold]Creative/explanation:[/] {total_creative:,}")
    console.print(f"[bold]Continuations:[/] {total_continuation:,}")
    console.print(f"[bold]Total:[/] {total_creative + total_continuation:,}")
    console.print(f"[bold green]Saved to {args.output}")

    console.print("\n[bold]Samples:")
    with open(args.output) as f:
        lines = f.readlines()
        for line in random.sample(lines, min(5, len(lines))):
            pair = json.loads(line)
            console.print(f"  Q: {pair['messages'][0]['content']}")
            console.print(f"  A: {pair['messages'][1]['content'][:150]}...")
            console.print()


if __name__ == "__main__":
    main()
