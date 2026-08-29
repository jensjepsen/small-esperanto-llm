"""Build text→question (T→Q) training data by inverting existing Danish
context+question SFT rows. No new LLM calls needed — we already have the
raw material in several datasets.

Sources with (context, question, answer) triples we can invert:
  - jensjepsen/danish-wiki-grounded-sft-v2 (default) — closed_qa +
    information_extraction categories
  - jensjepsen/danish-sciq (default) — rows with non-empty da_support

For each triple, emit an SFT row where user provides text and asks for a
question, assistant emits the question. Multiple user-turn phrasings are
sampled per row to widen the surface pattern the model sees.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

# Rotated user-turn templates so the model sees varied phrasings for the
# same underlying task.
TEMPLATES = [
    "Læs teksten og skriv ét spørgsmål, den kan besvare:\n\n{text}",
    "Tekst:\n{text}\n\nSkriv ét spørgsmål der kan besvares ud fra teksten.",
    "{text}\n\nHvilket spørgsmål kan man stille om teksten?",
    "{text}\n\nStil et spørgsmål om indholdet.",
    "Baseret på teksten nedenfor, skriv ét spørgsmål:\n\n{text}",
    "Læs:\n{text}\n\nGenerér et spørgsmål ud fra teksten.",
    "{text}\n\nSkriv et enkelt spørgsmål der kan besvares ud fra teksten.",
    "Følgende tekst indeholder en information:\n{text}\n\nSkriv et spørgsmål om den.",
]


def messages_row(context: str, question: str, rng: random.Random,
                 source: str) -> dict:
    tmpl = rng.choice(TEMPLATES)
    return {
        "messages": [
            {"role": "user",      "content": tmpl.format(text=context.strip())},
            {"role": "assistant", "content": question.strip()},
        ],
        "source": source,
    }


def from_wiki_grounded(rng: random.Random) -> list[dict]:
    """closed_qa + information_extraction rows have all three fields.
    Uses v3 (95k) — twice the pool of v2 (50k)."""
    ds = load_dataset("jensjepsen/danish-wiki-grounded-sft-v3",
                       "default", split="train")
    out = []
    kept_cats = {"closed_qa", "information_extraction"}
    for r in ds:
        if r.get("category") not in kept_cats:
            continue
        ctx = (r.get("context") or "").strip()
        q   = (r.get("instruction") or "").strip()
        if len(ctx) < 60 or len(q) < 8:
            continue
        out.append(messages_row(ctx, q, rng,
                                 source=f"wiki-grounded-v2:{r['category']}"))
    return out


def from_sciq(rng: random.Random) -> list[dict]:
    """SciQ has da_support (context) and da_question. Support is sometimes
    empty — skip those. Use train + validation splits (test held out for
    eval)."""
    out = []
    for split in ("train", "validation"):
        ds = load_dataset("jensjepsen/danish-sciq", "default", split=split)
        for r in ds:
            ctx = (r.get("da_support") or "").strip()
            q   = (r.get("da_question") or "").strip()
            if len(ctx) < 60 or len(q) < 8:
                continue
            out.append(messages_row(ctx, q, rng, source="sciq:" + split))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("harvesting wiki-grounded-v2 closed_qa + info_extraction…", flush=True)
    wiki = from_wiki_grounded(rng)
    print(f"  {len(wiki):,} rows", flush=True)

    print("harvesting sciq train + validation…", flush=True)
    sci = from_sciq(rng)
    print(f"  {len(sci):,} rows", flush=True)

    all_rows = wiki + sci
    rng.shuffle(all_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(all_rows):,} rows → {args.out}")
    print(f"source breakdown: {dict(Counter(r['source'] for r in all_rows))}")


if __name__ == "__main__":
    main()
