"""Smoke closed_qa density generation on a few salience-ranked articles.

For each article, generate N closed_qa items (grounded in article intro).
Uses gemma-3-12b via OpenRouter with a slightly modified prompt that asks
for N distinct question/answer pairs at once (cheaper than N separate
calls, and Gemma naturally spreads across different facts in the text).

Usage:
    uv run python scripts/smoke_wiki_closedqa_dense.py \\
        --salience /mnt/data2/da_wiki_curation/salience.tsv \\
        --n-articles 6 --n-qa-per 6 --tier T1_universal
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import aiohttp
from datasets import load_dataset

API = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemma-3-12b-it"

DENSE_PROMPT = """Du får en dansk Wikipedia-artikel og skal generere {n_qa}
forskellige spørgsmål-og-svar-par, som er GRUNDET i artiklen — dvs. hvert
svar SKAL kunne udledes direkte af teksten.

TITEL: {title}

ARTIKEL:
{source}

REGLER:

STAND-ALONE-KRAV (VIGTIGST):
Hvert spørgsmål SKAL kunne stilles til én, der ALDRIG har set artiklen. Det
betyder at spørgsmålet IKKE må referere til:
  - artiklen selv ("i artiklen", "i teksten", "ifølge kilden", "her nævnes")
  - Wikipedia
  - "denne artikel", "ovenstående", "nedenstående"
  - "omtales som", "beskrives som"  (når det henviser til at noget står i teksten)
  - noget kontekst-ord som "ifølge det ovenfor", "det følgende"

FORBUDT EKSEMPEL (må IKKE produceres):
  ❌ Q: "Hvilket sprog omtales som 'nynorsk' i artiklen?"
  ❌ Q: "Ifølge teksten, hvornår blev landet grundlagt?"
  ❌ Q: "Hvad står der om landbrug i artiklen?"

GODT EKSEMPEL (fri-stående faktaspørgsmål):
  ✓ Q: "Hvad hedder Norges andet officielle skriftsprog udover bokmål?"
  ✓ Q: "Hvornår blev Norge selvstændigt?"
  ✓ Q: "Hvilken erhvervsgren er dominerende i Sydsverige?"

ØVRIGE REGLER:
1. Alle {n_qa} spørgsmål skal handle om FORSKELLIGE fakta fra artiklen.
2. Ingen spørgsmål må være for generiske ("Hvad handler emnet om?").
3. Alle svar skal være konkrete og korrekte ifølge artiklen.
4. Blandede spørgsmålstyper: hvornår/hvem/hvor/hvordan/hvorfor/hvad.
5. Hvis du ikke kan formulere spørgsmålet fri-stående uden at referere til
   teksten, så SPRING det faktum over og vælg et andet.

OUTPUT: én JSON-liste med præcis {n_qa} objekter:
[
  {{"q": "...", "a": "..."}},
  {{"q": "...", "a": "..."}}
]

Kun JSON, ingen kommentarer, ingen markdown-fences."""


def clean_intro(text: str, max_chars: int = 1200) -> str:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out, total = [], 0
    for p in paras:
        if total + len(p) > max_chars:
            break
        out.append(p)
        total += len(p) + 1
    return "\n\n".join(out)


def parse_json(raw: str) -> list[dict] | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    ok = [d for d in data if isinstance(d, dict) and "q" in d and "a" in d]
    return ok or None


async def one_article(session, sem, key, title, source, n_qa):
    prompt = DENSE_PROMPT.format(title=title, source=source, n_qa=n_qa)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 1400,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://claude-code-wiki-closedqa-smoke",
        "X-Title": "wiki closedqa dense smoke",
    }
    async with sem:
        for _ in range(2):
            try:
                async with session.post(API, headers=headers, json=body, timeout=90) as r:
                    j = await r.json()
                    if "choices" in j and j["choices"]:
                        text = j["choices"][0]["message"]["content"]
                        usage = j.get("usage", {})
                        return {
                            "title": title, "text": text,
                            "parsed": parse_json(text),
                            "in": int(usage.get("prompt_tokens", 0)),
                            "out": int(usage.get("completion_tokens", 0)),
                        }
            except Exception:
                await asyncio.sleep(1)
        return {"title": title, "text": None, "parsed": None, "in": 0, "out": 0}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salience", type=Path, default=Path("/mnt/data2/da_wiki_curation/salience.tsv"))
    ap.add_argument("--tier", default="T1_universal")
    ap.add_argument("--n-articles", type=int, default=6)
    ap.add_argument("--n-qa-per", type=int, default=6)
    ap.add_argument("--key-file", type=Path, default=Path("/home/jepsen/or"))
    args = ap.parse_args()

    # Pick top-N articles from selected tier
    picks = []
    with args.salience.open() as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[5] != args.tier:
                continue
            picks.append((float(parts[6]), int(parts[0]), parts[1]))
    picks.sort(reverse=True)
    picks = picks[:args.n_articles]
    print(f"picked {len(picks)} articles from {args.tier}:", flush=True)
    for _, pid, title in picks:
        print(f"  pid={pid}  {title}")

    # Load wiki text from HF dataset (indexed by title)
    print("\nloading wikimedia/wikipedia 20231101.da …", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.da", split="train")
    want_titles = {t for _, _, t in picks}
    title_to_text = {}
    for row in ds:
        if row["title"] in want_titles:
            title_to_text[row["title"]] = row["text"]
            if len(title_to_text) == len(want_titles):
                break
    print(f"  matched {len(title_to_text)}/{len(want_titles)} titles", flush=True)

    key = args.key_file.read_text().strip()
    sem = asyncio.Semaphore(6)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _, _, title in picks:
            text = title_to_text.get(title)
            if not text:
                continue
            source = clean_intro(text, max_chars=1500)
            tasks.append(one_article(session, sem, key, title, source, args.n_qa_per))
        results = await asyncio.gather(*tasks)

    total_in = total_out = 0
    for r in results:
        total_in += r["in"]; total_out += r["out"]
        print("\n" + "=" * 72)
        print(f"TITLE: {r['title']}   [in={r['in']} out={r['out']}]")
        if r["parsed"]:
            for i, qa in enumerate(r["parsed"], 1):
                print(f"  {i:2d}. Q: {qa['q']}")
                print(f"      A: {qa['a']}")
        else:
            print("  [PARSE FAIL]")
            print("  RAW:", (r["text"] or "")[:400])
    cost = total_in * 0.05 / 1e6 + total_out * 0.15 / 1e6
    print(f"\n=== smoke total: in={total_in} out={total_out}  cost≈${cost:.4f} ===")


if __name__ == "__main__":
    asyncio.run(main())
