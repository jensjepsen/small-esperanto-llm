"""Generate Danish instruction paraphrases; structural checks only, then read.

A behavioural gate was built for this first -- run each candidate on probes
with known answers and keep the ones that score. It took three rounds of fixes
to make it discriminate at all, and its final signal was weaker than simply
reading the sentences: it cannot see near-duplicates, clumsy Danish, register
spread or stray English, all of which are obvious on the page. For a one-off
bank of ~200 short sentences, reading is the proportionate tool.

So this does only what a machine does better than a reader:
  - the {null} placeholder appears exactly once (or .format() breaks later)
  - length is sane
  - no English tokens
  - near-duplicates collapsed

The four planted controls are mixed into the output UNLABELLED. If a reader
misses them, reading is not sufficient after all -- same calibration idea as
the gate, without the machinery.

Usage:
  python scripts/gen_instruction_bank_simple.py --n 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_extraction_da import NULL, _ws  # noqa: E402

MODEL = "google/gemini-2.5-flash-lite"
URL = "https://openrouter.ai/api/v1/chat/completions"

GEN_SYS = """Du skriver instruktioner på dansk til en informationsudtrækker.

Opgaven, instruktionen skal beskrive, er ALTID den samme:
- læs en tekst og en liste af feltnavne
- angiv hvert felts værdi som en ORDRET tekststump fra teksten
- er feltet ikke nævnt i teksten, skrives NULLMARK
- én linje per felt

Skriv 25 FORSKELLIGE formuleringer af denne instruktion. Variér kraftigt:
- registret: kort og bydende, neutralt, høfligt, teknisk, næsten telegrafisk
- længden: fra 4 ord til 3 sætninger
- ordvalget: udtræk / find / hent / angiv / udfyld / gengiv / noter / kopier

Krav:
- det skal fremgå at værdier kopieres ORDRET
- det skal fremgå hvad man gør, når feltet ikke står i teksten
- brug NULLMARK præcis ét sted
- kun dansk, ingen engelske ord"""

CONTROLS = [
    "Opsummer teksten kort med dine egne ord, felt for felt. Mangler et felt, "
    "skriv NULLMARK.",
    "Udtræk felterne fra teksten, og omskriv dem pænt i dine egne ord. Brug "
    "NULLMARK ved manglende felter.",
    "Udfyld felterne ud fra teksten, og gæt hvis oplysningen ikke står der. "
    "NULLMARK bruges sjældent.",
    "Find felternes værdier og oversæt dem til engelsk. Manglende felter: "
    "NULLMARK.",
]

ENGLISH = re.compile(r"\b(output|answer|field|fields|text|passage|value|"
                     r"values|extract|please|the|and|with|from)\b", re.I)


def _key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for p in (Path.home() / "or", Path.home() / ".or"):
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("no OpenRouter key (~/or)")


def norm_key(t):
    """Collapse for near-duplicate detection: lowercase, drop punctuation."""
    return re.sub(r"[^\wæøå ]", "", t.lower()).strip()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="scratch/instruction_candidates.json")
    args = ap.parse_args()

    import aiohttp
    schema = {"type": "object", "properties": {"instruktioner": {
        "type": "array", "items": {"type": "string"}}},
        "required": ["instruktioner"], "additionalProperties": False}

    raw = []
    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}",
                     "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=180)) as s:
        async def one(i):
            body = {"model": MODEL, "temperature": 1.0,
                    "messages": [{"role": "system", "content": GEN_SYS},
                                 {"role": "user", "content": "Skriv dem nu."}],
                    "response_format": {"type": "json_schema", "json_schema": {
                        "name": "svar", "strict": True, "schema": schema}}}
            for _ in range(3):
                try:
                    async with s.post(URL, json=body) as r:
                        if r.status != 200:
                            await asyncio.sleep(2)
                            continue
                        d = await r.json()
                        return json.loads(
                            d["choices"][0]["message"]["content"])["instruktioner"]
                except Exception:
                    await asyncio.sleep(2)
            return []
        for batch in await asyncio.gather(*[one(i)
                                            for i in range((args.n + 24) // 25)]):
            raw += batch

    drop = {"placeholder": 0, "length": 0, "english": 0, "duplicate": 0}
    seen, keep = set(), []
    for t in raw + CONTROLS:
        t = _ws(t)
        if t.count("NULLMARK") != 1:
            drop["placeholder"] += 1
            continue
        if not (12 < len(t) < 400):
            drop["length"] += 1
            continue
        probe = t.replace("NULLMARK", "")
        if ENGLISH.search(probe):
            drop["english"] += 1
            continue
        k = norm_key(probe)
        if k in seen:
            drop["duplicate"] += 1
            continue
        seen.add(k)
        keep.append(t.replace("NULLMARK", "{null}"))

    print(f"generated {len(raw)}  ->  {len(keep)} after checks   dropped {drop}")
    print(f"(the 4 planted controls are somewhere in this list, unlabelled)\n")
    for i, t in enumerate(sorted(keep, key=len)):
        print(f"{i:3d}. {t}")
    Path(args.out).write_text(json.dumps(keep, ensure_ascii=False, indent=1))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
