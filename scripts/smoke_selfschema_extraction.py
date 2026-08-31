"""Smoke: let Gemini propose the extraction schema itself, per passage.

The alternative design is us sampling keys from a fixed vocabulary. Letting the
model choose means the schema fits the text -- an article about a star gets
`luminositet`, one about a treaty gets `underskrivere` -- which is the variation
`textman_extraction` lacks (one schema, 20,018 times).

What this smoke is actually testing, and the reason it prints per-passage
detail rather than just an aggregate:

  1. do the proposed keys VARY across passages, or does it converge on one
     favourite schema (the failure mode we are trying to escape)
  2. are the values verbatim spans of the passage (the gate that
     textman_extraction failed on 26% of its `numbers`)
  3. are the declared types honoured, so a type gate has something to check

Usage:
  python scripts/smoke_selfschema_extraction.py --n 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path

MODEL = "google/gemini-2.5-flash-lite"
URL = "https://openrouter.ai/api/v1/chat/completions"

# Two turns, deliberately separate calls rather than one combined answer.
# Proposing keys while already holding the values invites the model to pick
# whatever is easiest to fill; proposing them blind, then extracting in a fresh
# turn, is the task we actually want -- and it is also the shape the trained
# model will see (schema given, text given, extract).
SYS_KEYS = """Du er en dansk informationsarkitekt.

Du får en dansk tekst. Foreslå 3-6 felter (nøgler), som en struktureret
opsummering af netop denne tekst burde have.

Regler:
- Nøglerne skal være danske, konkrete og passe til DENNE tekst.
- Vælg felter som teksten faktisk siger noget om.
- Angiv en type for hvert felt: "tekst", "tal", "dato" eller "liste".
- Du må IKKE udfylde værdier. Foreslå kun feltnavne og typer."""

SYS_VALUES = """Du er en præcis dansk informationsudtrækker.

Du får en dansk tekst og en liste af felter. Udfyld hvert felt fra teksten.

Regler:
- Hver værdi SKAL være en ordret sammenhængende tekststump fra teksten. Kopier
  præcis, uden at omskrive, forkorte eller normalisere.
- Kan et felt ikke besvares ordret fra teksten, så returner en tom liste for
  det felt. Opfind ALDRIG en værdi.
- Er typen "liste", er værdien en liste af ordrette tekststumper."""

SCHEMA_KEYS = {
    "type": "object",
    "properties": {"felter": {"type": "array", "items": {
        "type": "object",
        "properties": {"navn": {"type": "string"},
                       "type": {"type": "string",
                                "enum": ["tekst", "tal", "dato", "liste"]}},
        "required": ["navn", "type"], "additionalProperties": False}}},
    "required": ["felter"], "additionalProperties": False,
}

SCHEMA_VALUES = {
    "type": "object",
    "properties": {"felter": {"type": "array", "items": {
        "type": "object",
        "properties": {"navn": {"type": "string"},
                       "vaerdi": {"type": "array", "items": {"type": "string"}}},
        "required": ["navn", "vaerdi"], "additionalProperties": False}}},
    "required": ["felter"], "additionalProperties": False,
}

NUM = re.compile(r"^[\d\s.,%-]+$")
DATE = re.compile(r"\d")


def key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for p in (Path.home() / "or", Path.home() / ".or"):
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("no OpenRouter key (~/or)")


async def _call(session, sys_msg, user_msg, schema, temp):
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": sys_msg},
                         {"role": "user", "content": user_msg}],
            "temperature": temp,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "svar", "strict": True,
                                                "schema": schema}}}
    for _ in range(3):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    await asyncio.sleep(2)
                    continue
                d = await r.json()
                return json.loads(d["choices"][0]["message"]["content"])
        except Exception:
            await asyncio.sleep(2)
    return None


async def one(session, sem, passage):
    """Turn 1: propose keys blind. Turn 2: fill them, fresh context."""
    async with sem:
        k = await _call(session, SYS_KEYS, passage, SCHEMA_KEYS, 0.8)
        if not k or not k.get("felter"):
            return None
        fields = k["felter"]
        spec = "\n".join(f"- {f['navn']} ({f['type']})" for f in fields)
        user = f"Tekst:\n{passage}\n\nFelter:\n{spec}"
        v = await _call(session, SYS_VALUES, user, SCHEMA_VALUES, 0.2)
        if not v:
            return None
        vals = {f["navn"]: f["vaerdi"] for f in v.get("felter", [])}
        return {"felter": [{"navn": f["navn"], "type": f["type"],
                            "vaerdi": vals.get(f["navn"], [])} for f in fields]}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--min-chars", type=int, default=350)
    ap.add_argument("--max-chars", type=int, default=1800)
    ap.add_argument("--dump", default="scratch/selfschema_smoke.jsonl")
    args = ap.parse_args()

    import aiohttp
    from datasets import load_dataset

    ds = load_dataset("jensjepsen/danish-vital-stem-da-v1", split="train")
    col = "text" if "text" in ds.column_names else ds.column_names[0]
    print(f"source: danish-vital-stem-da-v1  n={len(ds)}  col={col!r}", flush=True)
    passages = []
    for r in ds:
        t = (r[col] or "").strip()
        if args.min_chars <= len(t) <= args.max_chars:
            passages.append(t)
        if len(passages) >= args.n:
            break
    print(f"using {len(passages)} passages\n", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {key()}",
                     "Content-Type": "application/json",
                     "X-Title": "selfschema-smoke"},
            timeout=aiohttp.ClientTimeout(total=120)) as s:
        outs = await asyncio.gather(*[one(s, sem, p) for p in passages])

    keyc, rows = Counter(), []
    tot = grounded = typed_ok = empty = 0
    for p, o in zip(passages, outs):
        if not o:
            continue
        for f in o.get("felter", []):
            keyc[f["navn"]] += 1
            if not f["vaerdi"]:
                empty += 1
            for v in f["vaerdi"]:
                tot += 1
                g = v in p
                grounded += g
                t = f["type"]
                ok = (NUM.match(v) is not None) if t == "tal" else \
                     (DATE.search(v) is not None) if t == "dato" else True
                typed_ok += ok
        rows.append({"passage": p, "out": o})

    print(f"{'passage':<4}  proposed keys")
    for i, r in enumerate(rows[:12]):
        ks = ", ".join(f"{f['navn']}({f['type']})" for f in r["out"]["felter"])
        print(f"  {i:<4}{ks[:110]}")
    nf = sum(len(r["out"]["felter"]) for r in rows)
    print(f"\nfields proposed: {nf}   left empty at turn 2: {empty} "
          f"({100*empty/max(1,nf):.1f}%)")
    print(f"values: {tot}   verbatim: {grounded} ({100*grounded/max(1,tot):.1f}%)"
          f"   type-consistent: {typed_ok} ({100*typed_ok/max(1,tot):.1f}%)")
    print(f"distinct keys: {len(keyc)} over {len(rows)} passages "
          f"({len(keyc)/max(1,len(rows)):.1f} per passage)")
    print("most repeated keys:", keyc.most_common(8))
    Path(args.dump).parent.mkdir(exist_ok=True)
    with open(args.dump, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {args.dump}")


if __name__ == "__main__":
    asyncio.run(main())
