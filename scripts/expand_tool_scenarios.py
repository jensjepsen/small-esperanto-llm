"""Expand DA tool-call scenarios from 101 seeds to ~5000 via Gemini.

Async loop: each call samples 15 diverse anchors + growing pool, asks Gemini
for ~30 new Danish scenarios in the same style, dedups by ID + fuzzy
description hash, writes to output JSONL.

Usage:
    python scripts/expand_tool_scenarios.py \\
        --seeds data/tool_calls/scenarios_seed.jsonl \\
        --out data/tool_calls/scenarios_expanded.jsonl \\
        --target 5000 --concurrency 25
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

MODEL_ID = os.environ.get("TOOL_MODEL_ID", "gemini-3.1-flash-lite")
_CLIENT = None
_CFG = None


def _read_key_file(names: list[str]) -> str | None:
    for name in names:
        for p in [Path.home() / name, Path.home() / f".{name}"]:
            if p.exists():
                return p.read_text().strip()
    return None


async def call_gemini(prompt: str) -> str | None:
    global _CLIENT, _CFG
    if _CLIENT is None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise SystemExit("pip install google-genai")
        key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
               or _read_key_file(["gem", "gemini_key"]))
        if not key:
            raise SystemExit("No GOOGLE_API_KEY set.")
        _CLIENT = genai.Client(api_key=key)
        _CFG = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=1.0,   # push diversity high
        )
    try:
        resp = await _CLIENT.aio.models.generate_content(
            model=MODEL_ID, contents=prompt, config=_CFG)
        return (resp.text or "").strip() or None
    except Exception as e:
        print(f"  gemini error: {type(e).__name__}: {str(e)[:120]}",
              file=sys.stderr)
        return None


EXPAND_PROMPT = """Du hjælper med at bygge et bredt bibliotek af "værktøjs-scenarier" til træning af en dansk LLM med tool-calling.

Et scenario er én linje: en unik kort ID (snake_case) + en 1-linjes beskrivelse af et konkret arbejdsområde eller nichefag, hvor en person kunne bruge et sæt digitale værktøjer/API'er.

STIL-EKSEMPLER (fra vores eksisterende bank — brug dem KUN som stil-reference, KOPIÉR IKKE):
{anchors}

REGLER:
- Alt på DANSK.
- Undgå de mest oplagte kategorier (kalender, vejr, email, taxi, restaurant, søgning) — vi har for mange af dem.
- Vær ambitiøs: dæk brede vertikaler som håndværk, akademia, kunst, sport, industri, hobby, offentlig sektor, sundhed, forskning, kultur, natur, dyr, transport, mad, byggeri, IT-drift, biotek, gaming, uddannelse, religion, historie.
- Beskrivelsen skal antyde 3-5 konkrete arbejdsgange eller data-objekter (fx "batchlog, malegradsjustering, service").
- ID'et skal være kort (1-2 danske ord i snake_case, æøå ok).

Producer NØJAGTIG 30 nye scenarier — ikke duplikater af eksemplerne. Svar KUN med et JSON-array, ingen forklaring:

[{{"id": "xxx", "beskrivelse": "..."}}, {{"id": "yyy", "beskrivelse": "..."}}, ...]
"""


def _extract_array(text: str) -> list | None:
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Find first [...] block by bracket balance.
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                blob = text[start:i+1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return None
    return None


def _desc_hash(desc: str) -> str:
    """Fuzzy description hash — first 60 chars lowercased, punctuation stripped."""
    s = re.sub(r"[^\wæøåÆØÅ ]+", "", desc.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s[:60]


async def worker(rng: random.Random, seeds: list[dict],
                 pool: list[dict], seen_ids: set,
                 seen_desc_hashes: set) -> list[dict]:
    """One Gemini call → up to 30 new scenarios (deduped)."""
    # Sample 15 anchors: prefer seeds + some recent pool additions.
    anchor_pool = seeds + rng.sample(pool, min(len(pool), 30))
    anchors = rng.sample(anchor_pool, min(len(anchor_pool), 15))
    anchor_str = "\n".join(f'  - {a["id"]}: {a["beskrivelse"]}' for a in anchors)
    prompt = EXPAND_PROMPT.format(anchors=anchor_str)

    resp = await call_gemini(prompt)
    if not resp:
        return []
    arr = _extract_array(resp)
    if not isinstance(arr, list):
        return []

    fresh: list[dict] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        sid = item.get("id", "").strip()
        desc = item.get("beskrivelse", "").strip()
        if not sid or not desc:
            continue
        if sid in seen_ids:
            continue
        dh = _desc_hash(desc)
        if dh in seen_desc_hashes:
            continue
        seen_ids.add(sid)
        seen_desc_hashes.add(dh)
        fresh.append({"id": sid, "beskrivelse": desc})
    return fresh


async def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    seeds = [json.loads(l) for l in
             Path(args.seeds).read_text().splitlines() if l.strip()]
    print(f"loaded {len(seeds)} seed scenarios", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Start pool + seen sets with seeds so dedup covers the whole space.
    pool: list[dict] = list(seeds)
    seen_ids: set = {s["id"] for s in seeds}
    seen_desc_hashes: set = {_desc_hash(s["beskrivelse"]) for s in seeds}

    out_fh = out_path.open("w")
    # Write seeds first so consumers see one merged file.
    for s in seeds:
        out_fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    out_fh.flush()

    sem = asyncio.Semaphore(args.concurrency)
    written = len(seeds)

    async def one_call():
        nonlocal written
        async with sem:
            if written >= args.target:
                return
            fresh = await worker(rng, seeds, pool, seen_ids, seen_desc_hashes)
            for item in fresh:
                if written >= args.target:
                    break
                pool.append(item)
                out_fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                out_fh.flush()
                written += 1
            if written % 200 < 30:
                print(f"  written={written}/{args.target}", flush=True)

    # Overshoot: launch enough calls to hit target with dedup losses.
    # Each call yields ~15-25 fresh after dedup (declines as pool grows).
    # Start with n_calls = target/12; extend if we fall short.
    round_size = max(50, args.target // 15)
    while written < args.target:
        remaining = args.target - written
        # Scale calls per round to remaining budget.
        n_calls = min(round_size, max(20, remaining // 10 + 20))
        tasks = [asyncio.create_task(one_call()) for _ in range(n_calls)]
        await asyncio.gather(*tasks)
        print(f"  round done: written={written}/{args.target}  pool={len(pool)}",
              flush=True)
        # Fail-safe: if a round adds <10, we're saturating the space.
        if not tasks or written >= args.target:
            break

    out_fh.close()
    print(f"\nfinal: {written} unique scenarios written to {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--concurrency", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
