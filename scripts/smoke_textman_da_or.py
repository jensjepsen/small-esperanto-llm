"""Smoke test: text-manipulation on DA wiki via gemma-3-12b/OR.

Per article, gemma emits 3 outputs: (1) 3-bullet summary, (2) rewrite in own
words, (3) style-transferred version to a randomly chosen target register
(formal, casual, or plain).

Usage:
    uv run --no-project --with datasets --with aiohttp \\
        python scripts/smoke_textman_da_or.py --n 30 \\
        --out /tmp/textman_smoke.jsonl --concurrency 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

import aiohttp
from datasets import load_dataset


MODEL = "google/gemini-2.5-flash-lite"
API = "https://openrouter.ai/api/v1/chat/completions"

STYLE_TARGETS = ["formel akademisk", "afslappet talesprog", "let og letforståelig"]
GENRE_TARGETS = ["nyhedsoverskrift", "tweet", "email", "brev"]


PROMPT_TEMPLATE = """Du er en dansk sprogassistent. Læs følgende artikel og lav 6 tekstmanipulations-opgaver.

STRENGE REGLER:
  * summary: skriv præcis 3 bullet points der opsummerer artiklens vigtigste pointer. Hver bullet skal begynde med '- ' og være 1 sætning.
  * rewrite: omskriv de første 2-3 afsnit af artiklen i dine egne ord. Bevar al information, men brug HELT andre formuleringer end kilden.
  * style_transfer: omskriv de første 2-3 afsnit i en {style} stil. Bevar indholdet, tilpas sprog, tone og ordvalg.
  * extraction: udtræk struktureret information fra artiklen som JSON-objekter. Nøgler: 'people' (personer nævnt), 'places' (steder), 'dates' (årstal og datoer), 'numbers' (numeriske værdier med enhed/betydning). Hver værdi er en liste af strenge; tom liste hvis ingen findes.
  * elaborate: vælg en kort passage (1-2 sætninger) fra artiklen, og udvid den til et længere afsnit (4-6 sætninger) med baggrund, eksempler eller uddybning — enten fra artiklen selv eller almen viden. Angiv den valgte kilde-passage separat.
  * genre_transform: omskriv artiklens essens som en {genre}. Bevar de vigtigste fakta, men tilpas længde og form til genren (nyhedsoverskrift: 1 sætning på op til 15 ord; tweet: 1-2 sætninger på op til 280 tegn; email: kort formel/uformel mail med subject; brev: 1 afsnit i brevform med hilsen).
  * Alt skal være på flydende dansk.

Output KUN gyldig JSON i dette format, uden markdown eller kommentarer:
{{"summary": "- ...\\n- ...\\n- ...",
  "rewrite": "...",
  "style_transfer": {{"target_style": "{style}", "text": "..."}},
  "extraction": {{"people": [], "places": [], "dates": [], "numbers": []}},
  "elaborate": {{"source_passage": "...", "expanded": "..."}},
  "genre_transform": {{"target_genre": "{genre}", "text": "..."}}}}

ARTIKEL (titel: {title}):
{text}"""


PARSE_RE = re.compile(r"\{.*\}", re.S)


def _relax_json(s: str) -> str:
    """Escape unescaped \\n/\\r/\\t inside JSON string literals.

    Gemma routinely writes real newlines inside "..." values instead of the
    JSON-required \\n escape, tripping json.loads with 'Invalid control
    character'. Walk the string, track whether we're inside a "-delimited
    string with backslash escapes, and replace bare control chars with their
    escaped form.
    """
    out = []
    in_str = False
    esc = False
    for c in s:
        if esc:
            out.append(c); esc = False; continue
        if c == "\\":
            out.append(c); esc = True; continue
        if c == '"':
            in_str = not in_str; out.append(c); continue
        if in_str and c in "\n\r\t":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[c])
        else:
            out.append(c)
    return "".join(out)


def parse_output(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()
    m = PARSE_RE.search(text)
    if not m:
        return None
    js = m.group(0)
    try:
        obj = json.loads(js)
    except json.JSONDecodeError:
        try:
            obj = json.loads(_relax_json(js))
        except json.JSONDecodeError:
            return None
    if not all(k in obj for k in ("summary", "rewrite", "style_transfer",
                                    "extraction", "elaborate", "genre_transform")):
        return None
    if not isinstance(obj["style_transfer"], dict) or "text" not in obj["style_transfer"]:
        return None
    if not isinstance(obj["extraction"], dict):
        return None
    for k in ("people", "places", "dates", "numbers"):
        if k not in obj["extraction"] or not isinstance(obj["extraction"][k], list):
            return None
    if not isinstance(obj["elaborate"], dict) or "expanded" not in obj["elaborate"]:
        return None
    if not isinstance(obj["genre_transform"], dict) or "text" not in obj["genre_transform"]:
        return None
    return obj


async def gen_row(session, sem, key, art, style, genre, provider):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content":
                       PROMPT_TEMPLATE.format(title=art["title"], text=art["text"],
                                              style=style, genre=genre)}],
        "temperature": 0.4,
        "max_tokens": 3200,
        "provider": {"order": ["Google AI Studio","Google"], "allow_fallbacks": True},
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://claude-code-textman-smoke",
        "X-Title": "DA-TextMan-Smoke",
    }
    base = {"orig_idx": art["orig_idx"], "title": art["title"],
            "text": art["text"], "text_len": len(art["text"]),
            "style_target": style, "genre_target": genre}
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(API, headers=headers, json=body, timeout=90) as resp:
                    data = await resp.json()
                if "choices" not in data:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt); continue
                    return {**base, "items": None, "cost": 0, "reject": f"api:{json.dumps(data)[:180]}"}
                raw = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                in_t = int(usage.get("prompt_tokens", 0))
                out_t = int(usage.get("completion_tokens", 0))
                cost = float(usage.get("cost", 0) or 0)
                items = parse_output(raw)
                if items is None:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt); continue
                    return {**base, "items": None, "in_t": in_t, "out_t": out_t,
                            "cost": cost, "reject": "parse_fail", "raw": raw}
                return {**base, "items": items, "in_t": in_t, "out_t": out_t, "cost": cost}
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt); continue
                return {**base, "items": None, "cost": 0, "reject": f"exc:{str(e)[:200]}"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--min-chars", type=int, default=2000)
    ap.add_argument("--max-chars", type=int, default=4500)
    ap.add_argument("--key-file", type=Path, default=Path("/home/jepsen/or"))
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--provider", default="DeepInfra",
                    help="OpenRouter provider slug — pinned via provider.order")
    args = ap.parse_args()

    key = args.key_file.read_text().strip()

    print("loading wikimedia/wikipedia 20231101.da...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.da", split="train")
    print(f"  {len(ds):,} articles total", flush=True)

    rng = random.Random(args.seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    picked = []
    for i in idxs:
        row = ds[i]
        n = len(row["text"])
        if args.min_chars <= n <= args.max_chars:
            picked.append({"orig_idx": i, "title": row["title"], "text": row["text"]})
        if len(picked) >= args.n:
            break
    print(f"  sampled {len(picked)} articles", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    prior_cost = 0.0
    if args.out.exists():
        for line in args.out.open():
            try:
                r = json.loads(line)
                if r.get("reject"): continue
                done.add(r["orig_idx"])
                prior_cost += r.get("cost", 0) or 0
            except Exception:
                pass
        print(f"  resume: {len(done):,} rows already done (${prior_cost:.4f})", flush=True)
    todo = [a for a in picked if a["orig_idx"] not in done]
    print(f"  {len(todo):,} rows to generate", flush=True)
    if not todo:
        return

    t0 = time.time()
    n_ok = n_rej = 0
    tok_in = tok_out = 0
    cost = 0.0
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(args.concurrency)
        tasks = [asyncio.create_task(gen_row(session, sem, key, art,
                                              rng.choice(STYLE_TARGETS),
                                              rng.choice(GENRE_TARGETS),
                                              args.provider))
                 for art in todo]
        with args.out.open("a") as fout:
            for coro in asyncio.as_completed(tasks):
                r = await coro
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                if r.get("reject"):
                    n_rej += 1
                else:
                    n_ok += 1
                tok_in += r.get("in_t", 0)
                tok_out += r.get("out_t", 0)
                cost += r.get("cost", 0) or 0
                d = n_ok + n_rej
                el = time.time() - t0
                print(f"  {d}/{len(todo)}  ok={n_ok} rej={n_rej}  "
                      f"tok in={tok_in:,} out={tok_out:,}  "
                      f"cost=${cost:.4f}  elapsed={el:.0f}s", flush=True)

    print(f"\nDone: {n_ok} ok, {n_rej} rejected. Cost ${cost:.4f} "
          f"(+prior ${prior_cost:.4f}). "
          f"Extrapolated per 40k: ${(cost+prior_cost) * 40000 / max(len(picked),1):.1f}",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
