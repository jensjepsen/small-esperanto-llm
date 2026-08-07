"""Smoke test: RC (reading comprehension) generation on DA wiki via gemma-3-12b/OR.

Samples N mid-length DA wiki articles, asks gemma-3-12b to generate 3-4
inference-requiring questions per article (explicitly forbid first-sentence
lookup). Writes JSONL for spot-checking parse rate, task fidelity, DA quality.

Usage:
    uv run --no-project --with datasets --with aiohttp \\
        python scripts/smoke_rc_da_or.py --n 30 \\
        --out /tmp/rc_smoke.jsonl --concurrency 10
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

PROMPT_TEMPLATE = """Du er en dansk læringssassistent. Læs følgende artikel og lav præcis 4 læseforståelsesspørgsmål, ét fra hver af typerne nedenfor. Vælg typer, der passer til artiklen — hvis en type ikke er anvendelig (f.eks. 'numeric' på en artikel uden tal), vælg 'multi_fact' i stedet.

Typer (vælg 4 forskellige om muligt):
  * multi_fact        — kombinér information fra 2+ afsnit til én sammenhængende observation.
  * numeric           — kræver at læseren udleder eller sammenligner tal, mængder eller datoer fra artiklen.
  * attribution       — 'hvem sagde/mente/gjorde X?' — spor kilden eller aktøren gennem teksten.
  * ordering          — sæt begivenheder A, B, C i kronologisk rækkefølge baseret på teksten.
  * causal_inference  — 'hvorfor'-spørgsmål der kræver at læseren identificerer en årsagskæde eller motivation.

STRENGE REGLER:
  * Ingen spørgsmål må kunne besvares alene ud fra artiklens første sætning eller titel.
  * Svarene skal være støttet af teksten men formuleret i egne ord (ikke ordret citat), 1-4 sætninger.
  * Både spørgsmål og svar skal være på flydende dansk.

Output KUN gyldig JSON i dette format, uden markdown eller kommentarer:
{{"questions": [{{"type": "multi_fact", "q": "...", "a": "..."}}, {{"type": "numeric", "q": "...", "a": "..."}}, ...]}}

ARTIKEL (titel: {title}):
{text}"""

RC_TYPES = {"multi_fact", "numeric", "attribution", "ordering", "causal_inference"}


PARSE_RE = re.compile(r"\{.*\}", re.S)


def _relax_json(s: str) -> str:
    out, in_str, esc = [], False, False
    for c in s:
        if esc: out.append(c); esc = False; continue
        if c == "\\": out.append(c); esc = True; continue
        if c == '"': in_str = not in_str; out.append(c); continue
        if in_str and c in "\n\r\t":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[c])
        else:
            out.append(c)
    return "".join(out)


def parse_output(text: str) -> list[dict] | None:
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
    qs = obj.get("questions")
    if not isinstance(qs, list) or not qs:
        return None
    ok = []
    for q in qs:
        if not isinstance(q, dict): continue
        if not isinstance(q.get("q"), str) or not isinstance(q.get("a"), str):
            continue
        t = q.get("type", "multi_fact")
        if t not in RC_TYPES:
            t = "multi_fact"
        ok.append({"type": t, "q": q["q"].strip(), "a": q["a"].strip()})
    return ok or None


async def gen_row(session, sem, key, art, provider):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content":
                       PROMPT_TEMPLATE.format(title=art["title"], text=art["text"])}],
        "temperature": 0.3,
        "max_tokens": 1500,
        "provider": {"order": ["Google AI Studio","Google"], "allow_fallbacks": True},
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://claude-code-rc-smoke",
        "X-Title": "DA-RC-Smoke",
    }
    base = {"orig_idx": art["orig_idx"], "title": art["title"],
            "text": art["text"], "text_len": len(art["text"])}
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(API, headers=headers, json=body, timeout=90) as resp:
                    data = await resp.json()
                if "choices" not in data:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt); continue
                    return {**base, "qs": None, "cost": 0, "reject": f"api:{json.dumps(data)[:180]}"}
                raw = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                in_t = int(usage.get("prompt_tokens", 0))
                out_t = int(usage.get("completion_tokens", 0))
                cost = float(usage.get("cost", 0) or 0)
                qs = parse_output(raw)
                if qs is None:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt); continue
                    return {**base, "qs": None, "in_t": in_t, "out_t": out_t,
                            "cost": cost, "reject": "parse_fail", "raw": raw[:400]}
                return {**base, "qs": qs, "in_t": in_t, "out_t": out_t, "cost": cost}
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt); continue
                return {**base, "qs": None, "cost": 0, "reject": f"exc:{str(e)[:200]}"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--min-chars", type=int, default=2000)
    ap.add_argument("--max-chars", type=int, default=4500)
    ap.add_argument("--key-file", type=Path, default=Path("/home/jepsen/or"))
    ap.add_argument("--seed", type=int, default=42)
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
    print(f"  sampled {len(picked)} articles ({args.min_chars}-{args.max_chars} chars)",
          flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Resume: skip orig_idx already present in output.
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
        tasks = [asyncio.create_task(gen_row(session, sem, key, art, args.provider))
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
