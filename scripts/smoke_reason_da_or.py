"""Smoke test: reasoning generation on DA wiki via gemma-3-12b/OR.

Per article, gemma emits 3 reasoning items in one JSON: (1) causal chain,
(2) fact-check with verdict + evidence, (3) argumentation with pro/con.
Compare-and-contrast is a separate build (needs paired articles) — not
covered here.

Usage:
    uv run --no-project --with datasets --with aiohttp \\
        python scripts/smoke_reason_da_or.py --n 30 \\
        --out /tmp/reason_smoke.jsonl --concurrency 10
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

PROMPT_TEMPLATE = """Du er en dansk læringssassistent. Læs følgende artikel og lav 6 ræsonnements-øvelser: én af hver type nedenfor.

STRENGE REGLER:
  * causal_chain: identificér et årsag-virkning-forhold i artiklen, formulér et 'hvorfor'-spørgsmål, og skriv et svar som forklarer kæden af mekanismer (2-4 sætninger). Ikke gengivelse — forklaring.
  * fact_check: formulér en påstand (vælg SAND eller FALSK 50/50), og skriv en dom ('SAND' eller 'FALSK') samt en 1-3 sætningers begrundelse med henvisning til teksten.
  * argumentation: stil et 'bør X'- eller 'er X en god idé'-spørgsmål baseret på et emne i artiklen, og skriv en balanceret argumentation med både fordele og ulemper (3-5 sætninger).
  * multi_step: formulér et spørgsmål der kræver 2 eller flere logiske trin (f.eks. 'hvis X er sandt og Y følger af X, hvad kan vi så sige om Z?'), og skriv et svar der eksplicit viser hvert trin.
  * ranking: bed læseren om at rangere 3 elementer fra artiklen efter et kriterium (vigtighed, størrelse, tid, indflydelse el.lign.), og skriv en rangering med kort begrundelse for hver plads.
  * analogy: formulér en analogi der forbinder et koncept i artiklen med noget mere velkendt ('X ligner Y, fordi ...'), og forklar hvor analogien holder og hvor den bryder sammen (2-4 sætninger).
  * Hvis en type ikke passer artiklen naturligt, konstrukér alligevel et rimeligt eksempel — udelad ikke nøgler.
  * Alt skal være på flydende dansk. Ingen ordret gengivelse — brug egne ord.

Output KUN gyldig JSON i dette format, uden markdown eller kommentarer:
{{"causal_chain": {{"q": "...", "a": "..."}},
  "fact_check": {{"claim": "...", "verdict": "SAND|FALSK", "reasoning": "..."}},
  "argumentation": {{"q": "...", "a": "..."}},
  "multi_step": {{"q": "...", "a": "..."}},
  "ranking": {{"q": "...", "a": "..."}},
  "analogy": {{"q": "...", "a": "..."}}}}

ARTIKEL (titel: {title}):
{text}"""


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
    for k in ("causal_chain", "fact_check", "argumentation",
              "multi_step", "ranking", "analogy"):
        if k not in obj or not isinstance(obj[k], dict):
            return None
    if "verdict" not in obj["fact_check"] or obj["fact_check"]["verdict"] not in ("SAND", "FALSK"):
        return None
    return obj


async def gen_row(session, sem, key, art, provider):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content":
                       PROMPT_TEMPLATE.format(title=art["title"], text=art["text"])}],
        "temperature": 0.4,
        "max_tokens": 2500,
        "provider": {"order": ["Google AI Studio","Google"], "allow_fallbacks": True},
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://claude-code-reason-smoke",
        "X-Title": "DA-Reason-Smoke",
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
                            "cost": cost, "reject": "parse_fail", "raw": raw[:400]}
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
    ap.add_argument("--seed", type=int, default=43)
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
