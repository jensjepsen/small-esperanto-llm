"""Smoke test: balanced SAND/FALSK fact-check generation on DA STEM wiki.

Per article, gemma emits N true + N false claims (mixed), each with a
verdict and short reasoning. By construction the training data is
balanced 50/50 — no verdict-bias when this feeds SFT.

Usage:
    uv run --no-project --with datasets --with aiohttp python \\
        scripts/smoke_fact_check_da.py \\
        --tsv data/da_wiki_stem/pageids_d2_core_filtered.tsv \\
        --n 12 --out /tmp/factcheck_smoke.jsonl
"""
from __future__ import annotations
import argparse, asyncio, json, random, re
from pathlib import Path
import aiohttp
from datasets import load_dataset

MODEL = "google/gemini-2.5-flash-lite"
API = "https://openrouter.ai/api/v1/chat/completions"

PROMPT_TEMPLATE = """Du er en dansk faktakontrol-generator. Læs artiklen og lav OP TIL {n_each} SANDE og OP TIL {n_each} FALSKE påstande baseret på indholdet.

STRENGE REGLER:
  * SANDE påstande skal være DIREKTE understøttet af artiklen (ikke pseudo-sande gæt).
  * FALSKE påstande skal være PLAUSIBLE misforståelser eller tal-fejl, som artiklen tydeligt afkræfter — ikke tilfældigt vrøvl.
  * Blandet rækkefølge (ikke alle SANDE først).
  * Hver påstand: 1-2 sætninger, konkret, verificerbar mod teksten.
  * Reasoning: 1-2 sætninger med henvisning til hvad artiklen siger.
  * Alt på flydende dansk.
  * Hvis artiklen kun understøtter færre — brug færre. Kvalitet > kvantitet.
  * Verdict skal være PRÆCIS 'SAND' eller 'FALSK' (ingen anden formulering).

Output KUN gyldig JSON:
{{"claims": [
    {{"claim": "...", "verdict": "SAND", "reasoning": "..."}},
    {{"claim": "...", "verdict": "FALSK", "reasoning": "..."}},
    ...
  ]}}

ARTIKEL (titel: {title}):
{text}"""

PARSE_RE = re.compile(r"\{.*\}", re.S)


def _relax(s):
    valid = set('"\\/bfnrtu')
    out, in_str, esc = [], False, False
    for c in s:
        if esc:
            if in_str and c not in valid: out.append("\\")
            out.append(c); esc = False; continue
        if c == "\\": out.append(c); esc = True; continue
        if c == '"': in_str = not in_str; out.append(c); continue
        if in_str and c in "\n\r\t":
            out.append({"\n":"\\n","\r":"\\r","\t":"\\t"}[c])
        else:
            out.append(c)
    return "".join(out)


def parse(raw):
    t = raw.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0].strip()
    m = PARSE_RE.search(t)
    if not m: return None
    js = m.group(0)
    try: obj = json.loads(js)
    except json.JSONDecodeError:
        try: obj = json.loads(_relax(js))
        except json.JSONDecodeError: return None
    claims = obj.get("claims")
    if not isinstance(claims, list): return None
    ok = []
    for c in claims:
        if not isinstance(c, dict): continue
        if c.get("verdict") not in ("SAND", "FALSK"): continue
        if not isinstance(c.get("claim"), str) or not isinstance(c.get("reasoning"), str):
            continue
        ok.append({"claim": c["claim"].strip(), "verdict": c["verdict"],
                   "reasoning": c["reasoning"].strip()})
    return ok if ok else None


async def gen(session, sem, key, art, n_each):
    body = {
        "model": MODEL,
        "messages": [{"role": "user",
                       "content": PROMPT_TEMPLATE.format(n_each=n_each,
                                                        title=art["title"],
                                                        text=art["text"])}],
        "temperature": 0.3,
        "max_tokens": 4000,
        "provider": {"order": ["Google AI Studio", "Google"], "allow_fallbacks": True},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://claude-code-fc", "X-Title": "DA-FactCheck"}
    base = {"pageid": art["pageid"], "title": art["title"],
            "text": art["text"], "text_len": len(art["text"])}
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(API, headers=headers, json=body, timeout=90) as resp:
                    data = await resp.json()
                if "choices" not in data:
                    if attempt < 2: await asyncio.sleep(2 ** attempt); continue
                    return {**base, "reject": f"api:{json.dumps(data)[:150]}"}
                raw = data["choices"][0]["message"]["content"]
                cost = float(data.get("usage", {}).get("cost", 0) or 0)
                claims = parse(raw)
                if claims is None:
                    if attempt < 2: await asyncio.sleep(2 ** attempt); continue
                    return {**base, "cost": cost, "reject": "parse_fail", "raw": raw[:400]}
                return {**base, "claims": claims, "cost": cost}
            except Exception as e:
                if attempt < 2: await asyncio.sleep(2 ** attempt); continue
                return {**base, "reject": f"exc:{str(e)[:150]}"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path,
                    default=Path("data/da_wiki_stem/pageids_d2_core_filtered.tsv"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--n-each", type=int, default=6,
                    help="Target claims per verdict per article")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--min-chars", type=int, default=1500)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--key-file", type=Path, default=Path("/home/jepsen/or"))
    args = ap.parse_args()

    key = args.key_file.read_text().strip()

    entries = []
    for line in args.tsv.open():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            entries.append((int(parts[0]), parts[1]))
    print(f"loaded {len(entries)} STEM titles", flush=True)

    print("loading wikimedia/wikipedia 20231101.da...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.da", split="train")
    title_to_text = {r["title"]: r["text"] for r in ds}
    print(f"  indexed {len(title_to_text)} wiki titles", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(entries)
    picked = []
    for pageid, title in entries:
        text = title_to_text.get(title)
        if not text: continue
        if not (args.min_chars <= len(text) <= args.max_chars): continue
        picked.append({"pageid": pageid, "title": title, "text": text})
        if len(picked) >= args.n: break
    print(f"  sampled {len(picked)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Resume: skip pageids already done successfully
    done = set()
    prior_cost = 0.0
    if args.out.exists():
        for line in args.out.open():
            try:
                r = json.loads(line)
                if not r.get("reject"):
                    done.add(r["pageid"])
                    prior_cost += r.get("cost", 0) or 0
            except Exception: pass
        print(f"  resume: {len(done)} done (${prior_cost:.4f})", flush=True)
    todo = [a for a in picked if a["pageid"] not in done]
    print(f"  {len(todo)} to go", flush=True)
    if not todo:
        rows = []
    else:
        async with aiohttp.ClientSession() as session:
            sem = asyncio.Semaphore(args.concurrency)
            tasks = [asyncio.create_task(gen(session, sem, key, a, args.n_each)) for a in todo]
            rows = []
            with args.out.open("a") as fout:
                for coro in asyncio.as_completed(tasks):
                    r = await coro
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    fout.flush()
                    rows.append(r)
                    if len(rows) % 100 == 0:
                        run_cost = sum(x.get("cost", 0) for x in rows)
                        print(f"  {len(rows)}/{len(todo)}  ok={sum(1 for x in rows if not x.get('reject'))}  "
                              f"cost=${run_cost:.4f}", flush=True)

    ok = [r for r in rows if not r.get("reject")]
    rej = [r for r in rows if r.get("reject")]
    cost = sum(r.get("cost", 0) for r in rows)
    total_claims = sum(len(r["claims"]) for r in ok)
    sand = sum(1 for r in ok for c in r["claims"] if c["verdict"] == "SAND")
    falsk = sum(1 for r in ok for c in r["claims"] if c["verdict"] == "FALSK")
    print(f"\nDone: {len(ok)}/{len(rows)} ok, {len(rej)} rej. Cost ${cost:.4f}",
          flush=True)
    print(f"total claims: {total_claims}  SAND: {sand} ({100*sand/max(total_claims,1):.0f}%)  "
          f"FALSK: {falsk} ({100*falsk/max(total_claims,1):.0f}%)  "
          f"avg/article: {total_claims/max(len(ok),1):.1f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
