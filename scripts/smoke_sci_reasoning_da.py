"""Smoke test: 3 new science-reasoning generation styles on DA STEM wiki.

Reads the existing STEM TSV (pageid, title, len), fetches article text from
wikimedia/wikipedia:20231101.da via a title index, then runs each article
through 3 new prompt templates that target REASONING (not factual recall,
which is already covered by wiki-closedqa-stem-v1):

  * worked_calc   — multi-step numerical problem + worked solution
                    (only for articles with quantitative content)
  * mechanism     — "How does X work" in 3-5 explicit steps
  * counterfactual — "If X changes, what happens?" with reasoning

Misconceptions are handled separately in smoke_fact_check_da.py — that
script produces balanced SAND/FALSK claims per article, which avoids the
verdict-bias problem that misconceptions alone would cause in SFT.

Usage:
    uv run --no-project --with datasets --with aiohttp python \\
        scripts/smoke_sci_reasoning_da.py \\
        --tsv data/da_wiki_stem/pageids_d2_core_filtered.tsv \\
        --n 15 --out /tmp/sci_reason_smoke.jsonl
"""
from __future__ import annotations
import argparse, asyncio, json, random, re
from pathlib import Path
import aiohttp
from datasets import load_dataset

MODEL = "google/gemini-2.5-flash-lite"
API = "https://openrouter.ai/api/v1/chat/completions"


PROMPT_TEMPLATE = """Du er en dansk naturvidenskabslærer. Læs artiklen og lav OP TIL 6 forskellige ræsonnements-øvelser i HVER af de 3 typer beskrevet nedenfor. Hver øvelse skal dække et NYT aspekt eller en ny vinkel — ingen minor-varianter af samme spørgsmål.

TYPER:
  * worked_calc — Numeriske problemer baseret på tal, formler eller enheder fra artiklen. Hver skal have en trin-for-trin løsning (mindst 3 trin, inkl. formler og enheder). Hvis artiklen har MEGET LIDT numerisk indhold, generér færre (helt ned til 0 = tom liste).
  * mechanism — 'Hvordan virker X?' / 'Hvordan foregår Y?' — mekanistiske forklaringer. Svaret skal være 3-5 nummererede trin. Hver Q skal handle om en FORSKELLIG proces eller mekanisme i artiklen.
  * counterfactual — 'Hvad hvis X ændres?' — hver skal ændre en FORSKELLIG variabel og forudsige+forklare effekten.

STRENGE REGLER:
  * Alle svar baseret på artiklens indhold. Ingen fabrikation.
  * Alt på flydende dansk. Ingen ordret gengivelse.
  * Sigte 6 pr. type, men EMPTY LIST hvis artiklen ikke understøtter flere. Kvalitet > kvantitet.

Output KUN gyldig JSON:
{{"worked_calc": [{{"q": "...", "a": "..."}}, ...],
  "mechanism": [{{"q": "...", "a": "..."}}, ...],
  "counterfactual": [{{"q": "...", "a": "..."}}, ...]}}

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
    keys = {"worked_calc", "mechanism", "counterfactual"}
    if not all(k in obj for k in keys): return None
    for k in keys:
        if not isinstance(obj[k], list): return None
    return obj


async def gen(session, sem, key, art):
    body = {
        "model": MODEL,
        "messages": [{"role": "user",
                       "content": PROMPT_TEMPLATE.format(title=art["title"], text=art["text"])}],
        "temperature": 0.3,
        "max_tokens": 8000,
        "provider": {"order": ["Google AI Studio", "Google"], "allow_fallbacks": True},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://claude-code-sci-reason", "X-Title": "DA-SciReason"}
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
                items = parse(raw)
                if items is None:
                    if attempt < 2: await asyncio.sleep(2 ** attempt); continue
                    return {**base, "cost": cost, "reject": "parse_fail", "raw": raw[:400]}
                return {**base, "items": items, "cost": cost}
            except Exception as e:
                if attempt < 2: await asyncio.sleep(2 ** attempt); continue
                return {**base, "reject": f"exc:{str(e)[:150]}"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path,
                    default=Path("data/da_wiki_stem/pageids_d2_core_filtered.tsv"))
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--min-chars", type=int, default=1500)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--key-file", type=Path, default=Path("/home/jepsen/or"))
    args = ap.parse_args()

    key = args.key_file.read_text().strip()

    # Read TSV: pageid<TAB>title<TAB>len
    entries = []
    for line in args.tsv.open():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            entries.append((int(parts[0]), parts[1]))
    print(f"loaded {len(entries)} STEM titles from tsv", flush=True)

    # Load wiki + index by title
    print("loading wikimedia/wikipedia 20231101.da...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.da", split="train")
    title_to_text = {}
    for r in ds:
        title_to_text[r["title"]] = r["text"]
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
    print(f"  sampled {len(picked)} STEM articles in length window", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Resume: skip pageids already successfully generated
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
            tasks = [asyncio.create_task(gen(session, sem, key, a)) for a in todo]
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
    print(f"\nDone: {len(ok)}/{len(rows)} ok, {len(rej)} rej. Cost ${cost:.4f}",
          flush=True)
    # subtype counts (total items per subtype across all articles)
    from collections import Counter
    counts = Counter()
    for r in ok:
        for k in ("worked_calc", "mechanism", "counterfactual"):
            counts[k] += len(r["items"].get(k, []))
    total = sum(counts.values())
    print(f"items per subtype (total across {len(ok)} articles): {dict(counts)}",
          flush=True)
    print(f"total items: {total}  avg/article: {total/max(len(ok),1):.1f}",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
