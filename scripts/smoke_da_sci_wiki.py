"""Smoke test: filter DA Wikipedia for science-relevant articles.

Samples N random articles, asks gemini-2.5-flash-lite to classify each
(is it primarily science? which domain? confidence 0-1). Reports the
science-rate + per-domain distribution so we can budget for a full pass.

Usage:
    uv run --no-project --with datasets --with aiohttp python \\
        scripts/smoke_da_sci_wiki.py --n 100 --out /tmp/da_sci_smoke.jsonl
"""
from __future__ import annotations
import argparse, asyncio, json, random, re
from pathlib import Path
import aiohttp
from datasets import load_dataset

MODEL = "google/gemini-2.5-flash-lite"
API = "https://openrouter.ai/api/v1/chat/completions"

DOMAINS = ["fysik", "kemi", "biologi", "matematik", "astronomi", "geologi",
           "medicin", "datalogi", "andet"]

PROMPT = """Du er en klassificerings-assistent. Læs følgende danske Wikipedia-artikel og vurdér:

1. Er artiklen PRIMÆRT om et naturvidenskabeligt emne? (fx fysik, kemi, biologi, matematik, astronomi, geologi, medicin, datalogi)
   - JA: emnet er en videnskabelig proces, koncept, teori, metode, formel, enhed, art, fænomen etc.
   - NEJ: emnet er en person, sted, film, band, sportsklub, politisk begivenhed, virksomhed, etc. — selvom en person kan være videnskabsmand, er artiklen om personen, ikke videnskaben.

2. Hvis JA, hvilket domæne? (fysik, kemi, biologi, matematik, astronomi, geologi, medicin, datalogi, andet)

3. Konfidens 0.0-1.0.

Output KUN gyldig JSON: {{"is_science": true|false, "domain": "..." eller null, "confidence": 0.85, "reason": "1 sætning"}}

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
    if "is_science" not in obj: return None
    return obj


async def classify(session, sem, key, art):
    body = {
        "model": MODEL,
        "messages": [{"role": "user",
                       "content": PROMPT.format(title=art["title"],
                                                text=art["text"][:3500])}],
        "temperature": 0.1,
        "max_tokens": 400,
        "provider": {"order": ["Google AI Studio", "Google"], "allow_fallbacks": True},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://claude-code-da-sci", "X-Title": "DA-SciFilter"}
    base = {"orig_idx": art["orig_idx"], "title": art["title"],
            "text_len": len(art["text"])}
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(API, headers=headers, json=body, timeout=60) as resp:
                    data = await resp.json()
                if "choices" not in data:
                    if attempt < 2: await asyncio.sleep(2 ** attempt); continue
                    return {**base, "reject": f"api:{json.dumps(data)[:150]}"}
                raw = data["choices"][0]["message"]["content"]
                cost = float(data.get("usage", {}).get("cost", 0) or 0)
                obj = parse(raw)
                if obj is None:
                    if attempt < 2: await asyncio.sleep(2 ** attempt); continue
                    return {**base, "cost": cost, "reject": "parse_fail", "raw": raw[:200]}
                return {**base, **obj, "cost": cost}
            except Exception as e:
                if attempt < 2: await asyncio.sleep(2 ** attempt); continue
                return {**base, "reject": f"exc:{str(e)[:100]}"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=15)
    ap.add_argument("--min-chars", type=int, default=800)
    ap.add_argument("--max-chars", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--key-file", type=Path, default=Path("/home/jepsen/or"))
    args = ap.parse_args()

    key = args.key_file.read_text().strip()
    print("loading wikimedia/wikipedia 20231101.da...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.da", split="train")
    print(f"  {len(ds):,} total", flush=True)

    rng = random.Random(args.seed)
    idxs = list(range(len(ds))); rng.shuffle(idxs)
    picked = []
    for i in idxs:
        n = len(ds[i]["text"])
        if args.min_chars <= n <= args.max_chars:
            picked.append({"orig_idx": i, "title": ds[i]["title"], "text": ds[i]["text"]})
        if len(picked) >= args.n: break
    print(f"  sampled {len(picked)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(args.concurrency)
        tasks = [asyncio.create_task(classify(session, sem, key, a)) for a in picked]
        rows = []
        with args.out.open("w") as fout:
            for coro in asyncio.as_completed(tasks):
                r = await coro
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                rows.append(r)

    ok = [r for r in rows if not r.get("reject")]
    rej = [r for r in rows if r.get("reject")]
    sci = [r for r in ok if r.get("is_science")]
    cost = sum(r.get("cost", 0) for r in rows)
    print(f"\nclassified {len(ok)}/{len(rows)} ok, {len(rej)} rej. Cost ${cost:.4f}",
          flush=True)
    print(f"is_science: {len(sci)}/{len(ok)} = {100*len(sci)/max(len(ok),1):.1f}%",
          flush=True)
    from collections import Counter
    dom = Counter(r.get("domain") or "n/a" for r in sci)
    print(f"domain dist: {dict(dom)}")
    print(f"\nSample science titles:")
    for r in sci[:10]:
        print(f"  [{r.get('domain'):>10s}] conf={r.get('confidence',0):.2f}  {r['title']}")
    print(f"\nSample non-science titles:")
    for r in [r for r in ok if not r.get("is_science")][:5]:
        print(f"  {r['title']}")


if __name__ == "__main__":
    asyncio.run(main())
