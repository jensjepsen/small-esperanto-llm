"""Post-hoc LLM judge for wp_reword_v1.jsonl.

For each successfully-rewritten row, ask Gemma-3-12b-it (via OpenRouter)
whether q_orig and q_new are semantically equivalent — i.e. does the
reword ask for the same thing, using the same facts, arriving at the
same #### N answer? Outputs a JSONL with the verdict per row.

Catches the semantic-drift cases the regex filter misses:
- role/name-to-number swaps
- fraction vs ratio distortions ("en tredjedel" for "1:3")
- structural-connector changes ("hver" vs "tilsammen")
- invented time offsets that contradict "unknown" premises
- underspecification (missing info required to solve)

Usage:
    python scripts/judge_wp_reword.py \\
        --in /mnt/data2/wp_reword_v1.jsonl \\
        --out /mnt/data2/wp_reword_v1.judged.jsonl \\
        --concurrency 200 --report-every 500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import aiohttp

API = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3.1-flash-lite"

JUDGE_PROMPT = """Du får to matematik-tekstopgaver på dansk og facit-tallet.

ORIGINAL:
{q_orig}

OMSKREVET:
{q_new}

FACIT: {gold}

Er OMSKREVET semantisk ækvivalent med ORIGINAL — dvs.:
1) Alle nødvendige oplysninger fra ORIGINAL er bevaret i OMSKREVET
2) OMSKREVET introducerer INGEN ny information der ændrer opgaven (fx tilføjer et tal der ikke var der før, eller ændrer forhold-fortolkning)
3) OMSKREVET spørger om det samme
4) OMSKREVET kan løses til facit-tallet {gold}

Særlige faldgruber at kontrollere:
- Forhold som "3:4" er IKKE det samme som "en tredjedel" eller "3/10"
- "Hver gruppe har X" er IKKE det samme som "grupperne har X tilsammen"
- Hvis ORIGINAL siger "ukendt antal timer", må OMSKREVET ikke sætte det til fx "en time senere"
- Hvis roller er byttet om (fx "Merete har 4" vs "Merete har 3"), er det ikke ækvivalent

Svar KUN i formatet:
VERDICT: JA
eller
VERDICT: NEJ
GRUND: <én kort sætning>"""


VERDICT_RE = re.compile(r"VERDICT:\s*(JA|NEJ)", re.IGNORECASE)
REASON_RE = re.compile(r"GRUND:\s*(.+?)(?:\n|$)", re.IGNORECASE)
GOLD_RE = re.compile(r"####\s*(-?\d[\d,\.]*)")


def extract_gold(a: str) -> str:
    m = GOLD_RE.search(a)
    return m.group(1) if m else "?"


async def call_gemma(session, sem, key, prompt):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 80,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://claude-code-wp-judge",
        "X-Title": "wp reword judge",
    }
    async with sem:
        for backoff in [1, 3, 8, 20]:
            try:
                async with session.post(API, headers=headers, json=body, timeout=90) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(backoff * 2)
                        continue
                    j = await resp.json()
                    if "choices" in j and j["choices"]:
                        usage = j.get("usage", {}) or {}
                        return {
                            "text": j["choices"][0]["message"]["content"].strip(),
                            "in":   int(usage.get("prompt_tokens", 0)),
                            "out":  int(usage.get("completion_tokens", 0)),
                        }
                    await asyncio.sleep(backoff)
            except (asyncio.TimeoutError, aiohttp.ClientError):
                await asyncio.sleep(backoff)
    return None


def parse_verdict(text: str) -> tuple[str, str]:
    if not text:
        return "unknown", ""
    m = VERDICT_RE.search(text)
    v = m.group(1).upper() if m else "unknown"
    r = REASON_RE.search(text)
    reason = r.group(1).strip() if r else ""
    return v, reason


async def judge_row(session, sem, key, row):
    q_orig = row["q_orig"]
    q_new  = row["q_new"]
    gold   = extract_gold(row.get("a", ""))
    prompt = JUDGE_PROMPT.format(q_orig=q_orig, q_new=q_new, gold=gold)
    r = await call_gemma(session, sem, key, prompt)
    if r is None:
        return {"orig_idx": row["orig_idx"], "verdict": "api_fail",
                "reason": "", "raw": "", "in": 0, "out": 0}
    v, reason = parse_verdict(r["text"])
    return {"orig_idx": row["orig_idx"], "verdict": v, "reason": reason,
            "raw": r["text"], "in": r["in"], "out": r["out"]}


def load_done(out_path: Path) -> set[int]:
    if not out_path.exists():
        return set()
    seen = set()
    with out_path.open() as f:
        for line in f:
            try:
                seen.add(json.loads(line)["orig_idx"])
            except Exception:
                pass
    return seen


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--report-every", type=int, default=500)
    ap.add_argument("--key-file", type=Path, default=Path("/home/jepsen/or"))
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    key = args.key_file.read_text().strip()

    # Load rewords, skip already-judged.
    rows_all: list[dict] = []
    with open(args.in_path) as f:
        for line in f:
            r = json.loads(line)
            # Only judge rows the reword phase considered good.
            if r.get("status") in ("ok", "ok_retry") and r.get("q_new"):
                rows_all.append(r)
    if args.n > 0:
        rows_all = rows_all[:args.n]
    done = load_done(Path(args.out))
    todo = [r for r in rows_all if r["orig_idx"] not in done]
    print(f"total rewrites: {len(rows_all):,}  already judged: {len(done):,}  todo: {len(todo):,}",
          flush=True)
    if not todo:
        print("all done"); return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    n_ja = n_nej = n_unknown = n_api_fail = 0
    tok_in = tok_out = 0
    t0 = time.time()

    async def process(row, session, f_out):
        nonlocal n_ja, n_nej, n_unknown, n_api_fail, tok_in, tok_out
        res = await judge_row(session, sem, key, row)
        async with lock:
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
            f_out.flush()
            if res["verdict"] == "JA":
                n_ja += 1
            elif res["verdict"] == "NEJ":
                n_nej += 1
            elif res["verdict"] == "api_fail":
                n_api_fail += 1
            else:
                n_unknown += 1
            tok_in  += res.get("in", 0)
            tok_out += res.get("out", 0)
            done_now = n_ja + n_nej + n_unknown + n_api_fail
            if done_now % args.report_every == 0 or done_now == len(todo):
                el = time.time() - t0
                eta = el * (len(todo) - done_now) / max(done_now, 1)
                cost = tok_in * 0.25 / 1e6 + tok_out * 1.50 / 1e6  # gemini-3.1-flash-lite
                print(
                    f"[{done_now:6d}/{len(todo)}] "
                    f"JA={n_ja} NEJ={n_nej} unk={n_unknown} api_fail={n_api_fail}  "
                    f"pass={100*n_ja/done_now:.1f}%  "
                    f"cost=${cost:.2f}  eta={eta/60:.0f}m", flush=True)

    async with aiohttp.ClientSession() as session:
        with out_path.open("a") as f_out:
            tasks = [process(r, session, f_out) for r in todo]
            await asyncio.gather(*tasks)

    print("\n=== judge done ===")
    print(f"JA:        {n_ja:,}")
    print(f"NEJ:       {n_nej:,}")
    print(f"unknown:   {n_unknown:,}  (unparseable verdict)")
    print(f"api_fail:  {n_api_fail:,}")
    print(f"cost:      ${tok_in * 0.25 / 1e6 + tok_out * 1.50 / 1e6:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
