"""Reword all Danish wp-v2 questions in natural GSM8K-style Danish via
gemma-3-12b-it on OpenRouter. Assistant answer (recipe-style solution)
is kept verbatim — only the QUESTION is rewritten to strip artificial
recipe cues and add situational flavor.

Reject filter validated on 100-row smoke: ~83% first-pass clean.
Rejects are retried once with different seed/temperature.

Output JSONL, one row per successful rewrite:
    {orig_idx, q_orig, q_new, a, source_tags, attempts, tokens_in, tokens_out}

Usage:
    python scripts/gen_wp_reword.py --out /mnt/data2/wp_reword_v1.jsonl \\
        --concurrency 50 --report-every 500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import aiohttp
from datasets import load_dataset

# Reuse the reject-filter logic verified on the smoke:
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_wp_rephrase import check, PROMPT  # noqa: E402

API = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemma-3-12b-it"


def get_qa(row):
    q = next(m["content"] for m in row["messages"] if m["role"] == "user")
    a = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    return q, a


async def call_gemma(session, sem, key, prompt, temperature, max_tokens=400):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://claude-code-wp-reword",
        "X-Title": "wp-v2 → GSM8K-style",
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
                    # sometimes a rate/moderation error surfaces here
                    err = j.get("error", {}).get("message", "unknown")
                    await asyncio.sleep(backoff)
            except (asyncio.TimeoutError, aiohttp.ClientError):
                await asyncio.sleep(backoff)
    return None


async def reword_row(session, sem, key, idx, row):
    q, a = get_qa(row)
    prompt = PROMPT.format(q=q, a=a)

    # First attempt at temp 0.3 (as in the smoke).
    r = await call_gemma(session, sem, key, prompt, temperature=0.3)
    if r is None:
        return {"orig_idx": idx, "q_orig": q, "a": a, "q_new": None,
                "status": "api_fail", "attempts": 1, "in": 0, "out": 0}
    reason = check(q, r["text"])
    if reason is None:
        return {"orig_idx": idx, "q_orig": q, "a": a, "q_new": r["text"],
                "status": "ok", "attempts": 1,
                "in": r["in"], "out": r["out"]}

    # Retry once at higher temp for diversity.
    r2 = await call_gemma(session, sem, key, prompt, temperature=0.7)
    tokens_in  = r["in"]  + (r2["in"]  if r2 else 0)
    tokens_out = r["out"] + (r2["out"] if r2 else 0)
    if r2 is None:
        return {"orig_idx": idx, "q_orig": q, "a": a, "q_new": None,
                "status": f"retry_api_fail:{reason}", "attempts": 2,
                "in": tokens_in, "out": tokens_out}
    reason2 = check(q, r2["text"])
    if reason2 is None:
        return {"orig_idx": idx, "q_orig": q, "a": a, "q_new": r2["text"],
                "status": "ok_retry", "attempts": 2,
                "in": tokens_in, "out": tokens_out}
    # Both rejected — save the first attempt's text for debugging.
    return {"orig_idx": idx, "q_orig": q, "a": a, "q_new": r["text"],
            "q_retry": r2["text"], "status": f"reject:{reason}|{reason2}",
            "attempts": 2, "in": tokens_in, "out": tokens_out}


def load_done_indices(out_path: Path) -> set[int]:
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="jensjepsen/danish-word-problems-v2")
    ap.add_argument("--config", default="sft")
    ap.add_argument("--split", default="train")
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--report-every", type=int, default=500)
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--key-file", type=Path, default=Path("/home/jepsen/or"))
    args = ap.parse_args()

    key = args.key_file.read_text().strip()

    ds = load_dataset(args.dataset, args.config, split=args.split)
    total = len(ds) if args.n == 0 else min(args.n, len(ds))
    print(f"total rows: {total:,}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_indices(out_path)
    print(f"already done: {len(done):,}", flush=True)

    todo_indices = [i for i in range(total) if i not in done]
    print(f"todo: {len(todo_indices):,}", flush=True)
    if not todo_indices:
        print("all done")
        return

    sem = asyncio.Semaphore(args.concurrency)
    n_ok = n_retry_ok = n_reject = n_api_fail = 0
    tok_in = tok_out = 0
    t0 = time.time()
    lock = asyncio.Lock()

    async def process_and_write(idx, session, out_f):
        nonlocal n_ok, n_retry_ok, n_reject, n_api_fail, tok_in, tok_out
        try:
            row = ds[idx]
        except Exception:
            return
        result = await reword_row(session, sem, key, idx, row)
        async with lock:
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            if result["status"] == "ok":
                n_ok += 1
            elif result["status"] == "ok_retry":
                n_retry_ok += 1
            elif result["status"].startswith("api_fail") or result["status"].startswith("retry_api_fail"):
                n_api_fail += 1
            else:
                n_reject += 1
            tok_in += result.get("in", 0)
            tok_out += result.get("out", 0)
            done_now = n_ok + n_retry_ok + n_reject + n_api_fail
            if done_now % args.report_every == 0 or done_now == len(todo_indices):
                el = time.time() - t0
                eta = el * (len(todo_indices) - done_now) / max(done_now, 1)
                # cost estimate ($0.10/M in + $0.40/M out — rough gemma-3-12b)
                cost = tok_in * 0.10 / 1e6 + tok_out * 0.40 / 1e6
                print(
                    f"[{done_now:6d}/{len(todo_indices)}] "
                    f"ok={n_ok} retry_ok={n_retry_ok} reject={n_reject} "
                    f"api_fail={n_api_fail}  "
                    f"pass={100*(n_ok+n_retry_ok)/done_now:.1f}%  "
                    f"tokens={tok_in/1e6:.2f}M/{tok_out/1e6:.2f}M  "
                    f"cost≈${cost:.2f}  "
                    f"eta={eta/60:.0f}m", flush=True)

    async with aiohttp.ClientSession() as session:
        with out_path.open("a") as out_f:
            tasks = [process_and_write(i, session, out_f) for i in todo_indices]
            # Process concurrently — semaphore inside call_gemma limits real
            # concurrency; asyncio.gather just fans them all out.
            await asyncio.gather(*tasks)

    print("\n=== done ===")
    print(f"ok:         {n_ok:,}")
    print(f"ok_retry:   {n_retry_ok:,}")
    print(f"reject:     {n_reject:,}")
    print(f"api_fail:   {n_api_fail:,}")
    print(f"tokens:     in={tok_in/1e6:.2f}M  out={tok_out/1e6:.2f}M")
    print(f"cost est:   ${tok_in * 0.10 / 1e6 + tok_out * 0.40 / 1e6:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
