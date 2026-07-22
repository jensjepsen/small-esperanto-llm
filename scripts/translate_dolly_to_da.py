"""Translate Dolly-15k → Danish via Gemini 3.1 Flash Lite.

Resumable JSONL output: on restart, skips rows whose id is already present
with a successful translation. Failures (parse_fail, api_error) are retried.
Each row stored as {"id", "category", "en":{...}, "da":{...}, "meta":{...}}
so the file doubles as an EN↔DA parallel corpus.

Usage:
  export GOOGLE_API_KEY=$(cat ~/gem)
  uv run --with google-genai --with datasets python scripts/translate_dolly_to_da.py \\
      --out /mnt/data2/dolly_da_full.jsonl --workers 30
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from google import genai
from datasets import load_dataset


MODEL = "gemini-3.1-flash-lite"

TRANSLATE_PROMPT = """You are a professional English→Danish translator.
Translate the JSON object below to Danish. Preserve the JSON structure exactly
(same keys, no extra fields). Rules:
  - Use natural, idiomatic Danish, not a word-for-word rendering.
  - Keep proper names, numbers, dates, URLs, code identifiers verbatim.
  - Keep the same tone/register as the source (formal/casual/technical).
  - If a field is empty (empty string), keep it empty in the output.
  - Return ONLY the JSON object; no commentary before or after.

INPUT (English):
{payload}

OUTPUT (Danish, same JSON schema):"""


def load_done_ids(out_path: Path) -> set[int]:
    """Read existing output; keep ids of rows with successful translation."""
    done = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("da") is not None and not r.get("meta", {}).get("error"):
                done.add(r["id"])
    return done


async def translate_row(client, sem, idx, row):
    payload = json.dumps({
        "instruction": row["instruction"],
        "context": row["context"],
        "response": row["response"],
    }, ensure_ascii=False, indent=2)
    prompt = TRANSLATE_PROMPT.format(payload=payload)

    en_pack = {
        "instruction": row["instruction"],
        "context": row["context"],
        "response": row["response"],
    }

    async with sem:
        for attempt in range(4):
            try:
                t0 = time.time()
                resp = await asyncio.to_thread(
                    client.models.generate_content,
                    model=MODEL, contents=prompt,
                )
                dt = time.time() - t0
                raw = (resp.text or "").strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw
                    raw = raw.rsplit("```", 1)[0].strip()
                usage = getattr(resp, "usage_metadata", None)
                in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
                out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as e:
                    return {
                        "id": idx,
                        "category": row["category"],
                        "en": en_pack, "da": None,
                        "meta": {
                            "error": f"parse_fail:{str(e)[:120]}",
                            "raw": raw[:800],
                            "latency_s": dt,
                            "in_tok": in_tok, "out_tok": out_tok,
                            "attempt": attempt + 1,
                        },
                    }
                # Basic schema sanity: all 3 keys must be strings
                if not all(k in parsed and isinstance(parsed[k], str)
                           for k in ("instruction", "context", "response")):
                    if attempt < 3:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return {
                        "id": idx,
                        "category": row["category"],
                        "en": en_pack, "da": None,
                        "meta": {"error": "schema_bad",
                                 "raw": raw[:800],
                                 "latency_s": dt,
                                 "in_tok": in_tok, "out_tok": out_tok,
                                 "attempt": attempt + 1},
                    }
                return {
                    "id": idx,
                    "category": row["category"],
                    "en": en_pack,
                    "da": parsed,
                    "meta": {
                        "latency_s": dt,
                        "in_tok": in_tok, "out_tok": out_tok,
                        "attempt": attempt + 1,
                    },
                }
            except Exception as e:
                err = str(e)[:200]
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    "id": idx,
                    "category": row["category"],
                    "en": en_pack, "da": None,
                    "meta": {"error": f"api_error:{err}",
                             "attempt": attempt + 1},
                }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/mnt/data2/dolly_da_full.jsonl"))
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only translate this many rows (for testing)")
    args = ap.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY", file=sys.stderr)
        sys.exit(1)

    print(f"loading dolly-15k…", flush=True)
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    n_total = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"  {len(ds)} rows total, translating {n_total}", flush=True)

    done = load_done_ids(args.out)
    print(f"  {len(done)} rows already complete → skipping", flush=True)

    todo = [i for i in range(n_total) if i not in done]
    print(f"  {len(todo)} rows to translate", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    client = genai.Client(api_key=api_key)
    sem = asyncio.Semaphore(args.workers)

    t0 = time.time()
    n_done_local = 0
    n_success = 0
    n_fail = 0
    total_in = total_out = 0
    tasks = [translate_row(client, sem, i, ds[i]) for i in todo]

    with args.out.open("a") as f:
        for coro in asyncio.as_completed(tasks):
            r = await coro
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n_done_local += 1
            if r["da"] is not None:
                n_success += 1
            else:
                n_fail += 1
            total_in += r["meta"].get("in_tok", 0) or 0
            total_out += r["meta"].get("out_tok", 0) or 0
            if n_done_local % 100 == 0 or n_done_local == len(todo):
                elapsed = time.time() - t0
                rate = n_done_local / elapsed
                eta = (len(todo) - n_done_local) / rate if rate > 0 else 0
                cost = total_in * 0.25 / 1e6 + total_out * 1.50 / 1e6
                print(f"  {n_done_local}/{len(todo)}  ok={n_success} fail={n_fail}  "
                      f"{rate:.1f} rows/s  eta={eta:.0f}s  cost=${cost:.2f}",
                      flush=True)

    elapsed = time.time() - t0
    print(f"\ndone in {elapsed:.0f}s; success={n_success} fail={n_fail}")
    cost = total_in * 0.25 / 1e6 + total_out * 1.50 / 1e6
    print(f"tokens: {total_in:,} in + {total_out:,} out; cost = ${cost:.2f}")
    print(f"output → {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
