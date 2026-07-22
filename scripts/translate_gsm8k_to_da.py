"""Translate openai/gsm8k → Danish via Gemini 3.1 Flash Lite.

Both train (7473) and test (1319) splits. Resumable JSONL storing EN + DA
per row (parallel corpus).

GSM8K has TWO structural elements that must survive translation intact:
  1. `<<expr=result>>` calculator annotations inside the answer text
  2. `#### N` final-answer marker at the end of the answer

Numbers/equations inside those constructs must stay in digit form (not
wordified) so downstream eval scripts that regex `#### N` and rely on
digit expressions still work.

Usage:
  export GOOGLE_API_KEY=$(cat ~/gem)
  uv run --with google-genai --with datasets python \\
      scripts/translate_gsm8k_to_da.py \\
      --out /mnt/data2/gsm8k_da/full.jsonl --workers 30
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

TRANSLATE_PROMPT = """You are a professional English→Danish translator
specialising in mathematical word problems (GSM8K).

Translate the JSON object below to Danish. Keys stay `question` and `answer`.
Return ONLY the JSON object.

CRITICAL preservation rules:
  1. `<<expr=result>>` calculator annotations MUST stay VERBATIM. Do not
     change the syntax, the numbers inside, or the surrounding brackets.
     Example: `<<48/2=24>>` stays `<<48/2=24>>`.
  2. The final `#### N` answer marker MUST stay intact and at the end.
     Never remove it, never wordify N.
  3. All numeric expressions must stay as digits. Do NOT translate numbers
     to Danish words ("otte" for 8, "tres" for 60, etc.). GSM8K answers
     need parseable digit arithmetic.
  4. Proper names, units (dollars, hours, km, etc.) may be lightly
     localised (dollars → kroner is OK if the problem is currency-agnostic;
     otherwise keep dollars). Prefer keeping the source unit.

General rules:
  - Use natural idiomatic Danish for the prose framing of the problem
    and the connecting narration between calculations.
  - Preserve the step-by-step structure of the answer — one calculation
    per line matching the source.

INPUT (English):
{payload}

OUTPUT (Danish, same JSON schema):"""


def load_done_ids(out_path: Path) -> set[str]:
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


async def translate_row(client, sem, split, idx, row):
    en_pack = {"question": row["question"], "answer": row["answer"]}
    payload = json.dumps(en_pack, ensure_ascii=False, indent=2)
    prompt = TRANSLATE_PROMPT.format(payload=payload)
    row_id = f"{split}/{idx}"

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
                        "id": row_id, "split": split, "idx": idx,
                        "en": en_pack, "da": None,
                        "meta": {
                            "error": f"parse_fail:{str(e)[:120]}",
                            "raw": raw[:800],
                            "latency_s": dt,
                            "in_tok": in_tok, "out_tok": out_tok,
                            "attempt": attempt + 1,
                        },
                    }
                if not all(k in parsed and isinstance(parsed[k], str)
                           for k in ("question", "answer")):
                    if attempt < 3:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return {
                        "id": row_id, "split": split, "idx": idx,
                        "en": en_pack, "da": None,
                        "meta": {"error": "schema_bad", "raw": raw[:800],
                                 "latency_s": dt, "in_tok": in_tok,
                                 "out_tok": out_tok, "attempt": attempt + 1},
                    }
                # Sanity: the DA answer MUST still contain `####` — else the
                # downstream GSM eval script can't extract the gold answer.
                if "####" not in parsed["answer"]:
                    if attempt < 3:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return {
                        "id": row_id, "split": split, "idx": idx,
                        "en": en_pack, "da": None,
                        "meta": {"error": "hash_marker_lost", "raw": raw[:800],
                                 "latency_s": dt, "in_tok": in_tok,
                                 "out_tok": out_tok, "attempt": attempt + 1},
                    }
                return {
                    "id": row_id, "split": split, "idx": idx,
                    "en": en_pack, "da": parsed,
                    "meta": {"latency_s": dt, "in_tok": in_tok,
                             "out_tok": out_tok, "attempt": attempt + 1},
                }
            except Exception as e:
                err = str(e)[:200]
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {"id": row_id, "split": split, "idx": idx,
                        "en": en_pack, "da": None,
                        "meta": {"error": f"api_error:{err}",
                                 "attempt": attempt + 1}}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/mnt/data2/gsm8k_da/full.jsonl"))
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY", file=sys.stderr); sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"loading openai/gsm8k…", flush=True)
    ds_train = load_dataset("openai/gsm8k", "main", split="train")
    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    n_train = len(ds_train) if args.limit is None else min(args.limit, len(ds_train))
    n_test = len(ds_test) if args.limit is None else min(args.limit, len(ds_test))
    print(f"  train {n_train} + test {n_test} = {n_train+n_test}", flush=True)

    done = load_done_ids(args.out)
    print(f"  {len(done)} rows already complete → skipping", flush=True)

    todo = []
    for i in range(n_train):
        rid = f"train/{i}"
        if rid not in done:
            todo.append(("train", i, ds_train[i]))
    for i in range(n_test):
        rid = f"test/{i}"
        if rid not in done:
            todo.append(("test", i, ds_test[i]))
    print(f"  {len(todo)} rows to translate", flush=True)
    if not todo:
        return

    client = genai.Client(api_key=api_key)
    sem = asyncio.Semaphore(args.workers)

    t0 = time.time()
    n_done = n_ok = n_fail = 0
    total_in = total_out = 0
    tasks = [translate_row(client, sem, sp, ix, rw) for sp, ix, rw in todo]

    with args.out.open("a") as f:
        for coro in asyncio.as_completed(tasks):
            r = await coro
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            if r["da"] is not None:
                n_ok += 1
            else:
                n_fail += 1
            total_in += r["meta"].get("in_tok", 0) or 0
            total_out += r["meta"].get("out_tok", 0) or 0
            if n_done % 100 == 0 or n_done == len(todo):
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (len(todo) - n_done) / rate if rate > 0 else 0
                cost = total_in * 0.25 / 1e6 + total_out * 1.50 / 1e6
                print(f"  {n_done}/{len(todo)}  ok={n_ok} fail={n_fail}  "
                      f"{rate:.1f} rows/s  eta={eta:.0f}s  cost=${cost:.2f}",
                      flush=True)

    elapsed = time.time() - t0
    cost = total_in * 0.25 / 1e6 + total_out * 1.50 / 1e6
    print(f"\ndone in {elapsed:.0f}s; success={n_ok} fail={n_fail}")
    print(f"tokens: {total_in:,} in + {total_out:,} out; cost = ${cost:.2f}")
    print(f"output → {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
