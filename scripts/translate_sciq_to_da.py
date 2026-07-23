"""Translate allenai/sciq → Danish via Gemini 3.1 Flash Lite.

All 3 splits (train 11679 + validation 1000 + test 1000). Resumable JSONL
storing EN + DA per row (parallel corpus).

SciQ preservation rules baked into the prompt:
  1. Four options (correct + 3 distractors) MUST stay as four DISTINCT strings
  2. Scientific terms rendered in standard Danish vocabulary
  3. Numbers, units, chemical formulas kept verbatim
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

TRANSLATE_PROMPT = """You are a professional English→Danish translator specialising
in science exam questions (SciQ multiple-choice format).

Translate the JSON object below to Danish. Preserve the JSON schema EXACTLY
(same keys: question, correct_answer, distractor1, distractor2, distractor3, support).
Return ONLY the JSON object.

CRITICAL rules for multi-choice preservation:
  - Keep the four answer options (correct_answer + 3 distractors) as FOUR
    DISTINCT strings in Danish. Do NOT merge synonyms, do NOT drop any option.
  - Preserve source distractor content faithfully — even if a distractor looks
    like a typo (e.g. "light source4"), translate it literally, don't clean up.
  - Keep proper science terms accurate — mesophile → mesofil, mitochondria →
    mitokondrier, gymnosperms → nøgenfrøede planter, etc. Prefer standard
    Danish scientific vocabulary.
  - Keep numbers, units, chemical formulas verbatim.
  - Support text: translate the full explanation to natural Danish.
  - If a field is empty, keep it empty.

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
    en_pack = {
        "question": row["question"],
        "correct_answer": row["correct_answer"],
        "distractor1": row["distractor1"],
        "distractor2": row["distractor2"],
        "distractor3": row["distractor3"],
        "support": row["support"],
    }
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
                        "meta": {"error": f"parse_fail:{str(e)[:120]}",
                                 "raw": raw[:800], "latency_s": dt,
                                 "in_tok": in_tok, "out_tok": out_tok,
                                 "attempt": attempt + 1},
                    }
                required = ("question","correct_answer","distractor1",
                            "distractor2","distractor3","support")
                if not all(k in parsed and isinstance(parsed[k], str)
                           for k in required):
                    if attempt < 3:
                        await asyncio.sleep(2 ** attempt); continue
                    return {"id": row_id, "split": split, "idx": idx,
                            "en": en_pack, "da": None,
                            "meta": {"error": "schema_bad", "raw": raw[:800],
                                     "latency_s": dt, "in_tok": in_tok,
                                     "out_tok": out_tok, "attempt": attempt + 1}}
                # Distinct-4-options sanity check — retry if merged
                opts = {parsed["correct_answer"].lower().strip(),
                        parsed["distractor1"].lower().strip(),
                        parsed["distractor2"].lower().strip(),
                        parsed["distractor3"].lower().strip()}
                if len(opts) < 4 and attempt < 3:
                    await asyncio.sleep(2 ** attempt); continue
                return {"id": row_id, "split": split, "idx": idx,
                        "en": en_pack, "da": parsed,
                        "meta": {"latency_s": dt, "in_tok": in_tok,
                                 "out_tok": out_tok, "attempt": attempt + 1,
                                 "distinct_opts": len(opts)}}
            except Exception as e:
                err = str(e)[:200]
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt); continue
                return {"id": row_id, "split": split, "idx": idx,
                        "en": en_pack, "da": None,
                        "meta": {"error": f"api_error:{err}",
                                 "attempt": attempt + 1}}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/mnt/data2/sciq_da/full.jsonl"))
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY", file=sys.stderr); sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("loading allenai/sciq…", flush=True)
    splits = {}
    for s in ("train","validation","test"):
        splits[s] = load_dataset("allenai/sciq", split=s)
        if args.limit is not None:
            splits[s] = splits[s].select(range(min(args.limit, len(splits[s]))))
    total = sum(len(v) for v in splits.values())
    print(f"  train={len(splits['train'])}  val={len(splits['validation'])}  "
          f"test={len(splits['test'])}  total={total}", flush=True)

    done = load_done_ids(args.out)
    print(f"  {len(done)} rows already complete → skipping", flush=True)
    todo = []
    for split_name, ds in splits.items():
        for i in range(len(ds)):
            rid = f"{split_name}/{i}"
            if rid not in done:
                todo.append((split_name, i, ds[i]))
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
            if r["da"] is not None: n_ok += 1
            else: n_fail += 1
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
