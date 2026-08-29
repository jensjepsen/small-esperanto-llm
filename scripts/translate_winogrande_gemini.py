"""Translate Winogrande (allenai/winogrande, xl config) train + validation
to Esperanto via Gemini Flash Lite.

Preserves the `_` blank marker in the sentence, plus option1/option2 and
answer (1/2). Test split has private labels — skipped.

Output JSONL rows contain: id, split, en_sentence, en_option1, en_option2,
eo_sentence, eo_option1, eo_option2, answer.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset
from google import genai
from google.genai import types

PROMPT_TEMPLATE = """Translate this English Winograd-schema fill-in-the-blank to Esperanto.

Preserve exactly:
  - the `_` character where the blank goes (this is critical)
  - the two named options (usually people or entities)
  - the pronoun resolution ambiguity

Output format — ONLY these lines, nothing else:
FRAZO: <esperanto sentence with _ where the blank is>
OPCIO1: <esperanto option 1>
OPCIO2: <esperanto option 2>

English:
FRAZO: {sentence}
OPCIO1: {option1}
OPCIO2: {option2}"""


_PREAMBLE_PATTERNS = [
    re.compile(r"^(jen|jena|jenajn?)\s+(la|estas)?\s*trad", re.I),
    re.compile(r"^```", re.M),
]

_FIELD_RES = {
    "sent": re.compile(r"^\s*(?:FRAZO|SENTENCE)\s*[:\.\)]\s*(.+?)\s*$", re.I),
    "op1":  re.compile(r"^\s*(?:OPCIO1|OPTION1|OPCIO 1|OPTION 1)\s*[:\.\)]\s*(.+?)\s*$", re.I),
    "op2":  re.compile(r"^\s*(?:OPCIO2|OPTION2|OPCIO 2|OPTION 2)\s*[:\.\)]\s*(.+?)\s*$", re.I),
}


def parse_response(text: str):
    if not text:
        return None
    lines = [l for l in text.strip().splitlines() if l.strip()]
    result = {}
    for line in lines:
        stripped = line.strip()
        for key, pat in _FIELD_RES.items():
            m = pat.match(stripped)
            if m and key not in result:
                result[key] = m.group(1).strip()
                break
    if len(result) != 3:
        return None
    if "_" not in result["sent"]:
        return None  # blank marker dropped by translation
    return result["sent"], result["op1"], result["op2"]


def looks_bad(en_sent: str, en_op1: str, en_op2: str,
              eo_sent: str, eo_op1: str, eo_op2: str) -> str | None:
    if not eo_sent or not eo_op1 or not eo_op2:
        return "missing_field"
    src_len = len(en_sent) + len(en_op1) + len(en_op2)
    out_len = len(eo_sent) + len(eo_op1) + len(eo_op2)
    if src_len == 0:
        return "empty_source"
    ratio = out_len / src_len
    if ratio < 0.5 or ratio > 2.5:
        return f"len_ratio={ratio:.2f}"
    for pat in _PREAMBLE_PATTERNS:
        if pat.search(eo_sent[:120]):
            return "preamble"
    if "_" not in eo_sent:
        return "no_blank"
    return None


async def translate_row(
    client: genai.Client, model: str, semaphore: asyncio.Semaphore,
    row: dict, split: str, idx: int,
) -> dict:
    sent = row["sentence"]
    op1 = row["option1"]
    op2 = row["option2"]
    ans = int(row["answer"]) if row["answer"] else 0
    prompt = PROMPT_TEMPLATE.format(sentence=sent, option1=op1, option2=op2)
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await asyncio.to_thread(
                    client.models.generate_content, model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0)))
                raw = (resp.text or "").strip()
                parsed = parse_response(raw)
                if parsed is None:
                    return {
                        "id": idx, "split": split,
                        "en_sentence": sent, "en_option1": op1, "en_option2": op2,
                        "eo_sentence": None, "eo_option1": None, "eo_option2": None,
                        "answer": ans,
                        "raw_response": raw[:500],
                        "reject_reason": "parse_fail",
                    }
                eo_sent, eo_op1, eo_op2 = parsed
                bad = looks_bad(sent, op1, op2, eo_sent, eo_op1, eo_op2)
                return {
                    "id": idx, "split": split,
                    "en_sentence": sent, "en_option1": op1, "en_option2": op2,
                    "eo_sentence": eo_sent, "eo_option1": eo_op1, "eo_option2": eo_op2,
                    "answer": ans,
                    "reject_reason": bad,
                }
            except Exception as e:
                err = str(e)[:200]
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    "id": idx, "split": split,
                    "en_sentence": sent, "en_option1": op1, "en_option2": op2,
                    "eo_sentence": None, "eo_option1": None, "eo_option2": None,
                    "answer": ans,
                    "reject_reason": f"error:{err}",
                }
    return {}


async def run(args) -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    all_rows: list[tuple[str, int, dict]] = []
    print("loading allenai/winogrande [winogrande_xl]...", flush=True)
    ds = load_dataset("allenai/winogrande", "winogrande_xl")
    for split in ("train", "validation"):
        for i, row in enumerate(ds[split]):
            # Sentence-position-based idx; ensures uniqueness across splits
            uid = f"{split}-{i}"
            all_rows.append((split, uid, row))
    print(f"total rows: {len(all_rows):,}\n", flush=True)

    out_path = Path(args.out)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
                done.add(r["id"])
            except Exception:
                pass
        print(f"resume: {len(done):,} already done", flush=True)

    todo = [(s, uid, r) for (s, uid, r) in all_rows if uid not in done]
    print(f"todo: {len(todo):,}\n", flush=True)
    if not todo:
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    n_ok = n_rej = 0

    async def worker(split: str, uid: str, row: dict) -> dict:
        return await translate_row(client, args.model, semaphore, row, split, uid)

    with out_path.open("a") as fout:
        tasks = [asyncio.create_task(worker(s, uid, r)) for (s, uid, r) in todo]
        for coro in asyncio.as_completed(tasks):
            r = await coro
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            if r.get("reject_reason"):
                n_rej += 1
            else:
                n_ok += 1
            total = n_ok + n_rej
            if total % args.log_every == 0:
                el = time.time() - t0
                rate = total / el
                eta = (len(todo) - total) / rate
                print(f"  {total:,}/{len(todo):,}  ok={n_ok:,}  "
                      f"rej={n_rej:,} ({100*n_rej/total:.1f}%)  "
                      f"rate={rate:.1f}/s  eta={eta/60:.1f}min", flush=True)

    print(f"\ndone: {n_ok:,} translated, {n_rej:,} rejected in "
          f"{(time.time()-t0)/60:.1f} min → {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("/mnt/data2/winogrande_eo.jsonl"))
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
