"""Translate HellaSwag (Rowan/hellaswag) train + validation to Esperanto
via Gemini Flash Lite.

Preserves the 4-choice next-sentence structure. Test split is skipped
(labels are private/empty on HF).

Output JSONL rows contain: id, split, en_ctx, en_endings, eo_ctx,
eo_endings, label, activity_label, source_id.

Example:
    export GOOGLE_API_KEY=$(cat ~/gem)
    uv run python scripts/translate_hellaswag_gemini.py \\
        --out /mnt/data2/hellaswag_eo.jsonl --concurrency 48
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

PROMPT_TEMPLATE = """Translate this English commonsense scenario to Esperanto.

Preserve exactly:
  - the meaning and plausibility of the context
  - the four candidate next-sentence endings as separate lines
  - relative subject/pronoun consistency across endings

Output format — ONLY these lines, nothing else:
KUNTEKSTO: <esperanto context>
A: <esperanto ending 0>
B: <esperanto ending 1>
C: <esperanto ending 2>
D: <esperanto ending 3>

English:
KUNTEKSTO: {ctx}
A: {e0}
B: {e1}
C: {e2}
D: {e3}"""


_PREAMBLE_PATTERNS = [
    re.compile(r"^(jen|jena|jenajn?)\s+(la|estas)?\s*trad", re.I),
    re.compile(r"^```", re.M),
]


_LINE_RE = re.compile(r"^\s*([A-D])\s*[:\.\)]\s*(.+?)\s*$")
_CTX_RE = re.compile(r"^\s*(?:KUNTEKSTO|CONTEXT)\s*[:\.\)]\s*(.+?)\s*$", re.I)


def parse_response(text: str):
    if not text:
        return None
    lines = [l for l in text.strip().splitlines() if l.strip()]
    ctx = None
    endings = {}
    for line in lines:
        stripped = line.strip()
        m_c = _CTX_RE.match(stripped)
        if m_c and ctx is None:
            ctx = m_c.group(1).strip()
            continue
        m_e = _LINE_RE.match(stripped)
        if m_e:
            L = m_e.group(1)
            if L not in endings:
                endings[L] = m_e.group(2).strip()
    if ctx is None or len(endings) != 4:
        return None
    return ctx, [endings["A"], endings["B"], endings["C"], endings["D"]]


def looks_bad(en_ctx: str, en_endings: list[str], eo_ctx: str, eo_endings: list[str]) -> str | None:
    if not eo_ctx or len(eo_ctx) < 5:
        return "ctx_too_short"
    src_len = len(en_ctx) + sum(len(e) for e in en_endings)
    out_len = len(eo_ctx) + sum(len(e) for e in eo_endings)
    if src_len == 0:
        return "empty_source"
    ratio = out_len / src_len
    if ratio < 0.5 or ratio > 2.5:
        return f"len_ratio={ratio:.2f}"
    for pat in _PREAMBLE_PATTERNS:
        if pat.search(eo_ctx[:120]):
            return "preamble"
    return None


async def translate_row(
    client: genai.Client, model: str, semaphore: asyncio.Semaphore,
    row: dict, split: str,
) -> dict:
    ctx = row["ctx"]
    endings = row["endings"]
    prompt = PROMPT_TEMPLATE.format(
        ctx=ctx, e0=endings[0], e1=endings[1], e2=endings[2], e3=endings[3])
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
                        "id": row["ind"], "split": split,
                        "en_ctx": ctx, "en_endings": endings,
                        "eo_ctx": None, "eo_endings": None,
                        "label": int(row["label"]),
                        "activity_label": row.get("activity_label", ""),
                        "source_id": row.get("source_id", ""),
                        "raw_response": raw[:500],
                        "reject_reason": "parse_fail",
                    }
                eo_ctx, eo_endings = parsed
                bad = looks_bad(ctx, endings, eo_ctx, eo_endings)
                return {
                    "id": row["ind"], "split": split,
                    "en_ctx": ctx, "en_endings": endings,
                    "eo_ctx": eo_ctx, "eo_endings": eo_endings,
                    "label": int(row["label"]),
                    "activity_label": row.get("activity_label", ""),
                    "source_id": row.get("source_id", ""),
                    "reject_reason": bad,
                }
            except Exception as e:
                err = str(e)[:200]
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    "id": row["ind"], "split": split,
                    "en_ctx": ctx, "en_endings": endings,
                    "eo_ctx": None, "eo_endings": None,
                    "label": int(row["label"]),
                    "activity_label": row.get("activity_label", ""),
                    "source_id": row.get("source_id", ""),
                    "reject_reason": f"error:{err}",
                }
    return {}


async def run(args) -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    all_rows: list[tuple[str, dict]] = []
    print("loading Rowan/hellaswag...", flush=True)
    ds = load_dataset("Rowan/hellaswag")
    for split in ("train", "validation"):
        for row in ds[split]:
            all_rows.append((split, row))
    print(f"total rows: {len(all_rows):,}\n", flush=True)

    out_path = Path(args.out)
    done: set[int] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
                done.add(r["id"])
            except Exception:
                pass
        print(f"resume: {len(done):,} already done", flush=True)

    todo = [(s, r) for (s, r) in all_rows if r["ind"] not in done]
    print(f"todo: {len(todo):,}\n", flush=True)
    if not todo:
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    n_ok = n_rej = 0

    async def worker(split: str, row: dict) -> dict:
        return await translate_row(client, args.model, semaphore, row, split)

    with out_path.open("a") as fout:
        tasks = [asyncio.create_task(worker(s, r)) for (s, r) in todo]
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
    ap.add_argument("--out", type=Path, default=Path("/mnt/data2/hellaswag_eo.jsonl"))
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
