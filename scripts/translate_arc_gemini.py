"""Translate AI2 ARC (Challenge + Easy) to Esperanto via Gemini Flash Lite.

Preserves the multiple-choice structure: question + N choices (usually 4,
occasionally 3 or 5). answerKey and IDs pass through untouched.

Outputs JSONL with fields:
    id, source (arc_challenge | arc_easy), split (train | validation | test),
    en_question, en_choices [{label, text}],
    eo_question, eo_choices [{label, text}],
    answerKey, reject_reason

Resume-safe: skips rows whose id is already present in --out.

Example
-------
    export GOOGLE_API_KEY=$(cat ~/gem)
    uv run python scripts/translate_arc_gemini.py \\
        --out /mnt/data2/arc_eo.jsonl --concurrency 48
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

PROMPT_TEMPLATE = """Translate this English grade-school science multiple-choice question to Esperanto.

Preserve exactly:
  - the meaning and difficulty of the question
  - any numbers, units, chemical symbols, and scientific notation
  - the four (or three/five) answer choices as separate lines

Use standard Esperanto science terminology (e.g. akvo, energio, forto, ĉelo,
plantoj, bestoj, sunsistemo, temperaturo, magneto, elektro).

Output format — ONLY these lines, nothing else:
DEMANDO: <esperanto question>
A: <esperanto choice A>
B: <esperanto choice B>
C: <esperanto choice C>
D: <esperanto choice D>

(Use the same set of letters as the input — if it uses 1/2/3/4 use those; if
only three choices, output only three; etc.)

English:
DEMANDO: {question}
{choices}"""


_PREAMBLE_PATTERNS = [
    re.compile(r"^(jen|jena|jenajn?)\s+(la|estas)?\s*trad", re.I),
    re.compile(r"^```", re.M),
    re.compile(r"^(here|the)\s+is\s+the\s+trans", re.I),
]

_LINE_RE = re.compile(r"^\s*([A-Z0-9]+)\s*[:\.\)]\s*(.+?)\s*$")


def parse_response(text: str, expected_labels: list[str]) -> tuple[str, list[dict]] | None:
    """Parse the model's structured response. Returns (question, choices) or None on failure."""
    if not text:
        return None
    lines = [l for l in text.strip().splitlines() if l.strip()]
    question = None
    choices: list[dict] = []
    seen_labels: set[str] = set()
    for line in lines:
        stripped = line.strip()
        # DEMANDO: ...
        m_q = re.match(r"^\s*(?:DEMANDO|QUESTION)\s*[:\.\)]\s*(.+?)\s*$", stripped, re.I)
        if m_q and question is None:
            question = m_q.group(1).strip()
            continue
        m_c = _LINE_RE.match(stripped)
        if m_c:
            label = m_c.group(1).strip()
            if label in expected_labels and label not in seen_labels:
                choices.append({"label": label, "text": m_c.group(2).strip()})
                seen_labels.add(label)
    if question is None or len(choices) != len(expected_labels):
        return None
    # Reorder choices to match expected_labels order
    by_label = {c["label"]: c["text"] for c in choices}
    if set(by_label) != set(expected_labels):
        return None
    return question, [{"label": L, "text": by_label[L]} for L in expected_labels]


def looks_bad(en_q: str, en_choices: list[dict], eo_q: str, eo_choices: list[dict]) -> str | None:
    if not eo_q or len(eo_q) < 5:
        return "q_too_short"
    src_len = len(en_q) + sum(len(c["text"]) for c in en_choices)
    out_len = len(eo_q) + sum(len(c["text"]) for c in eo_choices)
    if src_len == 0:
        return "empty_source"
    ratio = out_len / src_len
    if ratio < 0.5 or ratio > 2.5:
        return f"len_ratio={ratio:.2f}"
    for pat in _PREAMBLE_PATTERNS:
        if pat.search(eo_q[:120]):
            return "preamble"
    return None


async def translate_row(
    client: genai.Client, model: str, semaphore: asyncio.Semaphore,
    row: dict, source: str, split: str,
) -> dict:
    en_q = row["question"]
    en_choices = [{"label": L, "text": T}
                  for L, T in zip(row["choices"]["label"], row["choices"]["text"])]
    labels = [c["label"] for c in en_choices]
    choices_str = "\n".join(f"{c['label']}: {c['text']}" for c in en_choices)
    prompt = PROMPT_TEMPLATE.format(question=en_q, choices=choices_str)

    async with semaphore:
        for attempt in range(3):
            try:
                resp = await asyncio.to_thread(
                    client.models.generate_content, model=model, contents=prompt)
                raw = (resp.text or "").strip()
                parsed = parse_response(raw, labels)
                if parsed is None:
                    return {
                        "id": row["id"], "source": source, "split": split,
                        "en_question": en_q, "en_choices": en_choices,
                        "eo_question": None, "eo_choices": None,
                        "answerKey": row["answerKey"],
                        "raw_response": raw[:500],
                        "reject_reason": "parse_fail",
                    }
                eo_q, eo_choices = parsed
                bad = looks_bad(en_q, en_choices, eo_q, eo_choices)
                return {
                    "id": row["id"], "source": source, "split": split,
                    "en_question": en_q, "en_choices": en_choices,
                    "eo_question": eo_q, "eo_choices": eo_choices,
                    "answerKey": row["answerKey"],
                    "reject_reason": bad,
                }
            except Exception as e:
                err = str(e)[:200]
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    "id": row["id"], "source": source, "split": split,
                    "en_question": en_q, "en_choices": en_choices,
                    "eo_question": None, "eo_choices": None,
                    "answerKey": row["answerKey"],
                    "reject_reason": f"error:{err}",
                }
    return {}


async def run(args) -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    all_rows: list[tuple[str, str, dict]] = []
    for cfg, tag in (("ARC-Challenge", "arc_challenge"), ("ARC-Easy", "arc_easy")):
        print(f"loading {cfg}...", flush=True)
        ds = load_dataset("allenai/ai2_arc", cfg)
        for split in ("train", "validation", "test"):
            for row in ds[split]:
                all_rows.append((tag, split, row))
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

    todo = [(t, s, r) for (t, s, r) in all_rows if r["id"] not in done]
    print(f"todo: {len(todo):,}\n", flush=True)
    if not todo:
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    n_ok = n_rej = 0

    async def worker(tag: str, split: str, row: dict) -> dict:
        return await translate_row(client, args.model, semaphore, row, tag, split)

    with out_path.open("a") as fout:
        tasks = [asyncio.create_task(worker(t, s, r)) for (t, s, r) in todo]
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
    ap.add_argument("--out", type=Path, default=Path("/mnt/data2/arc_eo.jsonl"))
    ap.add_argument("--model", default="gemini-flash-lite-latest")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
