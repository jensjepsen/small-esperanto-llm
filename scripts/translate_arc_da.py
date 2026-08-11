"""Translate AI2 ARC (Challenge + Easy) to Danish via Gemini Flash Lite.

Preserves the multiple-choice structure: question + N choices (usually 4,
occasionally 3 or 5). answerKey and IDs pass through untouched.

Outputs JSONL with fields:
    id, source (arc_challenge | arc_easy), split (train | validation | test),
    en_question, en_choices [{label, text}],
    da_question, da_choices [{label, text}],
    answerKey, reject_reason

Resume-safe: skips rows whose id is already present in --out.

Example
-------
    export GOOGLE_API_KEY=$(cat ~/gem)
    uv run python scripts/translate_arc_da.py \\
        --out data/arc_da.jsonl --concurrency 48
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

PROMPT_TEMPLATE = """Oversæt dette engelske naturfags-flervalgsspørgsmål til dansk.

Bevar præcist:
  - spørgsmålets betydning og sværhedsgrad
  - alle tal, enheder, kemiske symboler og videnskabelig notation
  - de fire (eller tre/fem) svarmuligheder som separate linjer

Brug standard dansk videnskabelig terminologi (fx vand, energi, kraft, celle,
planter, dyr, solsystem, temperatur, magnet, elektricitet).

Outputformat — KUN disse linjer, intet andet:
SPØRGSMÅL: <dansk spørgsmål>
A: <dansk svar A>
B: <dansk svar B>
C: <dansk svar C>
D: <dansk svar D>

(Brug samme sæt bogstaver som input — hvis input bruger 1/2/3/4, brug de samme;
hvis kun tre svar, output kun tre; osv.)

Engelsk:
QUESTION: {question}
{choices}"""


_PREAMBLE_PATTERNS = [
    re.compile(r"^(her|følgende|dette)\s+(er|er\s+den)?\s*oversæt", re.I),
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
        # SPØRGSMÅL: ... (accept QUESTION: too for robustness)
        m_q = re.match(r"^\s*(?:SPØRGSMÅL|SPORGSMAL|QUESTION)\s*[:\.\)]\s*(.+?)\s*$", stripped, re.I)
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
    by_label = {c["label"]: c["text"] for c in choices}
    if set(by_label) != set(expected_labels):
        return None
    return question, [{"label": L, "text": by_label[L]} for L in expected_labels]


def looks_bad(en_q: str, en_choices: list[dict], da_q: str, da_choices: list[dict]) -> str | None:
    if not da_q or len(da_q) < 5:
        return "q_too_short"
    src_len = len(en_q) + sum(len(c["text"]) for c in en_choices)
    out_len = len(da_q) + sum(len(c["text"]) for c in da_choices)
    if src_len == 0:
        return "empty_source"
    ratio = out_len / src_len
    if ratio < 0.5 or ratio > 2.5:
        return f"len_ratio={ratio:.2f}"
    for pat in _PREAMBLE_PATTERNS:
        if pat.search(da_q[:120]):
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
                        "da_question": None, "da_choices": None,
                        "answerKey": row["answerKey"],
                        "raw_response": raw[:500],
                        "reject_reason": "parse_fail",
                    }
                da_q, da_choices = parsed
                bad = looks_bad(en_q, en_choices, da_q, da_choices)
                return {
                    "id": row["id"], "source": source, "split": split,
                    "en_question": en_q, "en_choices": en_choices,
                    "da_question": da_q, "da_choices": da_choices,
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
                    "da_question": None, "da_choices": None,
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("--out", type=Path, default=Path("data/arc_da.jsonl"))
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
