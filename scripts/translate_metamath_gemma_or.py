"""Translate MetaMathQA to Danish via gemma-3-12b on OpenRouter.

Same skeleton as translate_metamath_gemini.py (and translate_orcamath_gemini.py)
but via OpenRouter's gemma-3-12b-it rather than Gemini Flash Lite. Uses a
STRICT literal-translation prompt that prevents the fabrication failure mode
observed with gemma when using a naive "translate to Danish" prompt.

Output schema (per row, resumable via orig_idx):
    {orig_idx, type, original_question,
     q_en, q_da, a_en, a_da,
     input_tokens, output_tokens, cost, reject_reason?}

Usage:
    uv run --no-project --with datasets --with aiohttp \\
        python scripts/translate_metamath_gemma_or.py \\
        --out /mnt/data2/metamath_gsm_da.jsonl --concurrency 60
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

import aiohttp
from datasets import load_dataset


MODEL = "google/gemma-3-12b-it"
API   = "https://openrouter.ai/api/v1/chat/completions"

PROMPT_TEMPLATE = """You are a STRICT literal translator, not a math problem generator.

Your job: translate the English text below WORD FOR WORD into Danish. You are
translating, not rewriting.

CRITICAL — read carefully:
  * Preserve every named entity EXACTLY. If the source says "Sue", the Danish
    says "Sue". If it says "Mark", "Diego", "Evan" — those names stay. Never
    invent new names.
  * Preserve every SUBJECT of the problem. If the source is about football
    players buying equipment, the Danish MUST be about football players buying
    equipment — NOT about baking cakes, not about cars, not about anything
    else. Same objects, same scenario.
  * Preserve every number, every unit, every equation, every "#### N" marker
    at the end.
  * Preserve step-by-step structure and line breaks.

If you find yourself writing about a completely different topic than the
source (e.g. source is about a machine and dogs, but you're writing about
cookies and a birthday party), you are hallucinating — STOP and re-read the
source.

EXAMPLE OF WHAT NOT TO DO:
  SOURCE: "Evan's dog weighs 63 pounds; it weighs x times as much as Ivan's dog."
  BAD: "Diego bagte 12 kager til sin søster..." (wrong — this is a new problem)
  GOOD: "Evans hund vejer 63 pund; den vejer x gange så meget som Ivans hund."

Use natural Danish math terminology:
  ligning = equation, brøk = fraction, procent = percent, gennemsnit = mean,
  differens = difference, produkt = product, kvadratrod = square root,
  hypotenuse = hypotenuse, katete = leg, uligheden = inequality

Output ONLY these two blocks with these exact headers:

QUESTION:
<Danish translation of the question — preserving all names, numbers, scenario>

ANSWER:
<Danish translation of the answer — preserving all steps and the #### N marker>

No preamble, no code fences, no notes about your process.

SOURCE QUESTION:
{q}

SOURCE ANSWER:
{a}"""


PARSE_RE = re.compile(r"QUESTION\s*:\s*(.*?)\s*ANSWER\s*:\s*(.*)", re.S | re.I)

# Presence check: any of these = looks Danish. Absent = probably English
# passed through untranslated. Include æ/ø/å chars + high-frequency
# Danish function words that virtually every translation will contain.
DA_MARKERS_RE = re.compile(
    r"[æøåÆØÅ]|\b(er|og|hvis|det|den|af|til|på|for|som|har|kan|skal|"
    r"også|ikke|hvor|mange|hver|når|så|men|eller|blev|blive|været|"
    r"med|fra|mod|efter|nogle|nogle|alle|deres|hendes|hans|vil|vi|"
    r"få|får|siden|derfor|dvs|altså)\b",
    re.I,
)


def looks_danish(text: str) -> bool:
    """Reject strings that look like untranslated English (rare English
    parroting failure mode observed in ~2% of gemma-3-12b outputs).

    Very short strings (< 30 chars) pass automatically — no reliable signal.
    Longer strings must contain at least one Danish marker.
    """
    if not text or len(text) < 30:
        return True
    return bool(DA_MARKERS_RE.search(text))


def parse_output(text: str) -> tuple[str, str] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()
    m = PARSE_RE.search(text)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


async def translate_row(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                         key: str, row: dict) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content":
                       PROMPT_TEMPLATE.format(q=row["query"], a=row["response"])}],
        "temperature": 0.1,
        "max_tokens": 1800,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://claude-code-metamath",
        "X-Title": "MetaMathQA→DA",
    }
    base = {
        "orig_idx": row["orig_idx"],
        "type": row.get("type", ""),
        "original_question": row.get("original_question", ""),
        "q_en": row["query"],
        "a_en": row["response"],
    }
    async with sem:
        for attempt in range(4):
            try:
                async with session.post(
                    API, headers=headers, json=body, timeout=90,
                ) as resp:
                    data = await resp.json()
                if "choices" not in data:
                    err = json.dumps(data)[:180]
                    if attempt < 3:
                        await asyncio.sleep(2 ** attempt); continue
                    return {**base, "q_da": None, "a_da": None,
                            "input_tokens": 0, "output_tokens": 0, "cost": 0,
                            "reject_reason": f"api:{err}"}
                raw = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                in_t  = int(usage.get("prompt_tokens", 0))
                out_t = int(usage.get("completion_tokens", 0))
                cost  = float(usage.get("cost", 0) or 0)
                parsed = parse_output(raw)
                if parsed is None:
                    if attempt < 3:
                        await asyncio.sleep(2 ** attempt); continue
                    return {**base, "q_da": None, "a_da": None,
                            "input_tokens": in_t, "output_tokens": out_t,
                            "cost": cost, "reject_reason": "parse_fail",
                            "raw": raw[:300]}
                q_da, a_da = parsed
                # Reject untranslated pass-through: gemma sometimes
                # returns the English source verbatim in the Danish slot.
                if q_da == row["query"] or a_da == row["response"] \
                        or not looks_danish(q_da) or not looks_danish(a_da):
                    if attempt < 3:
                        await asyncio.sleep(2 ** attempt); continue
                    return {**base, "q_da": q_da, "a_da": a_da,
                            "input_tokens": in_t, "output_tokens": out_t,
                            "cost": cost, "reject_reason": "untranslated"}
                return {**base, "q_da": q_da, "a_da": a_da,
                        "input_tokens": in_t, "output_tokens": out_t,
                        "cost": cost}
            except Exception as e:
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt); continue
                return {**base, "q_da": None, "a_da": None,
                        "input_tokens": 0, "output_tokens": 0, "cost": 0,
                        "reject_reason": f"exc:{str(e)[:200]}"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of source rows (for smoke tests)")
    ap.add_argument("--include-math", action="store_true",
                    help="also include MATH_* subtypes (default: only GSM_*)")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--key-file", type=Path, default=Path("/home/jepsen/or"))
    args = ap.parse_args()

    key = args.key_file.read_text().strip()

    print("loading meta-math/MetaMathQA…", flush=True)
    full = load_dataset("meta-math/MetaMathQA", split="train")

    def keep(t: str) -> bool:
        if args.include_math:
            return True
        return t.startswith("GSM_")

    rows: list[dict] = []
    for i, r in enumerate(full):
        if not keep(r.get("type", "")):
            continue
        rows.append({
            "orig_idx": i,
            "type": r.get("type", ""),
            "query": r["query"],
            "response": r["response"],
            "original_question": r.get("original_question", ""),
        })
        if args.limit and len(rows) >= args.limit:
            break
    print(f"  kept {len(rows):,} rows after filter "
          f"({'include_math' if args.include_math else 'GSM_ only'})",
          flush=True)

    # Resume via orig_idx
    done: set[int] = set()
    prior_in = prior_out = 0
    prior_cost = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        for line in args.out.open():
            try:
                r = json.loads(line)
                done.add(r["orig_idx"])
                prior_in  += r.get("input_tokens", 0)
                prior_out += r.get("output_tokens", 0)
                prior_cost += r.get("cost", 0) or 0
            except Exception:
                pass
        print(f"  resume: {len(done):,} rows already translated (${prior_cost:.2f})",
              flush=True)

    todo = [r for r in rows if r["orig_idx"] not in done]
    print(f"  {len(todo):,} rows to translate", flush=True)
    if not todo:
        return

    t0 = time.time()
    n_ok = n_rej = 0
    tok_in = tok_out = 0
    cost_run = 0.0
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(args.concurrency)
        tasks = [asyncio.create_task(translate_row(session, sem, key, r))
                 for r in todo]
        with args.out.open("a") as fout:
            for coro in asyncio.as_completed(tasks):
                r = await coro
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                if r.get("reject_reason"):
                    n_rej += 1
                else:
                    n_ok += 1
                tok_in  += r.get("input_tokens", 0)
                tok_out += r.get("output_tokens", 0)
                cost_run += r.get("cost", 0) or 0
                total = n_ok + n_rej
                if total % args.log_every == 0 or total == len(todo):
                    el = time.time() - t0
                    rate = total / el
                    eta = (len(todo) - total) / max(rate, 1e-6)
                    print(f"  {total:,}/{len(todo):,}  ok={n_ok:,}  "
                          f"rej={n_rej:,} ({100*n_rej/total:.1f}%)  "
                          f"{rate:.1f}/s  eta={eta/60:.1f}min  "
                          f"tok={tok_in/1e6:.1f}M/{tok_out/1e6:.1f}M  "
                          f"$this={cost_run:.2f}  "
                          f"$total={cost_run+prior_cost:.2f}",
                          flush=True)

    el = time.time() - t0
    print(f"\ndone: {n_ok:,} translated, {n_rej:,} rejected in {el/60:.1f} min")
    print(f"cost: ${cost_run:.2f} this run, ${cost_run+prior_cost:.2f} total")


if __name__ == "__main__":
    asyncio.run(main())
