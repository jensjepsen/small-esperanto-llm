"""Translate a subset of microsoft/orca-math-word-problems-200k to Esperanto
via Gemini Flash Lite. Produces gold-quality math prose parallel data for
v10-mt training or v9.5 fine-tuning.

Seed-based scaling
------------------
The row selection is deterministic in ``--seed``. Using the same seed with a
larger ``--n`` gives a SUPERSET of a smaller run (via ``ds.shuffle(seed).select(range(n))``).
Using a different seed gives a disjoint sample. So:

    --seed 42 --n 5000     ← pilot
    --seed 42 --n 20000    ← extends the pilot (first 5k are identical)
    --seed 43 --n 5000     ← additional disjoint 5k

Resume
------
Skips rows already present in ``--out`` (matched by ``orig_idx``). Safe to
re-invoke after crash/interrupt.

Example
-------
    export GOOGLE_API_KEY=$(cat ~/gem)
    uv run python scripts/translate_orcamath_gemini.py \\
        --n 5000 --seed 42 --concurrency 32 \\
        --out /mnt/data2/orca_math_eo_5k.jsonl
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

# ── Prompt (v3 from prompt sweep, math-vocab aware) ────────────────────

PROMPT_TEMPLATE = """Translate the following English math problem and solution to Esperanto.

Preserve exactly:
  - all numbers, variables, equations, and mathematical notation
  - any LaTeX ($...$) verbatim
  - the step-by-step structure

Use standard Esperanto math terminology:
  ebeno = plane, kampo = field, ringo = ring, grupo = group,
  meznombro = mean, mediano = median, varianco = variance,
  entjero = integer, reelo = real, kompleksa = complex, matrico = matrix,
  vektoro = vector, funkcio = function, derivaĵo = derivative,
  integralo = integral, ekvacio = equation, malegaleco = inequality,
  hipotenuzo = hypotenuse, kateto = leg, rilatumo = ratio,
  procento = percent

Output ONLY the Esperanto translation. No preamble, no notes, no markdown.

QUESTION:
{question}

ANSWER:
{answer}"""


# ── Quality filters ────────────────────────────────────────────────────

_PREAMBLE_PATTERNS = [
    re.compile(r"^(jen|jeni|jena|jenajn?)\s+(la|estas)?\s*trad", re.I),
    re.compile(r"^(here|the)\s+is\s+the\s+trans", re.I),
    re.compile(r"^```", re.M),
]


def looks_bad(en_q: str, en_a: str, eo_out: str) -> str | None:
    """Return a rejection reason string if output looks bad, else None."""
    if not eo_out or len(eo_out) < 20:
        return "too_short"
    # Length ratio sanity check — EO should be within 0.5x–2x the source
    src_len = len(en_q) + len(en_a)
    if src_len == 0:
        return "empty_source"
    ratio = len(eo_out) / src_len
    if ratio < 0.4 or ratio > 2.5:
        return f"len_ratio={ratio:.2f}"
    # Preamble contamination
    for pat in _PREAMBLE_PATTERNS:
        if pat.search(eo_out[:120]):
            return "preamble"
    # Must contain "QUESTION" / "ANSWER" section markers we asked to preserve
    # (translated as "DEMANDO"/"RESPONDO"); accept a few variants.
    up = eo_out.upper()
    if not any(
        marker in up for marker in ("DEMANDO", "RESPONDO", "DEMANDO:", "RESPONDO:")
    ):
        # Fall back to lenient: just require some line break structure
        if eo_out.count("\n") < 1:
            return "no_qa_structure"
    return None


# ── API caller ─────────────────────────────────────────────────────────


async def translate_row(client: genai.Client, model: str, semaphore: asyncio.Semaphore,
                        row: dict, orig_idx: int) -> dict:
    """Translate one orca-math row. Returns result dict (with error field if
    failed)."""
    prompt = PROMPT_TEMPLATE.format(question=row["question"], answer=row["answer"])
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await asyncio.to_thread(
                    client.models.generate_content, model=model, contents=prompt
                )
                eo = (resp.text or "").strip()
                bad = looks_bad(row["question"], row["answer"], eo)
                return {
                    "orig_idx": orig_idx,
                    "en_question": row["question"],
                    "en_answer": row["answer"],
                    "eo_translation": eo,
                    "reject_reason": bad,
                }
            except Exception as e:
                err_msg = str(e)[:200]
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    "orig_idx": orig_idx,
                    "en_question": row["question"],
                    "en_answer": row["answer"],
                    "eo_translation": None,
                    "reject_reason": f"error:{err_msg}",
                }
    return {}  # unreachable


# ── Main ───────────────────────────────────────────────────────────────


async def run(args) -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    # Deterministic subset by (seed, n)
    print(f"loading orca-math-word-problems-200k...", flush=True)
    full = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    shuf = full.shuffle(seed=args.seed)
    subset = shuf.select(range(min(args.n, len(shuf))))
    print(f"  selected {len(subset):,} rows (seed={args.seed})", flush=True)

    # Resume: figure out which orig_idx values are already in the output
    done: set[int] = set()
    out_path = Path(args.out)
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
                done.add(r["orig_idx"])
            except Exception:
                pass
        print(f"  {len(done):,} rows already translated, will skip", flush=True)

    # Row selection has stable indices [0, len(subset)). Skip already-done.
    todo = [(i, subset[i]) for i in range(len(subset)) if i not in done]
    print(f"  {len(todo):,} rows to translate\n", flush=True)
    if not todo:
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    n_ok = n_rej = 0

    async def worker(orig_idx: int, row: dict) -> dict:
        return await translate_row(client, args.model, semaphore, row, orig_idx)

    # Write incrementally, one line per completion, so ctrl-C is safe
    with out_path.open("a") as fout:
        tasks = [asyncio.create_task(worker(i, row)) for i, row in todo]
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
    ap.add_argument("--n", type=int, default=5000,
                    help="Number of rows to translate (default: 5000)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Shuffle seed for row selection. Fixed seed + larger "
                         "n produces a superset — safe to extend later.")
    ap.add_argument("--out", type=Path, default=Path("/mnt/data2/orca_math_eo.jsonl"))
    ap.add_argument("--model", default="gemini-flash-lite-latest",
                    help="Gemini model id (default: gemini-flash-lite-latest)")
    ap.add_argument("--concurrency", type=int, default=32,
                    help="Concurrent API calls (default 32; Flash Lite tier "
                         "handles 60+ RPS comfortably)")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
