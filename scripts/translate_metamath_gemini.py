"""Translate MetaMathQA to Esperanto via Gemini Flash Lite.

Same skeleton as translate_orcamath_gemini.py; MetaMath's row schema uses
``query``/``response`` (not question/answer) and includes a per-row ``type``
we preserve. By default only ``GSM_*`` rows are translated (the GSM8K-derived
word-problem subtypes); pass ``--include-math`` to also cover ``MATH_*``.

Output schema (matches translate_metamath_pod.py so shards concat cleanly),
plus per-row token usage for cost reporting:
  {orig_idx, type, q_en, q_eo, a_en, a_eo, original_question,
   input_tokens, output_tokens}

Example:
    export GEMINI_API_KEY=$(cat ~/gem)
    uv run --extra gemini python scripts/translate_metamath_gemini.py \\
        --out /mnt/data2/metamath_gsm_eo.jsonl --concurrency 64
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


PROMPT_TEMPLATE = """Translate the following English math problem and its solution to Esperanto.

Preserve exactly:
  - all numbers, variables, equations, and mathematical notation
  - any LaTeX ($...$, \\[...\\], \\boxed{{}}, \\frac, \\sqrt, \\cdot etc.) verbatim
  - the step-by-step structure and line breaks in the solution
  - the "#### N" answer marker if present

Use standard Esperanto math terminology:
  ebeno = plane, entjero = integer, reelo = real, meznombro = mean,
  ekvacio = equation, malegaleco = inequality, funkcio = function,
  derivaĵo = derivative, integralo = integral, procento = percent,
  rilatumo = ratio, hipotenuzo = hypotenuse, kateto = leg

Output ONLY two blocks with exactly these section headers:

DEMANDO:
<Esperanto translation of the question>

RESPONDO:
<Esperanto translation of the answer>

No preamble, no notes, no markdown fences.

QUESTION:
{question}

ANSWER:
{answer}"""


DEMANDO_RE = re.compile(r"DEMANDO\s*:\s*(.*?)\s*RESPONDO\s*:\s*(.*)", re.S | re.I)


def parse_output(text: str) -> tuple[str, str] | None:
    m = DEMANDO_RE.search(text)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def _usage(resp) -> tuple[int, int, int]:
    """Extract (prompt_tokens, output_tokens, thoughts_tokens) from a Gemini
    response. Thoughts (Gemini 2.5+ thinking tokens) are billed at the output
    rate — off for Flash Lite by default but included so a future model
    change doesn't silently under-report cost."""
    u = getattr(resp, "usage_metadata", None)
    if u is None:
        return 0, 0, 0
    return (
        int(getattr(u, "prompt_token_count", 0) or 0),
        int(getattr(u, "candidates_token_count", 0) or 0),
        int(getattr(u, "thoughts_token_count", 0) or 0),
    )


async def translate_row(client, model, semaphore, row) -> dict:
    prompt = PROMPT_TEMPLATE.format(question=row["query"], answer=row["response"])
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await asyncio.to_thread(
                    client.models.generate_content, model=model, contents=prompt
                )
                out = (resp.text or "").strip()
                in_tok, out_tok, thought_tok = _usage(resp)
                base = {
                    "orig_idx": row["orig_idx"],
                    "type": row.get("type", ""),
                    "original_question": row.get("original_question", ""),
                    "q_en": row["query"],
                    "a_en": row["response"],
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "thoughts_tokens": thought_tok,
                }
                parsed = parse_output(out)
                if parsed is None:
                    return {**base, "q_eo": None, "a_eo": None,
                            "raw": out[:400], "reject_reason": "no_section_markers"}
                q_eo, a_eo = parsed
                return {**base, "q_eo": q_eo, "a_eo": a_eo}
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    "orig_idx": row["orig_idx"],
                    "type": row.get("type", ""),
                    "original_question": row.get("original_question", ""),
                    "q_en": row["query"],
                    "a_en": row["response"],
                    "q_eo": None, "a_eo": None,
                    "input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0,
                    "reject_reason": f"error:{str(e)[:200]}",
                }
    return {}


async def run(args) -> None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    print("loading MetaMathQA...", flush=True)
    full = load_dataset("meta-math/MetaMathQA", split="train")

    # Filter by type — GSM_* by default (word-problem subtypes; MATH_* subtypes
    # opt-in via --include-math)
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
          f"({'include_math' if args.include_math else 'GSM_ only'})", flush=True)

    # Resume: skip orig_idx values already in the output file
    done: set[int] = set()
    prior_in = prior_out = prior_thought = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
                done.add(r["orig_idx"])
                prior_in += r.get("input_tokens", 0)
                prior_out += r.get("output_tokens", 0)
                prior_thought += r.get("thoughts_tokens", 0)
            except Exception:
                pass
        print(f"  {len(done):,} rows already translated — will skip", flush=True)

    todo = [r for r in rows if r["orig_idx"] not in done]
    print(f"  {len(todo):,} rows to translate\n", flush=True)
    if not todo:
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    n_ok = n_rej = 0
    tok_in = tok_out = tok_thought = 0

    async def worker(row):
        return await translate_row(client, args.model, semaphore, row)

    # Cost: thinking tokens (when present) bill at the output rate.
    def cost_of(t_in: int, t_out: int, t_thought: int) -> float:
        return (t_in * args.price_input
                + (t_out + t_thought) * args.price_output) / 1_000_000

    with out_path.open("a") as fout:
        tasks = [asyncio.create_task(worker(r)) for r in todo]
        for coro in asyncio.as_completed(tasks):
            r = await coro
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            if r.get("reject_reason"):
                n_rej += 1
            else:
                n_ok += 1
            tok_in += r.get("input_tokens", 0)
            tok_out += r.get("output_tokens", 0)
            tok_thought += r.get("thoughts_tokens", 0)
            total = n_ok + n_rej
            if total % args.log_every == 0 or total == len(todo):
                el = time.time() - t0
                rate = total / el
                eta = (len(todo) - total) / max(rate, 1e-6)
                run_cost = cost_of(tok_in, tok_out, tok_thought)
                grand_cost = cost_of(tok_in + prior_in,
                                     tok_out + prior_out,
                                     tok_thought + prior_thought)
                thought_str = (f"  thought={tok_thought / 1e6:.2f}M"
                               if tok_thought else "")
                print(
                    f"  {total:,}/{len(todo):,}  ok={n_ok:,}  "
                    f"rej={n_rej:,} ({100 * n_rej / total:.1f}%)  "
                    f"rate={rate:.1f}/s  eta={eta / 60:.1f}min  "
                    f"tok(in/out)={tok_in / 1e6:.2f}M/{tok_out / 1e6:.2f}M"
                    f"{thought_str}  "
                    f"$this={run_cost:.2f}  $total={grand_cost:.2f}",
                    flush=True,
                )

    el = time.time() - t0
    grand_in = tok_in + prior_in
    grand_out = tok_out + prior_out
    grand_thought = tok_thought + prior_thought
    print(
        f"\ndone: {n_ok:,} translated, {n_rej:,} rejected in {el / 60:.1f} min"
        f" → {out_path}\n"
        f"session tokens: in={tok_in:,}  out={tok_out:,}"
        f"  thought={tok_thought:,}"
        f"  session cost: ${cost_of(tok_in, tok_out, tok_thought):.2f}\n"
        f"cumulative tokens (incl. prior): in={grand_in:,}  out={grand_out:,}"
        f"  thought={grand_thought:,}"
        f"  cumulative cost: ${cost_of(grand_in, grand_out, grand_thought):.2f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/mnt/data2/metamath_gsm_eo.jsonl"))
    ap.add_argument("--model", default="gemini-flash-lite-latest")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop scanning source after this many kept rows (0 = no limit)")
    ap.add_argument("--include-math", action="store_true",
                    help="Also translate MATH_* subtypes (default: GSM_* only)")
    ap.add_argument("--price-input", type=float, default=0.10,
                    help="USD per 1M input tokens (Flash Lite ≈ 0.10)")
    ap.add_argument("--price-output", type=float, default=0.40,
                    help="USD per 1M output tokens (Flash Lite ≈ 0.40)")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
