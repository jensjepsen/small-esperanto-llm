"""Gemini rewrap pass: takes procedural word problems and asks Flash Lite to
generate N alternate phrasings of each question while keeping numbers identical.

The chain and answer are preserved as-is. Verification: each rewrap must
contain every multi-digit number from the original (skip percent-numbers,
same logic as the chain verifier).

Input: JSONL from word_problems_procedural.py (or generate_word_problems.py).
Output: JSONL with one row per accepted rewrap, plus an "original" row per
problem if --keep-original is set.

Usage:
  GOOGLE_API_KEY=... uv run --extra gemini python scripts/word_problems_rewrap.py \\
    --input data/word_problems/ratio_proc.jsonl \\
    --output data/word_problems/ratio_rewrap.jsonl \\
    --variants 2 --batch-size 10
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import the verifier from the existing module so we use the same battle-tested
# regex (lookbehind for non-alphabetic boundary, etc).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_word_problems import verify_chain, verify_question_numbers_in_chain


def question_numbers(q: str) -> set[str]:
    """Multi-digit non-percent numbers that must survive a rewrap."""
    q_stripped = re.sub(r"\d+\s*%", " ", q)
    return set(re.findall(r"\b\d{2,}\b", q_stripped))


def parse_response(text: str) -> list[list[str]]:
    """Expect a JSON list-of-lists: outer = one entry per problem, inner = N variants."""
    text = text.strip()
    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.strip()
            if chunk.startswith("json"):
                text = chunk[4:].strip()
                break
            if chunk.startswith("["):
                text = chunk
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("["), text.rfind("]")
        if i >= 0 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except json.JSONDecodeError:
                pass
    return []


def build_prompt(batch: list[dict], variants: int) -> str:
    problems_block = "\n\n".join(
        f"PROBLEMO {i+1}:\nDEMANDO: {p['question_eo']}\nSOLVO:\n{p['chain_eo']}"
        for i, p in enumerate(batch)
    )
    return f"""Reverkis la sekvajn esperantajn matematik-problemojn KAJ iliajn solvojn.
Por ĈIU problemo, generu EKZAKTE {variants} alternativajn versiojn de la
(DEMANDO + SOLVO) paro.

KRITIKAJ REGULOJ:
- ĈIUJ nombroj devas resti EKZAKTE SAMAJ kaj en la demando kaj en la solvo.
- La MATEMATIKA STRUKTURO devas resti la sama. La fina respondo NE povas ŝanĝi.
- En la SOLVO, ĉiu aritmetika paŝo devas resti sur sia propra linio kun signo `=`,
  kaj la lasta linio devas esti "#### N" kun la sama N kiel originale.
- Vi povas ŝanĝi: vorto-elekto, frazstrukturo, ordo de ne-aritmetikaj paŝoj, nomoj de variabloj (x→a, d→j), klarigaj komentoj inter paŝoj.
- Varia stilo: alternaj kuntekstoj (lernejo, vendejo, hejmo, sporto), pasinta/estanta/futura tempo, malsamaj frazstrukturoj.
- Gramatike ĝusta Esperanto.

{problems_block}

Respondu kun JSON: listo de {len(batch)} listoj, ĉiu enhavanta {variants} objektoj kun ŝlosiloj "question" kaj "chain".
Ekzemplo por 1 problemo, 2 variantoj:
  [[{{"question":"...","chain":"...\\n#### 42"}}, {{"question":"...","chain":"...\\n#### 42"}}]]

Respondu NUR la JSON, sen ```markdown, sen klariga teksto.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--variants", type=int, default=2,
                    help="rewraps per original problem")
    ap.add_argument("--batch-size", type=int, default=10,
                    help="originals per Gemini call")
    ap.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    ap.add_argument("--max-calls", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap inputs processed (0 = all)")
    ap.add_argument("--keep-original", action="store_true",
                    help="also emit the original problem")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel API workers (rate-limit aware: keep ≤ 20)")
    ap.add_argument("--report-every", type=int, default=30,
                    help="seconds between progress reports")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(2)
    from google import genai
    client = genai.Client(api_key=api_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    originals = []
    with args.input.open() as f:
        for line in f:
            try:
                originals.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if args.limit:
        originals = originals[: args.limit]
    print(f"loaded {len(originals)} originals from {args.input}", flush=True)

    stats = Counter()
    stats_lock = threading.Lock()
    out_lock = threading.Lock()
    counter = {"written": 0, "calls": 0}
    out_f = args.output.open("w")

    def process_batch(batch_idx: int, batch: list[dict]):
        with stats_lock:
            counter["calls"] += 1
            this_call = counter["calls"]
        prompt = build_prompt(batch, args.variants)
        try:
            resp = client.models.generate_content(model=args.model, contents=prompt)
            text = resp.text or ""
        except Exception as e:
            with stats_lock:
                stats["api-error"] += 1
            return

        parsed = parse_response(text)
        if not parsed or not isinstance(parsed, list):
            with stats_lock:
                stats["parse-fail"] += 1
            return

        if len(parsed) != len(batch):
            with stats_lock:
                stats["batch-length-mismatch"] += 1

        rows_out = []
        local_stats = Counter()
        for orig, variants_list in zip(batch, parsed):
            if not isinstance(variants_list, list):
                local_stats["variants-not-list"] += 1
                continue
            orig_q_nums = question_numbers(orig["question_eo"])
            if args.keep_original:
                r = dict(orig); r["rewrap_idx"] = 0; r["is_original"] = True
                rows_out.append(r)
                local_stats["original-kept"] += 1
            for vi, variant in enumerate(variants_list):
                local_stats["variants-seen"] += 1
                if not isinstance(variant, dict):
                    local_stats["variant-not-dict"] += 1
                    continue
                new_q = (variant.get("question") or "").strip()
                new_c = (variant.get("chain") or "").strip()
                if not new_q or not new_c:
                    local_stats["variant-missing-field"] += 1
                    continue
                if orig_q_nums - question_numbers(new_q):
                    local_stats["variant-q-num-missing"] += 1
                    continue
                ok, why = verify_chain(new_c, orig["answer"])
                if not ok:
                    local_stats[f"variant-chain:{why.split(':')[0]}"] += 1
                    continue
                ok2, why2 = verify_question_numbers_in_chain(new_q, new_c)
                if not ok2:
                    local_stats[f"variant-qnum:{why2.split(':')[0]}"] += 1
                    continue
                r = dict(orig)
                r["question_eo"] = new_q
                r["chain_eo"] = new_c
                r["rewrap_idx"] = vi + 1
                r["is_original"] = False
                rows_out.append(r)
                local_stats["variants-accepted"] += 1

        with out_lock:
            for r in rows_out:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()
        with stats_lock:
            counter["written"] += len(rows_out)
            stats.update(local_stats)

    batches = [
        originals[i : i + args.batch_size]
        for i in range(0, len(originals), args.batch_size)
    ]
    if args.max_calls:
        batches = batches[: args.max_calls]
    total_batches = len(batches)

    t0 = time.time()
    stop_event = threading.Event()

    def reporter():
        last_written = 0
        last_time = time.time()
        while not stop_event.is_set():
            time.sleep(args.report_every)
            now = time.time()
            with stats_lock:
                w = counter["written"]
                c = counter["calls"]
                snapshot = dict(stats)
            dt_recent = now - last_time
            recent_rate = (w - last_written) / max(0.1, dt_recent) * 60
            cum_rate = w / max(0.1, now - t0) * 60
            done_frac = c / max(1, total_batches)
            eta_min = (now - t0) / max(0.001, done_frac) * (1 - done_frac) / 60
            print(
                f"  [{c}/{total_batches} calls, {done_frac*100:.0f}%]  "
                f"written={w}  recent={recent_rate:.0f}/min  "
                f"avg={cum_rate:.0f}/min  ETA={eta_min:.0f}min  "
                f"stats={snapshot}",
                flush=True,
            )
            last_written, last_time = w, now

    rep_thread = threading.Thread(target=reporter, daemon=True)
    rep_thread.start()

    print(f"launching {args.workers} workers over {total_batches} batches "
          f"(batch_size={args.batch_size}, variants={args.variants}, "
          f"keep_original={args.keep_original})", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_batch, i, b) for i, b in enumerate(batches)]
        for _ in as_completed(futures):
            pass

    stop_event.set()
    out_f.close()
    print(f"\ndone: {counter['written']} rows → {args.output}")
    print(f"  {counter['calls']} API calls, {time.time()-t0:.1f}s wall")
    print(f"  stats: {dict(stats)}")


if __name__ == "__main__":
    main()
