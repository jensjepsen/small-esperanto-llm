"""Danish Wikipedia → grounded SFT rows via Gemini Flash Lite 3.1.

For each Danish Wikipedia article that passes the quality + Danish-relevance
filters, pick a random Dolly category and prompt Gemini to write a Danish row
grounded ONLY in the article intro. Resumable JSONL.

Filter: skips list/disambig/stub articles + articles about clearly foreign
subjects (unless Danish-crossover mention present).
"""
import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset
from google import genai

MODEL = "gemini-3.1-flash-lite"

CATEGORIES = ["open_qa", "closed_qa", "general_qa", "brainstorming",
              "creative_writing", "information_extraction", "summarization",
              "classification"]

CATEGORY_RULES = """Category rules (STRICT — do not drift):
  * open_qa → answer from general knowledge (informed by source), no context field in output
  * closed_qa → context (from source) + question whose answer is IN the context
  * general_qa → open-ended question about the topic, response is a paragraph
  * brainstorming → generate a LIST of ideas or items derivable from the source
  * creative_writing → produce a short creative piece (story, poem, letter) inspired by the topic
  * summarization → context = a longer text from source, response = short summary
  * information_extraction → context (from source) + task to extract specific facts from it
  * classification → present a LIST of items (some from source) and ask which fit a category. Response labels/sorts."""

PROMPT = """You are building a Danish instruction-tuning dataset. Generate ONE
Danish row grounded strictly in the source.

CATEGORY: {category}

SOURCE (Danish Wikipedia article "{title}"):
{source}

{rules}

STRICT constraints:
  - Every named entity, date, place, and number in your response MUST appear in
    the SOURCE. Do NOT introduce facts not in the source.
  - Write natural Danish. Do not mention "Wikipedia" or "kilden".
  - If the source does not contain enough material for a good {category} row
    (one-line stub, boring list, etc.), return {{"skip": true, "reason": "..."}}.

Output ONE JSON object:
  {{"category": "{category}",
    "instruction": "<Danish instruction>",
    "context": "<Danish context OR empty string if the category doesn't use one>",
    "response": "<Danish response, grounded in source>"}}

OUTPUT JSON:"""


DISQUALIFYING_TITLE_PREFIXES = ("Liste over", "Liste af", "Kategori:", "Skabelon:",
                                "Portal:", "Wikipedia:")
DISQUALIFYING_TEXT_MARKERS = ("kan henvise til:", "kan referere til:",
                              "flertydig", "kan være:")

FOREIGN_MARKERS = (
    "amerikansk", "britisk", "engelsk", "irsk", "skotsk", "walisisk",
    "tysk", "østrigsk", "schweizisk",
    "fransk", "belgisk", "hollandsk", "nederlandsk", "luxembourgsk",
    "italiensk", "spansk", "portugisisk", "græsk",
    "russisk", "ukrainsk", "polsk", "tjekkisk", "slovakisk",
    "ungarsk", "rumænsk", "bulgarsk", "serbisk", "kroatisk", "slovensk",
    "tyrkisk", "kinesisk", "japansk", "koreansk", "indisk", "pakistansk",
    "australsk", "newzealandsk", "canadisk", "mexicansk",
    "brasiliansk", "argentinsk", "chilensk", "colombiansk",
    "sydafrikansk", "nigeriansk", "kenyansk", "egyptisk",
    "iransk", "irakisk", "syrisk", "israelsk", "libanesisk", "jordansk",
    "svensk", "norsk", "islandsk", "finsk",
)

DANISH_MARKERS = ("dansk", "danmark", "københavn", "aarhus", "odense",
                  "aalborg", "kongeriget danmark", "rigsfællesskabet",
                  "grønland", "grønlandsk", "færøerne", "færøsk", "danskere")


def _has_foreign_subject_marker(intro_lc: str) -> bool:
    head = intro_lc[:220]
    for m in FOREIGN_MARKERS:
        if re.search(rf"\b(er|var) (en|et) {m}\b", head):
            return True
        if re.search(rf"\({m}\b", head):
            return True
    return False


def _mentions_denmark(intro_lc: str) -> bool:
    return any(m in intro_lc for m in DANISH_MARKERS)


def looks_useful(row) -> bool:
    title = row["title"]
    text  = row["text"]
    if any(title.startswith(p) for p in DISQUALIFYING_TITLE_PREFIXES):
        return False
    intro = text[:800]
    intro_lc = intro.lower()
    if any(m in intro_lc for m in DISQUALIFYING_TEXT_MARKERS):
        return False
    if len(intro.strip()) < 300:
        return False
    if intro.count("\n") > 15:
        return False
    if _has_foreign_subject_marker(intro_lc) and not _mentions_denmark(intro_lc):
        return False
    return True


def clean_intro(text: str, max_chars: int = 900) -> str:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out, total = [], 0
    for p in paras:
        if total + len(p) > max_chars:
            break
        out.append(p)
        total += len(p) + 1
    return "\n\n".join(out)


def load_done_ids(out_path: Path) -> set[int]:
    done = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = r.get("seed_idx")
            if idx is not None and not r.get("error"):
                done.add(idx)
    return done


async def generate_one(client, sem, seed_idx, row, category):
    source = clean_intro(row["text"])
    prompt = PROMPT.format(category=category, title=row["title"],
                            source=source, rules=CATEGORY_RULES)

    async with sem:
        for attempt in range(3):
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
                in_tok  = getattr(usage, "prompt_token_count", 0) if usage else 0
                out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as e:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt); continue
                    return {"seed_idx": seed_idx, "title": row["title"],
                            "category": category, "error": f"parse: {str(e)[:100]}",
                            "raw": raw[:400],
                            "meta": {"latency_s": dt, "in_tok": in_tok, "out_tok": out_tok}}
                parsed["seed_idx"] = seed_idx
                parsed["title"] = row["title"]
                parsed["meta"] = {"latency_s": dt, "in_tok": in_tok, "out_tok": out_tok,
                                  "attempt": attempt + 1}
                return parsed
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt); continue
                return {"seed_idx": seed_idx, "title": row["title"],
                        "category": category, "error": f"api: {str(e)[:200]}",
                        "meta": {"attempt": attempt + 1}}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15000, help="target rows to generate")
    ap.add_argument("--pool-scan", type=int, default=25000,
                    help="how many wiki articles to scan for the filtered pool")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/data2/wiki_grounded_da/full.jsonl"))
    args = ap.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY", file=sys.stderr); sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading Danish Wikipedia (streaming, will scan first {args.pool_scan})…",
          flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.da", split="train",
                       streaming=True)

    pool = []
    for i, row in enumerate(ds):
        if i >= args.pool_scan:
            break
        if looks_useful(row):
            pool.append({"seed_idx": i, "title": row["title"], "text": row["text"]})
        if (i + 1) % 5000 == 0:
            print(f"  scanned {i+1}, kept {len(pool)}", flush=True)
    print(f"  final pool: {len(pool)} / {args.pool_scan} "
          f"({100*len(pool)/args.pool_scan:.1f}%)", flush=True)

    rng = random.Random(args.seed)
    # Sample WITH replacement if pool < args.n; otherwise without.
    if len(pool) >= args.n:
        selected = rng.sample(pool, args.n)
    else:
        selected = [rng.choice(pool) for _ in range(args.n)]
        print(f"  ⚠ pool ({len(pool)}) < target ({args.n}); sampling with replacement",
              flush=True)
    categories = [rng.choice(CATEGORIES) for _ in selected]

    done = load_done_ids(args.out)
    print(f"  {len(done)} rows already complete → skipping", flush=True)
    tasks_in = [(i, row, cat)
                for i, (row, cat) in enumerate(zip(selected, categories))
                if i not in done]
    print(f"  {len(tasks_in)} rows to generate", flush=True)
    if not tasks_in:
        return

    client = genai.Client(api_key=api_key)
    sem = asyncio.Semaphore(args.workers)
    coros = [generate_one(client, sem, i, row, cat)
             for i, row, cat in tasks_in]

    t0 = time.time()
    n_done = n_ok = n_skip = n_fail = 0
    total_in = total_out = 0
    with args.out.open("a") as f:
        for coro in asyncio.as_completed(coros):
            r = await coro
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            if r.get("error"):     n_fail += 1
            elif r.get("skip"):    n_skip += 1
            else:                  n_ok += 1
            meta = r.get("meta", {})
            total_in  += meta.get("in_tok", 0) or 0
            total_out += meta.get("out_tok", 0) or 0
            if n_done % 200 == 0 or n_done == len(tasks_in):
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta  = (len(tasks_in) - n_done) / rate if rate > 0 else 0
                cost = total_in * 0.25 / 1e6 + total_out * 1.50 / 1e6
                print(f"  {n_done}/{len(tasks_in)}  ok={n_ok} skip={n_skip} fail={n_fail}  "
                      f"{rate:.1f} rows/s  eta={eta:.0f}s  cost=${cost:.2f}",
                      flush=True)

    elapsed = time.time() - t0
    cost = total_in * 0.25 / 1e6 + total_out * 1.50 / 1e6
    print(f"\ndone in {elapsed:.0f}s; ok={n_ok} skip={n_skip} fail={n_fail}")
    print(f"tokens: {total_in:,} in + {total_out:,} out; cost = ${cost:.2f}")
    print(f"cost per 1000 ok rows: ${cost / max(n_ok, 1) * 1000:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
