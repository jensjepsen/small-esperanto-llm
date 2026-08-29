"""Dolly → Danish-perspective reframing via Gemini Flash Lite 3.1.

Take an English Dolly-15k row and produce a NEW Danish row in the same
category. A specific Danish topic is injected per row (rotated through a
curated 600-topic pool) to prevent Gemini from defaulting to a small
canonical safe-set. Resumable JSONL.
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

from datasets import load_dataset
from google import genai

sys.path.insert(0, str(Path(__file__).resolve().parent))
from da_topics import TOPICS_FLAT

MODEL = "gemini-3.1-flash-lite"

REFRAME_PROMPT = """You are helping build a Danish-language instruction-tuning
dataset. Write ONE Danish row in the given CATEGORY, about the given TOPIC.

CATEGORY: {category}
TOPIC (subject area = {subject}): {topic}

Task-shape rules per category (STRICT — do not drift):
  * open_qa → answer from general knowledge, no context field
  * closed_qa → context + question whose answer is IN the context
  * general_qa → open-ended question, general knowledge, no context
  * brainstorming → generate a LIST of ideas/items
  * creative_writing → produce a creative piece (story, poem, letter)
  * summarization → context = a longer text, response = a short summary
  * information_extraction → context + task to extract specific facts from it
  * classification → present a LIST of items and ask which fit a category,
    or which category each item belongs to. Response labels/sorts the items.
    Danish example: "Hvilke af disse er hovedstæder: København, Aarhus, Oslo,
    Odense, Stockholm?" → "København, Oslo og Stockholm er hovedstæder.
    Aarhus og Odense er ikke."

Constraints:
  - The row MUST be about the TOPIC above. Anchor the instruction on it.
  - Do NOT mention the topic name as a meta-reference — write the row
    naturally as if the topic were the subject of a genuine user query.
  - Include only facts you are confident are correct. If you cannot construct
    a factually safe row about this topic in this category, return
    {{"skip": true, "reason": "brief reason"}}
  - No English text; all output in Danish.

For an English seed row (task-shape reference only, IGNORE its content):
{seed_payload}

Output ONE JSON object:
  {{"category": "{category}",
    "instruction": "<Danish instruction, grounded in the topic>",
    "context": "<Danish context OR empty string if the category doesn't use one>",
    "response": "<Danish response, grounded in context if provided>"}}

OUTPUT JSON:"""


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


async def reframe_one(client, sem, seed_idx, seed_row, subject, topic):
    seed_payload = json.dumps({
        "category": seed_row["category"],
        "instruction": seed_row["instruction"],
        "context": seed_row["context"],
        "response": seed_row["response"],
    }, ensure_ascii=False, indent=2)
    prompt = REFRAME_PROMPT.format(
        category=seed_row["category"],
        subject=subject,
        topic=topic,
        seed_payload=seed_payload,
    )

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
                in_tok  = getattr(usage, "prompt_token_count", 0) if usage else 0
                out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as e:
                    if attempt < 3:
                        await asyncio.sleep(2 ** attempt); continue
                    return {"seed_idx": seed_idx, "seed_category": seed_row["category"],
                            "topic": topic, "subject": subject,
                            "error": f"parse: {str(e)[:100]}",
                            "raw": raw[:400],
                            "meta": {"latency_s": dt, "in_tok": in_tok, "out_tok": out_tok}}
                parsed["seed_idx"] = seed_idx
                parsed["seed_category"] = seed_row["category"]
                parsed["topic"] = topic
                parsed["subject"] = subject
                parsed["meta"] = {"latency_s": dt, "in_tok": in_tok, "out_tok": out_tok,
                                  "attempt": attempt + 1}
                return parsed
            except Exception as e:
                err = str(e)[:200]
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt); continue
                return {"seed_idx": seed_idx, "seed_category": seed_row["category"],
                        "topic": topic, "subject": subject,
                        "error": f"api: {err}", "meta": {"attempt": attempt + 1}}


def build_topic_schedule(n_rows: int, seed: int) -> list[tuple[str, str]]:
    """Assign one (subject, topic) per row. Rotate the pool so no topic
    is over-used: each topic is used ceil(n_rows / n_topics) times."""
    rng = random.Random(seed)
    n_topics = len(TOPICS_FLAT)
    reps = (n_rows + n_topics - 1) // n_topics
    pool: list[tuple[str, str]] = []
    for _ in range(reps):
        shuffled = list(TOPICS_FLAT)
        rng.shuffle(shuffled)
        pool.extend(shuffled)
    return pool[:n_rows]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/data2/dolly_da_reframe_v2/full.jsonl"))
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of rows (for smokes)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY", file=sys.stderr); sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("loading databricks/databricks-dolly-15k…", flush=True)
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    total = len(ds)
    if args.limit:
        total = min(args.limit, total)
    print(f"  {total} Dolly rows will be reframed", flush=True)
    print(f"  topic pool: {len(TOPICS_FLAT)} topics across "
          f"{len(set(s for s, _ in TOPICS_FLAT))} subject areas", flush=True)

    schedule = build_topic_schedule(total, args.seed)

    done = load_done_ids(args.out)
    print(f"  {len(done)} rows already complete → skipping", flush=True)
    todo = [(i, ds[i], schedule[i]) for i in range(total) if i not in done]
    print(f"  {len(todo)} rows to reframe", flush=True)
    if not todo:
        return

    client = genai.Client(api_key=api_key)
    sem = asyncio.Semaphore(args.workers)
    tasks = [reframe_one(client, sem, i, r, subj, topic) for i, r, (subj, topic) in todo]

    t0 = time.time()
    n_done = n_ok = n_skip = n_fail = 0
    total_in = total_out = 0
    with args.out.open("a") as f:
        for coro in asyncio.as_completed(tasks):
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
            if n_done % 100 == 0 or n_done == len(todo):
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta  = (len(todo) - n_done) / rate if rate > 0 else 0
                cost = total_in * 0.25 / 1e6 + total_out * 1.50 / 1e6
                print(f"  {n_done}/{len(todo)}  ok={n_ok} skip={n_skip} fail={n_fail}  "
                      f"{rate:.1f} rows/s  eta={eta:.0f}s  cost=${cost:.2f}",
                      flush=True)

    elapsed = time.time() - t0
    cost = total_in * 0.25 / 1e6 + total_out * 1.50 / 1e6
    print(f"\ndone in {elapsed:.0f}s; ok={n_ok} skip={n_skip} fail={n_fail}")
    print(f"tokens: {total_in:,} in + {total_out:,} out; cost = ${cost:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
