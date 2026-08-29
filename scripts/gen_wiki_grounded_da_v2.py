"""100-row smoke via gemma-3-12b on OpenRouter with the FIXED prompt.

Fix: for categories with empty context, instruct the model NOT to write
source-referring phrases ("baseret på teksten" etc.) in the instruction.
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

import aiohttp
from datasets import load_dataset


CATEGORIES = ["open_qa", "closed_qa", "general_qa", "brainstorming",
              "creative_writing", "information_extraction", "summarization",
              "classification"]

CATEGORY_RULES = """Category rules (STRICT — do not drift):

Categories WITH context (context field is populated from source):
  * closed_qa → context + question whose answer is IN the context
  * summarization → context = a longer text from source, response = short summary
  * information_extraction → context + task to extract specific facts from it

Categories WITHOUT context (context field MUST be empty string):
  * open_qa → question the user could ask about the topic; response answers from general knowledge
  * general_qa → open-ended question about the topic, response is a paragraph
  * brainstorming → generate a LIST of ideas or items ABOUT the topic (not
    extracted FROM a source text). Draw from general knowledge.
  * creative_writing → produce a short creative piece (story, poem, letter) inspired by the topic
  * classification → the instruction MUST contain a comma-separated list of
    at least 4 concrete candidates inline (e.g. "Hvilke af følgende er
    hovedstæder: København, Aarhus, Oslo, Odense, Stockholm?"). The response
    MUST use ONLY items from that inline candidate list — never add items
    outside the list, even if you know additional facts from the source.
    The response labels/sorts those exact candidates. If you cannot
    construct such a candidate list from the source, return
    {"skip": true, "reason": "..."}.

STAND-ALONE INSTRUCTION rule (categories WITHOUT context):
  For open_qa, general_qa, brainstorming, creative_writing, classification —
  the instruction MUST make sense on its own without any accompanying text.
  You are FORBIDDEN from writing ANY phrase that points at a source, article,
  text, or passage the user cannot see. This includes but is not limited to:
    "baseret på teksten", "ifølge teksten", "der optræder i teksten",
    "ud fra teksten", "i teksten", "baseret på kilden", "ifølge kilden",
    "baseret på oplysningerne", "ifølge oplysningerne", "ifølge nedenstående",
    "ifølge artiklen", "nævnes i artiklen", "der beskrives", "i teksten om",
    "ovennævnte", "det følgende", "det ovennævnte", "nedenstående",
    "baseret på informationen", "fra teksten", "fra kilden", "fra artiklen"

  ANY grammatical form of "nævne i teksten" / "nævne i artiklen" is
  FORBIDDEN — this includes: nævnes/nævnt/omtales/omtalt/beskrevet/beskrives
  followed by i teksten/i artiklen/i kilden. Same for "der optræder i".

  If you find yourself needing such a phrase, you're constructing the wrong
  task shape — rewrite the instruction to ask about the TOPIC directly using
  a real semantic category rather than "mentioned in the text".

  Example (BAD): "Skriv et digt om producenten baseret på teksten."
  Example (GOOD): "Skriv et digt om producenten i naturens kredsløb."
  Example (BAD): "Hvilke steder nævnes i artiklen om Jylland?"
  Example (GOOD): "Nævn nogle karakteristiske steder i Jylland."
  Example (BAD, classification): "Hvilke af følgende fugle er nævnt i
    teksten: Trane, And, Fiskehejre, Stork, Papegøje?"
  Example (GOOD, classification): "Hvilke af følgende er vandfugle: Trane,
    And, Fiskehejre, Stork, Papegøje?"

  You may still USE facts from the source in your response — but reference
  them naturally (e.g. name Mogens Sandfær directly) rather than pointing at
  a text that isn't shown at inference time."""

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
    "context": "<Danish context OR empty string>",
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


_PAGEID_WHITELIST: set[int] | None = None


def load_pageid_whitelist(path="/mnt/data2/da_wiki_curation/pageids.tsv") -> set[int]:
    """Load the DA-wiki category-tree curation whitelist as a set of pageids."""
    global _PAGEID_WHITELIST
    if _PAGEID_WHITELIST is None:
        with open(path) as f:
            ids: set[int] = set()
            for line in f:
                s = line.split("\t", 1)[0].strip()
                if s.isdigit():
                    ids.add(int(s))
            _PAGEID_WHITELIST = ids
    return _PAGEID_WHITELIST


FOREIGN_COUNTRY_NOUNS = (
    "usa", "storbritannien", "england", "skotland", "wales", "irland",
    "tyskland", "østrig", "schweiz",
    "frankrig", "belgien", "nederlandene", "holland", "luxembourg",
    "italien", "spanien", "portugal", "grækenland", "cypern",
    "rusland", "ukraine", "polen", "tjekkiet", "slovakiet",
    "ungarn", "rumænien", "bulgarien", "serbien", "kroatien", "slovenien",
    "tyrkiet", "kina", "japan", "korea", "sydkorea", "nordkorea",
    "indien", "pakistan", "bangladesh", "vietnam", "thailand", "indonesien",
    "australien", "newzealand", "canada", "mexico", "cuba",
    "brasilien", "argentina", "chile", "colombia", "peru", "venezuela",
    "sydafrika", "nigeria", "kenya", "egypten", "marokko", "algeriet",
    "iran", "irak", "syrien", "israel", "libanon", "jordan",
    "sverige", "norge", "island", "finland", "estland", "letland", "litauen",
)

# Any standalone occurrence of a foreign country name in the intro is a
# strong signal the article is about a foreign subject.
FOREIGN_COUNTRY_ANY_RE = re.compile(
    r"\b(" + "|".join(FOREIGN_COUNTRY_NOUNS) + r")\b"
)


def _has_foreign_subject_marker(intro_lc):
    head = intro_lc[:400]
    # Adjective descriptor: "er en/et amerikansk X"
    for m in FOREIGN_MARKERS:
        if re.search(rf"\b(er|var) (en|et) {m}\b", head):
            return True
        if re.search(rf"\({m}\b", head):
            return True
    # Any foreign country name mentioned as standalone word.
    if FOREIGN_COUNTRY_ANY_RE.search(head):
        return True
    return False


def _mentions_denmark(intro_lc):
    return any(m in intro_lc for m in DANISH_MARKERS)


def looks_useful(row):
    """Pageid-whitelist ONLY: article must be in the DA-wiki category-tree
    curation. Plus minimal quality gates (disambig/stub rejection) to skip
    articles that would produce bad training rows even when in-scope."""
    pid = int(row.get("id", 0))
    if pid not in load_pageid_whitelist():
        return False
    # Minimal quality: reject stubs and disambig regardless of whitelist.
    text = row["text"]
    intro = text[:800]
    intro_lc = intro.lower()
    if any(m in intro_lc for m in DISQUALIFYING_TEXT_MARKERS):
        return False
    if len(intro.strip()) < 300 or intro.count("\n") > 15:
        return False
    return True


def clean_intro(text, max_chars=900):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out, total = [], 0
    for p in paras:
        if total + len(p) > max_chars:
            break
        out.append(p)
        total += len(p) + 1
    return "\n\n".join(out)


def source_for_category(row, category):
    """Category-specific source seeding.

    Full-source categories (closed_qa, summarization, information_extraction,
    classification): pass the full ~900-char intro.

    open_qa, general_qa: full intro as background knowledge (not for
    extraction), model draws general-knowledge answers.

    creative_writing: only the first sentence or two (~200 chars) — enough
    to identify the topic without forcing fact extraction.

    brainstorming: TITLE + first sentence only (~120 chars) — the model must
    brainstorm from general knowledge about the topic, not extract items
    from the source.
    """
    full = clean_intro(row["text"], max_chars=900)
    if category == "brainstorming":
        # Grab just the first sentence for a topic hint.
        first_sent = re.split(r"(?<=[.!?])\s+", full, maxsplit=1)[0]
        return first_sent[:180]
    if category == "creative_writing":
        # First 1-2 sentences (~250 chars).
        parts = re.split(r"(?<=[.!?])\s+", full)
        take = " ".join(parts[:2])
        return take[:280]
    return full


def parse_response(raw):
    """Strip code fences, parse JSON, normalize response array→string."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None, "parse_fail"
    # response array → "\n- " joined string
    resp = parsed.get("response")
    if isinstance(resp, list):
        parsed["response"] = "\n".join(
            f"- {x}" if not str(x).lstrip().startswith(("-", "*", "•")) else str(x)
            for x in resp)
    return parsed, None


async def generate_one(session, sem, key, seed_idx, row, category):
    source = source_for_category(row, category)
    prompt = PROMPT.format(category=category, title=row["title"],
                            source=source, rules=CATEGORY_RULES)
    body = {
        "model": "google/gemma-3-12b-it",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 800,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://claude-code-smoke",
        "X-Title": "wiki-grounded-da gemma smoke",
    }

    async with sem:
        for attempt in range(3):
            try:
                t0 = time.time()
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers, json=body, timeout=60,
                ) as resp:
                    data = await resp.json()
                dt = time.time() - t0
                if "choices" not in data:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt); continue
                    return {"seed_idx": seed_idx, "category": category,
                            "title": row["title"],
                            "error": f"api: {json.dumps(data)[:200]}"}
                raw = data["choices"][0]["message"]["content"]
                cost = data.get("usage", {}).get("cost", 0)
                in_tok = data.get("usage", {}).get("prompt_tokens", 0)
                out_tok = data.get("usage", {}).get("completion_tokens", 0)
                parsed, err = parse_response(raw)
                if err:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt); continue
                    return {"seed_idx": seed_idx, "category": category,
                            "title": row["title"], "error": err,
                            "raw": raw[:400],
                            "meta": {"latency_s": dt, "in_tok": in_tok,
                                     "out_tok": out_tok, "cost": cost}}
                parsed["seed_idx"] = seed_idx
                parsed["title"] = row["title"]
                parsed["meta"] = {"latency_s": dt, "in_tok": in_tok,
                                  "out_tok": out_tok, "cost": cost,
                                  "attempt": attempt + 1}
                return parsed
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt); continue
                return {"seed_idx": seed_idx, "category": category,
                        "title": row["title"], "error": f"exc: {str(e)[:200]}"}


def load_done_task_ids(out_path: Path) -> set[int]:
    """Load task_id (index in the shuffled schedule) values that already
    completed successfully. Errors are re-tried on resume."""
    done = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = r.get("task_id")
            if tid is not None and not r.get("error"):
                done.add(tid)
    return done


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--pool-scan", type=int, default=800)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/gemma12b_smoke_100.jsonl"))
    args = ap.parse_args()

    key = open("/home/jepsen/or").read().strip()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading Danish Wikipedia (streaming, scan {args.pool_scan})…", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.da", split="train",
                       streaming=True)
    pool = []
    t_scan = time.time()
    for i, row in enumerate(ds):
        if i >= args.pool_scan:
            break
        if looks_useful(row):
            pool.append({"seed_idx": i, "title": row["title"], "text": row["text"]})
        if (i + 1) % 5000 == 0:
            el = time.time() - t_scan
            eta = el * (args.pool_scan - i - 1) / (i + 1)
            print(f"  scanned {i+1}/{args.pool_scan}  kept {len(pool)} "
                  f"({100*len(pool)/(i+1):.1f}%)  {el:.0f}s  eta={eta:.0f}s",
                  flush=True)
    print(f"  final pool: {len(pool):,} / {args.pool_scan:,} "
          f"({100*len(pool)/args.pool_scan:.1f}%) in {time.time()-t_scan:.0f}s",
          flush=True)

    # Deterministic schedule: shuffle pool, assign categories, use task_id
    # as stable identifier for resume.
    rng = random.Random(args.seed)
    selected = rng.sample(pool, min(args.n, len(pool)))
    categories = [rng.choice(CATEGORIES) for _ in selected]

    # Resume: any task_id that already has a non-error row → skip.
    done = load_done_task_ids(args.out)
    print(f"resume: {len(done):,} rows already complete → skipping", flush=True)
    todo = [(tid, row, cat)
            for tid, (row, cat) in enumerate(zip(selected, categories))
            if tid not in done]
    print(f"to generate: {len(todo):,} rows", flush=True)
    if not todo:
        print("nothing to do")
        return

    t0 = time.time()
    n_done = n_ok = n_skip = n_fail = 0
    total_cost = 0.0
    total_in = total_out = 0
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(args.workers)
        tasks = [
            (tid, generate_one(session, sem, key, row["seed_idx"], row, cat))
            for tid, row, cat in todo
        ]
        # For as_completed we need the tid attached to the result; wrap:
        async def _run(tid, coro):
            r = await coro
            r["task_id"] = tid
            return r
        wrapped = [_run(tid, c) for tid, c in tasks]

        # APPEND mode — never truncate on resume
        with args.out.open("a") as f:
            for coro in asyncio.as_completed(wrapped):
                r = await coro
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1
                if r.get("error"):    n_fail += 1
                elif r.get("skip"):   n_skip += 1
                else:                 n_ok += 1
                meta = r.get("meta", {})
                total_cost += meta.get("cost", 0) or 0
                total_in   += meta.get("in_tok", 0) or 0
                total_out  += meta.get("out_tok", 0) or 0
                # More frequent gen-phase reporting.
                if n_done % 100 == 0 or n_done == len(wrapped):
                    el = time.time() - t0
                    rate = n_done / el
                    eta = (len(wrapped) - n_done) / rate if rate else 0
                    print(f"  {n_done:,}/{len(wrapped):,}  "
                          f"ok={n_ok:,} skip={n_skip} fail={n_fail}  "
                          f"{rate:.1f} rows/s  eta={eta:.0f}s  "
                          f"cost=${total_cost:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"\ndone in {elapsed:.0f}s; ok={n_ok} skip={n_skip} fail={n_fail}")
    print(f"tokens: {total_in:,} in + {total_out:,} out")
    print(f"cost: ${total_cost:.4f} (per 1000 ok: ${total_cost/max(n_ok,1)*1000:.2f})")
    if n_done:
        print(f"projected 15k cost: ${total_cost / n_done * 15000:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
