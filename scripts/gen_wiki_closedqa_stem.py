"""STEM-focused dense closed_qa generation for Danish Wikipedia.

Reads a STEM pageid TSV (from `fetch_da_wiki_curation.py` + length-filter,
format: pageid<TAB>title<TAB>text_length), fetches each article's intro
from wikimedia/wikipedia HF dataset, and generates N distinct factual
Q/A pairs per article via gemma-3-12b on OpenRouter.

Fork of `gen_wiki_closedqa_v4.py`. Differences:
  - Input: 3-col STEM tsv instead of 7-col salience tsv (no --tiers)
  - Defaults: --n-qa-per 16 (was 6), --wiki-max-chars 2500 (was 1500)
  - Output rows tagged `tier="stem"` for downstream provenance

Output: JSONL, one row per successful Q/A pair:
    {orig_pageid, orig_title, tier="stem", q, a}

Resume-safe: reads existing output, skips already-processed pageids.

Usage:
    python scripts/gen_wiki_closedqa_stem.py \\
        --tsv data/da_wiki_stem/pageids_d2_core_filtered.tsv \\
        --out data/wiki_closedqa_stem/rows.jsonl \\
        --n-qa-per 16 --concurrency 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import aiohttp
from datasets import load_dataset

API = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemma-3-12b-it"

DENSE_PROMPT = """Du får en dansk Wikipedia-artikel og skal generere {n_qa}
forskellige spørgsmål-og-svar-par, som er GRUNDET i artiklen — dvs. hvert
svar SKAL kunne udledes direkte af teksten.

TITEL: {title}

ARTIKEL:
{source}

REGLER:

STAND-ALONE-KRAV (VIGTIGST):
Hvert spørgsmål SKAL kunne stilles til én, der ALDRIG har set artiklen —
dvs. den, der læser spørgsmålet, må IKKE kunne vide at der findes en
kildetekst. Spørgsmålet SKAL kun bruge alment kendt kontekst.

DERFOR FORBUDT:
  - Direkte referencer til kilden: "i artiklen", "i teksten", "ifølge kilden",
    "her nævnes", "Wikipedia", "denne artikel", "ovenstående", "nedenstående"
  - Meta-vendinger: "omtales som", "beskrives som", "hvad står der om"
    (når det henviser til at noget står i teksten)
  - Implicitte referencer: "de fire nævnte", "af de tre ovenfor",
    "blandt de nævnte", "af de ovenstående", "af de fem punkter"
  - Ifølge-vendinger: "ifølge det ovenfor", "det følgende", "ovenfor omtalte"

FORBUDTE EKSEMPLER (må IKKE produceres):
  ❌ Q: "Hvilket sprog omtales som 'nynorsk' i artiklen?"
  ❌ Q: "Ifølge teksten, hvornår blev landet grundlagt?"
  ❌ Q: "Hvilken kommune har det største areal blandt de fire nævnte?"
  ❌ Q: "Hvad står der om landbrug i artiklen?"
  ❌ Q: "Hvem er ifølge kilden nuværende statsminister?"

GODE EKSEMPLER (fri-stående faktaspørgsmål):
  ✓ Q: "Hvad hedder Norges andet officielle skriftsprog udover bokmål?"
  ✓ Q: "Hvornår blev Norge selvstændigt?"
  ✓ Q: "Hvilken kommune i hovedstadsområdet har det største areal?"
  ✓ Q: "Hvilken erhvervsgren er dominerende i Sydsverige?"

ØVRIGE REGLER:
1. Alle {n_qa} spørgsmål skal handle om FORSKELLIGE fakta fra artiklen.
2. Ingen spørgsmål må være for generiske ("Hvad handler emnet om?").
3. Alle svar skal være konkrete og korrekte ifølge artiklen.
4. Blandede spørgsmålstyper: hvornår/hvem/hvor/hvordan/hvorfor/hvad.
5. Hvis du ikke kan formulere spørgsmålet fri-stående uden at referere til
   teksten, så SPRING det faktum over og vælg et andet.
6. Hold svarene KONCISE — helst under 25 ord pr. svar. Kun det centrale
   faktum, ingen unødige forklaringer eller sidebemærkninger.

OUTPUT: én JSON-liste med præcis {n_qa} objekter:
[
  {{"q": "...", "a": "..."}},
  {{"q": "...", "a": "..."}}
]

Kun JSON, ingen kommentarer, ingen markdown-fences."""


# Post-hoc reject filter: any of these substrings in a question means it's
# a self-reference. Cheap belt-and-suspenders to catch what the prompt misses.
_SELF_REF_PATTERNS = [
    re.compile(r"\bi (artiklen|teksten|kilden|kilden(s)?)\b", re.I),
    re.compile(r"\bif[øo]lge (artiklen|teksten|kilden|det ovenfor|"
               r"nedenst[åa]ende|ovenst[åa]ende)\b", re.I),
    re.compile(r"\bomtales? som\b", re.I),
    re.compile(r"\bbeskrives? som\b", re.I),
    re.compile(r"\bhvad st[åa]r der\b", re.I),
    re.compile(r"\b(denne|denne her) (artikel|tekst)\b", re.I),
    re.compile(r"\b(ovenn[æa]vnte|nedenn[æa]vnte|ovenst[åa]ende|"
               r"nedenst[åa]ende)\b", re.I),
    # "de X nævnte" / "af de X ovenfor" / "blandt de nævnte"
    re.compile(r"\b(af |blandt )?de\s+(to|tre|fire|fem|seks|syv|otte|ni|ti|"
               r"\d+)\s+(n[æa]vnte|ovenfor|ovenn[æa]vnte|nedenn[æa]vnte)\b", re.I),
    re.compile(r"\bblandt de n[æa]vnte\b", re.I),
    re.compile(r"\baf de (ovenst[åa]ende|nedenst[åa]ende)\b", re.I),
]


def has_self_ref(q: str) -> bool:
    return any(p.search(q) for p in _SELF_REF_PATTERNS)


def clean_intro(text: str, max_chars: int = 1500) -> str:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out, total = [], 0
    for p in paras:
        if total + len(p) > max_chars:
            break
        out.append(p)
        total += len(p) + 1
    return "\n\n".join(out)


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0].strip()
    return raw


def _validate_qa_list(data) -> list[dict] | None:
    if not isinstance(data, list):
        return None
    ok = [d for d in data
          if isinstance(d, dict) and "q" in d and "a" in d
          and isinstance(d["q"], str) and isinstance(d["a"], str)
          and d["q"].strip() and d["a"].strip()]
    return ok or None


def salvage_partial_json(raw: str) -> list[dict] | None:
    """Extract as many valid {q,a} objects as possible from a truncated
    JSON array. Trims the raw text at the last complete `}` before the
    truncation point, closes the array with `]`, and retries json.loads.
    Falls back to per-object regex extraction if that still fails."""
    raw = _strip_fences(raw)
    if not raw.startswith("["):
        return None
    # Find the last closing brace that ends a valid object at top level
    depth = 0
    last_good = -1
    for i, c in enumerate(raw):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                last_good = i
    if last_good == -1:
        return None
    salvaged = raw[:last_good + 1] + "]"
    try:
        data = json.loads(salvaged)
        return _validate_qa_list(data)
    except json.JSONDecodeError:
        pass
    # Last-ditch: extract each top-level {…} block individually
    objs, depth, start = [], 0, -1
    for i, c in enumerate(raw):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    d = json.loads(raw[start:i + 1])
                    if isinstance(d, dict) and "q" in d and "a" in d:
                        objs.append(d)
                except json.JSONDecodeError:
                    pass
                start = -1
    return _validate_qa_list(objs)


def parse_json_list(raw: str) -> list[dict] | None:
    raw = _strip_fences(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return salvage_partial_json(raw)
    return _validate_qa_list(data)


async def gen_one_article(session, sem, key, pageid, title, tier, source, n_qa):
    prompt = DENSE_PROMPT.format(title=title, source=source, n_qa=n_qa)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 3200,  # 16 Q/A × ~180 tokens each + JSON overhead
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://claude-code-wiki-closedqa-v4",
        "X-Title": "wiki-closedqa-v4",
    }
    async with sem:
        for backoff in (1, 3, 8):
            try:
                async with session.post(API, headers=headers, json=body, timeout=90) as r:
                    if r.status == 429:
                        await asyncio.sleep(backoff * 2); continue
                    j = await r.json()
                    if "choices" not in j or not j["choices"]:
                        await asyncio.sleep(backoff); continue
                    text = j["choices"][0]["message"]["content"]
                    usage = j.get("usage", {}) or {}
                    in_t = int(usage.get("prompt_tokens", 0))
                    out_t = int(usage.get("completion_tokens", 0))
                    parsed = parse_json_list(text)
                    return {
                        "pageid": pageid, "title": title, "tier": tier,
                        "parsed": parsed, "raw": text,
                        "in": in_t, "out": out_t,
                    }
            except (asyncio.TimeoutError, aiohttp.ClientError):
                await asyncio.sleep(backoff)
    return {"pageid": pageid, "title": title, "tier": tier,
            "parsed": None, "raw": None, "in": 0, "out": 0}


def load_stem_tsv(path: Path) -> list[tuple[float, int, str, str]]:
    """Read pageid<TAB>title<TAB>text_length TSV, return list matching the
    (score, pageid, title, tier) shape expected downstream. Score uses
    text_length as a rough proxy so longer articles get priority."""
    picks = []
    with path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            pid = int(parts[0])
            title = parts[1]
            length = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            picks.append((float(length), pid, title, "stem"))
    picks.sort(reverse=True)
    return picks


def load_done_pageids(out_path: Path) -> set[int]:
    if not out_path.exists():
        return set()
    seen: set[int] = set()
    with out_path.open() as f:
        for line in f:
            try:
                seen.add(json.loads(line)["orig_pageid"])
            except Exception:
                pass
    return seen


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path,
                    default=Path("/home/jepsen/src/espllm/data/da_wiki_stem/pageids_d2_core_filtered.tsv"),
                    help="STEM pageid TSV: pageid<TAB>title<TAB>text_length")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-qa-per", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--key-file", type=Path, default=Path("/home/jepsen/or"))
    ap.add_argument("--wiki-max-chars", type=int, default=2500)
    args = ap.parse_args()

    key = args.key_file.read_text().strip()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    picks = load_stem_tsv(args.tsv)
    if args.n:
        picks = picks[:args.n]
    print(f"STEM tsv={args.tsv}  picked={len(picks):,} articles", flush=True)

    done = load_done_pageids(args.out)
    print(f"already done: {len(done):,}", flush=True)
    todo = [(s, pid, t, tier) for s, pid, t, tier in picks if pid not in done]
    print(f"todo: {len(todo):,}", flush=True)
    if not todo:
        print("nothing to do."); return

    # Load wiki dataset and build title→text lookup (one pass over the ds)
    want_titles = {t for _, _, t, _ in todo}
    print(f"loading wikimedia/wikipedia 20231101.da (looking up {len(want_titles):,} titles) …",
          flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.da", split="train")
    title_to_text: dict[str, str] = {}
    for row in ds:
        if row["title"] in want_titles:
            title_to_text[row["title"]] = row["text"]
            if len(title_to_text) == len(want_titles):
                break
    print(f"  matched {len(title_to_text):,}/{len(want_titles):,} in wiki dump",
          flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    n_articles_ok = n_rows = n_reject_ref = n_parse_fail = n_api_fail = 0
    tok_in = tok_out = 0
    t0 = time.time()

    async def process(seed, session, f_out):
        nonlocal n_articles_ok, n_rows, n_reject_ref, n_parse_fail, n_api_fail
        nonlocal tok_in, tok_out
        _, pid, title, tier = seed
        text = title_to_text.get(title)
        if not text:
            return  # skip; article missing from wiki dump
        source = clean_intro(text, args.wiki_max_chars)
        result = await gen_one_article(session, sem, key, pid, title, tier,
                                        source, args.n_qa_per)
        rows_to_write = []
        if result["parsed"] is None:
            if result["raw"] is None:
                n_api_fail += 1
            else:
                n_parse_fail += 1
        else:
            n_articles_ok += 1
            kept = [qa for qa in result["parsed"] if not has_self_ref(qa["q"])]
            n_reject_ref += (len(result["parsed"]) - len(kept))
            for qa in kept:
                rows_to_write.append({
                    "orig_pageid": pid, "orig_title": title, "tier": tier,
                    "q": qa["q"].strip(), "a": qa["a"].strip(),
                })
            n_rows += len(rows_to_write)
        async with lock:
            for r in rows_to_write:
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
            f_out.flush()
            tok_in += result.get("in", 0)
            tok_out += result.get("out", 0)
            done_now = n_articles_ok + n_parse_fail + n_api_fail
            if done_now % args.report_every == 0 or done_now == len(todo):
                el = time.time() - t0
                eta = el * (len(todo) - done_now) / max(done_now, 1)
                # gemma-3-12b-it real pricing: $0.05/M in, $0.15/M out
                cost = tok_in * 0.05 / 1e6 + tok_out * 0.15 / 1e6
                print(
                    f"[{done_now:6d}/{len(todo)}] "
                    f"ok={n_articles_ok} parse_fail={n_parse_fail} api_fail={n_api_fail}  "
                    f"rows={n_rows} ref_reject={n_reject_ref}  "
                    f"cost=${cost:.2f} eta={eta/60:.0f}m", flush=True)

    async with aiohttp.ClientSession() as session:
        with args.out.open("a") as f_out:
            tasks = [process(s, session, f_out) for s in todo]
            await asyncio.gather(*tasks)

    cost = tok_in * 0.05 / 1e6 + tok_out * 0.15 / 1e6
    print("\n=== gen_wiki_closedqa_v4 done ===")
    print(f"articles ok:     {n_articles_ok:,}")
    print(f"parse failures:  {n_parse_fail:,}")
    print(f"api failures:    {n_api_fail:,}")
    print(f"rows written:    {n_rows:,}")
    print(f"ref-rejects:     {n_reject_ref:,}  (post-hoc filter)")
    print(f"cost:            ${cost:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
