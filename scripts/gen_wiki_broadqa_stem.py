"""STEM-focused BROAD Q/A generation for Danish Wikipedia.

Fork of `gen_wiki_closedqa_stem.py` optimized for KNOWLEDGE INSTILLATION
rather than fact retrieval. Generates fewer, longer Q/A per article where
answers are explanatory prose (150-400 words) rather than 3-15 word labels.

Rationale: v11's STEM addition (short Q/A, +0.9pp on SciQ) showed the
per-answer gradient signal is too thin for real knowledge embedding at
400M scale. Broad Q/A gives 200-400 token gradient per fact, teaching the
model to compose facts in context rather than lookup labels.

Reads same STEM pageid TSV. Differences from closedqa_stem:
  - --n-qa-per 4 (was 16) — fewer, longer Q/As per article
  - --wiki-max-chars 4000 (was 2500) — broader Qs need more source context
  - max_tokens 4000 (was 3200) — 4 answers × ~500 tokens + JSON overhead
  - Prompt: emphasizes explanatory questions and prose answers, forbids
    the "under 25 words" concise-answer instruction

Output: JSONL, one row per Q/A pair (same schema as gen_wiki_closedqa_stem):
    {orig_pageid, orig_title, tier="stem-broad", q, a}

Usage:
    python scripts/gen_wiki_broadqa_stem.py \\
        --tsv data/da_wiki_stem/pageids_d2_core_filtered.tsv \\
        --out data/wiki_broadqa_stem/rows.jsonl \\
        --n-qa-per 4 --concurrency 50
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
BREDE, forklarende spørgsmål-og-svar-par, som er GRUNDET i artiklen.

Formålet er IKKE at teste enkelte fakta — det er at give en LM træningsmateriale
til at LÆRE emnet i sammenhæng. Spørgsmålene skal derfor invitere til
uddybende svar (150-400 ord), og svarene skal væve fakta sammen i
sammenhængende dansk prosa med kausal argumentation.

TITEL: {title}

ARTIKEL:
{source}

SPØRGSMÅLSTYPER (vælg {n_qa} der passer godt til artiklen):
  - Mekanisme:      "Hvordan fungerer X?" / "Hvad sker der, når X?"
  - Betydning:      "Hvorfor er X vigtig?" / "Hvilken rolle spiller X?"
  - Sammenligning:  "Hvordan adskiller X sig fra Y?" (kun hvis Y nævnes)
  - Historie:       "Beskriv X's udvikling og opdagelse."
  - Anvendelse:     "Hvordan bruges X i praksis?"
  - Overblik:       "Forklar X og dets vigtigste træk."
  - Sammensætning:  "Hvad består X af, og hvordan hænger delene sammen?"

STAND-ALONE-KRAV (VIGTIGST):
Hvert spørgsmål SKAL kunne stilles til én, der ALDRIG har set artiklen.

FORBUDT (må aldrig indgå — hverken i Q eller A):
  - Ordene "artiklen", "teksten", "kilden", "passagen", "indholdet",
    "uddraget" i ENHVER form (subjekt-position, objekt-position, ejefald)
  - Præpositions-referencer: "i artiklen", "i teksten", "ifølge kilden",
    "her nævnes", "denne artikel", "ovenstående"
  - Subjekt-referencer: "Artiklen nævner...", "Teksten beskriver...",
    "Kilden argumenterer..." — FORBUDT, uanset ordstilling
  - Meta-vendinger: "omtales som", "beskrives som", "hvad står der om"
  - Implicitte referencer: "de tre nævnte", "af de ovenfor", "blandt de nævnte"

FORBUDTE EKSEMPLER (må IKKE produceres):
  ❌ Q: "Artiklen nævner, at fysikkens ligninger ofte 'gættes'. Hvad betyder det?"
  ❌ Q: "Artiklen introducerer begreberne 'ket' og 'bra'. Hvorfor?"
  ❌ Q: "Ifølge teksten, hvordan fungerer X?"
  ❌ A: "Ifølge artiklen består X af..."

SVARENES FORM (VIGTIGT):
  ✓ Skal være 150-400 ord (typisk 3-5 sammenhængende sætninger eller mere).
  ✓ Skal være i sammenhængende prosa, IKKE punktopstillinger.
  ✓ Skal bruge kausal argumentation ("fordi", "derfor", "på grund af", "hvilket
    medfører"), sammenligninger, og konkrete eksempler fra artiklen.
  ✓ Skal binde flere fakta sammen (ikke bare svare på det direkte spørgsmål,
    men også placere det i kontekst med relaterede oplysninger fra artiklen).
  ✗ Ikke bare ét kort faktum eller ét ord.
  ✗ Ikke opremsninger uden forklaring.

GODE EKSEMPLER (BREDE spørgsmål + UDDYBENDE svar):
  ✓ Q: "Hvordan omdanner planter sollys til kemisk energi via fotosyntese?"
    A: "Fotosyntese foregår primært i planternes kloroplaster, hvor klorofyl
       fanger fotoner fra sollys. Denne energi bruges til at splitte vandmolekyler
       i den lysafhængige fase, hvilket frigiver ilt som biprodukt og genererer
       ATP og NADPH. Disse energibærere driver derefter Calvin-cyklus, hvor
       kuldioxid fra atmosfæren omdannes til glukose. Processen er fundamental
       for alt liv på Jorden, fordi den både producerer den ilt vi indånder og
       den kemiske energi, der driver næsten alle fødekæder..."

  ✓ Q: "Hvorfor er erbium vigtigt inden for optisk kommunikation?"
    A: "Erbium spiller en central rolle i moderne fiberoptik, fordi erbium-ioner
       kan forstærke lyssignaler ved bølgelængden 1550 nanometer — netop den
       bølgelængde, hvor optiske fibre har mindst signaltab. Erbiumdopede
       fiberforstærkere (EDFA) blev udviklet i 1980'erne og revolutionerede
       telekommunikation ved at gøre det muligt at forstærke signaler over
       lange afstande uden først at konvertere dem til elektricitet..."

DÅRLIGE eksempler (for smalle):
  ✗ Q: "Hvornår blev fotosyntese opdaget?" (for snævert, ét faktum)
  ✗ Q: "Hvad er kemisk symbol for erbium?" (retrieval, ikke forklaring)

OUTPUT: én JSON-liste med præcis {n_qa} objekter:
[
  {{"q": "...", "a": "..."}},
  {{"q": "...", "a": "..."}}
]

Kun JSON, ingen kommentarer, ingen markdown-fences."""


# Post-hoc reject filter: any of these substrings in a question means it's
# a self-reference. Cheap belt-and-suspenders to catch what the prompt misses.
_SELF_REF_PATTERNS = [
    # Any occurrence of artiklen/teksten/kilden — subject OR object form.
    # In STEM domain almost no legitimate use ("hvad handler artiklen om"
    # is exactly what we're filtering; "artikel" without definite article
    # is fine — think grammatical article, industrial article etc.)
    re.compile(r"\b(artiklen|artiklens|teksten|tekstens|kilden|kildens|"
               r"passagen|uddraget|indholdet)\b", re.I),
    # Reference verbs commonly used with implicit source ("nævnes", "omtales",
    # "beskrives", "diskuteres" — all in passive form referring to what "the
    # article/text" says)
    re.compile(r"\b(n[æa]vnes|omtales?|beskrives?|diskuteres|forklares|"
               r"pr[æa]senteres|angives?)\s+(det|som|at|hvordan|hvorfor|hvad)\b", re.I),
    re.compile(r"\bhvad st[åa]r der\b", re.I),
    re.compile(r"\b(denne|denne her|denne ovenst[åa]ende) (artikel|tekst|passage|kilde)\b", re.I),
    re.compile(r"\b(ovenn[æa]vnte|nedenn[æa]vnte|ovenst[åa]ende|"
               r"nedenst[åa]ende|foreg[åa]ende)\b", re.I),
    # "de X nævnte" / "af de X ovenfor" / "blandt de nævnte"
    re.compile(r"\b(af |blandt )?de\s+(to|tre|fire|fem|seks|syv|otte|ni|ti|"
               r"\d+)\s+(n[æa]vnte|ovenfor|ovenn[æa]vnte|nedenn[æa]vnte)\b", re.I),
    re.compile(r"\bblandt de n[æa]vnte\b", re.I),
    re.compile(r"\baf de (ovenst[åa]ende|nedenst[åa]ende)\b", re.I),
    # "ifølge X" where X is any source-word
    re.compile(r"\bif[øo]lge\s+(artiklen|teksten|kilden|indholdet|"
               r"det ovenfor|nedenst[åa]ende|ovenst[åa]ende)\b", re.I),
]


def has_self_ref(text: str) -> bool:
    """Check both Q and A for source references."""
    return any(p.search(text) for p in _SELF_REF_PATTERNS)


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
        "temperature": 0.5,
        "max_tokens": 4000,  # 4 Q/A × ~700 tokens each + JSON overhead
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
            picks.append((float(length), pid, title, "stem-broad"))
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
    ap.add_argument("--n-qa-per", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--key-file", type=Path, default=Path("/home/jepsen/or"))
    ap.add_argument("--wiki-max-chars", type=int, default=4000)
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
            for qa in result["parsed"]:
                rejected = has_self_ref(qa["q"]) or has_self_ref(qa["a"])
                if rejected:
                    n_reject_ref += 1
                rows_to_write.append({
                    "orig_pageid": pid, "orig_title": title, "tier": tier,
                    "rejected": rejected,
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
