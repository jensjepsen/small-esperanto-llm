"""Smoke: rephrase wp-v2 questions into natural GSM8K-style Danish via
gemma-3-12b-it on OpenRouter, keeping the recipe-style ANSWER unchanged.

Idea: bridge the surface-form gap between wp-v2's recipe-cue phrasing
("Vi går baglæns", "Decimaltallet for X% er 0.X") and GSM8K's bare
natural language. Prints before/after side-by-side so we can eyeball
whether the rewrites (a) are natural Danish, (b) preserve numbers and
answer, (c) drop the recipe cues.

Usage:
    python scripts/smoke_wp_rephrase.py --n 20 --key-file /home/jepsen/or
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import aiohttp
from datasets import load_dataset

API = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemma-3-12b-it"

PROMPT = """Du får et matematisk tekstopgave-spørgsmål på dansk og facit.
Din opgave: skriv opgaven om til et **naturligt dansk tekstopgave-spørgsmål**
i samme stil som GSM8K — en typisk hverdagssituation, ingen skabelon-cues.

REGLER (VIGTIGT):
1. Alle tal skal bevares nøjagtigt.
2. Facit skal være det samme.
3. FJERN alle "opskrifts-vendinger" som "Vi går baglæns gennem procentregningen",
   "Decimaltallet for X% er ...", "Gang for at vende om", parentetiske hints
   som "(75% af Camillas andel)".
4. Skriv i naturligt hverdagsdansk med konkrete personer, situationer.
5. Kun ét spørgsmål til sidst — ingen mellemregninger i selve spørgsmålet.

SPØRGSMÅL:
{q}

FACIT:
{a}

Returnér KUN det omskrevne spørgsmål på én linje eller et kort afsnit, intet andet.
Ingen forklaring, ingen "Her er det omskrevne spørgsmål:", ingen kommentar."""


async def rephrase(session, sem, key, row):
    q = next(m["content"] for m in row["messages"] if m["role"] == "user")
    a = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(q=q, a=a)}],
        "temperature": 0.3,
        "max_tokens": 400,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    async with sem:
        for _ in range(3):
            try:
                async with session.post(API, headers=headers, json=body, timeout=60) as resp:
                    j = await resp.json()
                    if "choices" in j and j["choices"]:
                        return {"q_orig": q, "a": a, "q_new": j["choices"][0]["message"]["content"].strip()}
            except Exception as e:
                await asyncio.sleep(1.5)
        return {"q_orig": q, "a": a, "q_new": None}


NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")

# Danish number-word ↔ digit mapping. Both directions accepted so a wp-v2
# source with "5 timer" can be reworded as "fem timer" and vice versa.
# For 1 we accept the common forms: "en" (utrum, and general), "et" (neuter),
# "én" (stressed / when ambiguous with the indefinite article).
_WORD_TO_DIGIT: dict[str, str] = {
    "nul": "0",
    # NOTE: unaccented "en"/"et" are omitted — they overwhelmingly function
    # as the indefinite article in Danish, and treating them as the number
    # 1 causes constant false-positive number-added rejects. Only the
    # stressed form "én" is unambiguously numeric.
    "én": "1",
    "to": "2", "tre": "3", "fire": "4", "fem": "5",
    "seks": "6", "syv": "7", "otte": "8", "ni": "9",
    "ti": "10", "elleve": "11", "tolv": "12",
}
_DIGIT_TO_WORDS: dict[str, set[str]] = {}
for w, d in _WORD_TO_DIGIT.items():
    _DIGIT_TO_WORDS.setdefault(d, set()).add(w)
_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(_WORD_TO_DIGIT, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


# Ordinal contexts to skip: "7. klasse", "1. verdenskrig", "3. gang" etc.
# A digit immediately followed by `. letter` is an ordinal, not a numeric
# quantity relevant to the problem. Numeric measurements never take this form.
_ORDINAL_AFTER_RE = re.compile(r"^\.\s+[a-zæøå]")


def canon_numbers(text: str) -> set[str]:
    """Return canonical (digit-form) set of numeric tokens in `text`.
    Digits map to themselves; recognised Danish number-words (0-12) map to
    the corresponding digit string. Ordinal patterns (`7. klasse`) are
    excluded — Gemma frequently adds these as flavor identifiers."""
    out: set[str] = set()
    for m in NUM_RE.finditer(text):
        # Skip if this digit is functioning as an ordinal marker.
        if _ORDINAL_AFTER_RE.match(text[m.end():m.end() + 3]):
            continue
        out.add(m.group(0))
    for m in _WORD_RE.finditer(text):
        out.add(_WORD_TO_DIGIT[m.group(1).lower()])
    return out


# Danish fraction/portion phrases in NEW that can legitimately substitute
# for percentage/fraction digits in ORIG (e.g. "halvdelen" for "50%" or
# "1/2"). If NEW contains one of these phrases, the covered digits are
# subtracted from the missing_nums set — the semantic content survived
# even though the surface digit was replaced.
_FRACTION_SATISFIERS: list[tuple[re.Pattern[str], set[str]]] = [
    (re.compile(r"\bhalvdel\w*\b",         re.I), {"50", "2", "1", "0.5"}),
    (re.compile(r"\ben\s+tredjedel\b",     re.I), {"33", "3", "1"}),
    (re.compile(r"\btre\s+tredjedele?\b",  re.I), {"3"}),
    (re.compile(r"\ben\s+fjerdedel\b",     re.I), {"25", "4", "1"}),
    (re.compile(r"\btre\s+fjerdedele?\b",  re.I), {"75", "3", "4"}),
    (re.compile(r"\ben\s+femtedel\b",      re.I), {"20", "5", "1"}),
    (re.compile(r"\bto\s+femtedele?\b",    re.I), {"40", "2", "5"}),
    (re.compile(r"\ben\s+sjettedel\b",     re.I), {"6", "1"}),
    (re.compile(r"\bto\s+tredjedele?\b",   re.I), {"67", "2", "3"}),
]


def fraction_satisfies(text: str) -> set[str]:
    """Union of digit-strings that Danish fraction phrases in `text` cover."""
    covered: set[str] = set()
    for pat, digits in _FRACTION_SATISFIERS:
        if pat.search(text):
            covered |= digits
    return covered


# Ratio patterns "a:b" or "a/b" (small integer ratios) in ORIG. When Gemma
# rewrites "forholdet 2:5" as "2 ud af hver 7 bog", it introduces the SUM
# (a+b) as a new numeric token. That's a legitimate expansion, not new
# information. Similarly rewriting "2:3" as "2/5, 3/5 af det hele" uses
# the sum as a fraction denominator.
_RATIO_RE = re.compile(r"\b(\d+)\s*[:/]\s*(\d+)\b")


def ratio_sums(text: str) -> set[str]:
    """Digit-strings equal to a+b for each ratio a:b (or a/b) in text."""
    out: set[str] = set()
    for m in _RATIO_RE.finditer(text):
        try:
            a, b = int(m.group(1)), int(m.group(2))
            out.add(str(a + b))
        except ValueError:
            pass
    return out


# Numbers on the RHS of explicit arithmetic in ORIG ("70 + 60 = 130 km/t")
# are derived intermediates. Gemma correctly drops these per instruction #5
# ("ingen mellemregninger i selve spørgsmålet") — they should not count as
# missing when absent from NEW.
_DERIVED_RE = re.compile(
    r"\d+\s*[+\-*×·/]\s*\d+\s*=\s*(\d+(?:[.,]\d+)?)",
)


def derived_intermediates(text: str) -> set[str]:
    """Digit-strings that appear as `= N` after an explicit calc in text."""
    return {m.group(1) for m in _DERIVED_RE.finditer(text)}


RECIPE_CUES = [
    "Vi går baglæns", "vi går baglæns",
    "Decimaltallet for", "decimaltallet for",
    "Multiplikatoren er", "multiplikatoren er",
    "Gang for at vende om", "gang for at vende om",
    "Vend multiplikationen om",
    "altså tæller", "Ved at fordele",
]
DA_MARKERS = re.compile(
    r"\b(og|er|har|den|det|de|i|på|til|for|med|"
    r"hvad|hvor|mange|hvilken|beregn|find|antal|kroner|per)\b", re.I)
META_STARTS = ("Her er", "Det omskrevne", "Omskrevet:", "```", "Her kommer",
               "Her følger", "Følgende er")


def check(orig_q: str, new_q: str) -> str | None:
    """Return rejection reason string, or None if passes."""
    if new_q is None:
        return "no_response"
    q = new_q.strip()
    if len(q) < 30:
        return "too_short"
    # Short origs (e.g. 40 chars) naturally expand more than 3×; use
    # max(3×, 250) so we still cap runaway rambles on long origs.
    if len(q) > max(3 * len(orig_q), 250) and len(q) > 3 * len(orig_q):
        # keep hard cap for long rambles only when the ratio is high
        if len(q) > 800:
            return "too_long"
    nums_orig = canon_numbers(orig_q)
    nums_new  = canon_numbers(q)
    # Missing: strip fraction-word substitutes and derived intermediates from
    # orig before comparing (both are legitimately absent from new).
    missing = nums_orig - nums_new - fraction_satisfies(q) - derived_intermediates(orig_q)
    # Added: strip ratio sums (a+b for "a:b" in orig) — Gemma legitimately
    # expands "2:5" as "2 ud af hver 7 bog" or "2/7, 5/7".
    # Also handle the en/én asymmetry: wp-v2's "en pumpe" (article form of 1,
    # ambiguous, so not canonicalized) becomes Gemma's "én pumpe" (stressed,
    # canonicalized to 1). Count that "1" as already-present when ORIG has
    # a bare indefinite article.
    orig_has_article_one = re.search(r"\b(en|et)\b", orig_q) is not None
    exempt_added = ratio_sums(orig_q) | ({"1"} if orig_has_article_one else set())
    added   = nums_new - nums_orig - exempt_added
    if missing:
        return f"missing_nums={sorted(missing)}"
    if added:
        return f"added_nums={sorted(added)}"
    for cue in RECIPE_CUES:
        if cue in q:
            return f"recipe_leak='{cue}'"
    if any(q.startswith(m) for m in META_STARTS):
        return "meta_start"
    if not DA_MARKERS.search(q):
        return "not_danish"
    if not (q.rstrip().endswith("?") or
            re.search(r"\b(hvor mange|hvad|beregn|find)\b", q, re.I)):
        return "not_a_question"
    return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--key-file", type=Path, default=Path("/home/jepsen/or"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--focus", choices=["all", "percent", "rate"], default="all")
    ap.add_argument("--show-passes", type=int, default=15)
    ap.add_argument("--show-rejects", type=int, default=10)
    args = ap.parse_args()

    key = args.key_file.read_text().strip()
    ds = load_dataset("jensjepsen/danish-word-problems-v2", "sft", split="train")

    if args.focus != "all":
        pat_pct = re.compile(r"\d+\s*%|procent|rabat|baglæns")
        pat_rate = re.compile(r"km/t|km/time|/time|/dag|pr\.?\s*(min|time|dag|uge|år)|Gang for at vende om", re.I)
        pat = pat_pct if args.focus == "percent" else pat_rate
        rows = []
        for r in ds.shuffle(seed=args.seed):
            q = next(m["content"] for m in r["messages"] if m["role"] == "user")
            if pat.search(q):
                rows.append(r)
                if len(rows) >= args.n: break
    else:
        rows = list(ds.shuffle(seed=args.seed).select(range(args.n)))
    print(f"picked {len(rows)} rows (focus={args.focus})", flush=True)

    sem = asyncio.Semaphore(10)
    async with aiohttp.ClientSession() as sess:
        tasks = [rephrase(sess, sem, key, r) for r in rows]
        results = await asyncio.gather(*tasks)

    passes, rejects = [], []
    reject_counts: dict[str, int] = {}
    for r in results:
        reason = check(r["q_orig"], r["q_new"])
        if reason is None:
            passes.append(r)
        else:
            rejects.append({**r, "reason": reason})
            reject_counts[reason.split("=")[0]] = reject_counts.get(reason.split("=")[0], 0) + 1

    n = len(results)
    print(f"\n=== RESULTS ({n} rows) ===")
    print(f"pass:   {len(passes)}  ({100*len(passes)/n:.0f}%)")
    print(f"reject: {len(rejects)} ({100*len(rejects)/n:.0f}%)")
    if reject_counts:
        print("reject reasons:")
        for k, v in sorted(reject_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {v:3d}  {k}")

    import random as _r
    _r.seed(3)
    print(f"\n=== {args.show_passes} PASSES ===")
    for i, r in enumerate(_r.sample(passes, min(args.show_passes, len(passes)))):
        print("-" * 72)
        print(f"[{i}]")
        print("ORIG:", r["q_orig"])
        print("NEW :", r["q_new"])
        print("ANS :", r["a"][:220])

    print(f"\n=== up to {args.show_rejects} REJECTS ===")
    for i, r in enumerate(rejects[:args.show_rejects]):
        print("-" * 72)
        print(f"[{i}] REASON: {r['reason']}")
        print("ORIG:", r["q_orig"])
        print("NEW :", r["q_new"])


if __name__ == "__main__":
    asyncio.run(main())
