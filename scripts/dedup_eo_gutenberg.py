"""Title-level dedup of our trusted EN-Gutenberg queues against EO Gutenberg.

Strategy:
  1. Extract proper nouns from EO titles (capitalized non-initial words).
     Strip Esperanto noun endings (-o, -oj, -ino, -ina, -on) to recover
     the EN equivalent stem (Alicio → alic; Robinsono → robinson).
  2. For each EN book in our queue, if SAME author has an EO entry AND
     one of EO's stripped proper nouns appears in the EN title (case-
     insensitive substring), flag as duplicate.
  3. Otherwise keep it.

This catches "Hedda Gabler" ↔ "Hedda Gabler", "Alice's Adventures" ↔
"La Aventuroj de Alicio en Mirlando", "Robinson Crusoe" ↔ "Robinsono
Kruso", etc. Misses translation-renames like "An Enemy of the People"
↔ "Popolmalamiko" — those just stay duplicated, which is acceptable
noise (an extra MT pass on 5-10 books out of 926).
"""
import argparse
import csv
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

CATALOG = Path("/mnt/data2/sci_books/gutenberg_catalog.csv")

EO_SUFFIXES = ("ojn", "oj", "in", "ino", "ina", "on", "o")


def strip_eo_suffix(word: str) -> str:
    """Strip Esperanto noun ending; return the bare stem."""
    w = word.lower()
    for suf in EO_SUFFIXES:
        if w.endswith(suf) and len(w) > len(suf) + 1:
            return w[: -len(suf)]
    return w


EO_STOP = {"la", "el", "de", "kaj", "tri", "du", "unu", "kvin", "ses", "nau",
           "dek", "cent", "mil", "en", "al", "kun", "sub", "sur", "pri",
           "the", "a", "an", "of", "and", "or"}

# Generic genre/structure words that occur in many titles independently;
# matches via these aren't real duplicate evidence.
STEM_BLACKLIST = {"novel", "stori", "rakont", "poem", "poemoj", "tales",
                  "drama", "dram", "tragedi", "komedi", "verkoj",
                  "elekt", "selekt", "selectiv", "select", "rom", "roman",
                  "memori", "memoir", "biograf", "bigraf", "pri",
                  "verk", "libr", "book", "epos", "aktoj", "act",
                  "essai", "essay", "essa", "lectur", "lectu"}


def proper_nouns(title: str) -> list[str]:
    """All content tokens 4+ chars (case-insensitive) minus stopwords; EO-stripped stems.
    EO titles are sentence-case so we can't rely on capitalization for proper nouns."""
    tokens = re.findall(r"[A-Za-zĈĜĤĴŜŬĉĝĥĵŝŭ]+", title)
    out = []
    for t in tokens:
        if t.lower() in EO_STOP:
            continue
        if len(t) < 4:
            continue
        out.append(strip_eo_suffix(t))
    return out


def en_tokens(title: str) -> list[str]:
    """Lowercase word tokens from EN title."""
    return re.findall(r"[a-z]+", title.lower())


def prefix_overlap(a: str, b: str) -> int:
    """Length of common leading prefix between a and b."""
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n


def fuzzy_match(eo_stem: str, en_words: list[str], min_overlap: int = 4) -> bool:
    """True if any EN word shares >= min_overlap leading chars with EO stem."""
    if len(eo_stem) < min_overlap:
        return False
    for w in en_words:
        if prefix_overlap(eo_stem, w) >= min_overlap:
            return True
    return False


def author_surname(authors_field: str) -> str:
    """First author's surname (text before first comma in first segment)."""
    if not authors_field:
        return ""
    first = authors_field.split(";")[0].strip()
    return first.split(",")[0].strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=["sci", "lit", "econ"])
    args = ap.parse_args()

    in_path = Path(f"/mnt/data2/sci_books/{args.profile}_trusted.jsonl")
    out_path = Path(f"/mnt/data2/sci_books/{args.profile}_deduped.jsonl")

    # Build EO Gutenberg index: surname → list of (title, proper_noun_stems)
    catalog = list(csv.DictReader(CATALOG.open()))
    eo_books = [r for r in catalog
                if r.get("Language", "").lower() == "eo"
                and r.get("Type", "") == "Text"]
    eo_index: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for r in eo_books:
        surname = author_surname(r.get("Authors", ""))
        if surname:
            eo_index[surname].append((r["Title"], proper_nouns(r["Title"])))
    print(f"EO Gutenberg: {len(eo_books)} books, {len(eo_index)} unique authors")

    # Load our queue
    rows = [json.loads(l) for l in in_path.open()]
    print(f"input: {len(rows)} books in profile '{args.profile}'")

    # Pre-build EO content-token sets per author (skip stopwords + blacklist)
    eo_content: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for r in eo_books:
        surname = author_surname(r.get("Authors", ""))
        if not surname:
            continue
        toks = {strip_eo_suffix(t)
                for t in re.findall(r"[A-Za-zĈĜĤĴŜŬĉĝĥĵŝŭ]+", r["Title"].lower())
                if len(t) >= 4 and t not in EO_STOP}
        toks -= STEM_BLACKLIST
        if toks:
            eo_content[surname].append((r["Title"], toks))

    kept, dropped = [], []
    for r in rows:
        surname = author_surname(r.get("authors", ""))
        eo_matches = eo_content.get(surname, [])
        en_toks = {t for t in en_tokens(r["title"])
                   if len(t) >= 4 and t not in EO_STOP and t not in STEM_BLACKLIST}

        match_reason = None
        for eo_title, eo_toks in eo_matches:
            # Token-level fuzzy match: how many EO tokens have a close EN token?
            hits = 0
            best_pair = None
            for eo_t in eo_toks:
                close = difflib.get_close_matches(eo_t, en_toks, n=1, cutoff=0.75)
                if close:
                    hits += 1
                    if best_pair is None:
                        best_pair = (eo_t, close[0])
            # Duplicate when:
            #   - 2+ EO tokens match  (very strong signal across translations)
            #   - OR 1 hit AND EO has ≤3 content tokens (likely a proper-noun-heavy
            #     title; the one matched token is decisive)
            if hits >= 2 or (hits >= 1 and len(eo_toks) <= 3):
                match_reason = (f"{eo_title!r}  eo_tok='{best_pair[0]}' ↔ "
                                f"en_tok='{best_pair[1]}'  ({hits}/{len(eo_toks)} matched)")
                break
        if match_reason:
            r["dup_reason"] = match_reason
            dropped.append(r)
        else:
            kept.append(r)

    with out_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nkept:    {len(kept)}")
    print(f"dropped: {len(dropped)}")
    if dropped:
        print("\nDropped (sample of 15):")
        for r in dropped[:15]:
            print(f"  {r['title'][:50]:50s} ← {r['dup_reason']}")
    print(f"\nwrote -> {out_path}")


if __name__ == "__main__":
    main()
