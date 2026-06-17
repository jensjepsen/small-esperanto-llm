"""Post-filter mc4_eo.jsonl for actual Esperanto content.

The Verifier in clean_mc4_eo.py checks Esperanto grammar but not language ID.
Foreign text with a few Esperanto-suffix-shaped tokens slips through (~50%
of mc4_eo.jsonl is Spanish/French/English/multilingual junk per audit).

This script reads the existing mc4_eo.jsonl and rejects docs whose
"Esperanto-ness" score falls below --threshold. Three signals combined:
  1. Esperanto function-word density (high-confidence words like estas, kaj, ĉi)
  2. Verb/correlative ending density (-as, -is, -os, -us, -aŭ on words ≥4 chars)
  3. Esperanto-specific accented-character density (ĉĝĥĵŝŭ)

Tokens count toward "Esperanto" if any of the three signals fire.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Function words that are highly Esperanto-specific (rare in
# Spanish/French/Italian/English even where short forms collide).
EO_FUNCTION_WORDS = frozenset({
    # Copulas (the -as/-is/-os/-us/-i forms of esti are fingerprints)
    "estas", "estis", "estos", "estus", "esti", "estu",
    "havas", "havis", "havos", "havus", "havi",
    "povas", "povis", "povos", "povus", "povi",
    "devas", "devis", "devos", "devi",
    "faras", "faris", "faros", "fari",
    "iras", "iris", "iros", "iri",
    # Conjunctions
    "kaj", "sed", "aŭ", "nek", "ĉar", "ke", "ol", "dum", "kvankam",
    # Esperanto-specific pronouns / determiners
    "ĉi", "ĉu", "ĝi", "ŝi", "li", "ili", "oni",
    "ĝia", "ŝia", "lia", "ilia", "nia", "via", "mia", "sia",
    "ĝin", "ŝin", "lin", "ilin", "nin", "vin", "min", "sin",
    # Correlatives
    "kio", "kiu", "kie", "kiam", "kiel", "kial", "kies", "kiom",
    "tio", "tiu", "tie", "tiam", "tiel", "tial", "ties", "tiom",
    "io", "iu", "ie", "iam", "iel", "ies", "iom",
    "ĉio", "ĉiu", "ĉie", "ĉiam", "ĉiel", "ĉies", "ĉiom",
    "nenio", "neniu", "nenie", "neniam", "neniel", "nenies", "neniom",
    "kion", "tion", "ion", "ĉion", "nenion",
    "kiun", "tiun", "iun", "ĉiun", "neniun",
    # Adverbs / particles
    "ne", "jes", "ja", "do", "ankaŭ", "nur", "tre", "jam", "plu",
    "almenaŭ", "tamen", "ankoraŭ", "preskaŭ", "baldaŭ", "hodiaŭ",
    "morgaŭ", "hieraŭ",
})

# Accented chars that are nearly unique to Esperanto in modern UTF-8 text.
EO_ACCENTED = set("ĉĝĥĵŝŭĈĜĤĴŜŬ")

# -aŭ ending captures the adverbs (almenaŭ, hodiaŭ, antaŭ, etc.). The verb
# endings -as/-is/-os/-us require length ≥4 to avoid matching English "as",
# "is", etc. (in EO, "as"/"is" alone aren't words). The participle endings
# are fingerprints when ≥6 chars (-anta, -inta, -onta, -ata, -ita, -ota).
ENDING_RE = re.compile(
    r"^[a-zĉĝĥĵŝŭ']{2,}"
    r"(?:as|is|os|us|aŭ|"
    r"anta|antaj|antan|antajn|"
    r"inta|intaj|intan|intajn|"
    r"onta|ontaj|ontan|ontajn|"
    r"ata|ataj|atan|atajn|"
    r"ita|itaj|itan|itajn|"
    r"ota|otaj|otan|otajn"
    r")$",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ']+")


def eo_score(text: str) -> tuple[float, dict]:
    """Return (score, breakdown). Score = fraction of tokens that look Esperanto."""
    tokens = TOKEN_RE.findall(text)
    n_tokens = len(tokens)
    if n_tokens == 0:
        return 0.0, {"tokens": 0, "func": 0, "ending": 0, "accent_tok": 0}

    # Per-token signals (a token counts as "Esperanto-looking" if ANY fires).
    func_hits = 0
    ending_hits = 0
    accent_hits = 0
    eo_tokens = 0
    for tok in tokens:
        low = tok.lower()
        f = low in EO_FUNCTION_WORDS
        # length filter on -as/-is/-os/-us so 2-letter English words don't match.
        # The regex itself enforces ≥2 stem chars before ending; we still need
        # the whole-token length floor to dodge "is", "as", "us" themselves.
        e = bool(ENDING_RE.match(low)) and len(low) >= 4
        a = any(c in EO_ACCENTED for c in tok)
        if f:
            func_hits += 1
        if e:
            ending_hits += 1
        if a:
            accent_hits += 1
        if f or e or a:
            eo_tokens += 1

    score = eo_tokens / n_tokens
    return score, {
        "tokens": n_tokens,
        "func": func_hits,
        "ending": ending_hits,
        "accent_tok": accent_hits,
        "eo": eo_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file", type=Path,
                    default=Path("/mnt/data/espllm/data/mc4_filtered/mc4_eo.jsonl"))
    ap.add_argument("--out-file", type=Path,
                    default=Path("/mnt/data/espllm/data/mc4_filtered/mc4_eo_lang_filtered.jsonl"))
    ap.add_argument("--threshold", type=float, default=0.20,
                    help="Min fraction of tokens that must look Esperanto (default 0.20)")
    ap.add_argument("--show-rejects", type=int, default=8,
                    help="Print N borderline rejections for inspection")
    ap.add_argument("--show-keeps", type=int, default=4,
                    help="Print N borderline keeps (just above threshold)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Score everything, print stats + samples, don't write")
    args = ap.parse_args()

    if not args.in_file.exists():
        sys.exit(f"Input not found: {args.in_file}")

    n_in = 0
    n_kept = 0
    score_buckets = Counter()
    borderline_rejects: list[tuple[float, str]] = []
    borderline_keeps: list[tuple[float, str]] = []
    obvious_rejects: list[tuple[float, str]] = []

    fout = None if args.dry_run else open(args.out_file, "w")

    try:
        with open(args.in_file) as fin:
            for line in fin:
                n_in += 1
                d = json.loads(line)
                text = d.get("text", "")
                score, _ = eo_score(text)

                # bucket scores in 0.05 bins for histogram
                bucket = round(score * 20) / 20
                score_buckets[bucket] += 1

                if score >= args.threshold:
                    n_kept += 1
                    if fout is not None:
                        fout.write(line)
                    # Collect a few low-margin keeps for inspection.
                    if (score < args.threshold + 0.05
                            and len(borderline_keeps) < args.show_keeps):
                        borderline_keeps.append((score, text))
                else:
                    # Just-below-threshold rejects are the most informative.
                    if (score >= args.threshold - 0.05
                            and len(borderline_rejects) < args.show_rejects):
                        borderline_rejects.append((score, text))
                    elif (score < 0.05
                            and len(obvious_rejects) < 3):
                        obvious_rejects.append((score, text))
    finally:
        if fout is not None:
            fout.close()

    # ---- Report ----
    print(f"Input docs:    {n_in:,}")
    print(f"Kept:          {n_kept:,}  ({100*n_kept/max(n_in,1):.1f}%)")
    print(f"Rejected:      {n_in - n_kept:,}")
    print(f"Threshold:     {args.threshold}")
    if not args.dry_run:
        print(f"Wrote → {args.out_file}")

    print("\nScore histogram (0.05 buckets):")
    for b in sorted(score_buckets):
        n = score_buckets[b]
        bar = "█" * min(60, int(60 * n / max(score_buckets.values())))
        print(f"  {b:.2f}  {n:>7,}  {bar}")

    def show(label, samples):
        if not samples: return
        print(f"\n=== {label} ({len(samples)}) ===")
        for sc, txt in samples:
            print(f"\n[score={sc:.3f}]  {txt[:300]}{'…' if len(txt) > 300 else ''}")

    show("Borderline REJECTS (just below threshold — would now lose)",
         borderline_rejects)
    show("Borderline KEEPS (just above threshold — barely surviving)",
         borderline_keeps)
    show("Obvious REJECTS (score < 0.05)", obvious_rejects)


if __name__ == "__main__":
    main()
