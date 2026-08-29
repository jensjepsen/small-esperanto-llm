"""Filter Gutenberg catalog for science/philosophy works (English plaintext).

Reads /mnt/data2/sci_books/gutenberg_catalog.csv, filters by:
  - Language == en
  - LoCC code in target domain prefixes (Q*=Science, R*=Medicine, T*=Tech,
    B*=Philosophy, BF=Psychology, G*=Geography/Anthro, H*=Social Sciences)
  - Type == Text (skip Audio, Compilations, etc.)

Outputs:
  - /mnt/data2/sci_books/sci_candidates.jsonl     (one row per book with metadata)
  - prints a summary by top-level LoCC code

We deliberately don't download anything yet — this is just selection.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

CATALOG = Path("/mnt/data2/sci_books/gutenberg_catalog.csv")
OUT = Path("/mnt/data2/sci_books/sci_candidates.jsonl")

# LoCC top-level codes we consider in-domain.
# Reference: https://www.loc.gov/catdir/cpso/lcco/
DOMAIN_PREFIXES = {
    # Q = Science (math, astronomy, physics, chemistry, geology, biology...)
    "Q":  "science (general)",
    "QA": "mathematics",
    "QB": "astronomy",
    "QC": "physics",
    "QD": "chemistry",
    "QE": "geology",
    "QH": "natural history / biology",
    "QK": "botany",
    "QL": "zoology",
    "QM": "human anatomy",
    "QP": "physiology",
    "QR": "microbiology",
    # R = Medicine
    "R":  "medicine (general)",
    "RA": "public health",
    "RB": "pathology",
    "RC": "internal medicine",
    "RD": "surgery",
    # T = Technology (selected — avoid pure how-to manuals)
    "T":  "technology (general)",
    "TA": "civil engineering",
    "TJ": "mechanical engineering",
    "TK": "electrical engineering",
    "TL": "motor vehicles / aeronautics",
    "TP": "chemical technology",
    # B = Philosophy & Psychology
    "B":  "philosophy (general)",
    "BC": "logic",
    "BD": "speculative philosophy",
    "BF": "psychology",
    "BH": "aesthetics",
    "BJ": "ethics",
    # G = Geography & Anthropology
    "GE": "environmental sciences",
    "GF": "human ecology / anthropogeography",
    "GN": "anthropology",
    "GR": "folklore",
    # H = Social Sciences (selected)
    "HM": "sociology (general)",
    "HN": "social history",
    "HQ": "family, marriage, sex",
    # Education (philosophical aspects)
    "LB": "theory & practice of education",
}


def in_domain(locc: str) -> list[str]:
    """Return list of matching prefixes for a semicolon-separated LoCC field."""
    if not locc:
        return []
    matches = []
    for code in (c.strip().upper() for c in locc.split(";")):
        if not code:
            continue
        # Try exact 2-char prefix first, then 1-char
        if code[:2] in DOMAIN_PREFIXES:
            matches.append(code[:2])
        elif code[:1] in DOMAIN_PREFIXES:
            matches.append(code[:1])
    return matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap output (0 = all)")
    ap.add_argument("--year-min", type=int, default=0,
                    help="drop books issued before this year (0 = no min)")
    ap.add_argument("--year-max", type=int, default=0,
                    help="drop books issued after this year (0 = no max)")
    ap.add_argument("--loccs", default="",
                    help="comma-separated whitelist of LoCC prefixes; "
                         "empty = include all DOMAIN_PREFIXES")
    ap.add_argument("--out", default=str(OUT),
                    help="output JSONL path")
    args = ap.parse_args()
    allowed_loccs = set(args.loccs.split(",")) if args.loccs else set(DOMAIN_PREFIXES)

    print(f"reading {CATALOG}...", flush=True)
    rows = list(csv.DictReader(CATALOG.open()))
    print(f"  {len(rows):,} total entries")

    en_text = [r for r in rows
               if r.get("Language", "").lower() == "en"
               and r.get("Type", "") == "Text"]
    print(f"  {len(en_text):,} English texts")

    in_scope = []
    by_domain = defaultdict(int)
    for r in en_text:
        matches = in_domain(r.get("LoCC", ""))
        # Restrict to allowed LoCC prefixes
        matches = [m for m in matches if m in allowed_loccs]
        if not matches:
            continue
        # Year filter (issued field is "YYYY-MM-DD")
        if args.year_min or args.year_max:
            issued = r.get("Issued", "")
            try:
                year = int(issued[:4])
            except (ValueError, IndexError):
                continue
            if args.year_min and year < args.year_min:
                continue
            if args.year_max and year > args.year_max:
                continue
        in_scope.append({
            "gutenberg_id": int(r["Text#"]),
            "title": r["Title"],
            "authors": r.get("Authors", ""),
            "issued": r.get("Issued", ""),
            "subjects": r.get("Subjects", ""),
            "locc": r.get("LoCC", ""),
            "bookshelves": r.get("Bookshelves", ""),
            "domain": matches[0],
        })
        for m in matches:
            by_domain[m] += 1

    print(f"  {len(in_scope):,} in-domain (LoCC matches science/phil/med/tech/social)")
    print()
    print("breakdown by LoCC prefix:")
    for prefix in sorted(DOMAIN_PREFIXES.keys()):
        count = by_domain.get(prefix, 0)
        if count > 0:
            print(f"  {prefix:>3}: {count:>5,}  {DOMAIN_PREFIXES[prefix]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.limit:
        in_scope = in_scope[: args.limit]
    with out_path.open("w") as f:
        for rec in in_scope:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(in_scope):,} candidates -> {out_path}")


if __name__ == "__main__":
    main()
