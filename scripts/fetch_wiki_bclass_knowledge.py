"""Fetch B-class Wikipedia articles from knowledge-domain WikiProjects.

Approach:
  1. Enumerate all ~3k sub-categories of Category:B-Class_articles
  2. Filter names client-side by knowledge-domain keywords
  3. For each surviving sub-category, list its Talk-namespace members
     and strip 'Talk:' prefix → article titles
  4. Union + dedup against an existing titles file

Output: additional B-class titles NOT in the input file.

Usage:
    uv run python scripts/fetch_wiki_bclass_knowledge.py \\
        --existing /mnt/data2/wiki_gaps/wiki_quality_titles.txt \\
        --out /mnt/data2/wiki_gaps/bclass_knowledge_titles.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
UA = "espllm-pretrain-data-gathering/1.0 (jens.jepsen@gmail.com)"


# Positive keywords: category name must contain at least one of these
KNOWLEDGE_KEYWORDS = re.compile(
    r"\b("
    # Natural sciences
    r"physics|chemistry|biology|biochemistry|molecular biology|genetics|"
    r"astronomy|cosmology|astrophysics|"
    r"geology|geoscience|earth science|mineralogy|paleontology|palaeontology|"
    r"meteorology|oceanography|climatology|hydrology|volcanology|"
    r"botany|zoology|ecology|entomology|ornithology|mycology|"
    r"microbiology|virology|bacteriology|"
    # Formal / applied science
    r"mathematics|statistics|probability|logic|combinatorics|topology|"
    r"algebra|number theory|geometry|calculus|analysis|"
    r"computer science|computing|programming|software|cryptography|"
    r"engineering|electronics|robotics|electrical|mechanical|civil|chemical|aerospace|"
    r"physics of|"
    # Medicine and health
    r"medicine|medical|anatomy|physiology|neuroscience|neurology|"
    r"pharmacology|pharmacy|immunology|cardiology|oncology|psychiatry|"
    r"public health|epidemiology|nutrition|surgery|dentistry|nursing|"
    r"psychology|"
    # Humanities
    r"history|prehistory|archaeology|archeology|ancient|medieval|byzantine|"
    r"military history|renaissance|reformation|"
    r"philosophy|ethics|epistemology|metaphysics|logic|"
    r"religion|theology|christianity|islam|buddhism|hinduism|judaism|"
    r"linguistics|philology|"
    r"classical|classics|literature|"
    # Social sciences
    r"economics|finance|econometrics|"
    r"sociology|anthropology|ethnography|"
    r"political science|international relations|political philosophy|"
    r"geography|cartography|demographics|"
    # Cross-cutting knowledge
    r"encyclopedia|reference|academic|scholarship|scientific|"
    r"education|pedagogy|library|"
    # Specific but knowledge-heavy topics
    r"aviation|maritime|transport"
    r")\b",
    re.IGNORECASE
)

# Negative keywords: reject if the name contains any of these
POP_CULTURE = re.compile(
    r"\b("
    r"video games?|film|films|movie|movies|television|tv series|tv show|"
    r"actor|actress|celebrity|celebrities|"
    r"albums?|songs?|music|band|singer|discography|"
    r"football|soccer|baseball|basketball|hockey|cricket|rugby|golf|tennis|"
    r"wrestling|boxing|racing|nascar|"
    r"anime|manga|comic|comics|cartoon|"
    r"pornography|porn"
    r")\b",
    re.IGNORECASE
)


def api_get(params: dict, retries: int = 3) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 ** attempt)
    return {}


def enumerate_subcategories() -> list[str]:
    """List all sub-categories of Category:B-Class_articles."""
    cats: list[str] = []
    cont = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:B-Class articles",
            "cmnamespace": "14",  # Category namespace
            "cmlimit": "500",
        }
        if cont:
            params["cmcontinue"] = cont
        data = api_get(params)
        for m in data.get("query", {}).get("categorymembers", []):
            cats.append(m["title"])
        cont_d = data.get("continue", {})
        cont = cont_d.get("cmcontinue")
        if not cont:
            break
        time.sleep(0.05)
    return cats


def is_knowledge(cat_name: str) -> bool:
    """Filter sub-cat name for knowledge relevance."""
    if POP_CULTURE.search(cat_name):
        return False
    return bool(KNOWLEDGE_KEYWORDS.search(cat_name))


def fetch_members(cat: str) -> list[str]:
    """List article titles (from Talk pages) in a B-class category."""
    titles = []
    cont = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cat,
            "cmnamespace": "1",  # Talk namespace
            "cmlimit": "500",
        }
        if cont:
            params["cmcontinue"] = cont
        data = api_get(params)
        for m in data.get("query", {}).get("categorymembers", []):
            t = m["title"]
            if t.startswith("Talk:"):
                titles.append(t[5:])
        cont_d = data.get("continue", {})
        cont = cont_d.get("cmcontinue")
        if not cont:
            break
        time.sleep(0.05)
    return titles


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--existing", type=Path,
                    default=Path("/mnt/data2/wiki_gaps/wiki_quality_titles.txt"))
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/data2/wiki_gaps/bclass_knowledge_titles.txt"))
    ap.add_argument("--merged-out", type=Path,
                    default=Path("/mnt/data2/wiki_gaps/wiki_all_titles.txt"))
    ap.add_argument("--subcat-list-out", type=Path,
                    default=Path("/mnt/data2/wiki_gaps/bclass_subcats_matched.txt"))
    args = ap.parse_args()

    existing = {l.strip() for l in args.existing.read_text().splitlines() if l.strip()}
    print(f"existing titles: {len(existing):,}", flush=True)

    print("\nenumerating B-class sub-categories...", flush=True)
    all_subcats = enumerate_subcategories()
    print(f"  found {len(all_subcats):,} sub-categories", flush=True)

    knowledge_subcats = [c for c in all_subcats if is_knowledge(c)]
    print(f"  knowledge-matched: {len(knowledge_subcats):,}", flush=True)
    args.subcat_list_out.write_text("\n".join(knowledge_subcats) + "\n")

    print("\nfetching members...", flush=True)
    all_titles: set[str] = set()
    for i, cat in enumerate(knowledge_subcats):
        try:
            members = fetch_members(cat)
        except Exception as e:
            print(f"  [{i+1}/{len(knowledge_subcats)}] ERR {cat}: {e}",
                  file=sys.stderr, flush=True)
            continue
        all_titles.update(members)
        if (i + 1) % 20 == 0 or i == len(knowledge_subcats) - 1:
            print(f"  [{i+1}/{len(knowledge_subcats)}] {cat[:70]}: "
                  f"{len(members):>6}  total unique: {len(all_titles):,}",
                  flush=True)

    new_titles = all_titles - existing
    print(f"\ntotal unique B-class knowledge titles: {len(all_titles):,}")
    print(f"  new (not in existing): {len(new_titles):,}")
    print(f"  already in existing:   {len(all_titles) - len(new_titles):,}")

    args.out.write_text("\n".join(sorted(new_titles)) + "\n")
    merged = sorted(existing | new_titles)
    args.merged_out.write_text("\n".join(merged) + "\n")
    print(f"\nwrote new titles → {args.out}")
    print(f"wrote merged list ({len(merged):,} titles) → {args.merged_out}")


if __name__ == "__main__":
    main()
