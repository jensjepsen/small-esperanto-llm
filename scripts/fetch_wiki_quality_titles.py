"""Fetch Featured Articles + Good Articles lists from English Wikipedia.

Uses the MediaWiki `categorymembers` API to enumerate:
  - Category:Featured articles (~6.5k)
  - Category:Good articles (~42.9k)

Category members are `Talk:` pages (categories are placed on talk pages);
strip the `Talk:` prefix to get the mainspace article title.

Union with an existing titles file (e.g. Vital L5) and dedup.

Output: expanded titles list, one per line.

Usage:
    uv run python scripts/fetch_wiki_quality_titles.py \\
        --l5 /mnt/data2/wiki_gaps/vital_articles_level5.txt \\
        --out /mnt/data2/wiki_gaps/wiki_quality_titles.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
UA = "espllm-pretrain-data-gathering/1.0 (jens.jepsen@gmail.com)"


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


def category_members(category: str) -> list[str]:
    """List all page titles in a category (paginated). Strips Talk: prefix."""
    titles = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "500",
            "cmnamespace": "0",  # Mainspace articles
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(params)
        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            titles.append(m["title"])
        cont = data.get("continue", {})
        cmcontinue = cont.get("cmcontinue")
        if not cmcontinue:
            break
        if len(titles) % 5000 < 500 and len(titles) > 0:
            print(f"  ...{len(titles):,} fetched", flush=True)
        time.sleep(0.05)
    return titles


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--l5", type=Path,
                    default=Path("/mnt/data2/wiki_gaps/vital_articles_level5.txt"))
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/data2/wiki_gaps/wiki_quality_titles.txt"))
    ap.add_argument("--breakdown-out", type=Path,
                    default=Path("/mnt/data2/wiki_gaps/wiki_quality_source.jsonl"),
                    help="Per-title source annotations")
    args = ap.parse_args()

    l5_titles = [l.strip() for l in args.l5.read_text().splitlines() if l.strip()]
    print(f"L5 titles: {len(l5_titles):,}", flush=True)

    print("fetching Featured Articles (~6.5k)...", flush=True)
    featured = category_members("Category:Featured articles")
    print(f"  {len(featured):,} featured", flush=True)

    print("fetching Good Articles (~42.9k)...", flush=True)
    good = category_members("Category:Good articles")
    print(f"  {len(good):,} good", flush=True)

    # Union and dedup
    all_titles: dict[str, list[str]] = {}
    for t in l5_titles:
        all_titles.setdefault(t, []).append("L5")
    for t in featured:
        all_titles.setdefault(t, []).append("FA")
    for t in good:
        all_titles.setdefault(t, []).append("GA")

    print(f"\ntotal unique: {len(all_titles):,}", flush=True)
    print(f"  L5 only:            {sum(1 for s in all_titles.values() if s == ['L5']):,}")
    print(f"  FA only:            {sum(1 for s in all_titles.values() if s == ['FA']):,}")
    print(f"  GA only:            {sum(1 for s in all_titles.values() if s == ['GA']):,}")
    print(f"  L5 ∩ FA:            {sum(1 for s in all_titles.values() if 'L5' in s and 'FA' in s):,}")
    print(f"  L5 ∩ GA:            {sum(1 for s in all_titles.values() if 'L5' in s and 'GA' in s):,}")
    print(f"  FA ∩ GA:            {sum(1 for s in all_titles.values() if 'FA' in s and 'GA' in s):,}")

    args.out.write_text("\n".join(sorted(all_titles.keys())) + "\n")
    with args.breakdown_out.open("w") as f:
        for t, srcs in sorted(all_titles.items()):
            f.write(json.dumps({"title": t, "sources": srcs},
                               ensure_ascii=False) + "\n")

    print(f"\nwrote titles → {args.out}")
    print(f"wrote sources → {args.breakdown_out}")


if __name__ == "__main__":
    main()
