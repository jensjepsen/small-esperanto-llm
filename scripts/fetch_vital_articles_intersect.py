"""Intersect Wikipedia's curated "Vital Articles" list with our EN-only gap set.

The pageview-ranked en_only_candidates.jsonl surfaces pop culture (TV
shows, athletes, regional celebrities) because that's what people read.
Vital Articles is editor-curated: "what every encyclopedia should
cover" — math, science, history, philosophy, geography, foundational
concepts. Intersection gives us foundational EN-only articles.

Two phases:
  fetch-vital:  crawl all Wikipedia:Vital_articles/Level/{N}/* subpages
                via MediaWiki API, extract mainspace titles → TSV file
  intersect:    filter en_only_candidates.jsonl to titles in the vital
                set; emit reduced JSONL ranked by views (or length)

Usage:
    uv run python scripts/fetch_vital_articles_intersect.py fetch-vital --level 4
    uv run python scripts/fetch_vital_articles_intersect.py intersect \\
        --vital /mnt/data2/wiki_gaps/vital_articles_level4.txt \\
        --out /mnt/data2/wiki_gaps/en_only_vital_candidates.jsonl
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "espllm-pretrain-data-gathering/1.0 (jens.jepsen@gmail.com)"
DATA_DIR = Path("/mnt/data2/wiki_gaps")
DEFAULT_INPUT = DATA_DIR / "en_only_candidates.jsonl"


def api_get(params: dict) -> dict:
    """Polite GET to MediaWiki API."""
    params = {**params, "format": "json"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def list_subpages(prefix: str) -> list[str]:
    """All pages in Wikipedia: namespace (4) under <prefix>."""
    pages = []
    apcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "apnamespace": "4",  # Wikipedia: namespace
            "apprefix": prefix,
            "aplimit": "500",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue
        data = api_get(params)
        for p in data.get("query", {}).get("allpages", []):
            pages.append(p["title"])
        cont = data.get("continue", {})
        apcontinue = cont.get("apcontinue")
        if not apcontinue:
            break
        time.sleep(0.2)
    return pages


def fetch_mainspace_links(page: str) -> set[str]:
    """All mainspace (namespace 0) links from a single page."""
    links = set()
    plcontinue = None
    while True:
        params = {
            "action": "query",
            "titles": page,
            "prop": "links",
            "plnamespace": "0",
            "pllimit": "500",
        }
        if plcontinue:
            params["plcontinue"] = plcontinue
        data = api_get(params)
        for pdata in data.get("query", {}).get("pages", {}).values():
            for link in pdata.get("links", []):
                links.add(link["title"])
        cont = data.get("continue", {})
        plcontinue = cont.get("plcontinue")
        if not plcontinue:
            break
        time.sleep(0.2)
    return links


def cmd_fetch_vital(args):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Wikipedia reorganized: "Level 4" (with space) is the current path segment,
    # not "Level/4" (which now just redirects).
    prefix = f"Vital articles/Level {args.level}/"
    print(f"listing subpages of Wikipedia:{prefix}...", flush=True)
    subpages = list_subpages(prefix)
    print(f"  {len(subpages)} subpages found", flush=True)

    # Also include the top-level page itself
    subpages.insert(0, f"Wikipedia:Vital articles/Level {args.level}")

    all_titles: set[str] = set()
    for i, page in enumerate(subpages, 1):
        before = len(all_titles)
        try:
            links = fetch_mainspace_links(page)
        except Exception as e:
            print(f"  [{i:3d}/{len(subpages)}] ERROR {page}: {e}", flush=True)
            continue
        all_titles.update(links)
        added = len(all_titles) - before
        print(f"  [{i:3d}/{len(subpages)}] {page}: +{added} (total {len(all_titles)})",
              flush=True)

    out_path = DATA_DIR / f"vital_articles_level{args.level}.txt"
    with out_path.open("w") as f:
        for t in sorted(all_titles):
            f.write(t + "\n")
    print(f"\nwrote {len(all_titles):,} titles -> {out_path}", flush=True)


def cmd_intersect(args):
    vital_path = Path(args.vital)
    if not vital_path.exists():
        raise SystemExit(f"ERROR: {vital_path} missing — run fetch-vital first")
    with vital_path.open() as f:
        vital = set(line.strip() for line in f if line.strip())
    print(f"loaded {len(vital):,} vital titles from {vital_path}", flush=True)

    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_seen = n_kept = 0
    kept: list[dict] = []
    with input_path.open() as f:
        for line in f:
            n_seen += 1
            row = json.loads(line)
            if row["title"] in vital:
                kept.append(row)
                n_kept += 1

    if args.rank_by == "views":
        kept.sort(key=lambda r: -r.get("views", 0))
    elif args.rank_by == "length":
        kept.sort(key=lambda r: -r["length"])

    with out_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_chars = sum(r["length"] for r in kept)
    print(f"\nscanned {n_seen:,}; matched vital: {n_kept:,}", flush=True)
    print(f"wrote {len(kept):,} rows ({total_chars/1e6:.1f}M chars, "
          f"~{total_chars/4/1e6:.1f}M tokens) -> {out_path}", flush=True)

    if kept:
        print(f"\ntop 10 by {args.rank_by}:", flush=True)
        for r in kept[:10]:
            print(f"  views={r.get('views',0):>10,}  len={r['length']:>6,}  {r['title']}")


def cmd_vital_direct(args):
    """For each vital title: API-check EO sitelink + fetch full text.
    Bypasses the pageview-ranked 50k entirely — gets foundational EN-only
    articles directly via Wikipedia API. ~11k titles in ~5-10 min."""
    vital_path = Path(args.vital)
    if not vital_path.exists():
        raise SystemExit(f"ERROR: {vital_path} missing — run fetch-vital first")
    with vital_path.open() as f:
        titles = [t.strip() for t in f if t.strip()]
    print(f"loaded {len(titles):,} vital titles", flush=True)

    # Optional: load pageviews if present, for annotating output
    views_path = DATA_DIR / "en_pageviews.tsv"
    views_map: dict[str, int] = {}
    if views_path.exists():
        print(f"loading pageviews from {views_path}...", flush=True)
        with views_path.open() as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    views_map[parts[0]] = int(parts[1])
        print(f"  {len(views_map):,} titles with views", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_path.open("w")
    n_emitted = n_has_eo = n_short = 0

    BATCH = 20  # API supports 50 but extracts+langlinks together is heavy
    for i in range(0, len(titles), BATCH):
        batch = titles[i : i + BATCH]
        # One call: titles → langlinks=eo + extract (plain text intro+body)
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "langlinks|extracts",
            "lllang": "eo",
            "lllimit": "max",
            "explaintext": "1",
            "exsectionformat": "plain",
            "exlimit": "max",
            "redirects": "1",
        }
        try:
            data = api_get(params)
        except Exception as e:
            print(f"  batch {i//BATCH}: ERROR {e}", flush=True)
            time.sleep(1.0)
            continue

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "missing" in page:
                continue
            title = page.get("title")
            has_eo = bool(page.get("langlinks"))
            text = page.get("extract", "")
            if has_eo:
                n_has_eo += 1
                continue
            if len(text) < args.min_length:
                n_short += 1
                continue
            rec = {
                "page_id": page.get("pageid"),
                "title": title,
                "length": len(text),
                "text": text,
                "views": views_map.get(title.replace(" ", "_"), 0),
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_emitted += 1

        if (i // BATCH) % 50 == 0:
            print(f"  [{i+BATCH:>5}/{len(titles)}]  emitted={n_emitted}  "
                  f"has-eo={n_has_eo}  too-short={n_short}", flush=True)
        time.sleep(0.2)

    out_f.close()
    print(f"\ndone. {n_emitted:,} vital EN-only articles written -> {out_path}",
          flush=True)
    print(f"  had EO sitelink (skipped): {n_has_eo:,}", flush=True)
    print(f"  too short (skipped): {n_short:,}", flush=True)


def main():
    ap = argparse.ArgumentParser(__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("fetch-vital",
                        help="scrape Wikipedia:Vital_articles/Level/N/* sublists")
    p1.add_argument("--level", type=int, default=4, choices=[3, 4, 5],
                    help="Vital articles tier (3=1k essential, 4=10k, 5=50k)")
    p1.set_defaults(func=cmd_fetch_vital)

    p2 = sub.add_parser("intersect",
                        help="filter en_only_candidates.jsonl to vital titles "
                             "(legacy: limited by pageview-ranked top-50k)")
    p2.add_argument("--vital", required=True)
    p2.add_argument("--input", default=str(DEFAULT_INPUT))
    p2.add_argument("--out", required=True)
    p2.add_argument("--rank-by", choices=["views", "length", "stream"],
                    default="views")
    p2.set_defaults(func=cmd_intersect)

    p3 = sub.add_parser("vital-direct",
                        help="for each vital title, API-fetch text + check EO sitelink")
    p3.add_argument("--vital", required=True)
    p3.add_argument("--out", required=True)
    p3.add_argument("--min-length", type=int, default=2000,
                    help="drop articles whose extract is shorter than this")
    p3.set_defaults(func=cmd_vital_direct)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
