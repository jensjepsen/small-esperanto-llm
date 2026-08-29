"""Fetch lead paragraphs for all Wikipedia Vital Articles Level 5 titles.

Reads /mnt/data2/wiki_gaps/vital_articles_level5.txt (~51k titles) and
uses the MediaWiki extracts API (`exintro=1&explaintext=1`) to fetch the
plain-text lead section for each. Batches 20 titles per request.

Output JSONL rows:
    {"title": str, "pageid": int, "lead": str}

Resume-safe: skips titles already present in the output file.

Usage:
    uv run python scripts/fetch_vital_l5_leads.py \\
        --titles /mnt/data2/wiki_gaps/vital_articles_level5.txt \\
        --out /mnt/data2/wiki_gaps/vital_level5_leads.jsonl
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
USER_AGENT = "espllm-pretrain-data-gathering/1.0 (jens.jepsen@gmail.com)"


def api_get(params: dict, retries: int = 3) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 ** attempt)
    return {}


def fetch_batch(titles: list[str]) -> list[dict]:
    """Return list of {title, pageid, lead} for titles that resolved."""
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "redirects": 1,
        "titles": "|".join(titles),
    }
    data = api_get(params)
    out = []
    # Map normalized titles back to originals
    original_by_normalized = {t: t for t in titles}
    for n in data.get("query", {}).get("normalized", []) or []:
        original_by_normalized[n["to"]] = n["from"]
    for r in data.get("query", {}).get("redirects", []) or []:
        # If redirect target found in later pages, credit original request title
        orig = original_by_normalized.get(r["from"], r["from"])
        original_by_normalized[r["to"]] = orig

    for page in data.get("query", {}).get("pages", []):
        if page.get("missing"):
            continue
        title = page.get("title", "")
        # Find which requested title got us here
        requested = original_by_normalized.get(title, title)
        lead = page.get("extract", "").strip()
        if not lead:
            continue
        out.append({
            "title": requested,
            "resolved_title": title,
            "pageid": page.get("pageid"),
            "lead": lead,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--titles", type=Path,
                    default=Path("/mnt/data2/wiki_gaps/vital_articles_level5.txt"))
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/data2/wiki_gaps/vital_level5_leads.jsonl"))
    ap.add_argument("--batch", type=int, default=20,
                    help="Titles per API request (max 50 for extracts API)")
    ap.add_argument("--pause", type=float, default=0.1,
                    help="Sleep seconds between requests")
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()

    titles = [l.strip() for l in args.titles.read_text().splitlines() if l.strip()]
    print(f"total titles: {len(titles):,}", flush=True)

    done_titles: set[str] = set()
    if args.out.exists():
        with args.out.open() as f:
            for line in f:
                try:
                    done_titles.add(json.loads(line)["title"])
                except Exception:
                    pass
        print(f"resume: {len(done_titles):,} already fetched", flush=True)

    todo = [t for t in titles if t not in done_titles]
    print(f"todo: {len(todo):,}\n", flush=True)
    if not todo:
        return

    t0 = time.time()
    n_ok = 0
    n_missing = 0
    with args.out.open("a") as fout:
        for start in range(0, len(todo), args.batch):
            batch = todo[start:start + args.batch]
            try:
                results = fetch_batch(batch)
            except Exception as e:
                print(f"[error batch {start}] {e}", file=sys.stderr, flush=True)
                results = []
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            n_ok += len(results)
            n_missing += len(batch) - len(results)
            if args.pause > 0:
                time.sleep(args.pause)
            processed = start + len(batch)
            if processed % args.log_every == 0 or processed >= len(todo):
                el = time.time() - t0
                rate = processed / el
                eta = (len(todo) - processed) / rate if rate else 0
                print(f"  {processed:,}/{len(todo):,}  ok={n_ok:,}  "
                      f"missing={n_missing:,} ({100*n_missing/processed:.1f}%)  "
                      f"rate={rate:.1f}/s  eta={eta/60:.1f}min", flush=True)

    print(f"\ndone: {n_ok:,} leads fetched, {n_missing:,} missing in "
          f"{(time.time()-t0)/60:.1f} min → {args.out}")


if __name__ == "__main__":
    main()
