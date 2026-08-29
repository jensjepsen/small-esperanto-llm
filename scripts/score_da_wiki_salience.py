"""Score Danish Wikipedia articles by general-knowledge salience.

Combines three signals from the wiki SQL dumps:
  1. page_len       — article length in bytes (from page.sql.gz)
  2. langlinks      — # of other-language editions with this article
                       (from langlinks.sql.gz)
  3. category-based negative filter — kill stub-heavy trees
                       (villages, sports match reports, taxonomy)

(Pageviews would be a stronger signal but require a separate dump fetch;
 not included in this first cut. Length × langlinks alone reproduces most
 of the salience ranking in practice.)

Output: TSV with pageid, title, page_len, langlinks, tier, score
        + summary of tier bucket sizes.

Usage:
    # First-time (downloads ~50MB langlinks dump if missing):
    uv run --no-project python scripts/score_da_wiki_salience.py \\
        --pageids /mnt/data2/da_wiki_curation/pageids.tsv \\
        --page-sql /mnt/data2/da_wiki_dumps/dawiki-latest-page.sql.gz \\
        --dumps-dir /mnt/data2/da_wiki_dumps \\
        --out /mnt/data2/da_wiki_curation/salience.tsv
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys
import time
import urllib.request
from pathlib import Path

DUMPS_URL = "https://dumps.wikimedia.org/dawiki/latest"
UA = "espllm-pretrain-data-gathering/1.0 (jens.jepsen@gmail.com)"


def ensure_dump(name: str, dumps_dir: Path) -> Path:
    p = dumps_dir / name
    if p.exists():
        return p
    url = f"{DUMPS_URL}/{name}"
    print(f"downloading {url} → {p} …", flush=True)
    dumps_dir.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as r, tmp.open("wb") as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.rename(p)
    return p


# page schema (12 cols):
#   page_id, page_namespace, 'page_title', page_is_redirect, page_is_new,
#   page_random, 'page_touched', 'page_links_updated' | NULL, page_latest,
#   page_len, 'page_content_model' | NULL, 'page_lang' | NULL
_PAGE_ROW_RE = re.compile(
    rb"\((\d+),(\d+),"                             # 1:pid  2:ns
    rb"'(?:[^'\\]|\\.)*',"                         # title (skip)
    rb"(\d+),\d+,\d+(?:\.\d+)?,"                   # 3:is_redirect, is_new, random
    rb"'\d+',(?:'\d+'|NULL),\d+,"                  # touched, links_updated, latest
    rb"(\d+),"                                     # 4:page_len
)


def parse_page_lens(path: Path) -> dict[int, int]:
    """Yield {pageid: byte_length} for namespace-0, non-redirect pages."""
    out = {}
    with gzip.open(path, "rb") as f:
        for line in f:
            if not line.startswith(b"INSERT INTO `page` VALUES "):
                continue
            for m in _PAGE_ROW_RE.finditer(line):
                pid, ns, is_redir, plen = (int(m.group(i)) for i in (1, 2, 3, 4))
                if ns != 0 or is_redir:
                    continue
                out[pid] = plen
    return out


# langlinks: (ll_from, ll_lang, ll_title). Count distinct ll_from values.
_LANGLINKS_ROW_RE = re.compile(rb"\((\d+),")


def parse_langlinks_counts(path: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    with gzip.open(path, "rb") as f:
        for line in f:
            if not line.startswith(b"INSERT INTO `langlinks` VALUES "):
                continue
            for m in _LANGLINKS_ROW_RE.finditer(line):
                pid = int(m.group(1))
                counts[pid] = counts.get(pid, 0) + 1
    return counts


# linktarget schema (3 cols):
#   lt_id (bigint), lt_namespace (int), 'lt_title' (varbinary)
_LINKTARGET_ROW_RE = re.compile(rb"\((\d+),(\d+),'((?:[^'\\]|\\.)*)'\)")


def parse_linktarget_by_ns(path: Path, ns_wanted: int) -> dict[int, str]:
    """{lt_id: title} for link targets in a specific namespace.
    ns=0 → mainspace pages, ns=14 → categories."""
    out: dict[int, str] = {}
    with gzip.open(path, "rb") as f:
        for line in f:
            if not line.startswith(b"INSERT INTO `linktarget` VALUES "):
                continue
            for m in _LINKTARGET_ROW_RE.finditer(line):
                lt_id, ns, title = int(m.group(1)), int(m.group(2)), m.group(3)
                if ns != ns_wanted:
                    continue
                out[lt_id] = title.decode("utf-8", errors="replace")
    return out


# Backwards-compat alias
def parse_linktarget_ns0(path: Path) -> dict[int, str]:
    return parse_linktarget_by_ns(path, 0)


# Category patterns that identify stub-heavy categories to demote members of.
# Compiled as unicode-str regex; we decode the raw category bytes before matching
# so non-ASCII characters (æ, ø, å, ñ) work.
_STUB_CATEGORY_PATTERNS = [
    # Olympic events per year
    re.compile(r"sommer-OL_\d{4}", re.I),
    re.compile(r"vinter-OL_\d{4}", re.I),
    re.compile(r"_under_(sommer|vinter)-OL_\d{4}", re.I),
    re.compile(r"ungdoms-OL", re.I),
    re.compile(r"paralympiske_lege", re.I),
    # Cycling races per year
    re.compile(r"tour_de_france_\d{4}", re.I),
    re.compile(r"vuelta_a_españa_\d{4}", re.I),
    re.compile(r"giro_d'italia_\d{4}", re.I),
    # Championships per year (any specific "world championship in <sport>")
    re.compile(r"VM_i_(fodbold|h[åa]ndbold|volleyball|basketball|ishockey|"
               r"badminton|curling|skisport|skiskydning|orientering|roning)", re.I),
    re.compile(r"EM_i_(fodbold|h[åa]ndbold|volleyball|basketball|ishockey|"
               r"badminton|curling|skisport|skiskydning|orientering|roning)", re.I),
    re.compile(r"fodbold-VM_\d{4}", re.I),
    re.compile(r"fodbold-EM_\d{4}", re.I),
    # League seasons
    re.compile(r"-s[æa]son(en)?_\d{4}", re.I),
    re.compile(r"_s[æa]sonen", re.I),
    re.compile(r"superliga(en)?_\d{4}", re.I),
    re.compile(r"Champions_League_\d{4}", re.I),
    re.compile(r"Europa_League_\d{4}", re.I),
    # Sports-athlete-by-nationality (bulk biography stubs)
    re.compile(r"(danske|norske|svenske|engelske|tyske|franske|italienske|"
               r"spanske|amerikanske|brasilianske|argentinske|australske|"
               r"kanadiske|kinesiske|japanske|hollandske)_"
               r"(fodbold|h[åa]ndbold|basketball|ishockey|cykel|golf|tennis|"
               r"volleyball)(spillere|ryttere|klubber)", re.I),
    # F1 seasons
    re.compile(r"Formel_1_i_\d{4}", re.I),
    re.compile(r"_Grand_Prix_\d{4}", re.I),
    # Local elections / municipal detail pages
    re.compile(r"kommunalvalg_\d{4}", re.I),
    # Sport-by-year categories (catches Vuelta 2017 via "Cykelløb i 2017" etc.)
    re.compile(r"(cyklel[øo]b|cyklerace|sport|sportsbegivenheder|sportstur|"
               r"fodbold|tennis|golf|h[åa]ndbold|basketball|ishockey|"
               r"skisport|olympiske_lege)_i_\d{4}", re.I),
    re.compile(r"\d{4}_i_(sport|cykling|fodbold|tennis|golf|h[åa]ndbold|"
               r"basketball|ishockey|skisport)", re.I),
    # Sports tournament yearly editions (Wimbledon 2006, US Open 2008 etc.)
    re.compile(r"(wimbledon|us_open|australian_open|french_open|"
               r"roland_garros|us_masters|the_open)[_-]?\d{4}", re.I),
    # Sports clubs categories (any language)
    re.compile(r"fodboldklub(ber)?(_i_|$)", re.I),
    re.compile(r"h[åa]ndboldklub(ber)?(_i_|$)", re.I),
    re.compile(r"basketballklub(ber)?", re.I),
    re.compile(r"ishockeyklub(ber)?", re.I),
    re.compile(r"(fodbold|h[åa]ndbold|basketball|ishockey)hold_(fra|i)_", re.I),
    # Landsholds (national teams) per country
    re.compile(r"landshold(_i_)?", re.I),
]


def load_stub_categorized_pageids(
    categorylinks_path: Path, lt_id_to_cat: dict[int, str]
) -> set[int]:
    """Return set of pageids that belong to at least one blacklisted category.

    Uses regex on the CATEGORY NAME (from linktarget ns=14). Iterates
    categorylinks once, joining each cl_target_id against pre-computed
    blacklisted-lt_id set for speed."""
    # First: which lt_ids are blacklisted? (regex on decoded str)
    blacklisted_lt_ids: set[int] = set()
    for lt_id, cat_name in lt_id_to_cat.items():
        for pat in _STUB_CATEGORY_PATTERNS:
            if pat.search(cat_name):
                blacklisted_lt_ids.add(lt_id)
                break
    print(f"    {len(blacklisted_lt_ids):,} categories flagged stub-heavy",
          flush=True)

    # Second: walk categorylinks, collect pageids in flagged categories
    row_re = re.compile(
        rb"\((\d+),'(?:[^'\\]|\\.)*','[^']*','(?:[^'\\]|\\.)*','(page|subcat|file)',\d+,(\d+)\)"
    )
    demote: set[int] = set()
    with gzip.open(categorylinks_path, "rb") as f:
        for line in f:
            if not line.startswith(b"INSERT INTO `categorylinks` VALUES "):
                continue
            for m in row_re.finditer(line):
                if m.group(2) != b"page":
                    continue  # ignore subcat/file rows
                if int(m.group(3)) in blacklisted_lt_ids:
                    demote.add(int(m.group(1)))
    return demote


# pagelinks schema (3 cols):
#   pl_from (int), pl_from_namespace (int), pl_target_id (bigint)
_PAGELINKS_ROW_RE = re.compile(rb"\((\d+),(\d+),(\d+)\)")


def parse_pagelinks_inbound_counts(
    path: Path, lt_id_to_title: dict[int, str]
) -> dict[str, int]:
    """Return {title: inbound_count} — how many namespace-0 pages link to this
    target title. Only counts links FROM mainspace pages, so navigation links
    from portals/categories don't inflate the count.
    """
    counts: dict[str, int] = {}
    with gzip.open(path, "rb") as f:
        for line in f:
            if not line.startswith(b"INSERT INTO `pagelinks` VALUES "):
                continue
            for m in _PAGELINKS_ROW_RE.finditer(line):
                pl_from_ns, pl_target_id = int(m.group(2)), int(m.group(3))
                if pl_from_ns != 0:
                    continue
                title = lt_id_to_title.get(pl_target_id)
                if title is None:
                    continue
                counts[title] = counts.get(title, 0) + 1
    return counts


# Stub-heavy category prefixes to demote. Category names are what the
# curation script wrote — we don't rebuild the full categorylinks join here
# in this first cut (adds complexity). Instead we use title-shape heuristics
# to catch the biggest stub families:
_STUB_TITLE_PATTERNS = [
    re.compile(r"^\d+$"),                                    # pure-year like "1873"
    re.compile(r"^Liste over "),                             # list articles
    re.compile(r" Sogn( \(.+\))?$"),                         # Danish parish stubs
    re.compile(r" \(landsby\)"),                             # explicit village tag
    re.compile(r"^Portal:"),                                 # portals
]


def is_stub_title(title: str) -> bool:
    return any(p.search(title) for p in _STUB_TITLE_PATTERNS)


def score_and_tier(
    page_len: int, langlinks: int, inbound: int
) -> tuple[str, float]:
    """Three-signal tiering. `inbound` = # of other da.wiki articles that
    link INTO this one, our proxy for Danish-community relevance."""
    import math
    if page_len <= 0:
        return "excluded", 0.0
    score = (
        math.log10(page_len)
        * (1 + math.log1p(langlinks) / 2)
        * (1 + math.log1p(inbound) / 3)
    )
    # Tiers by intersection of thresholds. Inbound thresholds + a ratio
    # requirement (inbound ≥ langlinks/10) kills the "globally popular but
    # not Danish-relevant" tail (foreign celebrities, niche religions with
    # dedicated fan wikis) while keeping mainstream figures Danes reference.
    danish_relevance = inbound >= max(1, langlinks / 10)
    if langlinks >= 25 and page_len >= 8000 and inbound >= 100 and danish_relevance:
        tier = "T1_universal"
    elif langlinks >= 10 and page_len >= 5000 and inbound >= 40 and danish_relevance:
        tier = "T2_mainstream"
    elif langlinks >= 5 and page_len >= 3000 and inbound >= 10 and danish_relevance:
        tier = "T3_adequate"
    else:
        tier = "T4_niche"
    return tier, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pageids", type=Path,
                    default=Path("/mnt/data2/da_wiki_curation/pageids.tsv"))
    ap.add_argument("--page-sql", type=Path,
                    default=Path("/mnt/data2/da_wiki_dumps/dawiki-latest-page.sql.gz"))
    ap.add_argument("--dumps-dir", type=Path,
                    default=Path("/mnt/data2/da_wiki_dumps"))
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/data2/da_wiki_curation/salience.tsv"))
    ap.add_argument("--show-top", type=int, default=30,
                    help="Show top-N of each tier for eyeball")
    args = ap.parse_args()

    # 1) pageids.tsv → {pid: title}
    print("loading pageids …", flush=True)
    titles: dict[int, str] = {}
    with args.pageids.open() as f:
        for line in f:
            pid, title = line.rstrip("\n").split("\t", 1)
            titles[int(pid)] = title
    print(f"  {len(titles):,} pageids", flush=True)

    # 2) page.sql → {pid: page_len}
    print("parsing page.sql for page_len …", flush=True)
    t0 = time.time()
    page_lens = parse_page_lens(args.page_sql)
    print(f"  {len(page_lens):,} ns-0 pages ({time.time() - t0:.1f}s)", flush=True)

    # 3) langlinks.sql → {pid: langlinks_count}
    ll_path = ensure_dump("dawiki-latest-langlinks.sql.gz", args.dumps_dir)
    print("parsing langlinks.sql …", flush=True)
    t0 = time.time()
    langlinks = parse_langlinks_counts(ll_path)
    print(f"  {len(langlinks):,} pages with langlinks ({time.time() - t0:.1f}s)",
          flush=True)

    # 4) linktarget.sql → {lt_id: title} for both ns=0 (pages) and ns=14 (categories)
    lt_path = ensure_dump("dawiki-latest-linktarget.sql.gz", args.dumps_dir)
    print("parsing linktarget.sql (ns=0 and ns=14) …", flush=True)
    t0 = time.time()
    lt_id_to_title = parse_linktarget_by_ns(lt_path, 0)
    lt_id_to_cat = parse_linktarget_by_ns(lt_path, 14)
    print(f"  {len(lt_id_to_title):,} ns-0 pages / "
          f"{len(lt_id_to_cat):,} categories ({time.time() - t0:.1f}s)", flush=True)

    # 5) pagelinks.sql → {title: inbound_count}
    pl_path = ensure_dump("dawiki-latest-pagelinks.sql.gz", args.dumps_dir)
    print("parsing pagelinks.sql (inbound-link counts) …", flush=True)
    t0 = time.time()
    inbound = parse_pagelinks_inbound_counts(pl_path, lt_id_to_title)
    print(f"  {len(inbound):,} titles with inbound links ({time.time() - t0:.1f}s)",
          flush=True)

    # 5b) categorylinks.sql → pages in stub-heavy categories → demote set
    cl_path = args.dumps_dir / "dawiki-latest-categorylinks.sql.gz"
    if not cl_path.exists():
        cl_path = ensure_dump("dawiki-latest-categorylinks.sql.gz", args.dumps_dir)
    print("parsing categorylinks.sql (stub-heavy category members) …", flush=True)
    t0 = time.time()
    demote_pids = load_stub_categorized_pageids(cl_path, lt_id_to_cat)
    print(f"  {len(demote_pids):,} pageids in stub-heavy categories "
          f"({time.time() - t0:.1f}s)", flush=True)

    # 6) Score
    from collections import Counter
    tier_counts: Counter[str] = Counter()
    per_tier: dict[str, list[tuple[int, str, int, int, int, float]]] = {
        "T1_universal": [], "T2_mainstream": [], "T3_adequate": [], "T4_niche": [],
        "excluded": [],
    }
    with args.out.open("w") as f_out:
        f_out.write("pageid\ttitle\tpage_len\tlanglinks\tinbound\ttier\tscore\n")
        for pid, title in titles.items():
            plen = page_lens.get(pid, 0)
            ll = langlinks.get(pid, 0)
            inb = inbound.get(title.replace(" ", "_"), 0)
            if inb == 0:
                # try the space-form too, in case pageids.tsv uses spaces
                inb = inbound.get(title, 0)
            if is_stub_title(title):
                tier, score = "excluded", 0.0
            elif pid in demote_pids:
                tier, score = "T4_niche", 0.0
            else:
                tier, score = score_and_tier(plen, ll, inb)
            tier_counts[tier] += 1
            per_tier[tier].append((pid, title, plen, ll, inb, score))
            f_out.write(f"{pid}\t{title}\t{plen}\t{ll}\t{inb}\t{tier}\t{score:.3f}\n")

    print(f"\n=== tier bucket sizes ===")
    for tier in ["T1_universal", "T2_mainstream", "T3_adequate", "T4_niche", "excluded"]:
        print(f"  {tier:16s}  {tier_counts[tier]:7,d}")

    for tier in ["T1_universal", "T2_mainstream", "T3_adequate"]:
        rows = sorted(per_tier[tier], key=lambda r: -r[5])[:args.show_top]
        print(f"\n=== top {args.show_top} of {tier} (by score) ===")
        for pid, title, plen, ll, inb, sc in rows:
            print(f"  {sc:6.2f}  ll={ll:3d}  in={inb:5d}  len={plen:6d}  {title}")

    # Diagnostic: show that the problematic outliers now get demoted
    watchlist = ["Corbin Bleu", "Den Sande Jesus Kirke", "Interlingue",
                 "Neil Armstrong", "Bill Gates", "Galileo Galilei",
                 "Danmark", "Sverige"]
    print(f"\n=== diagnostic watchlist ===")
    print(f"  {'tier':16s}  {'ll':>4s}  {'in':>5s}  {'len':>6s}  title")
    for pid, title in titles.items():
        if title in watchlist:
            plen = page_lens.get(pid, 0)
            ll = langlinks.get(pid, 0)
            inb = inbound.get(title.replace(" ", "_"), inbound.get(title, 0))
            tier, _ = score_and_tier(plen, ll, inb) if not is_stub_title(title) else ("excluded", 0)
            print(f"  {tier:16s}  {ll:4d}  {inb:5d}  {plen:6d}  {title}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
