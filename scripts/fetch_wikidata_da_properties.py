"""Fetch Wikidata properties with Danish labels + datatypes, and map YAGO's
predicates onto them.

The YAGO dump gives an extraction-shaped property set (things prose actually
asserts about entities) and real datatypes taken from the object values, but
its names are schema.org English. Danish names have to come from somewhere
that is not a hand-written translation table -- Wikidata carries `da` labels
for a large share of its properties, and YAGO's predicates are derived from
Wikidata in the first place, so this is a lookup rather than a translation.

Matching is by English label, after normalising schema.org camelCase
("birthPlace" -> "birth place"). Unmatched YAGO predicates are reported rather
than silently dropped.

Usage:
  python scripts/fetch_wikidata_da_properties.py \
      --yago scratch/yago_extraction_props.json \
      --out scratch/da_extraction_fields.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = "https://query.wikidata.org/sparql"
API = "https://www.wikidata.org/w/api.php"
UA = "espllm-dataset-builder/0.1 (research; contact via github.com/jensjepsen)"

QUERY = """
SELECT ?p ?dt ?en ?da WHERE {
  ?p wikibase:propertyType ?dt .
  ?p rdfs:label ?en . FILTER(lang(?en) = "en")
  OPTIONAL { ?p rdfs:label ?da . FILTER(lang(?da) = "da") }
}
"""


def camel_to_words(s: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", s).lower().replace("_", " ").strip()


def _api(params, tries=5):
    """Wikidata rate-limits hard during outages (SPARQL to 1 req/min, the API
    to 429 after a burst). Back off rather than dropping the row -- a silent
    gap here becomes a silently narrower field vocabulary."""
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    delay = 2.0
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def fetch_for(names, cache: Path):
    """Resolve English property names to Wikidata properties via the MediaWiki
    API. The SPARQL endpoint would give all ~12k in one query but is currently
    rate-limited to 1 req/min during a WDQS outage; this needs only ~90 small
    calls against a different service.
    """
    if cache.exists():
        print(f"using cached {cache}")
        return json.loads(cache.read_text())
    part = cache.with_suffix(".partial.json")
    found = json.loads(part.read_text()) if part.exists() else {}
    if found:
        print(f"resuming with {len(found)} already resolved")
    for i, (orig, words) in enumerate(names, 1):
        if orig in found:
            continue
        try:
            r = _api({"action": "wbsearchentities", "type": "property",
                      "language": "en", "uselang": "en", "limit": 5,
                      "search": words})
        except Exception as e:
            print(f"  [{i}] {words!r} search failed: {type(e).__name__}")
            continue
        hits = r.get("search") or []
        # require the English label to match what we searched for, so a fuzzy
        # hit does not silently become a different property
        exact = [h for h in hits if h.get("label", "").lower() == words]
        pick = (exact or hits[:1])
        if pick:
            found[orig] = pick[0]["id"]
        part.write_text(json.dumps(found, ensure_ascii=False))
        time.sleep(1.0)
        if i % 20 == 0:
            print(f"  resolved {len(found)}/{i}", flush=True)
    ids = sorted(set(found.values()))
    meta = {}
    for j in range(0, len(ids), 50):
        chunk = ids[j:j + 50]
        r = _api({"action": "wbgetentities", "ids": "|".join(chunk),
                  "props": "labels|datatype", "languages": "da|en"})
        for pid, e in (r.get("entities") or {}).items():
            labs = e.get("labels", {})
            meta[pid] = {"pid": pid, "datatype": e.get("datatype"),
                         "en": labs.get("en", {}).get("value"),
                         "da": labs.get("da", {}).get("value")}
        time.sleep(0.2)
    out = {"resolved": found, "meta": meta}
    cache.write_text(json.dumps(out, ensure_ascii=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yago", type=Path, default=Path("scratch/yago_extraction_props.json"))
    ap.add_argument("--types", type=Path, default=Path("/tmp/yago_pred_types.txt"),
                    help="pred -> object datatype, derived from the dump")
    ap.add_argument("--cache", type=Path, default=Path("scratch/wikidata_properties.json"))
    ap.add_argument("--out", type=Path, default=Path("scratch/da_extraction_fields.json"))
    args = ap.parse_args()

    yago_pre = json.loads(args.yago.read_text())
    names = [(y["name_en"], camel_to_words(y["name_en"])) for y in yago_pre]
    got = fetch_for(names, args.cache)
    resolved, meta = got["resolved"], got["meta"]
    n_da = sum(1 for m in meta.values() if m.get("da"))
    print(f"  properties resolved: {len(resolved)}/{len(names)}; "
          f"with Danish label: {n_da}/{len(meta)}")

    # object datatype straight from the dump beats guessing from the name
    dump_dt = {}
    if args.types.exists():
        for line in args.types.read_text().splitlines():
            parts = line.split()
            if len(parts) == 3:
                _, pred, dt = parts
                dump_dt.setdefault(pred, dt)     # first = most frequent
    DT = {":dateTime": "date", ":decimal": "number", ":integer": "number",
          ":double": "number", "iri": "entity", "literal": "string"}

    out, missing = [], []
    for y in yago_pre:
        pid = resolved.get(y["name_en"])
        hit = meta.get(pid) if pid else None
        dt = DT.get(dump_dt.get(y["prop"], ""), y["type"])
        if not hit or not hit.get("da"):
            missing.append((y["name_en"], camel_to_words(y["name_en"]), bool(hit)))
            continue
        out.append({"name_da": hit["da"], "name_en": y["name_en"],
                    "pid": hit["pid"], "type": dt,
                    "wikidata_datatype": hit["datatype"],
                    "yago_uses": y["yago_uses"]})

    out.sort(key=lambda r: -r["yago_uses"])
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    from collections import Counter
    print(f"\nmatched {len(out)}/{len(yago_pre)} YAGO predicates to a Danish label")
    print("types:", dict(Counter(r["type"] for r in out)))
    print(f"\n{'da':<28}{'en':<22}{'type':<9}{'pid'}")
    for r in out[:30]:
        print(f"  {r['name_da']:<26}{r['name_en']:<22}{r['type']:<9}{r['pid']}")
    if missing:
        print(f"\nunmatched ({len(missing)}):")
        for n, k, had_en in missing[:20]:
            print(f"  {n:<22} searched={k!r} en_found={had_en}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
