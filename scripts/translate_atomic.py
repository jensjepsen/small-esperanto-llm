"""Translate ATOMIC 2020 commonsense KG to Esperanto, component-by-component.

Strategy: dedupe heads and tails, translate each unique string once via Gemini,
then reassemble triples in Esperanto. The 23 relations have a hand-written
Esperanto lookup, no API translation needed.

Pre-processing substitutions (done before sending to translator):
- PersonX  → Petro    (concrete name reads more naturally in Esperanto)
- PersonY  → Maria
- PersonZ  → Anna
- ___      → ion      (placeholder "something")
- "none"   → no API call, mapped to "neniu" directly

Outputs:
- atomic_dict.json:  english_str → esperanto_str  (the translation cache)
- atomic_eo.jsonl:   one Esperanto triple per line {head, relation, tail}
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

csv.field_size_limit(sys.maxsize)


RELATION_EO = {
    # Social — PersonX (subject)
    "xIntent":   "intencas",
    "xAttr":     "estas (priskribo)",
    "xWant":     "deziras poste",
    "xNeed":     "antaŭe bezonis",
    "xEffect":   "rezulte okazas al li",
    "xReact":    "sentas",
    "xReason":   "ĉar",
    # Social — PersonY (other)
    "oWant":     "alia persono deziras",
    "oReact":    "alia persono sentas",
    "oEffect":   "al alia persono okazas",
    # Event / temporal
    "isBefore":  "antaŭ tio okazas",
    "isAfter":   "post tio okazas",
    "HasSubEvent":"inkluzivas la paŝon",
    "Causes":    "kaŭzas",
    "HinderedBy":"estas malhelpata de",
    "isFilledBy":"povas plenigi (la mankon)",
    # Physical / object
    "AtLocation":"troviĝas en",
    "ObjectUse": "uzata por",
    "CapableOf": "kapablas",
    "HasProperty":"havas econ",
    "MadeUpOf":  "konsistas el",
    "Desires":   "deziras",
    "NotDesires":"ne deziras",
}


# Pre/post-processing of head/tail strings
PERSON_MAP_EN = [("PersonX", "Petro"), ("PersonY", "Maria"), ("PersonZ", "Anna")]
NONE_TAIL = "none"
NONE_TAIL_EO = "neniu"


def preprocess(s: str) -> str:
    """Replace ATOMIC placeholders with natural Esperanto stand-ins.

    We do this BEFORE translation so the LLM sees natural English with
    real names, not 'PersonX' which it might leave untranslated or mangle.
    """
    out = s.strip()
    for src, dst in PERSON_MAP_EN:
        # Possessives first to avoid "Petro 's" artifacts
        out = re.sub(rf"{re.escape(src)}'s", f"{dst}'s", out)
        out = re.sub(rf"\b{re.escape(src)}\b", dst, out)
    # ___ → "something"  (translator will pick natural EO equivalent)
    out = out.replace("___", "something")
    return out


def load_unique_strings(path: Path):
    """Load TSV, return (unique_strings, triple_iterator-callable)."""
    heads, tails = set(), set()
    triples = []
    with open(path) as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3:
                continue
            h, r, t = row[0].strip(), row[1].strip(), row[2].strip()
            if not h or not t or r not in RELATION_EO:
                continue
            heads.add(h)
            tails.add(t)
            triples.append((h, r, t))
    return heads, tails, triples


def translate_batch_via_gemini(client, strings: list[str], model_name: str, timeout_s: int = 60) -> dict:
    """Send a batch of strings to Gemini, get a dict mapping en → eo."""
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(strings))
    prompt = f"""Traduku la sekvajn anglajn frazetojn al natura Esperanto.

Reguloj:
- Traduku NUR al Esperanto. Konservu nomojn (Petro, Maria, Anna) kiel ili estas.
- Uzu ĝustajn supersignojn (ĉ, ĝ, ĥ, ĵ, ŝ, ŭ).
- "something" → "io" aŭ "ion" laŭ kunteksto.
- Konservu mallongajn, koncizan stilon. Se la angla estas mallonga frazeto, traduku same mallonge.
- Traduku ĉion en sama linio, sen klarigoj.

Respondu kiel JSON-listo de objektoj kun kampoj "i" (la indekso) kaj "eo" (la traduko).
Ekzemplo: [{{"i": 1, "eo": "petro iras hejmen"}}, {{"i": 2, "eo": "ĝoja"}}]

Frazetoj:
{numbered}

Respondu NUR per la JSON, sen alia teksto."""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={"http_options": {"timeout": timeout_s * 1000}},
    )
    text = response.text.strip()
    if "```" in text:
        for part in text.split("```"):
            if part.lstrip().startswith(("json", "[")):
                text = part.removeprefix("json").strip()
                break
    try:
        results = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}; got: {text[:200]}", file=sys.stderr)
        return {}
    out = {}
    for r in results:
        idx = r.get("i")
        eo = r.get("eo", "").strip()
        if idx is None or not eo:
            continue
        if 1 <= idx <= len(strings):
            out[strings[idx - 1]] = eo
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/tmp/dl/atomic2020_data-feb2021/train.tsv"))
    parser.add_argument("--cache", type=Path, default=Path("data/atomic_eo/atomic_dict.json"))
    parser.add_argument("--out-triples", type=Path, default=Path("data/atomic_eo/atomic_eo.jsonl"))
    parser.add_argument("--max-strings", type=int, default=None,
                        help="Limit total strings to translate (for sampling)")
    parser.add_argument("--max-heads", type=int, default=None,
                        help="Keep only the N most-frequent unique heads (symmetric)")
    parser.add_argument("--balanced", action="store_true",
                        help="With --max-heads, take half PersonX and half object heads")
    parser.add_argument("--personx-heads", type=int, default=None,
                        help="Asymmetric mode: number of PersonX heads to keep (overrides --max-heads)")
    parser.add_argument("--object-heads", type=int, default=None,
                        help="Asymmetric mode: number of non-PersonX heads to keep")
    parser.add_argument("--max-tails-per-relation", type=int, default=3,
                        help="For each (head, relation), keep at most N distinct tails")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--parallel", type=int, default=8,
                        help="Number of concurrent API calls")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be sent to the API; don't actually call it")
    parser.add_argument("--n-dry", type=int, default=20,
                        help="In dry-run, show this many sample strings")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    heads, tails, triples = load_unique_strings(args.input)
    print(f"  triples: {len(triples):,}  unique heads: {len(heads):,}  unique tails: {len(tails):,}")

    # Filter heads to most-frequent if requested
    if args.personx_heads is not None or args.object_heads is not None:
        # Asymmetric: separate caps for PersonX vs object-type heads.
        # PersonX side is well-curated deep into the ranks; object heads
        # degrade (typos, freq < 5) past ~2-3k ranks — cap them tighter.
        head_counts = collections.Counter(h for h, _, _ in triples)
        n_px = args.personx_heads or 0
        n_ob = args.object_heads or 0
        personx = [h for h, _ in head_counts.most_common() if "PersonX" in h][:n_px]
        other   = [h for h, _ in head_counts.most_common() if "PersonX" not in h][:n_ob]
        keep_heads = set(personx) | set(other)
        triples = [(h, r, t) for h, r, t in triples if h in keep_heads]
        heads = set(h for h, _, _ in triples)
        tails = set(t for _, _, t in triples)
        print(f"  asymmetric filter: {n_px} PersonX + {n_ob} object heads → "
              f"triples={len(triples):,} heads={len(heads):,} tails={len(tails):,}")
    elif args.max_heads:
        head_counts = collections.Counter(h for h, _, _ in triples)
        if args.balanced:
            half = args.max_heads // 2
            personx = [h for h, _ in head_counts.most_common() if "PersonX" in h][:half]
            other   = [h for h, _ in head_counts.most_common() if "PersonX" not in h][:args.max_heads - half]
            keep_heads = set(personx) | set(other)
        else:
            keep_heads = set(h for h, _ in head_counts.most_common(args.max_heads))
        triples = [(h, r, t) for h, r, t in triples if h in keep_heads]
        heads = set(h for h, _, _ in triples)
        tails = set(t for _, _, t in triples)
        print(f"  after head filter: triples={len(triples):,} heads={len(heads):,} tails={len(tails):,}")

    # Cap tails per (head, relation) to reduce noise/duplication
    if args.max_tails_per_relation:
        bucket = collections.defaultdict(list)
        for h, r, t in triples:
            if t == NONE_TAIL:
                continue
            bucket[(h, r)].append(t)
        triples = []
        for (h, r), ts in bucket.items():
            # Keep the top-N most-common (or first N) tails
            counts = collections.Counter(ts)
            kept = [t for t, _ in counts.most_common(args.max_tails_per_relation)]
            for t in kept:
                triples.append((h, r, t))
        heads = set(h for h, _, _ in triples)
        tails = set(t for _, _, t in triples)
        print(f"  after tail cap: triples={len(triples):,} heads={len(heads):,} tails={len(tails):,}")

    # All unique strings to translate (heads + tails)
    all_strings = sorted(heads | tails)
    if args.max_strings:
        all_strings = all_strings[:args.max_strings]
    print(f"Strings to translate: {len(all_strings):,}")

    # Pre-process
    needs_translation = []
    raw_to_clean = {}
    for s in all_strings:
        clean = preprocess(s)
        raw_to_clean[s] = clean
        # 'none' is handled directly
        if s == NONE_TAIL:
            continue
        needs_translation.append(clean)

    needs_translation = list(dict.fromkeys(needs_translation))  # preserve order, dedupe
    print(f"After dedupe by cleaned form: {len(needs_translation):,} unique strings")

    # Load existing cache
    cache: dict[str, str] = {}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text())
        print(f"Cache hits available: {len(cache):,}")
    todo = [s for s in needs_translation if s not in cache]
    print(f"Strings still to translate: {len(todo):,}")

    if args.dry_run:
        print(f"\n--- DRY RUN: first {args.n_dry} strings, before/after preprocess ---")
        for s in all_strings[:args.n_dry]:
            print(f"  {s!r}")
            if raw_to_clean[s] != s:
                print(f"    → {raw_to_clean[s]!r}")
        print(f"\n--- DRY RUN: example batch payload (first {min(args.batch_size, 10)} strings) ---")
        sample = todo[:min(args.batch_size, 10)]
        if sample:
            numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sample))
            print(numbered)
        print(f"\nWould make ~{(len(todo) + args.batch_size - 1) // args.batch_size} API calls")
        return

    # Real translation
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable.", file=sys.stderr)
        sys.exit(1)
    from google import genai
    client = genai.Client(api_key=api_key)

    # Ensure output dirs
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.out_triples.parent.mkdir(parents=True, exist_ok=True)

    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    print(f"Translating in {n_batches} batches of {args.batch_size}, {args.parallel}-way parallel...", flush=True)
    cache_lock = threading.Lock()
    start_t = time.time()

    def do_batch(batch):
        try:
            return translate_batch_via_gemini(client, batch, args.model)
        except Exception as e:
            try:
                return translate_batch_via_gemini(client, batch, args.model)
            except Exception as e2:
                return {"__error__": str(e2)}

    completed = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(do_batch, todo[bi * args.batch_size : (bi + 1) * args.batch_size]): bi
            for bi in range(n_batches)
        }
        for fut in as_completed(futures):
            bi = futures[fut]
            results = fut.result()
            err = results.pop("__error__", None) if isinstance(results, dict) else None
            with cache_lock:
                cache.update(results or {})
                completed += 1
                cache_size = len(cache)
                if completed % 10 == 0 or completed == n_batches:
                    args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
            elapsed = time.time() - start_t
            eta = elapsed / completed * (n_batches - completed)
            tag = f"ERR ({err})" if err else f"+{len(results)}"
            print(f"  [{completed:>3}/{n_batches}] batch {bi+1}: {tag}; cache={cache_size:,}; ETA={eta/60:.1f}m",
                  flush=True)

    args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    print(f"Wrote translation cache ({len(cache):,} entries) → {args.cache}")

    # Reassemble triples
    print("Reassembling Esperanto triples...")
    out_count = 0
    with open(args.out_triples, "w") as f:
        for h, r, t in triples:
            h_clean = raw_to_clean.get(h, preprocess(h))
            t_clean = NONE_TAIL_EO if t == NONE_TAIL else raw_to_clean.get(t, preprocess(t))
            h_eo = NONE_TAIL_EO if h == NONE_TAIL else cache.get(h_clean)
            t_eo = NONE_TAIL_EO if t == NONE_TAIL else cache.get(t_clean)
            if not h_eo or not t_eo:
                continue
            f.write(json.dumps({
                "head": h_eo,
                "relation": r,
                "relation_eo": RELATION_EO[r],
                "tail": t_eo,
            }, ensure_ascii=False) + "\n")
            out_count += 1
    print(f"Wrote {out_count:,} Esperanto triples → {args.out_triples}")


if __name__ == "__main__":
    main()
