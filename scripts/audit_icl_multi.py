"""Format-agnostic audit of multi-format ICL rows.

The single-format audit json.loads()ed every answer, which is meaningless once
a row can be rendered as tsv or <tag>value</tag>. Everything here goes through
the generator's own canon() with the row's declared format, so the audit and
the generator cannot drift apart in what they consider parseable.

Adds one check the single-format version had no reason to make: the format
must be CONSTANT within a row. A prompt whose demonstrations mix formats
teaches nothing about which to produce -- the same class of defect as a key
that is never demonstrated.
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "scripts")
from gen_icl_json import canon, NULL, SYMBOLS  # noqa: E402

SPLITS = ("train", "eval_schema", "eval_format", "eval_both", "val")


def blocks(user: str):
    """-> ([(passage, answer)], target_passage). Splitting on the literal
    'Tekst:\\n' delimiter rather than on blank lines, because passages are
    bullet lists that contain newlines of their own."""
    parts = user.split("\n\nTekst:\n")
    demos = []
    for b in parts[1:-1]:
        p, a = b.rsplit("\nSvar:", 1)
        demos.append((p.strip(), a.strip()))
    tgt = parts[-1].rsplit("\nSvar:", 1)[0].strip() if len(parts) > 1 else ""
    return demos, tgt


def main(d: Path):
    F, N, H = Counter(), Counter(), Counter()
    ex = defaultdict(list)
    total = 0

    def hit(bucket, name, r, det=""):
        bucket[name] += 1
        if len(ex[name]) < 2:
            ex[name].append((r["meta"], det))

    for sp in SPLITS:
        f = d / f"{sp}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            total += 1
            fmt = r["meta"]["format"]
            demos, tgt_p = blocks(r["messages"][0]["content"])
            ans = r["messages"][1]["content"]

            keys = None
            keyset = _keys_of(r)
            tg = canon(ans, fmt, keyset)
            if tg is None:
                hit(F, "target does not parse under its own format", r,
                    f"{fmt}: {ans[:70]}")
                continue
            keys = set(tg)

            dgs = []
            for p, a in demos:
                g = canon(a, fmt, keyset)
                if g is None:
                    hit(F, "an exemplar does not parse under the row format",
                        r, f"{fmt}: {a[:70]}")
                else:
                    dgs.append((p, g))

            # every key used by the answer must be demonstrated
            dk = {k for _, g in dgs for k in g}
            if keys - dk:
                hit(F, "target key never appears in an exemplar", r,
                    str(sorted(keys - dk)))
            # ... and demonstrated with a real value at least once
            nonnull = {k for _, g in dgs for k, v in g.items()
                       if v != [NULL]}
            need = {k for k, v in tg.items() if v != [NULL]}
            if need - nonnull:
                hit(F, "target key only ever demonstrated as null", r,
                    str(sorted(need - nonnull)))

            # every value must be recoverable from its own passage
            for p, g in dgs + [(tgt_p, tg)]:
                low = p.lower()
                low = re.sub(r"\s+", " ", low)   # canon() collapses the
                #   value's whitespace, so the passage must be collapsed too
                #   or a line-wrapped "spaghe\nttien" never matches
                for k, vs in g.items():
                    for v in vs:
                        # booleans are exempt by construction: true/false
                        # never appear literally in a Danish passage
                        if v in (NULL, "true", "false") or len(v) < 3:
                            continue
                        if v not in low and not _numeric_ok(v, p):
                            hit(F, "value not recoverable from its passage",
                                r, f"{k}={v!r}")

            # Format consistency is already proven above: every exemplar is
            # parsed with the ROW's format and a failure is reported as
            # "does not parse under the row format". A surface-shape
            # heuristic on top of that only produced false positives
            # ("15. maj -> kat_c" read as numbered, "[a@b] -> x" as bracket).

            if ans.strip() in [a.strip() for _, a in demos]:
                hit(N, "target answer identical to an exemplar", r)
            ps = [p for p, _ in demos] + [tgt_p]
            if any(i != j and len(x) > 20 and x in y
                   for i, x in enumerate(ps) for j, y in enumerate(ps)):
                hit(N, "one passage contained in another", r)
            if "felterne" in r["messages"][0]["content"]:
                hit(H, "instruction text leaked", r)
            if len(demos) != r["meta"]["shots"]:
                hit(H, "shots metadata disagrees with prompt", r,
                    f"{len(demos)} vs {r['meta']['shots']}")

    print(f"audited {total} rows from {d}")
    for lab, b in (("FATAL", F), ("NOISE", N), ("HYGIENE", H)):
        print(f"\n--- {lab} ---")
        if not b:
            print("  (none)")
        for k, v in b.most_common():
            print(f"  {v:>5}  {k}")
            for m, det in ex[k][:2]:
                print(f"          fmt={m['format']} schema={m['schema'][:40]}"
                      + (f"\n          {det[:120]}" if det else ""))


def _shape(ans: str) -> str:
    a = ans.strip()
    if a.startswith("{"):
        return "json"
    if a.startswith("<"):
        return "tagged"
    if re.match(r"^\d+\.\s", a):
        return "numbered"
    if "\t" in a:
        return "tsv"
    if re.search(r"^\[[^\]]+\]\s", a, re.M):
        return "bracket"
    if "->" in a:
        return "arrow"
    if re.search(r"^[^\s:=]+=", a, re.M):
        return "eq"
    return "colon"


def _keys_of(r):
    """The key set this row uses, derived from metadata rather than by
    regexing the rendered answer.

    A per-format pattern table is the same maintenance trap as a per-format
    break mutation: adding bracket_pair/brace_pair broke both. The schema and
    the symbol scheme fully determine the keys, and the symbol shuffle only
    permutes the assignment, never the set.
    """
    m = r.get("meta", r)
    if m.get("symbols", "none") == "none":
        return set(m["schema"].split("|"))
    return set(SYMBOLS[m["symbols"]][:m["n_fields"]])


def _numeric_ok(v: str, p: str) -> bool:
    try:
        x = float(v.replace(",", "."))
    except ValueError:
        return False
    s = "%g" % x
    forms = {s, s.replace(".", ",")}
    if x.is_integer():
        forms.add(f"{int(x):,}".replace(",", "."))
    return any(f in p for f in forms)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/icl_multi"))
