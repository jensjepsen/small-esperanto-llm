"""Audit a distractor-injected corpus. Every claim the stage makes, checked.

A distractor that the answer cites is not a distractor, it is a wrong answer.
A distractor that holds the same value in every row is not a distractor, it is
a constant the model learns to skip. A distractor the catalogue never declared
is a schema violation. None of these are visible in a spot check of ten rows,
and all of them would train the model on something other than what was
intended -- so each is a hard check here with a pass/fail, not a number to
eyeball.

Runs on the injected file, and on the twins file if given, because the twins
rename `parameters` while distractors live in `returns`: a renaming that
reached them would symbolize a key the catalogue documents in Danish.

    uv run python scripts/audit_distractors.py \
        --rows $OUT/sft_dist.jsonl --map $OUT/proposed_distractors.jsonl \
        --twins $OUT/sft_twins.jsonl --baseline $OUT/sft_answered.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inject_distractors import DEC, SEP, _render, signature  # noqa: E402
from render_toolmind_sft import param_symbol  # noqa: E402

SYM = re.compile(r"^[a-z][0-9a-f]{4}$")
NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _cited(answer: str, v) -> bool:
    """Would a reader see this value in the answer?

    Numbers compare numerically -- `8` must not match inside `2018` -- and
    strings need length, since a two-character value matches everywhere.
    """
    low = (answer or "").lower()
    if isinstance(v, bool) or v is None:
        return False
    if isinstance(v, (int, float)):
        for m in NUM.finditer(low):
            try:
                if float(m.group().replace(",", ".")) == float(v):
                    return True
            except ValueError:
                continue
        return False
    s = str(v).strip().lower()
    return len(s) > 3 and s in low


def rows_of(path):
    return [json.loads(l) for l in path.open() if l.strip()]


def catalogue(row):
    head = row["messages"][0]["content"]
    if SEP not in head:
        return None
    try:
        cat, _ = DEC.raw_decode(head.split(SEP, 1)[1].lstrip())
        return cat
    except Exception:
        return None


def preexisting(path):
    """(tool, field) pairs whose payload ALREADY carried the field.

    Without this the variety check judges SOURCE data: `get_movie_details`
    ships a real `cast` list, injection correctly skips it as already-present,
    and the audit then flagged the source repeating one film as a constant
    distractor. 5 false alarms in the first full run, all of them source
    fields this stage never touched.
    """
    out = set()
    if not path:
        return out
    for r in rows_of(path):
        msgs = r["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "tool_result":
                continue
            call = None
            for j in range(i - 1, -1, -1):
                if msgs[j]["role"] == "tool_call":
                    try:
                        call = json.loads(msgs[j]["content"])
                    except Exception:
                        pass
                    break
            name = (call or {}).get("name")
            try:
                p = json.loads(m["content"])
            except Exception:
                continue
            if name and isinstance(p, dict):
                for k in p:
                    out.add((name, k))
    return out


def constancy_bar(dmap, p=0.01):
    """Rows needed before one repeated value is suspicious, per (tool, field).

    With a pool of N, k rows land identically with probability N^-(k-1), so
    the bar is the smallest k where that drops below `p`. Wide-pool shapes
    (dates, ids, integers) get the floor of 3; an eight-value enum needs 4;
    a three-value one needs 6.
    """
    from toolmind_distractor_values import ENUM, FREE, classify
    bar = {}
    for sig_, fields in dmap.items():
        tool = sig_.split("(")[0]
        for f in fields:
            exs = f.get("eksempler") or []
            kind = classify(exs)
            pool = len(set(exs)) if kind in (ENUM, FREE) else 10 ** 6
            pool = max(pool, 2)
            k = 1 + math.ceil(math.log(1 / p) / math.log(pool))
            bar[(tool, f["felt"])] = max(3, k)
    return bar


def audit(rows, dmap, label, twins=False, skip=frozenset(), min_rows=None):
    # A TWIN CANNOT BE LOOKED UP BY ITS OWN SIGNATURE. symbolize_twins renames
    # `parameters` as well as `returns`, so a twin's signature reads
    # `get_stock_price(v466b)` and misses the map completely -- the first
    # attempt matched only zero-parameter tools, 4.7% where half was expected.
    # Each twin is paired to its original by `idx`, and the ORIGINAL supplies
    # both the signature and the English field names to symbolize.
    origin = {}
    if twins:
        for r in rows:
            if not r.get("sym"):
                origin[r.get("idx")] = r
    fail = Counter()
    stat = Counter()
    min_rows = min_rows or {}
    values = defaultdict(set)          # (tool, field) -> distinct values seen
    seen = Counter()                   # (tool, field) -> how often it occurred
    cites = []
    for r in rows:
        cat = catalogue(r)
        if cat is None:
            fail["catalogue-unreadable"] += 1
            continue
        by_name = {t["name"]: t for t in cat if "name" in t}
        msgs = r["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "tool_result":
                continue
            call = None
            for j in range(i - 1, -1, -1):
                if msgs[j]["role"] == "tool_call":
                    try:
                        call = json.loads(msgs[j]["content"])
                    except Exception:
                        pass
                    break
            fn = by_name.get((call or {}).get("name"))
            if not fn:
                continue
            lookup = fn
            if twins and r.get("sym"):
                src = origin.get(r.get("idx"))
                if src is None:
                    fail["twin-without-original"] += 1
                    continue
                try:
                    scat, _ = DEC.raw_decode(
                        src["messages"][0]["content"].split(SEP, 1)[1].lstrip())
                except Exception:
                    continue
                lookup = next((x for x in scat
                               if x.get("name") == fn.get("name")), None)
                if lookup is None:
                    continue
            elif twins:
                continue          # the original half is audited on its own pass
            fields = dmap.get(signature(lookup))
            if not fields:
                continue
            try:
                payload = json.loads(m["content"])
            except Exception:
                fail["payload-unparseable"] += 1
                continue
            if not isinstance(payload, dict):
                continue
            declared = ((fn.get("returns") or {}).get("properties") or {})
            answer = (msgs[i + 1]["content"]
                      if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant"
                      else "")
            stat["payloads"] += 1
            for f in fields:
                # IN A TWIN, THE FIELD IS NOT CALLED WHAT THE MAP CALLS IT.
                # symbolize_twins renames `returns` as well as `parameters`
                # (a parameters-only pass leaked the real names through the
                # returns block in 9,412 rows), so looking the distractor up
                # by its English name finds nothing and every check below
                # silently passes. The symbol is a pure function of
                # (row idx, key, namespace), so recompute it.
                # `name` is how the field appears in THIS row; `orig` is what
                # it is called everywhere else. Group on `orig` so the twin and
                # original passes bucket identically -- keying on the symbol
                # groups per-idx instead of per-field, which discarded the
                # per-pool constancy bar and produced 3 false alarms.
                orig = f["felt"]
                name = (param_symbol(r.get("idx", 0), orig, "r")
                        if twins else orig)
                present = name in payload
                if present:
                    stat["distractors-present"] += 1
                    values[(fn["name"], orig)].add(str(payload[name]))
                    seen[(fn["name"], orig)] += 1
                    # THE property: the gold answer must not cite it.
                    if answer and _cited(answer, payload[name]):
                        fail["ANSWER-CITES-DISTRACTOR"] += 1
                        if len(cites) < 8:
                            cites.append((fn["name"], orig, payload[name],
                                          answer[:110]))
                    # and it must not duplicate a real value
                    others = {x for k, v in payload.items() if k != name
                              for x in _render(v)}
                    if any(x in others for x in _render(payload[name])):
                        fail["duplicates-another-field"] += 1
                else:
                    stat["distractors-absent"] += 1
                # declared in the catalogue the model actually sees?
                if present and name not in declared:
                    fail["PAYLOAD-KEY-NOT-DECLARED"] += 1
                # In the ORIGINAL half a symbol-shaped key would mean the
                # twin transform leaked into it; in the twin half the key is
                # SUPPOSED to be a symbol, and a plain English one would mean
                # the renaming missed the distractor.
                if twins and not SYM.match(name):
                    fail["twin-distractor-NOT-symbolized"] += 1
                if not twins and SYM.match(name):
                    fail["original-distractor-symbolized"] += 1

    print(f"\n=== {label}: {len(rows):,} rows, {stat['payloads']:,} payloads "
          f"with a distractor-bearing tool ===")
    tot = stat["distractors-present"] + stat["distractors-absent"]
    if tot:
        print(f"  injected {stat['distractors-present']:,}/{tot:,} "
              f"({100*stat['distractors-present']/tot:.1f}%)")
    if values:
        # A pair is only CONSTANT if it had chances to vary. A tool occurring
        # once in the corpus trivially has one value, and counting that as a
        # constant made the first run cry wolf over 99 of 300 pairs.
        import statistics as st
        # A CONSTANT means the generator failed, not that a small closed set
        # repeated. `get_temperature.unit` draws from three values; seeing the
        # same one in three rows is a 1-in-9 coincidence, and across ~1,000
        # judgeable pairs several such collisions are expected. So the bar is
        # probabilistic: flag only when identical values are less than 1%
        # likely by chance, given how many values the field can take. A flat
        # `>=3 rows` bar produced 5 false alarms, every one of them an enum.
        const = [(k, seen[k]) for k, v in values.items()
                 if len(v) == 1 and k not in skip
                 and seen[k] >= min_rows.get(k, 3)]
        multi = [v for v in values.values() if len(v) > 1]
        print(f"  {len(values):,} (tool, field) pairs; "
              f"{sum(1 for k in values if seen[k] < 3):,} occur <3 times "
              f"(cannot judge variety)")
        if multi:
            print(f"  distinct values per varying pair: mean "
                  f"{st.mean(len(v) for v in multi):.1f}, "
                  f"max {max(len(v) for v in multi)}")
        if const:
            fail["CONSTANT-ACROSS-ROWS"] += len(const)
            print(f"  {len(const)} pairs are the SAME VALUE in every row "
                  f"they appear in:")
            for (t, f_), n_ in sorted(const, key=lambda x: -x[1])[:6]:
                print(f"    {t}.{f_}  ({n_} rows, 1 value)")
    if fail:
        print("  FAILURES:")
        for k, v in fail.most_common():
            print(f"    {k:<32} {v:,}")
    else:
        print("  no failures")
    for t, f, v, a in cites:
        print(f"    cited: {t}.{f} = {v!r}  in  {a!r}")
    return fail


def distribution(rows, label):
    """The statistic the whole exercise targets: does the payload pose a choice?"""
    w, n, buckets = Counter(), Counter(), Counter()
    for r in rows:
        msgs = r["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "tool_result":
                continue
            try:
                p = json.loads(m["content"])
            except Exception:
                continue
            if not isinstance(p, dict):
                continue
            w[min(len(p), 6)] += 1
            nums = [v for v in p.values()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
            n[min(len(nums), 4)] += 1
            ans = (msgs[i + 1]["content"]
                   if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant"
                   else "")
            if not ans or not nums:
                continue
            c = [_cited(ans, v) for v in nums]
            if len(nums) == 1:
                buckets["one number: nothing to discriminate"] += 1
            elif all(c):
                buckets["several, gold cites ALL"] += 1
            elif any(c):
                buckets["several, gold cites SOME <-- the target"] += 1
            else:
                buckets["several, gold cites NONE"] += 1
    t = sum(w.values()) or 1
    b = sum(buckets.values()) or 1
    print(f"\n--- {label} ---")
    print("  payload keys:", {k: f"{100*v/t:.0f}%" for k, v in sorted(w.items())})
    print("  numeric:     ", {k: f"{100*v/t:.0f}%" for k, v in sorted(n.items())})
    for k, v in buckets.most_common():
        print(f"    {100*v/b:5.1f}%  {k}")
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--twins", type=Path, default=None)
    ap.add_argument("--baseline", type=Path, default=None)
    args = ap.parse_args()

    dmap = {}
    for line in args.map.open():
        if line.strip():
            r = json.loads(line)
            dmap[r["sig"]] = r["fields"]
    print(f"{len(dmap):,} signatures carry distractors")

    rows = rows_of(args.rows)
    skip = preexisting(args.baseline)
    if skip:
        print(f"{len(skip):,} (tool, field) pairs already present before "
              f"injection; excluded from the variety check")
    bar = constancy_bar(dmap)
    fails = audit(rows, dmap, args.rows.name, skip=skip, min_rows=bar)
    if args.twins:
        fails += audit(rows_of(args.twins), dmap, args.twins.name,
                       twins=True, skip=skip, min_rows=bar)

    if args.baseline:
        distribution(rows_of(args.baseline), f"BEFORE ({args.baseline.name})")
    distribution(rows, f"AFTER ({args.rows.name})")

    hard = {"ANSWER-CITES-DISTRACTOR", "PAYLOAD-KEY-NOT-DECLARED",
            "twin-distractor-NOT-symbolized", "original-distractor-symbolized",
            "CONSTANT-ACROSS-ROWS"}
    bad = {k: v for k, v in fails.items() if k in hard}
    print()
    if bad:
        raise SystemExit(f"AUDIT FAILED: {bad}")
    print("AUDIT PASSED: no cited distractors, no undeclared payload keys, "
          "no symbolized distractor names")


if __name__ == "__main__":
    main()
