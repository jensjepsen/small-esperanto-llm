"""Group return observations by FUNCTION, not by tool name.

The returns map keys on `(tool_name, field)` and accumulates every field ever
seen under that name. Two things get merged that should not:

  across functions   43.3% of the 875 names carry more than one parameter
                     schema -- `search_movies` has 63 -- because glaive
                     invented each dialogue independently. Unioning their
                     returns gave `search_quotes` an AAPL stock ticker.

  within a function  different rows for the SAME signature named one concept
                     several ways (`final_price` / `new_price` /
                     `discounted_price`), and those were unioned too.

Measured consequence: across 8,875 observed payloads, only 65.9% of the fields
a spec declares are actually present in the payload that spec describes; the
median is 50% and `calculate_discount` manages 17.7% over 435 payloads. The
specs promise about twice what the tools return, payload generation is
schema-constrained so it filled the surplus, and the answer generator cited it.

The fix: bucket observations by `(name, parameter-schema fingerprint)` and keep
the fields present in a majority of that bucket's real payloads. That is a
contract describing what this function returns, rather than what everything
sharing its label ever returned.

Reports declared-vs-present coverage before and after, which is the success
criterion: it should approach 100% by construction.
"""
import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path


def param_fp(spec) -> str:
    """Identity of a function: its parameter property names.

    Names collide; signatures mostly do not. Sorted rather than ordered because
    JSON object key order is not meaningful and the same function is written
    with different key orders across rows.
    """
    props = ((spec.get("parameters") or {}).get("properties") or {})
    return ",".join(sorted(props))


def walk_leaves(obj, prefix=""):
    """Leaf paths of a payload, arrays collapsed to `field[]` like the map."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_leaves(v, f"{prefix}[]")
    else:
        yield prefix


def observations(path: Path):
    """(name, param_fp, payload_paths, top_level_fields) per observed result."""
    out = []
    for line in path.open():
        if not line.strip():
            continue
        da = (json.loads(line).get("da") or {})
        specs = {}
        for t in da.get("tools") or []:
            f = t.get("function") if isinstance(t, dict) else t
            if isinstance(f, dict) and f.get("name"):
                specs[f["name"]] = f
        last = None
        for m in da.get("conversations") or []:
            for tc in (m.get("tool_calls") or []):
                last = (tc.get("function") or {}).get("name") or last
            if m.get("role") != "tool" or not last or last not in specs:
                continue
            try:
                obj = json.loads(m.get("content") or "")
            except Exception:
                continue
            paths = {p for p in walk_leaves(obj) if p}
            if not paths:
                continue
            out.append((last, param_fp(specs[last]), paths,
                        set(obj) if isinstance(obj, dict) else set(),
                        specs[last]))
    return out


def top(path):
    return path.split(".")[0].removesuffix("[]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=Path("scratch/toolmind_da_v3/translated.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("scratch/toolmind_da_v3/returns_groups.jsonl"))
    ap.add_argument("--min-share", type=float, default=0.5,
                    help="keep fields present in >= this share of a group's "
                         "payloads")
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    obs = observations(args.src)
    print(f"{len(obs):,} observed payloads", flush=True)

    groups = defaultdict(list)
    for name, fp, paths, tops, spec in obs:
        groups[(name, fp)].append(paths)
    names = {n for n, _ in groups}
    print(f"{len(names):,} tool names -> {len(groups):,} (name, signature) "
          f"groups   [+{len(groups)-len(names):,} splits]", flush=True)

    # majority field set per group
    keep = {}
    for key, plist in groups.items():
        c = Counter()
        for paths in plist:
            for p in paths:
                c[p] += 1
        n = len(plist)
        keep[key] = {p for p, k in c.items() if k / n >= args.min_share}

    # coverage: declared vs actually present, old union vs new group
    union = defaultdict(set)
    for name, fp, paths, tops, spec in obs:
        union[name] |= paths

    old, new = [], []
    for name, fp, paths, tops, spec in obs:
        o = union[name]
        if o:
            old.append(len({top(p) for p in o} & tops)
                       / len({top(p) for p in o}))
        k = keep[(name, fp)]
        if k:
            new.append(len({top(p) for p in k} & tops)
                       / len({top(p) for p in k}))
    print(f"\ndeclared fields present in the payload they describe:")
    print(f"  OLD (union by name)      mean {100*st.mean(old):5.1f}%   "
          f"median {100*st.median(old):5.1f}%")
    print(f"  NEW (majority by signature) mean {100*st.mean(new):5.1f}%   "
          f"median {100*st.median(new):5.1f}%")
    print(f"  fields declared per payload: "
          f"{st.mean([len({top(p) for p in union[n]}) for n,_,_,_,_ in obs]):.1f}"
          f" -> {st.mean([len({top(p) for p in keep[(n,f)]}) for n,f,_,_,_ in obs]):.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for (name, fp), fields in sorted(keep.items()):
            fh.write(json.dumps({"tool": name, "signature": fp,
                                 "n": len(groups[(name, fp)]),
                                 "fields": sorted(fields)},
                                ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}", flush=True)

    split = sorted(((len({f for _, f in groups if _ == n}), n) for n in names),
                   reverse=True)[:args.show]
    print("\nnames that split into the most functions:")
    for cnt, n in split:
        print(f"  {n:<28} {cnt} signatures")
        for (nm, fp), fields in keep.items():
            if nm != n:
                continue
            print(f"      ({fp or 'no params'}) -> {sorted(fields)[:6]}")
            if cnt > 3:
                break


if __name__ == "__main__":
    main()
