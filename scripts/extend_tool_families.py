"""Stage 2: give existing tools a sibling, so a scenario can chain.

Reads a frozen catalogue, picks anchors that REQUIRE an identifier they cannot
produce, and invents one sibling per anchor that returns it. Appends the
siblings to the catalogue; every existing tool is left byte-identical, so the
frozen catalogue stays frozen and rows already built stay reproducible.

Only anchors with an identifier-shaped required parameter are candidates --
those are the ones a chain can hand something to. Everything else is skipped
without an API call.

    uv run --no-project --with aiohttp --with langdetect python \\
        scripts/extend_tool_families.py \\
          --tools data/tool_calls/tools_v2.jsonl \\
          --out   data/tool_calls/tools_v3.jsonl \\
          --n 15 --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tool_dialogues_proc import (  # noqa: E402
    IDENTIFYING, FatalAPIError, _key, catalogue_faults, family_index,
    build_producer, find_link, gate_sibling, gate_tool, gate_tool_examples,
    invent_lookup, tool_signature,
)


def candidates(tools, min_handles=1):
    """(anchor, [keys]) -- EVERY handle it requires and cannot produce.

    All of them, not the first: a consumer needing `detector_id` AND
    `experiment_id` wants two lookups, and two lookups are what make the row
    fan-in -- two independent calls in one turn, either order correct. One
    sibling would leave the second handle to be spoken by the user, which is a
    valid row but the weaker one.
    """
    out = []
    for t in tools:
        rets = {r["name"] for r in (t.get("returns") or [])}
        keys = [p.get("name") for p in (t.get("parameters") or [])
                if p.get("required") and IDENTIFYING.search(p.get("name") or "")
                and p.get("name") not in rets]
        if len(keys) >= min_handles:
            out.append((t, keys))
    return out


async def run(args):
    import aiohttp
    tools = [json.loads(l) for l in args.tools.open() if l.strip()]
    fam = family_index(tools)
    # RESUMABLE PER HANDLE, not per family. Skipping a whole scenario because
    # it already has one sibling would strand every partially-extended family:
    # a consumer needing two handles that got one could never receive the
    # second, so fan-in would be unreachable for exactly the families a first
    # pass touched. `_link_key` records what each sibling covers, so a later
    # run asks only for what is missing -- feed the OUTPUT back as --tools and
    # the count can be raised any number of times without re-buying anything.
    covered = {(t.get("_scenario"), t.get("_link_key")) for t in tools
               if t.get("_link_key")}
    cands = []
    for t, ks in candidates(tools, args.min_handles):
        todo = [k for k in ks if (t.get("_scenario"), k) not in covered]
        if todo:
            cands.append((t, todo))
    print(f"{len(tools):,} tools   {len(fam):,} families   "
          f"{len(cands):,} anchors needing >={args.min_handles} handle(s)   "
          f"{sum(len(k) for k in (c[1] for c in cands)):,} handles still "
          f"uncovered ({len(covered):,} already done)",
          flush=True)
    cands = cands[:args.n] if args.n else cands
    print(f"inventing {len(cands):,} siblings", flush=True)

    stats, made = Counter(), []
    sem = asyncio.Semaphore(args.concurrency)
    sigs = {tool_signature(t) for t in tools}

    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}"}) as session:
        async def one_key(anchor, key):
            """One producer for one handle. Returns the sibling or None."""
            for attempt in range(2):
                # The model describes only the human-sayable input; the link
                # half is copied from the consumer by build_producer, so format
                # mismatch and circularity cannot arise.
                spec, u = await invent_lookup(
                    session, anchor, key, 0.9 if not attempt else 0.6)
                sib = build_producer(anchor, key, spec)
                if sib is None:
                    stats["reject:unusable-spec"] += 1
                    continue
                why = (gate_sibling(sib, anchor, key)
                       or gate_tool(sib) or gate_tool_examples(sib))
                if why:
                    stats[f"reject:{why.split(':')[0]}"] += 1
                    continue
                    # Catalogue faults are NOT a rejection here. The frozen
                    # catalogue was built the same way -- invent, then repair
                    # offline -- and that pass added 12,947 envelopes alone. A
                    # sibling rejected for a fault the repair exists to fix
                    # would be thrown away for the same reason most of the
                    # original 4,991 would have been. Repair runs after the
                    # gather, in one batch, and the fault check happens then.
                sib["_scenario"] = anchor.get("_scenario")
                sib["_sibling_of"] = anchor.get("name")
                sib["_link_key"] = key
                if tool_signature(sib) in sigs:
                    stats["reject:duplicate-signature"] += 1
                    return None
                sigs.add(tool_signature(sib))
                if not find_link(anchor, [sib, anchor]):
                    stats["reject:no-link-after-all"] += 1
                    return None
                stats["invented"] += 1
                return sib
            stats["reject:gave-up"] += 1
            return None

        async def one(anchor, keys):
            async with sem:
                got = []
                for k in keys[:args.max_siblings]:
                    sib = await one_key(anchor, k)
                    if sib is not None:
                        got.append(sib)
                if len(got) > 1:
                    stats["family:fan-in"] += 1
                elif got:
                    stats["family:single-link"] += 1
                made.extend(got)

        await asyncio.gather(*[one(t, ks) for t, ks in cands])

    # ── repair, then judge ────────────────────────────────────────────────
    # Shelled out rather than imported: repair_tool_catalogue.py is the script
    # that produced the frozen catalogue, it is a pure file->file transform,
    # and refactoring it into a library to save one subprocess would risk the
    # one piece of this pipeline that is known-good.
    if made:
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as td:
            src, dst = Path(td) / "in.jsonl", Path(td) / "out.jsonl"
            src.write_text("\n".join(json.dumps(t, ensure_ascii=False)
                                     for t in made) + "\n")
            r = subprocess.run([sys.executable,
                                str(Path(__file__).with_name(
                                    "repair_tool_catalogue.py")),
                                str(src), str(dst)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"repair failed:\n{r.stderr[-800:]}")
            repaired = [json.loads(x) for x in dst.open() if x.strip()]
        print(f"\nrepair: {len(made)} in -> {len(repaired)} out")
        for line in r.stdout.strip().splitlines():
            if line.strip():
                print(f"   {line.strip()}")
        # Keyed on (name, scenario), NOT name. 888 tools share 370 names -- the
        # catalogue keeps collisions on purpose -- so a name-only lookup
        # re-gates a sibling against a DIFFERENT tool that happens to share its
        # anchor's name, and then rejects it for a format mismatch against
        # examples its real anchor never had. The family key is the identity.
        keep = []
        by_key = {(t["name"], t.get("_scenario")): t for t in tools}
        for sib in repaired:
            faults = catalogue_faults(sib)
            if faults:
                stats[f"reject:after-repair:{faults[0][0]}"] += 1
                continue
            anchor = by_key.get((sib.get("_sibling_of"), sib.get("_scenario")))
            # RE-GATE. The repair demotes required parameters it judges
            # unaskable, and it stripped the last one from a lookup tool in the
            # first smoke -- `lookup_hs_code` came out with `required: []`, a
            # lookup that takes nothing, so the user has no way to supply the
            # thing being looked up and the chain has no starting point. The
            # pre-repair gate cannot see this because the demotion happens
            # afterwards.
            why = anchor and gate_sibling(sib, anchor, sib.get("_link_key"))
            if why:
                stats[f"reject:after-repair:{why}"] += 1
                continue
            # The link must survive the repair: it demotes and deletes
            # parameters, and a demoted required key is no longer a chain.
            if not anchor or not find_link(anchor, [sib, anchor]):
                stats["reject:link-lost-in-repair"] += 1
                continue
            stats["kept"] += 1
            keep.append(sib)
        made = keep

    print("\n" + "\n".join(f"   {v:>5}  {k}" for k, v in
                           sorted(stats.items())), flush=True)
    if made:
        with args.out.open("w") as fh:
            for t in tools:
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")
            for t in made:
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"\n-> {args.out}  ({len(tools):,} + {len(made):,} = "
              f"{len(tools) + len(made):,} tools)", flush=True)
        for s in made[:args.show]:
            a = next(t for t in tools if t["name"] == s["_sibling_of"])
            print(f"\n  FAMILY {s['_scenario']}  link={s['_link_key']}")
            print(f"    producer {s['name']:<34} {s.get('description','')[:58]}")
            print(f"      required: {[p['name'] for p in (s.get('parameters') or []) if p.get('required')]}")
            print(f"      returns : {[r['name'] for r in (s.get('returns') or [])]}")
            print(f"    consumer {a['name']:<34} {a.get('description','')[:58]}")
            print(f"      required: {[p['name'] for p in (a.get('parameters') or []) if p.get('required')]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=15, help="0 = every candidate")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--min-handles", type=int, default=1,
                    help="only anchors requiring at least this many handles. "
                         "2 targets the fan-in case.")
    ap.add_argument("--max-siblings", type=int, default=2,
                    help="most producers to invent per anchor")
    ap.add_argument("--show", type=int, default=4)
    args = ap.parse_args()
    try:
        asyncio.run(run(args))
    except FatalAPIError as e:
        raise SystemExit(f"ABORTED: {e}")


if __name__ == "__main__":
    main()
