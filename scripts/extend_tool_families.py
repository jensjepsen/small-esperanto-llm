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
    find_link, gate_sibling, gate_tool, gate_tool_examples, invent_sibling,
    tool_signature,
)


def candidates(tools):
    """Anchors that require a handle they do not themselves produce."""
    out = []
    for t in tools:
        rets = {r["name"] for r in (t.get("returns") or [])}
        for p in (t.get("parameters") or []):
            n = p.get("name") or ""
            if p.get("required") and IDENTIFYING.search(n) and n not in rets:
                out.append((t, n))
                break
    return out


async def run(args):
    import aiohttp
    tools = [json.loads(l) for l in args.tools.open() if l.strip()]
    fam = family_index(tools)
    have = {sc for sc, ms in fam.items() if len(ms) > 1}
    cands = [(t, k) for t, k in candidates(tools)
             if t.get("_scenario") not in have]
    print(f"{len(tools):,} tools   {len(fam):,} families   "
          f"{len(cands):,} anchors that could take a sibling", flush=True)
    cands = cands[:args.n] if args.n else cands
    print(f"inventing {len(cands):,} siblings", flush=True)

    stats, made = Counter(), []
    sem = asyncio.Semaphore(args.concurrency)
    sigs = {tool_signature(t) for t in tools}

    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}"}) as session:
        async def one(anchor, key):
            async with sem:
                for attempt in range(2):
                    sib, u = await invent_sibling(
                        session, anchor, key, 0.9 if not attempt else 0.6)
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
                        return
                    sigs.add(tool_signature(sib))
                    # The real test: does the pair actually chain?
                    if not find_link(anchor, [sib, anchor]):
                        stats["reject:no-link-after-all"] += 1
                        return
                    stats["invented"] += 1
                    made.append(sib)
                    return
                stats["reject:gave-up"] += 1

        await asyncio.gather(*[one(t, k) for t, k in cands])

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
        keep, by_name = [], {t["name"]: t for t in tools}
        for sib in repaired:
            faults = catalogue_faults(sib)
            if faults:
                stats[f"reject:after-repair:{faults[0][0]}"] += 1
                continue
            anchor = by_name.get(sib.get("_sibling_of"))
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
    ap.add_argument("--show", type=int, default=4)
    args = ap.parse_args()
    try:
        asyncio.run(run(args))
    except FatalAPIError as e:
        raise SystemExit(f"ABORTED: {e}")


if __name__ == "__main__":
    main()
