"""Sample a value bank for distractor fields whose shape cannot be generated.

`toolmind_distractor_values.generate` covers everything with inferable form --
integers, decimals, dates, timestamps, id-like codes, number+unit, and closed
token sets. What it returns None for is free text: `author`, `meal_breakdown`,
`content_summary`. Those have no shape to rescale, and the corpus is no help
either -- of 8,325 observed string payload fields only 20 carry 20 or more
distinct values, the largest being `message` at 323. There is nothing to
sample from.

So they are sampled from the model instead, ONCE PER FIELD rather than once
per row: ~200 values for a field costs about a thousand output tokens, and
even two thousand such fields comes in under a dollar. Per-row generation
would cost four orders of magnitude more for values nobody reads closely.

The bank is a POOL, and a pool is exactly what the whole exercise rejects for
per-schema exemplars -- so it has to be big enough that recurrence is rarer
than the answer field's own recurrence. `--min-distinct` enforces that and the
gate rejects a bank that came back narrow; a field that cannot be filled is
better dropped than shipped as a constant.

    uv run python scripts/gen_distractor_value_banks.py \
        --map scratch/distractors/map.jsonl \
        --out scratch/distractors/banks.jsonl --per-field 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_return_distractors import MODEL, URL, _key  # noqa: E402
from toolmind_distractor_values import FREE, classify  # noqa: E402

SYS = """Du laver en liste af realistiske vaerdier til ét felt i et vaerktoejs-API.

Du faar feltets navn, dets danske beskrivelse og nogle eksempler paa formen.
Returnér mange forskellige vaerdier, som feltet realistisk kunne have.

Regler:
- Vaerdierne skal passe til feltets betydning og ligne eksemplerne i form.
- De skal vaere INDBYRDES FORSKELLIGE. En gentagelse er spildt plads.
- Spaend bredt: hvis feltet er en kilde, saa naevn baade store og smaa; hvis
  det er en tekst, saa varier laengde og indhold.
- Dansk, medmindre eksemplerne klart er noget andet (navne, koder, titler).
- Ingen nummerering, ingen forklaring, ingen pladsholdere.
"""

SCHEMA = {"type": "object", "properties": {
    "vaerdier": {"type": "array", "items": {"type": "string"}}},
    "required": ["vaerdier"], "additionalProperties": False}


def gate(values, exemplars, want):
    """Why each check exists is in the reason string; None if clean."""
    vals = [str(v).strip() for v in (values or []) if str(v).strip()]
    if not vals:
        return "empty-bank"
    uniq = list(dict.fromkeys(vals))
    if len(uniq) < want // 4:
        return f"too-few-distinct:{len(uniq)}/{want}"
    # A bank that drifts off-shape is worse than no bank: the distractor stops
    # resembling the field it is supposed to be confusable with.
    if exemplars:
        want_kind = classify(exemplars)
        got_kind = classify(uniq[:8])
        if want_kind != FREE and got_kind != want_kind:
            return f"shape-drift:{got_kind}!={want_kind}"
    longest = max(len(v) for v in uniq)
    if longest > 400:
        return f"value-too-long:{longest}"
    return None


# Exemplars are FREE-shaped throughout, because that is the only kind this
# pass ever sees -- `free_fields` selects on exactly that, so the shape-drift
# branch is skipped on the real path. The first cut used `["Ritzau"]`, which
# classifies as ENUM, so drift fired before the check under test and the
# control was measuring a different rule than the one it named.
_FREE_EX = ["Mahatma Gandhi", "Søren Kierkegaard"]
CONTROLS = [
    ([], _FREE_EX, 200, "empty-bank"),
    (["a"] * 50, _FREE_EX, 200, "too-few-distinct"),
    ([f"navn nummer {i}" for i in range(80)] + ["x" * 500], _FREE_EX, 200,
     "value-too-long"),
    ([f"v{i}" for i in range(80)], ["kvadratmeter", "hektar"], 200,
     "shape-drift"),
]


def check_controls():
    for vals, exs, want, why in CONTROLS:
        got = gate(vals, exs, want)
        if got is None or not got.startswith(why):
            raise SystemExit(f"control {why!r} -> {got!r}")
    ok = gate([f"kilde nummer {i}" for i in range(120)], _FREE_EX, 200)
    if ok is not None:
        raise SystemExit(f"gate rejects a clean bank: {ok}")
    print(f"gate: {len(CONTROLS)} planted defects caught, 1 clean bank passes",
          flush=True)


def draw_counts(rows_path: Path, dmap):
    """How many payloads will actually draw from each bank.

    Sizing every bank alike wastes almost all of the work: the median bank is
    drawn TWICE, 86% are drawn ten times or fewer, and only ~20 exceed a
    hundred. A flat 200 is ~30x oversized for the median and is also what
    padded `bmi_category` -- a genuinely six-member concept -- into invented
    Danish.
    """
    from inject_distractors import DEC, SEP, signature
    draws = Counter()
    for line in rows_path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        msgs = r["messages"]
        head = msgs[0]["content"]
        if SEP not in head:
            continue
        try:
            cat, _ = DEC.raw_decode(head.split(SEP, 1)[1].lstrip())
        except Exception:
            continue
        by = {t["name"]: t for t in cat if "name" in t}
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
            fn = by.get((call or {}).get("name"))
            if not fn:
                continue
            for f in dmap.get(signature(fn)) or []:
                if classify(f.get("eksempler")) == FREE:
                    draws[(f["felt"],
                           f["beskrivelse"].strip().lower())] += 1
    return draws


def size_for(draws_n, lo=24, hi=120, mult=3):
    """Bank size from demand. Floor keeps a rarely-drawn field from becoming a
    two-value pool; ceiling keeps a heavily-drawn one from being padded past
    what the concept can bear."""
    return max(lo, min(hi, mult * max(1, draws_n)))


def free_fields(map_path: Path):
    """Distractor fields the generator cannot fill, deduped by (field, shape).

    Deduped because banks are per FIELD MEANING, not per tool: `source` on
    twelve news tools wants one bank, not twelve. Keyed on the field name plus
    its Danish description so two unrelated `status` fields do not share.
    """
    seen = {}
    for line in map_path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        for f in r["fields"]:
            if classify(f.get("eksempler")) != FREE:
                continue
            k = (f["felt"], f["beskrivelse"].strip().lower())
            seen.setdefault(k, {"felt": f["felt"],
                                "beskrivelse": f["beskrivelse"],
                                "eksempler": f.get("eksempler") or [],
                                "tools": []})
            seen[k]["tools"].append(r["name"])
    return list(seen.values())


async def fetch(session, spec, want, tries=3):
    body = {"model": MODEL, "temperature": 1.0, "max_tokens": 12000,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": json.dumps(
                             {"felt": spec["felt"],
                              "beskrivelse": spec["beskrivelse"],
                              "eksempler": spec["eksempler"],
                              "antal": want}, ensure_ascii=False)}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "bank", "strict": True, "schema": SCHEMA}}}
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                vals = json.loads(
                    d["choices"][0]["message"]["content"])["vaerdier"]
                return vals, (d.get("usage") or {})
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return None, {}


async def main_async(args):
    import aiohttp
    check_controls()
    specs = free_fields(args.map)
    print(f"{len(specs):,} distinct free-text distractor fields "
          f"(from {sum(len(s['tools']) for s in specs):,} tool-field pairs)",
          flush=True)
    if args.n:
        specs = specs[:args.n]
        print(f"SMOKE: {len(specs)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists() and not args.overwrite:
        for line in args.out.open():
            if line.strip():
                r = json.loads(line)
                done.add((r["felt"], r["beskrivelse"].strip().lower()))
    todo = [s for s in specs
            if (s["felt"], s["beskrivelse"].strip().lower()) not in done]

    if args.rows:
        dmap = {}
        for line in args.map.open():
            if line.strip():
                r = json.loads(line)
                dmap[r["sig"]] = r["fields"]
        draws = draw_counts(args.rows, dmap)
        for sp in todo:
            k = (sp["felt"], sp["beskrivelse"].strip().lower())
            sp["_want"] = size_for(draws.get(k, 0))
        undrawn = sum(1 for sp in todo
                      if draws.get((sp["felt"],
                                    sp["beskrivelse"].strip().lower()), 0) == 0)
        sizes = Counter(sp["_want"] for sp in todo)
        print(f"draw-count sizing: {dict(sorted(sizes.items()))}  "
              f"({undrawn:,} never drawn -> floor)", flush=True)
        print(f"  total values to generate: "
              f"{sum(sp['_want'] for sp in todo):,} "
              f"(flat {args.per_field} would be "
              f"{len(todo)*args.per_field:,})", flush=True)
    print(f"{len(todo):,} to fetch ({len(done):,} cached)", flush=True)

    why = Counter()
    tot_in = tot_out = 0
    sem = asyncio.Semaphore(args.concurrency)
    fh = args.out.open("a")
    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}"}) as session:
        async def one(spec):
            nonlocal tot_in, tot_out
            want = spec.get("_want", args.per_field)
            async with sem:
                vals, usage = await fetch(session, spec, want)
            tot_in += usage.get("prompt_tokens", 0)
            tot_out += usage.get("completion_tokens", 0)
            if vals is None:
                why["no-response"] += 1
                return
            bad = gate(vals, spec["eksempler"], want)
            if bad:
                why[bad.split(":")[0]] += 1
                return
            uniq = list(dict.fromkeys(str(v).strip() for v in vals
                                      if str(v).strip()))
            why["ok"] += 1
            why["_values"] += len(uniq)
            fh.write(json.dumps({"felt": spec["felt"],
                                 "beskrivelse": spec["beskrivelse"],
                                 "n": len(uniq), "vaerdier": uniq},
                                ensure_ascii=False) + "\n")
            fh.flush()

        for i in range(0, len(todo), args.concurrency):
            await asyncio.gather(*(one(s)
                                   for s in todo[i:i + args.concurrency]))
            print(f"  {sum(v for k, v in why.items() if not k.startswith('_')):,}"
                  f"/{len(todo):,}  ok={why['ok']:,}", flush=True)
    fh.close()
    print(f"\n{ {k: v for k, v in why.items()} }")
    if why["ok"]:
        print(f"mean bank size: {why['_values'] / why['ok']:.0f} values")
    cost = tot_in / 1e6 * 0.10 + tot_out / 1e6 * 0.40
    print(f"tokens: in={tot_in:,} out={tot_out:,}   cost ${cost:.4f}",
          flush=True)
    if todo:
        print(f"per field: ${cost / len(todo):.5f}  ->  all {len(specs):,}: "
              f"${cost / len(todo) * len(specs):.2f}", flush=True)
    print(f"-> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-field", type=int, default=60,
                    help="flat size; overridden per field when --rows given")
    ap.add_argument("--rows", type=Path, default=None,
                    help="rendered rows -- size each bank by how often it is "
                         "actually drawn from")
    ap.add_argument("--n", type=int, default=0, help="smoke: first N fields")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
