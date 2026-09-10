"""Propose a `returns` contract for tools that were never observed returning.

The returns map is built by watching real payloads: `return_fields` walks the
`tool_result` leaves and describes each (tool, field) it finds. glaive splits
its tool vocabulary almost exactly in half -- 445 of 890 tools never get a
result anywhere in the raw data -- so those tools have no observed payload, get
no spec, and are skipped when generating answers. The gap is self-sustaining:
no result, no spec, no answer, still no result.

For these the name, Danish description and parameter list are enough to say
what the tool returns. `check_bus_schedule` returns departure times and a
status; `validate_password_strength` returns a verdict and a score.

This INVENTS a contract rather than observing one, which is a real cost: for
these tools nothing in the source says what they actually return. It buys a
documented catalogue and an answer turn for half the vocabulary. Where genuine
multi-turn data exists (ToolMind's APIGen-MT and tau-train files carry results
for 85% and 80% of calls), taking it is strictly better than inventing here.
"""
import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

MODEL = "google/gemini-2.5-flash-lite"
URL = "https://openrouter.ai/api/v1/chat/completions"


def _key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for p in (Path.home() / "or", Path.home() / ".or"):
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("no OpenRouter key (~/or)")


SYS = """Du beskriver, hvad et værktøjs-API returnerer.

Du får et værktøj: navn, dansk beskrivelse og hvilke parametre det tager. Du
skal foreslå, hvilke felter værktøjet returnerer, når det kaldes.

For hvert felt: et engelsk feltnavn i snake_case, og en kort dansk beskrivelse
af hvad feltet indeholder.

Regler:
- 1 til 5 felter. Kun det værktøjet faktisk giver tilbage.
- Returnér RESULTATET, ikke inputtet. Et værktøj, der opretter en bruger, giver
  et bruger-id og en status tilbage -- ikke brugernavnet det fik ind.
- Feltnavne er engelske og i snake_case. Beskrivelser er på dansk.
- Skriv beskrivelsen som en feltbeskrivelse: "Antal kopper kaffe tilbage",
  ikke "Dette felt indeholder antallet af kopper".
- Er værktøjet en handling uden data at give tilbage, så returnér en status og
  en kvitteringsbesked.

Eksempel for get_lyrics(song, artist):
  lyrics    -> "Sangteksten"
  title     -> "Sangens titel"
  artist    -> "Kunstnerens navn"
"""

SCHEMA = {"type": "object", "properties": {"vaerktoejer": {
    "type": "array", "items": {"type": "object", "properties": {
        "navn": {"type": "string"},
        "felter": {"type": "array", "items": {"type": "object", "properties": {
            "felt": {"type": "string"}, "beskrivelse": {"type": "string"}},
            "required": ["felt", "beskrivelse"],
            "additionalProperties": False}}},
        "required": ["navn", "felter"], "additionalProperties": False}}},
    "required": ["vaerktoejer"], "additionalProperties": False}

SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")


def _danish(text: str) -> bool:
    if len(text.split()) < 3:
        return True                     # too short to judge
    try:
        from langdetect import DetectorFactory, detect_langs
        DetectorFactory.seed = 0
        return not any(x.lang == "en" and x.prob > 0.9
                       for x in detect_langs(text))
    except ImportError:
        return True


def gate(tool, fields, params):
    """Why each check exists is in the reason string; None if clean."""
    if not fields:
        return "no-fields"
    if len(fields) > 5:
        return "too-many-fields"
    names = [f.get("felt", "") for f in fields]
    if len(set(names)) != len(names):
        return "duplicate-fields"
    for n in names:
        if not SNAKE.match(n or ""):
            return f"field-not-snake_case:{n}"
    # The failure mode this pass invites: describing the INPUT as the output.
    # A tool that takes (song, artist) "returning" song and artist has
    # documented nothing and would train the model to echo its own call back.
    if params and set(names) <= set(params):
        return "returns-only-echoes-parameters"
    for f in fields:
        d = f.get("beskrivelse", "")
        if not d.strip():
            return "empty-description"
        if not _danish(d):
            return f"description-not-danish:{d[:40]}"
    return None


# Planted defects, one per check that can fire. A gate that never fires is
# indistinguishable from clean data.
CONTROLS = [
    ("t", [], ["a"], "no-fields"),
    ("t", [{"felt": f"f{i}", "beskrivelse": "Beskrivelse af feltet"}
           for i in range(6)], ["a"], "too-many-fields"),
    ("t", [{"felt": "a", "beskrivelse": "En ting"},
           {"felt": "a", "beskrivelse": "En ting"}], ["x"], "duplicate-fields"),
    ("t", [{"felt": "Not Snake", "beskrivelse": "En ting"}], ["x"],
     "field-not-snake_case"),
    ("get_lyrics", [{"felt": "song", "beskrivelse": "Sangens titel"},
                    {"felt": "artist", "beskrivelse": "Kunstnerens navn"}],
     ["song", "artist"], "returns-only-echoes-parameters"),
    ("t", [{"felt": "a", "beskrivelse": ""}], ["x"], "empty-description"),
    ("t", [{"felt": "a", "beskrivelse": "The name of the artist to look up"}],
     ["x"], "description-not-danish"),
]

CLEAN = [
    ("get_lyrics", [{"felt": "lyrics", "beskrivelse": "Sangteksten"},
                    {"felt": "title", "beskrivelse": "Sangens titel"}],
     ["song", "artist"]),
    ("create_user", [{"felt": "user_id", "beskrivelse": "Brugerens id"},
                     {"felt": "status", "beskrivelse": "Statuskode for kaldet"}],
     ["name", "email"]),
]


def check_controls():
    for tool, fields, params, want in CONTROLS:
        got = gate(tool, fields, params)
        if got is None or not got.startswith(want):
            raise SystemExit(f"control {want!r} -> {got!r}")
    for tool, fields, params in CLEAN:
        why = gate(tool, fields, params)
        if why is not None:
            raise SystemExit(f"gate rejects a clean proposal: {why}")
    print(f"gate: {len(CONTROLS)} planted defects caught, "
          f"{len(CLEAN)} clean proposals pass", flush=True)


def signature(fn) -> str:
    """A function's identity: name plus its parameter property names.

    Keyed on the SIGNATURE, not the name. 379 of 875 names carry more than one
    parameter schema -- `search_movies` carries 63 -- so proposing one contract
    per name means proposing it for 63 unrelated functions at once. That is the
    same identity bug that gave `search_quotes` an AAPL stock ticker.
    """
    props = ((fn.get("parameters") or {}).get("properties") or {})
    return f"{fn.get('name')}({','.join(sorted(props))})"


def missing_tools(path: Path, include_uncalled: bool = True):
    """EVERY catalogue signature carrying no returns block, with its Danish spec.

    Includes catalogue PADDING by default, though a contract is normally
    derived from an observed payload and a tool nothing calls has none.
    Skipping them makes the `returns` block itself identify the tool to call.

    How badly depends on how much of the tool vocabulary is reused. Glaive
    reuses names heavily, so nearly every padding tool is called in some other
    row and picks up a contract there -- on v9 the called tool is the only one
    in its catalogue carrying `returns` in 1.7% of train rows. ToolACE has a
    long tail called nowhere, 382 of 652 signatures, and there the called tool
    was the ONLY one with a block in 66.7% of rows: "pick the tool formatted
    with returns" solved two thirds of the corpus without reading a name or a
    description. Covering padding put that at 1.3%, in line with v9.

    Proposing for padding does not introduce a subtler version of the same
    tell. Invented contracts run thinner than observed ones (mean 1.69 return
    fields against 2.46), but the called tool holds the most fields in its
    catalogue in 21.2% of rows -- 10.9% counting ties as losses -- against
    ~16.7% for a blind guess among six.

    `--called-only` restores the old behaviour for a corpus whose vocabulary
    is reused enough not to need this, where it is just spend.
    """
    have, want = set(), {}
    called = Counter()
    for line in path.open():
        if not line.strip():
            continue
        da = (json.loads(line).get("da") or {})
        specs = {}
        for t in da.get("tools", []) or []:
            fn = t.get("function") if isinstance(t, dict) else None
            if not fn or not fn.get("name"):
                continue
            specs[fn["name"]] = fn
            sig = signature(fn)
            if fn.get("returns"):
                have.add(sig)
            else:
                want.setdefault(sig, fn)
        for m in da.get("conversations", []) or []:
            for tc in (m.get("tool_calls") or []):
                n = (tc.get("function") or {}).get("name")
                if n and n in specs:
                    called[signature(specs[n])] += 1
    return [(sig, f, called[sig]) for sig, f in want.items()
            if sig not in have and (include_uncalled or called[sig])], called


async def propose(session, chunk, tries=3):
    shown = []
    for _sig, fn, _ in chunk:
        props = ((fn.get("parameters") or {}).get("properties") or {})
        shown.append({"navn": fn.get("name"),
                      "beskrivelse": fn.get("description"),
                      "parametre": list(props)})
    body = {"model": MODEL, "temperature": 0.3, "max_tokens": 4000,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": json.dumps(
                             shown, ensure_ascii=False, indent=1)}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "returns", "strict": True, "schema": SCHEMA}}}
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                out = json.loads(
                    d["choices"][0]["message"]["content"])["vaerktoejer"]
                # POSITIONAL. Matching on the returned `navn` broke the moment
                # chunks became signatures: the model echoes the tool name, not
                # the signature string, so every lookup missed and the whole
                # batch came back as no-proposal.
                if len(out) == len(chunk):
                    return {sig: x.get("felter") or []
                            for (sig, _fn, _n), x in zip(chunk, out)}
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return {}


async def main_async(args):
    import aiohttp
    check_controls()
    tools, called = missing_tools(args.src / "translated.jsonl",
                                  not args.called_only)
    tools.sort(key=lambda x: -x[2])
    scope = "called" if args.called_only else "catalogue"
    print(f"{len(tools):,} {scope} tools carry no returns block "
          f"({sum(t[2] for t in tools):,} calls, "
          f"{sum(1 for t in tools if not t[2]):,} never called)", flush=True)
    if args.n:
        tools = tools[:args.n]
        print(f"smoke: the {len(tools)} most-called of them", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    have = {}
    if args.out.exists():
        for line in args.out.open():
            rec = json.loads(line)
            have.setdefault(rec.get("signature") or rec["tool"], []).append(rec)
    todo = [t for t in tools if t[0] not in have]
    print(f"{len(have):,} cached, {len(todo):,} to propose", flush=True)
    if args.dry_run:
        return

    chunks = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    reasons = Counter()
    kept = []
    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}",
                     "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=300)) as s:
        sem = asyncio.Semaphore(args.concurrency)

        async def propose_split(chunk):
            """Halve on a count mismatch instead of losing the batch.

            A reply with the wrong number of items -- truncation, or the model
            collapsing `get_news(category)` and `get_news(category,country)`
            into one because they share a name -- used to discard all twelve
            entries as no-proposal. Splitting isolates the offender.
            """
            got = await propose(s, chunk)
            if got:
                return got
            if len(chunk) == 1:
                return {}
            mid = len(chunk) // 2
            left = await propose_split(chunk[:mid])
            right = await propose_split(chunk[mid:])
            return {**left, **right}

        async def run(chunk):
            async with sem:
                got = await propose_split(chunk)
            out = []
            for name, fn, n in chunk:
                fields = got.get(name)
                if fields is None:
                    reasons["no-proposal"] += 1
                    continue
                params = list(((fn.get("parameters") or {})
                               .get("properties") or {}))
                why = gate(name, fields, params)
                if why:
                    reasons[why.split(":")[0]] += 1
                    continue
                out.append((name, fn, n, fields))
            return out
        for res in await asyncio.gather(*[run(c) for c in chunks]):
            kept.extend(res)

    with args.out.open("a", buffering=1) as fh:
        for sig, fn, n, fields in kept:
            fh.write(json.dumps({"tool": fn.get("name"), "signature": sig,
                                 "calls": n, "felter": fields},
                                ensure_ascii=False) + "\n")
    if args.merge_into:
        # MERGING IS PART OF THE PASS, not a shell step. Proposals must never
        # shadow observation-derived contracts: a signature whose fields fell
        # below the majority threshold still has real evidence behind it, and
        # replacing that with a guess scored 2,005 observed payloads at F1 25.4%
        # where observation-derived contracts score 97.4%.
        from translate_toolmind_da import _return_key, RETURN_SEP
        have, recs = set(), []
        if args.merge_into.exists():
            for line in args.merge_into.open():
                if line.strip():
                    r = json.loads(line)
                    have.add(r["k"])
                    recs.append(r)
        # observed = signatures with at least one OBSERVATION-derived key.
        # Presence alone would include proposals merged on an earlier run.
        observed = {tuple(r["k"].split(RETURN_SEP)[:2]) for r in recs
                    if r.get("src", "observed") == "observed"}
        added = skipped = 0
        with args.merge_into.open("a", buffering=1) as fh:
            for sig, fn, _n, fields in kept:
                name = sig.split("(")[0]
                params = sig[len(name) + 1:-1]
                if (name, params) in observed:
                    skipped += 1        # we have real evidence for this one
                    continue
                for f in fields:
                    k = _return_key(name, f["felt"], params)
                    if k in have:
                        continue
                    have.add(k)
                    added += 1
                    fh.write(json.dumps({"k": k, "da": f["beskrivelse"],
                                         "src": "proposed"},
                                        ensure_ascii=False) + "\n")
        print(f"merged into {args.merge_into}: +{added:,} keys, "
              f"{skipped:,} signatures skipped as already observed", flush=True)

    print(f"\nproposed for {len(kept):,}/{len(todo):,} tools", flush=True)
    if reasons:
        print("rejected:")
        for w, c in reasons.most_common():
            print(f"  {w:<34} {c:,}")

    for name, fn, n, fields in kept[:args.show]:
        props = list(((fn.get("parameters") or {}).get("properties") or {}))
        print("-" * 74)
        print(f"  {name}  ({n} calls)")
        print(f"    beskrivelse : {fn.get('description')}")
        print(f"    parametre   : {props}")
        for f in fields:
            print(f"      {f['felt']:<24} {f['beskrivelse']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("scratch/toolmind_da_v3"))
    ap.add_argument("--out", type=Path,
                    default=Path("scratch/toolmind_da_v3/proposed_returns.jsonl"))
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--called-only", action="store_true",
                    help="skip catalogue PADDING that nothing calls. Cheaper, "
                         "but then carrying a `returns` block identifies the "
                         "tool to call -- 66.7% of ToolACE rows were solvable "
                         "on that cue alone")
    ap.add_argument("--merge-into", type=Path, default=None,
                    help="returns_map.jsonl to append accepted proposals to, "
                         "skipping signatures that already have observed keys")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
