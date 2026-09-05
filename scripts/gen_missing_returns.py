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


def missing_tools(path: Path):
    """Called tools carrying no returns block, with their Danish specs."""
    have, want = {}, {}
    called = Counter()
    for line in path.open():
        if not line.strip():
            continue
        da = (json.loads(line).get("da") or {})
        for t in da.get("tools", []) or []:
            fn = t.get("function") if isinstance(t, dict) else None
            if not fn or not fn.get("name"):
                continue
            if fn.get("returns"):
                have[fn["name"]] = True
            else:
                want.setdefault(fn["name"], fn)
        for m in da.get("conversations", []) or []:
            for tc in (m.get("tool_calls") or []):
                n = (tc.get("function") or {}).get("name")
                if n:
                    called[n] += 1
    return [(n, f, called[n]) for n, f in want.items()
            if n not in have and called[n]], called


async def propose(session, chunk, tries=3):
    shown = []
    for name, fn, _ in chunk:
        props = ((fn.get("parameters") or {}).get("properties") or {})
        shown.append({"navn": name,
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
                return {x["navn"]: x["felter"] for x in out}
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return {}


async def main_async(args):
    import aiohttp
    check_controls()
    tools, called = missing_tools(args.src / "translated.jsonl")
    tools.sort(key=lambda x: -x[2])
    print(f"{len(tools):,} called tools carry no returns block "
          f"({sum(t[2] for t in tools):,} calls)", flush=True)
    if args.n:
        tools = tools[:args.n]
        print(f"smoke: the {len(tools)} most-called of them", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    have = {}
    if args.out.exists():
        for line in args.out.open():
            rec = json.loads(line)
            have.setdefault(rec["tool"], []).append(rec)
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

        async def run(chunk):
            async with sem:
                got = await propose(s, chunk)
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
        for name, _fn, n, fields in kept:
            fh.write(json.dumps({"tool": name, "calls": n, "felter": fields},
                                ensure_ascii=False) + "\n")
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
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
