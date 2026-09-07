"""Second pass over value-lexicon entries the translator left unchanged.

54.5% of the value map (3,023 of 5,544 entries) came back identical to the
English. Most of that is correct -- `New York`, `Inception`, `J.K. Rowling`,
`Bohemian Rhapsody` are names, and `broccoli`, `Celsius`, `Fahrenheit` are the
same word in Danish. But ordinary words hide in the same bucket: `Italian`
appears in 145 calls, `chess` in a dialogue whose result says "Spillet skak",
`User` where the payload says "Bruger".

The first pass could not tell them apart because it translates a mixed list and
a cautious model leaves anything name-shaped alone. This pass asks the only
question that matters for these: is it a name, or a word? It sees just the
unchanged entries, so the judgement is not diluted by the values that already
translated cleanly.

Only entries the model CHANGES are rewritten, so a correct "leave it" costs
nothing and a name that survives the second look is confirmed rather than
merely untouched.
"""
import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate_toolmind_da import VALUE_SEP, CODE_VALUE  # noqa: E402

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


SYS = """Du får en nummereret liste af VÆRDIER fra kald til et værktøjs-API.
De blev alle sammen ladt uoversatte i første omgang. Nogle med rette, nogle
ikke.

For hver linje: skriv den danske form, hvis værdien er et almindeligt ord.
Skriv værdien UÆNDRET, hvis den er et navn eller en kode.

Uændret:
- personnavne (John Doe), stednavne (New York, Paris), firmanavne (Apple)
- titler på film, bøger og sange (Inception, Bohemian Rhapsody)
- koder, valutaer, symboler, enheder (AAPL, USD, Celsius)
- ord der staves ens på dansk (broccoli, chokolade)

Oversæt:
- almindelige navneord og tillægsord (Italian -> italiensk, chess -> skak,
  laptop -> bærbar computer, User -> Bruger, action -> action)
- kategorier og genrer, når de har en dansk form

Du får feltets navn som kontekst. Svar med præcis lige så mange linjer i samme
rækkefølge, kun værdien, uden numre."""


async def ask(session, chunk, tries=3):
    shown = [f"{slot}: {val}" for slot, val in chunk]
    body = {"model": MODEL, "temperature": 0.1, "max_tokens": 3000,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": "\n".join(
                             f"{i+1}. {x}" for i, x in enumerate(shown))}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "vaerdier", "strict": True, "schema": {
                    "type": "object",
                    "properties": {"linjer": {"type": "array",
                                              "items": {"type": "string"}}},
                    "required": ["linjer"], "additionalProperties": False}}}}
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                out = json.loads(
                    d["choices"][0]["message"]["content"])["linjer"]
                if len(out) == len(chunk):
                    return [x.strip() for x in out]
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return None


# Slots whose value is a UNIT. Translating these is not a rendering choice, it
# changes what the call means: Danish "mil" is 10 km, so miles -> mil turns a
# correct call into a wrong one against a result computed in miles. Same reason
# the palindrome and word-count tools are pinned -- the value's literal
# identity is load-bearing. Matched on the SLOT name, so it holds for units we
# have not seen.
UNIT_SLOT = re.compile(r"(^|_)(unit|units|from_unit|to_unit|measurement|"
                       r"scale|uom)$", re.I)


def suspect(new, old, slot=""):
    """Reasons to distrust a proposed change."""
    if not new or new == old:
        return "unchanged"
    if UNIT_SLOT.search(slot or ""):
        return "unit-slot-pinned"
    if CODE_VALUE.match(new):
        return "became-a-code"          # a word cannot translate into a token
    if len(new) > 3 * max(4, len(old)):
        return "too-long"               # an explanation, not a translation
    if re.search(r"[.!?]\s|\bkan\b|\bikke\b", new) and len(new.split()) > 4:
        return "looks-like-prose"
    return None


async def main_async(args):
    import aiohttp
    vm = [json.loads(l) for l in args.map.open() if l.strip()]
    same = [r for r in vm if r["k"].split(VALUE_SEP, 1)[1] == r["da"]]
    print(f"{len(vm):,} entries, {len(same):,} unchanged -> revisiting",
          flush=True)
    if args.n:
        same = same[:args.n]

    chunks = [same[i:i + args.batch] for i in range(0, len(same), args.batch)]
    sem = asyncio.Semaphore(args.concurrency)
    changed, reasons = {}, Counter()
    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}",
                     "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=300)) as s:
        async def run(chunk):
            pairs = [(r["k"].split(VALUE_SEP, 1)[0].split(".")[-1],
                      r["k"].split(VALUE_SEP, 1)[1]) for r in chunk]
            async with sem:
                out = await ask(s, pairs)
            if out is None:
                reasons["api"] += len(chunk)
                return
            for r, new in zip(chunk, out):
                old = r["k"].split(VALUE_SEP, 1)[1]
                slot = r["k"].split(VALUE_SEP, 1)[0].split(".")[-1]
                why = suspect(new, old, slot)
                if why:
                    reasons[why] += 1
                    continue
                changed[r["k"]] = new
        await asyncio.gather(*[run(c) for c in chunks])

    print(f"\nproposed changes: {len(changed):,} of {len(same):,}", flush=True)
    for w, c in reasons.most_common():
        print(f"  {w:<20} {c:,}")
    for k, v in list(changed.items())[:args.show]:
        print(f"    {k.split(VALUE_SEP,1)[1]!r:<24} -> {v!r}   "
              f"[{k.split(VALUE_SEP,1)[0]}]")

    if args.apply and changed:
        out = []
        for r in vm:
            if r["k"] in changed:
                r = {**r, "da": changed[r["k"]]}
            out.append(r)
        with args.map.open("w") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nrewrote {args.map} with {len(changed):,} updated entries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=Path,
                    default=Path("scratch/toolmind_da_v6fresh/value_map.jsonl"))
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
