"""Propose DISTRACTOR return fields, so copying the payload stops working.

WHY. Measured on v8 at tooldev step-11328, the answer turn is not
gradient-starved -- it is 28% of trained tokens and 71% of the loss. But
inside it the payload values are the CHEAPEST tokens (CE 0.020 against 0.197
for the Danish around them): they are COPIED, not selected, because the value
appeared verbatim in the tool_result a few tokens earlier.

Copying is a perfect policy because the data never punishes it. Of 20,566
train answer turns carrying a number:

    87.9%  the payload holds exactly ONE number -- nothing to discriminate
    10.6%  several, and the gold answer cites them ALL -- no choice made
     1.2%  several, and gold cites only SOME -- a real choice

And the thinness is in the SCHEMA, not the fill: 78.0% of tools document a
single `returns` field and payloads mirror them at a ratio of 1.04. So the fix
belongs here, at the contract, not in a post-hoc sprinkle of junk keys.

WHAT MAKES A DISTRACTOR WORK. Three properties, none of which "a random key"
has:

  plausible sibling   `temperature_kelvin` on a dice roll is ignored on sight.
                      `floor: 4` beside `cups_left: 8` is what actually
                      defeats the model, because it belongs there.
  same type/magnitude a 4 against an 8 forces a choice; a 4 against
                      1725036492 does not.
  strings too         `capacity: 10` -> "10 meter højt" had only ONE number in
                      the payload, so numeric distractors alone miss it.

THIS SCRIPT ONLY PROPOSES THE CONTRACT -- one call per DISTINCT TOOL (867),
not per row (17k). The VALUES follow from it downstream.

ORDER: STAGE 2.5, BEFORE `--annotate` AND BEFORE THE RENDERER.

Not after the answers, where the symbolized twins sit. The twins only rename
keys the payload already has, so they cannot invalidate an answer. A
distractor is different: `gen_tool_answer_turns` SYNTHESISES the payload from
the `returns` block ("invent a result that conforms to the tool's returns
schema") and gates it with `result-extra-fields` / `result-missing-fields`
against `_spec_fields(returns)`. So

  - a field injected into a payload after stage 5 is rejected as an extra
    field, because the contract never declared it; and
  - even if it were let through, the answer was written without ever seeing
    the distractor, so nothing verified that the answer declines to cite it --
    which is the entire property we are buying.

Declaring it here instead means the payload comes back carrying it, and the
answer is gated against the enriched contract.

THIS COSTS ANSWER REGENERATION. The answer cache keys on
(tool, arguments, question, fingerprint of `returns`) precisely because the
schema is an input to the payload, so every touched signature re-buys its
answers at stage 5. That is the price of the ordering, and it is the correct
price to pay.

    # smoke: proposal quality only, against the published corpus
    uv run python scripts/gen_return_distractors.py \
        --corpus jensjepsen/danish-tool-dialogues-v8 \
        --n 15 --out scratch/distractors/smoke.jsonl

    # real: pipeline dir, merged into the returns map before stage 3
    uv run python scripts/gen_return_distractors.py --src $OUT \
        --out $OUT/proposed_distractors.jsonl \
        --merge-into $OUT/returns_map.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

MODEL = "google/gemini-2.5-flash-lite"
URL = "https://openrouter.ai/api/v1/chat/completions"
SEP = "Værktøjer:"
SYM = re.compile(r"^[a-z][0-9a-f]{4}$")
SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")
# Infrastructure telemetry, not domain content. Banned in the prompt too, but
# the prompt alone does not hold: at the temperature the retries escalate to,
# `calculate_area` came back with `calculation_time_seconds` anyway. A rule
# that matters needs a gate, not a sentence.
PLACEHOLDER = re.compile(r"^(x+|n/?a|null|none|string|value|tekst|\?+|"
                         r"eksempel|example|todo|\.\.\.)$", re.I)
NUMERIC = re.compile(r"^-?\d+([.,]\d+)?$")
INSTRUMENT = re.compile(
    r"^(calculation|computation|processing|execution|analysis|response|"
    r"elapsed|runtime)_?time|_time_(ms|seconds|sec)$|^(request|trace|"
    r"correlation)_id$|^api_version$|^status_code$")


def _key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for p in (Path.home() / "or", Path.home() / ".or"):
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("no OpenRouter key (~/or)")


SYS = """Du udvider et værktøjs-API med ekstra returfelter.

Du får et værktøj: navn, dansk beskrivelse, parametre og de felter det ALLEREDE
returnerer. Du skal foreslå 1-3 EKSTRA felter, som det samme værktøj
realistisk også ville returnere.

De ekstra felter er BIFELTER. De er ikke svaret på brugerens spørgsmål -- de
er den slags ledsagende oplysninger, et rigtigt API sender med.

Regler:
- Feltet skal høre naturligt til netop dette værktøj. En kaffemaskine
  returnerer "etage" og "sidste_service"; den returnerer ikke "aktiekurs".
- Det må IKKE være en omskrivning af et felt, værktøjet allerede returnerer.
- Det må IKKE bare gentage en parameter, værktøjet fik ind.
- Mindst ét felt skal have samme type som et eksisterende returfelt, så de kan
  forveksles. Returnerer værktøjet et tal, så giv mindst ét tal mere.
- Feltnavne er engelske og i snake_case. Beskrivelser er på dansk.
- Skriv beskrivelsen som en feltbeskrivelse: "Etagen maskinen står på", ikke
  "Dette felt indeholder etagen".
- "type" er en af: number, integer, string.
- IKKE teknisk instrumentering: "beregningstid_ms", "analysis_time",
  "request_id", "api_version". De hører til ethvert værktøj og siger intet om
  netop dette. Feltet skal handle om SAGEN -- forsendelsen, skatten, filmen.
- IKKE et felt, der gentager en parameter værktøjet fik ind.

- "eksempler" er 8 FORSKELLIGE realistiske vaerdier feltet kunne have. De
  fastlaegger feltets FORM og OMRAADE -- den endelige vaerdi i hver raekke
  udledes af dem. De skal se ud som noget et rigtigt API ville sende: datoer
  som "2026-03-14", enheder som "kvadratmeter", kilder som "Ritzau". Tal
  skrives som tal. Ingen pladsholdere som "xxx" eller "string".
- De 8 vaerdier skal spaende BREDT i stoerrelse og form: er feltet et tal, saa
  vis baade smaa og store; er det en dato, saa vis flere aar. En smal
  eksempelmaengde giver et smalt felt, og et smalt felt er nemt at genkende og
  ignorere.

Eksempel -- coffee_machine_status(floor), returnerer allerede cups_left:
  floor         integer  "Etagen maskinen står på"    eksempler: 2, 3, 5, 11
  last_service  string   "Dato for seneste service"
                         eksempler: 2026-02-11, 2025-11-30, 2026-01-04, 2025-08-22
"""

SCHEMA = {"type": "object", "properties": {"vaerktoejer": {
    "type": "array", "items": {"type": "object", "properties": {
        "navn": {"type": "string"},
        "felter": {"type": "array", "items": {"type": "object", "properties": {
            "felt": {"type": "string"},
            "type": {"type": "string", "enum": ["number", "integer", "string"]},
            "beskrivelse": {"type": "string"},
            "eksempler": {"type": "array",
                          "items": {"type": "string"}}},
            "required": ["felt", "type", "beskrivelse", "eksempler"],
            "additionalProperties": False}}},
        "required": ["navn", "felter"], "additionalProperties": False}}},
    "required": ["vaerktoejer"], "additionalProperties": False}


def _danish(text: str) -> bool:
    if len(text.split()) < 3:
        return True                      # too short to judge
    try:
        from langdetect import DetectorFactory, detect_langs
        DetectorFactory.seed = 0
        return not any(x.lang == "en" and x.prob > 0.9
                       for x in detect_langs(text))
    except ImportError:
        return True


def gate(fields, existing, params):
    """Why each check exists is in the reason string; None if clean."""
    if not fields:
        return "no-fields"
    if len(fields) > 3:
        return "too-many-fields"
    names = [f.get("felt", "") for f in fields]
    if len(set(names)) != len(names):
        return "duplicate-fields"
    for n in names:
        if not SNAKE.match(n or ""):
            return f"field-not-snake_case:{n}"
    # Clashes are handled by `drop_clashes` BEFORE this gate, not here. The
    # model restates the existing contract before extending it -- 12 of 15
    # smoke proposals led with the field the tool already returns -- so
    # rejecting the proposal over a duplicate throws away the good fields
    # attached to it. Left as an assertion: if a clash reaches the gate, the
    # caller skipped the filter.
    clash = set(names) & set(existing)
    if clash:
        return f"clashes-with-existing:{sorted(clash)[0]}"
    # Telemetry is not a distractor. It attaches to every tool alike, so the
    # model can learn to skip the whole shape without ever reading a field.
    for n in names:
        if INSTRUMENT.search(n or ""):
            return f"instrumentation-not-domain:{n}"
    # ANY parameter echoed back, not just a proposal made entirely of them.
    # The smoke produced `symbol` for get_stock_price(symbol) and `category`
    # for get_news(category): a real API does return those, but as a
    # DISTRACTOR it is the worst possible kind -- the value the user just
    # supplied, which the answer gate already penalises as an echo. A tool
    # contract may legitimately restate its input; a distractor may not.
    echo = set(names) & set(params or ())
    if echo:
        return f"echoes-parameter:{sorted(echo)[0]}"
    # The whole point is confusability. If nothing shares a type with an
    # existing field, the model can separate them on type alone and the
    # distractor costs tokens without buying pressure.
    if existing:
        et = {t for t in existing.values() if t}
        ft = {f.get("type") for f in fields}
        num = {"number", "integer"}
        if et and not (ft & et) and not (ft & num and et & num):
            return f"no-type-overlap:{sorted(ft)}/{sorted(et)}"
    for f in fields:
        d = f.get("beskrivelse", "")
        if not d.strip():
            return "empty-description"
        if not _danish(d):
            return f"description-not-danish:{d[:40]}"
        # Examples are USED, not decoration: they seed the value injected into
        # real payloads, so a placeholder would ship as data.
        #
        # And they must be a POOL, not one value. A single exemplar is
        # identical in every row of that tool, which makes the distractor the
        # constant beside a varying answer -- a fresh shortcut in place of the
        # one this whole exercise removes.
        exs = [str(x).strip() for x in (f.get("eksempler") or []) if str(x).strip()]
        if not exs:
            return "empty-examples"
        if len(set(exs)) < 3:
            return f"examples-not-varied:{exs[:3]}"
        for ex in exs:
            if PLACEHOLDER.match(ex):
                return f"placeholder-example:{ex[:30]}"
            if f.get("type") in ("number", "integer") and not NUMERIC.match(ex):
                return f"example-not-numeric:{ex[:30]}"
    return None


# Planted defects, one per check that can fire. A gate that never fires is
# indistinguishable from clean data.
F = lambda n, t="integer", d="Beskrivelse af feltet", e=None: {   # noqa: E731
    "felt": n, "type": t, "beskrivelse": d,
    "eksempler": e if e is not None else (
        ["7", "12", "31", "4"] if t != "string"
        else ["en vaerdi", "en anden", "en tredje", "en fjerde"])}
CONTROLS = [
    ([], {"a": "integer"}, ["x"], "no-fields"),
    ([F(f"f{i}") for i in range(4)], {"a": "integer"}, ["x"],
     "too-many-fields"),
    ([F("a"), F("a")], {"z": "integer"}, ["x"], "duplicate-fields"),
    ([F("Not Snake")], {"z": "integer"}, ["x"], "field-not-snake_case"),
    ([F("cups_left")], {"cups_left": "integer"}, ["floor"],
     "clashes-with-existing"),
    ([F("song"), F("artist")], {"lyrics": "string"}, ["song", "artist"],
     "echoes-parameter"),
    ([F("symbol", "string", "Aktiesymbolet"), F("high", "number", "Dagens højeste")],
     {"price": "number"}, ["symbol"], "echoes-parameter"),
    ([F("calculation_time_seconds")], {"area": "integer"}, ["shape"],
     "instrumentation-not-domain"),
    ([F("response_time_ms")], {"area": "integer"}, ["shape"],
     "instrumentation-not-domain"),
    ([F("note", "string")], {"count": "integer"}, ["x"], "no-type-overlap"),
    ([F("a", "integer", "En ting", [])], {"z": "integer"}, ["x"],
     "empty-examples"),
    ([F("a", "integer", "En ting", ["7", "7", "7", "7"])], {"z": "integer"},
     ["x"], "examples-not-varied"),
    ([F("a", "integer", "En ting", ["7", "12", "xxx", "4"])], {"z": "integer"},
     ["x"], "placeholder-example"),
    ([F("a", "integer", "En ting", ["7", "12", "syv", "4"])], {"z": "integer"},
     ["x"], "example-not-numeric"),
    ([F("a", "integer", "")], {"z": "integer"}, ["x"], "empty-description"),
    ([F("a", "integer", "The floor the machine is on")], {"z": "integer"},
     ["x"], "description-not-danish"),
]
CLEAN = [
    ([F("floor", "integer", "Etagen maskinen står på"),
      F("last_service", "string", "Dato for seneste service")],
     {"cups_left": "integer"}, ["floor_query"]),
    ([F("capacity", "integer", "Hvor mange lokalet kan rumme")],
     {"room": "string", "floor": "integer"}, ["people", "time"]),
]


def check_controls():
    for fields, existing, params, want in CONTROLS:
        got = gate(fields, existing, params)
        if got is None or not got.startswith(want):
            raise SystemExit(f"control {want!r} -> {got!r}")
    for fields, existing, params in CLEAN:
        why = gate(fields, existing, params)
        if why is not None:
            raise SystemExit(f"gate rejects a clean proposal: {why}")
    print(f"gate: {len(CONTROLS)} planted defects caught, "
          f"{len(CLEAN)} clean proposals pass", flush=True)


def drop_clashes(fields, existing):
    """Strip fields the tool already returns, keep the rest, cap at 3.

    Both filters exist because of what the model actually does: it recites the
    current contract and then extends it. `convert_currency` came back with
    all three existing fields plus `exchange_rate` -- one good distractor
    behind three duplicates, which the old gate scored as `too-many-fields`.
    """
    kept, dropped = [], 0
    for f in fields:
        if f.get("felt") in existing:
            dropped += 1
            continue
        kept.append(f)
    return kept[:3], dropped, max(0, len(kept) - 3)


def signature(fn) -> str:
    """Name plus sorted parameter names -- a NAME is not a function.

    379 of 875 glaive names carry more than one parameter schema
    (`search_movies` carries 63), so keying on the name proposes one contract
    for dozens of unrelated tools.
    """
    props = ((fn.get("parameters") or {}).get("properties") or {})
    return f"{fn.get('name')}({','.join(sorted(props))})"


def collect_tools(corpus: str, split: str = "train"):
    """Distinct called signatures across EVERY split named in `split`.

    HELD-OUT TOOLS ARE THE POINT. Reading `train` alone -- which is what the
    first v9 build did -- leaves the eval splits unenriched by construction,
    because a held-out tool appears nowhere in train. Measured on the pushed
    v9: 169 of 169 `eval_unseen_tools` signatures (100%, all 870 calls) and
    103 of 404 `eval_seen_tools` signatures had no distractor, so the split
    that exists to measure field selection on novel tools was the one split
    the change did not reach.
    """
    """Distinct called signatures, with how many payloads each one has.

    Ranked by call count so a smoke run exercises the tools that carry the
    most rows rather than an arbitrary corner of the vocabulary.
    """
    from collections import Counter
    from datasets import load_dataset
    dec = json.JSONDecoder()
    parts = [x.strip() for x in str(split).split(",") if x.strip()]
    rows = []
    for sp in parts:
        try:
            rows.extend(load_dataset(corpus, "sft", split=sp))
        except Exception as e:
            print(f"  split {sp}: unavailable ({type(e).__name__})", flush=True)
    print(f"  read {len(rows):,} rows from {len(parts)} split(s): "
          f"{', '.join(parts)}", flush=True)
    specs, called, single, obs = {}, Counter(), Counter(), {}
    for r in rows:
        msgs = r["messages"]
        head = msgs[0]["content"]
        if SEP not in head:
            continue
        try:
            cat, _ = dec.raw_decode(head.split(SEP, 1)[1].lstrip())
        except Exception:
            continue
        keys = [k for t in cat
                for k in ((t.get("parameters") or {}).get("properties") or {})]
        if keys and all(SYM.match(k) for k in keys):
            continue                     # symbolized twin: same tools, renamed
        by_name = {t["name"]: t for t in cat if "name" in t}
        for i, m in enumerate(msgs):
            if m["role"] != "tool_call":
                continue
            try:
                call = json.loads(m["content"])
            except Exception:
                continue
            fn = by_name.get(call.get("name"))
            if not fn:
                continue
            sig = signature(fn)
            specs[sig] = fn
            called[sig] += 1
            # does this tool's payload already pose a choice?
            if i + 1 < len(msgs) and msgs[i + 1]["role"] == "tool_result":
                try:
                    p = json.loads(msgs[i + 1]["content"])
                except Exception:
                    continue
                if isinstance(p, dict) and len(p) <= 1:
                    single[sig] += 1
                # OBSERVED types. A v4 `returns` leaf carries a Danish
                # description and NO type, so the declared schema cannot say
                # whether a field is a number -- 13 of 15 smoke tools had every
                # declared type as None, which silently disabled the
                # confusability check. The payloads know.
                if isinstance(p, dict):
                    for k, v in p.items():
                        if isinstance(v, bool) or v is None:
                            continue
                        t = ("integer" if isinstance(v, int) else
                             "number" if isinstance(v, float) else
                             "string" if isinstance(v, str) else None)
                        if t:
                            obs.setdefault(sig, {}).setdefault(k, t)
    for sig, fn in specs.items():
        fn["_obs"] = obs.get(sig, {})
    return [(sig, specs[sig], called[sig], single[sig])
            for sig in sorted(specs, key=lambda s: -called[s])]


def _existing(fn) -> dict:
    """Declared return fields, typed from OBSERVATION where the schema is silent."""
    props = ((fn.get("returns") or {}).get("properties") or {})
    obs = fn.get("_obs") or {}
    out = {k: (v or {}).get("type") for k, v in props.items()}
    for k, t in obs.items():
        if not out.get(k):
            out[k] = t
    return out


async def propose(session, chunk, tries=3, hints=None, temp=0.4):
    shown = []
    for _sig, fn, _c, _s in chunk:
        props = ((fn.get("parameters") or {}).get("properties") or {})
        rets = ((fn.get("returns") or {}).get("properties") or {})
        entry = {
            "navn": fn.get("name"),
            "beskrivelse": fn.get("description"),
            "parametre": list(props),
            # TYPES ARE SENT. The prompt asks for a field that shares a type
            # with an existing one; without the types it was guessing, and the
            # smoke produced string-only extras for numeric tools.
            "returnerer_allerede": [
                {"felt": k, "type": (fn.get("_obs") or {}).get(k)
                 or (v or {}).get("type") or "ukendt",
                 "beskrivelse": (v or {}).get("description", "")}
                for k, v in rets.items()] or [
                {"felt": k, "type": t, "beskrivelse": ""}
                for k, t in (fn.get("_obs") or {}).items()],
        }
        # RETRY FEEDBACK. A bare resample repeats the same miss: the model was
        # not told which constraint it broke. Naming the required type turns a
        # coin-flip into a targeted second attempt.
        if hints and hints.get(_sig):
            entry["krav"] = hints[_sig]
        shown.append(entry)
    body = {"model": MODEL, "temperature": temp, "max_tokens": 4000,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": json.dumps(
                             shown, ensure_ascii=False, indent=1)}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "distractors", "strict": True, "schema": SCHEMA}}}
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                out = json.loads(
                    d["choices"][0]["message"]["content"])["vaerktoejer"]
                usage = d.get("usage") or {}
                # POSITIONAL, not by returned name: the model echoes the tool
                # NAME while chunks are keyed by SIGNATURE, so name lookup
                # misses every time and the batch reads as no-proposal.
                if len(out) == len(chunk):
                    return ({sig: x.get("felter") or []
                             for (sig, _f, _c, _s), x in zip(chunk, out)},
                            usage)
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return {}, {}



def hint_for(reason, fn, existing):
    """Turn a gate verdict into an instruction the next attempt can act on."""
    params = list(((fn.get("parameters") or {}).get("properties") or {}))
    if reason.startswith("no-type-overlap"):
        want = [f"{k} ({t})" for k, t in existing.items() if t]
        types = sorted({t for t in existing.values() if t})
        return (f"Værktøjet returnerer allerede: {', '.join(want)}. Mindst ét "
                f"af dine felter SKAL have typen {' eller '.join(types)}, "
                f"ellers kan de ikke forveksles med hinanden.")
    if reason.startswith("no-fields") or reason.startswith("clashes"):
        return (f"Disse felter findes ALLEREDE og må ikke foreslås igen: "
                f"{', '.join(sorted(existing))}. Find andre.")
    if reason.startswith("echoes-parameter"):
        return (f"Disse navne er parametre værktøjet FÅR IND og må ikke "
                f"bruges som returfelter: {', '.join(params)}.")
    if reason.startswith("description-not-danish"):
        return "Beskrivelserne skal være på DANSK."
    if reason.startswith("too-many-fields"):
        return "Højst 3 felter."
    return None


async def main_async(args):
    import aiohttp
    check_controls()
    tools = collect_tools(args.corpus, args.split)
    # A tool with no observed returns has nothing to distract FROM: the
    # proposal becomes its whole contract, so `sort_numbers` was handed
    # `sorted_numbers` as a "distractor" -- the answer field itself. Those
    # belong to stage 2 (gen_missing_returns), which invents contracts on
    # purpose. 9 of 106 accepted proposals in the 120-tool spread were this.
    n_raw = len(tools)
    tools = [t for t in tools if _existing(t[1])]
    if len(tools) != n_raw:
        print(f"skipped {n_raw - len(tools):,} signatures with no observed "
              f"returns (stage 2's job, not this one)", flush=True)
    n_all = len(tools)
    print(f"{n_all:,} distinct called signatures "
          f"({sum(t[2] for t in tools):,} calls, "
          f"{sum(t[3] for t in tools):,} single-key payloads)", flush=True)
    if args.n:
        if args.spread:
            # EVENLY SPACED across the ranked list, not the head. The top of
            # the distribution is calculators and news feeds; a coverage
            # number measured only there does not describe the 2,867.
            step = max(1, len(tools) // args.n)
            tools = tools[::step][:args.n]
            print(f"SMOKE: {len(tools)} spread across all {n_all:,} "
                  f"(every {step}th by call count)", flush=True)
        else:
            tools = tools[:args.n]
            print(f"SMOKE: first {len(tools)} by call count", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists() and not args.overwrite:
        for line in args.out.open():
            if line.strip():
                done.add(json.loads(line)["sig"])
        print(f"resuming: {len(done):,} already proposed", flush=True)
    todo = [t for t in tools if t[0] not in done]
    chunks = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"{len(todo):,} to propose in {len(chunks):,} requests", flush=True)

    from collections import Counter
    why = Counter()
    tot_in = tot_out = 0
    sem = asyncio.Semaphore(args.concurrency)
    fh = args.out.open("a")
    headers = {"Authorization": f"Bearer {_key()}"}
    async with aiohttp.ClientSession(headers=headers) as session:
        async def one(chunk, hints, temp, failed):
            nonlocal tot_in, tot_out
            async with sem:
                got, usage = await propose(session, chunk, hints=hints,
                                           temp=temp)
            tot_in += usage.get("prompt_tokens", 0)
            tot_out += usage.get("completion_tokens", 0)
            for sig, fn, calls, single in chunk:
                fields = got.get(sig)
                ex = _existing(fn)
                if fields is None:
                    failed.append((sig, fn, calls, single, "no-proposal"))
                    continue
                fields, ndrop, nover = drop_clashes(fields, ex)
                if ndrop:
                    why["_clash-fields-dropped"] += ndrop
                if nover:
                    why["_over-3-truncated"] += nover
                bad = gate(fields, ex, list(
                    ((fn.get("parameters") or {}).get("properties") or {})))
                if bad:
                    failed.append((sig, fn, calls, single, bad))
                    continue
                why["ok"] += 1
                fh.write(json.dumps(
                    {"sig": sig, "name": fn.get("name"), "calls": calls,
                     "existing": ex, "fields": fields},
                    ensure_ascii=False) + "\n")
            fh.flush()

        # A rejection says WHICH constraint was missed, so the retry can state
        # it instead of resampling blind. Bare resampling reproduces the same
        # miss: the model was never told what was wrong.
        pending, hints = todo, {}
        for attempt in range(args.retries + 1):
            if not pending:
                break
            # Warmer each round; at the same temperature the second attempt
            # tends to return the answer that was just rejected.
            temp = 0.4 if not attempt else min(1.0, 0.4 + 0.3 * attempt)
            if attempt:
                print(f"\n  retry {attempt}: {len(pending):,} tools "
                      f"(temp={temp})", flush=True)
            failed = []
            ch = [pending[i:i + args.batch]
                  for i in range(0, len(pending), args.batch)]
            for i in range(0, len(ch), args.concurrency):
                await asyncio.gather(*(one(c, hints, temp, failed)
                                       for c in ch[i:i + args.concurrency]))
                print(f"    ok={why['ok']:,}  failed-this-round="
                      f"{len(failed):,}", flush=True)
            for _s, _f, _c, _g, reason in failed:
                why[f"a{attempt}:{reason.split(':')[0]}"] += 1
            hints = {sig: hint_for(reason, fn, _existing(fn))
                     for sig, fn, _c, _g, reason in failed}
            pending = [(s_, f_, c_, g_) for s_, f_, c_, g_, _r in failed]
        why["FINAL-no-distractor"] += len(pending)
    fh.close()
    print(f"\n{dict(why)}")
    print(f"tokens: in={tot_in:,} out={tot_out:,}")
    # gemini-2.5-flash-lite via OpenRouter, Sep 2026 list price
    cost = tot_in / 1e6 * 0.10 + tot_out / 1e6 * 0.40
    print(f"cost this run: ${cost:.4f}", flush=True)
    if todo:
        per = cost / len(todo)
        print(f"per tool: ${per:.6f}  ->  all {n_all:,} tools: "
              f"${per * n_all:.3f}", flush=True)
    print(f"-> {args.out}", flush=True)


def merge_into(path: Path, records):
    """Append distractor keys to the returns map, marked `src=distractor`.

    Marked, not anonymous: `gen_missing_returns` already distinguishes
    observation-derived contracts from invented ones and refuses to let a
    guess shadow evidence. A distractor is invented on purpose, and anything
    downstream that wants to weigh or strip them needs to be able to find
    them.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from translate_toolmind_da import _return_key
    have = set()
    if path.exists():
        for line in path.open():
            if line.strip():
                have.add(json.loads(line)["k"])
    added = 0
    with path.open("a", buffering=1) as fh:
        for rec in records:
            name = rec["sig"].split("(")[0]
            params = rec["sig"][len(name) + 1:-1]
            for f in rec["fields"]:
                k = _return_key(name, f["felt"], params)
                if k in have:
                    continue
                have.add(k)
                added += 1
                fh.write(json.dumps(
                    {"k": k, "da": f["beskrivelse"], "type": f["type"],
                     "src": "distractor"}, ensure_ascii=False) + "\n")
    print(f"merged into {path}: +{added:,} distractor keys", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    help="pipeline dir with translated.jsonl (real run)")
    ap.add_argument("--corpus", default="jensjepsen/danish-tool-dialogues-v8",
                    help="published dataset (smoke: proposal quality only)")
    ap.add_argument("--split",
                    default="train,eval_seen_tools,eval_unseen_tools",
                    help="comma-separated; MUST include the eval splits or "
                         "held-out tools get no distractors at all")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--merge-into", type=Path,
                    help="returns_map.jsonl to append to (stage 2.5)")
    ap.add_argument("--n", type=int, default=0, help="smoke: first N tools")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--retries", type=int, default=2,
                    help="extra attempts for tools that got no usable distractor")
    ap.add_argument("--spread", action="store_true",
                    help="sample evenly across the ranked list, not the head")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))
    if args.merge_into:
        merge_into(args.merge_into,
                   [json.loads(l) for l in args.out.open() if l.strip()])


if __name__ == "__main__":
    main()
