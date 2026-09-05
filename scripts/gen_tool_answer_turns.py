"""Answer the dangling terminal call in every tool dialogue.

The source corpus always cuts at the last call, so 17,123 of 25,411 calls
(67.4%) have no result behind them and no answer after them. The consequence is
a lopsided corpus: reasoning-before-a-call is 78.7% of the trained tokens and
grounded answers are 3.5%. Both live in the `<|assistant|>` slot, so after a
`<|tool_result|>` the model picks between two learned registers at ~22:1 odds
and reliably picks the wrong one -- on four unseen tools it fabricated
`4 * 12 = 48` while the tool had just returned `dog_years: 28`.

This fills in the missing half-turn: for each dangling call, invent a result
that conforms to the tool's `returns` schema, then write the Danish answer that
reads it. Existing turns are never touched.

Cache is keyed on (tool, arguments, question, fingerprint of the tool's
`returns` block) and appended as it goes, so a rerun costs only what is new.
The fingerprint is in the key because the schema is an INPUT to the payload:
edit `returns` -- nest it, dedupe its synonym fields -- and every cached result
was built to a contract that no longer holds.
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

MODEL = "google/gemini-2.5-flash-lite"
URL = "https://openrouter.ai/api/v1/chat/completions"

# Meta-talk. The failure this pass exists to fix is the model narrating the
# call instead of answering ("Så det korrekte værktøjskald er..."), so an
# answer that does the same thing is worse than no answer at all.
META = re.compile(r"\b(værktøjskald|værktøjet|funktionskald|tool_call|"
                  r"json|parameter|parametre|argument(er)?|api|"
                  r"jeg (skal|vil) (nu |bare )?(kalde|formatere|generere))\b",
                  re.I)
# No sign. `2024-03-15` is a date, not two negative numbers, and reading it as
# one made every reformatted date ("15. marts 2024") look invented.
NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for p in (Path.home() / "or", Path.home() / ".or"):
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("no OpenRouter key (~/or)")


ANSWER_SYS = """Du laver træningsdata til en dansk assistent, der bruger værktøjer.

Du får et værktøj (navn, beskrivelse, parametre og hvilke felter det
returnerer), brugerens spørgsmål, og det kald assistenten har lavet.

Lav to ting:

1. "resultat": et realistisk svar fra værktøjet, som JSON-objekt. Brug PRÆCIS
   de felter værktøjet siger det returnerer -- ingen ekstra felter, ingen
   udeladte. Værdierne skal passe til kaldets argumenter og være konkrete
   (rigtige tal, rigtige navne), ikke pladsholdere som "abc" eller 0.

2. "svar": assistentens svar til brugeren, på dansk, 1-3 sætninger.

Krav til "svar":
- Det skal BRUGE tallene og navnene fra "resultat". Skriv dem ud.
- Det skal svare på brugerens spørgsmål, ikke beskrive hvad du har gjort.
- Skriv ALDRIG om værktøjer, kald, funktioner, parametre eller JSON.
- Ingen engelske ord eller sætninger.
- Regn ikke videre på tallene og find ikke på tal, der ikke står i "resultat".

Eksempel på et godt svar: "Der er 8 kopper kaffe tilbage på 4. etage, og
maskinen virker fint."
Eksempel på et dårligt svar: "Jeg har kaldt værktøjet og fået resultatet."
"""


def _nums(text):
    """Numbers appearing in a piece of text, as floats."""
    out = set()
    for m in NUM.finditer(str(text)):
        try:
            out.add(float(m.group().replace(",", ".")))
        except ValueError:
            continue
    return out


DA_THOUSANDS = re.compile(r"^\d{1,3}(?:\.\d{3})+$")


def _readings(tok):
    """Every number a Danish token could mean, with its decimal precision.

    Danish writes thousands with a period and decimals with a comma, so
    `76.500` is seventy-six thousand five hundred -- reading it as 76.5 marked
    correct answers as fabrications.
    """
    out = []
    if DA_THOUSANDS.match(tok):
        out.append((float(tok.replace(".", "")), 0))
    t = tok.replace(",", ".")
    try:
        out.append((float(t), len(t.split(".")[1]) if "." in t else 0))
    except ValueError:
        pass
    return out


def _traces_to(tok, pool):
    """Is `tok` a faithful rendering of some number in `pool`?

    Answers restate values rather than copy them: 153.9380400259 is written
    "153.94", 0.98 is written "98%", 76500 is written "76.500". All three are
    correct readings of the payload; a literal comparison calls them
    fabrications, which is what rejected 45 clean rows.
    """
    for val, d in _readings(tok):
        for p in pool:
            if val == p or round(p, d) == val:
                return True
            if d == 0 and int(val) == int(p):
                return True
            if round(p * 100, d) == val:      # 0.98 -> 98%
                return True
    return False


def _coerce(obj):
    """Numeric strings back to numbers.

    Leaves in a v4 `returns` block carry a Danish description and no type, so
    the schema cannot state one and the provider returns every value as a
    string: `{"area": "50.26548245743669"}`. That is both unnatural as a tool
    result and invisible to the grounding check, which reads numbers.
    """
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce(v) for v in obj]
    if isinstance(obj, str) and re.fullmatch(r"-?\d+(?:\.\d+)?", obj.strip()):
        f = float(obj)
        return int(f) if f.is_integer() and "." not in obj else f
    return obj


def _leaves(obj, prefix=""):
    """Scalar leaves of a result payload, as (path, value)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def _arrays(obj):
    """Every list inside a payload, so its length can be cited."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _arrays(v)
    elif isinstance(obj, list):
        yield obj
        for v in obj:
            yield from _arrays(v)


def _spec_fields(returns):
    """Top-level field names a `returns` block declares.

    Normalised to the first path segment because v4 stores the tree flattened
    (`restaurants[].address`) while v5 nests it. Comparing a real payload's
    `restaurants` key against the flat form marked every array-valued result
    as carrying extra fields -- 29 of 51 rejects in the first smoke.
    """
    if not isinstance(returns, dict):
        return set()
    props = returns.get("properties")
    if not isinstance(props, dict):
        return set()
    return {k.split(".")[0].removesuffix("[]") for k in props}


def _catalogue(msgs):
    for m in msgs:
        c = m.get("content") or ""
        if c.startswith("Værktøjer:"):
            try:
                return json.loads(c.split("Værktøjer:", 1)[1].strip()
                                  .split("\n\n")[0])
            except Exception:
                return []
    return []


def _last_user(msgs, upto):
    for m in reversed(msgs[:upto]):
        if m["role"] == "user":
            c = m.get("content") or ""
            return c.split("\n\n", 1)[-1] if c.startswith("Værktøjer:") else c
    return ""


def dangling(row):
    """(index, call, spec, question) for the row's unanswered terminal call.

    A call is dangling when no tool_result follows it. Only the terminal one
    can be -- mid-dialogue calls always got their result from the source.
    """
    msgs = row["messages"]
    call_at = None
    for i, m in enumerate(msgs):
        if m["role"] == "tool_call":
            nxt = msgs[i + 1]["role"] if i + 1 < len(msgs) else None
            if nxt != "tool_result":
                call_at = i
    if call_at is None:
        return None
    try:
        call = json.loads(msgs[call_at]["content"])
    except Exception:
        return None
    spec = next((t for t in _catalogue(msgs)
                 if t.get("name") == call.get("name")), None)
    if not spec:
        return None
    return call_at, call, spec, _last_user(msgs, call_at)


def spec_fingerprint(spec):
    """Hash of the contract the payload was generated against.

    The `returns` block is an INPUT to the result -- it becomes the response
    schema -- so a payload built for one version of it is not a valid answer
    for another. v4 stores the tree flattened and v5 nests it, and deduping
    the synonym fields (`final_price`/`new_price`/`discounted_price` all
    holding the same number) changes it again. Without this in the key those
    edits are invisible and the cache serves payloads shaped by a schema that
    no longer exists.

    Hashed over the CANONICAL form -- the JSON Schema actually sent to the
    provider -- not the raw block. v4 stores the tree flat and v5 nests it, but
    `_schema_from_returns` nests v4 before generating, so both produce the same
    contract and the same payload. Hashing the raw block would treat that pure
    representation change as a semantic one and discard 15,141 valid payloads;
    hashing the canonical form invalidates only tools whose field set really
    moved, such as those whose synonym fields get deduped.
    """
    canon = _schema_from_returns(spec.get("returns") or {})
    return hashlib.sha1(json.dumps(canon, sort_keys=True,
                                   ensure_ascii=False).encode()
                        ).hexdigest()[:12]


def cache_key(call, question, spec):
    return json.dumps([call.get("name"), call.get("arguments") or {},
                       question[:200], spec_fingerprint(spec)],
                      sort_keys=True, ensure_ascii=False)


# ── gate ────────────────────────────────────────────────────────────────────

def gate(result, answer, spec, context=""):
    """Why each check exists is in the reason string; returns None if clean.

    `context` is the question plus the call -- numbers the user supplied are
    fair to repeat ("på 4. etage"), so only numbers appearing in NEITHER the
    payload nor the context count as invented.
    """
    if not isinstance(result, dict) or not result:
        return "result-not-an-object"
    if not isinstance(answer, str) or not answer.strip():
        return "answer-empty"
    answer = answer.strip()
    if len(answer.split()) > 70:
        return "answer-too-long"
    if META.search(answer):
        return "answer-is-meta"

    declared = _spec_fields(spec.get("returns") or {})
    if declared:
        got = set(result)
        if got - declared:
            return f"result-extra-fields:{sorted(got - declared)[:3]}"
        if declared - got:
            return f"result-missing-fields:{sorted(declared - got)[:3]}"

    # Grounding. The whole point of the turn is that the answer reads the
    # result, so an answer that shares no value with it is the failure mode
    # being trained away, not a mild stylistic miss.
    vals = [v for _, v in _leaves(result)
            if not isinstance(v, bool) and v not in (None, "")]
    low = answer.lower()
    # Numbers the payload states, including those inside strings ("10 dollars")
    # and list lengths ("der er 3 actionfilm").
    pool = {float(v) for v in vals if isinstance(v, (int, float))}
    pool |= {float(m.group().replace(",", "."))
             for v in vals if isinstance(v, str) for m in NUM.finditer(v)}
    pool |= {float(len(v)) for v in _arrays(result)}

    # Substring is the honest test and covers 84.4% on its own. Numeric
    # tolerance adds 7.5pp and every point of it is real: Danish writes 898.09
    # as "898,09" and 12000.0 as "12.000", and answers round 50.26548245743669
    # to "50.27". None of those are substrings of the payload.
    cites = any(str(v).lower() in low for v in vals) or \
        any(_traces_to(m.group(), pool) for m in NUM.finditer(low))
    # A payload that states no fact -- {"message": "E-mail sendt", "status":
    # "succes"} -- has nothing to cite; the title and recipient in the answer
    # came from the question. 6.9% of pairs. Requiring a citation there
    # rejects correct answers for having nothing to quote.
    if not cites and pool:
        return "answer-not-grounded"

    # Invented numbers. `4 * 12 = 48` on a payload holding 28 is exactly what
    # the probe caught the model doing; an answer that does it here would
    # teach it.
    # Token sets, not substring search: `4` inside "på 4. etage" is followed by
    # a period, so a `(?![\d.,])` guard rejects the very number it should
    # allow -- which is what the clean-pair control caught.
    allowed = _nums(json.dumps(result, ensure_ascii=False) + " " + context)
    allowed |= pool
    for m in NUM.finditer(answer):
        if not _traces_to(m.group(), allowed):
            return f"answer-invents-number:{m.group()}"
    return None


# Planted defects, one per check that can fire on generated content. A gate
# that never fires is indistinguishable from clean data, so the run asserts it
# catches all of these before it trusts a single pass verdict.
_CUPS = {"returns": {"properties": {"cups_left": {}}}}
_CTX = "Er der kaffe tilbage på 4. etage?"
CONTROLS = [
    ({"cups_left": 8}, "Jeg har kaldt værktøjet og formateret parametrene.",
     _CUPS, _CTX),
    ({"cups_left": 8}, "Der er kaffe tilbage på etagen.", _CUPS, _CTX),
    ({"dog_years": 28}, "Din hund er 4 * 12 = 48 hundeår gammel.",
     {"returns": {"properties": {"dog_years": {}}}},
     "Min hund er 4 menneskeår gammel."),
    ({"cups_left": 8, "extra": 1}, "Der er 8 kopper tilbage.", _CUPS, _CTX),
    ({}, "Der er 8 kopper tilbage.", {"returns": {"properties": {}}}, _CTX),
    ({"cups_left": 8}, "   ", _CUPS, _CTX),
]


# Clean pairs that MUST pass. Each is a shape the first smoke wrongly rejected:
# a value followed by a period, a rounded float, a reformatted date, and a
# probability written as a percentage. Counts alone read all four as the model
# generating badly; the rejects file showed the gate was the problem.
CLEAN = [
    ({"cups_left": 8}, "Der er 8 kopper kaffe tilbage på 4. etage.",
     _CUPS, _CTX),
    ({"number": 42}, "Jeg har genereret et tilfældigt tal, som er 42.",
     {"returns": {"properties": {"number": {}}}}, "Giv mig et tilfældigt tal"),
    ({"area": 153.9380400259}, "Arealet af cirklen er 153.94 kvadratenheder.",
     {"returns": {"properties": {"area": {}}}}, "radius på 7"),
    ({"estimated_delivery": "2024-03-15", "status": "Leveret"},
     "Din pakke blev leveret den 15. marts 2024.",
     {"returns": {"properties": {"estimated_delivery": {}, "status": {}}}},
     "Hvor er min pakke?"),
    ({"confidence": 0.98, "sentiment": "negativ"},
     "Teksten udtrykker en negativ følelse med 98% sikkerhed.",
     {"returns": {"properties": {"confidence": {}, "sentiment": {}}}},
     "Analyser denne tekst"),
    ({"restaurants": [{"name": "Carbone", "rating": 4.7}]},
     "Jeg fandt Carbone med en bedømmelse på 4.7.",
     {"returns": {"properties": {"restaurants[]": {},
                                 "restaurants[].name": {},
                                 "restaurants[].rating": {}}}},
     "italienske restauranter"),
]


def check_controls():
    bad = [i for i, c in enumerate(CONTROLS)
           if gate(*c) is None]
    if bad:
        raise SystemExit(f"gate is blind: controls {bad} passed")
    for i, c in enumerate(CLEAN):
        why = gate(*c)
        if why is not None:
            raise SystemExit(f"gate rejects clean pair {i}: {why}")
    print(f"gate: {len(CONTROLS)} planted defects caught, "
          f"{len(CLEAN)} clean pairs pass", flush=True)


# ── generation ──────────────────────────────────────────────────────────────

def _schema_from_returns(returns):
    """The tool's `returns` block as a JSON Schema for the result payload.

    Gating field membership after generation detects the wrong shape; handing
    the provider the schema prevents it. v4 stores the tree flattened
    (`restaurants[].address`), so it is nested first with the same helper the
    generator uses. Leaves carry only a Danish description and no type, so
    they stay type-free -- the point is the key set, which is what the model
    got wrong.
    """
    if not isinstance(returns, dict):
        return None
    props = returns.get("properties")
    if not isinstance(props, dict) or not props:
        return None
    if any("." in k or k.endswith("[]") for k in props):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from translate_toolmind_da import _nest_paths
        props = _nest_paths({k: (v if isinstance(v, dict) else {})
                             for k, v in props.items()})

    def conv(node):
        if not isinstance(node, dict):
            return {}
        t = node.get("type")
        if t == "array":
            return {"type": "array", "items": conv(node.get("items") or {})}
        sub = node.get("properties")
        if isinstance(sub, dict) and sub:
            return {"type": "object",
                    "properties": {k: conv(v) for k, v in sub.items()},
                    "required": sorted(sub),
                    "additionalProperties": False}
        return {"type": t} if t else {}

    return {"type": "object",
            "properties": {k: conv(v) for k, v in props.items()},
            "required": sorted(props), "additionalProperties": False}


async def one_answer(session, call, spec, question, tries=3):
    shown = {"navn": spec.get("name"),
             "beskrivelse": spec.get("description"),
             "parametre": spec.get("parameters"),
             "returnerer": spec.get("returns")}
    user = (f"VÆRKTØJ:\n{json.dumps(shown, ensure_ascii=False)}\n\n"
            f"BRUGERENS SPØRGSMÅL:\n{question}\n\n"
            f"KALDET:\n{json.dumps(call, ensure_ascii=False)}")
    rs = _schema_from_returns(spec.get("returns") or {})
    # With a schema the payload is constrained to the declared keys; without
    # one (1,532 tools carry no `returns`) the model returns it as a JSON
    # string and the gate is the only check. Those rows also invent redundant
    # synonym fields -- three keys all holding 160 -- so they are skipped
    # unless explicitly asked for.
    if rs is None:
        if not ALLOW_UNSPECED[0]:
            return None, "no-returns-spec"
        payload = {"type": "string"}
    else:
        payload = rs
    body = {"model": MODEL, "temperature": 0.4, "max_tokens": 1600,
            "messages": [{"role": "system", "content": ANSWER_SYS},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "svar", "strict": True, "schema": {
                    "type": "object",
                    "properties": {"resultat": payload,
                                   "svar": {"type": "string"}},
                    "required": ["resultat", "svar"],
                    "additionalProperties": False}}}}
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    if a == tries - 1:
                        return None, f"http-{r.status}"
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                out = json.loads(d["choices"][0]["message"]["content"])
                res = out["resultat"]
                if isinstance(res, str):
                    res = json.loads(res)
                return _coerce(res), out["svar"]
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return None, "exhausted"


ALLOW_UNSPECED = [False]


async def main_async(args):
    import aiohttp
    from datasets import load_dataset

    check_controls()
    if args.rows:
        rows = [json.loads(l) for l in args.rows.open() if l.strip()]
    else:
        ds = load_dataset(args.repo, "sft", split=args.split)
        rows = [dict(r) for r in ds]
    if args.n:
        rows = rows[:args.n]
    print(f"{len(rows):,} rows from "
          f"{args.rows or args.repo + ':' + args.split}", flush=True)

    jobs = []
    for i, r in enumerate(rows):
        d = dangling(r)
        if d:
            jobs.append((i, d))
    print(f"{len(jobs):,} dangling terminal calls "
          f"({100*len(jobs)/max(1,len(rows)):.1f}% of rows)", flush=True)

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    have = {}
    if args.cache.exists():
        for line in args.cache.open():
            try:
                rec = json.loads(line)
                have[rec["k"]] = (rec["resultat"], rec["svar"])
            except Exception:
                continue
    todo = [(i, d) for i, d in jobs
            if cache_key(d[1], d[3], d[2]) not in have]
    print(f"{len(have):,} cached, {len(todo):,} to generate", flush=True)
    if args.dry_run:
        # What a run would COST, before it costs it. The fingerprint means a
        # spec edit shows up here as a jump in `to generate`, which is the
        # signal worth seeing before launching.
        free = sum(1 for _, d in todo
                   if _schema_from_returns(d[2].get("returns") or {}) is None)
        print(f"dry-run: {len(todo)-free:,} would hit the API, "
              f"{free:,} skipped as no-returns-spec", flush=True)
        return

    # Generation and gating are separate passes: caching happens BEFORE the
    # verdict, so re-gating after a gate change costs nothing and a rejected
    # generation is never re-bought. Every gate fix in this file was found by
    # reading rejects, so that property paid for itself several times.
    done = [0]
    if todo:
        sem = asyncio.Semaphore(args.concurrency)
        lock = asyncio.Lock()
        failed = Counter()
        async with aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {_key()}",
                         "Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=300)) as s:
            with args.cache.open("a", buffering=1) as fh:
                async def run(i, d):
                    _, call, spec, question = d
                    async with sem:
                        res, ans = await one_answer(s, call, spec, question)
                    done[0] += 1
                    if done[0] % 100 == 0:
                        print(f"  generated {done[0]:,}/{len(todo):,}"
                              f"  ({sum(failed.values()):,} failed)",
                              flush=True)
                    if res is None:
                        failed[ans or "api-failed"] += 1
                        return
                    k = cache_key(call, question, spec)
                    async with lock:
                        have[k] = (res, ans)
                        fh.write(json.dumps(
                            {"k": k, "tool": spec.get("name"),
                             "resultat": res, "svar": ans},
                            ensure_ascii=False) + "\n")
                await asyncio.gather(*[run(i, d) for i, d in todo])
        if failed:
            print("generation failures: "
                  + ", ".join(f"{w} {c:,}" for w, c in failed.most_common()),
                  flush=True)

    reasons = Counter()
    accepted = {}
    with args.rejects.open("w", buffering=1) as rej:
        for i, d in jobs:
            _, call, spec, question = d
            k = cache_key(call, question, spec)
            got = have.get(k)
            if not got:
                continue
            res, ans = got
            why = gate(res, ans, spec,
                       question + " " + json.dumps(call, ensure_ascii=False))
            if why:
                reasons[why.split(":")[0]] += 1
                # Rejects are written out, not just counted. A gate tuned from
                # counts alone cannot tell over-strictness from dirty
                # generations, and the two want opposite fixes.
                rej.write(json.dumps(
                    {"why": why, "tool": spec.get("name"),
                     "declared": sorted(_spec_fields(spec.get("returns") or {})),
                     "resultat": res, "svar": ans, "q": question[:200]},
                    ensure_ascii=False) + "\n")
                continue
            accepted[k] = (res, ans)

    kept = sum(1 for i, d in jobs
               if cache_key(d[1], d[3], d[2]) in accepted)
    print(f"accepted {kept:,}/{len(jobs):,} calls "
          f"({100*kept/max(1,len(jobs)):.1f}%)", flush=True)
    if reasons:
        print("rejected:")
        for w, c in reasons.most_common():
            print(f"  {w:<28} {c:,}")
    have = accepted

    # splice
    attached = 0
    for i, d in jobs:
        call_at, call, spec, question = d
        got = have.get(cache_key(call, question, spec))
        if not got:
            continue
        res, ans = got
        msgs = rows[i]["messages"]
        msgs[call_at + 1:call_at + 1] = [
            {"role": "tool_result",
             "content": json.dumps(res, ensure_ascii=False)},
            {"role": "assistant", "content": ans}]
        attached += 1
    print(f"attached to {attached:,} rows", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}", flush=True)

    if args.show:
        print("\n" + "=" * 74)
        for i, d in random.Random(0).sample(jobs, min(args.show, len(jobs))):
            for m in rows[i]["messages"][d[0] - 1:d[0] + 3]:
                c = m["content"]
                print(f"  [{m['role']:<11}] {c[:400]}")
            print("-" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="jensjepsen/danish-tool-dialogues-v4")
    ap.add_argument("--split", default="train")
    ap.add_argument("--rows", type=Path, default=None,
                    help="local jsonl instead of the hub")
    ap.add_argument("--out", type=Path,
                    default=Path("scratch/tool_answers/answered.jsonl"))
    ap.add_argument("--cache", type=Path,
                    default=Path("scratch/tool_answers/answers.jsonl"))
    ap.add_argument("--rejects", type=Path,
                    default=Path("scratch/tool_answers/rejects.jsonl"))
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true",
                    help="report how many calls would hit the API, then exit")
    ap.add_argument("--allow-unspeced", action="store_true",
                    help="also answer calls whose tool declares no returns; "
                         "their payloads are unconstrained and pad with "
                         "synonym fields")
    args = ap.parse_args()
    ALLOW_UNSPECED[0] = args.allow_unspeced
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
