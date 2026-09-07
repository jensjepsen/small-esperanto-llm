"""Build a GRPO set where the gold is known by CONSTRUCTION, not by reference.

The v5 answer gate could only ask "does the reply cite something from the
payload". That rewards reciting every field -- 43% of v5's answers do -- and
the tool_answer eval scores the same way (recall over all payload values), so
a padded answer outscores the reference reply. Measured: on one eval row the
gold answer scored 26.1% and a padded model answer 47.8%.

The fix needs a notion of WHICH field was asked for. Deriving it from a gold
reply is possible but proxy-ish -- it penalises a model for citing a real
payload value the reference happened to skip. Deriving it by construction is
exact: choose the field FIRST, then generate a question that asks for it.

Each row therefore carries everything a deterministic reward needs:

    tool + arguments     -> gold call          (task A)
    payload + target     -> gold answer field  (task B)

and emits two single-turn GRPO prompts, because TRL's GRPO generates one
completion per prompt and a call-then-answer rollout would need multi-turn
machinery that buys nothing here:

    task A  prompt: catalogue + question              -> expect a tool call
    task B  prompt: ... + call + tool_result          -> expect an answer

No reference text is stored for either. The reward is computed from the
metadata.
"""
import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tool_answer_turns import (  # noqa: E402
    _leaves, _schema_from_returns, _spec_fields, _coerce,
)

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


SYS = """Du laver træningsdata til en dansk assistent, der bruger værktøjer.

Du får ét værktøj og ét MÅLFELT blandt de felter, værktøjet returnerer.

Lav tre ting:

1. "argumenter": realistiske værdier til værktøjets parametre. Konkrete
   værdier, ikke pladsholdere.

2. "spoergsmaal": et dansk spørgsmål fra en bruger, som
   - kun kan besvares ved at kalde værktøjet med præcis de argumenter,
   - spørger om MÅLFELTET og ikke om de andre felter,
   - nævner argumenternes værdier, så de kan udledes af spørgsmålet,
   - IKKE bruger feltets engelske navn og ikke gengiver feltbeskrivelsen
     ordret. Skriv som en almindelig bruger ville spørge.

3. "resultat": et realistisk svar fra værktøjet. Udfyld ALLE felterne, ikke kun
   målfeltet -- de andre felter er distraktorer og skal have konkrete,
   forskellige værdier. Målfeltets værdi må ikke være den samme som et andet
   felts værdi.

Eksempel: værktøj coffee_machine_status(floor), målfelt cups_left
  argumenter  {"floor": 4}
  spørgsmål   "Er der kaffe nok tilbage på 4. etage til et møde?"
  resultat    {"cups_left": 8, "days_since_service": 12, "floor": 4}
"""


def load_tools(repo, split, limit=0):
    """Distinct tool specs carrying a returns block, from the published corpus."""
    from datasets import load_dataset
    ds = load_dataset(repo, "sft", split=split)
    out = {}
    for r in ds:
        for m in r["messages"]:
            c = m.get("content") or ""
            if not c.startswith("Værktøjer:"):
                continue
            try:
                cat = json.loads(c.split("Værktøjer:", 1)[1].strip()
                                 .split("\n\n")[0])
            except Exception:
                break
            for t in cat:
                if t.get("name") and t.get("returns") and t["name"] not in out:
                    out[t["name"]] = t
            break
        if limit and len(out) >= limit:
            break
    return list(out.values())


# Fields that carry no information to select FOR. Asking "what was the status"
# and rewarding "succes" trains recitation of a constant, not field selection.
BOILERPLATE = {"status", "message", "success", "result", "code", "error",
               "data", "response", "info"}


def target_fields(spec):
    """Leaf fields worth asking about.

    Excludes three kinds. Nested/array fields, which are ambiguous to ask for.
    Boilerplate status keys, whose value is a constant. And fields that ECHO an
    input parameter -- `retrieve_movie_details(title=...)` returning `title`
    means the answer is already in the question, so the row is degenerate and
    the reward is free. The first smoke produced exactly that row.
    """
    props = ((spec.get("returns") or {}).get("properties") or {})
    params = set(((spec.get("parameters") or {}).get("properties") or {}))
    out = []
    for k, v in props.items():
        if "." in k or k.endswith("[]"):
            continue
        if isinstance(v, dict) and v.get("type") in ("object", "array"):
            continue
        if k.lower() in BOILERPLATE or k in params:
            continue
        out.append(k)
    return out


async def one_row(session, spec, field, tries=3):
    rs = _schema_from_returns(spec.get("returns") or {})
    if rs is None:
        return None
    shown = {"navn": spec.get("name"), "beskrivelse": spec.get("description"),
             "parametre": spec.get("parameters"),
             "returnerer": spec.get("returns"), "maalfelt": field}
    body = {"model": MODEL, "temperature": 0.7, "max_tokens": 1200,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": json.dumps(
                             shown, ensure_ascii=False)}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "row", "strict": True, "schema": {
                    "type": "object", "properties": {
                        "argumenter": {"type": "string"},
                        "spoergsmaal": {"type": "string"},
                        "resultat": rs},
                    "required": ["argumenter", "spoergsmaal", "resultat"],
                    "additionalProperties": False}}}}
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                o = json.loads(d["choices"][0]["message"]["content"])
                args = o["argumenter"]
                if isinstance(args, str):
                    args = json.loads(args)
                return _coerce(args), o["spoergsmaal"], _coerce(o["resultat"])
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return None


def gate(spec, field, args, question, payload):
    """Reasons a constructed row cannot carry a clean reward."""
    if not isinstance(payload, dict) or field not in payload:
        return "target-missing-from-payload"
    target = payload[field]
    if target is None or target == "":
        return "target-empty"
    declared = _spec_fields(spec.get("returns") or {})
    if declared and set(payload) - declared:
        return "payload-extra-fields"

    vals = [(k, v) for k, v in payload.items()]
    others = [v for k, v in vals if k != field]
    # A target that equals a distractor makes precision unscoreable: citing the
    # value cannot be attributed to the right field.
    if any(str(v).strip().lower() == str(target).strip().lower()
           for v in others):
        return "target-value-collides"
    # With no competitor of the same kind, precision is free and the row
    # teaches nothing about selection.
    same_kind = [v for v in others
                 if isinstance(v, (int, float)) == isinstance(target,
                                                              (int, float))
                 and not isinstance(v, bool)]
    if len(same_kind) < 1 or len(others) < 2:
        return "too-few-competitors"

    q = (question or "").strip()
    if len(q.split()) < 4:
        return "question-too-short"
    if field.lower() in q.lower():
        return "question-names-the-field"     # trivially solvable
    # The value itself must not be in the question. target_fields already drops
    # fields that echo a parameter, but the generator can still write the
    # answer into the prompt ("Hvad hedder filmen ... og hedder The Matrix?").
    tv = str(target).strip()
    if len(tv) > 2 and tv.lower() in q.lower():
        return "question-contains-the-answer"
    desc = (((spec.get("returns") or {}).get("properties") or {})
            .get(field) or {})
    d = (desc.get("description") or "").strip().lower()
    if d and len(d) > 8 and d in q.lower():
        return "question-quotes-the-description"
    # Task A is only verifiable if the arguments follow from the question.
    if isinstance(args, dict):
        for v in args.values():
            if isinstance(v, (int, float)) and str(v) not in q:
                return f"argument-not-in-question:{v}"
            if isinstance(v, str) and len(v) > 2 and v.lower() not in q.lower():
                return f"argument-not-in-question:{v[:24]}"
    return None


CATALOG_LABEL = "Værktøjer"


def emit(spec, field, args, question, payload, pool, rng):
    """The two single-turn GRPO prompts, plus the reward metadata."""
    cat = [spec] + rng.sample([t for t in pool if t["name"] != spec["name"]],
                              k=min(5, max(0, len(pool) - 1)))
    rng.shuffle(cat)                       # position must carry no signal
    catalog = json.dumps(cat, ensure_ascii=False)
    call = {"name": spec["name"], "arguments": args}
    base = f"<|user|>{CATALOG_LABEL}:\n{catalog}\n\n{question}<|end|><|assistant|>"
    distractors = [v for k, v in payload.items() if k != field]
    return [
        {"task": "call", "prompt": base,
         "gold": {"name": spec["name"], "arguments": args}},
        {"task": "answer",
         "prompt": (base + json.dumps(call, ensure_ascii=False).join(
             ["<|tool_call|>", "<|/tool_call|>"]) + "<|end|>"
             + f"<|tool_result|>{json.dumps(payload, ensure_ascii=False)}"
               "<|/tool_result|><|assistant|>"),
         "gold": {"field": field, "value": payload[field],
                  "distractors": distractors}},
    ]


async def main_async(args):
    import aiohttp
    pool = load_tools(args.repo, args.split)
    print(f"{len(pool):,} distinct tools with a returns block", flush=True)
    rng = random.Random(args.seed)
    picks = []
    for spec in pool:
        fs = target_fields(spec)
        if not fs:
            continue
        for f in rng.sample(fs, k=min(args.per_tool, len(fs))):
            picks.append((spec, f))
    rng.shuffle(picks)
    if args.n:
        picks = picks[:args.n]
    print(f"{len(picks):,} (tool, target-field) pairs to build", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    reasons = Counter()
    rows, kept = [], []
    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}",
                     "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=300)) as s:
        async def run(spec, field):
            async with sem:
                got = await one_row(s, spec, field)
            if not got:
                reasons["api-failed"] += 1
                return
            a, q, p = got
            why = gate(spec, field, a, q, p)
            if why:
                reasons[why.split(":")[0]] += 1
                return
            kept.append((spec, field, a, q, p))
        await asyncio.gather(*[run(sp, f) for sp, f in picks])

    for spec, field, a, q, p in kept:
        rows.extend(emit(spec, field, a, q, p, pool, rng))
    print(f"\nkept {len(kept):,}/{len(picks):,} rows "
          f"-> {len(rows):,} GRPO prompts", flush=True)
    if reasons:
        print("rejected:")
        for w, c in reasons.most_common():
            print(f"  {w:<32} {c:,}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}", flush=True)

    for spec, field, a, q, p in kept[:args.show]:
        print("=" * 76)
        print(f"  tool     : {spec['name']}   target field: {field}")
        print(f"  desc     : {(((spec.get('returns') or {}).get('properties') or {}).get(field) or {}).get('description')}")
        print(f"  question : {q}")
        print(f"  gold call: {json.dumps({'name': spec['name'], 'arguments': a}, ensure_ascii=False)}")
        print(f"  payload  : {json.dumps(p, ensure_ascii=False)[:200]}")
        print(f"  REWARD   : cite {p[field]!r}; do NOT cite "
              f"{[v for k, v in p.items() if k != field][:4]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="jensjepsen/danish-tool-dialogues-v5")
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-tool", type=int, default=1)
    ap.add_argument("--n", type=int, default=0, help="0 = all pairs")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
