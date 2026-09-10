"""Does an answer contradict the arguments its own call was made with?

One variable, one question. The judge sees ONLY the call arguments and the
answer -- not the payload, not the catalogue, not the rest of the dialogue --
so it cannot reject for grounding, style or completeness. It can only rule on
whether the answer asserts something the arguments disprove.

The defect this measures, from gen_ub9:

    tool_call    {"include_failed_prints": true}
    assistant    "Der er 5 jobs i printkøen. Fejlloggen medregnes ikke."

The answers prompt used to receive only (question, payload), so the model had
no way to know what was requested and invented a qualifier.

    uv run python scripts/judge_contradictions.py \
        --a scratch/gen_ub7/sft.jsonl  --a-label without-args \
        --b scratch/gen_ub11/sft.jsonl --b-label with-args
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tool_dialogues_da import strip_catalogue  # noqa: E402

MODEL = "google/gemini-2.5-flash-lite"
URL = "https://openrouter.ai/api/v1/chat/completions"
SEP = "Værktøjer:"

SYS = """Du afgør ÉN ting: modsiger svaret de argumenter, kaldet blev lavet med?

Du får `argumenter` og `svar`. Du får IKKE resultatet, og du skal IKKE vurdere,
om svaret er velbegrundet, fyldestgørende eller velformuleret.

MODSIGELSE (`modsiger: true`) er kun:
- svaret siger, at noget ikke er med, som argumenterne slår TIL
  (kaldt med `include_failed_prints: true`, men svaret siger, fejlloggen
  ikke er medregnet)
- svaret siger, at noget er med, som argumenterne slår FRA
- svaret nævner en anden værdi for et argument, end det der blev sendt
  (kaldt med `city: "Aarhus"`, men svaret taler om København)
- svaret hævder en afgrænsning -- en periode, en type, et filter -- som
  argumenterne ikke sætter

Alt andet er `modsiger: false`. Er du i tvivl, så svar false."""

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["domme"],
    "properties": {"domme": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["modsiger", "problem"],
        "properties": {"modsiger": {"type": "boolean"},
                       "problem": {"type": "string"}}}}}}

CONTROLS = [
    ({"argumenter": {"include_failed_prints": True},
      "svar": "Der er 5 jobs i printkøen. Fejlloggen medregnes ikke."}, True),
    ({"argumenter": {"city": "Aarhus"},
      "svar": "Vejret i København er 12 grader."}, True),
    ({"argumenter": {"include_historical_data": True},
      "svar": "Der blev ikke inkluderet historiske data i analysen."}, True),
    ({"argumenter": {"log_type": "all"},
      "svar": "Her er kun vaccinationsloggen for koen."}, True),
    ({"argumenter": {"include_failed_prints": True},
      "svar": "Der er 5 jobs i printkøen, inklusive de mislykkede."}, False),
    ({"argumenter": {"city": "Aarhus"},
      "svar": "Vejret i Aarhus er 12 grader."}, False),
    ({"argumenter": {"pig_ids": [1, 2, 3]},
      "svar": "Slagtevægten for grisene er 119 kg."}, False),
]


def _key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for p in (Path.home() / "or", Path.home() / ".or"):
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("no OpenRouter key (~/or)")


def items_from(path, n, seed=0):
    """(arguments, answer) pairs: the assistant turn that answers each call."""
    out = []
    for line in Path(path).open():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        # ALL calls that share an answer, not one of them. A parallel turn
        # makes two calls and answers both at once; pairing the answer with a
        # single call made the judge flag the other call's subject as
        # "not in the arguments" -- an artefact, not a contradiction.
        pending = []
        for m in msgs:
            if m["role"] == "tool_call":
                try:
                    a = json.loads(m["content"]).get("arguments") or {}
                except Exception:
                    a = {}
                if a:
                    pending.append(a)
                continue
            if m["role"] == "tool_result":
                continue
            if m["role"] == "assistant" and pending:
                text = str(m.get("content") or "").strip()
                if text:
                    out.append({"argumenter": pending[0] if len(pending) == 1
                                else pending, "svar": text[:400]})
                pending = []
            elif m["role"] == "user":
                pending = []
    random.Random(seed).shuffle(out)
    return out[:n]


async def ask(session, items, tries=3):
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": 1200,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": json.dumps(
                             {"opgaver": items}, ensure_ascii=False, indent=1)}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "contradiction", "strict": True, "schema": SCHEMA}}}
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                v = json.loads(d["choices"][0]["message"]["content"])["domme"]
                if len(v) == len(items):
                    return v, d.get("usage") or {}
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return None, {}


async def run(session, items, batch, tok):
    out = []
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        v, u = await ask(session, chunk)
        tok["in"] += u.get("prompt_tokens", 0)
        tok["out"] += u.get("completion_tokens", 0)
        out.extend(v or [{"modsiger": False, "problem": "(judge failed)"}
                         for _ in chunk])
    return out


async def main_async(a):
    import aiohttp
    tok = {"in": 0, "out": 0}
    res = {}
    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}"}) as session:
        cv = await run(session, [c[0] for c in CONTROLS], 7, tok)
        want = [c[1] for c in CONTROLS]
        got = [bool(v.get("modsiger")) for v in cv]
        miss = [i for i, (w, g) in enumerate(zip(want, got)) if w != g]
        print(f"controls: {len(want) - len(miss)}/{len(want)} correct"
              + (f"   MISSED {miss}" if miss else ""), flush=True)
        if miss:
            for i in miss:
                print(f"    [{i}] wanted modsiger={want[i]}, got {got[i]}: "
                      f"{CONTROLS[i][0]['svar'][:60]}", flush=True)
        if len(miss) > 1:
            raise SystemExit("judge not calibrated; rates would be meaningless")
        for path, label in ((a.a, a.a_label), (a.b, a.b_label)):
            items = items_from(path, a.n, a.seed)
            v = await run(session, items, a.batch, tok)
            bad = [(it, d) for it, d in zip(items, v) if d.get("modsiger")]
            res[label] = (len(bad), len(items))
            print(f"\n{label:<14} {len(bad)}/{len(items)} answers contradict "
                  f"their call  ({100 * len(bad) / max(len(items), 1):.1f}%)")
            for it, d in bad[:5]:
                print(f"    args: {json.dumps(it['argumenter'], ensure_ascii=False)[:70]}")
                print(f"    svar: {it['svar'][:90]}")
                print(f"     why: {d.get('problem','')[:90]}")
    (ka, kb) = list(res)
    (xa, na), (xb, nb) = res[ka], res[kb]
    pa, pb = xa / max(na, 1), xb / max(nb, 1)
    se = math.sqrt(pa * (1 - pa) / max(na, 1) + pb * (1 - pb) / max(nb, 1))
    d = pb - pa
    print(f"\n{ka} {pa:.3f}  vs  {kb} {pb:.3f}")
    if se:
        print(f"difference {d:+.3f}   95% CI [{d - 1.96 * se:+.3f}, "
              f"{d + 1.96 * se:+.3f}]   "
              f"{'significant' if abs(d) > 1.96 * se else 'NOT significant'}")
    cost = tok["in"] / 1e6 * 0.10 + tok["out"] / 1e6 * 0.40
    print(f"judge cost: ${cost:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
