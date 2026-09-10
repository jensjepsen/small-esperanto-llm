"""Score payload plausibility across two corpora, with planted controls.

Answers one question: does generating payloads against the call ARGUMENTS
produce more plausible data than deriving them from a hash of the contract's
examples? Everything else in the two runs is held constant.

The judge is shown only (question, arguments, payload) -- never the answer,
never the catalogue -- so it is rating the DATA, not the dialogue.

    uv run python scripts/judge_payloads.py \
        --a scratch/gen_ub7/sft.jsonl  --a-label hash \
        --b scratch/gen_ub10/sft.jsonl --b-label conditioned --n 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tool_dialogues_da import strip_catalogue  # noqa: E402

SEP = "Værktøjer:"
MODEL = "google/gemini-2.5-flash-lite"
URL = "https://openrouter.ai/api/v1/chat/completions"

SYS = """Du vurderer, om et VÆRKTØJSRESULTAT er troværdige data.

Du får brugerens spørgsmål, de argumenter værktøjet blev kaldt med, og det
resultat, værktøjet gav. Du skal IKKE vurdere sproget eller et svar -- kun om
selve resultatet er data, et rigtigt API kunne have returneret.

Resultatet er DÅRLIGT hvis:
- det modsiger argumenterne (bedt om vegetarisk mad, men retten indeholder kød;
  spurgt til ét id, men resultatet handler om et andet)
- felterne modsiger hinanden (pigmentet er Vermilion, men hovedgrundstoffet
  er kobber; arealet i m2 passer ikke med arealet i hektar)
- en værdi er fysisk umulig eller absurd i sin sammenhæng (en rugemaskine på
  50 grader, en luftfugtighed på 224%, en gris der tager 7 kg på om dagen)
- en værdi er tydeligt en pladsholder eller uden indhold

Ellers er det GODT. Vær konkret i `problem`."""

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["domme"],
    "properties": {"domme": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["ok", "problem"],
        "properties": {"ok": {"type": "boolean"},
                       "problem": {"type": "string"}}}}}}

# Planted: the judge must catch these, or its rates mean nothing.
CONTROLS = [
    ({"spoergsmaal": "Planlæg en vegetarisk menu til 200 elever.",
      "argumenter": {"dietary_restrictions": "vegetar"},
      "resultat": {"planned_menu": "Mandag: Hakkebøf med bløde løg",
                   "cost_dkk": 4200}}, False),
    ({"spoergsmaal": "Hvad er status på rugemaskinen?",
      "argumenter": {"incubator_id": "3"},
      "resultat": {"current_temperature_celsius": 50.1,
                   "target_temperature_celsius": 37.5}}, False),
    ({"spoergsmaal": "Hvad er slagtevægten?",
      "argumenter": {"current_weight_kg": 75},
      "resultat": {"estimated_slaughter_weight_kg": 115,
                   "average_daily_gain_kg": 7}}, False),
    ({"spoergsmaal": "Hvad er luftfugtigheden i drivhuset?",
      "argumenter": {"greenhouse_id": 1},
      "resultat": {"current_humidity": 224, "target_humidity": 80}}, False),
    ({"spoergsmaal": "Hvad er status på rugemaskinen?",
      "argumenter": {"incubator_id": "3"},
      "resultat": {"current_temperature_celsius": 37.6,
                   "target_temperature_celsius": 37.5,
                   "eggs_hatched": 8, "eggs_total": 15}}, True),
    ({"spoergsmaal": "Hvad er slagtevægten for en gris på 75 kg?",
      "argumenter": {"current_weight_kg": 75, "age_days": 120},
      "resultat": {"estimated_slaughter_weight_kg": 115,
                   "average_daily_gain_kg": 0.9}}, True),
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
    """(question, arguments, payload) triples -- one per call."""
    out = []
    for line in Path(path).open():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        last_user = ""
        for i, m in enumerate(msgs):
            if m["role"] == "user":
                c = str(m.get("content") or "")
                # The catalogue rides in the first user turn and is full of
                # `]`. Splitting on the first one landed mid-JSON and fed the
                # judge a slice of an unrelated tool's schema as "the
                # question" -- which it then, correctly, called irrelevant.
                last_user = strip_catalogue(c)
            if m["role"] != "tool_call":
                continue
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            if not nxt or nxt["role"] != "tool_result":
                continue
            try:
                call = json.loads(m["content"])
                pay = json.loads(nxt["content"])
            except Exception:
                continue
            if not isinstance(pay, dict) or not last_user.strip():
                continue
            out.append({"spoergsmaal": last_user[:300],
                        "argumenter": call.get("arguments") or {},
                        "resultat": pay})
    random.Random(seed).shuffle(out)
    return out[:n]


async def ask(session, items, tries=3):
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": 1200,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": json.dumps(
                             {"opgaver": items}, ensure_ascii=False, indent=1)}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "payload_judge", "strict": True, "schema": SCHEMA}}}
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
    verdicts = []
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        v, u = await ask(session, chunk)
        tok["in"] += u.get("prompt_tokens", 0)
        tok["out"] += u.get("completion_tokens", 0)
        verdicts.extend(v or [{"ok": True, "problem": "(judge failed)"}
                              for _ in chunk])
    return verdicts


async def main_async(a):
    import aiohttp
    tok = {"in": 0, "out": 0}
    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}"}) as session:
        cv = await run(session, [c[0] for c in CONTROLS], 6, tok)
        want = [c[1] for c in CONTROLS]
        got = [bool(v.get("ok")) for v in cv]
        miss = [i for i, (w, g) in enumerate(zip(want, got)) if w != g]
        print(f"controls: {len(want) - len(miss)}/{len(want)} correct"
              + (f"  MISSED {miss}" if miss else ""), flush=True)
        if len(miss) > 1:
            raise SystemExit("judge is not calibrated; rates below would be "
                             "meaningless")
        for path, label in ((a.a, a.a_label), (a.b, a.b_label)):
            items = items_from(path, a.n, a.seed)
            v = await run(session, items, a.batch, tok)
            bad = [(i, x.get("problem", "")) for i, x in enumerate(v)
                   if not x.get("ok")]
            print(f"\n{label:<12} {len(items) - len(bad)}/{len(items)} payloads "
                  f"plausible  ({100 * (len(items) - len(bad)) / max(len(items), 1):.1f}%)")
            for _i, why in bad[:6]:
                print(f"    - {why[:100]}")
    cost = tok["in"] / 1e6 * 0.10 + tok["out"] / 1e6 * 0.40
    print(f"\njudge cost: ${cost:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
