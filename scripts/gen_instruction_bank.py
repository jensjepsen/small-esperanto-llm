"""Generate Danish instruction paraphrases and gate them BEHAVIOURALLY.

The instruction is part of the task specification, not decoration: the gold
answer assumes values are copied verbatim, absent fields get the null marker,
and the output is one line per field. A paraphrase that says "opsummer" instead
of "udtræk ordret" silently makes the gold wrong for every row that uses it.

So the gate is a measurement, not an opinion. Each candidate is run in
INSTRUCTION-ONLY mode -- no demonstrations, so the wording alone has to carry
the task -- against held-out (passage, fields, gold) triples, and scored with
the same canon() exact-match the training data is built on. A paraphrase that
drops verbatim-ness produces rewritten values and fails; one that omits the
null rule invents values for absent fields and fails those rows.

PLANTED CONTROLS are included in every run. If the gate does not reject them,
its pass rate means nothing -- an uncalibrated judge is the failure mode this
design exists to avoid.

Usage:
  python scripts/gen_instruction_bank.py --n 200 --probes 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_extraction_da import (BLANKS, INSTR_EXTRACT, NULL, _ws,  # noqa: E402
                               canon, is_verbatim)

MODEL = "google/gemini-2.5-flash-lite"
URL = "https://openrouter.ai/api/v1/chat/completions"

GEN_SYS = """Du skriver instruktioner på dansk til en informationsudtrækker.

Opgaven, instruktionen skal beskrive, er ALTID den samme:
- læs en tekst og en liste af feltnavne
- angiv hvert felts værdi som en ORDRET tekststump fra teksten
- er feltet ikke nævnt i teksten, skrives {null}
- én linje per felt

Skriv {n} FORSKELLIGE formuleringer af denne instruktion. Variér:
- registret: kort og bydende, neutralt, høfligt, teknisk
- længden: fra 5 ord til 3 sætninger
- ordvalget: udtræk / find / hent / angiv / udfyld / gengiv

Krav til hver formulering:
- den skal gøre det klart at værdier skal kopieres ORDRET
- den skal nævne hvad man gør, når feltet ikke står i teksten
- brug pladsholderen {null} præcis ét sted
- kun dansk, ingen engelske ord"""

# Deliberately broken. The gate MUST reject these; if it does not, it is too
# lax and the pass rate on real candidates is uninterpretable.
CONTROLS = [
    ("ctrl:summarise", "Opsummer teksten kort med dine egne ord."),
    ("ctrl:paraphrase", "Udtræk felterne fra teksten, og omskriv dem pænt "
                        "med dine egne ord."),
    ("ctrl:no_null", "Udfyld felterne ud fra teksten."),
    ("ctrl:unrelated", "Oversæt teksten til engelsk."),
]


def _key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for p in (Path.home() / "or", Path.home() / ".or"):
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("no OpenRouter key (~/or)")


async def _call(session, sys_msg, user_msg, schema=None, temp=0.9, tries=4):
    body = {"model": MODEL, "temperature": temp,
            "messages": [{"role": "system", "content": sys_msg},
                         {"role": "user", "content": user_msg}]}
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "svar", "strict": True, "schema": schema}}
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                return d["choices"][0]["message"]["content"]
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return None


async def generate(session, n):
    schema = {"type": "object", "properties": {"instruktioner": {
        "type": "array", "items": {"type": "string"}}},
        "required": ["instruktioner"], "additionalProperties": False}
    out = []
    for _ in range(0, n, 25):
        txt = await _call(session, GEN_SYS.replace("{n}", "25").replace(
            "{null}", NULL), "Skriv dem nu.", schema)
        if not txt:
            continue
        try:
            out += json.loads(txt)["instruktioner"]
        except Exception:
            pass
    # the placeholder must survive, or .format(null=...) breaks at render time
    keep = []
    for t in out:
        t = _ws(t)
        if t.count(NULL) == 1 and 10 < len(t) < 400:
            keep.append(t.replace(NULL, "{null}"))
    return sorted(set(keep))


def probes(raw_path, k, rng):
    """Held-out (passage, fields, gold) triples, spanning registers."""
    rows = [json.loads(l) for l in open(raw_path)]
    by = {}
    for r in rows:
        by.setdefault(r.get("meta", {}).get("register", "?"), []).append(r)
    # Scan widely: a probe needs >=2 present AND >=1 absent field, and only
    # ~4.6% of fields come back empty, so sampling 3 rows per register yielded
    # 2 usable probes and the gate measured nothing.
    out = []
    for reg, rs in by.items():
        for r in rng.sample(rs, min(400, len(rs))):
            present = [f for f in r["felter"]
                       if [v for v in f["vaerdi"] if is_verbatim(v, r["passage"])]]
            absent = [f for f in r["felter"] if not f["vaerdi"]]
            if len(present) < 2 or not absent:
                continue                           # no absent field -> the null
            chosen = present[:3] + absent[:1]      # rule is not exercised at all
            gold = {}                              # the null rule is exercised
            for f in chosen:
                vs = [_ws(v) for v in f["vaerdi"] if is_verbatim(v, r["passage"])]
                gold[f["navn"]] = vs or [NULL]
            out.append((r["passage"], [f["navn"] for f in chosen], gold))
            if len(out) >= k:
                return out
    if len(out) < k:
        print(f"  WARNING: only {len(out)} probes found (wanted {k})", flush=True)
    return out


async def score(session, instr, items):
    """Mean fraction of FIELDS answered exactly right, instruction only.

    All-or-nothing per item has no resolution: requiring every field of every
    probe to match exactly floored all 25 candidates at 0.17 and put a control
    that omits the null rule on the same score as the good ones. Grading per
    field separates them -- an instruction that drops the null convention gets
    the present fields right and the absent one wrong, which is visible.

    Absent fields are reported separately because that is precisely what the
    null rule governs, and it is one field per probe against three present
    ones -- averaged together it would be swamped.
    """
    tot = hit = a_tot = a_hit = 0
    for passage, names, gold in items:
        prompt = (instr.format(null=NULL) + "\n\nTekst:\n" + passage +
                  "\nFelter: " + ", ".join(names) + "\nSvar:")
        out = await _call(session, "Du er en præcis dansk udtrækker.",
                          prompt, temp=0.0)
        if not out:
            continue
        # First block only. The training answer is one line per field; on a
        # catalogue-like passage the model answers correctly and then keeps
        # going, producing 16 values per field against a gold of one. Scoring
        # the whole output measured "does the model stop", not "does the
        # instruction convey the task" -- the reference scored 0.00 on absent
        # fields while visibly emitting the right `felt: -` line.
        first = out.split("\n\n")[0]
        got = canon(first, "kv_colon", names) or {}
        for n, v in gold.items():
            is_absent = (v == [NULL])
            right = sorted(got.get(n, [])) == sorted(v)
            if is_absent:
                a_tot += 1
                a_hit += right
            else:
                tot += 1
                hit += right
    return (hit / max(1, tot), a_hit / max(1, a_tot))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="scratch/extraction_full/raw.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--probes", type=int, default=10)
    ap.add_argument("--margin", type=float, default=0.8,
                    help="keep candidates scoring >= margin x reference score")
    ap.add_argument("--out", default="scratch/instruction_bank.json")
    args = ap.parse_args()

    import aiohttp
    rng = random.Random(7)
    items = probes(args.raw, args.probes, rng)
    print(f"{len(items)} probe items", flush=True)

    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}",
                     "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=180)) as s:
        cands = await generate(s, args.n)
        print(f"generated {len(cands)} candidates with a usable placeholder\n",
              flush=True)

        ref = INSTR_EXTRACT[0]
        ref_p, ref_a = await score(s, ref, items)
        ref_score = ref_p
        print(f"reference: present-fields {ref_p:.2f}  absent-fields {ref_a:.2f}\n",
              flush=True)

        print("planted controls (all should fall well below reference):")
        for name, ctrl in CONTROLS:
            cp, ca = await score(s, ctrl, items)
            print(f"  present {cp:.2f}  absent {ca:.2f}  {name:<18} {ctrl[:52]}",
                  flush=True)

        print("\ncandidates:", flush=True)
        keep = []
        sem = asyncio.Semaphore(8)

        async def one(c):
            async with sem:
                return (c, *await score(s, c, items))
        for c, cp, ca in await asyncio.gather(*[one(c) for c in cands]):
            # both axes must hold: present-field accuracy shows verbatim
            # copying survived, absent-field accuracy shows the null rule did
            if cp >= args.margin * ref_p and ca >= args.margin * max(ref_a, .5):
                keep.append(((cp + ca) / 2, c))
        keep.sort(reverse=True)
        for sc, c in keep[:30]:
            print(f"  {sc:.2f}  {c[:100]}")
        print(f"\nkept {len(keep)}/{len(cands)}")
        Path(args.out).write_text(json.dumps(
            {"reference_score": ref_score,
             "instructions": [c for _, c in keep]}, ensure_ascii=False, indent=1))
        print(f"-> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
