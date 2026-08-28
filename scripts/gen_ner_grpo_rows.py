"""Build GRPO NER rows with a FRESHLY GENERATED instruction per row.

Per row: sample a dane_plus sentence and a subset of entity types, have an LLM
write a new instruction for THAT text and THOSE types, then rejection-sample —
run the instruction through an oracle and keep it only if the oracle's answer
scores above threshold against that row's own gold.

Why per row rather than a bank of phrasings: the policy has twice keyed off
surface tokens instead of reading the instruction — it converged on a
memorised {person, places, dates, numbers} object, and renaming one key
"org"->"organisation" moved org recall 0%->21.7% while explaining the concept
moved it 0%->0%. A shared bank still gives every row one of N fixed prefixes.
A per-row instruction gives the policy nothing constant to latch onto.

Rejection sampling also localises risk: a bad phrasing in a 22-item bank
corrupts ~4.5% of the dataset, a bad phrasing here corrupts one row — and the
oracle check catches it first.

The TYPE SUBSET and the JSON KEY NAMES are chosen by us and pinned in the
generation prompt, because the verifier keys off them. Everything else about
the instruction is the model's.

Usage:
  python scripts/gen_ner_grpo_rows.py --n 2400 --out data/ner_grpo_rows.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from esperanto_lm.rl_rewards import (  # noqa: E402
    _NER_BUCKET_ORDER, _NER_GLOSS, _NER_KEY_FOR_BUCKET, _NER_PLURAL,
    _da_list, ner_prompt, reward_ner,
)

GEN = "google/gemini-2.5-flash-lite"
ORACLE = "google/gemini-2.5-flash-lite"

CANON = {"PERSON": "person", "PER": "person", "ORGANIZATION": "org",
         "ORG": "org", "GPE": "sted", "LOCATION": "sted", "LOC": "sted",
         "FACILITY": "sted", "DATE": "dato"}

# The required clauses are shuffled per call and a STYLE is sampled, because
# with a fixed clause order the model reproduces it: a first pass produced
# "Identificer venligst..." in 8/33 rows and "venligst" in 16/33, all with the
# same identify -> classify -> JSON -> empty -> verbatim ordering. Unique
# strings, but not structurally different instructions — and a recurring prefix
# is exactly what the policy latches onto.
STYLES = [
    "en kort, kontant bydeform uden høflighedsord",
    "et direkte spørgsmål til modellen",
    "en høflig anmodning",
    "en overskrift efterfulgt af korte punkter",
    "en teknisk specifikation i nøgtern tone",
    "en henvendelse i du-form",
    "en kort systeminstruktion uden indledende høflighed",
    "en opgavebeskrivelse der starter med formålet",
]

CLAUSES = [
    "den skal angive JSON-formen {shape} og hvilken type der hører til hvilken nøgle",
    "den skal sige at en tom liste bruges hvis en type ikke forekommer",
    "den skal sige at enhederne skal skrives ordret som i teksten",
    "den skal nævne præcis de ønskede typer, ingen andre",
]

ASK = """Du skriver en instruktion på dansk til en sprogmodel.

Modellen skal finde navngivne enheder i denne tekst:
---
{text}
---

Instruktionen skal bede om PRÆCIS disse typer: {types}.
Svaret skal afgives som JSON med præcis disse nøgler: {keys}.

Skriv EN instruktion i denne stil: {style}.

Krav (i vilkårlig rækkefølge i din instruktion):
{clauses}

Desuden:
- skriv IKKE selve teksten ind i instruktionen, og afslør IKKE hvilke enheder
  der findes — instruktionen skal fungere for enhver tekst
- undgå at starte med de samme ord som en typisk instruktion; variér
  sætningsbygning og ordvalg
- kun instruktionen, ingen forklaring

Svar med JSON: {{"instruction": "..."}}"""


def gold_of(row):
    out = []
    for e in row["ents"] or []:
        raw = str(e.get("label", "")).upper()
        lab = CANON.get(raw) or CANON.get(raw.replace(" ", "_"))
        if lab:
            s = row["text"][e["start"]:e["end"]].strip()
            if s:
                out.append((s, lab))
    return out


def key():
    for nm in ("or", "openrouter"):
        for p in (Path.home()/nm, Path.home()/f".{nm}"):
            if p.exists():
                return p.read_text().strip()
    return os.environ.get("OPENROUTER_API_KEY")


COST = {"v": 0.0, "calls": 0}


async def call(sess, model, prompt, max_tokens, json_mode=False, temp=1.0):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temp, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    for _ in range(4):
        try:
            async with sess.post("https://openrouter.ai/api/v1/chat/completions",
                                 json=body) as r:
                if r.status != 200:
                    await asyncio.sleep(2)
                    continue
                d = await r.json()
                COST["v"] += float((d.get("usage") or {}).get("cost", 0) or 0)
                COST["calls"] += 1
                return d["choices"][0]["message"]["content"]
        except Exception:
            await asyncio.sleep(2)
    return None


PROG = {"done": 0, "kept": 0}


async def make_row(sess, sem, row, buckets, gold_json, args, total):
    bs = [b for b in _NER_BUCKET_ORDER if b in set(buckets)]
    keys = [_NER_KEY_FOR_BUCKET[b] for b in bs]
    shape = "{" + ", ".join(f'"{k}": []' for k in keys) + "}"
    rrng = random.Random(hash(row["text"]) & 0xFFFFFFFF)
    clauses = [c.format(shape=shape) for c in CLAUSES]
    rrng.shuffle(clauses)
    ask = ASK.format(text=row["text"], types=_da_list(_NER_PLURAL[b] for b in bs),
                     keys=", ".join(f'"{k}"' for k in keys),
                     style=rrng.choice(STYLES),
                     clauses="\n".join("- " + c for c in clauses))

    best = None
    for attempt in range(1, args.retries + 1):
        async with sem:
            raw = await call(sess, GEN, ask, 700, json_mode=True)
        instr = None
        m = re.search(r"\{.*\}", raw or "", re.S)
        if m:
            try:
                instr = (json.loads(m.group(0)).get("instruction") or "").strip()
            except Exception:
                instr = None
        # mechanical gate: must name every requested key, no unrequested ones,
        # and must not have smuggled the passage in
        if not instr or not (60 < len(instr) < 900):
            continue
        low = instr.lower()
        if any(f'"{k}"' not in instr and k not in low for k in keys):
            continue
        unwanted = [_NER_KEY_FOR_BUCKET[b] for b in _NER_BUCKET_ORDER if b not in bs]
        if any(f'"{u}"' in instr for u in unwanted):
            continue
        if row["text"][:40].lower() in low:
            continue

        prompt = f'{instr}\n\n"{row["text"]}"'
        async with sem:
            out = await call(sess, ORACLE, prompt, 400)
        score = reward_ner(out or "", gold_json)
        if best is None or score > best[1]:
            best = (instr, score, attempt)
        if score >= args.min_score:
            break

    PROG["done"] += 1
    if PROG["done"] % 100 == 0 or PROG["done"] == total:
        print(f"  {PROG['done']}/{total} rows  kept={PROG['kept']}  "
              f"${COST['v']:.3f}", flush=True)
    if best and best[1] >= args.min_score:
        PROG["kept"] += 1
        return {"text": row["text"], "ents": row["ents_pairs"],
                "buckets": bs, "instruction": best[0],
                "oracle_score": round(best[1], 4), "attempts": best[2]}
    return {"text": row["text"], "ents": row["ents_pairs"], "buckets": bs,
            "instruction": None,
            "oracle_score": round(best[1], 4) if best else 0.0,
            "attempts": args.retries}


async def main():
    import aiohttp
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=0, help="0 = whole split")
    ap.add_argument("--empty-frac", type=float, default=0.28)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=0.75,
                    help="reward_ner floor the oracle must reach for the "
                         "instruction to be accepted")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/ner_grpo_rows.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("KennethEnevoldsen/dane_plus", split=args.split)
    rng = random.Random(args.seed)
    with_e, no_e = [], []
    for r in ds:
        t = (r.get("text") or "").strip()
        if not t:
            continue
        pairs = gold_of(r)
        rec = {"text": t, "ents_pairs": [[s, l] for s, l in pairs]}
        (with_e if pairs else no_e).append(rec)
    if args.empty_frac >= 0 and with_e:
        want = int(round(len(with_e) * args.empty_frac / max(1e-6, 1 - args.empty_frac)))
        no_e = rng.sample(no_e, min(want, len(no_e)))
    rows = with_e + no_e
    rng.shuffle(rows)
    if args.n:
        rows = rows[:args.n]

    # Sample WITH REPLACEMENT so the row count is not capped by the source
    # (2357 sentences). Two reasons this is better than letting
    # interleave_datasets cycle the split: a cycle repeats the IDENTICAL
    # prompt, whereas a redraw gets a fresh instruction and a different type
    # subset — and the same sentence under two different subsets has two
    # different correct answers, which is direct pressure to read the schema
    # rather than emit a memorised object.
    #
    # A given (text, subset) pair is never drawn twice; once a sentence has
    # used all subsets compatible with its gold, it stops being redrawn.
    SIZES, W = [1, 2, 3, 4], [0.10, 0.20, 0.30, 0.40]
    ALL_SUBSETS = []
    for m in range(1, 16):
        bs = tuple(b for i, b in enumerate(_NER_BUCKET_ORDER) if m >> i & 1)
        ALL_SUBSETS.append(bs)

    # Balance the empty share at the JOB level, not the sentence level.
    # Entity-bearing sentences yield several jobs (one per subset) while an
    # empty sentence yields interchangeable ones, so a sentence-level 28% came
    # out as 16% of jobs once redraws were capped, and as far more when they
    # were not. Drawing the two pools to an explicit job quota makes the number
    # land where it is set — which matters because the policy is already an
    # 82-93% abstainer and the abstention plateau is the failure this whole
    # --empty-frac knob exists to control.
    target = args.n or len(rows)
    want_empty = int(round(target * max(0.0, args.empty_frac)))
    pools = {"empty": [r for r in rows if not r["ents_pairs"]],
             "full": [r for r in rows if r["ents_pairs"]]}
    quota = {"empty": want_empty, "full": target - want_empty}

    used = {}                      # text -> set of subsets already drawn
    jobs, guard = [], 0
    while len(jobs) < target and guard < target * 60:
        guard += 1
        kind = "empty" if quota["empty"] > 0 and (
            quota["full"] <= 0 or rng.random() < quota["empty"] /
            max(1, quota["empty"] + quota["full"])) else "full"
        if not pools[kind]:
            quota[kind] = 0
            continue
        r = rng.choice(pools[kind])
        have = {l for _, l in r["ents_pairs"]}
        seen = used.setdefault(r["text"], set())
        # subsets that keep this row non-degenerate and are not already drawn
        cands = [b for b in ALL_SUBSETS
                 if b not in seen and (not have or (have & set(b)))]
        if not cands:
            continue
        k = rng.choices(SIZES, weights=W)[0]
        sized = [b for b in cands if len(b) == k] or cands
        bs = list(rng.choice(sized))
        seen.add(tuple(bs))
        quota[kind] -= 1
        ents = [[sfc, l] for sfc, l in r["ents_pairs"] if l in bs]
        gold_json = json.dumps({"ents": ents, "buckets": bs}, ensure_ascii=False)
        jobs.append((r, bs, gold_json))

    _reuse = Counter(len(v) for v in used.values() if v)
    _empty_jobs = sum(1 for _, _, gj in jobs if not json.loads(gj)["ents"])
    print(f"  sampled with replacement: {len(jobs)} jobs from "
          f"{sum(1 for v in used.values() if v)} distinct sentences  "
          f"(draws per sentence: {dict(sorted(_reuse.items()))})", flush=True)
    print(f"  entity-free jobs: {_empty_jobs}/{len(jobs)} = "
          f"{100*_empty_jobs/max(1,len(jobs)):.0f}%  (target {100*args.empty_frac:.0f}%)",
          flush=True)

    print(f"{len(jobs)} rows  (retries={args.retries}, min_score={args.min_score}, "
          f"gen={GEN}, oracle={ORACLE})", flush=True)

    K = key()
    assert K, "no OpenRouter key"
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    async with aiohttp.ClientSession(
        headers={"Authorization": f"Bearer {K}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://claude-code-ner", "X-Title": "ner-grpo-rows"},
        timeout=aiohttp.ClientTimeout(total=300)) as sess:
        out = await asyncio.gather(*[
            make_row(sess, sem, r, bs, gj, args, len(jobs)) for r, bs, gj in jobs])
    dt = time.time() - t0

    kept = [o for o in out if o["instruction"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for o in kept:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")

    uniq = len({o["instruction"] for o in kept})
    att = Counter(o["attempts"] for o in kept)
    sizes = Counter(len(o["buckets"]) for o in kept)
    print(f"\nkept {len(kept)}/{len(out)} in {dt:.0f}s "
          f"({60*len(kept)/max(dt,1):.0f} rows/min)")
    print(f"unique instructions: {uniq}/{len(kept)}")
    print(f"attempts: {dict(sorted(att.items()))}")
    print(f"subset sizes: {dict(sorted(sizes.items()))}")
    print(f"cost ${COST['v']:.4f} over {COST['calls']} calls "
          f"(${1000*COST['v']/max(1,len(kept)):.2f}/1k rows)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
