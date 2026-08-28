"""Generate and VALIDATE Danish paraphrases for the NER prompt's opening slot.

Openings are the free part of the prompt — the schema spec and key names are
the contract and never vary. But a paraphrase can still silently change the
task ("opsummér teksten" is a fine paraphrase of nothing we want), so every
candidate is validated BEHAVIOURALLY rather than by judging whether it "means
the same".

The test: build the full prompt with the candidate opening and the canonical
conditions, run it on gold examples through a strong model, and keep the
opening only if that model still produces correct, correctly-keyed output. A
phrasing that survives that is task-preserving by demonstration.

This is the same discipline as the control-gated verifiers: assert the
behaviour, don't trust the description.

Usage:
  python scripts/gen_ner_openings.py --n 30 --probe-rows 8
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from esperanto_lm.rl_rewards import (  # noqa: E402
    _NER_BUCKET_ORDER, _NER_GLOSS, _NER_KEY_FOR_BUCKET, _NER_PLURAL, _da_list,
    NER_COND_ONLYKEYS, NER_COND_EMPTY, NER_COND_VERBATIM,
    NER_OPENINGS, parse_ner,
)

GEN = "google/gemini-2.5-flash-lite"      # candidate author
ORACLE = "google/gemini-2.5-flash-lite"   # behavioural validator

ASK = """Du skal skrive danske instruktions-åbninger til en opgave om at finde
navngivne enheder i en tekst.

Hver åbning er en SKABELON der indeholder pladsholderen {{types}}, som senere
udfyldes med en dansk opremsning af de ønskede typer, fx "personer",
"personer og steder" eller "organisationer, steder og datoer".

Eksisterende skabeloner:
{examples}

Skriv {n} NYE, forskellige skabeloner. Krav:
- hver skabelon SKAL indeholde {{types}} præcis én gang
- {{types}} skal passe grammatisk uanset om der indsættes én eller flere typer
- ingen nævnelse af JSON, nøgler, formater eller anførselstegn
- må IKKE henvise til hvor teksten står (ikke "nedenfor", "herunder",
  "ovenfor") — teksten kan stå både før og efter instruktionen
- variér sætningstypen: brug både bydeform, spørgsmål, høflige
  forespørgsler og overskrifts-agtige formuleringer
- naturligt dansk

Svar med JSON: {{{{"openings": ["...", "..."]}}}}"""


def build_prompt(opening: str, buckets) -> str:
    bs = [b for b in _NER_BUCKET_ORDER if b in set(buckets)]
    keys = [_NER_KEY_FOR_BUCKET[b] for b in bs]
    # doubled braces: the returned string is a .format() template with a {t}
    # slot, so the JSON shape's own braces must be escaped
    shape = "{{" + ", ".join(f'"{k}": []' for k in keys) + "}}"
    gloss = ", ".join(_NER_GLOSS[b] for b in bs)
    conds = " ".join([NER_COND_ONLYKEYS[0], NER_COND_EMPTY[0], NER_COND_VERBATIM[0]])
    return (f'{opening}:\n\n"{{t}}"\n\n'
            f'Svar kun med JSON på formen {shape} — {gloss}. {conds}')


def key():
    for nm in ("or", "openrouter"):
        for p in (Path.home()/nm, Path.home()/f".{nm}"):
            if p.exists():
                return p.read_text().strip()
    return os.environ.get("OPENROUTER_API_KEY")


COST = {"v": 0.0, "calls": 0}


async def call(sess, model, prompt, max_tokens, json_mode=False):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0 if not json_mode else 0.0,
            "max_tokens": max_tokens}
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


def gold_of(row):
    CANON = {"PERSON": "person", "PER": "person", "ORGANIZATION": "org",
             "ORG": "org", "GPE": "sted", "LOCATION": "sted", "LOC": "sted",
             "FACILITY": "sted", "DATE": "dato"}
    out = set()
    for e in row["ents"] or []:
        raw = str(e.get("label", "")).upper()
        lab = CANON.get(raw) or CANON.get(raw.replace(" ", "_"))
        if lab:
            s = row["text"][e["start"]:e["end"]].strip()
            if s:
                out.add((s.lower(), lab))
    return out


async def main():
    import aiohttp
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="candidates per slot")
    ap.add_argument("--probe-rows", type=int, default=8)
    ap.add_argument("--min-f1", type=float, default=0.55,
                    help="oracle F1 floor for a candidate to be kept")
    ap.add_argument("--out", default="scratch/ner_openings.json")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("KennethEnevoldsen/dane_plus", split="dev")
    rng = random.Random(11)
    pool = [{"text": r["text"].strip(), "gold": gold_of(r)}
            for r in ds if r["text"].strip() and gold_of(r)]
    probe = rng.sample(pool, args.probe_rows)
    print(f"validating against {len(probe)} gold rows "
          f"({sum(len(p['gold']) for p in probe)} entities), oracle={ORACLE}\n",
          flush=True)

    K = key()
    assert K, "no OpenRouter key"
    async with aiohttp.ClientSession(
        headers={"Authorization": f"Bearer {K}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://claude-code-ner", "X-Title": "ner-openings"},
        timeout=aiohttp.ClientTimeout(total=240)) as sess:

        # baseline: how does the oracle do with the CANONICAL opening?
        async def score_opening(opening, buckets):
            tpl = build_prompt(opening, buckets)
            outs = await asyncio.gather(*[
                call(sess, ORACLE, tpl.format(t=p["text"]), 400) for p in probe])
            tp = fp = fn = 0
            for p, o in zip(probe, outs):
                pred = set(parse_ner(o or "") or [])
                g = {x for x in p["gold"] if x[1] in set(buckets)}
                tp += len(pred & g); fp += len(pred - g); fn += len(g - pred)
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            return (2 * pr * rc / (pr + rc)) if pr + rc else 0.0

        # every template is scored on TWO subsets — a single type and three —
        # so a template that only reads well for one arity is caught
        PROBE_SETS = [("person",), ("org", "sted", "dato")]

        async def score_template(tpl_str):
            fs = []
            for bset in PROBE_SETS:
                types = _da_list(_NER_PLURAL[b] for b in bset)
                fs.append(await score_opening(tpl_str.format(types=types), bset))
            return sum(fs) / len(fs)

        results = {}
        for slot, seeds, ask in [("templates",
                                  [o for o, _ in NER_OPENINGS], ASK)]:
            base = await score_template(seeds[0])
            print(f"[{slot}] canonical template oracle F1 = {base:.3f}", flush=True)

            raw = await call(sess, GEN,
                             ask.format(examples="\n".join("- " + x for x in seeds),
                                        n=args.n), 2000, json_mode=True)
            cands = []
            m = re.search(r"\{.*\}", raw or "", re.S)
            if m:
                try:
                    cands = [c.strip().rstrip(":").strip()
                             for c in json.loads(m.group(0)).get("openings", [])
                             if isinstance(c, str) and 15 < len(c.strip()) < 160]
                except Exception:
                    pass
            cands = [c for c in dict.fromkeys(cands)
                     if c not in seeds and c.count("{types}") == 1
                     and not re.search(r"nedenfor|herunder|ovenfor|nedenunder", c, re.I)]
            print(f"[{slot}] {len(cands)} candidates generated", flush=True)

            kept, rejected = [], []
            scores = await asyncio.gather(*[score_template(c) for c in cands])
            for c, f in zip(cands, scores):
                (kept if f >= args.min_f1 else rejected).append((c, round(f, 3)))
            kept.sort(key=lambda x: -x[1])
            print(f"[{slot}] kept {len(kept)}/{len(cands)} "
                  f"(F1 >= {args.min_f1})", flush=True)
            for c, f in kept[:8]:
                print(f"    {f:.3f}  {c}")
            if rejected:
                print(f"  rejected (task drift):")
                for c, f in sorted(rejected, key=lambda x: x[1])[:4]:
                    print(f"    {f:.3f}  {c}")
            results[slot] = {"canonical_f1": round(base, 3),
                             "kept": kept, "rejected": rejected}
            print(flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"cost ${COST['v']:.4f} over {COST['calls']} calls")
    print(f"-> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
