"""Danish extraction dataset from Wikipedia prose, with a per-passage schema.

Two phases, because the LLM half is slow and paid for and the rendering half is
cheap and worth re-running:

  --phase extract   two Gemini turns per passage, cached to raw.jsonl
                      turn 1: propose 3-6 fields BLIND (no values)
                      turn 2: fill them from the text, fresh context
  --phase render    raw.jsonl -> training rows across formats / key-namings /
                      prompt modes, gated on a canon() round-trip

Why two turns rather than one: proposing fields while already holding the
values invites picking whatever is easy to fill. Proposing blind then
extracting means ~25% of fields legitimately come back empty, which is the
abstention signal a fixed-schema set cannot produce.

Why this source rather than the existing sets: `textman_extraction` has ONE
schema across 20,018 rows and 26% of its `numbers` are not in the passage;
`danish-json-grpo-v1` has 134 field-sets over 9,815 rows on synthetic
passages. Here the schema is per-passage and the text is real prose.

Every value is checked to be a verbatim span. Values that are not are dropped
rather than trusted -- the smoke measured 97% compliance, so 3% would
otherwise be teaching invention.

Usage:
  python scripts/gen_extraction_da.py --phase extract --n 4000
  python scripts/gen_extraction_da.py --phase render --rows-per-passage 4
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_icl_schema_format import FORMATS, NULL, SYMBOLS, canon, render  # noqa: E402

MODEL = "google/gemini-2.5-flash-lite"
URL = "https://openrouter.ai/api/v1/chat/completions"
USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"

SYS_KEYS = """Du er en dansk informationsarkitekt.

Du får en dansk tekst. Foreslå 3-6 felter (nøgler), som en struktureret
opsummering af netop denne tekst burde have.

Regler:
- Nøglerne skal være danske, konkrete og passe til DENNE tekst.
- Et feltnavn er en KORT navneordsfrase på 1-3 ord. Ikke en sætning, ikke et
  spørgsmål, ikke "Definition af ...".
- Skriv feltnavne med lille begyndelsesbogstav.
- Vælg felter som teksten faktisk siger noget om.
- Angiv en type for hvert felt: "tekst", "tal", "dato" eller "liste".
- Du må IKKE udfylde værdier."""

SYS_VALUES = """Du er en præcis dansk informationsudtrækker.

Du får en dansk tekst og en liste af felter. Udfyld hvert felt fra teksten.

Regler:
- Hver værdi SKAL være en ordret sammenhængende tekststump fra teksten. Kopier
  præcis, uden at omskrive, forkorte eller normalisere.
- Kan et felt ikke besvares ordret fra teksten, så returner en tom liste for
  det felt. Opfind ALDRIG en værdi.
- Er typen "liste", er værdien en liste af ordrette tekststumper."""

SCHEMA_KEYS = {
    "type": "object",
    "properties": {"felter": {"type": "array", "items": {
        "type": "object",
        "properties": {"navn": {"type": "string"},
                       "type": {"type": "string",
                                "enum": ["tekst", "tal", "dato", "liste"]}},
        "required": ["navn", "type"], "additionalProperties": False}}},
    "required": ["felter"], "additionalProperties": False,
}
SCHEMA_VALUES = {
    "type": "object",
    "properties": {"felter": {"type": "array", "items": {
        "type": "object",
        "properties": {"navn": {"type": "string"},
                       "vaerdi": {"type": "array", "items": {"type": "string"}}},
        "required": ["navn", "vaerdi"], "additionalProperties": False}}},
    "required": ["felter"], "additionalProperties": False,
}

# a field name should be a short noun phrase; the smoke produced things like
# "Definition af dværgkanin" and "Årsager til dværgvækst", which are captions
KEY_OK = re.compile(r"^[a-zæøå][\wæøåÆØÅ /-]{1,28}$")
NUMRE = re.compile(r"^[\d\s.,%+-]+$")
MAX_VALUE_CHARS = 80


def _key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for p in (Path.home() / "or", Path.home() / ".or"):
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("no OpenRouter key (~/or)")


async def _call(session, sys_msg, user_msg, schema, temp):
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": sys_msg},
                         {"role": "user", "content": user_msg}],
            "temperature": temp,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "svar", "strict": True,
                                                "schema": schema}}}
    for attempt in range(4):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                d = await r.json()
                return json.loads(d["choices"][0]["message"]["content"])
        except Exception:
            await asyncio.sleep(1.5 * (attempt + 1))
    return None


async def extract_one(session, sem, pid, passage):
    async with sem:
        k = await _call(session, SYS_KEYS, passage, SCHEMA_KEYS, 0.9)
        if not k or not k.get("felter"):
            return None
        fields = [f for f in k["felter"] if KEY_OK.match(f["navn"].strip())]
        if len(fields) < 2:
            return None
        spec = "\n".join(f"- {f['navn']} ({f['type']})" for f in fields)
        v = await _call(session, SYS_VALUES,
                        f"Tekst:\n{passage}\n\nFelter:\n{spec}",
                        SCHEMA_VALUES, 0.2)
        if not v:
            return None
        got = {f["navn"]: f["vaerdi"] for f in v.get("felter", [])}
        out = []
        for f in fields:
            # verbatim gate: keep only spans that really occur in the passage
            # length cap: the smoke had 14% of values over 80 chars (max 329)
            # -- whole sentences rather than extraction spans
            vals = [x for x in got.get(f["navn"], [])
                    if x and x in passage and len(x) <= MAX_VALUE_CHARS]
            out.append({"navn": f["navn"].strip(), "type": f["type"],
                        "vaerdi": vals})
        return {"pid": pid, "passage": passage, "felter": out}


async def phase_extract(args):
    import aiohttp
    from datasets import load_dataset

    raw = args.out / "raw.jsonl"
    args.out.mkdir(parents=True, exist_ok=True)
    done = set()
    if raw.exists():
        for line in raw.open():
            try:
                done.add(json.loads(line)["pid"])
            except Exception:
                pass
        print(f"resuming: {len(done)} passages already extracted", flush=True)

    ds = load_dataset(args.source, split=args.split)
    col = args.text_col if args.text_col in ds.column_names else ds.column_names[0]
    todo = []
    for i, r in enumerate(ds):
        t = (r[col] or "").strip()
        pid = hashlib.md5(t.encode()).hexdigest()[:16]
        if pid in done or not (args.min_chars <= len(t) <= args.max_chars):
            continue
        todo.append((pid, t))
        if len(todo) >= args.n:
            break
    print(f"source={args.source}:{args.split} col={col!r}  to extract: {len(todo)}",
          flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    fh = raw.open("a")
    ok = 0
    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}",
                     "Content-Type": "application/json",
                     "X-Title": "extraction-da"},
            timeout=aiohttp.ClientTimeout(total=180)) as s:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            res = await asyncio.gather(*[extract_one(s, sem, p, t)
                                         for p, t in chunk])
            for r in res:
                if r:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    ok += 1
            fh.flush()
            print(f"  {min(i+args.batch, len(todo))}/{len(todo)}  kept {ok}",
                  flush=True)
    fh.close()
    print(f"\nextracted {ok}/{len(todo)} -> {raw}")


def phase_render(args):
    raw = args.out / "raw.jsonl"
    rows_in = [json.loads(l) for l in raw.open() if l.strip()]
    # Re-apply the value gates here as well as in extract: raw.jsonl may
    # predate a gate, and re-rendering a cached extraction should not
    # resurrect values a later rule would have dropped.
    dropped = 0
    for r in rows_in:
        for f in r["felter"]:
            keep = [v for v in f["vaerdi"]
                    if v in r["passage"] and len(v) <= MAX_VALUE_CHARS]
            dropped += len(f["vaerdi"]) - len(keep)
            f["vaerdi"] = keep
    print(f"{len(rows_in)} extracted passages "
          f"({dropped} values dropped by the re-applied gates)", flush=True)

    fmts = sorted(FORMATS)
    held_f = set(args.held_formats)
    seen_f = [f for f in fmts if f not in held_f]
    assert seen_f, "every format held out"
    rng = random.Random(args.seed)

    # partition on PASSAGE and on SCHEMA, so eval_both is unseen on both axes
    def bucket(s, mod):
        return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod

    out = defaultdict(list)
    stats = Counter()
    for r in rows_in:
        present = [f for f in r["felter"] if f["vaerdi"]]
        absent = [f for f in r["felter"] if not f["vaerdi"]]
        if not present:
            stats["no_present_fields"] += 1
            continue
        schema_key = "|".join(sorted(f["navn"] for f in r["felter"]))
        held_schema = bucket(schema_key, 10) < args.schema_heldout_pct
        held_pass = bucket(r["pid"], 10) < args.passage_heldout_pct

        for _ in range(args.rows_per_passage):
            k = rng.randint(1, min(len(present), 4))
            chosen = rng.sample(present, k)
            # absent fields are real abstention targets: the model must emit the
            # empty marker rather than invent, and we know they are empty
            chosen += rng.sample(absent, min(len(absent), rng.randint(0, 2)))
            names = [f["navn"] for f in chosen]
            sym = rng.random() < args.sym_frac
            scheme = rng.choice(sorted(SYMBOLS)) if sym else None
            km = ({n: SYMBOLS[scheme][i] for i, n in enumerate(names)}
                  if sym else {n: n for n in names})
            fmt = rng.choice(sorted(held_f)) if (held_schema or held_pass) and \
                rng.random() < 0.5 else rng.choice(seen_f)
            obj = {n: (f["vaerdi"] if len(f["vaerdi"]) != 1 else f["vaerdi"][0])
                   or None for n, f in zip(names, chosen)}
            ans = render(obj, names, km, fmt)
            keys = list(km.values())
            if canon(ans, fmt, keys) != canon(ans, fmt, keys):   # parser stable
                stats["unstable"] += 1
                continue
            if canon(ans, fmt, keys) is None:
                stats["unparseable"] += 1
                continue
            mode = rng.choices(["icl", "instruction", "both"],
                               weights=args.mode_frac)[0]
            args._scheme = scheme
            prompt = build_prompt(r["passage"], names, km, fmt, mode, rng,
                                  rows_in, args)
            split = ("eval_both" if held_schema and held_pass else
                     "eval_schema" if held_schema else
                     "eval_passage" if held_pass else "train")
            out[split].append({
                "messages": [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": ans}],
                "meta": {"format": fmt, "mode": mode,
                         "symbols": scheme or "none", "n_fields": len(names),
                         "schema": "|".join(names), "pid": r["pid"]}})
            stats[f"split:{split}"] += 1

    for split, rows in out.items():
        p = args.out / f"{split}.jsonl"
        with p.open("w") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"  {split:<14} {len(rows)}")
    print("\nstats:", dict(stats))


def build_prompt(passage, names, km, fmt, mode, rng, pool, args):
    """Every block -- demonstrations and query alike -- carries its OWN text and
    its OWN key list. The demonstrations therefore teach the task (extract the
    named fields, in this format, empty marker when absent), not a particular
    schema; the keys ride along with each item.

    An earlier version relabelled each demonstration's fields with the QUERY's
    key names, which produced demos where a bridge article answered under
    `ordensnavn`/`familier`. It also made per-passage schemas look like a
    blocker for ICL when they are the point: each demo showing a different key
    set is what teaches "read the keys you were given" -- the axis
    textman_extraction could not teach, because its four keys never varied.
    """
    def block(text, ns, keymap, obj, with_answer):
        spec = ", ".join(keymap[n] for n in ns)
        head = f"Tekst:\n{text}\nFelter: {spec}\nSvar:"
        return head + (f"\n{render(obj, ns, keymap, fmt)}" if with_answer else "")

    parts = []
    if mode in ("instruction", "both"):
        parts.append(f"Udtræk de angivne felter fra hver tekst. Svar i samme "
                     f"format som felterne er navngivet. Er et felt ikke nævnt "
                     f"i teksten, så skriv {NULL}.")
    if mode in ("icl", "both"):
        demos = []
        for other in rng.sample(pool, min(len(pool), args.shots + 6)):
            if other["passage"] == passage:
                continue
            pres = [f for f in other["felter"] if f["vaerdi"]]
            if len(pres) < 2:
                continue
            dn = [f["navn"] for f in pres[:4]]
            # the demo keeps its own names, or its own symbol assignment --
            # never the query's
            dkm = ({n: SYMBOLS[km_scheme][i] for i, n in enumerate(dn)}
                   if (km_scheme := args._scheme) else {n: n for n in dn})
            dobj = {n: (f["vaerdi"] if len(f["vaerdi"]) != 1 else f["vaerdi"][0])
                    for n, f in zip(dn, pres)}
            demos.append(block(other["passage"][:700], dn, dkm, dobj, True))
            if len(demos) >= args.shots:
                break
        if demos:
            parts.append("Eksempler:\n\n" + "\n\n".join(demos))
    parts.append(block(passage, names, km, {}, False))
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["extract", "render"], required=True)
    ap.add_argument("--source", default="jensjepsen/danish-vital-stem-da-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--out", type=Path, default=Path("scratch/extraction_da"))
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--batch", type=int, default=120)
    ap.add_argument("--min-chars", type=int, default=350)
    ap.add_argument("--max-chars", type=int, default=1800)
    ap.add_argument("--rows-per-passage", type=int, default=4)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--sym-frac", type=float, default=0.35)
    ap.add_argument("--mode-frac", nargs=3, type=float, default=[0.5, 0.2, 0.3])
    ap.add_argument("--held-formats", nargs="*",
                    default=["kv_eq", "bracket_pair", "brace_pair"])
    ap.add_argument("--schema-heldout-pct", type=int, default=2)
    ap.add_argument("--passage-heldout-pct", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    if args.phase == "extract":
        asyncio.run(phase_extract(args))
    else:
        phase_render(args)


if __name__ == "__main__":
    main()
