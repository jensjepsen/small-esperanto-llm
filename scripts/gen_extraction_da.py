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

# Register mix. The generator previously read one wiki-STEM corpus, so every
# passage was an encyclopedic lead paragraph -- the model could learn
# "Wikipedia lead" rather than "extraction". These are dynaword `source`
# values, grouped by register, with a share each.
#
# Conversation, subtitles, social media and poetry are deliberately absent:
# dialogue and verse have little to extract, so the proposer would spend calls
# on fields that come back empty.
REGISTERS = {
    "encyclopedic": (["wikipedia", "wikisource"],                     0.30),
    # ncc_* are Norwegian Colossal Corpus derivatives and carry Norwegian
    # text despite being in a Danish corpus: 43/300 sampled ncc_newspaper docs
    # match nei|jeg heter|hva|ikkje against only 10 with Danish equivalents.
    # tv2r and nordjyllandnews sample clean (0/300).
    "news":         (["tv2r", "nordjyllandnews"],                     0.25),
    "legal":        (["retsinformationdk", "domsdatabasen",
                      "retspraksis", "skat", "fm-udgivelser"],        0.20),
    "web_admin":    (["ai-aktindsigt", "miljoeportalen"],             0.15),
    "medical":      (["health_hovedstaden"],                          0.05),
    "books":        (["memo", "adl"],                                 0.05),
}

MODEL = "google/gemini-2.5-flash-lite"
# bump when SYS_KEYS / SYS_VALUES change, so a mixed cache is detectable
PROMPT_V = 2
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
# Length is NOT a correctness constraint. An 80-char cap was imposed on the
# aesthetic view that "extraction values should be short spans"; measured, it
# was the direct cause of definitional fields coming back empty 60-100% of the
# time (`definition`, `beskrivelse`), because a passage's definition of a term
# is a sentence, not a phrase. That taught a shortcut: emit the empty marker
# whenever the field name looks abstract, without reading the passage.
#
# What actually matters is verifiability: the value must be verbatim, must not
# contain a newline (it would break the line-based formats), and must survive a
# render/canon round trip. Checked against 33 long values (median 123, max 233
# chars): none contained a newline and all ten formats round-tripped them.
MAX_VALUE_CHARS = 400          # backstop against a runaway paste, not a shape rule
NEWLINE = re.compile(r"[\r\n]")
_WS = re.compile(r"\s+")


def _ws(t):
    return _WS.sub(" ", t).strip()


def is_verbatim(value, passage):
    """Verbatim up to whitespace.

    An exact `value in passage` test is stricter than the scorer: canon()
    collapses whitespace when parsing, and hard-wrapped sources put newlines
    mid-phrase. Measured on the register pilot, that alone accounted for the
    books register's 76.3% verbatim rate against 94-99% elsewhere -- the model
    was copying correctly from text that reads
    'en af \\nkunstnernes ateliers'.
    """
    return _ws(value) in _ws(passage)

# Danish prose spells numbers out ("Tre", "ni") and qualifies them
# ("4,22 millioner", "5,5 millioner ar siden", "1950'erne"). A strict numeric
# test rejected 19.4% of `tal` values in the smoke -- all of them CORRECT
# verbatim answers. So `tal` requires a numeric SIGNAL, not a bare number:
# digits, or a spelled-out Danish numeral. A value with neither is a genuine
# mislabel and is dropped.
_DA_NUMWORD = (r"nul|en|et|to|tre|fire|fem|seks|syv|otte|ni|ti|elleve|tolv|"
               r"tretten|fjorten|femten|seksten|sytten|atten|nitten|tyve|"
               r"tredive|fyrre|halvtreds|tres|halvfjerds|firs|halvfems|"
               r"hundrede|tusind|million|milliard|billion")
HAS_NUM = re.compile(r"(\d|\b(" + _DA_NUMWORD + r")\w*)", re.IGNORECASE)
HAS_DATE = re.compile(r"\d")


def type_ok(t, v):
    """A value must carry the signal its declared type implies."""
    if t == "tal":
        return bool(HAS_NUM.search(v))
    if t == "dato":
        return bool(HAS_DATE.search(v))
    return True


def chunk_article(text, lo, hi):
    """Split an article into passage-sized chunks on paragraph boundaries.

    The corpus is full Wikipedia articles (median 10,941 chars), so a length
    FILTER keeps only 3% -- and biases that 3% toward stubs, the articles with
    least to extract. Chunking instead gives several DIFFERENT schemas per
    article: a physics article's history section proposes different fields than
    its maths section.

    Bounds are enforced on the ACTUAL joined length, not a running counter. A
    first version tracked `len(para) + 2` per paragraph and checked the limit
    before appending, which let chunks out at 349 and 1784 chars against a
    [350, 1500] window -- close enough to look right in a spot check.
    """
    out, buf = [], []

    def flush():
        if buf:
            joined = "\n\n".join(buf)
            if len(joined) >= lo:
                out.append(joined)
        buf.clear()

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para or len(para) > hi:
            continue                      # oversized paragraph: not splittable
        if buf and len("\n\n".join(buf + [para])) > hi:
            flush()
        buf.append(para)
    flush()
    return out


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
        # Store what the model RETURNED, ungated. Gates run at render time
        # only. An earlier version filtered here, which permanently discarded
        # values a later rule would have kept -- lifting the 80-char cap then
        # meant re-paying for the whole extraction. Raw output is the expensive
        # artifact; gate decisions are cheap and revisable.
        return {"pid": pid, "passage": passage,
                "felter": [{"navn": f["navn"].strip(), "type": f["type"],
                            "vaerdi": got.get(f["navn"], [])} for f in fields],
                "meta": {"model": MODEL, "prompt_v": PROMPT_V}}


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

    todo = []
    if args.registers:
        # Per-source parquet, one file per dynaword source:
        #   data/tv2r/data.parquet, data/retsinformationdk/data.parquet, ...
        # so the FILE is the filter and only the ~14 sources in REGISTERS are
        # read. Loading the whole dataset instead cost 28GB of arrow cache and
        # 25GB resident on a 31GB box before it was killed -- it materialises
        # 7.4M rows to filter on a column that the directory layout already
        # encodes.
        want = ({reg: max(1, args.n // len(REGISTERS)) for reg in REGISTERS}
                if args.uniform else
                {reg: int(args.n * share) for reg, (_, share) in REGISTERS.items()})
        got = Counter()
        for reg, (srcs, _) in REGISTERS.items():
            per_src = max(1, want[reg] // len(srcs))
            for src in srcs:
                if got[reg] >= want[reg]:
                    break
                try:
                    ds = load_dataset(args.source,
                                      # data.parquet only: the directory also
                                      # holds metadata.parquet with a different
                                      # schema, and a *.parquet glob picks up
                                      # both -> CastError mid-generation
                                      data_files=f"data/{src}/data.parquet",
                                      split="train")
                except Exception as e:
                    print(f"    [{reg}/{src}] unavailable: "
                          f"{type(e).__name__}", flush=True)
                    continue
                n_before = got[reg]
                for r in ds:
                    if got[reg] >= want[reg] or got[reg] - n_before >= per_src:
                        break
                    t = (r.get(args.text_col) or "").strip()
                    if not t:
                        continue
                    for piece in chunk_article(
                            t, args.min_chars,
                            args.max_chars)[:args.max_chunks_per_article]:
                        if got[reg] >= want[reg]:
                            break
                        pid = hashlib.md5(piece.encode()).hexdigest()[:16]
                        if pid in done:
                            continue
                        todo.append((pid, piece, reg))
                        got[reg] += 1
                print(f"    [{reg}/{src}] +{got[reg]-n_before} "
                      f"(source has {len(ds):,} docs)", flush=True)
                del ds
        print(f"  register mix: {dict(got)}", flush=True)
    else:
        ds = load_dataset(args.source, split=args.split)
        col = args.text_col if args.text_col in ds.column_names else ds.column_names[0]
        n_art = 0
        for r in ds:
            t = (r[col] or "").strip()
            if not t:
                continue
            n_art += 1
            pieces = (chunk_article(t, args.min_chars, args.max_chars)
                      if args.chunk else
                      ([t] if args.min_chars <= len(t) <= args.max_chars else []))
            for piece in pieces[:args.max_chunks_per_article]:
                pid = hashlib.md5(piece.encode()).hexdigest()[:16]
                if pid in done:
                    continue
                todo.append((pid, piece, "encyclopedic"))
            if len(todo) >= args.n:
                break
        print(f"  {n_art} articles -> {len(todo)} passages", flush=True)
    todo = todo[:args.n]
    print(f"source={args.source}  to extract: {len(todo)} passages", flush=True)

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
                                         for p, t, _ in chunk])
            for r, (_, _, reg) in zip(res, chunk):
                if r:
                    r["meta"]["register"] = reg
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
    why = Counter()
    for r in rows_in:
        for f in r["felter"]:
            keep, seen_v = [], set()
            for v in f["vaerdi"]:
                v = _ws(v)
                if not v or v in seen_v:
                    why["duplicate/empty"] += 1
                elif not is_verbatim(v, r["passage"]):
                    why["not-verbatim"] += 1
                elif NEWLINE.search(v):
                    why["newline"] += 1
                elif len(v) > MAX_VALUE_CHARS:
                    why["too-long"] += 1
                elif not type_ok(f["type"], v):
                    why[f"type:{f['type']}"] += 1
                else:
                    seen_v.add(v)
                    keep.append(v)
                    continue
            f["vaerdi"] = keep
    print(f"{len(rows_in)} extracted passages; gate drops: {dict(why)}", flush=True)

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
    ap.add_argument("--uniform", action="store_true",
                    help="equal quota per register instead of REGISTERS shares")
    ap.add_argument("--registers", action="store_true",
                    help="draw a register-balanced mix from dynaword "
                         "(implies --source danish-foundation-models/"
                         "danish-dynaword). Without it, one corpus is read "
                         "whole and every passage is the same register.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--out", type=Path, default=Path("scratch/extraction_da"))
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--batch", type=int, default=120)
    ap.add_argument("--min-chars", type=int, default=350)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--chunk", action="store_true", default=True,
                    help="split articles on paragraph boundaries (default). "
                         "--no-chunk filters by length instead, which keeps 3%% "
                         "of this corpus and biases toward stubs.")
    ap.add_argument("--no-chunk", dest="chunk", action="store_false")
    ap.add_argument("--max-chunks-per-article", type=int, default=3,
                    help="cap per article so a few long articles cannot "
                         "dominate the passage pool")
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
