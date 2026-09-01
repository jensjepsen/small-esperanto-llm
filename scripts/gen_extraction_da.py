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
import time
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
# Blank markers for the reverse (fill-the-gap) task. Deliberately many and
# structurally different: a single marker teaches the marker, not the task, and
# `___` in particular collides with OCR artefacts in the books register. The
# {i} slot is filled with a 1-based index when numbering is on, so a passage
# with several gaps can name which is which.
BLANKS = [
    "____", "_____", "[...]", "[BLANK]", "[MANGLER]", "<mangler>", "###",
    "***", "(?)", "<?>", "{{}}", "[__]", "«...»", "…", "[X]", "[FELT]",
    "[UDFYLD]", "<udfyld>", "[?]", "\u2588\u2588\u2588",
]
BLANKS_NUM = [
    "[{i}]", "[BLANK {i}]", "[MANGLER {i}]", "<{i}>", "___{i}___",
    "[FELT {i}]", "(({i}))", "#{i}#", "[UDFYLD {i}]", "<felt{i}>",
]

# Prompt-surface variation. Everything here is Danish -- no English labels,
# since the model is Danish-only and an English scaffold would be a token the
# task never actually appears with.
#
# Before this there were exactly TWO instruction sentences across 60,376 rows
# and one fixed set of labels, so the wording was more uniform than
# textman_extraction's (whose three commonest phrasings covered only 80 of
# 400 rows). A model can key on the literal string instead of the task, which
# is the surface-token dependence that made `org` unreachable in the NER work.
#
# Labels are drawn per row but held CONSISTENT within a row, so demonstrations
# and query agree -- varying them inside one prompt would teach noise.
L_TEXT = ["Tekst", "Kildetekst", "Uddrag", "Tekststykke", "Afsnit",
          "Tekstuddrag"]
L_TEXT_GAP = ["Tekst med huller", "Tekst med manglende dele", "Hullet tekst",
              "Tekst med tomme felter", "Ufuldstændig tekst",
              "Tekst med udeladelser"]
L_FIELDS = ["Felter", "Nøgler", "Feltnavne", "Ønskede felter",
            "Felter der skal udtrækkes", "Efterspurgte felter"]
L_ANSWER = ["Svar", "Resultat", "Uddrag", "Udtræk", "Besvarelse",
            "Oplysninger"]
L_FILL = ["Udfyld", "Udfyldning", "Manglende tekst", "Svar", "Indsæt"]
L_DEMOS = ["Eksempler", "Eksempler på opgaven", "Løste eksempler",
           "Her er nogle eksempler", "Demonstrationer"]

def load_instr_bank():
    """Hand-written shapes only.

    A generated bank of 181 was produced and read. It gave lexical variety
    (~40 distinct verbs) but a single structure: essentially every one was
    `[verb] ordret [felterne]. [mangel-clause] {null}.` It also carried
    grammatical slips -- five uninflected `ordret` where `ordrette` is
    required, `feltsværdier`, a dangling `der svarer til` -- so it would have
    needed hand-checking anyway, which is most of the work of writing them.

    The 34 hand-written ones span nine shapes the generator never produced:
    questions, telegraphic fragments, numbered rules, null-rule-first,
    role framing, negative framing, explanatory paragraphs.
    """
    try:
        from instruction_shapes_da import ALL as _shapes
        return _shapes
    except Exception:
        return INSTR_EXTRACT


INSTR_EXTRACT = [
    "Udtræk de angivne felter fra hver tekst. Svar i samme format som "
    "felterne er navngivet. Er et felt ikke nævnt i teksten, så skriv {null}.",
    "Find værdien for hvert af de nævnte felter i teksten. Kopier ordret. "
    "Mangler et felt i teksten, skriv {null}.",
    "Læs teksten og udfyld de efterspurgte felter med ordrette tekststumper. "
    "Brug {null} for felter teksten ikke nævner.",
    "Nedenfor er en tekst og en liste af felter. Angiv hvert felts værdi, "
    "præcis som det står i teksten. Ukendte felter markeres med {null}.",
    "Udfyld felterne ud fra teksten. Værdier skal være ordret afskrift. "
    "Er oplysningen der ikke, så skriv {null}.",
    "Gennemgå teksten og hent de oplysninger, felterne beder om. Skriv dem "
    "af ordret. Findes en oplysning ikke, skriv {null}.",
    "Til hver tekst hører nogle felter. Udfyld dem med ordret tekst fra "
    "passagen, og brug {null} hvor teksten intet siger.",
    "Uddrag de nævnte oplysninger fra teksten. Gengiv dem ordret. "
    "Manglende oplysninger angives som {null}.",
]
INSTR_FILL = [
    "Teksten mangler nogle stykker. Udfyld hvert hul med den ordrette tekst, "
    "der hører til det angivne felt. Svar med en linje per hul.",
    "Der er fjernet nogle tekststumper. Genskab hver enkelt ud fra feltnavnet "
    "og sammenhængen. En linje per hul.",
    "Nogle passager er erstattet af markeringer. Angiv for hver markering "
    "hvilken tekst der manglede. Svar med en linje per hul.",
    "Udfyld hullerne i teksten. Hvert hul svarer til et af de nævnte felter. "
    "Skriv en linje per hul.",
    "Teksten er ufuldstændig. Gengiv den manglende tekst for hvert hul, "
    "en linje ad gangen.",
    "Hullerne nedenfor dækker over konkrete tekststumper. Skriv hvad der "
    "hørte hjemme i hvert hul, en linje per hul.",
]

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
FAILS = Counter()
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
    last = None
    for attempt in range(4):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    last = f"HTTP {r.status}"
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                d = await r.json()
                return json.loads(d["choices"][0]["message"]["content"])
        except Exception as e:
            last = type(e).__name__
            await asyncio.sleep(1.5 * (attempt + 1))
    # Exhausted retries. Reported, not swallowed: a silent None is
    # indistinguishable from a hang, and a 17-minute stall at concurrency 100
    # looked exactly like one -- no rows, no errors, connections open.
    FAILS[last or "unknown"] += 1
    n = sum(FAILS.values())
    if n <= 5 or n % 50 == 0:
        print(f"  [call failed after 4 tries: {last}]  total failures={n} "
              f"{dict(FAILS)}", flush=True)
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
                        # Count EVERY candidate against the quota, including
                        # ones already extracted -- then drop the done ones
                        # from todo. Counting only new passages made the
                        # candidate set grow on every rerun: already-done work
                        # did not fill the quota, so the sampler walked deeper
                        # into each source and collected fresh passages instead
                        # of the ones it had missed. A rerun meant to recover
                        # 2,325 dropped passages queued 18,758 new ones.
                        #
                        # With this, the candidate set is identical on every
                        # run for the same --n/--uniform, so todo is exactly
                        # (candidates - done) and a rerun recovers the gaps.
                        got[reg] += 1
                        if pid in done:
                            continue
                        todo.append((pid, piece, reg))
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
    lock = asyncio.Lock()
    t0 = time.time()

    async def run_one(s, pid, text, reg):
        """Write as each passage completes -- no batch barrier.

        The previous version gathered a fixed chunk of 120 and waited for all
        of them before writing, printing and starting the next chunk. With
        concurrency 100 that drains the pool at every boundary: 100 start, 20
        trickle in, then everything idles waiting for the slowest straggler.
        The chunking existed only to get incremental flushes and a progress
        line, and neither needs a barrier.
        """
        nonlocal ok
        r = await extract_one(s, sem, pid, text)
        if not r:
            return
        r["meta"]["register"] = reg
        async with lock:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            ok += 1
            if ok % 200 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"  {ok}/{len(todo)} kept  {ok/el:.1f}/s  "
                      f"eta {(len(todo)-ok)/max(ok/el, .01)/60:.0f}min",
                      flush=True)

    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}",
                     "Content-Type": "application/json",
                     "X-Title": "extraction-da"},
            timeout=aiohttp.ClientTimeout(total=180)) as s:
        await asyncio.gather(*[run_one(s, pid, text, reg)
                               for pid, text, reg in todo])
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
    args._bank = load_instr_bank()
    print(f"instruction bank: {len(args._bank)} hand-written instructions",
          flush=True)

    # partition on PASSAGE and on SCHEMA, so eval_both is unseen on both axes
    def bucket(s, mod=100):
        """Stable hash bucket in [0, mod). mod MUST be 100: the thresholds are
        PERCENTAGES, and an earlier version used mod=10, which turned
        `--schema-heldout-pct 5` into 5-of-10 = 50% per axis and collapsed
        train to a quarter of the data. At the original pct=2 it was 20% per
        axis, which is why the pilot's train split was 62% and not ~96%."""
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
        held_schema = bucket(schema_key) < args.schema_heldout_pct
        held_pass = bucket(r["pid"]) < args.passage_heldout_pct

        for _ in range(args.rows_per_passage):
            task = rng.choices(["extract", "fill"],
                               weights=[1 - args.fill_frac, args.fill_frac])[0]
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
            # With --held-formats empty, every format is trained and eval rows
            # draw from the same pool: format transfer has been measured and is
            # ~0 (6/6 parse with demos on a trained format, 0/6 with demos on a
            # held-out one), so spending half of every eval row on a settled
            # negative costs resolution on the axis that can still move.
            fmt = (rng.choice(sorted(held_f))
                   if held_f and (held_schema or held_pass) and rng.random() < 0.5
                   else rng.choice(seen_f))
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
            # Vary the shot count per row. Pinned at 3, the model never sees
            # 1-shot or 5-shot and cannot learn to use more context when given
            # it -- the shot-scaling probe measured 53.2 / 55.9 / 58.5 / 59.8 /
            # 56.6 across 1-5 shots, so the useful range is wider than one
            # value. 0 is included so `icl` mode also covers the no-example
            # case, which is what an instruction-only prompt looks like.
            args._shots = rng.randint(args.min_shots, args.shots)
            # one surface per row: demos and query must agree, or the labels
            # become noise rather than a learnable convention
            args._lab = {
                "text": rng.choice(L_TEXT), "gap": rng.choice(L_TEXT_GAP),
                "fields": rng.choice(L_FIELDS), "answer": rng.choice(L_ANSWER),
                "fill": rng.choice(L_FILL), "demos": rng.choice(L_DEMOS),
            }
            marker = None
            if task == "fill":
                built = build_fill(r, chosen, names, km, mode, rng,
                                   rows_in, args)
                if built is None:
                    stats["fill_unmaskable"] += 1
                    continue
                prompt, ans, marker = built
            else:
                prompt = build_prompt(r["passage"], names, km, fmt, mode, rng,
                                      rows_in, args)
            split = ("eval_both" if held_schema and held_pass else
                     "eval_schema" if held_schema else
                     "eval_passage" if held_pass else "train")
            out[split].append({
                "messages": [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": ans}],
                "meta": {"task": task, "format": fmt, "mode": mode,
                         "marker": marker,
                         "shots": (getattr(args, "_shots", args.shots)
                                   if mode in ("icl", "both") else 0),
                         "symbols": scheme or "none", "n_fields": len(names),
                         "schema": "|".join(names), "pid": r["pid"]}})
            stats[f"split:{split}"] += 1
            stats[f"task:{task}"] += 1

    for split, rows in out.items():
        p = args.out / f"{split}.jsonl"
        with p.open("w") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"  {split:<14} {len(rows)}")
    print("\nstats:", dict(stats))


def _mask(passage, spans, marker, numbered, rng):
    """Blank out `spans` in `passage`, left to right, no overlaps.

    Returns (masked_text, [(marker_shown, original_span)]) or None if any span
    cannot be located -- which happens when whitespace differs, since values
    were normalised at gate time but the passage was not.
    """
    hits = []
    for sp in spans:
        # The span must occur EXACTLY once. Masking only the first occurrence
        # leaves later copies visible, and the model reads the answer straight
        # off the prompt -- measured at 1,167 of 4,000 rows (29%) before this
        # check, which defeats the point of the task.
        if passage.count(sp) != 1:
            return None
        i = passage.find(sp)
        if i < 0:
            return None
        hits.append((i, i + len(sp), sp))
    hits.sort()
    for a, b in zip(hits, hits[1:]):          # overlapping spans are ambiguous
        if a[1] > b[0]:
            return None
    out, prev, pairs = [], 0, []
    for n, (i, j, sp) in enumerate(hits, 1):
        shown = marker.format(i=n) if numbered else marker
        out.append(passage[prev:i]); out.append(shown)
        pairs.append((shown, sp))
        prev = j
    out.append(passage[prev:])
    return "".join(out), pairs


def build_fill(r, chosen, names, km, mode, rng, pool, args):
    """Reverse task: the passage has gaps, the model supplies what belongs.

    Forward teaches 'find this in the text'; this teaches 'this text is missing
    something of this type'. Same supervision, opposite direction, and copying
    cannot shortcut it -- the answer is absent from the prompt by construction.

    The marker is drawn from a large pool on purpose. A single fixed blank
    teaches the marker rather than the task, and `___` collides with OCR
    artefacts in the books register.
    """
    # only fields that actually have values can be blanked
    fillable = [(n, f) for n, f in zip(names, chosen) if f["vaerdi"]]
    if not fillable:
        return None
    numbered = rng.random() < args.numbered_blank_frac
    marker = rng.choice(BLANKS_NUM if numbered else BLANKS)
    # one span per field: masking every occurrence of a multi-value field makes
    # the count ambiguous, and the model cannot know how many to supply
    spans = [f["vaerdi"][0] for _, f in fillable]
    got = _mask(r["passage"], spans, marker, numbered, rng)
    if got is None:
        return None
    masked, pairs = got

    def block(text, pairs_, with_answer):
        # The spec line MUST list the gap markers, because the answer is keyed
        # by markers. Listing field names instead made the two vocabularies
        # disjoint in 100% of fill rows (20,091/20,091 in v1) and put the count
        # off in 13.6%, since `names` includes absent fields that can never be
        # masked. That made the line pure noise on this task -- and, worse,
        # taught 20% of the corpus that the field-name line does not determine
        # the answer keys, which is exactly what `extract` relies on it for.
        # It also made fill unlearnable in `instruction` mode, where there are
        # no demonstrations to reveal the marker vocabulary.
        spec = ", ".join(shown for shown, _ in pairs_)
        head = (f"{args._lab['gap']}:\n{text}\n"
                f"{args._lab['fields']}: {spec}\n{args._lab['fill']}:")
        if not with_answer:
            return head
        body = "\n".join(f"{shown} = {sp}" for shown, sp in pairs_)
        return head + "\n" + body

    parts = []
    if mode in ("instruction", "both"):
        parts.append(rng.choice(INSTR_FILL) + " Formen er: hul = tekst.")
    if mode in ("icl", "both"):
        demos, want = [], getattr(args, "_shots", args.shots)
        for other in rng.sample(pool, min(len(pool), want + 8)):
            if len(demos) >= want or other["pid"] == r["pid"]:
                continue
            pres = [f for f in other["felter"] if f["vaerdi"]][:3]
            if len(pres) < 2:
                continue
            dgot = _mask(other["passage"][:700],
                         [f["vaerdi"][0] for f in pres], marker, numbered, rng)
            if dgot is None:
                continue
            demos.append(block(dgot[0], dgot[1], True))
        if demos:
            parts.append(f"{args._lab['demos']}:\n\n" + "\n\n".join(demos))
    parts.append(block(masked, pairs, False))
    answer = "\n".join(f"{shown} = {sp}" for shown, sp in pairs)
    return "\n\n".join(parts), answer, marker


def format_clause(fmt):
    """A one-line statement of the required output format, in the format.

    Instruction-mode prompts previously named no format at all: the whole
    instruction was e.g. "Findes oplysningen? Så skriv den af ordret. Findes
    den ikke? Så -." while the gold was TSV. One format out of ten, with
    nothing in the prompt to say which -- unanswerable, and measured at 0/6
    parse against 6/6 for the same trained format when demonstrations were
    present. That is ~21% of rows teaching that the requested format is
    unknowable.

    The clause is RENDERED, not written by hand, so it cannot drift from the
    renderer. Control characters are shown as escapes (a literal tab in the
    prompt is invisible and unlearnable as a spec).
    """
    example = render({"felt": "værdi"}, ["felt"], {"felt": "felt"}, fmt)
    shown = example.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
    return f"Svar i formatet: {shown}"


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
        head = (f"{args._lab['text']}:\n{text}\n"
                f"{args._lab['fields']}: {spec}\n{args._lab['answer']}:")
        return head + (f"\n{render(obj, ns, keymap, fmt)}" if with_answer else "")

    parts = []
    if mode in ("instruction", "both"):
        # `args._bank`, not a local named `pool` -- that is the parameter
        # holding the passage rows, and shadowing it made the demonstration
        # loop iterate over instruction strings
        parts.append(rng.choice(args._bank).format(null=NULL)
                     + " " + format_clause(fmt))
    if mode in ("icl", "both"):
        demos = []
        want_shots = getattr(args, "_shots", args.shots)
        if want_shots <= 0:
            demos = []
        for other in rng.sample(pool, min(len(pool), want_shots + 6)):
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
            if len(demos) >= getattr(args, "_shots", args.shots):
                break
        if demos:
            parts.append(f"{args._lab['demos']}:\n\n" + "\n\n".join(demos))
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
    ap.add_argument("--shots", type=int, default=5,
                    help="max demonstrations; each row draws "
                         "randint(--min-shots, --shots)")
    ap.add_argument("--min-shots", type=int, default=1)
    ap.add_argument("--fill-frac", type=float, default=0.3,
                    help="share of rows rendered as the reverse "
                         "fill-the-gap task instead of extraction")
    ap.add_argument("--numbered-blank-frac", type=float, default=0.4,
                    help="share of fill rows using an indexed marker "
                         "([BLANK 1], <2>, ...) rather than a bare one")
    ap.add_argument("--sym-frac", type=float, default=0.35)
    ap.add_argument("--mode-frac", nargs=3, type=float, default=[0.5, 0.2, 0.3])
    ap.add_argument("--held-formats", nargs="*", default=[],
                    help="Formats withheld from training, to measure format "
                         "transfer. Now EMPTY by default: transfer was "
                         "measured at 0/6 (demos + held-out format) against "
                         "6/6 (demos + trained format), so holding three of "
                         "ten out spent ~50%% of every eval row re-confirming "
                         "a settled negative. Pass them explicitly to restore "
                         "the transfer measurement.")
    # 5% on each axis, so eval_both (their intersection) is ~0.25% of
    # passages rather than the 0.04% that 2%x2% gives -- the pilot produced
    # 9 rows there, too few to read anything from.
    ap.add_argument("--schema-heldout-pct", type=int, default=5)
    ap.add_argument("--passage-heldout-pct", type=int, default=5)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    if args.phase == "extract":
        asyncio.run(phase_extract(args))
    else:
        phase_render(args)


if __name__ == "__main__":
    main()
