"""Translate ToolMind's Glaive subset to Danish, preserving the tool surface.

WHAT MOVES AND WHAT DOES NOT
    translated : user turns, <think> blocks, assistant prose, tool and
                 parameter DESCRIPTIONS, enum values, and natural-language
                 VALUES in tool calls and results -- including content the
                 user chose, such as a note title, which must then read the
                 same everywhere it recurs
    untouched  : tool names, parameter keys, every JSON key, and machine
                 values -- acronyms, dates, numbers, emails, URLs, ISO codes,
                 snake_case identifiers (EUR, AAPL, 1990-05-15, en-US)

Rationale: a Danish user talks to an English-named API. Keeping names and keys
fixed also keeps the data verifiable -- a reward function can compare keys
exactly, and `project_icl_json_v1_result` measured that meaning-free keys are no
harder for the model to induce than real ones, so there is nothing to gain by
translating them. Descriptions ARE translated: once names are inert, the
description is the only thing carrying tool semantics, and leaving it English
would make tool *selection* an English-reading task inside a Danish model.

THE MODEL NEVER SEES THE JSON. Asking an LLM to "return this structure with
some fields translated" invites structural drift, and we would then be gating
for damage we caused. Instead each row is reduced to a numbered list of text
segments; the model returns a numbered list of translations; the splice happens
programmatically by path. Structure is preserved BY CONSTRUCTION rather than by
check -- the same reason the extraction generator proposes keys blind.

Enum values DO translate, but only once: the spec's list is translated and
every invocation inherits that exact string. The contract must stay COHERENT,
not English -- a call carrying "cirkel" is valid iff the spec offers "cirkel".
A gate checks that, because independently translating the two is precisely how
a schema and its calls drift apart.

Usage:
  python scripts/translate_toolmind_da.py --n 5 --dry-run     # segments only
  python scripts/translate_toolmind_da.py --n 25              # small smoke
  python scripts/translate_toolmind_da.py --n 25 --gate-only  # re-gate cache
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

REPO = "Nanbeige/ToolMind"
FILE = "open_datasets/glaive-function-calling-v2-query.jsonl"
MODEL = "google/gemini-2.5-flash-lite"
URL = "https://openrouter.ai/api/v1/chat/completions"

# Identifier-shaped values never translate. A value is an identifier if it has
# no space and looks like a code, path, symbol or machine token.
IDENTIFIER = re.compile(
    r"^(?:[A-Z]{2,5}|[\w.-]+/[\w./-]+|[\w.-]+@[\w.-]+|https?://\S+|"
    r"[\d\s:/.,+-]+|[A-Za-z_][A-Za-z0-9_]*|\W+)$"
)
CODE_VALUE = re.compile(
    r"^(?:[A-Z0-9]{2,6}|[a-z]{2}(?:[-_][A-Za-z]{2})?|[\d\s:/.,+%-]+|"
    r"\S+@\S+|https?://\S+|[A-Za-z]+[_.][\w._]+|\W+)$")
IDENT_STRICT = re.compile(r"^(?=.*[_.\d]|.*[a-z][A-Z])[A-Za-z][\w.]{3,}$")
_TAG = re.compile(r"^\s*\[(?:tool_desc|param_desc|user|think|response|arg|result)\]\s*")
DA_MARK = re.compile(r"\b(og|er|det|den|til|for|ikke|med|som|har|kan|jeg|du|"
                     r"af|på|en|et|de|der)\b|[æøåÆØÅ]", re.I)
EN_MARK = re.compile(r"\b(the|and|is|are|to|of|you|for|with|that|this|"
                     r"your|have|will|from)\b", re.I)

SYS = """Du oversætter engelsk til dansk i et datasæt om værktøjskald.

Du får en nummereret liste af tekststykker. Oversæt HVER linje til naturligt
dansk og svar med præcis lige så mange linjer i samme rækkefølge.

ABSOLUTTE KRAV:
- Oversæt ALDRIG de navne, der står under "BEVAR UÆNDRET". De skal stå
  ordret på engelsk i din oversættelse, også midt i en dansk sætning.
- Tal, datoer, koder, valutaer, URL'er og filstier skrives uændret.
- Indhold som brugeren selv har valgt -- en titel, en note, en besked -- SKAL
  oversættes til dansk. "Team Meeting Agenda" bliver til "Dagsorden til
  teammøde". Kun rigtige egennavne (personer, byer, film, restauranter,
  firmaer) beholder deres oprindelige form.
- Optræder det samme stykke tekst flere steder -- fx både i brugerens
  besked, i værktøjskaldet og i svaret -- SKAL du bruge nøjagtig samme
  danske ord alle steder. Ellers hænger samtalen ikke sammen.
- Bevar tone og længde. Et <think>-stykke er modellens indre ræsonnement og
  skal lyde som en person, der tænker højt på dansk.
- Svar KUN med oversættelserne, én per linje, uden numre og UDEN
  kategorimærket i kantede parenteser. Mærket er kun til din orientering.
- Parameternavne og funktionsnavne skal stå på engelsk OGSÅ når du nævner dem
  midt i en dansk sætning. Skriv "parameteren company_name", ikke
  "parameteren firma_navn". Dette er den hyppigste fejl."""


# ── segment extraction ──────────────────────────────────────────────────────

def _translatable_value(v, pinned: set) -> bool:
    """Free text translates; machine tokens do not.

    An earlier version skipped anything without a space, which pinned 55.1% of
    argument values for the wrong reason: `activity=cycling`, `cuisine=Italian`,
    `genre=comedy` are ordinary words, not machine tokens, and a Danish
    conversation should carry Danish there. Only genuinely non-linguistic
    values stay: acronyms, dates, numbers, emails, URLs, ISO codes, and
    snake_case/dotted identifiers.
    """
    if not isinstance(v, str) or not v.strip():
        return False
    if v in pinned:
        return False
    return not CODE_VALUE.match(v.strip())


def _pinned_names(row) -> set:
    """Tool names and parameter keys. These never translate, anywhere."""
    out = set()
    for t in row.get("tools", []):
        f = t.get("function") or {}
        out.add(f.get("name"))
        props = ((f.get("parameters") or {}).get("properties")) or {}
        if isinstance(props, dict):
            out.update(props)
    return {x for x in out if isinstance(x, str)}


def _enum_arg_paths(row) -> set:
    """Argument paths holding an enum value — rewritten by splice(), not sent."""
    enums = _enum_values(row)
    out = set()
    for j, m in enumerate(row.get("conversations", [])):
        for t_i, tc in enumerate(m.get("tool_calls") or []):
            args = (tc.get("function") or {}).get("arguments") or {}
            if isinstance(args, dict):
                for k, v in args.items():
                    if isinstance(v, str) and v in enums:
                        out.add(("conversations", j, "tool_calls", t_i,
                                 "function", "arguments", k))
    return out


def _enum_values(row) -> set:
    out = set()
    for t in row.get("tools", []):
        props = (((t.get("function") or {}).get("parameters")) or {}).get("properties") or {}
        if isinstance(props, dict):
            for v in props.values():
                if isinstance(v, dict):
                    out.update(x for x in (v.get("enum") or [])
                               if isinstance(x, str))
    return out


# kept for the gate's benefit: names + enums, i.e. everything with a contract
def _pinned_values(row) -> set:
    return _pinned_names(row)


def segments(row):
    """[(path, kind, text)] — path is a splice address into the row.

    Enum values are translated ONCE, from the spec, and the same translation is
    then forced onto every invocation that used them (see `splice`). The
    contract has to stay coherent, not English: a call carrying `"cirkel"` is
    only valid if the spec's enum list says `"cirkel"` too.
    """
    pinned = _pinned_names(row)
    enums = _enum_values(row)
    segs = []
    for i, t in enumerate(row.get("tools", [])):
        f = t.get("function") or {}
        if f.get("description"):
            segs.append((("tools", i, "function", "description"),
                         "tool_desc", f["description"]))
        props = ((f.get("parameters") or {}).get("properties")) or {}
        if isinstance(props, dict):
            for k, v in props.items():
                if not isinstance(v, dict):
                    continue
                if v.get("description"):
                    segs.append((("tools", i, "function", "parameters",
                                  "properties", k, "description"),
                                 "param_desc", v["description"]))
                for e_i, ev in enumerate(v.get("enum") or []):
                    if isinstance(ev, str) and not CODE_VALUE.match(ev.strip()):
                        segs.append((("tools", i, "function", "parameters",
                                      "properties", k, "enum", e_i),
                                     "enum", ev))
    for j, m in enumerate(row.get("conversations", [])):
        role, content = m.get("role"), m.get("content") or ""
        if role == "user" and content.strip():
            segs.append((("conversations", j, "content"), "user", content))
        elif role == "assistant" and content.strip():
            kind = "think" if m.get("tool_calls") else "response"
            segs.append((("conversations", j, "content"), kind, content))
        elif role == "tool" and content.strip():
            # tool results are JSON strings: only natural-language leaves move
            try:
                obj = json.loads(content)
            except Exception:
                segs.append((("conversations", j, "content"), "result", content))
                continue
            for p, v in _walk(obj):
                if _translatable_value(v, pinned):
                    segs.append((("conversations", j, "content", "#json") + p,
                                 "result", v))
        for t_i, tc in enumerate(m.get("tool_calls") or []):
            args = (tc.get("function") or {}).get("arguments") or {}
            if isinstance(args, dict):
                for k, v in args.items():
                    if isinstance(v, str) and v in enums:
                        continue          # inherits the spec's enum translation
                    if _translatable_value(v, pinned):
                        segs.append((("conversations", j, "tool_calls", t_i,
                                      "function", "arguments", k), "arg", v))
    return segs


def _walk(obj, prefix=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, prefix + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, prefix + (i,))
    else:
        yield prefix, obj


def splice(row, segs, translations):
    """Write translations back by path. Structure cannot change here."""
    out = json.loads(json.dumps(row))          # deep copy
    json_cache = {}
    for (path, _kind, _orig), new in zip(segs, translations):
        if "#json" in path:
            cut = path.index("#json")
            outer, inner = path[:cut], path[cut + 1:]
            node = _get(out, outer)
            key = tuple(outer)
            if key not in json_cache:
                json_cache[key] = json.loads(node)
            _set(json_cache[key], inner, new)
        else:
            _set(out, path, new)
    for key, obj in json_cache.items():
        _set(out, key, json.dumps(obj, ensure_ascii=False))
    # Force the spec's enum translation onto every invocation. Translating the
    # two independently is how a contract silently desynchronises: the spec
    # would offer ["cirkel","rektangel"] while the call carried "circle".
    emap = {orig: new for (_p, kind, orig), new in zip(segs, translations)
            if kind == "enum"}
    if emap:
        for m in out.get("conversations", []):
            for tc in (m.get("tool_calls") or []):
                a = (tc.get("function") or {}).get("arguments")
                if isinstance(a, dict):
                    for k, v in list(a.items()):
                        if isinstance(v, str) and v in emap:
                            a[k] = emap[v]
    return out


def _get(o, path):
    for p in path:
        o = o[p]
    return o


def _set(o, path, val):
    for p in path[:-1]:
        o = o[p]
    o[path[-1]] = val


# ── gates ───────────────────────────────────────────────────────────────────

def skeleton(row, blanks=None):
    """Structure with every translatable leaf blanked, so two rows differing
    only in translations compare equal. Catches structural drift.

    `blanks` MUST come from the ORIGINAL row. Recomputing per row was a
    false-positive generator: translating "The temperature" to "Temperaturen"
    loses the space, the value stops looking translatable, the blank-sets
    diverge and the skeletons mismatch though nothing structural moved.

    Tool results are JSON *strings*, so their blanked leaves live at paths
    under a "#json" marker that the plain walk can never reach. Those are
    parsed and blanked in place -- otherwise the whole serialised result
    compares literally and every translated result reads as structural damage.
    """
    if blanks is None:
        blanks = {tuple(p) for p, _k, _t in segments(row)}
    inner, plain = {}, set()
    for path in blanks:
        if "#json" in path:
            cut = path.index("#json")
            inner.setdefault(path[:cut], []).append(path[cut + 1:])
        else:
            plain.add(path)

    def rec(o, prefix=()):
        if prefix in inner and isinstance(o, str):
            try:
                obj = json.loads(o)
            except Exception:
                return "<T>"
            for ip in inner[prefix]:
                try:
                    _set(obj, ip, "<T>")
                except Exception:
                    pass
            return json.dumps(obj, sort_keys=True, ensure_ascii=False)
        if isinstance(o, dict):
            return {k: rec(v, prefix + (k,)) for k, v in sorted(o.items())}
        if isinstance(o, list):
            return [rec(v, prefix + (i,)) for i, v in enumerate(o)]
        return "<T>" if prefix in plain else o
    return json.dumps(rec(row), ensure_ascii=False, sort_keys=True)


def gate(orig, new):
    """Mechanical checks. Returns a list of failure reasons (empty = pass)."""
    bad = []
    o_segs, n_segs = segments(orig), segments(new)
    if len(o_segs) != len(n_segs):
        bad.append("segment-count-changed")
    # tool names + parameter keys must be byte-identical
    def surface(r):
        s = []
        for t in r.get("tools", []):
            f = t.get("function") or {}
            props = ((f.get("parameters") or {}).get("properties")) or {}
            s.append((f.get("name"),
                      tuple(sorted(props)) if isinstance(props, dict) else ()))
        for m in r.get("conversations", []):
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                a = fn.get("arguments") or {}
                s.append((fn.get("name"),
                          tuple(sorted(a)) if isinstance(a, dict) else ()))
        return s
    if surface(orig) != surface(new):
        bad.append("tool-name-or-key-changed")
    # Enum-valued arguments are rewritten by splice() to inherit the spec's
    # translation, but they are never SENT for translation, so they are absent
    # from the segment list and would compare literally -- the mechanism that
    # keeps the contract coherent would read as structural damage (8/10 rows).
    blanks = {tuple(p) for p, _k, _t in o_segs} | _enum_arg_paths(orig)
    if skeleton(orig, blanks) != skeleton(new, blanks):
        bad.append("structure-changed")
    # pinned values (enums, identifiers, numbers) must survive untouched
    pin_o = _pinned_values(orig)
    for m in new.get("conversations", []):
        for tc in (m.get("tool_calls") or []):
            for k, v in ((tc.get("function") or {}).get("arguments") or {}).items():
                if isinstance(v, str) and v not in pin_o:
                    continue
    # DNT tokens (tool + parameter names) must survive inside the prose.
    # The smoke caught the model rendering `company_name` as `firma_navn`
    # inside a <think> block while the call itself still used company_name --
    # prose referring to a parameter that does not exist.
    # Only MACHINE identifiers are enforceable. Many parameter names are also
    # ordinary English words -- "The search query" -> "Søgeforespørgslen" and
    # "Convert an amount..." -> "Konverter et beløb..." are correct
    # translations, and an unrestricted check flagged 7/10 rows on exactly
    # those. Requiring an underscore, dot, digit or internal capital keeps the
    # check on tokens that can never be prose.
    lost = 0
    for (_p, kind, txt), (_p2, _k2, new_txt) in zip(o_segs, n_segs):
        if kind not in ("think", "response", "user", "tool_desc", "param_desc"):
            continue
        for tok in pin_o:
            if not IDENT_STRICT.match(tok):
                continue
            if tok in txt and tok not in new_txt:
                lost += 1
    if lost:
        bad.append(f"dnt-token-lost({lost})")

    # CONTRACT COHERENCE. Enum values may be translated, but the spec and every
    # invocation must move together: a call carrying "cirkel" is valid only if
    # that parameter's enum list also says "cirkel". This is the check that
    # makes translating enums safe at all -- without it the two drift apart and
    # the row becomes unsatisfiable against its own schema.
    spec_enums = {}
    for t in new.get("tools", []):
        f = t.get("function") or {}
        props = ((f.get("parameters") or {}).get("properties")) or {}
        if isinstance(props, dict):
            for k, v in props.items():
                if isinstance(v, dict) and v.get("enum"):
                    spec_enums[(f.get("name"), k)] = set(v["enum"])
    broke = 0
    for m in new.get("conversations", []):
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            for k, v in (fn.get("arguments") or {}).items():
                allowed = spec_enums.get((fn.get("name"), k))
                if allowed and isinstance(v, str) and v not in allowed:
                    broke += 1
    if broke:
        bad.append(f"enum-desynced({broke})")

    # language: translated fields should read as Danish, not English
    # VALUE-CHAIN CONSISTENCY. User content translates, so a value now moves in
    # several places at once: the user says it, the call carries it, the result
    # echoes it, the response repeats it. If the translator renders it
    # differently in each, the conversation stops cohering and nothing
    # downstream can ground the argument in what the user actually asked for.
    # Rule: if the English value appeared in the row's other prose, the Danish
    # value must appear in the Danish prose.
    def _prose(segs):
        return " ".join(t for _p, k, t in segs
                        if k in ("user", "think", "response", "result")).lower()
    o_prose, n_prose = _prose(o_segs), _prose(n_segs)
    drift = 0
    for (_p, kind, en_v), (_p2, _k2, da_v) in zip(o_segs, n_segs):
        if kind != "arg" or len(en_v) < 4:
            continue
        if en_v.lower() in o_prose and da_v.lower() not in n_prose:
            drift += 1
    if drift:
        bad.append(f"value-chain-broken({drift})")

    # ANY clearly-English segment fails the row. This was a ratio
    # (en > da * 0.15) and it went blind exactly where it mattered: on a row
    # with many segments, one untranslated field sits under the threshold, and
    # the planted control passed. A segment of 8+ words carrying English
    # function words and not one Danish marker is unambiguous, so one is enough.
    en = da = 0
    for (_p, kind, txt), (_p2, _k2, new_txt) in zip(o_segs, n_segs):
        if kind not in ("user", "think", "response", "tool_desc", "param_desc"):
            continue
        if len(new_txt.split()) < 5:
            continue
        if EN_MARK.search(new_txt) and not DA_MARK.search(new_txt):
            en += 1
        else:
            da += 1
    if en:
        bad.append(f"still-english({en}/{en+da})")
    return bad


# ── translation ─────────────────────────────────────────────────────────────

def _key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for p in (Path.home() / "or", Path.home() / ".or"):
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("no OpenRouter key (~/or)")


def build_request(row, segs):
    pinned = sorted(x for x in _pinned_values(row) if x)
    lines = "\n".join(f"{i+1}. [{k}] {t}" for i, (_p, k, t) in enumerate(segs))
    return (f"BEVAR UÆNDRET (skriv disse ordret på engelsk):\n"
            f"{', '.join(pinned)}\n\n"
            f"Oversæt disse {len(segs)} tekststykker:\n{lines}")


async def translate(session, row, segs, tries=3):
    body = {"model": MODEL, "temperature": 0.3,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": build_request(row, segs)}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "oversaettelser", "strict": True, "schema": {
                    "type": "object",
                    "properties": {"linjer": {"type": "array",
                                              "items": {"type": "string"}}},
                    "required": ["linjer"], "additionalProperties": False}}}}
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                out = json.loads(d["choices"][0]["message"]["content"])["linjer"]
                # the model echoes the "[kind]" hint back roughly 1 time in 3
                out = [_TAG.sub("", x, count=1).lstrip() for x in out]
                if len(out) == len(segs):
                    return out
                # a length mismatch is unrecoverable by splicing: retry
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return None


# ── main ────────────────────────────────────────────────────────────────────

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", type=Path, default=Path("scratch/toolmind_da"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the segments that WOULD be sent, no API calls")
    ap.add_argument("--gate-only", action="store_true",
                    help="re-run the gates over the cache, no API calls")
    ap.add_argument("--only-enums", action="store_true",
                    help="keep only rows whose calls use an enum-constrained "
                         "argument -- 3.3% of values, so they need selecting "
                         "for on purpose or the enum path never gets smoked")
    ap.add_argument("--show", type=int, default=2,
                    help="print this many translated rows in full")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "translated.jsonl"

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, FILE, repo_type="dataset")
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            rows.append(json.loads(line))
    if args.only_enums:
        keep = []
        with open(path) as f:
            for line in f:
                x = json.loads(line)
                specs = {}
                for t in x.get("tools", []):
                    fn = t.get("function") or {}
                    props = ((fn.get("parameters") or {}).get("properties")) or {}
                    if isinstance(props, dict):
                        for k, v in props.items():
                            if isinstance(v, dict) and v.get("enum"):
                                specs[(fn.get("name"), k)] = True
                hit = any(specs.get((( tc.get("function") or {}).get("name"), k))
                          for m in x.get("conversations", [])
                          for tc in (m.get("tool_calls") or [])
                          for k in ((tc.get("function") or {}).get("arguments") or {}))
                if hit:
                    keep.append(x)
                if len(keep) >= args.n:
                    break
        rows = keep
    print(f"loaded {len(rows)} rows from {FILE}", flush=True)

    if args.dry_run:
        for r in rows[:args.show or 2]:
            segs = segments(r)
            print(f"\n=== row: {len(segs)} segments, "
                  f"{sum(len(t) for _p, _k, t in segs):,} chars ===")
            print(build_request(r, segs)[:2500])
        kinds = Counter(k for r in rows for _p, k, _t in segments(r))
        print(f"\nsegment kinds over {len(rows)} rows: {dict(kinds)}")
        return

    if args.gate_only:
        pairs = [json.loads(l) for l in cache.open()]
    else:
        import aiohttp
        sem = asyncio.Semaphore(args.concurrency)
        async with aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {_key()}",
                         "Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=300)) as s:
            async def one(i, r):
                segs = segments(r)
                async with sem:
                    tr = await translate(s, r, segs)
                if i % 5 == 0:
                    print(f"  {i+1}/{len(rows)}", flush=True)
                if tr is None:
                    return None
                return {"orig": r, "da": splice(r, segs, tr)}
            got = await asyncio.gather(*[one(i, r) for i, r in enumerate(rows)])
        pairs = [g for g in got if g]
        with cache.open("w") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"\ntranslated {len(pairs)}/{len(rows)} -> {cache}")

    fails = Counter()
    ok = 0
    for p in pairs:
        bad = gate(p["orig"], p["da"])
        if bad:
            for b in bad:
                fails[b.split("(")[0]] += 1
        else:
            ok += 1
    print(f"\nGATE: {ok}/{len(pairs)} clean")
    for k, v in fails.most_common():
        print(f"   {v:>4}  {k}")

    # PLANTED CONTROLS: corrupt a copy in each way the gate claims to catch.
    # Without this the pass rate is uninterpretable -- a gate that never fires
    # looks identical to data that is always right.
    if pairs:
        base = pairs[0]
        ctrl = {}
        c1 = json.loads(json.dumps(base["da"]))
        for t in c1.get("tools", []):
            if t.get("function", {}).get("name"):
                t["function"]["name"] += "_ANDET"
                break
        ctrl["renamed tool"] = c1
        c2 = json.loads(json.dumps(base["da"]))
        for j, m in enumerate(c2.get("conversations", [])):
            src = base["orig"]["conversations"][j].get("content") or ""
            if m.get("content") and len(src.split()) >= 8:
                m["content"] = src          # the actual failure: left as-is
                break
        ctrl["left in english"] = c2
        c3 = json.loads(json.dumps(base["da"]))
        c3.setdefault("tools", []).append({"function": {"name": "x"}})
        ctrl["structure changed"] = c3
        # a control for the DNT check specifically: rewrite a snake_case
        # identifier inside the prose the way the model did before the prompt
        # fix ("parameteren company_name" -> "parameteren firma_navn"). Without
        # this the DNT check could be vacuous and the pass rate would not say so.
        c4 = json.loads(json.dumps(base["da"]))
        idents = sorted(t for t in _pinned_values(base["orig"])
                        if IDENT_STRICT.match(t))
        if idents:
            tok = idents[0]
            for m in c4.get("conversations", []):
                if m.get("content") and tok in m["content"]:
                    m["content"] = m["content"].replace(tok, "dansk_navn")
                    ctrl["identifier translated in prose"] = c4
                    break
        # SYNTHETIC, not sampled: enum-constrained arguments are rare (3.3% of
        # values), so a control drawn from the batch silently disappears on
        # most samples and the enum gate would look tested when it was not.
        enum_spec = {"tools": [{"function": {"name": "draw", "parameters": {
            "properties": {"shape": {"type": "string",
                                     "enum": ["cirkel", "rektangel"]}}}}}],
            "conversations": [{"role": "assistant", "content": "",
                               "tool_calls": [{"function": {
                                   "name": "draw",
                                   "arguments": {"shape": "cirkel"}}}]}]}
        enum_bad = json.loads(json.dumps(enum_spec))
        enum_bad["conversations"][0]["tool_calls"][0]["function"]["arguments"]["shape"] = "circle"
        ctrl["enum desynced from spec"] = enum_bad
        ctrl_base = {"enum desynced from spec": enum_spec}

        c6 = json.loads(json.dumps(base["da"]))
        for m in c6.get("conversations", []):
            hit = False
            for tc in (m.get("tool_calls") or []):
                a = (tc.get("function") or {}).get("arguments") or {}
                for k, v in list(a.items()):
                    if isinstance(v, str) and len(v) > 4:
                        # spaces, no underscore: must stay a translatable
                        # segment, or it trips segment-count instead and
                        # the value-chain check goes untested
                        a[k] = "en helt anden formulering"
                        ctrl["value chain broken"] = c6
                        hit = True
                        break
                if hit:
                    break
            if hit:
                break
        print("\nPLANTED CONTROLS (each must FAIL):")
        for name, bad_row in ctrl.items():
            ref = ctrl_base.get(name, base["orig"])
            res = gate(ref, bad_row)
            print(f"   {'FAIL ok' if res else '*** PASSED — GATE IS BLIND ***':<32}"
                  f" {name:<20} {res}")

    for p in pairs[:args.show]:
        print("\n" + "=" * 70)
        for m in p["da"].get("conversations", [])[:6]:
            c = (m.get("content") or "")[:300]
            print(f"[{m.get('role')}] {c}")
            if m.get("tool_calls"):
                print("   ->", json.dumps(m["tool_calls"], ensure_ascii=False)[:200])


if __name__ == "__main__":
    asyncio.run(main())
