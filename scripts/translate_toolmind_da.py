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
# Tools whose SUBJECT is language. Translating the conversation destroys the
# premise: a user who says "translate this English sentence: 'I love to
# travel'" ends up saying "oversæt denne engelske sætning: 'Jeg elsker at
# rejse'" while the call still carries source_language="English". Same for
# detect_language -- the text whose language is being identified has changed
# language. 467 of 19,919 rows (2.3%); they are dropped, not repaired.
LANGUAGE_TOOL = re.compile(r"translat|language|lang_", re.I)


def is_language_tool_row(row) -> bool:
    return any(LANGUAGE_TOOL.search(((t.get("function") or {}).get("name") or ""))
               for t in row.get("tools", []))
# The model echoes the "[kind]" hint back, and TRANSLATES it: [svar] 1,408,
# [taenk] 1,219, [tanke], [respons]... The English-only list missed every
# Danish variant, so ~2,650 assistant turns kept a stray bracket tag. Match
# any short bracketed token at the start instead of enumerating words.
_TAG = re.compile(r"^\s*\[[^\]\n]{1,14}\]\s*")
def _is_english(text: str) -> bool:
    """Proper language identification, not a word list.

    This was a hand-rolled regex twice over and wrong twice: first it fired on
    a RATIO and went blind on long rows, then its "English" markers included
    `to`, `for`, `is`, `have` and `and` -- all Danish words too (to=two,
    is=ice, and=duck) -- so "Beregn afstanden mellem to lokationer" read as
    untranslated English, 15 of 18 apparent failures at n=200. langdetect was
    already a dependency the whole time. It is decisive even on three words
    ("Mødets dato" -> da:1.00), which is well below anything we check.
    """
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 0          # else results vary between calls
    try:
        top = detect_langs(text)[0]
    except Exception:
        return False                  # too short / no letters: not a failure
    return top.lang == "en" and top.prob >= 0.90

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
- Står der "FASTE OVERSÆTTELSER", er de danske ord dér allerede afgjort og
  bruges i værktøjets skema. Brug dem hver gang begrebet nævnes -- også når
  ordet skal bøjes ("rektangel" -> "rektangler", "rektanglets") og også når
  du nævner det uden anførselstegn midt i en sætning. Skriv ALDRIG det
  engelske ord, når der står en fast oversættelse for det.
- VIGTIGST: linjer mærket [arg] er værdier, der sendes til værktøjet. De
  SKAL oversættes til dansk med præcis det ord, brugeren selv brugte i sin
  besked. Siger brugeren "komediefilm", skal [arg] "comedy" blive til
  "komedie" -- ikke forblive "comedy". Et dansk spørgsmål med en engelsk
  værdi i kaldet er den hyppigste fejl i dette datasæt.
- Bevar tone og længde. Et <think>-stykke er modellens indre ræsonnement og
  skal lyde som en person, der tænker højt på dansk.
- Svar KUN med oversættelserne, én per linje, uden numre og UDEN
  kategorimærket i kantede parenteser. Mærket er kun til din orientering.
- Parameternavne og funktionsnavne skal stå på engelsk OGSÅ når du nævner dem
  midt i en dansk sætning. Skriv "parameteren company_name", ikke
  "parameteren firma_navn". Dette er den hyppigste fejl."""


# ── segment extraction ──────────────────────────────────────────────────────

# Quote delimiters seen around payload echoes in this corpus: straight double,
# straight single, and the curly pair the translator sometimes emits. Matching
# only `"x"` left 7 of 50 leaking assistant messages untouched -- every one of
# them wrote 'circle' in single quotes.
_Q = "\"'\u2018\u2019\u201c\u201d"


def _quoted(term):
    """Regex for `term` wrapped in any quote character."""
    return f"[{_Q}]{re.escape(term)}[{_Q}]"


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


# Tools whose answer depends on the LITERAL CHARACTERS of the value. These are
# not language tools -- the name filter does not catch them -- but they fail the
# same way: "racecar" is a palindrome and "racerbil" is not, so translating the
# value makes the assistant assert something false, against a tool result that
# was computed on the English string. check_word_count on a translated sentence
# counts different words; reverse_string reverses different characters.
# 363 of 17,138 rows (2.12%).
#
# Pinned, not dropped: a Danish user can perfectly well ask whether 'racecar'
# is a palindrome, so the row stays usable once the value holds still.
CHAR_DEPENDENT = re.compile(
    r"palindrom|anagram|spell|rhym|syllab|acronym|letter|reverse|cipher|"
    r"encod|decod|word_count|character_count", re.I)


def _char_dependent_values(row) -> set:
    """Argument values belonging to a character-dependent tool."""
    names = set()
    for t in row.get("tools", []) or []:
        f = t.get("function") or {}
        if (CHAR_DEPENDENT.search(f.get("name") or "")
                or CHAR_DEPENDENT.search(f.get("description") or "")):
            names.add(f.get("name"))
    if not names:
        return set()
    out = set()
    for m in row.get("conversations", []):
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            if fn.get("name") not in names:
                continue
            a = fn.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    continue
            if isinstance(a, dict):
                out.update(v.strip() for v in a.values()
                           if isinstance(v, str) and v.strip())
    return out


def _pinned_names(row) -> set:
    """Tool names, parameter keys, and character-dependent values. Never
    translated, anywhere -- and listed under BEVAR UAENDRET so the prose keeps
    them too, which is what stops the user turn asking about 'racerbil'."""
    out = set(_char_dependent_values(row))
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
                # enum entries are VALUES: see value_segments()
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
                # A scalar at the JSON root gives an EMPTY path, which _set()
                # cannot address (it indexes path[-1]) -- it crashed the run at
                # row ~2,250. 14 of 9,276 tool results are like this and every
                # one is double-encoded JSON ('"{\\"tax_amount\\": 10000}"'),
                # i.e. machine data with nothing to translate. Skip them.
                if not p:
                    continue
                if _translatable_value(v, pinned):
                    segs.append((("conversations", j, "content", "#json") + p,
                                 "result", v))
        # call arguments are VALUES and come from the global lexicon, not from
        # this per-row request: see value_segments()
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


# The tool catalogue belongs to the row, not to the conversation, so its text
# is row-independent and gets translated ONCE globally (see the spec pass).
# Conversation text is per-row and carries the do-not-translate list.
SPEC_KINDS = ("tool_desc", "param_desc")


# ---------------------------------------------------------------- value lexicon
#
# Argument VALUES used to be translated per row (kind="arg"), which is how the
# same string acquired two Danish forms in two different rows: `shape` came back
# "rektangel" 506 times and "rectangle" 69 times, `get_news.category` split
# sports/sport 202/85. Nothing in a prompt tells the model which one that row
# wants, so ~2,800 values were an unwinnable coin flip -- the model learns the
# majority form and is scored against the minority whenever gold disagrees.
#
# Enum entries never had this problem: they were already translated globally,
# keyed by the English string. Their failure mode is different and visible in
# ['cirkel','rektangel','triangle'] -- one string the model declined to
# translate, propagated CONSISTENTLY to every row. Consistent, so harmless to
# learn from; still wrong, and fixed by retrying that string, not by re-keying.
#
# Both now draw from ONE table keyed by (tool, arg_key, value). Slot-keying
# rather than bare-string keying keeps context: "Apple" under
# get_stock_price.company is a ticker to pin, under a shopping tool it is a
# fruit. And because an enum entry and a call value that share a slot share a
# key, the spec cannot offer ["cirkel"] while the call carries "circle" -- not
# by a post-hoc repair pass, but because they are the same lookup.
#
# 24,459 value occurrences are only 2,483 distinct strings, of which 1,195
# appear more than once -- a singleton cannot contradict itself, so the whole
# defect lives in those 1,195. Deduplicating is ~10x cheaper as a side effect.
VALUE_SEP = "\x00"


def _value_key(tool, arg_key, value):
    return f"{tool or ''}.{arg_key or ''}{VALUE_SEP}{value.strip()}"


def value_segments(row):
    """[(path, key, text)] for every translatable VALUE: enum entries and call
    arguments alike. Paths address the row; keys address the global lexicon."""
    pinned = _pinned_names(row)
    segs = []
    for i, t in enumerate(row.get("tools", [])):
        f = t.get("function") or {}
        name = f.get("name")
        props = ((f.get("parameters") or {}).get("properties")) or {}
        if isinstance(props, dict):
            for k, v in props.items():
                if not isinstance(v, dict):
                    continue
                for e_i, ev in enumerate(v.get("enum") or []):
                    if isinstance(ev, str) and _translatable_value(ev, pinned):
                        segs.append((("tools", i, "function", "parameters",
                                      "properties", k, "enum", e_i),
                                     _value_key(name, k, ev), ev))
    for j, m in enumerate(row.get("conversations", [])):
        for t_i, tc in enumerate(m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, dict):
                for k, v in args.items():
                    if isinstance(v, str) and _translatable_value(v, pinned):
                        segs.append((("conversations", j, "tool_calls", t_i,
                                      "function", "arguments", k),
                                     _value_key(fn.get("name"), k, v), v))
    return segs


def spec_segments(row):
    return [s for s in segments(row) if s[1] in SPEC_KINDS]


def conv_segments(row):
    return [s for s in segments(row) if s[1] not in SPEC_KINDS]


def splice(row, segs, translations, spec_map=None, value_map=None):
    """Write translations back by path. Structure cannot change here.

    `segs`/`translations` cover the conversation; `spec_map` supplies the tool
    catalogue and `value_map` every argument/enum VALUE, both from global passes.
    """
    out = json.loads(json.dumps(row))          # deep copy
    if spec_map:
        segs = list(segs) + [s for s in spec_segments(row) if s[2] in spec_map]
        translations = list(translations) + [
            spec_map[s[2]] for s in spec_segments(row) if s[2] in spec_map]
    if value_map:
        vsegs = [s for s in value_segments(row) if s[1] in value_map]
        segs = list(segs) + [(p, "value", t) for p, _k, t in vsegs]
        translations = list(translations) + [value_map[k] for _p, k, _t in vsegs]
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
    # No post-hoc enum sync any more. Spec enums and call arguments share a
    # lexicon key, so they receive the same string by construction rather than
    # being reconciled afterwards; the old pass could only fix rows whose spec
    # HAPPENED to declare an enum, which is 543 of 17,138.
    #
    # Prose echoes are the one place the lexicon cannot reach. Reasoning text
    # quotes the payload inline -- `Sa, shape: "rectangle", dimensions: {...}`
    # -- and that copy is free text, not a value path. 1,133 of the 4,943
    # values whose translation changes the string are quoted this way, across
    # 861 rows (5.0%). Untouched, the row contradicts itself: the call carries
    # "rektangel" while its own reasoning says "rectangle".
    #
    # Substitution is quote-delimited and assistant-only. Quotes separate the
    # payload echo from ordinary prose -- `"Italian"` is the echo, `italiensk`
    # in a sentence is Danish text -- and a user's own quoted words are theirs,
    # not an echo, so user turns are left alone.
    # Reasoning also CITES the spec: `Vaerktojets beskrivelse siger "the shape
    # to calculate the area for"`. Same defect one level up -- the description
    # is now Danish, so the citation is stale English. 681 assistant messages
    # (2.21%), 852 citations. Invisible to every structural check AND to
    # langdetect, since one quoted phrase does not shift a long segment's
    # verdict; found only by reading the output.
    repl = {}
    if value_map:
        for _p, key, en in value_segments(row):
            da = value_map.get(key)
            if da and da.strip().lower() != en.strip().lower():
                repl[en.strip()] = da
    if spec_map:
        for _p, _k, en in spec_segments(row):
            da = spec_map.get(en.strip())
            if da and da.strip().lower() != en.strip().lower():
                repl[en.strip()] = da
                # citations often drop a trailing period
                repl.setdefault(en.strip().rstrip("."), da.strip().rstrip("."))
    if repl:
        # longest first: a description can contain a value as a substring, and
        # rewriting the short one first would corrupt the long one.
        ordered = sorted(repl.items(), key=lambda kv: -len(kv[0]))
        for m in out.get("conversations", []):
            if m.get("role") != "assistant":
                continue
            c = m.get("content")
            if not isinstance(c, str) or not c:
                continue
            for en, da in ordered:
                # CASE-INSENSITIVE. A citation is routinely re-cased to fit the
                # sentence: the spec says "The shape to calculate the area for"
                # and the reasoning quotes "the shape to calculate the area
                # for". An exact match misses it, and the gate then declared
                # idx 72 -- the row that motivated this whole check -- clean.
                c = re.sub(_quoted(en), lambda _m, d=da: f'"{d}"',
                           c, flags=re.I)
            m["content"] = c
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


def gate(orig, new, value_map=None, spec_map=None):
    """Mechanical checks. Returns a list of failure reasons (empty = pass).

    `value_map` upgrades the value check from "did it change" to "is it the
    CANONICAL translation". A structural diff cannot tell a correct translation
    from an arbitrary replacement -- blanking value paths hides both, not
    blanking them flags both -- so before the lexicon existed there was no way
    to verify a value at all. Now there is exactly one right answer per key.
    """
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
    # Values come from the global lexicon and are therefore absent from
    # segments() -- exactly like enum args before them. Un-blanked, every
    # legitimately translated value reads as structural damage: 45 of 198 rows
    # on the first smoke after values went global.
    blanks = ({tuple(p) for p, _k, _t in o_segs}
              | {tuple(p) for p, _k, _t in value_segments(orig)}
              | _enum_arg_paths(orig))
    if skeleton(orig, blanks) != skeleton(new, blanks):
        bad.append("structure-changed")
    # Values are blanked above, so they must be verified HERE or not at all.
    if value_map:
        for path, key, _en in value_segments(orig):
            want = value_map.get(key)
            if want is None:
                continue
            try:
                got = _get(new, path)
            except Exception:
                bad.append("value-path-missing")
                continue
            if got != want:
                bad.append("value-not-canonical")
                break
        # A canonical call is not enough: the reasoning must not still quote the
        # pre-translation form of a value it is about to send. This is the check
        # that would have caught row 22 -- call "rektangel", prose "rectangle".
        stale = False
        for _path, key, en in value_segments(orig):
            want = value_map.get(key)
            en = en.strip()
            if not want or want.strip().lower() == en.lower():
                continue
            for m in new.get("conversations", []):
                if m.get("role") != "assistant":
                    continue
                c = m.get("content")
                if isinstance(c, str) and re.search(
                        _quoted(en), c, re.I):
                    stale = True
                    break
            if stale:
                break
        if stale:
            bad.append("stale-prose-echo")
    # Same, for descriptions the reasoning cites verbatim.
    if spec_map:
        for _path, _k, en in spec_segments(orig):
            en = en.strip()
            da = spec_map.get(en)
            if not da or da.strip().lower() == en.lower() or len(en) < 15:
                continue
            for m in new.get("conversations", []):
                if m.get("role") != "assistant":
                    continue
                c = m.get("content")
                if isinstance(c, str) and re.search(
                        _quoted(en.rstrip('.')).replace('[' + _Q + ']', 
                            '[' + _Q + ']', 1), c, re.I):
                    bad.append("stale-spec-citation")
                    break
            if "stale-spec-citation" in bad:
                break
    # pinned values (enums, identifiers, numbers) must survive untouched
    pin_o = _pinned_values(orig)
    for m in new.get("conversations", []):
        for tc in (m.get("tool_calls") or []):
            # `arguments` is a dict on all 4,000 rows I first surveyed, but not
            # across the full 20k -- some are lists, and an unguarded .items()
            # crashed the gate report AFTER a completed 20k translation. Every
            # other access already guards; this one did not.
            call_args = (tc.get("function") or {}).get("arguments")
            if not isinstance(call_args, dict):
                continue
            for k, v in call_args.items():
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
            fn_args = fn.get("arguments")
            if not isinstance(fn_args, dict):
                continue
            for k, v in fn_args.items():
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
        if len(new_txt.split()) < 3:
            continue
        if _is_english(new_txt):
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


SPEC_SYS = """Du oversætter engelske feltbeskrivelser fra et værktøjs-API til dansk.

Du får en nummereret liste. Oversæt HVER linje til naturligt, grammatisk dansk
og svar med præcis lige så mange linjer i samme rækkefølge.

- Skriv almindeligt dansk. "The name of the artist" bliver til "Kunstnerens
  navn", ikke "Navnet på artist".
- Tal, datoer, koder, valutaer og URL'er skrives uændret.
- Svar KUN med oversættelserne, én per linje, uden numre."""


async def translate_spec(session, strings, tries=3):
    """Translate distinct catalogue strings, WITHOUT a do-not-translate list.

    This pass exists because the per-row prompt pins every parameter key, and
    43% of those keys are ordinary English words -- title, location, category,
    amount, length, text, genre, weight, shape. Obeying the pin inside a
    description produced "Navnet på artist" and "Beregn procentdelen af et
    number" in 6,701 rows. Descriptions never refer to another parameter by
    identifier, so the pin has no business being applied to them; keeping them
    out of that request is a structural fix rather than a prompt plea.

    Deduplicating is the other half: 75,468 description segments are only
    10,124 distinct strings, so this is ~7.5x cheaper AND makes the rendering
    deterministic -- the same English description cannot come back as two
    different Danish ones in two different rows.
    """
    body = {"model": MODEL, "temperature": 0.2,
            "messages": [{"role": "system", "content": SPEC_SYS},
                         {"role": "user", "content": "\n".join(
                             f"{i+1}. {s}" for i, s in enumerate(strings))}],
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
                    if a == tries - 1:
                        print(f"  spec HTTP {r.status}: "
                              f"{(await r.text())[:120]}", flush=True)
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                out = json.loads(
                    d["choices"][0]["message"]["content"])["linjer"]
                out = [_TAG.sub("", x, count=1).lstrip() for x in out]
                if len(out) == len(strings):
                    return out
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return None


async def build_spec_map(session, rows, cache_path, batch=40, concurrency=24):
    """english -> danish for every distinct catalogue string. Resumable."""
    have = {}
    if cache_path.exists():
        for line in cache_path.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                have[r["en"]] = r["da"]
            except Exception:
                continue
    want = sorted({s[2].strip() for r in rows for s in spec_segments(r)})
    todo = [s for s in want if s not in have]
    print(f"spec strings: {len(want):,} distinct, {len(have):,} cached, "
          f"{len(todo):,} to translate", flush=True)
    if not todo:
        return have
    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    n = [0]
    with cache_path.open("a", buffering=1) as fh:
        async def one(chunk):
            async with sem:
                got = await translate_spec(session, chunk)
            n[0] += 1
            if n[0] % 20 == 0:
                print(f"  spec {n[0]}/{len(chunks)} batches", flush=True)
            if not got:
                return
            async with lock:
                for en, da in zip(chunk, got):
                    have[en] = da
                    fh.write(json.dumps({"en": en, "da": da},
                                        ensure_ascii=False) + "\n")
        await asyncio.gather(*[one(c) for c in chunks])
    print(f"spec map: {len(have):,} strings", flush=True)
    return have


VALUE_SYS = """Du oversætter VÆRDIER fra kald til et værktøjs-API til dansk.

Du får en nummereret liste. Hver linje har formen `felt: værdi`. Oversæt KUN
værdien og svar med præcis lige så mange linjer i samme rækkefølge.

- Feltnavnet er kun kontekst. Skriv det ikke i svaret.
- Almindelige ord oversættes: "comedy" bliver "komedie", "rectangle" bliver
  "rektangel", "sports" bliver "sport".
- Egennavne, firmanavne, titler på film og bøger, personnavne og stednavne
  skrives uændret.
- Koder, forkortelser, ISO-sprogkoder ("en", "da"), valutaer ("USD"),
  identifikatorer, tal, datoer, e-mails og URL'er skrives uændret.
- Svar KUN med værdierne, én per linje, uden numre og uden feltnavn."""


async def translate_values(session, keys, tries=3):
    """Translate distinct (slot, value) pairs. `keys` are lexicon keys."""
    shown = []
    for k in keys:
        slot, _, val = k.partition(VALUE_SEP)
        shown.append(f"{slot.split('.')[-1] or 'værdi'}: {val}")
    body = {"model": MODEL, "temperature": 0.2,
            "messages": [{"role": "system", "content": VALUE_SYS},
                         {"role": "user", "content": "\n".join(
                             f"{i+1}. {s}" for i, s in enumerate(shown))}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "vaerdier", "strict": True, "schema": {
                    "type": "object",
                    "properties": {"linjer": {"type": "array",
                                              "items": {"type": "string"}}},
                    "required": ["linjer"], "additionalProperties": False}}}}
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    if a == tries - 1:
                        print(f"  value HTTP {r.status}: "
                              f"{(await r.text())[:120]}", flush=True)
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                out = json.loads(
                    d["choices"][0]["message"]["content"])["linjer"]
                # the model sometimes echoes "felt: værdi" back despite the
                # instruction; keep only what follows the first colon when the
                # prefix matches the field we sent.
                cleaned = []
                for k, o in zip(keys, out):
                    slot = k.partition(VALUE_SEP)[0].split(".")[-1]
                    o = _TAG.sub("", o, count=1).lstrip()
                    if slot and o.lower().startswith(f"{slot.lower()}:"):
                        o = o.split(":", 1)[1].strip()
                    cleaned.append(o)
                if len(cleaned) == len(keys):
                    return cleaned
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return None


def _load_spec_map(cache_path):
    """The catalogue table, off disk. Keyed by the english string."""
    have = {}
    if cache_path.exists():
        for line in cache_path.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                have[r["en"]] = r["da"]
            except Exception:
                continue
    return have


def _load_value_map(cache_path):
    """The lexicon, straight off disk. No session, no network."""
    have = {}
    if cache_path.exists():
        for line in cache_path.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                have[r["k"]] = r["da"]
            except Exception:
                continue
    return have


async def build_value_map(session, rows, cache_path, batch=40, concurrency=24):
    """(tool, arg_key, value) -> danish, for every distinct value. Resumable.

    Same shape as build_spec_map. The point is not the cost saving but the
    determinism: one key, one translation, applied to the spec's enum and to
    every invocation that uses it, so the contract cannot desynchronise.
    """
    have = {}
    if cache_path.exists():
        for line in cache_path.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                have[r["k"]] = r["da"]
            except Exception:
                continue
    want = sorted({s[1] for r in rows for s in value_segments(r)})
    todo = [k for k in want if k not in have]
    print(f"values: {len(want):,} distinct (slot,value), {len(have):,} cached, "
          f"{len(todo):,} to translate", flush=True)
    if not todo:
        return have
    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    n = [0]
    with cache_path.open("a", buffering=1) as fh:
        async def one(chunk):
            async with sem:
                got = await translate_values(session, chunk)
            n[0] += 1
            if n[0] % 20 == 0:
                print(f"  values {n[0]}/{len(chunks)} batches", flush=True)
            if not got:
                return
            async with lock:
                for k, da in zip(chunk, got):
                    have[k] = da
                    fh.write(json.dumps({"k": k, "da": da},
                                        ensure_ascii=False) + "\n")
        await asyncio.gather(*[one(c) for c in chunks])
    print(f"value map: {len(have):,} entries", flush=True)
    return have


def _glossary(row, value_map=None, spec_map=None, limit=24):
    """Canonical terms for THIS row, for the per-row prose request.

    Both global passes finish before any row is translated, so by the time the
    conversation is sent we already know that this row's `rectangle` is
    `rektangel`. Handing that over is strictly better than rewriting the prose
    afterwards: substitution can only touch an exact quoted string, so it
    leaves inflected and unquoted mentions behind -- idx 72 kept saying
    "standardformer som circle, rectangle osv." with no quotes to anchor to --
    whereas a glossary lets the model decline the word properly (rektangler,
    rektanglets) and use it in running text.
    """
    terms = {}
    if value_map:
        for _p, key, en in value_segments(row):
            da = value_map.get(key)
            if da and da.strip().lower() != en.strip().lower():
                terms[en.strip()] = da.strip()
    # DESCRIPTIONS ARE DELIBERATELY EXCLUDED. Offering "The principal amount of
    # the loan" -> "Lanets hovedstol" taught the model to write the Danish
    # description where it had been keeping the English parameter identifier,
    # and dnt-token-lost went 0 -> 2 on a 198-row smoke (idx 108 `principal`,
    # idx 179 `price_range`). Descriptions are cited verbatim in only 2.21% of
    # rows and the substitution pass already handles those, so the glossary
    # buys almost nothing here and costs DNT adherence. Values are what recur
    # in prose and need inflecting, so values are what the glossary carries.
    # shortest first: single words are the ones that recur in prose, and a
    # truncated list should keep those rather than long descriptions
    return dict(sorted(terms.items(), key=lambda kv: len(kv[0]))[:limit])


def build_request(row, segs, value_map=None, spec_map=None):
    pinned = sorted(x for x in _pinned_values(row) if x)
    lines = "\n".join(f"{i+1}. [{k}] {t}" for i, (_p, k, t) in enumerate(segs))
    gloss = _glossary(row, value_map, spec_map)
    head = f"BEVAR UÆNDRET (skriv disse ordret på engelsk):\n{', '.join(pinned)}\n\n"
    if gloss:
        head += ("FASTE OVERSÆTTELSER (brug præcis disse danske ord, også når "
                 "du bøjer dem):\n"
                 + "\n".join(f'  "{en}" -> "{da}"' for en, da in gloss.items())
                 + "\n\n")
    return head + f"Oversæt disse {len(segs)} tekststykker:\n{lines}"


async def translate(session, row, segs, tries=3, value_map=None, spec_map=None):
    body = {"model": MODEL, "temperature": 0.3,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": build_request(
                             row, segs, value_map, spec_map)}],
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
                    # SURFACE THE STATUS. Swallowing it made a $150 key cap
                    # look like silent attrition: 6,345 rows returned None with
                    # zero exceptions logged and no clue why. The same 403 wall
                    # was misdiagnosed twice on the extraction job before the
                    # status code was printed.
                    if a == tries - 1:
                        body_txt = (await r.text())[:160]
                        print(f"  HTTP {r.status}: {body_txt}", flush=True)
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
    ap.add_argument("--retry-failed", action="store_true",
                    help="gate the cache, DROP the rows that fail, then run "
                         "normally so they are re-translated. Failures are "
                         "mostly recoverable -- a single skipped description "
                         "or an argument left English -- so retrying beats "
                         "discarding the conversation.")
    ap.add_argument("--respec", action="store_true",
                    help="rebuild the global catalogue map and re-splice tool/"
                         "parameter descriptions into an EXISTING cache, "
                         "leaving conversation text untouched. Repairs a cache "
                         "translated before the spec pass existed, without "
                         "paying to redo conversations that are already right.")
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
            print(build_request(
                r, segs,
                _load_value_map(args.out / "value_map.jsonl"),
                _load_spec_map(args.out / "spec_map.jsonl"))[:2500])
        kinds = Counter(k for r in rows for _p, k, _t in segments(r))
        print(f"\nsegment kinds over {len(rows)} rows: {dict(kinds)}")
        return

    # RESUME. Rows are appended as they return, keyed by their index in the
    # source file, and a rerun skips whatever is already there. Previously the
    # cache was written once after asyncio.gather, so a crash at row 19,000 of
    # 20,000 lost the run and everything it cost. Append-and-skip also makes
    # the job interruptible on purpose: stop it, inspect, restart.
    # Repair a torn tail BEFORE appending. A hard kill can leave a partial
    # final line with no newline; opening in append mode then writes the next
    # record directly onto it, corrupting both. Measured: one row silently lost
    # per interrupted run, and the in-memory count still reported success.
    if cache.exists() and cache.stat().st_size:
        with cache.open("rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                size = f.seek(0, os.SEEK_END)
                pos = size
                while pos > 0:
                    step = min(65536, pos)
                    pos -= step
                    f.seek(pos)
                    nl = f.read(step).rfind(b"\n")
                    if nl != -1:
                        keep = pos + nl + 1
                        f.truncate(keep)
                        break
                else:
                    keep = 0
                    f.truncate(0)
                print(f"repaired torn tail in {cache} "
                      f"({size - keep:,} bytes dropped)", flush=True)

    done = {}
    if cache.exists():
        with cache.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                except Exception:
                    continue          # a torn last line from a hard kill
                if "idx" in p:
                    done[p["idx"]] = p
        print(f"resume: {len(done):,} rows already in {cache}", flush=True)

    if args.retry_failed and done:
        keep, drop = {}, 0
        for i, p in done.items():
            if gate(p["orig"], p["da"]):
                drop += 1
            else:
                keep[i] = p
        if drop:
            with cache.open("w") as f:
                for i in sorted(keep):
                    f.write(json.dumps(keep[i], ensure_ascii=False) + "\n")
            done = keep
            print(f"retry-failed: dropped {drop:,} failing rows, "
                  f"{len(keep):,} kept", flush=True)

    if args.respec:
        import aiohttp
        pairs = list(done.values())
        async with aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {_key()}",
                         "Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=300)) as s:
            spec_map = await build_spec_map(
                s, [p["orig"] for p in pairs], args.out / "spec_map.jsonl")
            value_map = await build_value_map(
                s, [p["orig"] for p in pairs], args.out / "value_map.jsonl")
        # This is also the cheap upgrade path for an ALREADY translated corpus:
        # conversations are left alone and only catalogue text and values are
        # rewritten from the global tables, so v1 -> v2 costs one value pass
        # (2,483 strings) instead of a full retranslation (24,459 segments).
        fixed = fixed_v = 0
        for p in pairs:
            new = json.loads(json.dumps(p["da"]))
            for path, kind, en in spec_segments(p["orig"]):
                da = spec_map.get(en.strip())
                if da is None:
                    continue
                cur = _get(new, path)
                if cur != da:
                    _set(new, path, da)
                    fixed += 1
            # Values: spec enums AND call arguments, from one table. Addressed
            # by path off the ORIGINAL row, so an argument the per-row pass had
            # already translated is overwritten with the canonical form -- that
            # is the whole point, and it is what makes the two agree.
            for path, key, _en in value_segments(p["orig"]):
                da = value_map.get(key)
                if da is None:
                    continue
                try:
                    cur = _get(new, path)
                except Exception:
                    continue
                if cur != da:
                    _set(new, path, da)
                    fixed_v += 1
            p["da"] = new
        tmp = cache.with_suffix(".respec")
        with tmp.open("w") as f:
            for p in sorted(pairs, key=lambda x: x["idx"]):
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        tmp.replace(cache)
        print(f"respec: {fixed:,} description fields + {fixed_v:,} values "
              f"rewritten across {len(pairs):,} rows -> {cache}", flush=True)

    if args.gate_only or args.respec:
        pairs = list(done.values()) or [json.loads(l) for l in cache.open()]
    else:
        import aiohttp
        # Excluded by INDEX, never by filtering `rows` -- idx is the row's
        # position in the source file and the resume key, so dropping entries
        # from the list would renumber everything after them and invalidate the
        # entire cache.
        skip = {i for i, r in enumerate(rows) if is_language_tool_row(r)}
        if skip:
            print(f"skipping {len(skip):,} language-tool rows "
                  f"(translation destroys their premise)", flush=True)
        todo = [(i, r) for i, r in enumerate(rows)
                if i not in done and i not in skip]
        print(f"to translate: {len(todo):,} of {len(rows):,}", flush=True)
        sem = asyncio.Semaphore(args.concurrency)
        write_lock = asyncio.Lock()
        spec_map = {}
        n_done = [0]
        errs = Counter()
        # line-buffered append: each row is durable the moment it lands
        with cache.open("a", buffering=1) as fh:
            async with aiohttp.ClientSession(
                    headers={"Authorization": f"Bearer {_key()}",
                             "Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=300)) as s:
                # Catalogue text first, once, globally.
                spec_map = await build_spec_map(
                    s, [r for _i, r in todo], args.out / "spec_map.jsonl")
                # Then every distinct VALUE, also once, also globally. Both
                # tables are built before any row is spliced so a resumed run
                # reuses identical mappings -- a value translated on Monday and
                # a value translated on Friday must agree, or the corpus
                # re-acquires exactly the defect this pass removes.
                value_map = await build_value_map(
                    s, [r for _i, r in todo], args.out / "value_map.jsonl")

                async def one(i, r):
                    # One malformed row must not abort the job. asyncio.gather
                    # propagates the first exception and cancels the rest, so
                    # an IndexError on row 2,250 killed a 20k run outright.
                    # Failures return None and stay uncached, so a rerun
                    # retries them.
                    try:
                        segs = conv_segments(r)      # catalogue handled above
                        async with sem:
                            tr = await translate(s, r, segs, value_map=value_map,
                                             spec_map=spec_map)
                        n_done[0] += 1
                        if n_done[0] % 250 == 0:
                            print(f"  {n_done[0]}/{len(todo)}", flush=True)
                        if tr is None:
                            return None
                        rec = {"idx": i, "orig": r,
                               "da": splice(r, segs, tr, spec_map, value_map)}
                    except Exception as e:
                        errs[type(e).__name__] += 1
                        return None
                    async with write_lock:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    return rec
                got = await asyncio.gather(*[one(i, r) for i, r in todo])
        fresh = [g for g in got if g]
        pairs = list(done.values()) + fresh
        print(f"\ntranslated {len(fresh):,} new "
              f"({len(fresh) + len(done):,} total) -> {cache}")
        if errs:
            print(f"  row errors: {dict(errs)}", flush=True)
        failed = len(todo) - len(fresh)
        if failed:
            print(f"  {failed:,} rows returned nothing and are NOT cached; "
                  f"rerun to retry them", flush=True)

    # Verdicts are PERSISTED, not just printed. The gate takes ~10 minutes of
    # langdetect over the corpus, so anything downstream that needs to know
    # which rows passed -- the renderer, a publish step -- would otherwise have
    # to recompute it or, worse, silently use every row.
    fails = Counter()
    ok = 0
    # Read the lexicon off disk rather than requiring a session: --gate-only
    # and --respec both need it, and it is exactly the table the run just wrote.
    gate_vmap = _load_value_map(args.out / "value_map.jsonl")
    gate_smap = _load_spec_map(args.out / "spec_map.jsonl")
    print(f"gate: {len(gate_vmap):,} canonical values, "
          f"{len(gate_smap):,} catalogue strings loaded", flush=True)
    verdict_path = args.out / "gate_verdicts.jsonl"
    with verdict_path.open("w") as vf:
        for p in pairs:
            bad = gate(p["orig"], p["da"], gate_vmap, gate_smap)
            vf.write(json.dumps({"idx": p.get("idx"), "bad": bad}) + "\n")
            if bad:
                for b in bad:
                    fails[b.split("(")[0]] += 1
            else:
                ok += 1
    print(f"verdicts -> {verdict_path}", flush=True)
    print(f"\nGATE: {ok}/{len(pairs)} clean")
    for k, v in fails.most_common():
        print(f"   {v:>4}  {k}")

    # PLANTED CONTROLS: corrupt a copy in each way the gate claims to catch.
    # Without this the pass rate is uninterpretable -- a gate that never fires
    # looks identical to data that is always right.
    if pairs:
        base = pairs[0]
        ctrl, ctrl_base = {}, {}
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
        # SYNTHETIC, like the enum control. Sampling the batch's first row
        # made this blind: if that row happened to carry no snake_case
        # identifier inside a conversation segment, the control corrupted
        # nothing and PASSED, which is exactly what happened on the 19,347-row
        # run -- the dnt-token-lost count was real but unverifiable.
        dnt_ok = {"tools": [{"function": {
            "name": "get_stock_price", "description": "Hent aktiekurs",
            "parameters": {"properties": {"company_name": {"type": "string"}}}}}],
            "conversations": [
                {"role": "user", "content": "Hvad er kursen?"},
                {"role": "assistant",
                 "content": "Jeg bruger parameteren company_name til opslaget."}]}
        dnt_bad = json.loads(json.dumps(dnt_ok))
        dnt_bad["conversations"][1]["content"] = (
            "Jeg bruger parameteren firma_navn til opslaget.")
        ctrl["identifier translated in prose"] = dnt_bad
        ctrl_base["identifier translated in prose"] = dnt_ok

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
        ctrl_base["enum desynced from spec"] = enum_spec

        # SYNTHETIC. This was drawn from the batch -- it hunted for a call
        # argument with a string value over 4 chars -- and on a rerun no row
        # matched, so the control was never added and the output silently
        # listed 7 controls instead of 8. A vanished control is worse than a
        # failing one: nothing reports it, and the value check it guards was
        # left unverified. Caught only by diffing against a previous run.
        chain_ref = {"tools": [{"function": {
            "name": "create_note", "parameters": {"properties": {
                "title": {"type": "string"}}}}}],
            "conversations": [{"role": "assistant", "content": "",
                               "tool_calls": [{"function": {
                                   "name": "create_note",
                                   "arguments": {"title": "Shopping List"}}}]}]}
        chain_bad = json.loads(json.dumps(chain_ref))
        chain_bad["conversations"][0]["tool_calls"][0]["function"][
            "arguments"]["title"] = "en helt anden formulering"
        ctrl["value chain broken"] = chain_bad
        ctrl_base["value chain broken"] = chain_ref
        gate_vmap = dict(gate_vmap)
        gate_vmap[_value_key("create_note", "title", "Shopping List")] = "Indkøbsliste"

        # SYNTHETIC: a sampled row exercises this only 5% of the time.
        echo_ref = {"tools": [{"function": {
            "name": "search_restaurants", "parameters": {"properties": {
                "cuisine": {"type": "string"}}}}}],
            "conversations": [{"role": "assistant",
                               "content": 'I will use cuisine: "Italian".',
                               "tool_calls": [{"function": {
                                   "name": "search_restaurants",
                                   "arguments": {"cuisine": "Italian"}}}]}]}
        echo_bad = json.loads(json.dumps(echo_ref))
        # the CALL is canonical, so only the prose check can catch this
        echo_bad["conversations"][0]["tool_calls"][0]["function"][
            "arguments"]["cuisine"] = "italiensk"
        echo_bad["conversations"][0]["content"] = 'Jeg bruger cuisine: "Italian".'
        ctrl["stale prose echo"] = echo_bad
        ctrl_base["stale prose echo"] = echo_ref
        gate_vmap = dict(gate_vmap)
        gate_vmap[_value_key("search_restaurants", "cuisine", "Italian")] = "italiensk"

        # SYNTHETIC: only 2.21% of rows cite a description verbatim, so a
        # drawn control would leave this check looking tested when it was not.
        EN_D = "The shape to calculate the area for"
        cite_ref = {"tools": [{"function": {
            "name": "calculate_area", "parameters": {"properties": {
                "shape": {"type": "string", "description": EN_D}}}}}],
            "conversations": [{"role": "assistant",
                               "content": f'The description says "{EN_D}".'}]}
        cite_bad = json.loads(json.dumps(cite_ref))
        cite_bad["tools"][0]["function"]["parameters"]["properties"][
            "shape"]["description"] = "Formen, som arealet skal beregnes for"
        cite_bad["conversations"][0]["content"] = (
            f'Beskrivelsen siger "{EN_D}", saa jeg bruger den.')
        ctrl["stale spec citation"] = cite_bad
        ctrl_base["stale spec citation"] = cite_ref
        gate_smap = dict(gate_smap)
        gate_smap[EN_D] = "Formen, som arealet skal beregnes for"

        # A control that is never constructed reports nothing -- the output
        # just lists one line fewer, and the check it guards goes unverified.
        # That happened twice: the sampled value-chain control found no
        # matching row on a rerun, and an edit deleted the prose-echo control
        # outright. Both were caught only by diffing against an earlier run.
        EXPECTED_CONTROLS = 8
        if len(ctrl) != EXPECTED_CONTROLS:
            missing = EXPECTED_CONTROLS - len(ctrl)
            print(f"\n*** {missing} PLANTED CONTROL(S) MISSING: built "
                  f"{len(ctrl)} of {EXPECTED_CONTROLS} -- "
                  f"{sorted(ctrl)} ***", flush=True)
        print("\nPLANTED CONTROLS (each must FAIL):")
        for name, bad_row in ctrl.items():
            ref = ctrl_base.get(name, base["orig"])
            res = gate(ref, bad_row, gate_vmap, gate_smap)
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
