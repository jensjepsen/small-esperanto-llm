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
import random
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


# ── symbolized twins ────────────────────────────────────────────────────────
#
# Half the corpus is rendered with parameter/result keys replaced by inert
# symbols, so that scoring well REQUIRES reading the Danish descriptions rather
# than recalling that `price` means price. English keys are already inert to a
# Danish-only model -- `cups_left` tokenises to ['c','ups','_','le','ft'] -- but
# they are inert CONSISTENTLY, so they can be memorised per tool.
#
# Symbolizing happens HERE, on the English source, BEFORE translation. Doing it
# after translation was tried and fails: 93.2% of rows name a parameter key in
# their reasoning ("...der tager et company_name ... {"company_name": "Apple"}"),
# so a Danish-side rename leaves the prose contradicting the call AND handing
# the model the mapping it was supposed to have to look up. Renaming first
# means the translator sees a coherent row and writes coherent Danish about the
# symbol.
#
# It also moves the ambiguity into English, where it is tractable: bare `time`
# in Danish prose means "hour" (58 rows) and must not be touched, whereas in the
# English source the same token is unambiguously the identifier in context.
SYM_CTX = "(?:parameter|field|argument|key|value|property)"


def _sym_prose(text, kmap_all):
    """Rewrite identifier mentions in English reasoning.

    Only in contexts where the token is unambiguously the identifier: inside a
    JSON fragment, quoted, adjacent to a schema word, or shaped like an
    identifier (contains `_`). A bare common word is left alone.
    """
    if not text:
        return text
    for orig, sym in sorted(kmap_all.items(), key=lambda kv: -len(kv[0])):
        e = re.escape(orig)
        text = re.sub(rf'(["\'])({e})(["\'])', rf'\1{sym}\3', text)
        text = re.sub(rf'(?<![\w])({e})(\s*:)', rf'{sym}\2', text)
        text = re.sub(rf'(?<![\w])({e})(\s+{SYM_CTX})', rf'{sym}\2', text,
                      flags=re.I)
        text = re.sub(rf'({SYM_CTX}\s+)({e})(?![\w])', rf'\1{sym}', text,
                      flags=re.I)
        if "_" in orig:                      # identifier-shaped: safe bare
            text = re.sub(rf'(?<![\w]){e}(?![\w])', sym, text)
    return text


def source_key_map(row, idx):
    """The (tool -> {real key: symbol}) map symbolize_source() would use.

    Exposed because the RENDERER needs it: result-field keys carry no
    description, so unlike parameters they cannot be matched back through
    text. The function is deterministic in (row, idx), so recomputing here
    reproduces exactly what generation did.
    """
    return _symbolize(row, idx)[1]


def symbolize_source(row, idx):
    return _symbolize(row, idx)[0]


def _symbolize(row, idx):
    """Return an English row with parameter/result keys replaced by symbols.

    Consistent within the row across spec, call, result and prose; permuted per
    row so no symbol becomes a stable second name for a key.
    """
    row = json.loads(json.dumps(row))
    keys = {}
    for t in row.get("tools", []) or []:
        f = t.get("function") or {}
        keys.setdefault(f.get("name"), set()).update(
            ((f.get("parameters") or {}).get("properties") or {}))
    last = None
    for m in row.get("conversations", []):
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            last = fn.get("name") or last
            a = fn.get("arguments")
            if isinstance(a, str):
                try: a = json.loads(a)
                except Exception: a = None
            if isinstance(a, dict):
                keys.setdefault(last, set()).update(a)
        if m.get("role") == "tool" and last:
            try: obj = json.loads(m.get("content") or "")
            except Exception: continue
            def collect(o):
                if isinstance(o, dict):
                    keys[last].update(o)
                    for v in o.values(): collect(v)
                elif isinstance(o, list):
                    for v in o: collect(v)
            keys.setdefault(last, set())
            collect(obj)
    # TOOL NAMES ARE LEFT REAL, deliberately.
    #
    # Symbolizing them was tried and cost far more than it bought. It silently
    # defeated the language-tool filter, which matches on the NAME (467 rows
    # whose premise translation destroys sailed through). It collapsed the
    # distractor pool from 873 tools to 12, because every row reuses t1..tn, so
    # every catalogue drew the same twelve specs -- 3x prompt bloat and 7.2% of
    # rows truncated past their model turn, training nothing. And it made the
    # corpus's own returns map meaningless, since `t1.p1` denotes something
    # different in every row, which then needed a reverse map plus a join back
    # to pristine originals to undo.
    #
    # What it bought was small: names stay memorisable. But selection is
    # already tested by the shuffled 2-8 catalogue, and the point of the
    # symbolized half is the KEYS -- whether the model reads "Aktuel
    # aktiekurs" or recalls that `price` means price.
    tmap = {}
    kmap = {}
    for tool, ks in keys.items():
        ks = sorted(x for x in ks if isinstance(x, str))
        syms = [f"p{i+1}" for i in range(len(ks))]
        random.Random(f"{idx}:{tool}").shuffle(syms)
        kmap[tool] = dict(zip(ks, syms))
    flat = {k: v for m in kmap.values() for k, v in m.items()}
    flat.update(tmap)          # names are identifiers too, safe to rewrite bare

    def ren(o, m):
        if isinstance(o, dict):
            return {m.get(k, k): ren(v, m) for k, v in o.items()}
        if isinstance(o, list):
            return [ren(v, m) for v in o]
        return o

    for t in row.get("tools", []) or []:
        f = t.get("function") or {}
        m = kmap.get(f.get("name"), {})
        if f.get("name") in tmap:
            f["name"] = tmap[f["name"]]
        props = (f.get("parameters") or {}).get("properties")
        if isinstance(props, dict):
            f["parameters"]["properties"] = {m.get(k, k): v for k, v in props.items()}
            req = (f.get("parameters") or {}).get("required")
            if isinstance(req, list):
                f["parameters"]["required"] = [m.get(k, k) for k in req]
    last = None
    for msg in row.get("conversations", []):
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            last = fn.get("name") or last
            m = kmap.get(last, {})
            a = fn.get("arguments")
            if isinstance(a, dict):
                fn["arguments"] = {m.get(k, k): v for k, v in a.items()}
            if fn.get("name") in tmap:
                fn["name"] = tmap[fn["name"]]
        if msg.get("role") == "tool" and last:
            try: obj = json.loads(msg.get("content") or "")
            except Exception: pass
            else:
                msg["content"] = json.dumps(ren(obj, kmap.get(last, {})),
                                            ensure_ascii=False)
        if msg.get("role") in ("assistant", "user") and msg.get("content"):
            msg["content"] = _sym_prose(msg["content"], flat)
    return row, {"tools": tmap, "keys": kmap}


# ── canonical spec schema ───────────────────────────────────────────────────
#
# The source carries EIGHT distinct spec shapes. Beyond the intended
# name/description/parameters(/returns), it has: `response` instead of returns
# (1,409), `arguments` instead of parameters (417), a `required` list at the
# top level instead of inside parameters (1,200), and a {"type":"function",
# "function":{...}} wrapper. Types are spelled dict/str/float as often as
# object/string/number.
#
# That mattered because extraction was a WHITELIST of known paths, so the
# dialects were never walked -- their descriptions were never translated and
# the still-english gate never saw them, which is how "Name of the city to get
# the date and time for." reached the published corpus. Distractor padding then
# amplified rare dialects 16x, since the pool weights tools by distinct NAME
# rather than by frequency.
#
# Preserve-don't-touch was the right rule for TRANSLATION -- splice by path so
# structure cannot drift. It was never the right rule for the CATALOGUE, which
# we already synthesise: we add returns, shuffle, and pad with other rows'
# tools. Carrying source dialects through that is inheritance, not fidelity.
#
# So: project every tool onto one shape at ingest, before anything is extracted
# or translated, and let the generic walker below cover whatever remains.
_TYPE_ALIASES = {"dict": "object", "str": "string", "float": "number",
                 "int": "integer", "bool": "boolean", "list": "array"}


def _canon_types(node):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                out[k] = _TYPE_ALIASES.get(v.lower(), v)
            else:
                out[k] = _canon_types(v)
        return out
    if isinstance(node, list):
        return [_canon_types(v) for v in node]
    return node


def canonical_spec(fn):
    """One tool spec, one shape. Unknown keys are dropped, not carried."""
    if not isinstance(fn, dict):
        return None
    if "function" in fn and isinstance(fn["function"], dict):
        fn = fn["function"]                      # unwrap {"type":"function",...}
    name = fn.get("name")
    if not name:
        return None
    params = fn.get("parameters")
    if not isinstance(params, dict):
        # the `arguments` dialect: a bare property map, no envelope
        args = fn.get("arguments")
        params = ({"type": "object", "properties": args}
                  if isinstance(args, dict) else {"type": "object",
                                                  "properties": {}})
    params = dict(params)
    params.setdefault("type", "object")
    if not isinstance(params.get("properties"), dict):
        params["properties"] = {}
    # a `required` list sitting at the top level belongs inside parameters
    if isinstance(fn.get("required"), list) and "required" not in params:
        params["required"] = fn["required"]
    out = {"name": name,
           "description": fn.get("description") or "",
           "parameters": _canon_types(params)}
    # `response` is DROPPED, not normalised: where both exist they describe the
    # same fields twice, in two languages -- trading_login carries returns
    # {"status": "Status for login"} and response {"status": "Login status
    # message."}. Our returns block is derived from the payloads the tool
    # actually produced, so it is the better of the two; the 17 tools with only
    # `response` are never called in the corpus, so their outputs are never
    # shown to the model anyway.
    if isinstance(fn.get("returns"), dict):
        out["returns"] = _canon_types(fn["returns"])
    return out


def canonicalise_row(row):
    tools = []
    for t in row.get("tools", []) or []:
        c = canonical_spec(t.get("function") if isinstance(t, dict)
                           and t.get("function") else t)
        if c:
            tools.append({"type": "function", "function": c})
    row["tools"] = tools
    return row


def is_language_tool_row(row) -> bool:
    if row.get("_language_tool"):
        return True          # marked pre-symbolization, name no longer matches
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
# Word order is not fixed in this corpus: `word_count` and `count_words` are
# both present, as are `character_count` and `count_characters`. Spelling one
# order missed 59 instances (count_words 55, count_characters 4) whose values
# were then translated -- and a translated sentence has a different word count
# than the tool result recorded.
CHAR_DEPENDENT = re.compile(
    r"palindrom|anagram|spell|rhym|syllab|acronym|letter|reverse|cipher|"
    r"encod|decod|vowel|consonant|capitali[sz]|uppercase|lowercase|"
    r"word.?count|count.?word|char.?count|count.?char", re.I)

# Parameter keys whose value is a CREDENTIAL. Translating "password123" to
# "kodeord123" changes the secret, and the row then teaches the model to
# rewrite the thing it was asked to transmit verbatim.
CREDENTIAL_KEY = re.compile(
    r"password|passwd|secret|token|api_?key|access_?key|hash|salt|pin_?code",
    re.I)


def _char_dependent_values(row) -> set:
    """Values that must hold still: character-dependent tools, and any
    argument under a credential-shaped key."""
    names = set()
    for t in row.get("tools", []) or []:
        f = t.get("function") or {}
        if (CHAR_DEPENDENT.search(f.get("name") or "")
                or CHAR_DEPENDENT.search(f.get("description") or "")):
            names.add(f.get("name"))
    out = set()
    for m in row.get("conversations", []):
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            a = fn.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    continue
            if not isinstance(a, dict):
                continue
            char_tool = fn.get("name") in names
            for k, v in a.items():
                if not (isinstance(v, str) and v.strip()):
                    continue
                if char_tool or CREDENTIAL_KEY.search(str(k)):
                    out.add(v.strip())
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


def _all_descriptions(node, prefix):
    """(path, text) for EVERY description anywhere under `node`.

    Replaces an enumerated whitelist of known paths. The whitelist existed only
    because segments are spliced back by exact path -- but a walker emits exact
    paths too, so the restriction bought nothing and cost coverage: `response`,
    `arguments` and oneOf dialects were never extracted, so their text was
    never translated and the still-english gate never saw it.

    `returns` is EXCLUDED: we generate that block ourselves, in Danish, from
    the tool_result payloads. Translating our own output would be circular.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "returns":
                continue
            if k == "description" and isinstance(v, str) and v.strip():
                yield prefix + (k,), v
            else:
                yield from _all_descriptions(v, prefix + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _all_descriptions(v, prefix + (i,))


def _nested_desc_segments(node, base, depth=0):
    """Descriptions inside a parameter that is itself an object or array.

    Walks `properties` and `items` to any depth. Kept separate from the
    top-level loop so the path stays exact -- these are spliced back by
    address, never by search.
    """
    out = []
    if depth > 4 or not isinstance(node, dict):
        return out
    props = node.get("properties")
    if isinstance(props, dict):
        for k, v in props.items():
            if not isinstance(v, dict):
                continue
            if v.get("description"):
                out.append((base + ("properties", k, "description"),
                            "param_desc", v["description"]))
            out.extend(_nested_desc_segments(
                v, base + ("properties", k), depth + 1))
    items = node.get("items")
    if isinstance(items, dict):
        if items.get("description"):
            out.append((base + ("items", "description"),
                        "param_desc", items["description"]))
        out.extend(_nested_desc_segments(items, base + ("items",), depth + 1))
    return out


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
        # WALK EVERYTHING under the tool, not an enumerated set of paths.
        for path, txt in _all_descriptions(f, ("tools", i, "function")):
            kind = "tool_desc" if path[-2:] == ("function", "description") \
                else "param_desc"
            segs.append((path, kind, txt))
        # enum entries are VALUES, handled by value_segments()
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


def _strip_generated(row):
    """A copy without `returns`, which the annotate pass ADDS deliberately.

    The skeleton check compares structure against the English original, and
    `returns` exists in neither the source nor the translation -- it is
    generated metadata. Gating an annotated corpus without stripping it failed
    18,100 of 19,442 rows as structure-changed, which is the check working
    correctly on a difference we introduced on purpose.
    """
    row = json.loads(json.dumps(row))
    for t in row.get("tools", []) or []:
        f = t.get("function") if isinstance(t, dict) and t.get("function") else t
        if isinstance(f, dict):
            f.pop("returns", None)
    return row


def gate(orig, new, value_map=None, spec_map=None):
    new = _strip_generated(new)
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
    #
    # Tokens we DELIBERATELY keep English are removed before asking what
    # language the sentence is. They are not evidence of a failed translation
    # -- they are the translation working. Left in, "Hvad med Microsoft?" and
    # "Interessant! Hvad med 'computer'?" both read as English to langdetect,
    # which flips easily on a three-word sentence whose only content word is a
    # proper noun, a ticker, or a pinned palindrome word. That produced 967
    # false failures on the full corpus (373 user turns, 102 responses), and
    # the rate rose with pinning coverage -- i.e. the check was penalising the
    # fix. Strip them, then judge the Danish that remains.
    # Quoted spans go too. A quoted string in this corpus is a title, a name or
    # a citation -- "Hvad med filmen \"The Godfather\"?" is correct Danish that
    # langdetect calls English because the title outweighs the sentence, and
    # "Her er nogle actionfilm fra 2020: \"Tenet\", \"Extraction\", ..." even
    # more so. 443 further false failures, 99 responses and 73 user turns.
    # The planted control leaves an UNQUOTED segment in English, so it still
    # fires; a wholly untranslated sentence is not wrapped in quotes.
    keep_en = {t for t in _pinned_names(orig) if isinstance(t, str) and t}
    _QUOTED = re.compile(rf'[{_Q}][^{_Q}]{{0,120}}[{_Q}]')
    def _strip_pinned(s: str) -> str:
        s = _QUOTED.sub(" ", s)
        for t in sorted(keep_en, key=len, reverse=True):
            s = re.sub(rf'[{_Q}]?{re.escape(t)}[{_Q}]?', " ", s, flags=re.I)
        return s
    en = da = 0
    for (_p, kind, txt), (_p2, _k2, new_txt) in zip(o_segs, n_segs):
        if kind not in ("user", "think", "response", "tool_desc", "param_desc"):
            continue
        probe = _strip_pinned(new_txt)
        # after stripping, a short remainder carries too little signal to judge
        if len(probe.split()) < 5:
            continue
        if _is_english(probe):
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


# ── returns schema ──────────────────────────────────────────────────────────
#
# Tool specs describe their INPUTS and say nothing about what they return:
# 0 of 86,304 specs carry a returns/response/output key. So an answer turn
# gets `name`, a Danish `description`, a Danish description per input
# parameter -- and then a raw JSON blob whose keys are English.
#
# For a Danish-only model those keys are inert. `cups_left` tokenises to
# ['c','ups','_','le','ft'] and `dog_years` to ['d','og','_','ye','ars'] --
# fragments with no meaning in its space, one of which ('og') is Danish for
# "and". It cannot infer that cups_left is the remaining coffee any more than
# it could from `xk_92`.
#
# That is why grounding is near-gold on SEEN tools and guesswork on unseen
# ones: the pairing was memorised, never read. Probed on an unfamiliar
# get_coffee_status it answered with `last_service` instead of `cups_left`,
# and on calculate_dog_years it never reported dog_years at all.
#
# Inputs already have the fix -- a Danish description per field, which is why
# argument binding works at all. This gives results the same channel. The
# field names and types are read off the tool_result payloads already in the
# corpus; only the descriptions are generated, once per (tool, field), not per
# row.
RETURNS_MAX_FIELDS = 6
RETURN_SEP = "\x00"


def _return_key(tool, path):
    return f"{tool or ''}{RETURN_SEP}{path}"


def _walk_leaves(obj, prefix=""):
    """(path, json-type) per leaf. Lists descend into their first element so a
    list of records yields `events[].name` rather than an opaque `events`."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        if obj:
            yield from _walk_leaves(obj[0], f"{prefix}[]")
    else:
        t = ("boolean" if isinstance(obj, bool) else
             "number" if isinstance(obj, (int, float)) else
             "null" if obj is None else "string")
        yield prefix, t


def return_fields(row):
    """[(key, tool, path, type, example)] for every field this row's tools
    actually returned. Attributed to the tool whose call preceded the result."""
    out, last = [], None
    for m in row.get("conversations", []):
        for tc in (m.get("tool_calls") or []):
            last = (tc.get("function") or {}).get("name") or last
        if m.get("role") != "tool" or not last:
            continue
        try:
            obj = json.loads(m.get("content") or "")
        except Exception:
            continue
        for path, t in _walk_leaves(obj):
            if not path:
                continue
            out.append((_return_key(last, path), last, path, t,
                        (m.get("content") or "")[:120]))
    return out


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


# ── parallel gate ───────────────────────────────────────────────────────────
#
# langdetect over ~19.5k rows is the slow half of a build (~10 min single
# threaded) and the work is embarrassingly parallel -- gate() is pure.
#
# FOUR workers, deliberately. This box goes down under sustained load on more
# than four cores, so the pool is capped rather than sized from cpu_count().
#
# The maps go through `initializer`, not through each task: 10,057 catalogue
# strings plus 4,106 values is far past the point where pickling them per row
# costs more than the check itself.
_G_VMAP = None
_G_SMAP = None


def _gate_init(vmap, smap):
    global _G_VMAP, _G_SMAP
    _G_VMAP, _G_SMAP = vmap, smap


def _gate_one(pair):
    return pair.get("idx"), gate(pair["orig"], pair["da"], _G_VMAP, _G_SMAP)


def gate_all(pairs, vmap, smap, workers=4):
    """[(idx, bad)] in input order. Falls back to serial for tiny inputs."""
    if len(pairs) < 200 or workers <= 1:
        _gate_init(vmap, smap)
        return [_gate_one(p) for p in pairs]
    import multiprocessing as mp
    with mp.Pool(workers, initializer=_gate_init,
                 initargs=(vmap, smap)) as pool:
        # imap, not imap_unordered: the verdict file is joined on position by
        # downstream readers, so order is part of the contract
        out = []
        for i, r in enumerate(pool.imap(_gate_one, pairs, chunksize=64), 1):
            out.append(r)
            if i % 5000 == 0:
                print(f"  gate {i}/{len(pairs)}", flush=True)
        return out


RETURNS_SYS = """Du skriver korte danske feltbeskrivelser til et værktøjs-API.

Du får en nummereret liste. Hver linje beskriver ét felt, som et værktøj
returnerer: værktøjets navn, feltets sti, feltets type og et eksempel på
værktøjets svar.

Skriv én kort dansk beskrivelse per linje -- hvad feltet indeholder, set fra
brugerens synspunkt. Svar med præcis lige så mange linjer i samme rækkefølge.

- Skriv som en feltbeskrivelse, ikke som en sætning: "Antal kopper kaffe
  tilbage", ikke "Dette felt indeholder antallet af kopper kaffe tilbage".
- Feltnavnet er på engelsk og skal IKKE oversættes eller gentages i svaret.
- Brug værktøjets navn og eksemplet til at forstå, hvad feltet betyder.
- Er feltet en statuskode eller teknisk metadata, så skriv det ("Statuskode
  for kaldet").
- Svar KUN med beskrivelserne, én per linje, uden numre."""


async def translate_returns(session, keys, meta, tries=3):
    """Danish description per (tool, result-field). `meta` maps key -> (tool,
    path, type, example)."""
    shown = []
    for k in keys:
        tool, path, typ, ex = meta[k]
        shown.append(f"værktøj={tool} felt={path} type={typ} svar={ex[:70]}")
    # Explicit ceiling. None of these passes set one, so they ran on the
    # provider default -- and one batch failed every attempt because its
    # search_book examples carry large payloads, so the reply was cut short,
    # the length check rejected the chunk, and all 40 entries were lost. The
    # split-retry below is the backstop; this removes the cause. 40 short
    # Danish descriptions need well under 2k tokens, so 4k is generous.
    body = {"model": MODEL, "temperature": 0.2, "max_tokens": 4000,
            "messages": [{"role": "system", "content": RETURNS_SYS},
                         {"role": "user", "content": "\n".join(
                             f"{i+1}. {x}" for i, x in enumerate(shown))}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "beskrivelser", "strict": True, "schema": {
                    "type": "object",
                    "properties": {"linjer": {"type": "array",
                                              "items": {"type": "string"}}},
                    "required": ["linjer"], "additionalProperties": False}}}}
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status != 200:
                    if a == tries - 1:
                        print(f"  returns HTTP {r.status}: "
                              f"{(await r.text())[:120]}", flush=True)
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                out = json.loads(
                    d["choices"][0]["message"]["content"])["linjer"]
                out = [_TAG.sub("", x, count=1).strip() for x in out]
                if len(out) == len(keys):
                    return out
        except Exception:
            await asyncio.sleep(1.5 * (a + 1))
    return None


def _load_returns_map(cache_path):
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


async def build_returns_map(session, rows, cache_path, batch=40,
                            concurrency=24):
    """(tool, result-field) -> danish description. Resumable.

    Same shape as build_spec_map/build_value_map, so pointing --out at an
    existing corpus reuses every earlier pass and pays only for this one.
    """
    have = _load_returns_map(cache_path)
    meta = {}
    for r in rows:
        for key, tool, path, typ, ex in return_fields(r):
            meta.setdefault(key, (tool, path, typ, ex))
    want = sorted(meta)
    todo = [k for k in want if k not in have]
    print(f"return fields: {len(want):,} distinct (tool,field), "
          f"{len(have):,} cached, {len(todo):,} to describe", flush=True)
    if not todo:
        return have
    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    n = [0]
    with cache_path.open("a", buffering=1) as fh:
        async def run_chunk(chunk):
            """Translate a chunk, halving it on failure.

            One batch failed on every attempt and took all 40 of its entries
            with it -- a contiguous alphabetical run from save_contact.message
            to search_book.publication_year, i.e. exactly one slice. The
            search_book examples carry large JSON payloads, so the batch
            tripped a length limit and the all-or-nothing length check
            discarded the lot. Splitting isolates the offender instead of
            losing its neighbours.
            """
            async with sem:
                got = await translate_returns(session, chunk, meta)
            if got:
                return [(k, da) for k, da in zip(chunk, got)]
            if len(chunk) == 1:
                print(f"  returns: giving up on {chunk[0]!r}", flush=True)
                return []
            mid = len(chunk) // 2
            left = await run_chunk(chunk[:mid])
            right = await run_chunk(chunk[mid:])
            return left + right

        async def one(chunk):
            pairs_out = await run_chunk(chunk)
            n[0] += 1
            if n[0] % 20 == 0:
                print(f"  returns {n[0]}/{len(chunks)} batches", flush=True)
            if not pairs_out:
                return
            async with lock:
                for k, da in pairs_out:
                    have[k] = da
                    fh.write(json.dumps({"k": k, "da": da},
                                        ensure_ascii=False) + "\n")
        await asyncio.gather(*[one(c) for c in chunks])
    print(f"returns map: {len(have):,} entries", flush=True)
    return have


def attach_returns_to_row(row, rmap, symmap=None):
    """Write the `returns` block into the row's own tools, in the row's symbols.

    Done HERE rather than in the renderer because this is the only place that
    holds the returns map and the row's symbol map at once. Reconstructing it
    downstream needed three joins -- row by idx, tool by description, field by
    recomputing the symbolizer -- and each failure was silent: the documented
    keys simply came out disjoint from the payload.

    Row-present paths win the dedupe. The corpus spells one concept many ways
    (data.company / data.company_name), but sorted order put `current_price`
    ahead of `data.price` for the same description, dropping the very field the
    row returns.
    """
    kmap = (symmap or {}).get("keys") or {}
    tmap = (symmap or {}).get("tools") or {}
    rev_t = {v: k for k, v in tmap.items()}

    present, last = set(), None
    for m in row.get("conversations", []):
        for tc in (m.get("tool_calls") or []):
            last = (tc.get("function") or {}).get("name") or last
        if m.get("role") == "tool" and last:
            try:
                obj = json.loads(m.get("content") or "")
            except Exception:
                continue
            for pth, _t in _walk_leaves(obj):
                if pth:
                    present.add((last, pth))

    for t in row.get("tools", []) or []:
        f = t.get("function") if isinstance(t, dict) and t.get("function") else t
        if not isinstance(f, dict):
            continue
        shown = f.get("name")
        real = rev_t.get(shown, shown)          # symbolized rows map back
        fields = {k.split(RETURN_SEP, 1)[1]: v for k, v in rmap.items()
                  if k.startswith(f"{real}{RETURN_SEP}")}
        if not fields:
            continue
        exact = dict(kmap.get(real) or {})
        used, nxt = set(exact.values()), 1
        if exact or tmap:                        # symbolized row
            for path in sorted(fields):
                for part in path.split("."):
                    bare = part[:-2] if part.endswith("[]") else part
                    if bare in exact:
                        continue
                    while f"p{nxt}" in used:
                        nxt += 1
                    exact[bare] = f"p{nxt}"
                    used.add(f"p{nxt}")

        def ren(path):
            out = []
            for part in path.split("."):
                bare = part[:-2] if part.endswith("[]") else part
                new = exact.get(bare, bare)
                out.append(new + "[]" if part.endswith("[]") else new)
            return ".".join(out)

        here = {q for tn, q in present if tn == shown}
        ordered = sorted(fields.items(),
                         key=lambda kv: (ren(kv[0]) not in here, kv[0]))
        # CAP the block. Documenting every corpus-wide field for all 8
        # catalogue tools pushed the median symbolized prompt to 4,170 tokens
        # and 7.5% of rows past the 8,048 limit -- and because the model turn
        # sits at the END, truncation removed it entirely and the row trained
        # nothing. Row-present paths are ordered first, so the cap drops the
        # fields this row never returns rather than the ones it does.
        seen, props = set(), {}
        for path, desc in ordered:
            if len(props) >= RETURNS_MAX_FIELDS:
                break
            if desc in seen:
                continue
            seen.add(desc)
            props[ren(path)] = {"description": desc}
        f["returns"] = {"type": "object", "properties": props}
    return row


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
    # Ceiling sized from the INPUT. Without one the request ran on the provider
    # default and the longest rows came back truncated, failed the
    # segment-count check, and were abandoned after three tries -- 137 rows
    # whose median length is 7,015 chars against a corpus median of 842. They
    # failed identically at concurrency 96, 48 and 8, which is what ruled out
    # rate limiting and pointed at length. Danish runs longer than English, so
    # allow ~1.6 chars out per char in plus headroom for the JSON envelope.
    _chars = sum(len(t) for _p, _k, t in segs)
    body = {"model": MODEL, "temperature": 0.3,
            "max_tokens": max(4000, min(32000, int(_chars * 1.6 / 2.5) + 1500)),
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
    ap.add_argument("--returns-from", type=Path, default=None,
                    help="Read returns_map.jsonl from ANOTHER corpus. A "
                         "symbolized corpus's own map is keyed by per-row "
                         "symbols and is meaningless across rows; the real-key "
                         "corpus's map is the authority, resolved through the "
                         "stored symbol map.")
    ap.add_argument("--annotate", action="store_true",
                    help="Local pass, no API: store the symbol map on each row "
                         "and bake the `returns` block into its tools, so the "
                         "renderer needs no mapping logic at all.")
    ap.add_argument("--symbolize", action="store_true",
                    help="Replace parameter/result KEYS with inert symbols in "
                         "the ENGLISH source before translating, so the "
                         "reasoning is written about the symbol. Doing this "
                         "after translation fails: 93.2%% of rows name a "
                         "parameter key in their prose, which both contradicts "
                         "the call and hands the model the mapping.")
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
    sym_maps = {}
    # CANONICALISE FIRST. Every downstream stage -- language-tool filter,
    # symbolization, extraction, translation, gate -- then sees one spec shape
    # instead of eight, and the generic walker covers whatever text it holds.
    rows = [canonicalise_row(r) for r in rows]
    print(f"canonicalised {len(rows):,} rows to one spec shape", flush=True)

    if args.symbolize:
        # Mark language-tool rows BEFORE renaming. is_language_tool_row matches
        # on the tool NAME, so symbolizing to t1/t2 makes it match nothing and
        # the 467 rows whose premise translation destroys sail through -- a
        # user saying "translate this English sentence" becomes Danish while
        # the call still carries source_language="English". The flag survives
        # the rename; the filter reads it below.
        for r in rows:
            if is_language_tool_row(r):
                r["_language_tool"] = True
        # Load-time transform, so the symbolized corpus is an independent build
        # with its own caches rather than a second code path through main().
        # Descriptions are untouched, so spec_map can be copied in from the
        # real-key build and warms straight through; only the conversations,
        # values and return paths are new.
        out_rows = []
        for i, r in enumerate(rows):
            new, m = _symbolize(r, i)
            sym_maps[i] = m          # PERSISTED below: never recompute it
            out_rows.append(new)
        rows = out_rows
        print(f"SYMBOLIZED {len(rows):,} source rows before translation",
              flush=True)
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

    if args.annotate:
        rmap = _load_returns_map(
            (args.returns_from or args.out) / "returns_map.jsonl")
        # PRISTINE originals. A symbolized corpus stored its post-transform
        # input as `orig`, so recomputing the map from it symbolizes twice
        # ({"t1": "t2"}). The real-key corpus holds the untouched English rows
        # under the same idx, and is the only sound basis for the map.
        pristine = {}
        if args.returns_from:
            for line in (args.returns_from / "translated.jsonl").open():
                if line.strip():
                    rec0 = json.loads(line)
                    pristine[rec0["idx"]] = rec0["orig"]
            print(f"pristine originals: {len(pristine):,}", flush=True)
        pairs = [json.loads(l) for l in cache.open() if l.strip()]
        print(f"annotate: {len(pairs):,} rows, {len(rmap):,} return descriptions",
              flush=True)
        n_map = n_ret = 0
        for rec in pairs:
            if args.symbolize:
                base = pristine.get(rec["idx"])
                if base is None:
                    continue          # cannot map it soundly: leave untouched
                rec["symmap"] = _symbolize(base, rec["idx"])[1]
                n_map += 1
            before = json.dumps(rec["da"].get("tools"), ensure_ascii=False)
            rec["da"] = attach_returns_to_row(rec["da"], rmap,
                                              rec.get("symmap"))
            if json.dumps(rec["da"].get("tools"), ensure_ascii=False) != before:
                n_ret += 1
        tmp = cache.with_suffix(".annot")
        with tmp.open("w") as f:
            for rec in sorted(pairs, key=lambda x: x["idx"]):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(cache)
        print(f"annotate: stored {n_map:,} symbol maps, attached returns to "
              f"{n_ret:,} rows -> {cache}", flush=True)
        return

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
            await build_returns_map(
                s, [p["orig"] for p in pairs], args.out / "returns_map.jsonl")
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
                # Third global pass: what each tool RETURNS. Independent of the
                # other two -- it describes result fields rather than
                # translating anything -- so an existing corpus warms straight
                # through and pays only for this.
                await build_returns_map(
                    s, rows, args.out / "returns_map.jsonl")

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
                        da = splice(r, segs, tr, spec_map, value_map)
                        rec = {"idx": i, "orig": r, "da": da}
                        if sym_maps:
                            # The map is DATA, not something to reconstruct.
                            # Recomputing it downstream made correctness depend
                            # on _symbolize being unchanged since the row was
                            # written -- which nearly broke when tool-name
                            # symbolization landed mid-build.
                            rec["symmap"] = sym_maps.get(i)
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
    verdicts = gate_all(pairs, gate_vmap, gate_smap,
                        workers=int(os.environ.get("GATE_WORKERS", "4")))
    with verdict_path.open("w") as vf:
        for idx, bad in verdicts:
            vf.write(json.dumps({"idx": idx, "bad": bad}) + "\n")
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
