"""Generate Danish tool dialogues that pose a CHOICE in the payload.

Emits the `translated.jsonl` shape, so stages 4-8 of TOOLMIND_PIPELINE.md and
audit_distractors.py consume the output unchanged. Replaces stages 1-3.

See scripts/TOOL_GENERATOR_PLAN.md. The argument in one line: retrofitting a
distractor onto an existing payload is capped by no-type-overlap (76.9% of v8
tools, 46.1% of ToolACE's), but a tool we INVENT can be given same-type,
similar-magnitude competing return fields by construction.

Two LLM calls per dialogue:
  A. invent the tool AND its returns contract together, naming which field
     answers and which same-type field competes with it;
  C. write the Danish turns -- given the payload, which is synthesised
     locally first so the model writes its answer against real values rather
     than inventing them.

Conventions: English tool/parameter/field names, Danish descriptions and
turns, JSON-schema vocabulary, OpenRouter only.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from toolmind_distractor_values import classify, generate as gen_value  # noqa: E402
# The invented-number check is NOT reimplemented here. gen_tool_answer_turns'
# version is calibrated on the corpus: Danish writes 898.09 as "898,09" and
# 12000 as "12.000", answers round 50.26548 to "50.27" and render 0.98 as
# "98%", and a digit opening a line is a list enumerator, not a quantity.
# A naive float compare flags all of those as fabrications.
from gen_tool_answer_turns import (  # noqa: E402
    _nums as _answer_nums, _traces_to, LIST_ENUM, NUM)

MODEL = "google/gemini-2.5-flash-lite"
URL = "https://openrouter.ai/api/v1/chat/completions"
SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")
DA_CHARS = re.compile(r"[æøåÆØÅ]")
EN_WORDS = re.compile(r"\b(the|and|with|from|your|please|here|are|this|that|"
                      r"will|would|should|have|been|about|which)\b", re.I)


def _key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for p in (Path.home() / "or", Path.home() / ".or"):
        if p.exists():
            return p.read_text().strip()
    raise SystemExit("no OpenRouter key (~/or)")


# ── Axis 1: structural template ────────────────────────────────────────────
# Kept from gen_tool_call_sft.py, retyped to JSON Schema. The grammar is what
# makes schema-shape coverage measurable rather than whatever the model feels
# like emitting; `template_hash` is how you check it afterwards.

N_PARAM_WEIGHTS = {0: 0.04, 1: 0.26, 2: 0.30, 3: 0.22, 4: 0.12, 5: 0.06}
TYPE_WEIGHTS = {"string": 0.44, "integer": 0.20, "number": 0.12,
                "boolean": 0.08, "enum": 0.10, "object": 0.06}


def _weighted(rng, weights):
    r = rng.random() * sum(weights.values())
    acc = 0.0
    for k, w in weights.items():
        acc += w
        if r <= acc:
            return k
    return list(weights)[-1]


def sample_template(rng):
    n = _weighted(rng, N_PARAM_WEIGHTS)
    params = []
    for i in range(n):
        t = _weighted(rng, TYPE_WEIGHTS)
        p = {"slot": i, "type": t, "required": rng.random() < 0.60,
             "decorators": []}
        if t in ("integer", "number") and rng.random() < 0.45:
            p["decorators"].append("min_max")
        if t == "string" and rng.random() < 0.20:
            p["decorators"].append("length")
        if t == "string" and rng.random() < 0.10:
            p["decorators"].append("format")
        if t == "object":
            p["sub_params"] = [
                {"type": _weighted(rng, {"string": 0.5, "integer": 0.3,
                                         "number": 0.1, "boolean": 0.1}),
                 "required": rng.random() < 0.7}
                for _ in range(rng.randint(2, 3))]
        params.append(p)
    return {"n_params": n, "params": params}


def template_hash(tpl):
    parts = []
    for p in tpl["params"]:
        s = p["type"][:3] + ("R" if p["required"] else "O")
        if p["decorators"]:
            s += "".join(d[0] for d in p["decorators"])
        if p.get("sub_params"):
            s += "{" + ",".join(x["type"][:3] for x in p["sub_params"]) + "}"
        parts.append(s)
    return "|".join(parts) or "(none)"


def render_template(tpl):
    if not tpl["params"]:
        return "(ingen parametre)"
    out = []
    for p in tpl["params"]:
        req = "påkrævet" if p["required"] else "valgfri"
        line = f'  {p["slot"]+1}. {p["type"]} ({req})'
        if p["decorators"]:
            line += " + " + ", ".join(p["decorators"])
        for sp in p.get("sub_params", []):
            line += f'\n     - {sp["type"]}'
        out.append(line)
    return "\n".join(out)


# ── Axis 2 x plan ──────────────────────────────────────────────────────────
# Rates target what the corpus LACKS. The old generator emitted multi_chain in
# 23 of 37,032 rows (0.06%); ToolACE runs 38.2% parallel and 22.2% multi-turn.

PLAN_WEIGHTS = {"single": 0.40, "parallel": 0.25, "multi_turn": 0.20,
                "clarify": 0.10, "refuse": 0.05}

PLAN_TEXT = {
    "single": "Brugeren spørger om én ting. Assistenten kalder værktøjet én "
              "gang og svarer.",
    "parallel": "Brugeren beder om TO ting i samme tur. Assistenten laver TO "
                "kald i SAMME assistent-tur, og svarer derefter samlet.",
    "multi_turn": "Samtalen har mindst to runder: kald, svar, brugeren spørger "
                  "videre, nyt kald, nyt svar. Mindst én tur imellem er ren "
                  "samtale uden værktøj.",
    "clarify": "Brugeren udelader en PÅKRÆVET parameter. Assistenten spørger "
               "efter den præcise oplysning, brugeren giver den, og FØRST "
               "derefter kaldes værktøjet.",
    "refuse": "Brugeren beder om noget, værktøjet IKKE kan. Assistenten siger "
              "det ligeud og kalder ikke værktøjet. Der er INGEN kald og "
              "INGEN is_answer-tur: skriv afslaget som almindelig tekst.",
}

TOOL_SYS = """Du opfinder ét værktøj (et API-endepunkt) til et dansk scenarie.

NAVNE PÅ ENGELSK, BESKRIVELSER PÅ DANSK. Værktøjsnavn, parameternavne og
returfeltnavne er engelske og i snake_case. Alle beskrivelser er på dansk.

Værktøjet skal have en `returns`-kontrakt, og kontrakten SKAL stille et VALG:

- `answer_field` er det felt, der besvarer brugerens spørgsmål.
- `competitor_field` er et ANDET felt med SAMME TYPE og en værdi af SAMME
  STØRRELSESORDEN. Det skal være fristende at forveksle med answer_field,
  men det er ikke svaret.
- Eksempel: en kaffemaskine returnerer {"cups_left": 8, "floor": 4}. Begge er
  små heltal. Kun beskrivelsen afgør, hvilket der er svaret.
- Et felt må IKKE gentage en parameter, værktøjet fik ind.
- Drift-felter (status, request_id, latency_ms) tæller ikke som konkurrent.

Giv 2-3 eksempelværdier pr. returfelt. Værdierne skal være realistiske og
FORSKELLIGE fra hinanden."""

TOOL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["name", "description", "parameters", "returns",
                 "answer_field", "competitor_field", "user_goal"],
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "user_goal": {"type": "string"},
        "parameters": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "type", "description", "required"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "required": {"type": "boolean"},
                    "enum": {"type": "array", "items": {"type": "string"}},
                }}},
        "returns": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "type", "description", "examples"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "examples": {"type": "array",
                                 "items": {"type": "string"}},
                }}},
        "answer_field": {"type": "string"},
        "competitor_field": {"type": "string"},
    }}

# ── unbundled generation ───────────────────────────────────────────────────
# One prompt used to carry six constraints at once: plan shape, per-call
# payload selection, competitor avoidance, no fabrication, Danish-only, and
# valid JSON in `arguments_json`. The failures were what over-constrained
# single-shot generation looks like -- answers quoting the previous call's
# payload, `{` emitted as an assistant turn, duplicate turns.
#
# Split in two. The SKELETON never sees a payload, so it cannot fabricate from
# one; payloads are then synthesised from the arguments it actually produced,
# which is what finally keys them correctly; and each ANSWER is written
# against its own payload alone, so quoting a different call's data is not
# possible rather than merely gated.

SKELETON_SYS = """Du skriver skelettet af en kort dansk samtale mellem en
bruger og en assistent, der bruger et værktøj.

Du får IKKE værktøjets resultat, og du skal IKKE skrive assistentens svar på
et resultat. Marker i stedet den tur med `is_answer: true` og lad `text` være
tom -- svaret skrives bagefter.

`is_answer: true` gælder KUN en tur, der kommer LIGE EFTER et kald og svarer
på dets resultat. Alt andet assistenten siger -- et afslag, et opklarende
spørgsmål, en afsluttende bemærkning -- er almindelig tekst med
`is_answer: false`, som du skriver selv.

Regler:
- Følg planen præcist.
- FØRSTE tur er ALTID brugerens, og den skal være et rigtigt spørgsmål i hel
  sætning -- ikke et id, et navn eller et tal alene. Assistenten må aldrig
  åbne samtalen, heller ikke for at spørge om en manglende oplysning.
- ALT bruger og assistent siger er på DANSK. Kun værktøjsnavne og
  parameternavne er engelske.
- En tur med et kald har tom `text` og `is_answer: false`.
- EFTER hvert kald skal der komme præcis én tur med `is_answer: true`.
- Udfyld aldrig en påkrævet parameter med tom tekst. Mangler oplysningen, så
  lad assistenten spørge brugeren om den FØR kaldet.
- `arguments_json` skal være gyldig JSON og passe til parametrenes typer.
- Skriv ikke tankerækker."""

SKELETON_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["turns"],
    "properties": {"turns": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["role", "text", "is_answer"],
        "properties": {
            "role": {"type": "string", "enum": ["user", "assistant"]},
            "text": {"type": "string"},
            "is_answer": {"type": "boolean"},
            "calls": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "arguments_json"],
                "properties": {"name": {"type": "string"},
                               "arguments_json": {"type": "string"}}}},
        }}}}}

def _called_name(row):
    for m in row["da"]["conversations"]:
        for c in (m.get("tool_calls") or []):
            return (c.get("function") or {}).get("name")
    return None


ANSWERS_SYS = """Du skriver assistentens svar på et værktøjsresultat, på DANSK.

Du får en liste af opgaver. For HVER opgave får du brugerens spørgsmål, de
ARGUMENTER kaldet blev lavet med, og NØJAGTIGT det resultat, værktøjet gav.

Returner PRÆCIS ét svar pr. opgave, i samme rækkefølge som opgaverne. Får du
én opgave med flere resultater, skal de sammenfattes i ÉT svar.

Svaret skal stemme overens med `argumenter`: blev der kaldt med
`include_failed_prints: true`, må svaret ikke sige, at fejlloggen ikke er
med. Nævn kun en afgrænsning, hvis argumenterne faktisk sætter den.

Regler pr. svar:
- Brug feltet `answer_field` fra resultatet. Det er svaret.
- Nævn IKKE `competitor_field` -- hverken feltnavnet eller dets værdi, og
  heller ikke omskrevet.
- Brug KUN tal og tekst, der står i dette resultat, i argumenterne eller i
  spørgsmålet. Find intet på.
- Skriv aldrig feltnavne som `brood_pattern_score` i teksten. Sig hvad feltet
  BETYDER, på dansk.
- Ét kort, naturligt svar i hele sætninger -- ikke en rå gengivelse af
  feltets værdi og ikke JSON."""

ANSWERS_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["svar"],
    "properties": {"svar": {"type": "array", "items": {"type": "string"}}}}


# ── gates ──────────────────────────────────────────────────────────────────

def _danish(s):
    s = str(s or "")
    return bool(s.strip()) and (bool(DA_CHARS.search(s))
                                or len(EN_WORDS.findall(s)) == 0)


def gate_tool(t):
    """Why each check exists is in the reason; None means clean."""
    if not SNAKE.match(t.get("name") or ""):
        return f"tool-name-not-snake_case:{t.get('name')}"
    rets = t.get("returns") or []
    if len(rets) < 2:
        return "returns-under-2-fields"
    names = [r.get("name") for r in rets]
    if len(set(names)) != len(names):
        return "duplicate-return-fields"
    for r in rets:
        if not SNAKE.match(r.get("name") or ""):
            return f"return-not-snake_case:{r.get('name')}"
        if not _danish(r.get("description")):
            return f"return-description-not-danish:{r.get('name')}"
        if len(r.get("examples") or []) < 2:
            return f"return-examples-under-2:{r.get('name')}"
        if len(set(map(str, r.get("examples") or []))) < 2:
            return f"return-examples-not-varied:{r.get('name')}"
    params = t.get("parameters") or []
    for p in params:
        if not SNAKE.match(p.get("name") or ""):
            return f"param-not-snake_case:{p.get('name')}"
        if not _danish(p.get("description")):
            return f"param-description-not-danish:{p.get('name')}"
    # THE POINT OF THE SCRIPT. Without a same-type competitor the payload
    # states one fact and "emit the number you just saw" is a perfect policy.
    a, c = t.get("answer_field"), t.get("competitor_field")
    by = {r["name"]: r for r in rets if r.get("name")}
    if a not in by:
        return f"answer-field-not-in-returns:{a}"
    if c not in by:
        return f"competitor-field-not-in-returns:{c}"
    if a == c:
        return "answer-and-competitor-identical"
    if by[a].get("type") != by[c].get("type"):
        return f"competitor-type-differs:{by[a].get('type')}/{by[c].get('type')}"
    # A tool that names its return fields after the CONTRACT satisfies
    # `answer_field in returns` trivially and teaches nothing: one tool came
    # back returning literally {"answer_field": ..., "competitor_field": ...}.
    # A field called `data_value` or `requested_data` names no fact, so the
    # answer reads it as "a count of whatever was asked" -- `{"action":
    # "beredskabsplan"}` came back 11207 and was reported as "11207
    # beredskabsplaner". Matched by SHAPE: the exact-name list I wrote last
    # round missed every one of these.
    meta_hit = [k for k in by if CONTENTLESS.match(k)]
    if meta_hit:
        return f"return-field-named-after-contract:{sorted(meta_hit)[0]}"
    pnames = {p.get("name") for p in params}
    echo = pnames & set(by)
    if echo:
        return f"return-echoes-parameter:{sorted(echo)[:2]}"
    if not _danish(t.get("description")):
        return "tool-description-not-danish"
    return None


CATALOGUE_SEP = "Værktøjer:"


def strip_catalogue(content):
    """The user's actual words, with the tool catalogue removed.

    The catalogue rides inside the FIRST user turn and is full of JSON. Every
    ad-hoc `split("]")` or comma count I wrote against a raw first turn read
    the schema instead of the question -- four separate mis-measurements this
    session, including one reported as an 8% defect rate that was 0.6%.
    Import this instead of writing it again.
    """
    c = str(content or "")
    if CATALOGUE_SEP not in c:
        return c
    rest = c.split(CATALOGUE_SEP, 1)[1].lstrip()
    try:
        _cat, end = json.JSONDecoder().raw_decode(rest)
        return rest[end:].strip()
    except Exception:
        return ""


FAIL_WORDS = re.compile(
    r"(fejl|kunne ikke|mislyk|ikke muligt|ikke lykkedes|gik galt|"
    r"ingen data|ikke tilgængelig)", re.I)


def _mentions(text, value):
    """Does the answer cite this value?

    Numbers need TOKEN matching, not substring with a length floor: the whole
    point of a competitor is that it is a small value of the same magnitude as
    the answer, and `floor: 4` is one character. A substring rule with a
    minimum length -- which is what the answer-turn gate uses for prose --
    silently never fires on exactly the cases this gate exists to catch.
    """
    low = str(text).lower()
    s = str(value).strip()
    if not s:
        return False
    try:
        v = float(s.replace(",", "."))
    except ValueError:
        return len(s) > 2 and s.lower() in low
    for m in NUM.finditer(low):
        try:
            if float(m.group().replace(",", ".")) == v:
                return True
        except ValueError:
            continue
    return False


CONTENTLESS = re.compile(
    r"^(alternative_|alternate_|other_|requested_|returned_|primary_|"
    r"secondary_|main_)*"
    r"(answer|competitor|answer_field|competitor_field|result|value|data|"
    r"info|information|details|detail|item|entry|output|field|svar|felt)"
    r"(_field|_value|_data|_info|_result|_details)*$")

WORD = re.compile(r"[a-zA-ZæøåÆØÅ0-9]{4,}")
JSONISH = re.compile(r'^\s*[{}\[\]]|"\s*:\s*|[}\]]\s*$')
# The assistant's own question, wherever it lands. Two shapes: asking the
# user to supply something ("kan du oplyse"), and asking what the user WANTS
# -- which is the one that slipped, as an opening turn reading "Er det
# økologisk podning, du søger information om?". In a real user turn `du` is
# the one asked to act; here it is the one doing the wanting.
ASK_LIKE = re.compile(r"(vil du gerne|kan du oplyse|hvilken .* vil du|"
                      r"hvad vil du|kunne du oplyse|angiv venligst|"
                      r"\bdu (søger|ønsker|vil vide|tænker på|leder efter|"
                      r"mener|har brug for|er interesseret)|"
                      r"\b(ønsker|søger|leder) du\b|"
                      r"er du interesseret|har du brug for)", re.I)
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
# Scaffolding, not content. A fabricated week menu still echoes the weekday
# names of the real one -- "Mandag: Pasta bolognese" against "Mandag:
# Mexicansk gryderet" -- so counting those as shared content made the check
# pass the very row it was written for.
SCAFFOLD = {
    "mandag", "tirsdag", "onsdag", "torsdag", "fredag", "lørdag", "søndag",
    "januar", "februar", "marts", "april", "juni", "juli", "august",
    "september", "oktober", "november", "december", "uger", "ugen",
    "dagen", "dage", "klokken", "eller", "samt", "eksempel",
}


def _shares_content(answer, value, need=2):
    """Does the answer actually convey this string value?

    Word overlap, not substring: a correct answer reorders and reworders, so
    demanding the value verbatim rejects good prose. Two shared content words
    is deliberately weak -- this is here to catch WHOLESALE INVENTION, not to
    police faithfulness. A payload holding "Mandag: Mexicansk gryderet,
    Tirsdag: Grontsagssuppe" was answered "Mandag: Pasta bolognese, Tirsdag:
    Linsegryde" -- a different menu entirely, sharing only the weekday names.
    """
    a = {w.lower() for w in WORD.findall(str(answer))} - SCAFFOLD
    v = {w.lower() for w in WORD.findall(str(value))} - SCAFFOLD
    return len(a & v) >= need


def gate_dialogue(row, tool, plan=None):
    msgs = row["da"]["conversations"]
    roles = [m["role"] for m in msgs]
    if not msgs:
        return "no-turns"
    if roles[0] != "user":
        return "does-not-open-on-user"
    # A bare id or place name is not a question. Four of these shipped --
    # whole opening turns reading "Rønne-Køge" and "123" -- because the
    # detector that found them stayed in the review script and was never
    # promoted to a gate.
    opening = str(msgs[0].get("content") or "").strip()
    if len(opening.split()) < 3 or not re.search(r"[a-zæøåA-ZÆØÅ]{3,}", opening):
        return f"degenerate-opening:{opening[:24]}"
    for i, m in enumerate(msgs):
        c = str(m.get("content") or "")
        # The assistant's clarifying question landing in the USER's turn, so
        # the user appears to interview themselves.
        # ANY user turn, including the first. The original check started at
        # i > 0 because that is where I first saw it; procedural beats always
        # open on the user, so the dresser writes assistant-shaped questions
        # into turn 0 -- "Hvilken rugemaskine vil du gerne tjekke status for?"
        if m["role"] == "user" and strip_catalogue(c).rstrip().endswith("?") \
                and ASK_LIKE.search(strip_catalogue(c)):
            return "assistant-question-in-user-turn"
        # `{` and `}, "max_elevation_gain": 200 } }` were emitted as whole
        # assistant turns: structured output leaking into prose.
        if m["role"] == "assistant" and c.strip() \
                and JSONISH.search(c.strip()) \
                and not re.search(r"[a-zæøå]{4,}\s+[a-zæøå]{4,}", c.lower()):
            return "json-fragment-in-turn"
    for a, b in zip(msgs, msgs[1:]):
        if a["role"] == b["role"] == "assistant" \
                and str(a.get("content") or "").strip() \
                and str(b.get("content") or "").strip():
            return "consecutive-assistant-turns"
    for a, b in zip(roles, roles[1:]):
        if a == b == "user":
            return "user-after-user"
    names = {t["function"]["name"] for t in row["da"]["tools"]}
    byname = {t["function"]["name"]: t["function"] for t in row["da"]["tools"]}
    ncall = 0
    for i, m in enumerate(msgs):
        for c in (m.get("tool_calls") or []):
            ncall += 1
            fn = (c.get("function") or {})
            if fn.get("name") not in names:
                return "call-not-in-catalogue"
            if i + 1 >= len(msgs) or msgs[i + 1]["role"] != "tool":
                return "call-without-result"
            # A required parameter filled with "" is not a call, it is a
            # placeholder -- 6 rows called ferry_operations with every
            # argument set to the empty string.
            req = set((byname[fn["name"]].get("parameters")
                       or {}).get("required") or [])
            args = fn.get("arguments") or {}
            for k in req & set(args):
                if isinstance(args[k], str) and not args[k].strip():
                    return f"empty-required-argument:{k}"
                if args[k] is None:
                    return f"null-required-argument:{k}"
            # `{"lactate_threshold_1": null, "lactate_threshold_2": null,
            #  "max_lactate": null}` -- the model called the tool rather than
            # asking for data the user never gave.
            if args and all(v is None for v in args.values()):
                return "call-with-all-null-arguments"
    # A payload whose own dates contradict each other teaches the model to
    # read nonsense as fact. LEXICAL pairs only -- start/end, from/to. The
    # `runway_clear_time` vs `next_scheduled_maintenance` case found by eye is
    # semantic and this does NOT catch it.
    for m in msgs:
        if m["role"] != "tool":
            continue
        try:
            p = json.loads(m["content"])
        except Exception:
            continue
        if not isinstance(p, dict):
            continue
        dates = {k: v for k, v in p.items()
                 if isinstance(v, str) and ISO_DATE_RE.match(v)}
        for k1, v1 in dates.items():
            for k2, v2 in dates.items():
                lo, hi = k1.lower(), k2.lower()
                pair = (("start" in lo and "end" in hi)
                        or ("from" in lo and "to" in hi)
                        or ("begin" in lo and "slut" in hi))
                if pair and v1 > v2:
                    return f"payload-dates-out-of-order:{k1}>{k2}"
    # EVERY result must be answered. The end-of-dialogue check only looks at
    # the LAST turn, so a mid-dialogue result followed by the user speaking
    # again slipped through in 30.8% of rows -- the model would learn that a
    # tool result can simply be ignored.
    for i, m in enumerate(msgs):
        if m["role"] != "tool":
            continue
        nxt = next((x for x in msgs[i + 1:] if x["role"] != "tool"), None)
        if nxt is None or nxt["role"] != "assistant":
            return "result-not-answered"
    for m in msgs:
        if m["role"] in ("user", "assistant") and m.get("content"):
            if len(EN_WORDS.findall(str(m["content"]))) >= 4:
                return "turn-in-english"
    # Every plan but `refuse` must actually call the tool. A chat-only
    # dialogue passes every other check -- no calls means no result checks --
    # and then dies at the renderer's pedagogy filter, which dropped 53 of 121
    # rows against only 10 deliberate refusals.
    if ncall == 0 and plan not in (None, "refuse"):
        return "no-call-but-plan-requires-one"
    if roles[-1] != "assistant":
        return "does-not-end-on-assistant"
    final = str(msgs[-1].get("content") or "")
    if not final.strip():
        return "final-turn-empty"
    # the gold must USE the asked-for field and must NOT cite its competitor
    payloads = [json.loads(m["content"]) for m in msgs if m["role"] == "tool"]
    if ncall and not payloads:
        return "no-payload"
    # EVERY answer turn, not just the last. Checking only the final turn let
    # intermediate answers in multi-turn and parallel rows cite the
    # competitor -- 4 rows that audit_distractors caught as
    # ANSWER-CITES-DISTRACTOR after this gate had passed them.
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or not str(m.get("content") or "").strip():
            continue
        if not any(x["role"] == "tool" for x in msgs[:i]):
            continue
        for p in payloads:
            cv = p.get(tool["competitor_field"])
            if cv is not None and _mentions(m["content"], cv):
                return f"answer-cites-competitor:{tool['competitor_field']}"
    # FABRICATION. Checking only that the competitor is absent says nothing
    # about whether what IS cited is real: a payload holding
    # `health_score: 47` was answered "sundhedsscore er 95", and
    # `Tinglysningsnr. 11223/2021` came back as `Tinglysningsnr. 12345/2023`.
    # 20.2% of rows. Numbers the user supplied are fair to repeat, so the
    # allowed set is payloads + calls + user turns.
    # Scoped to the PREFIX, not the row. Allowing every number anywhere in the
    # dialogue lets an answer quote a payload that has not arrived yet: a turn
    # whose own result said 160 answered "145 ved 12.5 km/t", both values from
    # the NEXT call's payload, and a row-scoped check waved it through.
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or not str(m.get("content") or "").strip():
            continue
        if not any(x["role"] == "tool" for x in msgs[:i]):
            continue
        allowed = set()
        for x in msgs[:i]:
            if x["role"] in ("tool", "user"):
                allowed |= _answer_nums(x.get("content") or "")
            for c in (x.get("tool_calls") or []):
                allowed |= _answer_nums(json.dumps(
                    (c.get("function") or {}).get("arguments") or {},
                    ensure_ascii=False))
        scan = LIST_ENUM.sub(lambda x: " " * len(x.group()), m["content"])
        for tok in NUM.finditer(scan):
            if not _traces_to(tok.group(), allowed):
                return f"answer-invents-number:{tok.group()}"
    # A payload that says the call FAILED cannot be answered as though it
    # succeeded -- `{"status": "fejl", "data_value": 12310}` was reported as
    # "Der er 12310 valgstedsmedarbejdere."
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or not str(m.get("content") or "").strip():
            continue
        prev = msgs[i - 1] if i else None
        if not prev or prev["role"] != "tool":
            continue
        try:
            p = json.loads(prev["content"])
        except Exception:
            continue
        if not isinstance(p, dict):
            continue
        st = str(p.get("status") or p.get("state") or "").strip().lower()
        if st in ("fejl", "error", "failed", "failure", "mislykkedes",
                  "mislykket"):
            if not FAIL_WORDS.search(m["content"]):
                return "answer-ignores-failed-status"
    # DOCUMENTATION ECHO. A user turn that restates the answer field's own
    # description makes field selection a string match, which defeats the
    # same-type competitor: "Den aktuelle grundvandsstand målt i meter under
    # terræn" appeared as both the schema text and the entire utterance in
    # 21.1% of rows.
    af_spec = None
    for t in row["da"]["tools"]:
        fn = t.get("function") or {}
        props = ((fn.get("returns") or {}).get("properties") or {})
        if tool.get("answer_field") in props:
            af_spec = props[tool["answer_field"]].get("description")
            break
    if af_spec:
        dw = {w.lower() for w in WORD.findall(af_spec)} - SCAFFOLD
        for m in msgs:
            if m["role"] != "user":
                continue
            qw = {w.lower() for w in
                  WORD.findall(strip_catalogue(m.get("content")))} - SCAFFOLD
            if not dw or not qw:
                continue
            if len(dw & qw) >= max(2, int(0.8 * len(dw))) and len(qw - dw) <= 2:
                return "question-echoes-field-description"
    # PARROTING. The answers prompt is given the call arguments so it will not
    # contradict them -- that cut contradictions 21.3% -> 11.9%. The cost is
    # that the model can answer WITH an argument: asked about `dmx_address:
    # 285`, told `assigned_dmx_address: 202`, it replied "DMX-adressen 285 er
    # tildelt". Fluent, grounded-looking, and the wrong number.
    #
    # Fully deterministic: code knows the answer field's value and the
    # arguments it chose. Paraphrase is allowed for prose (word overlap) and
    # Danish number formatting for values, so only a genuine substitution --
    # answer field ABSENT, argument PRESENT -- is rejected.
    af = tool.get("answer_field")
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or not str(m.get("content") or "").strip():
            continue
        prev = msgs[i - 1] if i else None
        if not prev or prev["role"] != "tool":
            continue
        try:
            p = json.loads(prev["content"])
        except Exception:
            continue
        if not isinstance(p, dict) or af not in p:
            continue
        av = p[af]
        if isinstance(av, bool) or av in (None, ""):
            continue
        # A paraphrase counts as stating it, at ANY length: the answer field
        # "Træbeklædning (Skovfyr, trykimprægneret)" -- three words -- came
        # back as "træbeklædning af skovfyr, som er trykimprægneret", and a
        # >=4-word threshold called that a parrot.
        stated = (_mentions(m["content"], av) or
                  (isinstance(av, str) and _shares_content(m["content"], av)))
        if stated:
            continue
        # A FAILED call has no answer-field value to state, and naming the
        # subject you could not look up is the correct thing to say.
        st = str(p.get("status") or p.get("state") or "").strip().lower()
        if st in ("fejl", "error", "failed", "failure", "mislykkedes",
                  "mislykket") or FAIL_WORDS.search(m["content"]):
            continue
        call = next((c for x in reversed(msgs[:i])
                     for c in (x.get("tool_calls") or [])), None)
        # ONLY an argument naming the SAME QUANTITY as the answer field.
        # Mentioning an argument is normally correct -- you name the aircraft
        # you looked up. The defect is substitution: asked for `dmx_address`,
        # told `assigned_dmx_address`, answering with the former. Requiring the
        # names to share a stem took precision from 2/7 to the cases that are
        # actually wrong.
        af_stem = set(re.split(r"_+", af)) - {"assigned", "current", "next",
                                              "available", "estimated",
                                              "total", "actual", "requested"}
        for k, v in ((call or {}).get("function") or {}).get("arguments",
                                                             {}).items():
            if isinstance(v, bool) or v in (None, ""):
                continue
            if str(v).strip() == str(av).strip():
                continue
            k_stem = set(re.split(r"_+", k))
            if not (af_stem & k_stem):
                continue
            if _mentions(m["content"], v):
                return f"answer-parrots-argument:{k}~{af}"
    # STRING fabrication. The number check covers only half the space: an
    # invented menu contains no digits at all and sailed through. Scoped to
    # `answer_field`, which is by contract the value the turn reporting that
    # payload is supposed to convey.
    af = tool.get("answer_field")
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or not str(m.get("content") or "").strip():
            continue
        prev = msgs[i - 1] if i else None
        if not prev or prev["role"] != "tool":
            continue
        val = json.loads(prev["content"]).get(af)
        if not isinstance(val, str) or len(val.split()) < 6:
            continue
        if not _shares_content(m["content"], val):
            return f"answer-invents-text:{af}"
    return None


# ── payload synthesis (no LLM) ─────────────────────────────────────────────

def payload_for(tool, idx, args):
    """The payload this tool returns for THESE arguments.

    Keyed on the arguments themselves, which is only possible now that the
    skeleton is generated first. The previous design synthesised a fixed list
    up front and handed the k-th one to the k-th distinct argument set -- an
    approximation that collided past k and had to be gated.
    """
    salt = _hash(json.dumps(args or {}, sort_keys=True, ensure_ascii=False))
    return synth_payload(tool, idx ^ (salt & 0xFFFFFF))


PAYLOAD_SALT = 104729


def synth_payloads(tool, idx, k=5):
    """One payload PER CALL, not per row.

    k is 5, not 3: a dialogue with four distinct argument sets clamped its
    last two onto the same payload, so two different measurement types came
    back byte-identical.

    A row keyed on `idx` alone returned the byte-identical payload to every
    call in the dialogue -- 19 of 130 rows, 13 of them for demonstrably
    DIFFERENT arguments, so the corpus taught that a tool's output is
    independent of what you ask it. The k-th call gets the k-th payload, and
    the turn-writing prompt is told so.
    """
    return [synth_payload(tool, idx + i * PAYLOAD_SALT) for i in range(k)]


def synth_payload(tool, idx):
    """Values per row, derived from the contract.

    Deterministic in `idx` so a rerun reproduces the corpus, and varied across
    rows so a field is not a constant the model can memorise.

    COLLISIONS ARE RESAMPLED. Same-type competitors of similar magnitude land
    on the same value often -- 14 of 149 payloads in the first run -- and a
    competitor equal to the answer poses no choice at all, which is the one
    thing this generator exists to guarantee.
    """
    out = {}
    for r in tool["returns"]:
        ex = [_unquote(e) for e in (r.get("examples") or []) if str(e).strip()]
        kind = classify(ex)
        v = _value(r["name"], ex, idx, kind)
        for bump in range(1, 12):
            if str(v) not in {str(x) for x in out.values()}:
                break
            v = _value(r["name"], ex, idx + bump * 7919, kind)
        out[r["name"]] = v
    return out


def _unquote(s):
    """`'"ZONE-7"'` -> `ZONE-7`.

    The model returns examples as JSON strings and often quotes them a second
    time inside the string. Left alone the quotes reach the payload.
    """
    s = str(s).strip()
    if len(s) > 1 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1].strip()
    return s


def _hash(*parts):
    import hashlib
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8],
               16)


PCT_NAME = re.compile(r"(percent|percentage|pct|procent)")


def _numeric_span(examples, field=None):
    """(lo, hi, is_int) implied by the examples, or None."""
    vals = []
    for e in examples:
        m = NUM.search(str(e))
        if not m:
            return None
        try:
            vals.append(float(m.group().replace(",", ".")))
        except ValueError:
            return None
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    if lo == hi:
        lo, hi = lo * 0.6, hi * 1.6 if hi else 1.0
    span = hi - lo
    is_int = all(float(v).is_integer() for v in vals)
    lo_out, hi_out = lo - span * 0.4, hi + span * 0.4
    # A band around the examples must not cross zero when none of them do.
    # It produced `insured_value_dkk: -273092` and `parts_approval_pending: -1`
    # -- well-typed, in-range of the arithmetic, and meaningless as data.
    if lo >= 0:
        lo_out = max(0 if lo == 0 else min(vals) * 0.25, lo_out, 0)
    # And it must not cross 100 for a PERCENTAGE whose examples never do. The
    # band ran 40% past the examples with no ceiling, so utilisation examples
    # topping out at 88 produced 108, and a success RATE came back as 101.
    if field and PCT_NAME.search(str(field).lower()) and hi <= 100:
        hi_out = min(hi_out, 100.0)
    return lo_out, hi_out, is_int


def _value(field, examples, idx, kind):
    """A per-row value for one return field.

    FREE and NUMUNIT go through the EXAMPLES, not toolmind_distractor_values:
    that module resolves FREE against the LLM-sampled banks, which this
    generator never builds, so every free-text field came back None -- which
    then read as both CONSTANT-ACROSS-ROWS and duplicates-another-field. The
    contract already carries 2-3 examples per field; rotating them by a hash
    of (field, idx) is varied enough and costs nothing.

    NUMBERS ARE HELD TO THE MAGNITUDE THE EXAMPLES IMPLY. Generating freely
    within the TYPE produced a greenhouse at 224% humidity, a 3D printer at
    554.6 C and an incubator at 50.1 C -- all well-typed and all absurd, which
    teaches the model to read nonsense as fact. The examples carry the scale;
    stay inside a band around them.
    """
    if not examples:
        return None
    span = _numeric_span(examples, field)
    # A value that merely STARTS with a number is not a number+unit: dates,
    # log lines, ids and COORDINATE PAIRS all do. Rebuilding them as "<new
    # number> <rest>" produced "2023 -10-26: Modtog vaccination ...", and
    # treating "56.1250,10.1250" as numeric collapsed a lat/long pair to the
    # single float 57.63 -- half a position.
    if any(re.match(r"^\s*-?\d+[.,]?\d*\s*[-/:,;]", str(e)) for e in examples):
        return examples[_hash(field, idx) % len(examples)]
    if span and kind in ("int", "float", "numunit"):
        lo, hi, is_int = span
        frac = (_hash(field, idx) % 10_000) / 10_000
        v = lo + (hi - lo) * frac
        if is_int:
            v = int(round(v))
        else:
            v = round(v, 2)
        suffix = re.sub(r"^\s*-?[\d.,]+\s*", "", str(examples[0])).strip()
        return f"{v} {suffix}".strip() if suffix else v
    if kind in ("free", "numunit") or not kind:
        return examples[_hash(field, idx) % len(examples)]
    return gen_value(field, examples, idx, kind)


def to_spec(tool):
    """The tool as the pipeline's catalogue entry."""
    props, req = {}, []
    for p in tool.get("parameters") or []:
        d = {"type": p.get("type") or "string",
             "description": p.get("description") or ""}
        if p.get("enum"):
            d["enum"] = p["enum"]
        props[p["name"]] = d
        if p.get("required"):
            req.append(p["name"])
    return {"type": "function", "function": {
        "name": tool["name"], "description": tool["description"],
        "parameters": {"type": "object", "properties": props,
                       "required": req},
        "returns": {"type": "object", "properties": {
            r["name"]: {"type": r.get("type") or "string",
                        "description": r.get("description") or ""}
            for r in tool["returns"]}}}}


# ── controls ───────────────────────────────────────────────────────────────

_OK_TOOL = {
    "name": "get_coffee_status", "description": "Henter status for kaffemaskinen.",
    "user_goal": "Er der kaffe tilbage?",
    "parameters": [{"name": "machine_id", "type": "string",
                    "description": "Maskinens id", "required": True}],
    "returns": [
        {"name": "cups_left", "type": "integer",
         "description": "Antal kopper tilbage", "examples": ["8", "3"]},
        {"name": "floor", "type": "integer",
         "description": "Etagen maskinen står på", "examples": ["4", "2"]}],
    "answer_field": "cups_left", "competitor_field": "floor"}


_MENU_TOOL = {
    "name": "get_school_menu", "description": "Henter ugens skolemenu.",
    "user_goal": "Hvad er menuen?",
    "parameters": [{"name": "week_number", "type": "integer",
                    "description": "Ugenummer", "required": True}],
    "returns": [
        {"name": "planned_menu", "type": "string",
         "description": "Ugens menu",
         "examples": ["Mandag: Mexicansk gryderet, Tirsdag: Grøntsagssuppe",
                      "Mandag: Frikadeller, Tirsdag: Fiskefilet"]},
        {"name": "alternative_menu", "type": "string",
         "description": "Alternativ menu",
         "examples": ["Mandag: Vegetarlasagne", "Mandag: Bønnegryde"]}],
    "answer_field": "planned_menu", "competitor_field": "alternative_menu"}
_MENU_PAY = {"planned_menu": "Mandag: Mexicansk gryderet, Tirsdag: "
                             "Grøntsagssuppe med brød, Onsdag: Kyllingespyd",
             "alternative_menu": "Mandag: Vegetarlasagne med salat"}


def _mut(**kw):
    t = json.loads(json.dumps(_OK_TOOL))
    t.update(kw)
    return t


TOOL_CONTROLS = [
    (_mut(name="GetCoffee"), "tool-name-not-snake_case"),
    (_mut(competitor_field="cups_left"), "answer-and-competitor-identical"),
    (_mut(returns=[_OK_TOOL["returns"][0],
                   {"name": "location", "type": "string",
                    "description": "Placering", "examples": ["Kbh", "Aarhus"]}],
          competitor_field="location"), "competitor-type-differs"),
    (_mut(returns=[_OK_TOOL["returns"][0]]), "returns-under-2-fields"),
    (_mut(returns=[_OK_TOOL["returns"][0],
                   {"name": "machine_id", "type": "integer",
                    "description": "Maskinens id", "examples": ["1", "2"]}],
          competitor_field="machine_id"), "return-echoes-parameter"),
    (_mut(returns=[_OK_TOOL["returns"][0],
                   {"name": "floor", "type": "integer",
                    "description": "The floor it is on", "examples": ["4", "2"]}]),
     "return-description-not-danish"),
    (_mut(returns=[_OK_TOOL["returns"][0],
                   {"name": "floor", "type": "integer",
                    "description": "Etagen", "examples": ["4", "4"]}]),
     "return-examples-not-varied"),
    (_mut(answer_field="temperature"), "answer-field-not-in-returns"),
    (_mut(returns=[{"name": "data_value", "type": "integer",
                    "description": "Værdien", "examples": ["8", "3"]},
                   {"name": "alternative_data_value", "type": "integer",
                    "description": "Anden værdi", "examples": ["4", "2"]}],
          answer_field="data_value", competitor_field="alternative_data_value"),
     "return-field-named-after-contract"),
    (_mut(returns=[{"name": "answer_field", "type": "integer",
                    "description": "Svaret", "examples": ["8", "3"]},
                   {"name": "competitor_field", "type": "integer",
                    "description": "Ikke svaret", "examples": ["4", "2"]}],
          answer_field="answer_field", competitor_field="competitor_field"),
     "return-field-named-after-contract"),
]


def _dlg(turns, payload=None, tool=None):
    tool = tool or _OK_TOOL
    msgs = []
    for t in turns:
        m = {"role": t[0], "content": t[1]}
        if len(t) > 2 and t[2]:
            m["tool_calls"] = [{"function": {"name": tool["name"],
                                             "arguments": t[2]}}]
        msgs.append(m)
        if len(t) > 2 and t[2]:
            msgs.append({"role": "tool", "content": json.dumps(
                payload if payload is not None else {"cups_left": 8, "floor": 4},
                ensure_ascii=False)})
    return {"idx": 0, "da": {"tools": [to_spec(tool)], "conversations": msgs}}


_CALL = {"machine_id": "m1"}
DIALOGUE_CONTROLS = [
    (_dlg([("user", "Er der kaffe?"), ("assistant", "", _CALL),
           ("assistant", "Maskinen står på 4. etage.")]),
     "answer-cites-competitor"),
    (_dlg([("user", "Er der kaffe?"), ("user", "Nå?"),
           ("assistant", "Der er 8 kopper tilbage.")]), "user-after-user"),
    (_dlg([("user", "Er der kaffe?"), ("assistant", "", _CALL),
           ("assistant", "")]), "final-turn-empty"),
    (_dlg([("user", "Er der kaffe?"), ("assistant", "", _CALL),
           ("assistant", "There are 8 cups left and that is the number "
                         "which you have been asking about")]),
     "turn-in-english"),
    (_dlg([("assistant", "Hej"), ("user", "Er der kaffe?"),
           ("assistant", "Der er 8 kopper.")]), "does-not-open-on-user"),
    # a mid-dialogue result the assistant never answers -- 30.8% of the first
    # smoke, and invisible to the end-of-dialogue check
    (_dlg([("user", "Er der kaffe?"), ("assistant", "", _CALL),
           ("user", "Nå, og hvad med i morgen?"),
           ("assistant", "Der er 8 kopper tilbage.")]), "result-not-answered"),
    (_dlg([("user", "Er der kaffe?"), ("assistant", "", {"machine_id": "  "}),
           ("assistant", "Der er 8 kopper tilbage.")]),
     "empty-required-argument"),
    # a value stated in the answer that appears nowhere in the payload, the
    # call or the question -- 20.2% of the previous smoke
    (_dlg([("user", "Er der kaffe?"), ("assistant", "", _CALL),
           ("assistant", "Der er 12 kopper tilbage.")]),
     "answer-invents-number"),
]
# Quoting a payload that has not arrived yet. Its 99 is present in the row --
# in the SECOND result -- so only a prefix-scoped check rejects this, which is
# the whole point of the control.
def _menu_dlg(answer):
    return {"idx": 0, "da": {"tools": [to_spec(_MENU_TOOL)], "conversations": [
        {"role": "user", "content": "Hvad er menuen i uge 42?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": _MENU_TOOL["name"],
                          "arguments": {"week_number": 42}}}]},
        {"role": "tool", "content": json.dumps(_MENU_PAY, ensure_ascii=False)},
        {"role": "assistant", "content": answer},
    ]}}


DIALOGUE_CONTROLS.append((
    {"idx": 0, "da": {"tools": [to_spec(_OK_TOOL)], "conversations": [
        {"role": "user", "content": "Er der kaffe?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": _OK_TOOL["name"],
                                      "arguments": _CALL}}]},
        {"role": "tool", "content": json.dumps({"cups_left": 8, "floor": 4})},
        {"role": "assistant", "content": "Der er 99 kopper tilbage."},
        {"role": "user", "content": "Og nu?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": _OK_TOOL["name"],
                                      "arguments": {"machine_id": "m2"}}}]},
        {"role": "tool", "content": json.dumps({"cups_left": 99, "floor": 4})},
        {"role": "assistant", "content": "Nu er der 99 kopper."},
    ]}}, "answer-invents-number"))
# The four gates promoted from the review script. Each shipped into the
# corpus before it was a gate: 4 degenerate openings, 1 role-swapped question,
# 1 JSON fragment, 2 consecutive-turn rows.
# The user asking in the schema's own words -- `cups_left` is documented as
# "Antal kopper tilbage", so a turn that IS that phrase makes the choice a
# string match.
DIALOGUE_CONTROLS.append((
    _dlg([("user", "Antal kopper tilbage"), ("assistant", "", _CALL),
          ("assistant", "Der er 8 kopper tilbage.")]),
    "question-echoes-field-description"))

DIALOGUE_CONTROLS.append((
    {"idx": 0, "da": {"tools": [to_spec(_OK_TOOL)], "conversations": [
        {"role": "user", "content": "Er der kaffe tilbage i maskinen?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": _OK_TOOL["name"],
                          "arguments": {"machine_id": "m1"}}}]},
        {"role": "tool", "content": json.dumps(
            {"status": "fejl", "cups_left": 8, "floor": 4})},
        {"role": "assistant", "content": "Der er 8 kopper tilbage."}]}},
    "answer-ignores-failed-status"))

DIALOGUE_CONTROLS += [
    (_dlg([("user", "Kan du analysere mine resultater?"),
           ("assistant", "", {"machine_id": None}),
           ("assistant", "Der er 8 kopper tilbage.")]),
     "null-required-argument"),
    (_dlg([("user", "Rønne-Køge"), ("assistant", "", _CALL),
           ("assistant", "Der er 8 kopper tilbage.")]), "degenerate-opening"),
    (_dlg([("user", "12345"), ("assistant", "", _CALL),
           ("assistant", "Der er 8 kopper tilbage.")]), "degenerate-opening"),
    (_dlg([("user", "Er der kaffe tilbage?"), ("assistant", "", _CALL),
           ("assistant", "Der er 8 kopper tilbage."),
           ("user", "Og hvilken maskine vil du gerne se?"),
           ("assistant", "Maskine m1.")]),
     "assistant-question-in-user-turn"),
    (_dlg([("user", "Er der kaffe tilbage?"), ("assistant", "", _CALL),
           ("assistant", '}, "machine_id": "m1" } }')]),
     "json-fragment-in-turn"),
    (_dlg([("user", "Er der kaffe tilbage?"), ("assistant", "", _CALL),
           ("assistant", "Der er 8 kopper tilbage."),
           ("assistant", "Der er 8 kopper tilbage.")]),
     "consecutive-assistant-turns"),
]

# A verbatim quote of a payload containing HYPHENS. The gate once used
# `-?\d+` while the allowed-set builder used `\d+`, so `Na8-10Al6Si6O24S2-4`
# yielded "-10" and "-4" as invented numbers and rejected a perfect answer --
# 19 of 140 rows. Both now share one NUM.
_CHEM = {
    "name": "analyze_pigment_sample", "description": "Analyserer en pigmentprøve.",
    "user_goal": "Hvad består pigmentet af?",
    "parameters": [{"name": "sample_id", "type": "string",
                    "description": "Prøvens id", "required": True}],
    "returns": [
        {"name": "pigment_composition", "type": "string",
         "description": "Pigmentets kemiske sammensætning",
         "examples": ["Ultramarin (Na8-10Al6Si6O24S2-4)", "Blyhvidt (2PbCO3)"]},
        {"name": "secondary_composition", "type": "string",
         "description": "Sekundær sammensætning",
         "examples": ["Oxideret ultramarin", "Zinkhvidt"]}],
    "answer_field": "pigment_composition",
    "competitor_field": "secondary_composition"}
_CHEM_PAY = {"pigment_composition": "Ultramarin (Na8-10Al6Si6O24S2-4)",
             "secondary_composition": "Oxideret ultramarin"}

DIALOGUE_CLEAN = [
    _dlg([("user", "Er der kaffe tilbage i maskinen m1?"),
          ("assistant", "", _CALL),
          ("assistant", "Der er 8 kopper kaffe tilbage.")]),
]

# An entirely different menu, with no digits in it at all -- which is why the
# number check passed it.
DIALOGUE_CONTROLS.append((
    _menu_dlg("Mandag: Pasta bolognese, Tirsdag: Linsegryde, "
              "Onsdag: Laks med grøntsager."), "answer-invents-text"))
DIALOGUE_CLEAN.append({"idx": 0, "da": {
    "tools": [to_spec(_CHEM)], "conversations": [
        {"role": "user", "content": "Hvad består pigmentprøve 12345 af?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": _CHEM["name"],
                          "arguments": {"sample_id": "12345"}}}]},
        {"role": "tool", "content": json.dumps(_CHEM_PAY, ensure_ascii=False)},
        {"role": "assistant",
         "content": "Pigmentets sammensætning er Ultramarin "
                    "(Na8-10Al6Si6O24S2-4)."}]}})
DIALOGUE_CLEAN.append(_menu_dlg(
    "Menuen er mexicansk gryderet mandag, grøntsagssuppe tirsdag og "
    "kyllingespyd onsdag."))


def check_controls():
    for t, want in TOOL_CONTROLS:
        got = gate_tool(t)
        if got is None or not got.startswith(want):
            raise SystemExit(f"tool control {want!r} -> {got!r}")
    if gate_tool(_OK_TOOL) is not None:
        raise SystemExit(f"tool gate rejects a clean tool: {gate_tool(_OK_TOOL)}")
    for row, want in DIALOGUE_CONTROLS:
        t = _MENU_TOOL if row["da"]["tools"][0]["function"]["name"] == \
            _MENU_TOOL["name"] else _OK_TOOL
        got = gate_dialogue(row, t)
        if got is None or not got.startswith(want):
            raise SystemExit(f"dialogue control {want!r} -> {got!r}")
    for row in DIALOGUE_CLEAN:
        nm = row["da"]["tools"][0]["function"]["name"]
        t = ({_MENU_TOOL["name"]: _MENU_TOOL,
              _CHEM["name"]: _CHEM}).get(nm, _OK_TOOL)
        got = gate_dialogue(row, t)
        if got is not None:
            raise SystemExit(f"dialogue gate rejects a clean row: {got}")
    # payload synthesis must actually vary across rows
    vals = {json.dumps(synth_payload(_OK_TOOL, i), sort_keys=True)
            for i in range(40)}
    if len(vals) < 8:
        raise SystemExit(f"payloads barely vary: {len(vals)} distinct in 40")
    # the answer and its competitor must never coincide -- a payload where
    # they match poses no choice, which is the whole premise
    for i in range(300):
        p = synth_payload(_OK_TOOL, i)
        if str(p[_OK_TOOL["answer_field"]]) == str(p[_OK_TOOL["competitor_field"]]):
            raise SystemExit(f"answer == competitor at idx {i}: {p}")
    # Synthesis must not invent impossible values. Both of these shipped:
    # `insured_value_dkk: -273092` and `"2023 -10-26: Modtog vaccination"`.
    _neg = {"name": "value_dkk", "type": "integer", "description": "Beløb",
            "examples": ["220463", "473148"]}
    _log = {"name": "log_line", "type": "string", "description": "Log",
            "examples": ["2023-10-26: Modtog vaccination.",
                         "2022-04-02: Kontrol."]}
    tt = json.loads(json.dumps(_OK_TOOL))
    tt["returns"] = tt["returns"] + [_neg, _log]
    for i in range(200):
        pay = synth_payload(tt, i)
        if float(str(pay["value_dkk"]).split()[0]) < 0:
            raise SystemExit(f"negative synthesised value: {pay['value_dkk']}")
        if " -" in str(pay["log_line"]):
            raise SystemExit(f"corrupted date synthesised: {pay['log_line']!r}")
    # A percentage whose examples never exceed 100 must not either.
    for nm, exs in (("cpu_utilization_percent", ["75", "42", "90"]),
                    ("backup_success_rate_percent", ["99", "95", "100"]),
                    ("pump_uptime_percentage", ["99.8", "98.5", "100.0"])):
        tt = json.loads(json.dumps(_OK_TOOL))
        tt["returns"] = tt["returns"] + [
            {"name": nm, "type": "number", "description": "Pct", "examples": exs}]
        for i in range(200):
            v = float(str(synth_payload(tt, i)[nm]).split()[0])
            if v > 100:
                raise SystemExit(f"{nm} synthesised as {v}")
    # A coordinate pair must survive synthesis intact -- it collapsed to the
    # single float 57.63 and was answered as "position 57.63 grader nord".
    _pos = {"name": "position", "type": "string", "description": "Position",
            "examples": ["56.1250,10.1250", "55.9890,12.9890"]}
    tp = json.loads(json.dumps(_OK_TOOL))
    tp["returns"] = tp["returns"] + [_pos]
    for i in range(60):
        v = str(synth_payload(tp, i)["position"])
        if "," not in v:
            raise SystemExit(f"coordinate pair collapsed to {v!r}")
    # A free-text field must not come back None. It did for every FREE field
    # until _value stopped routing them through the unbuilt value banks.
    _free = {"name": "note", "description": "Fri note", "type": "string",
             "examples": ['"ZONE-7"', '"DISTRICT-B"', '"AREA-42"']}
    t = json.loads(json.dumps(_OK_TOOL))
    t["returns"] = t["returns"] + [_free]
    got = {str(synth_payload(t, i)["note"]) for i in range(60)}
    if None in got or "None" in got:
        raise SystemExit("free-text field synthesised as None")
    if len(got) < 2:
        raise SystemExit(f"free-text field is constant: {got}")
    if any('"' in g for g in got):
        raise SystemExit(f"example quotes reached the payload: {got}")
    print(f"gates: {len(TOOL_CONTROLS)} tool + {len(DIALOGUE_CONTROLS)} "
          f"dialogue defects caught, {1 + len(DIALOGUE_CLEAN)} clean pass, "
          f"{len(vals)}/40 distinct payloads", flush=True)


# ── generation ─────────────────────────────────────────────────────────────

class FatalAPIError(RuntimeError):
    """The account cannot make calls -- retrying and continuing only wastes."""


# Non-200s worth giving up on immediately rather than retrying 3x per call.
FATAL_STATUS = {401, 402, 403}


async def _ask(session, sys_prompt, user_payload, schema, name, tries=3,
               temp=0.7, max_tokens=3000):
    """Structured call. Raises FatalAPIError on auth/quota; None on soft fail.

    This used to swallow every status and exception alike and return None, so
    an exhausted key looked exactly like a model returning unparseable JSON: a
    1000-row build reported "1000 dlg:no-output", kept going, and spent the
    remaining credit inventing tools whose dialogues could never be written.
    """
    body = {"model": MODEL, "temperature": temp, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": user_payload}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": name, "strict": True, "schema": schema}}}
    last = None
    for a in range(tries):
        try:
            async with session.post(URL, json=body) as r:
                if r.status in FATAL_STATUS:
                    raise FatalAPIError(
                        f"HTTP {r.status}: {(await r.text())[:300]}")
                if r.status != 200:
                    last = f"HTTP {r.status}"
                    await asyncio.sleep(1.5 * (a + 1))
                    continue
                d = await r.json()
                return (json.loads(d["choices"][0]["message"]["content"]),
                        d.get("usage") or {})
        except FatalAPIError:
            raise
        except Exception as e:
            last = f"{type(e).__name__}: {e}"[:120]
            await asyncio.sleep(1.5 * (a + 1))
    _ask.last_error = last
    return None, {}


_ask.last_error = None


async def invent_tool(session, scenario, tpl):
    prompt = (f"SCENARIE: {scenario['id']} -- {scenario['beskrivelse']}\n\n"
              f"VÆRKTØJETS PARAMETRE SKAL HAVE DENNE FORM:\n"
              f"{render_template(tpl)}\n")
    return await _ask(session, TOOL_SYS, prompt, TOOL_SCHEMA, "tool")


HINTS = {
    "does-not-end-on-assistant":
        "SIDSTE tur SKAL være assistenten, der svarer brugeren i prosa. "
        "Slut ikke på et værktøjskald.",
    "final-turn-empty":
        "Assistentens sidste tur skal indeholde et rigtigt svar, ikke tom tekst.",
    "answer-cites-competitor":
        "Det sidste svar må IKKE nævne competitor_field -- hverken feltnavnet "
        "eller dets værdi. Brug kun answer_field.",
    "turn-in-english": "Skriv ALLE ture på dansk.",
    "call-not-in-catalogue":
        "Kald kun det værktøj, du har fået -- ikke et andet navn.",
    "user-after-user": "Bruger og assistent skal skiftes til at tale.",
    "result-not-answered":
        "HVERT værktøjsresultat skal efterfølges af et assistent-svar, der "
        "bruger netop det resultat. Lad ikke brugeren tale igen først.",
    "no-call-but-plan-requires-one":
        "Samtalen SKAL indeholde mindst ét rigtigt værktøjskald.",
    "empty-required-argument":
        "Udfyld aldrig en påkrævet parameter med tom tekst.",
    "answer-invents-number":
        "Brug KUN tal, der står i resultatet eller som brugeren selv har "
        "nævnt. Find aldrig på tal.",
    "answer-invents-text":
        "Gengiv indholdet af answer_field, som det STÅR i resultatet. Find "
        "ikke på en anden tekst.",
    "degenerate-opening":
        "Brugerens FØRSTE tur skal være et rigtigt spørgsmål i hel sætning "
        "-- ikke bare et id, et navn eller et tal.",
    "assistant-question-in-user-turn":
        "Assistentens opklarende spørgsmål skal stå i en ASSISTENT-tur, ikke "
        "i brugerens.",
    "json-fragment-in-turn":
        "En assistent-tur må kun indeholde dansk prosa -- aldrig JSON eller "
        "brudstykker af et kald.",
    "consecutive-assistant-turns":
        "Slå assistentens tekst sammen til ÉN tur; to assistent-ture i træk "
        "er ikke en samtale.",
    "null-required-argument":
        "Udfyld aldrig en påkrævet parameter med null. Mangler oplysningen, "
        "så lad assistenten spørge brugeren om den FØR kaldet.",
    "call-with-all-null-arguments":
        "Kald ikke værktøjet med tomme argumenter. Spørg brugeren om de "
        "oplysninger, kaldet kræver.",
    "question-echoes-field-description":
        "Brugerens replik må ikke gentage feltets beskrivelse. Spørg med "
        "brugerens egne, dagligdags ord.",
    "answer-ignores-failed-status":
        "Siger resultatet, at kaldet fejlede, skal svaret sige det ligeud -- "
        "ikke rapportere et tal, som om alt gik godt.",
    "payload-dates-out-of-order":
        "Vælg argumenter, hvor datoerne giver mening -- en slutdato må ikke "
        "ligge før startdatoen.",
}


async def write_skeleton(session, tool, plan, catalogue, hint=None, temp=0.7):
    prompt = json.dumps({
        "vaerktoej": {"name": tool["name"], "description": tool["description"],
                      "parameters": tool["parameters"]},
        "brugerens_maal": tool.get("user_goal", ""),
        "andre_vaerktoejer": [t["function"]["name"] for t in catalogue
                              if t["function"]["name"] != tool["name"]],
        "plan": PLAN_TEXT[plan],
        **({"ekstra_krav": hint} if hint else {}),
    }, ensure_ascii=False, indent=1)
    return await _ask(session, SKELETON_SYS, prompt, SKELETON_SCHEMA,
                      "skeleton", temp=temp)


async def write_answers(session, tool, items, temp=0.5, hint=None):
    """One call for every answer slot in a dialogue, each with its OWN payload."""
    prompt = json.dumps({
        "answer_field": tool["answer_field"],
        "competitor_field": tool["competitor_field"],
        "opgaver": [{"spoergsmaal": q,
                     "argumenter": ar[0] if len(ar) == 1 else ar,
                     "resultat": p[0] if len(p) == 1 else p}
                    for q, p, ar in items],
        **({"ret_dette": hint} if hint else {}),
    }, ensure_ascii=False, indent=1)
    return await _ask(session, ANSWERS_SYS, prompt, ANSWERS_SCHEMA, "answers",
                      temp=temp)


def plan_row(idx, tool, skeleton, catalogue):
    """Skeleton -> (messages with answer slots empty, list of (question, payload)).

    Payloads are synthesised HERE, from the arguments the skeleton actually
    produced, so an answer can only ever be written against its own call.
    """
    SLOT = "\x00ANSWER"
    msgs, items, seen_args = [], [], {}
    last_user = ""
    for t in skeleton:
        calls = t.get("calls") or []
        text = (t.get("text") or "").strip()
        if t["role"] == "user":
            last_user = text or last_user
        if calls and t["role"] == "assistant":
            parsed, pays = [], []
            for c in calls:
                try:
                    a = json.loads(c.get("arguments_json") or "{}")
                except Exception:
                    a = {}
                if not isinstance(a, dict):
                    a = {}
                parsed.append({"function": {"name": c.get("name"),
                                            "arguments": a}})
                # Distinct arguments must give distinct data. With only 2-3
                # examples per field the value space is small enough that two
                # different calls land on the same payload by chance, not by
                # hash collision -- resample until it differs.
                key = json.dumps(a, sort_keys=True, ensure_ascii=False)
                # Model-generated first, local synthesis as the FALLBACK. The
                # same-type competitor is the one thing this generator must
                # rather than trusted.
                pay = None
                if key in seen_args:
                    pays.append(seen_args[key])          # same args, same data
                    continue
                pay = payload_for(tool, idx, a)
                for bump in range(1, 8):
                    prior = {json.dumps(v, sort_keys=True, ensure_ascii=False)
                             for k2, v in seen_args.items() if k2 != key}
                    if json.dumps(pay, sort_keys=True,
                                  ensure_ascii=False) not in prior:
                        break
                    pay = payload_for(tool, idx + bump * 7919, a)
                seen_args[key] = pay
                pays.append(pay)
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": parsed})
            for p in pays:
                msgs.append({"role": "tool",
                             "content": json.dumps(p, ensure_ascii=False)})
            # One answer slot per call TURN, carrying EVERY payload that turn
            # produced. A parallel turn makes two calls and the user asked for
            # two things; handing the slot only the last payload made the
            # answer cover half the question, which the judge rejected as
            # incomplete in 7 of 17 probe rows.
            items.append((last_user, pays,
                          [(c.get("function") or {}).get("arguments") or {}
                           for c in parsed]))
            msgs.append({"role": "assistant", "content": SLOT})
        elif t.get("is_answer") and msgs and msgs[-1]["content"] == SLOT:
            continue          # the slot was just created by the call turn
        elif t.get("is_answer"):
            # `is_answer` on a turn with no call before it: a refusal, or a
            # closing pleasantry the model happened to mark. Dropping these
            # ended 14 rows on a user turn -- every `refuse` plan, and any
            # multi_turn whose sign-off carried the flag.
            msgs.append({"role": t["role"], "content": text})
        else:
            msgs.append({"role": t["role"], "content": text})
    # Only a leading PLEASANTRY is dropped. Popping a leading turn that
    # carries a call would strip the call and leave its result orphaned; let
    # the gate reject that skeleton instead.
    while msgs and msgs[0]["role"] == "assistant" and not msgs[0].get(
            "tool_calls") and msgs[0]["content"] != SLOT:
        msgs.pop(0)
    # Keeping refusals and sign-offs (above) can leave two assistant turns
    # adjacent. That is a formatting artefact, not a defect worth discarding a
    # dialogue for -- merge them into the one turn they should have been.
    merged = []
    for m in msgs:
        if (merged and merged[-1]["role"] == "assistant"
                and m["role"] == "assistant"
                and not merged[-1].get("tool_calls") and not m.get("tool_calls")
                and merged[-1]["content"] != SLOT and m["content"] != SLOT
                and str(merged[-1]["content"]).strip() and str(m["content"]).strip()):
            merged[-1] = {"role": "assistant",
                          "content": merged[-1]["content"].rstrip() + " "
                                     + m["content"].lstrip()}
            continue
        merged.append(m)
    msgs = merged
    # Slots are found by SCANNING, not by arithmetic: index maths did not
    # survive the pop above and ran off the end of the list.
    slots = [i for i, m in enumerate(msgs) if m["content"] == SLOT]
    return ({"idx": idx, "da": {"tools": catalogue, "conversations": msgs}},
            items, slots)


JUDGE_SYS = """Du er kvalitetskontrol på dansk træningsdata for værktøjsbrug.

For hver opgave får du brugerens spørgsmål, de ARGUMENTER værktøjet blev
kaldt med, værktøjets NØJAGTIGE resultat, og assistentens svar. Afgør om svaret er brugbart træningsdata.

Svaret er DÅRLIGT hvis noget af dette gælder:
- det påstår noget, der ikke står i resultatet (også omskrevet)
- det gengiver `competitor_field` -- også med andre ord end feltets egne
- det svarer ikke på spørgsmålet
- det modsiger argumenterne (kaldt med `include_failed_prints: true`, men
  svaret siger, at fejlloggen ikke er med)
- det skriver et RÅT feltnavn i teksten, altså identifikatoren selv med
  understreger eller engelsk stavemåde: `brood_pattern_score`, `alto_count`,
  `guest_count`. En DANSK oversættelse er derimod netop det, der ønskes:
  "81 altstemmer" for `alto_count` og "142 gæster" for `guest_count` er
  KORREKTE svar, ikke fejl.
- resultatet selv er meningsløst: enheder der ikke passer, en slutdato før
  startdatoen, en procent over 100, en værdi der bare gentager spørgsmålet
- det indeholder JSON, engelsk prosa eller afbrudt tekst

Vær konkret i `problem`. Er svaret fint, sæt `ok: true` og lad `problem` være
tom."""

JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["domme"],
    "properties": {"domme": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["ok", "problem"],
        "properties": {"ok": {"type": "boolean"},
                       "problem": {"type": "string"}}}}}}

# The judge needs its own controls: verdicts from an unvalidated judge are
# uncalibrated, which is the same trap the gates fell into for six rounds.
JUDGE_CONTROLS = [
    ({"spoergsmaal": "Er der kaffe tilbage?",
      "resultat": {"cups_left": 8, "floor": 4},
      "svar": "Maskinen står på 4. etage."}, False),      # cites competitor
    ({"spoergsmaal": "Er der kaffe tilbage?",
      "resultat": {"cups_left": 8, "floor": 4},
      "svar": "Der er 12 kopper tilbage."}, False),       # fabricated
    ({"spoergsmaal": "Hvad er sundhedsscoren?",
      "resultat": {"health_score": 47, "pest_level": 2},
      "svar": 'Feltet "health_score" er 47.'}, False),    # raw field name
    ({"spoergsmaal": "Hvornår slutter bookingen?",
      "resultat": {"start": "2024-05-10", "end": "2024-05-02"},
      "svar": "Bookingen slutter den 2. maj 2024."}, False),  # incoherent payload
    ({"spoergsmaal": "Er der kaffe tilbage?",
      "resultat": {"cups_left": 8, "floor": 4},
      "svar": "Der er 8 kopper kaffe tilbage."}, True),
    # A Danish translation of a field name is the DESIRED answer, not a
    # defect. The rubric's "no raw field names" line was read as forbidding
    # these, rejecting correct rows.
    ({"spoergsmaal": "Hvor mange altstemmer er der i koret?",
      "resultat": {"total_members": 201, "alto_count": 81},
      "svar": "Der er 81 altstemmer i koret."}, True),
    ({"spoergsmaal": "Hvor mange gæster har bekræftet?",
      "resultat": {"guest_count": 142, "table_assignments_completed": 135},
      "svar": "Der er 142 gæster, der har bekræftet."}, True),
    ({"spoergsmaal": "Hvad er slagtevægten i stald 5?",
      "resultat": {"average_slaughter_weight_kg": 121,
                   "expected_slaughter_weight_kg": 119},
      "svar": "Den gennemsnitlige slagtevægt i stald 5 er 121 kg."}, True),
]


async def judge(session, items, tool=None):
    """Batched verdicts. `items` are dicts of spoergsmaal/resultat/svar."""
    payload = {"opgaver": items}
    if tool:
        payload["answer_field"] = tool["answer_field"]
        payload["competitor_field"] = tool["competitor_field"]
    r, u = await _ask(session, JUDGE_SYS,
                      json.dumps(payload, ensure_ascii=False, indent=1),
                      JUDGE_SCHEMA, "judge", temp=0.0, max_tokens=1200)
    if not r or len(r.get("domme") or []) != len(items):
        return None, u
    return r["domme"], u


async def check_judge(session):
    items = [{k: v for k, v in c[0].items()} for c in JUDGE_CONTROLS]
    want = [c[1] for c in JUDGE_CONTROLS]
    verdicts, _ = await judge(session, items)
    if not verdicts:
        raise SystemExit("judge returned nothing on its controls")
    got = [bool(v.get("ok")) for v in verdicts]
    miss = [(i, w, g) for i, (w, g) in enumerate(zip(want, got)) if w != g]
    if miss:
        for i, w, g in miss:
            print(f"  judge control {i}: wanted ok={w}, got ok={g} "
                  f"({verdicts[i].get('problem','')[:60]})", flush=True)
        raise SystemExit(f"judge failed {len(miss)}/{len(want)} controls")
    print(f"judge: {len(want)} controls passed "
          f"({sum(1 for w in want if not w)} planted defects caught)",
          flush=True)


def build_row(idx, tool, payloads, turns, catalogue):
    """Turns -> the pipeline's conversation shape.

    An assistant turn carrying BOTH a call and prose is SPLIT: call, then the
    results, then the prose as its own turn. Blanking the text instead -- which
    is what the first version did -- both threw the answer away and left the
    row ending on a tool result, which was 11 of 13 rejections in the first
    probe. The model is not wrong to write it that way; the renderer just
    needs one turn per act.
    """
    msgs = []
    seen = {}
    for t in turns:
        calls = t.get("calls") or []
        text = (t.get("text") or "").strip()
        if calls and t["role"] == "assistant":
            parsed = []
            for c in calls:
                try:
                    args = json.loads(c.get("arguments_json") or "{}")
                except Exception:
                    args = {}
                parsed.append({"function": {"name": c.get("name"),
                                            "arguments": args}})
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": parsed})
            for c in parsed:
                # Keyed on the ARGUMENTS, not the call index. Same arguments
                # must return the same data -- a dialogue that asks twice for
                # the same thing and gets two different answers is teaching
                # that a tool's output is arbitrary. Distinct argument sets
                # still map to distinct payloads, which is what stopped every
                # call in a row returning byte-identical data.
                k = json.dumps((c.get("function") or {}).get("arguments") or {},
                               sort_keys=True, ensure_ascii=False)
                if k not in seen:
                    seen[k] = min(len(seen), len(payloads) - 1)
                msgs.append({"role": "tool", "content": json.dumps(
                    payloads[seen[k]], ensure_ascii=False)})
            if text:
                msgs.append({"role": "assistant", "content": text})
        else:
            msgs.append({"role": t["role"], "content": text})
    # a leading assistant pleasantry is not a defect worth discarding a whole
    # dialogue over
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return {"idx": idx, "da": {"tools": catalogue, "conversations": msgs}}


async def main_async(args):
    import aiohttp
    check_controls()
    rng = random.Random(args.seed)
    scenarios = [json.loads(l) for l in args.scenarios.open() if l.strip()]
    rng.shuffle(scenarios)
    print(f"{len(scenarios):,} scenarios", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    tok = Counter()
    headers = {"Authorization": f"Bearer {_key()}"}
    sem = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession(headers=headers) as session:
        # ── stage A: tools ────────────────────────────────────────────────
        want_tools = max(args.n // args.dialogues_per_tool, 1)
        tools = []

        async def one_tool(sc):
            async with sem:
                tpl = sample_template(rng)
                t, u = await invent_tool(session, sc, tpl)
                tok["in"] += u.get("prompt_tokens", 0)
                tok["out"] += u.get("completion_tokens", 0)
                if not t:
                    stats["tool:no-proposal"] += 1
                    return None
                why = gate_tool(t)
                if why:
                    stats[f"tool:{why.split(':')[0]}"] += 1
                    return None
                t["_hash"] = template_hash(tpl)
                t["_scenario"] = sc["id"]
                stats["tool:ok"] += 1
                return t

        seen_names = set()

        pool = scenarios[:want_tools * 3]
        done = 0
        for chunk_start in range(0, len(pool), args.concurrency * 2):
            if len(tools) >= want_tools:
                break
            chunk = pool[chunk_start:chunk_start + args.concurrency * 2]
            got = await asyncio.gather(*[one_tool(s) for s in chunk])
            for t in got:
                # Two tools sharing a NAME but not a schema collide in every
                # by-name lookup downstream -- `marina_crane_booking` was
                # invented twice with different parameters.
                if not t:
                    continue
                if t["name"] in seen_names:
                    stats["tool:duplicate-name"] += 1
                    continue
                seen_names.add(t["name"])
                tools.append(t)
            done += len(chunk)
            print(f"  tools {len(tools)}/{want_tools}  (tried {done})",
                  flush=True)
        tools = tools[:want_tools]
        if not tools:
            raise SystemExit("no tools survived the gate")
        (args.out / "tools.jsonl").write_text("\n".join(
            json.dumps(t, ensure_ascii=False) for t in tools) + "\n")

        # ── stages B-D: dialogues ─────────────────────────────────────────
        specs = [to_spec(t) for t in tools]
        tools_by_name = {t["name"]: t for t in tools}
        bad_skeletons = []
        rows = []

        async def one_dialogue(idx, tool):
            async with sem:
                plan = _weighted(rng, PLAN_WEIGHTS)
                others = [s for s in specs
                          if s["function"]["name"] != tool["name"]]
                rng.shuffle(others)
                cat = [to_spec(tool)] + others[:rng.randint(1, 5)]
                rng.shuffle(cat)
                hint, why = None, None
                # One hinted retry on the SKELETON. A gate verdict is an
                # instruction the next attempt can act on, so a second call is
                # cheaper than discarding a tool we already paid to invent.
                for attempt in range(2):
                    r, u = await write_skeleton(
                        session, tool, plan, cat, hint=hint,
                        temp=0.7 if not attempt else 1.0)
                    tok["in"] += u.get("prompt_tokens", 0)
                    tok["out"] += u.get("completion_tokens", 0)
                    if not r or not r.get("turns"):
                        why = "no-skeleton"
                        continue
                    row, items, slots = plan_row(idx, tool, r["turns"], cat)

                    row["_plan"] = plan
                    if not items and plan != "refuse":
                        why = "no-call-but-plan-requires-one"
                    else:
                        # answers written against their OWN payload, one call
                        a, u2 = await write_answers(session, tool, items) \
                            if items else ({"svar": []}, {})
                        tok["in"] += u2.get("prompt_tokens", 0)
                        tok["out"] += u2.get("completion_tokens", 0)
                        svar = (a or {}).get("svar") or []
                        if len(svar) != len(items) or len(slots) != len(items):
                            why = "answers-count-mismatch"
                        else:
                            conv = row["da"]["conversations"]
                            for s, text in zip(slots, svar):
                                conv[s]["content"] = (text or "").strip()
                            why = gate_dialogue(row, tool, plan)
                    if not why:
                        row["_judge_items"] = [
                            {"spoergsmaal": q,
                             "argumenter": ar[0] if len(ar) == 1 else ar,
                             "resultat": p[0] if len(p) == 1 else p,
                             "svar": row["da"]["conversations"][s]["content"]}
                            for (q, p, ar), s in zip(items, slots)]
                        stats["dlg:ok"] += 1
                        stats[f"plan:{plan}"] += 1
                        if attempt:
                            stats["dlg:ok-on-retry"] += 1
                        return row
                    hint = HINTS.get(why.split(":")[0])
                    if not hint:
                        break
                stats[f"dlg:{str(why).split(':')[0]}"] += 1
                bad_skeletons.append({
                    "why": str(why), "plan": plan, "tool": tool["name"],
                    "skeleton": (r or {}).get("turns"),
                    "answers": locals().get("svar"),
                    "items": [{"q": x[0], "p": x[1],
                               "args": x[2] if len(x) > 2 else None}
                              for x in (locals().get("items") or [])]})
                return None

        jobs = [(i, tools[i % len(tools)]) for i in range(args.n)]
        for s in range(0, len(jobs), args.concurrency * 2):
            batch = jobs[s:s + args.concurrency * 2]
            got = await asyncio.gather(*[one_dialogue(i, t) for i, t in batch])
            rows.extend([r for r in got if r])
            print(f"  dialogues {len(rows)}/{args.n}", flush=True)

        # ── stage E: LLM judge, only on rows the gates already passed ─────
        if not args.no_judge:
            await check_judge(session)
            todo = [(r, it) for r in rows for it in [r.pop("_judge_items", [])]
                    if it]
            flat = [(ri, it) for ri, (r, its) in enumerate(todo) for it in its]
            keep = [True] * len(todo)
            reasons = []
            repairs = []

            async def one_batch(chunk):
                async with sem:
                    verdicts, u = await judge(session, [it for _, it in chunk])
                    tok["jin"] += u.get("prompt_tokens", 0)
                    tok["jout"] += u.get("completion_tokens", 0)
                    if not verdicts:
                        return
                    for (ri, it), v in zip(chunk, verdicts):
                        if not v.get("ok"):
                            keep[ri] = False
                            reasons.append(v.get("problem", "")[:90])
                            repairs.append((ri, it, v.get("problem", "")))

            B = args.judge_batch
            chunks = [flat[i:i + B] for i in range(0, len(flat), B)]
            for s in range(0, len(chunks), args.concurrency):
                await asyncio.gather(*[one_batch(c)
                                       for c in chunks[s:s + args.concurrency]])
                print(f"  judged {min((s + args.concurrency) * B, len(flat))}"
                      f"/{len(flat)} turns", flush=True)
            # REPAIR, not just reject. A verdict is an instruction -- the same
            # reason the gates carry hints. Rewriting the answer against its
            # own payload plus the judge's complaint recovers most of the
            # rejected rows for one small call each, instead of discarding a
            # dialogue whose tool and skeleton are already paid for.
            if repairs:
                print(f"  repairing {len(repairs)} judged turns", flush=True)

                async def one_repair(ri, it, problem):
                    async with sem:
                        row = todo[ri][0]
                        tool = tools_by_name.get(_called_name(row))
                        if not tool:
                            return
                        _p = (it["resultat"] if isinstance(it["resultat"], list)
                              else [it["resultat"]])
                        _a = it.get("argumenter") or {}
                        _a = _a if isinstance(_a, list) else [_a]
                        a, u = await write_answers(
                            session, tool,
                            [(it["spoergsmaal"], _p, _a)],
                            temp=0.6, hint=problem)
                        tok["in"] += u.get("prompt_tokens", 0)
                        tok["out"] += u.get("completion_tokens", 0)
                        svar = (a or {}).get("svar") or []
                        if not svar:
                            return
                        conv = row["da"]["conversations"]
                        for m in conv:
                            if m["role"] == "assistant" and \
                                    m.get("content") == it["svar"]:
                                m["content"] = svar[0].strip()
                                it["svar"] = svar[0].strip()
                                break
                        v, u2 = await judge(session, [it])
                        tok["jin"] += u2.get("prompt_tokens", 0)
                        tok["jout"] += u2.get("completion_tokens", 0)
                        if v and v[0].get("ok") and gate_dialogue(
                                row, tool, row.get("_plan")) is None:
                            keep[ri] = True
                            stats["judge:repaired"] += 1

                for s2 in range(0, len(repairs), args.concurrency):
                    await asyncio.gather(*[one_repair(*r)
                                           for r in repairs[s2:s2 + args.concurrency]])
            rejected = [r for (r, _), k in zip(todo, keep) if not k]
            rows = [r for (r, _), k in zip(todo, keep) if k]
            stats["judge:rejected"] = len(rejected)
            stats["judge:kept"] = len(rows)
            (args.out / "judge_rejects.jsonl").write_text("\n".join(
                json.dumps(x, ensure_ascii=False) for x in rejected) + "\n")
            print(f"\njudge: {len(rows)} kept, {len(rejected)} rejected "
                  f"of {len(todo)}", flush=True)
            for x in Counter(reasons).most_common(8):
                print(f"    {x[1]:>3}  {x[0]}", flush=True)

    (args.out / "gen_rejects.jsonl").write_text("\n".join(
        json.dumps(x, ensure_ascii=False) for x in bad_skeletons) + "\n")
    for r in rows:
        r.pop("_judge_items", None)
    out = args.out / "translated.jsonl"
    with out.open("w") as f:
        for i, r in enumerate(rows):
            r["idx"] = i
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    for k, v in sorted(stats.items()):
        print(f"   {v:>5}  {k}")
    cost = tok["in"] / 1e6 * 0.10 + tok["out"] / 1e6 * 0.40
    jcost = tok["jin"] / 1e6 * 0.10 + tok["jout"] / 1e6 * 0.40
    print(f"\ngeneration: in={tok['in']:,} out={tok['out']:,}  ~${cost:.4f}")
    print(f"judge     : in={tok['jin']:,} out={tok['jout']:,}  ~${jcost:.4f}"
          f"  ({100*jcost/max(cost,1e-9):.1f}% of generation)")
    print(f"template shapes: {len({t['_hash'] for t in tools})} distinct "
          f"over {len(tools)} tools")
    print(f"-> {out}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("scratch/gen_da"))
    ap.add_argument("--scenarios", type=Path,
                    default=Path("data/tool_calls/scenarios_expanded.jsonl"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dialogues-per-tool", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the LLM verification pass")
    ap.add_argument("--judge-batch", type=int, default=10,
                    help="turns per judge call; the rubric is sent once")
    args = ap.parse_args()
    if args.controls_only:
        check_controls()
        return
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
