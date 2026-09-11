"""Procedural structure, LLM only for the words.

Prototype of the split measured across 22 runs of gen_tool_dialogues_da.py:
72% of every dialogue rejection was STRUCTURAL -- turn order, whether a call
happened, whether a result got answered, how many answers came back. Those are
decisions code can make correctly every time. Only 26% were prose defects,
which need a model either way.

    code owns:  turn order, which turns call, the arguments, the payloads,
                which field answers, the catalogue
    model owns: the tool's names and Danish descriptions, the user's
                utterances, the assistant's answers

One dressing call per dialogue instead of skeleton+answers, and the structural
defect classes become unreachable rather than gated.

    uv run --no-project --with aiohttp --with langdetect \
        python scripts/gen_tool_dialogues_proc.py --out scratch/proc --n 200

langdetect is optional: without it the whole-turn language check is skipped
and the run says so.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tool_dialogues_da import (  # noqa: E402
    FatalAPIError, _answer_nums, _ask, _hash, _key, _mentions, _shares_content,
    _traces_to,
    _unquote, _numeric_span, _weighted, check_judge, gate_dialogue, gate_tool,
    judge, payload_for, sample_template, render_template, strip_catalogue,
    template_hash, to_spec, LIST_ENUM, NUM, PLAN_WEIGHTS)

# Parameters need EXAMPLES so arguments can be sampled procedurally. The
# existing schema only describes them, which is enough for a model writing a
# call but not for code doing it.
TOOL_SYS = """Du opfinder ét værktøj (et API-endepunkt) til et dansk scenarie.

NAVNE PÅ ENGELSK, BESKRIVELSER PÅ DANSK. Værktøjsnavn, parameternavne og
returfeltnavne er engelske og i snake_case. Alle beskrivelser er på dansk.

Værktøjet skal have en `returns`-kontrakt, og kontrakten SKAL stille et VALG:
- `confusable_fields` er 2-3 returfelter med SAMME TYPE og samme
  størrelsesorden, som er lette at forveksle. Fx {"cups_left": 8,
  "floor": 4} -- kun beskrivelsen skiller dem.
- HVERT af dem skal kunne være svaret på et rimeligt spørgsmål. Brugeren kan
  spørge om et hvilket som helst af dem.
- `answer_field` og `competitor_field` skal begge være med i
  `confusable_fields`.
- Et returfelt må ikke gentage en parameter.
- Returfelter skal have RIGTIGE domænenavne. Aldrig `data_value`,
  `requested_data`, `alternative_data`, `result` eller `value` -- sig hvad
  feltet indeholder.
- Drift-felter (status, request_id) tæller ikke som konkurrent.

Har værktøjet en parameter, der VÆLGER hvilken oplysning der returneres --
fx `data_category` med valgmulighederne "stemmeoptælling" og
"beredskabsplan" -- så udfyld `selectors`: for hver valgmulighed skal du
angive, hvilket returfelt den peger på. Værdierne er på dansk og felterne på
engelsk, så sammenhængen kan ikke gættes ud fra navnene -- den skal skrives.
Har værktøjet ingen sådan parameter, så lad `selectors` være tom.

`required` er KUN sandt for en parameter, kaldet er meningsløst uden -- det,
der udpeger HVEM eller HVAD der spørges om. Filtre, grænser, tidsstempler,
perioder, tærskler og antalsbegrænsninger er ALTID `required: false`: brugeren
nævner dem sjældent, og et kald, der sender dem alligevel, opfinder en værdi,
ingen har bedt om.

Giv 2-3 realistiske og FORSKELLIGE eksempelværdier for HVER parameter og HVERT
returfelt. Eksemplerne bruges direkte som data, så de skal være rigtige
værdier -- ikke beskrivelser af værdier."""

TOOL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["name", "description", "parameters", "returns",
                 "answer_field", "competitor_field", "confusable_fields",
                 "selectors", "user_goal"],
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "user_goal": {"type": "string"},
        "parameters": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "type", "description", "required", "examples"],
            "properties": {
                "name": {"type": "string"}, "type": {"type": "string"},
                "description": {"type": "string"},
                "required": {"type": "boolean"},
                "examples": {"type": "array", "items": {"type": "string"}},
                "enum": {"type": "array", "items": {"type": "string"}}}}},
        "returns": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "type", "description", "examples"],
            "properties": {
                "name": {"type": "string"}, "type": {"type": "string"},
                "description": {"type": "string"},
                "examples": {"type": "array", "items": {"type": "string"}}}}},
        "answer_field": {"type": "string"},
        "competitor_field": {"type": "string"},
        "confusable_fields": {"type": "array",
                              "items": {"type": "string"}},
        "selectors": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["parameter", "option", "selects_field"],
            "properties": {"parameter": {"type": "string"},
                           "option": {"type": "string"},
                           "selects_field": {"type": "string"}}}}}}

DRESS_SYS = """Du skriver replikkerne i en dansk samtale, hvis FORLØB allerede
ligger fast.

Du får en liste af BEATS. Hvert beat har et nummer og en rolle. Du skriver KUN
tekst til de beats, der beder om det -- du må ikke tilføje, fjerne eller
ombytte beats.

- `bruger_beder_om`: skriv brugerens replik. Den skal naturligt føre til
  præcis de argumenter, der står -- nævn værdierne, men som almindeligt sprog,
  ikke som felter. `bruger_beder_om` er et FELTNAVN på engelsk: det siger,
  hvad brugeren vil vide, men ikke hvordan de siger det. Skriv replikken i
  brugerens egne, almindelige ord -- aldrig feltnavnet, og aldrig
  feltbeskrivelsen fra værktøjet. Spørgsmålet skal handle om netop det,
  `svar_felt` indeholder -- ikke om noget andet i resultatet.
  Det skal kunne besvares med præcis den værdi: spørg ikke "hvornår", hvis
  feltet er et antal eller en prioritet, og ikke "hvor stor en andel", hvis
  feltet tæller noget.
  Står der `udelad`, mangler brugeren netop DEN oplysning: nævn den ikke, og
  antyd den ikke -- heller ikke selvom den står i et senere beat. Replikken
  skal give mening uden.
- `assistent_spoerger_om`: skriv assistentens spørgsmål efter netop den
  manglende oplysning. Spørg kun om DEN -- ikke om noget, brugeren allerede
  har sagt.
- `bruger_oplyser`: brugeren svarer på det spørgsmål. Skriv KUN den værdi,
  der står i `argumenter`, som et kort svar: "Det er 262." eller "2023".
  Gentag ALDRIG spørgsmålet, og stil ikke et nyt -- den, der svarer, er ikke
  den, der spurgte.
- `assistent_svarer`: skriv assistentens svar. Brug feltet `svar_felt` fra
  `resultat`. Brug kun tal og tekst fra `resultat`, fra argumenterne eller
  fra det brugeren selv sagde. Skriv aldrig feltnavne i teksten; sig hvad
  feltet betyder. Altid en hel sætning -- aldrig kun værdien.
  Kald værdien det, den er: et antal er et antal, ikke en procent eller en
  rate. Har feltet ingen enhed, så find ikke på en.
  Svar på DET, DER BLEV SPURGT OM, og ikke mere: tilføj ikke oplysninger,
  brugeren ikke bad om. Decimaltal skrives med KOMMA på dansk: 3,43 --
  ikke 3.43.
- `assistent_afslaar`: skriv et kort, ligefremt afslag.

Siger `resultat`, at kaldet fejlede (fx `status: "fejl"`), skal svaret sige
det ligeud -- rapporter ALDRIG et tal fra et mislykket kald, som om alt gik
godt.

To replikker må aldrig være ens. Brugeren og assistenten er to forskellige
mennesker: assistenten gentager ikke brugerens sætning som sit svar.

Alt er på DANSK. Kun værktøjs-, parameter- og feltnavne er engelske."""

DRESS_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["ture"],
    "properties": {"ture": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["nr", "tekst"],
        "properties": {"nr": {"type": "integer"},
                       "tekst": {"type": "string"}}}}}}


# ── procedural: arguments ──────────────────────────────────────────────────

def sample_arg(param, idx, salt=0):
    """A value for this parameter, from its own declared examples."""
    ex = [_unquote(e) for e in (param.get("examples") or []) if str(e).strip()]
    enum = param.get("enum") or []
    h = _hash(param.get("name"), idx, salt)
    if enum:
        return enum[h % len(enum)]
    t = (param.get("type") or "string").lower()
    if t == "boolean":
        return bool(h % 2)
    if not ex:
        return None
    span = _numeric_span(ex)
    if span and t in ("integer", "number"):
        lo, hi, is_int = span
        v = lo + (hi - lo) * ((h % 10_000) / 10_000)
        return int(round(v)) if t == "integer" or is_int else round(v, 2)
    raw = ex[h % len(ex)]
    if t in ("integer", "number"):
        try:
            return int(raw) if t == "integer" else float(raw)
        except ValueError:
            return raw
    return raw


def _selector_for(param, answer_field, tool=None):
    """A choice this parameter offers that SELECTS the answer field.

    Some parameters pick which aspect the tool reports. Sampling them
    independently of the chosen answer field produced a call for
    `measurement_type: "pump_efficiency"` answering a question about
    `bacteria_count_per_ml` -- the arguments contradicting the question.
    """
    if tool:
        dec = (tool.get("_selectors") or {}).get(
            f"{param.get('name')}\u0000{answer_field}")
        if dec is not None:
            return dec
    opts = [str(o) for o in (param.get("enum") or [])] or \
           [_unquote(e) for e in (param.get("examples") or [])]
    if len(opts) < 2:
        return None
    af_toks = [w for w in re.split(r"_+", answer_field) if len(w) >= 4]
    best = None
    for o in opts:
        o_toks = [w for w in re.split(r"[_\s-]+", str(o).lower()) if len(w) >= 4]
        for a in af_toks:
            for b in o_toks:
                # prefix match, so bacteria~bacteriological counts
                n = min(len(a), len(b), 6)
                if a == b or (n >= 5 and a[:n] == b[:n]):
                    best = o
                    break
            if best:
                break
        if best:
            break
    return best


def governs_field(param, tool):
    """Does this parameter CHOOSE which return field the tool reports?

    Three sources, strongest first. An explicit NO (`_not_selectors`) beats the
    regex, because the regex is the thing that gets it wrong: `election_type`
    is condemned by its `_type` suffix while its options name elections rather
    than fields, and every row on that tool then dies as
    `dlg:no-args-for-answer-field`.
    """
    name = param.get("name")
    if name in (tool.get("_not_selectors") or ()):
        return False
    declared = {k.split(chr(0))[0] for k in (tool.get("_selectors") or {})}
    if declared:
        return name in declared
    opts = (param.get("enum") or []) or (param.get("examples") or [])
    return bool(SELECTOR_NAME.search(str(name or "")) and len(opts) > 1)


def selects_answer(param, value, tool):
    """Is this argument CODE's choice of WHICH field to report?

    `governs_field` alone was too generous: it exempts a parameter the tool
    lists as a selector for every answer field at once. `cheese_type` and
    `substrate_type` are listed there, and their values name cheeses and
    substrates rather than return fields, so `cheese_type: "Rococo"` rode
    into a call whose user had supplied only a batch number -- and the answer
    then called it "Rococo-osten". The exemption needs the value ACTUALLY
    SENT to be the one that selects THIS row's answer field.
    """
    af = (tool or {}).get("answer_field")
    if not af or not governs_field(param, tool):
        return False
    sel = _selector_for(param, af, tool)
    return sel is not None and str(sel).lower() == str(value).lower()


def sample_args(tool, idx, salt=0, optional_p=0.5, answer_field=None):
    """Required parameters always; optional ones sometimes. Never null.

    Returns None when the tool cannot be asked this row's question at all --
    see the selector case below.
    """
    out = {}
    af = answer_field or tool.get("answer_field")
    for p in tool.get("parameters") or []:
        # `sel` FIRST, because a selector whose value names the reported field
        # is not an optional extra -- it is the call's record of which field
        # was asked for, and the only place in a row where the choice of field
        # is written down rather than enforced by redaction.
        sel = _selector_for(p, af, tool) if af else None
        # THE COIN FLIP MUST NOT REACH IT. This test used to run first, so an
        # optional selector naming the answer field was dropped half the time.
        # Measured on the v2 build: of 6,363 rows whose tool has such a
        # selector, 997 calls omitted it -- and because `synth_payload` emits
        # every declared return field regardless of what was selected, the
        # answer field was present anyway in 100% of them. The corpus shipped
        # 997 worked examples of "skip the selector, answer correctly", and the
        # model learned it: 13.6% of unseen calls omit a selector gold sends,
        # 7.1% add one gold omits -- a marginal rate rather than a rule.
        if sel is None and not p.get("required") \
                and _hash("opt", p.get("name"), idx, salt) % 100 \
                >= optional_p * 100:
            continue
        # A parameter that PICKS the reported field, none of whose options
        # picks ours, cannot be filled by sampling: `health_metric:
        # "population_size"` was sent for a question about `honey_yield_kg`,
        # so the call itself asked for the wrong thing and the answer read it
        # off a payload the call had not requested. Optional -> leave it out.
        # Required -> the tool cannot express this question; drop the row and
        # let a different answer field be tried.
        if sel is None and af and governs_field(p, tool):
            if p.get("required"):
                return None
            continue
        v = sel if sel is not None else sample_arg(p, idx, salt)
        if v is not None and v != "":
            out[p["name"]] = v
    return order_ranges(out)


# A parameter that selects HOW to look something up -- by MMSI, by name, by
# call sign -- names the same subject a different way. Varying it must not
# change the data: the corpus had one vessel returning two different wrecks
# because `query_type` was treated as a second subject.
# HOW to look a thing up, not WHAT to look up. `^search_` used to be enough,
# which made `search_term` a lookup key and exempted it from having to be
# spoken: "Hvor mange dele er der til produktion 304?" went out with
# `search_term: "SK87B"` attached.
LOOKUP_PARAM = re.compile(
    r"^(by_|lookup_)|^search_(by|type|mode|field|method|kind)$|"
    r"(query|search|lookup|identifier|id|match|ref)_?(type|by|method|mode|field|kind)$|"
    r"^(method|mode|match_type|id_type|key_type)$", re.I)


# Fallback for a tool that declared no selectors at all.
SELECTOR_NAME = re.compile(r"(category|kategori|_type$|^type$|kind|aspect|"
                           r"metric|report)", re.I)


def is_lookup_param(name):
    return bool(LOOKUP_PARAM.search(str(name or "")))


# HOW a request is delivered or authorised, never WHAT it asks about. A
# closed technical set, like TYPE_WORDS -- not a domain vocabulary the next
# catalogue will word differently.
CORRELATION_PARAM = re.compile(
    r"^(request|correlation|trace|session|idempotency)_(id|key|token)$", re.I)
META_PARAM = re.compile(
    r"(^|_)format(_|$)"
    r"|(^|_)(api_?key|access_key|secret|token|auth|credential|password)(_|$)"
    r"|(^|_)(locale|encoding)(_|$)", re.I)


def is_meta_param(name):
    """A parameter that must not change the answer.

    Asking for the same data as PDF instead of JSON returned a different
    growth report; csv instead of html turned 1 power outage into 0, and 4
    signal failures into 10; two API keys gave two colony strengths. A format
    is not a second subject, and neither is a request id or a credential.
    """
    n = str(name or "")
    if CORRELATION_PARAM.search(n):
        return True
    # `format_check_id` NAMES a subject; `report_format` names a rendering.
    return bool(META_PARAM.search(n)) and not IDENTIFYING.search(n)


def is_filter_param(name):
    """A bound or a window: it narrows the query, it is not the subject."""
    n = str(name or "")
    return bool(BOUND_LO.search(n) or BOUND_HI.search(n)
                or BOUND_ANY.search(n)
                or RANGE_LO.search(n) or RANGE_HI.search(n))


# Fields that AGGREGATE over the thing they name. `number` is deliberately
# absent: `master_version_number` and `launch_sequence_number` are numbers
# that count nothing.
AGGREGATE = re.compile(r"(^|_)(count|antal|sum|average|avg|mean|total)(_|$)"
                       r"|^number_of_", re.I)


def _aggregates_over(field, param):
    """Does `field` count the things `param` identifies?

    Two ways to say it: a counting word beside the same subject
    (`student_count` next to `student_id`), or the subject's PLURAL
    (`absent_students`). The plural is what separates a count from a property:
    `sample_weight_kg` beside `sample_id` is the weight OF that sample and
    must keep following it.
    """
    if IDENTIFYING.search(field):
        return False
    if AGGREGATE.search(field) and _same_subject(param, field):
        return True
    ftoks = set(_subject_tokens(field))
    return any(t + "s" in ftoks or t + "er" in ftoks
               for t in _subject_tokens(param))


def field_key(field, args):
    """The arguments a given return field actually depends on.

    Two calls for one dive, differing only in `sample_id`, came back with two
    different `dive_depth_meters` -- a dive has one depth. The field shares a
    subject with `dive_id` and none with `sample_id`, so it is drawn from the
    dive alone, while `sample_weight_kg` still follows the sample. A field
    that matches no argument falls back to the subject minus filters.

    Filters are excluded even when they share a word: `species_count_threshold`
    and `samples_collected_count` both say "count", and letting that key the
    record moved the sample count from 218 to 56 when the threshold changed.
    A bound still binds -- `respect_constraints` runs afterwards.

    A COUNT of a thing cannot depend on WHICH one you name. `student_count`
    shares its subject with `student_id`, so the class size was drawn per
    pupil: one row reported 9C as having 20 students when asked about s12345
    and 21 when asked about s67890, and a parallel row answered "der er 29
    elever, som hedder s67890, og 28 elever, som hedder s11223". The member's
    identifier is dropped from the key, here and in the fallback, so the count
    follows whatever else the call names -- the class.
    """
    args = args or {}
    members = {k for k in args
               if IDENTIFYING.search(k) and _aggregates_over(field, k)}
    rel = {k: v for k, v in args.items()
           if _same_subject(k, field) and not is_filter_param(k)
           and not is_meta_param(k) and k not in members}
    return rel or {k: v for k, v in entity_args(args).items()
                   if k not in members}


def entity_args(args):
    """The arguments that say WHICH thing, without the ones that filter it.

    The payload is keyed on the arguments, so a filter in the key made the
    entity itself change: recipe 188 scaled 5->3 returned flour, sugar and
    eggs, and the same recipe scaled 6->3 returned rice and water. Keying
    the record on the subject alone keeps a thing's data its own; only the
    answer field is redrawn per query.
    """
    return {k: v for k, v in (args or {}).items()
            if not is_lookup_param(k) and not is_filter_param(k)
            and not is_meta_param(k)}


def vary_one(tool, args, idx, salt, numeric_ok=True):
    """The same request, for a different subject.

    A parallel turn is "do X for A AND for B" -- one parameter moves and the
    rest hold. Resampling every parameter independently, which is what the
    first version did, invents a second call the user never made: a rescue
    request for 3 crew with thermal cameras came out beside one for 6 crew
    with diving gear, and no user turn could motivate both.

    Returns (new_args, varied_parameter) or (None, None) when nothing sensible
    can vary -- a lone boolean flag is a display toggle, not a second subject.
    """
    af = tool.get("answer_field")
    # A SELECTOR chooses WHICH field is meaningful, so varying it leaves the
    # row's single answer field unable to express the question: asked for the
    # material log AND the contingency plan, the answer gave both fields from
    # both payloads -- the same pair twice. Vary a SUBJECT (the year, the
    # vessel) so both calls mean the same thing about different things.
    # Excluded for ANY declared field, not just this row's.
    declared_sel = {k.split(chr(0))[0] for k in (tool.get("_selectors") or {})}
    cands = [p for p in (tool.get("parameters") or [])
             if p["name"] in args
             and not is_lookup_param(p["name"])
             # A BOUND is not a second subject. "for the 86 busiest junctions"
             # and "for the 3 busiest" is one question asked twice, and the
             # payloads answered it with two different traffic volumes --
             # teaching that a page size changes the data. Same for
             # `max_depth: 2` vs `11` and `limit: 86` vs `3`.
             and not is_filter_param(p["name"])
             # A format, a credential, a request id: varying one of these
             # asks the SAME question twice and must give the same answer.
             and not is_meta_param(p["name"])
             and p["name"] not in declared_sel
             and not (not declared_sel and SELECTOR_NAME.search(p["name"]))
             and _selector_for(p, af, tool) is None
             and (p.get("type") or "string").lower() != "boolean"
             and len({str(_unquote(e)) for e in (p.get("examples") or [])
                      } | set(p.get("enum") or [])) > 1]
    if not numeric_ok:
        cands = [p for p in cands
                 if (p.get("type") or "string").lower() not in ("integer", "number")
                 or p.get("enum")]
    if not cands:
        return None, None
    # Prefer a SUBJECT, not a filter. Varying `max_container_capacity` from
    # 297 to 228 produced a second route request the user never made, while
    # varying `artwork_title` from "Stilleben med frugter" to "Skriget" reads
    # exactly like the two-part question it is meant to be. 20 of 73 varied
    # parameters were numeric filters.
    def rank(p):
        t = (p.get("type") or "string").lower()
        name = p.get("name", "")
        subjectish = any(k in name for k in
                         ("name", "id", "title", "type", "code", "number",
                          "navn", "sample", "area", "site", "location"))
        return (0 if (t == "string" and subjectish) else
                1 if t == "string" else
                2 if p.get("enum") else 3)
    ranked = sorted([p for p in cands if p.get("required")] or cands, key=rank)
    best = [p for p in ranked if rank(p) == rank(ranked[0])]
    pick = best[_hash("vary", idx, salt) % len(best)]
    out = dict(args)
    for bump in range(1, 6):
        v = sample_arg(pick, idx, salt * 31 + bump)
        if v is not None and str(v) != str(args.get(pick["name"])):
            out[pick["name"]] = v
            return order_ranges(drop_stale(out, pick["name"],
                                           args.get(pick["name"]), v)), pick
    return None, None


def drop_stale(args, varied, old, new):
    """Arguments that still describe the subject we just moved away from.

    Varying `dive_id` from DIVE-2023-10-26-005 to DIVE-2023-10-27-001 left
    `dive_metadata` saying `"date": "2023-10-26"` -- the second call carried
    the first call's date. A held constant that quotes the old value is not
    held, it is stale.
    """
    old, new = str(old), str(new)
    marks = {old[i:i + 6] for i in range(len(old) - 5)} - \
            {new[i:i + 6] for i in range(len(new) - 5)}
    if not marks:
        return args
    return {k: v for k, v in args.items()
            if k == varied or not isinstance(v, str)
            or not any(m in v for m in marks)}


# ── procedural: payload repair ─────────────────────────────────────────────
#
# The payload is synthesised from the field's own examples and knows nothing
# about the call that asked for it. Reading 176 rows, that showed up three
# ways, all of which teach the model to read a contradiction as fact:
#
#   "hvor mange bryllupper, hvor der er MINDST 129 begivenheder" -> 20
#   `minimum_member_count: 44` -> `total_members: 62, active_members: 275`
#   two parallel calls, one for 001001 and one for 12a345b, and the payload
#   for 001001 carrying `cadastral_map_url: ".../12a345b.pdf"`

# Parameter names follow the schema's own convention -- English snake_case,
# by the tool contract -- so these read the NAME, not the domain.
BOUND_LO = re.compile(r"(^|_)(min|minimum|least|lower|floor)(_|$)")
BOUND_HI = re.compile(r"(^|_)(max|maximum|upper|limit|cap)(_|$)")
# A bound whose DIRECTION the name does not give. It filters, so it must not
# be a row's axis and must not key the record -- `species_count_threshold`
# moving 65 -> 50 changed `samples_collected_count` from 218 to 56 -- but
# nothing here says which way to clamp, so it does not clamp.
BOUND_ANY = re.compile(r"(^|_)(threshold|cutoff|filter)(_|$)")
TOTAL_TOKEN = re.compile(r"(^|_)(total|overall)(_|$)")


def _subject_tokens(name):
    return [t for t in re.split(r"_+", str(name).lower()) if len(t) >= 3]


def token_weights(names):
    """How much each token distinguishes one of these names from the others.

    Replaces a hand-written stop-word list. `count`, `meters` and `average`
    are not domain knowledge I should be encoding -- they are simply tokens
    that recur across a tool's own field names, and a token shared by many
    names cannot say WHICH name is meant. Used to CHOOSE between candidates
    that all share something with a filter: `min_depth_meters` matches both
    `depth_measurement_meters` and `berth_rental_rate_dkk_per_meter`, and
    `meter` is worth a third of `depth` because three names carry it.
    """
    df = Counter()
    for n in names:
        for t in set(_subject_tokens(n)):
            df[t] += 1
    return {t: 1.0 / c for t, c in df.items()}


def _subject_score(a, b, weights=None):
    """How much two names talk about the same thing."""
    w, score = weights or {}, 0.0
    for x in _subject_tokens(a):
        for y in _subject_tokens(b):
            n = min(len(x), len(y), 5)
            if n >= 4 and x[:n] == y[:n]:
                score += min(w.get(x, 1.0), w.get(y, 1.0))
                break
    return score


# Half a point: a token shared by no more than two of the tool's names.
SUBJECT_FLOOR = 0.5


def _same_subject(a, b, weights=None):
    """`minimum_member_count` and `active_members` are about members."""
    return _subject_score(a, b, weights) >= SUBJECT_FLOOR


def _as_number(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return v


def _num_in(v):
    """The number a payload value holds, even when it is carried in a string.

    `_as_number` is the strict form and stays strict -- most callers compare
    two fields and must not start matching `"Rute C"` against a number. This
    one exists for the BOUNDS clamp, which was silently skipping every field
    whose value arrived as `"78 %"`: the envelope said 0-100 and the clamp
    never ran, so `"111 %"` shipped.
    """
    n = _as_number(v)
    if n is not None:
        return n
    if isinstance(v, str):
        m = NUMERIC_LITERAL.match(v)
        if m:
            return float(m.group(1))
    return None


def _with_num(v, n):
    """`n`, wearing whatever shape `v` had -- `"78 %"` stays `"95 %"`."""
    if not isinstance(v, str):
        return n
    m = NUMERIC_LITERAL.match(v)
    if not m:
        return n
    body = ("%g" % n) if float(n) != int(n) else str(int(n))
    return v[:m.start(1)] + body + v[m.end(1):]


# A number, then anything with NO further digit in it. The trailing `\D*` is
# what makes this safe without a unit vocabulary: `"111 %"` and `"25 DKK"`
# parse, `"2023-10-27"` does not, because its tail still carries digits. A
# decimal comma is deliberately NOT accepted -- `"1,500"` is 1.5 in one
# convention and 1500 in the other, and the schema examples are ASCII.
NUMERIC_LITERAL = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(\D*)$")


def example_number(e):
    """The number an EXAMPLE states, or None.

    Examples arrive as strings -- `"95"`, not `95` -- so `_as_number` rejects
    every one of them. Two audit rules were written on top of that and did
    nothing at all, while a third read `all([])` as True and counted a return
    with no examples as numeric.

    A unit carried INSIDE the example is still a number. `current_battery_level`
    declares `"78 %"`, so the strict form found no numbers, derived no
    envelope, and the repair never bounded it -- which is how a battery came
    back at `"111 %"`. Three such values shipped in the last build.
    """
    m = NUMERIC_LITERAL.match(str(_unquote(e)))
    return float(m.group(1)) if m else None


def example_envelope(field):
    """The range a field's OWN examples describe, widened by their spread.

    Derived, not named: `battery_level_percent` with examples 78/92/100 is
    bounded by what the schema itself claims the field looks like, and so is
    `air_exchange_rate` with 15/400/1100 -- which a rule keyed on the word
    "rate" would have wrongly crushed to 100. A payload that leaves this
    envelope is inventing a magnitude the tool never described.
    """
    raw = field.get("examples") or []
    vals = [example_number(e) for e in raw]
    if len(raw) < 2 or any(v is None for v in vals):
        return None
    lo, hi = min(vals), max(vals)
    spread = hi - lo
    if not spread:
        return None
    out = {"min": max(0.0, lo - spread) if lo >= 0 else lo - spread,
           "max": hi + spread}
    # Widening by the spread is too generous for a ratio: examples of 0.17,
    # 0.5 and 0.99 bought a ceiling of 1.81, and a `fingerprint_match_score`
    # came back as 1.18. A field whose every example sits in [0,1] is a
    # proportion, and 1 is its bound -- read off the values, not off the name,
    # which is why `_rate` and `score` and `andel` all get it for free. 180 of
    # 315 such fields could exceed 1 before this.
    if all(0.0 <= v <= 1.0 for v in vals):
        out["min"], out["max"] = max(0.0, out["min"]), 1.0
    return out


def _nudge(v, target, up, field, idx, taken=()):
    """Move a value onto the right side of a bound, without landing on it.

    Never onto a value already in the payload: the whole contract is that
    two same-type fields hold two different numbers.
    """
    step = max(1, int(abs(target) * 0.15)) if abs(target) >= 7 else 1
    for bump in range(8):
        off = (_hash(field, idx, bump) % step) + 1 + bump
        out = target + off if up else target - off
        if not up and target >= 0:
            out = max(out, 0)
        out = int(round(out)) if isinstance(v, int) else round(float(out), 2)
        if str(out) not in {str(x) for x in taken}:
            return out
    return out


def _bound_target(param, payload, answer_field, weights=None):
    """WHICH field a filter constrains, or None when nothing says.

    `minimum_member_count` beside `total_members`, `active_members` and
    `board_members` is a floor under the choir, not under its board, so a
    bound goes on the WHOLE: the total if there is one.

    A bound whose subject matches NO field is left alone. It used to fall
    back to the answer field on the theory that a filter must bound
    something, and that theory produced `max_results: 86` capping rush-hour
    traffic at 83 beside an average of 862, and `max_depth: 11` capping the
    scraped records at 10. A page size is not a bound on the data.

    TRIED AND REJECTED: falling back when another argument's VALUE names the
    reported field, so that "report brood_count, minimum 1101" would stop
    answering 1013. Replayed over the 15,281-row corpus it moves 86 fields,
    and reading them, ~20 are the intended repair and ~65 are new
    contradictions: `max_results: 91` capping `fault_current_amps` at 91,
    `max_items: 62` capping `material_required_kg`. Magnitude does not split
    the two populations either -- `max_results` is wrong when it misses by
    1.13x and when it misses by 25x, while `max_records: 207` beside
    `freight_manifest_count: 212` is right at 1.02x. What separates them is
    whether the parameter counts RETURNED RECORDS or measures the subject,
    and nothing in the schema says which. It needs a vocabulary; a vocabulary
    is what this file is trying not to have.
    """
    cand = [f for f, v in payload.items()
            if _as_number(v) is not None and _same_subject(param, f)]
    if not cand:
        return None
    whole = [f for f in cand if TOTAL_TOKEN.search(f)]
    cand = whole or cand
    return max(cand, key=lambda f: _subject_score(param, f, weights))


def _declared_direction(tool, param):
    """Which way a `threshold` / `cutoff` / `filter` bounds, per its own docs.

    The NAME carries no direction -- a "grænse" is neither a floor nor a
    ceiling -- so the bound used to be read as a filter and left the payload
    alone. That shipped "Hvad er signalstyrken, når grænsen er sat til 90?"
    answered with 62, the answer contradicting the constraint its own call had
    just sent. The parameter's Danish description says which way it goes
    ("Minimum signalstyrke i dBm"), and it is the same text the dresser read
    when it wrote "mindst" into the question.
    """
    p = next((x for x in (tool.get("parameters") or [])
              if x.get("name") == param), None)
    d = str((p or {}).get("description") or "")
    lo, hi = bool(UPPER_PHRASE.search(d)), bool(LOWER_PHRASE.search(d))
    if lo == hi:
        return None
    return "lo" if lo else "hi"


def respect_constraints(tool, payload, args, idx):
    """Make the payload obey the filters the call actually sent."""
    out = dict(payload)
    af = tool.get("answer_field")
    w = token_weights(list(out) + list(args or {}))
    for k, v in (args or {}).items():
        n = _as_number(v)
        if n is None:
            continue
        lo, hi = bool(BOUND_LO.search(k)), bool(BOUND_HI.search(k))
        if not (lo or hi) and BOUND_ANY.search(k):
            d = _declared_direction(tool, k)
            lo, hi = d == "lo", d == "hi"
        f = _bound_target(k, out, af, w)
        if f is None:
            continue
        cur = _as_number(out[f])
        others = [x for g, x in out.items() if g != f]
        if not (lo or hi):
            # No EQUALITY clamp. Forcing a field to equal a same-subject
            # argument fixed one row (`guest_count: 123` beside
            # `guest_list_size: 51`) and broke four: the answer field became
            # a copy of the argument, `temperature_celsius: 23.56` was
            # written into `ambient_temperature_celsius`, and `room_count:
            # 82` into `available_rooms` beside 86 occupied. Two names
            # sharing a word are not the same quantity.
            continue
        elif lo and cur < n:
            out[f] = _nudge(out[f], n, True, f, idx, others)
        elif hi and cur > n:
            out[f] = _nudge(out[f], n, False, f, idx, others)
    # Bounds the CATALOGUE declares, written once by the repair pass from the
    # field's own name: a percentage is 0-100 wherever it appears, and a stock
    # cannot exceed the silo it sits in. Declared per tool rather than guessed
    # per payload, so the relation is visible in the schema being trained on.
    for f, b in (tool.get("_bounds") or {}).items():
        cur = _num_in(out.get(f))
        if cur is None:
            continue
        others = [x for g, x in out.items() if g != f]
        lo, hi = b.get("min"), b.get("max")
        if b.get("max_field"):
            hi = min([x for x in (hi, _num_in(out.get(b["max_field"])))
                      if x is not None] or [None])
        if hi is not None and cur > hi:
            out[f] = _with_num(out[f], _nudge(cur, hi, False, f, idx, others))
        elif lo is not None and cur < lo:
            out[f] = _with_num(out[f], _nudge(cur, lo, True, f, idx, others))
    # A part cannot exceed its whole: `total_members: 27` beside
    # `active_members: 51`. The total already carries any bound, so it is the
    # parts that move.
    for t in [f for f in out if TOTAL_TOKEN.search(f)]:
        tv = _as_number(out[t])
        if tv is None:
            continue
        for f in list(out):
            fv = _as_number(out[f])
            if f == t or fv is None or not _same_subject(t, f) or fv <= tv:
                continue
            others = [x for g, x in out.items() if g != f]
            out[f] = _nudge(out[f], tv, False, f, idx, others)
    return out


# Fields that NAME a thing rather than measure it. Two calls for two subjects
# must not come back with the same one.
# A title and a name identify a subject exactly as an id does -- the user says
# "Den store Gatsby", not an accession number. Leaving them out cost twice: the
# catalogue repair demoted `book_title` as an unaskable required parameter and
# left the tool with nothing required at all, so a row asking about a named
# book called the tool with no arguments and credited the answer to the book
# anyway; and two clubs in one parallel row came back with one `club_name`,
# because sibling distinctness only guards identifiers.
IDENTIFYING = re.compile(r"(_id$|^id$|_ids$|identifier|url|uri|reference|"
                         r"_ref$|confirmation|number|kode|code|serial|_no$|"
                         r"_name$|^name$|_navn$|_title$|_titel$)",
                         re.I)


def _rekey(value, own, other):
    """Rewrite an identifier belonging to ANOTHER call's subject.

    `.../maps/12a345b.pdf` in the payload for property 001001 is not a
    coincidence: the field's examples embed the subject, and the payload is
    drawn from the examples. Substituting the call's own subject back in
    keeps the row usable instead of dropping it.
    """
    s = str(value)
    for pname, ov in (other or {}).items():
        ov = str(ov)
        if len(ov) < 3 or ov.lower() not in s.lower():
            continue
        mine = str((own or {}).get(pname, "")).strip()
        if not mine or mine.lower() == ov.lower():
            return None                     # no replacement available
        s = re.sub(re.escape(ov), mine, s, flags=re.I)
    return s


def _loose_key(value):
    """The value as a pattern that tolerates the separators a payload drops.

    A declared example is `OBJ-12345` and the payload writes `OBJ12345`; the
    case number `2024-0012-CD` turns up as `FP-2024-0012-CD-008`. Matching
    literally found neither.
    """
    s = str(value)
    runs = [r for r in re.split(r"[^0-9A-Za-zÆØÅæøå]+", s) if r]
    # A KEY, not a word. Without this, `record_type`'s examples --
    # "behandlingsjournal", "henvisning", "øvelsesprogram" -- were read as
    # other subjects' keys, and every payload sentence containing the ordinary
    # Danish word came back with it swapped: "Henvisning til ergoterapeut"
    # became "øvelsesprogram til ergoterapeut". Three rows of prose destroyed
    # before the guard.
    if not runs or sum(len(r) for r in runs) < 5 or not re.search(r"\d", s):
        return None
    return re.compile(r"[-_. ]?".join(re.escape(r) for r in runs), re.I)


def rekey_examples(tool, payload, args):
    """Identifiers that name a subject the call did not ask about.

    A payload value is drawn from the field's examples, and those examples
    embed whichever subject the tool's author had in mind. `_rekey` swaps out
    a SIBLING call's subject; the rest of the cast lives in the parameter's
    own declared examples, and they shipped: `restoration_log:
    "Restoration_Log_OBJ12345.txt"` answered for object SPECIMEN-ABCDE, and
    `fingerprint_id: "FP-2024-0012-CD-008"` answered for case 2022-9876-EF --
    both stated to the user as that subject's record.
    """
    params = {p.get("name"): p for p in (tool.get("parameters") or [])}
    subs = []
    for k, mine in (args or {}).items():
        # Only a parameter that NAMES a subject: an id, a case number, a
        # reference. A `record_type` or a `data_category` enumerates aspects,
        # and its values are words that belong in prose.
        if not isinstance(mine, str) or not str(mine).strip() \
                or not IDENTIFYING.search(k):
            continue
        for e in (params.get(k) or {}).get("examples") or []:
            e = _unquote(e)
            if not str(e).strip() or str(e).lower() == str(mine).lower():
                continue
            pat = _loose_key(e)
            if pat is not None:
                subs.append((pat, str(mine)))
    if not subs:
        return payload
    out = dict(payload)
    for f, v in out.items():
        if not isinstance(v, str):
            continue
        for pat, mine in subs:
            if pat.search(v) and not pat.fullmatch(v.strip()):
                # lambda, so a backslash or a `\1` inside an id is a literal
                v = pat.sub(lambda _m: mine, v)
        out[f] = v
    return out


def echo_identifiers(tool, payload, args, answer_field):
    """A returned identifier for the thing that was asked about IS that thing.

    `plot_identifier: "A-123"` came back with `plot_number: "A-050"` -- the
    tool answering about a different grave than the one requested. Never the
    answer field: an answer that merely repeats an argument is copyable, and
    only between two IDENTIFIERS -- `document_version: "2023-Q4"` was written
    into `historical_documentation_ref`, replacing a document reference with
    a quarter.
    """
    out = dict(payload)
    for f, v in out.items():
        if f == answer_field or not isinstance(v, str) \
                or not IDENTIFYING.search(f):
            continue
        for k, av in (args or {}).items():
            if not isinstance(av, str) or not av.strip() \
                    or not IDENTIFYING.search(k):
                continue
            if _same_subject(k, f):
                out[f] = av
                break
    return out


# Access refused, not a domain state. `compressor_status: "fault"` and
# `temperature_alarm: "active"` are things a tool legitimately reports; these
# say the caller was not allowed to have the data.
DENIED = re.compile(r"(n(æ|ae)gtet|afvist|afsl(å|aa)et|ikke tilladt|"
                    r"ingen adgang|uautoriseret|denied|deny|unauthorized|"
                    r"forbidden|blocked|blokeret|restricted)", re.I)


def clear_denials(tool, payload, answer_field):
    """A refusal in a field the user never asked about.

    `access_authorization_status: "denied"` sits beside `mast_corrosion_level:
    19`, and status fields are among the few non-answer fields the dresser can
    see. One row read it as a failed call and refused -- "Adgang til
    information om korrosion for tårn TOWER-CPH-456 er nægtet" -- with the
    corrosion level right there in the payload; the very next row, same denial,
    answered 17 without blinking. Both cannot be right, so the denial is
    swapped for another value its own field declares. A denial in the ANSWER
    field is the answer, and stays.
    """
    out = dict(payload)
    rets = {str(r.get("name")): r for r in (tool.get("returns") or [])}
    for f, v in list(out.items()):
        if f == answer_field or not isinstance(v, str) or not DENIED.search(v):
            continue
        alt = next((x for x in (rets.get(f) or {}).get("examples") or []
                    if isinstance(x, str) and x.strip()
                    and not DENIED.search(x)), None)
        if alt is None:
            out.pop(f)
        else:
            out[f] = alt
    return out


RANGE_LO = re.compile(r"(^|_)(start|from|begin|first|earliest|efter)(_|$)")
RANGE_HI = re.compile(r"(^|_)(end|until|last|latest|slut)(_|$)")


def order_ranges(args):
    """`start_year: 2024, end_year: 2022` is not a range, it is backwards."""
    if not args:
        return args
    out = dict(args)
    for a in [k for k in out if RANGE_LO.search(k)]:
        for b in [k for k in out if RANGE_HI.search(k)]:
            va, vb = _as_number(out[a]), _as_number(out[b])
            if va is None or vb is None:
                continue
            if RANGE_LO.sub("_", a.lower()) != RANGE_HI.sub("_", b.lower()):
                continue
            if va > vb:
                out[a], out[b] = out[b], out[a]
    return out


def key_payloads(tool, pays, arglist, idx, all_args=None):
    """Payloads that belong to their own call, and to no other.

    `all_args` is every call in the row. A multi-turn row asks about two
    subjects in two separate beats, and scoping this to one beat let the
    documentation reference for `hus_67890` come back as
    `.../ejendom_12345_historik.pdf` -- the other turn's building.
    """
    out = []
    pool = list(all_args or arglist)
    for k, (p, a) in enumerate(zip(pays, arglist)):
        others = [x for x in pool if x is not a]
        merged = {kk: vv for o in others for kk, vv in o.items()}
        q = dict(p)
        for f, v in list(q.items()):
            if not isinstance(v, str):
                continue
            fixed = _rekey(v, a, merged)
            if fixed is None:
                # Cannot be re-keyed: fall back to a value drawn for a
                # different row, which is at least not another subject's.
                alt = payload_for(tool, idx + 7919 * (k + 1), a).get(f)
                q[f] = alt if alt is not None else v
            else:
                q[f] = fixed
        # Two subjects, one berth number: `available_berth_id: "BERTH-E02"`
        # came back for both boats in a parallel booking.
        for f in [x for x in q if IDENTIFYING.search(x)]:
            for bump in range(1, 8):
                if all(str(q[f]) != str(prev.get(f)) for prev in out):
                    break
                alt = payload_for(tool, idx + 7919 * bump * (k + 2), a).get(f)
                if alt is None:
                    break
                q[f] = _rekey(alt, a, merged) or alt
        # Re-keying draws replacement values from other rows, which know
        # nothing about this call's filters.
        q = rekey_examples(tool, q, a)
        q = echo_identifiers(tool, q, a, tool.get("answer_field"))
        q = clear_denials(tool, q, tool.get("answer_field"))
        out.append(respect_constraints(tool, q, a, idx))
    # Two different subjects must not come back with the same ANSWER. The
    # distinctness loop above guards identifiers, and the synthesis step
    # guards the answer at draw time -- but this runs LAST, after
    # `respect_constraints` has clamped, and a clamp can walk two payloads
    # onto one number. It did: tower 170 and tower 145 both reported a signal
    # strength of -64.3, and two bike shops both reported 59 spare parts. 85
    # rows shipped that way, all of them teaching that the id does not matter.
    af = tool.get("answer_field")
    if af:
        for i, (q, a) in enumerate(zip(out, arglist)):
            cur = _num_in(q.get(af))
            if cur is None:
                continue
            if not any(arglist[j] != a and str(p.get(af)) == str(q[af])
                       for j, p in enumerate(out[:i])):
                continue
            taken = {str(p.get(af)) for p in out[:i]} | \
                    {str(v) for kk, v in q.items() if kk != af}
            q[af] = _with_num(q[af], _nudge(cur, cur, True, af, idx + i, taken))
    return out


# ── procedural: the beat sheet ─────────────────────────────────────────────

def can_vary(tool, idx, numeric_ok=True):
    """Can this tool support a second, different call at all?"""
    a = sample_args(tool, idx, 0)
    if a is None:
        return False
    return vary_one(tool, a, idx, 1, numeric_ok)[0] is not None


def answerable(tool, field):
    """Can a call actually ASK for this field?

    False when a REQUIRED parameter selects which field comes back and none
    of its options selects this one. Filtering the role rotation on this is
    cheaper than sampling an impossible row and dropping it.
    """
    for p in tool.get("parameters") or []:
        if p.get("required") and governs_field(p, tool) \
                and _selector_for(p, field, tool) is None:
            return False
    return True


# Chain is NOT in PLAN_WEIGHTS. Most families have one member and cannot chain
# at all, so a global weight would spend most of its draws on `return None`.
# It is offered only to rows whose family actually links, and at a rate high
# enough to matter given how few families qualify.
CHAIN_P = 0.5


def pick_plan(tool, idx, rng, family=None):
    """Only plans the TOOL can actually realise.

    A zero-parameter tool cannot be asked for two different things, so
    `parallel` and `multi_turn` silently collapsed to one call while the row
    kept the label -- the distribution said 5 parallel rows and none of them
    had two calls.
    """
    if family and len(family) > 1 and find_link(tool, family) \
            and _hash("chain", tool.get("name"), idx) % 100 < CHAIN_P * 100:
        return "chain"
    plan = _weighted(rng, PLAN_WEIGHTS)
    # "Do it for A and for B" needs a SUBJECT to differ. Moving a number --
    # capacity 297 vs 228 -- is a fine follow-up question but not a natural
    # two-at-once request, so parallel demands a non-numeric axis.
    if plan == "parallel" and not can_vary(tool, idx, numeric_ok=False):
        plan = "multi_turn" if can_vary(tool, idx) else "single"
    if plan in ("parallel", "multi_turn") and not can_vary(tool, idx):
        plan = _weighted(rng, {k: v for k, v in PLAN_WEIGHTS.items()
                               if k not in ("parallel", "multi_turn")})
    return plan


def _wants(tool):
    """What the user is asking for -- the FIELD NAME, not its description.

    The question still has to target the chosen answer field, or the role and
    the question diverge (a row answering `wreck_latitude` asked "find
    fartøjet" and reported the wreck's position as the vessel's). But handing
    over the field's Danish description got it back verbatim as the user's
    line in 21.1% of rows: "Den aktuelle grundvandsstand målt i meter under
    terræn" was both the schema text and the whole utterance. A user who
    phrases the request in the tool's own documentation makes field selection
    a string match, which is the shortcut the competitor field exists to
    prevent. Pass the identifier and let the model find its own words.
    """
    return tool["answer_field"]


def varied_params(beats):
    """Parameter names whose value differs between this row's calls."""
    calls = [a for b in beats for a in (b.get("kald") or [])]
    seen, out = {}, set()
    for a in calls:
        for k, v in a.items():
            if k in seen and str(seen[k]) != str(v):
                out.add(k)
            seen.setdefault(k, v)
    return out


def beats_for(plan, tool, idx, family=None):
    """The whole dialogue shape, decided in code.

    Every structural defect the gates chase -- opening on the assistant, a
    result nobody answers, a plan that never calls, two answers for one task --
    is a choice made here, correctly, once.

    `family` is the scenario's members. Only the chain plan uses it; every other
    plan calls one tool and ignores it.
    """
    b = []
    if plan == "chain":
        # `tool` IS the consumer: it carries this row's answer field, and the
        # answer comes from the second call.
        link = find_link(tool, family or [])
        if link is None:
            return None
        prod, key = link
        cons = tool
        # The user asks for what the CONSUMER reports and supplies only what
        # the PRODUCER needs. The handle between them is never spoken -- that
        # is the whole point: it has to come back from the first call.
        pa = sample_args(prod, idx, 0, answer_field=None)
        ca = sample_args(cons, idx, 1)
        if pa is None or ca is None:
            return None
        ca = dict(ca)
        ca[key] = LINK                       # resolved once the payload exists
        # ANY OTHER required argument must be SPOKEN. 210 consumers require more
        # than one identifier handle, and the lookup covers exactly one: the
        # rest were sampled from examples and never said, so `gate_call`
        # rejected the row for inventing them. They belong in the user's turn --
        # "for forsøg EXP-2024-07, hvad er strålens energi i Nordhallen?" is a
        # perfectly ordinary request, and the row then teaches the real lesson:
        # look up what you were not given, use what you were.
        spoken = {k: v for k, v in ca.items() if k != key}
        b.append({"rolle": "bruger", "bruger_beder_om": _wants(cons),
                  "args": {**pa, **spoken}})
        b.append({"rolle": "assistent", "kald": [pa], "_tool": prod})
        b.append({"rolle": "assistent", "kald": [ca], "_tool": cons,
                  "_link": {"key": key, "from": 1}})
        b.append({"rolle": "assistent", "assistent_svarer": True})
        return b
    a1 = sample_args(tool, idx, 0)
    if a1 is None:
        return None
    if plan == "refuse":
        b.append({"rolle": "bruger", "bruger_beder_om": "noget værktøjet "
                                                        "IKKE kan",
                  "args": None})
        b.append({"rolle": "assistent", "assistent_afslaar": True})
        return b
    if plan == "clarify":
        req = [p for p in (tool.get("parameters") or []) if p.get("required")]
        miss = req[_hash("miss", idx) % len(req)] if req else None
        b.append({"rolle": "bruger",
                  # `udelad` below says what to hold back. Saying it twice
                  # got it narrated: "Jeg vil gerne se adgangskontrol-
                  # begivenheder, men uden at oplyse patrulje-ID."
                  "bruger_beder_om": _wants(tool),
                  # The withheld value is named, because the model can see it
                  # in the later beat anyway and was writing it into this turn
                  # -- leaving the assistant asking which patrol, right after
                  # the user said "for patrulje 296".
                  "udelad": miss["name"] if miss else None,
                  "args": {k: v for k, v in a1.items()
                           if not miss or k != miss["name"]}})
        if miss:
            b.append({"rolle": "assistent",
                      "assistent_spoerger_om": miss["description"]})
            b.append({"rolle": "bruger",
                      "bruger_oplyser": miss["name"],
                      "args": {miss["name"]: a1.get(miss["name"])}})
        b.append({"rolle": "assistent", "kald": [a1]})
        b.append({"rolle": "assistent", "assistent_svarer": True})
        return b
    if plan == "parallel":
        a2, varied = vary_one(tool, a1, idx, 1, numeric_ok=False)
        if a2 is None:                    # nothing to vary -> a single call
            b.append({"rolle": "bruger", "bruger_beder_om": _wants(tool),
                      "args": a1})
            b.append({"rolle": "assistent", "kald": [a1]})
            b.append({"rolle": "assistent", "assistent_svarer": True})
            return b
        # Describe the SITUATION, never a phrase to repeat. "det SAMME for to
        # forskellige {description}" came back verbatim as the user's line --
        # including the raw parameter description -- in 4 of 157 rows.
        b.append({"rolle": "bruger",
                  "bruger_beder_om": f"{_wants(tool)} -- for BEGGE: nævn de "
                                     f"to konkrete værdier ved navn og bed om "
                                     f"det samme for dem begge",
                  "args": [a1, a2]})
        b.append({"rolle": "assistent", "kald": [a1, a2]})
        b.append({"rolle": "assistent", "assistent_svarer": True})
        return b
    b.append({"rolle": "bruger", "bruger_beder_om": _wants(tool),
              "args": a1})
    b.append({"rolle": "assistent", "kald": [a1]})
    b.append({"rolle": "assistent", "assistent_svarer": True})
    if plan == "multi_turn":
        a2, varied = vary_one(tool, a1, idx, 2)
        if a2 is None:
            return b
        b.append({"rolle": "bruger",
                  "bruger_beder_om": f"det samme -- {_wants(tool)} -- for en "
                                     f"ny værdi: nævn den ved navn",
                  "args": a2})
        b.append({"rolle": "assistent", "kald": [a2]})
        b.append({"rolle": "assistent", "assistent_svarer": True})
    return b


# The held-out split rule, duplicated from render_toolmind_sft.is_heldout_tool
# rather than imported: that module pulls in train_sft_packed and transformers,
# and this generator runs on boxes that have neither. If the percentage ever
# moves, it moves in both places -- push_tool_dialogues_hf takes it as a flag.
HELDOUT_PCT = 6


def is_heldout_tool(name):
    return int(hashlib.md5((name or "").encode()).hexdigest(), 16) % 100 \
        < HELDOUT_PCT


# ── families ───────────────────────────────────────────────────────────────
#
# A scenario used to hold exactly one tool -- 4,991 of 4,991 in the frozen
# catalogue -- and that is what makes chaining impossible. A chain needs two
# tools in ONE domain sharing an id vocabulary, where one's return is the
# other's required parameter. Across scenarios the key names do match (3,070
# producer/consumer pairs, 2,388 of them identifier-keyed) but the referents do
# not: `batch_id` produced by `brewery_data` and required by
# `slaughterhouse_trace` is a string match, not a chain.
#
# So `_scenario` becomes a FAMILY key that may hold several tools, and a row
# records which member it used. These two helpers are the only places that
# knowledge lives; everything else goes through them.


def family_index(tools):
    """scenario -> [tools], in catalogue order. One entry per member."""
    out = {}
    for t in tools:
        sc = t.get("_scenario")
        if sc:
            out.setdefault(sc, []).append(t)
    return out


def row_tool(row, fam, by_name=None):
    """The tool a ROW used, resolved without trusting the tool's NAME.

    Names are not unique -- the catalogue keeps collisions on purpose -- so the
    family key is the identity. `_tool` names the member; rows written before
    families existed have no `_tool` and their family has one member, which is
    unambiguous. `by_name` stays as the last resort for a row whose family is
    missing entirely.
    """
    key = str(row.get("_key") or "")
    members = fam.get(key.split("#")[0].split("/")[0]) or []
    want = row.get("_tool")
    if want:
        for t in members:
            if t.get("name") == want:
                return t
    if len(members) == 1:
        return members[0]
    if members:
        return members[0]
    return (by_name or {}).get(_called_name_row(row))


# A chain argument is not known when the beats are written: it is whatever the
# producer's payload turns out to hold, and payloads are synthesised later. The
# consumer's beat therefore carries a SENTINEL, and the payload loop swaps in
# the real value once the producer's payload exists. A sentinel rather than a
# plausible-looking placeholder, so a row that somehow escapes resolution fails
# loudly in a gate instead of shipping a made-up id.
LINK = "\u0000link\u0000"


def find_link(consumer, members):
    """(producer, key) that feeds THIS consumer, else None.

    Asked consumer-first, not pair-first, because the row's answer field is the
    consumer's: the question is what the second call reports, and the first call
    only exists to hand it a handle. Searching pairs instead would find chains
    whose answer belongs to a tool the row was not built for.

    The key must be a field the producer RETURNS and a parameter the consumer
    REQUIRES -- not merely declares -- since an optional parameter can be
    dropped by sampling and the chain would evaporate.

    Identifier-shaped only. `status` produced by one tool and accepted by
    another is a coincidence of vocabulary; `sample_id` is a handle to a thing.
    Chaining on a non-identifier would teach the model to pipe any matching
    name, which is the failure the cross-scenario pairs already illustrate
    (`batch_id`: brewery -> slaughterhouse, a string match with no referent).
    """
    need = [p.get("name") for p in (consumer.get("parameters") or [])
            if p.get("required") and IDENTIFYING.search(p.get("name") or "")]
    if not need:
        return None
    for a in members:
        if a.get("name") == consumer.get("name"):
            continue
        rets = {r["name"] for r in (a.get("returns") or [])}
        for k in need:
            if k in rets:
                return a, k
    return None


def tool_signature(t):
    """What makes a tool the same TOOL, rather than the same name.

    Two schemas may share a name -- the naming convention is narrow and a real
    corpus collides too -- but a tool whose name AND shape are already in the
    catalogue adds nothing, and its dialogues would be near-duplicates of ones
    already written. Types and required-ness are part of the shape; order and
    prose are not.
    """
    def side(fs, extra):
        return tuple(sorted(
            (str(f.get("name")), str(f.get("type") or "").lower())
            + tuple(str(f.get(k)) for k in extra)
            for f in (fs or [])))
    return (str(t.get("name")),
            side(t.get("parameters"), ("required",)),
            side(t.get("returns"), ()))


def _called_name_row(row):
    for m in row["da"]["conversations"]:
        for c in (m.get("tool_calls") or []):
            return (c.get("function") or {}).get("name")
    return None


# ── answer consistency ─────────────────────────────────────────────────────

def _visible(payload, answer_field):
    """The fields the dressing model is allowed to see for this answer.

    The answer field, and nothing else. Status fields used to be shown too,
    on the theory that a failed call must be reportable -- and the theory
    handed the dresser the competitor whenever the competitor happened to be
    named `..._status`: "Bogen står i Kælderrum B, Sektion 2, Række 5. Den er
    udlånt til John Smith" is `book_location` answered with `loan_status`
    attached, which is the shortcut the competitor exists to close. A third
    field went the same way -- "Trykket i reaktoren er 48 bar, og
    sikkerhedsventilen er lukket". A failure in the ANSWER field is still
    visible, and is still the answer.
    """
    return {k for k in payload if k == answer_field}


# Dates, clock times and years render as digits that belong to no field.
# Masking them before the unasked-field scan keeps "den 15. november 2023"
# from colliding with an unrelated `funeral_logs: 15`.
MONTHS = (r"januar|februar|marts|april|maj|juni|juli|august|september|"
          r"oktober|november|december")
DATEISH = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?"
    r"|\d{1,2}:\d{2}(?::\d{2})?"
    r"|\d{1,2}\.\s*(?:" + MONTHS + r")(?:\s+\d{4})?"
    r"|(?<!\d)(?:19|20)\d{2}(?!\d)", re.I)


# `\d+\.\d+` only, and only when the token IS the repr of a value this row
# holds. Numeric equality is not enough: Danish writes twelve thousand as
# `12.000`, which floats to 12.0, so an equality test would rewrite a
# thousands separator into a decimal comma and change the number.
DECIMAL = re.compile(r"(?<![\d.,-])\d+\.\d+(?![\d.,])")


def _floats_in(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            _floats_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _floats_in(v, out)
    elif isinstance(obj, float) and not isinstance(obj, bool):
        out.add(repr(obj))
    return out


def da_decimals(text, reprs):
    """3.43 -> 3,43. The payload is JSON; the turn is Danish prose.

    Roughly half of every read row wrote the period straight through, often
    next to a sibling turn that had written the comma, so the target itself
    was inconsistent about how a number looks.
    """
    return DECIMAL.sub(
        lambda m: m.group().replace(".", ",") if m.group() in reprs
        else m.group(), text)


def _answer_blocks(msgs):
    """(index of each answer turn, payloads it answers, calls it answers)."""
    out, pays, calls, said = [], [], [], []
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            said.append(str(m.get("content") or ""))
            pays, calls = [], []
            continue
        for c in (m.get("tool_calls") or []):
            calls.append((c.get("function") or {}).get("arguments") or {})
        if m["role"] == "tool":
            try:
                pays.append(json.loads(m["content"]))
            except Exception:
                pass
        elif m["role"] == "assistant" and str(m.get("content") or "").strip() \
                and pays:
            out.append((i, pays, calls, " ".join(said)))
            pays, calls = [], []
    return out


# An answer that hands back nothing. Not "der er en fejl" -- reporting a
# failure the payload states is correct -- but the caller being turned away.
REFUSES = re.compile(r"(er n(æ|ae)gtet|blev n(æ|ae)gtet|ingen adgang|"
                     r"ikke adgang|adgang (er )?afvist|ikke tilladt|"
                     r"kan ikke oplyse|kan ikke give dig|m(å|aa) ikke "
                     r"oplyse|uautoriseret)", re.I)


def gate_answers(row, tool):
    """What the answer turn says, beyond being true.

    The three classes reading found that no existing gate names: the answer
    volunteers a field nobody asked about, the answer drops a unit its own
    field name declares, and the decimal separator drifts between turns.
    """
    msgs = row["da"]["conversations"]
    af = tool["answer_field"]
    for i, pays, calls, said in _answer_blocks(msgs):
        text = str(msgs[i]["content"])
        scan = DATEISH.sub(lambda m: " " * len(m.group()), text)
        mine = {str(p.get(af)) for p in pays}
        from_call = {str(v) for c in calls for v in c.values()}
        # A question that quotes the answer verbatim -- "Hvad er kravene til
        # fotodokumentation af facader og interiør før renovering?" against a
        # payload holding exactly that string -- can be answered by copying
        # the question, which is the shortcut the competitor field exists to
        # close.
        av = pays[0].get(af) if pays else None
        # The value is in the payload and the answer declines to give it.
        # `clear_denials` removes the usual excuse before the dresser ever
        # sees one, so this is the backstop -- and it buys a retry, because
        # the row is sound apart from what the answer chose to say.
        if av not in (None, "") and REFUSES.search(text) \
                and not (isinstance(av, str) and DENIED.search(av)):
            return f"answer-withholds-present-value:{af}"
        if isinstance(av, str) and av.strip():
            # Verbatim at any length: "Kan du finde steriliserings-ID'et til
            # ST929341?" answered "ST929341" is a copy of the question.
            if len(av.strip()) >= 4 and av.strip().lower() in said.lower():
                return f"question-leaks-answer:{af}"
            if len(av.split()) >= 3 and _shares_content(said, av, need=3):
                return f"question-leaks-answer:{af}"
        for p in pays:
            for k, v in p.items():
                # No status exemption: see `_visible`. A field the dresser
                # cannot see is a field it must not name.
                if k == af or isinstance(v, bool) or v in (None, ""):
                    continue
                # A value the answer is entitled to say: the answer field's
                # own value, an argument, or something the user said first.
                if str(v) in mine or str(v) in from_call \
                        or _mentions(said, v):
                    continue
                if _mentions(scan, v):
                    return f"answer-cites-unasked-field:{k}"
        # Whatever survived da_decimals is either drift the model introduced
        # (`27.0` for a payload holding 27) or a Danish THOUSANDS separator,
        # which is correct and groups in threes.
        for m in DECIMAL.finditer(text):
            if not re.fullmatch(r"\d{1,3}(?:\.\d{3})+", m.group()):
                return f"answer-writes-english-decimal:{m.group()}"
    return None


# A whole row came back in English -- "What is the lift operational status for
# Val Thorens..." answered in kind. The word-list gate needs four hits from a
# 17-word list and that sentence has one. langdetect reads the whole turn: at
# p >= 0.6 it flagged those four turns and nothing else across 1182 Danish
# turns from three runs.
try:
    from langdetect import detect_langs, DetectorFactory  # noqa: E402
    DetectorFactory.seed = 0
    LANGDETECT = True
except ImportError:                                       # pragma: no cover
    LANGDETECT = False

EN_PROB = 0.6


def english_turn(text):
    """The turn is confidently English. Short turns are not judged: a name
    and a number carry no language."""
    t = " ".join(str(text or "").split())
    if not LANGDETECT or len(t.split()) < 5:
        return False
    try:
        top = detect_langs(t)[0]
    except Exception:
        return False
    return top.lang == "en" and top.prob >= EN_PROB


# ── the call must match what was said ──────────────────────────────────────

AASCII = str.maketrans({"æ": "ae", "ø": "oe", "å": "aa",
                        "Æ": "ae", "Ø": "oe", "Å": "aa"})


def _norm_tokens(s):
    """Comparable tokens: ASCII-folded, and `01` and `1` the same number."""
    out = []
    for t in re.split(r"[^a-z0-9]+", str(s).lower().translate(AASCII)):
        if not t:
            continue
        out.append(str(int(t)) if t.isdigit() else t)
    return out


def _token_hit(t, have):
    """Danish inflects: the argument `hest` is spoken as "heste", and the
    argument `hund` appears inside "hundeprøver"."""
    if t in have:
        return True
    # Four, not three: `for` is a prefix of `Forsikringsselskab`, and reading
    # a preposition as the company name made the user look like they had
    # already said it.
    if t.isdigit() or len(t) < 4:
        return False
    return any(x.startswith(t) or t.startswith(x)
               for x in have if len(x) >= 4)


NOW_WORDS = re.compile(r"(lige nu|netop nu|\bnu\b|aktuel|i dag|for tiden|"
                       r"nuv(æ|ae)rende|seneste|øjeblikket)", re.I)
MONTH_WORD = re.compile(r"(januar|februar|marts|april|maj|juni|juli|august|"
                        r"september|oktober|november|december)", re.I)
CLOCK = re.compile(r"\b\d{1,2}[:.]\d{2}\b")


def _epoch_said(v, text):
    """Is this epoch argument something the user actually asked for?

    None when the value is not an epoch at all. A "hvad er den nu"
    question legitimately carries a timestamp nobody spoke; a bare
    1678826425 beside a question with no time in it does not.
    """
    if isinstance(v, bool) or not isinstance(v, int):
        return None
    if not 1_000_000_000 <= v <= 2_000_000_000:
        return None
    if NOW_WORDS.search(text) or CLOCK.search(text):
        return True
    import datetime as _dt
    d = _dt.datetime.utcfromtimestamp(v)
    have = _answer_nums(text)
    # The year is often left out -- "den 26. oktober klokken 11:30" is a
    # perfectly specific request -- so a month and a day carry it alone.
    if float(d.day) in have and MONTH_WORD.search(text):
        return True
    return float(d.year) in have


def _leaves(v, out):
    """The values inside a structured argument, without its schema keys.

    `crawl_config: {"urls": ["example.com", "anothersite.org"],
    "rotate_proxies": true}` is spoken as "data fra example.com og
    anothersite.org, og du skal rotere proxyer". Matching the raw JSON as
    text found none of it, so the argument was ruled unspoken and dropped --
    leaving a call with no arguments under a question naming two sites.
    """
    if isinstance(v, dict):
        for x in v.values():
            _leaves(x, out)
    elif isinstance(v, (list, tuple)):
        for x in v:
            _leaves(x, out)
    elif not isinstance(v, bool) and v not in (None, ""):
        out.append(v)
    return out


def _structured(value):
    """A JSON object or array hiding in a string argument."""
    if isinstance(value, (dict, list)):
        return value
    s = str(value).strip()
    if not s.startswith(("{", "[")):
        return None
    try:
        v = json.loads(s)
    except Exception:
        return None
    return v if isinstance(v, (dict, list)) else None


def _said_value(value, text):
    """Did the user supply this value, in any wording?

    Tokens, not substrings, and folded to ASCII: the argument
    `adresse:Vejnavn 10, 1234 By` is the same information as "på Vejnavn 10
    i 1234 By", and `Renseanlaeg_Aarhus_Nord` is the same as "Renseanlæg
    Aarhus Nord".
    """
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return _traces_to(str(value), _answer_nums(text))
    inner = _structured(value)
    if inner is not None:
        # Every leaf, not a majority: a `venue_details` blob rode an address
        # nobody had mentioned into the call on the strength of its other two
        # fields.
        leaves = _leaves(inner, [])
        return all(_said_value(x, text) for x in leaves)
    toks = _norm_tokens(value)
    if not toks:
        return True
    have = set(_norm_tokens(text))
    hit = sum(1 for t in toks if _token_hit(t, have))
    # Short values must match WHOLE. At 60% a two-token value passed on one
    # token, so a user asking for `MFH-67890` was sent `MFM-67890` -- the
    # digits carried it and the prefix was free to differ.
    need = len(toks) if len(toks) <= 2 else max(1, int(round(0.6 * len(toks))))
    return hit >= need


def _spoken_pool(msgs, upto, extra_args):
    """Numbers the user is entitled to say by turn `upto`."""
    pool = set()
    for x in msgs[:upto]:
        if x["role"] in ("tool", "user"):
            pool |= _answer_nums(strip_catalogue(x.get("content") or ""))
        for c in (x.get("tool_calls") or []):
            extra_args.append((c.get("function") or {}).get("arguments") or {})
    import datetime as _dt
    for a in extra_args:
        pool |= _answer_nums(json.dumps(a, ensure_ascii=False))
        for v in a.values():
            if isinstance(v, int) and not isinstance(v, bool) \
                    and 1_000_000_000 <= v <= 2_000_000_000:
                d = _dt.datetime.utcfromtimestamp(v)
                pool |= {float(x) for x in (d.year, d.month, d.day,
                                            d.hour, d.minute, d.second)}
    return pool


LOWER_PHRASE = re.compile(r"(eller lavere|h(ø|oe)jst|maksimal|maks\.|op til|"
                          r"under|ikke over)", re.I)
UPPER_PHRASE = re.compile(r"(eller h(ø|oe)jere|mindst|minimum|mere end|over|"
                          r"fra og med)", re.I)


def unspoken_arguments(row, tool):
    """Arguments the conversation never supplies.

    `customer_id: "cust_67890"` under "hvad er mit receptnummer?" is the
    model being shown that an identifier may be produced from nothing. A
    selector or a lookup key is exempt: those are CODE's choice of which
    aspect to ask for, not a value the user hands over.

    Being an enum is NOT exemption. It was, on the theory that a closed list
    is code picking an aspect, and the theory covered two things it should
    not have: `log_level: "ERROR"` invented under a plain question about CO2,
    and `resort_name: "Trysil"` -- the required identifier, enumerated
    because the resorts are a fixed list -- invented under a question that
    named only an altitude and a date. A closed list says what the values may
    be, not that the user supplied one.
    """
    msgs = row["da"]["conversations"]
    params = {p["name"]: p for p in (tool.get("parameters") or [])}
    said, out = "", []
    for m in msgs:
        if m["role"] == "user":
            said += " " + strip_catalogue(m.get("content") or "")
            continue
        # A TOOL RESULT supplies values too. This walk is in turn order and
        # `said` only ever grows, so a call can cite a result that came BEFORE
        # it and not one that comes after -- which is exactly the chain rule.
        # Without this an id read out of a previous payload reads as invented
        # and gate_call kills the row, so no chain could survive the gate.
        # `_spoken_pool` already counted tool turns for NUMBERS; this is the
        # same rule for the string identifiers a chain actually passes.
        if m["role"] == "tool":
            said += " " + strip_catalogue(m.get("content") or "")
            continue
        for c in (m.get("tool_calls") or []):
            for k, v in (((c.get("function") or {}).get("arguments")) or {}
                         ).items():
                p = params.get(k) or {}
                # A boolean stays exempt: "Det haster" and `is_urgent: true`
                # share no token, so there is no sound way to tell a spoken
                # one from an invented one without reading the sentence.
                if isinstance(v, bool) or is_lookup_param(k) \
                        or is_meta_param(k) or selects_answer(p, v, tool):
                    continue
                ep = _epoch_said(v, said)
                ok = ep if ep is not None else _said_value(v, said)
                if not ok:
                    out.append((k, bool(p.get("required"))))
    return out


def drop_unspoken_optionals(row, tool, keep=()):
    """Remove optional arguments nobody asked for, rather than rerolling.

    These are filters CODE chose to include -- a `record_limit: 21` or a
    `timestamp` -- so the cheap repair is to stop sending them. The payload
    stays as it is: a result that happens to satisfy a filter that was never
    applied is not a contradiction, and the answer text already quotes it.

    `keep` is the axis a parallel or multi-turn row varies. Dropping it left
    two byte-identical calls returning different payloads -- a row teaching
    that the same call gives different answers. Anything held back here
    falls through to the gate, which asks the dresser to say the value
    instead.
    """
    drop = {k for k, req in unspoken_arguments(row, tool)
            if not req and k not in set(keep)}
    if not drop:
        return row, drop
    msgs = []
    for m in row["da"]["conversations"]:
        if m.get("tool_calls"):
            m = {**m, "tool_calls": [
                {**c, "function": {**c["function"], "arguments": {
                    k: v for k, v in (c["function"].get("arguments") or {}
                                      ).items() if k not in drop}}}
                for c in m["tool_calls"]]}
        msgs.append(m)
    return {**row, "da": {**row["da"], "conversations": msgs}}, drop


# The answer field is code's choice; the question is the dresser's words. When
# the two disagree about what KIND of thing is being asked for, the answer
# invents the difference: `master_version_number: 101` answered "der er 101
# masterversioner tilgængelige", and `upcoming_pelts_harvests: 1` answered
# "pelshøsten er om 1 uge".
WHEN_Q = re.compile(r"\bhvorn(å|aa)r\b", re.I)
COUNT_Q = re.compile(r"\bhvor mange\b|\bantallet af\b|\bantal\b", re.I)
TIME_FIELD = re.compile(r"(date|dato|time|tid|day|dag|hour|minut|minute|"
                        r"second|sekund|week|uge|month|m(å|aa)ned|year|"
                        r"(å|aa)r|schedule|plan|expiry|udl(ø|oe)b|deadline|"
                        r"frist|termin|timestamp|due)", re.I)
# Singular, named things. `_count`, `_units`, `_quantity`, `_days` and the
# like are absent: those answer "hvor mange" perfectly well.
NAMED_THING = re.compile(r"(_id|_ids|_number|_code|_reference|_ref|_url|"
                         r"_name|_status|_date|_time|_description)$", re.I)


def gate_kind(row, tool):
    """Does the question ask for the KIND of thing the answer field holds?"""
    af = str(tool.get("answer_field") or "")
    if not af:
        return None
    msgs = row["da"]["conversations"]
    said = " ".join(strip_catalogue(str(m.get("content") or ""))
                    for m in msgs if m["role"] == "user")
    val = None
    for m in msgs:
        if m["role"] != "tool":
            continue
        try:
            val = json.loads(m["content"]).get(af, val)
        except Exception:
            pass
        break
    dateish = isinstance(val, str) and bool(DATEISH.search(val))
    if WHEN_Q.search(said) and not TIME_FIELD.search(af) and not dateish:
        return f"question-asks-when-of-a-timeless-field:{af}"
    if COUNT_Q.search(said) and NAMED_THING.search(af):
        return f"question-asks-how-many-of-one-thing:{af}"
    return None


def gate_prose(row, tool):
    """Schema identifiers in the conversation.

    "Jeg vil gerne bede om sample_id for station 34", "inden for
    passing_accuracy", and an assistant offering to choose between
    'route_optimization' and 'fill_level_sensors'. The names are English
    snake_case by contract, so any of them appearing in a Danish turn is the
    schema leaking into the words -- and a question that names the field it
    wants makes field selection a string match.
    """
    names = {str(r.get("name")) for r in (tool.get("returns") or [])}
    names |= {str(p.get("name")) for p in (tool.get("parameters") or [])}
    for p in tool.get("parameters") or []:
        names |= {str(o) for o in (p.get("enum") or [])}
    msgs = row["da"]["conversations"]
    # The two speakers are two people. One row had the assistant answer with
    # the user's own opening sentence, word for word.
    said = {}
    for i, m in enumerate(msgs):
        t = " ".join(strip_catalogue(str(m.get("content") or "")).lower().split())
        if len(t) < 12:
            continue
        if t in said and said[t] != m["role"]:
            return "turn-repeated-by-other-speaker"
        said[t] = m["role"]
    # The clarify plan holds one value back so the assistant has something to
    # ask for. If the opening turn says it anyway, the assistant is asking
    # for what it just heard.
    #
    # An assistant turn BEFORE the first call, with a user turn after it, is
    # that question -- whatever its punctuation. Requiring a "?" missed "Jeg
    # mangler et værelsesnummer for at kunne tjekke rengøringsstatus." asked
    # of a user who had opened with "rengøringsstatus for værelse 301?", and
    # answered "301". Turns after a call are answers, not questions: matching
    # those would read every ordinary follow-up as a repeat.
    first_call = next((i for i, m in enumerate(msgs) if m.get("tool_calls")),
                      len(msgs))
    for i, m in enumerate(msgs[:first_call]):
        if m["role"] != "assistant" or m.get("tool_calls") \
                or not str(m.get("content") or "").strip():
            continue
        nxt = next((x for x in msgs[i + 1:] if x["role"] == "user"), None)
        if nxt is None:
            continue
        asked = strip_catalogue(str(nxt.get("content") or ""))
        earlier = " ".join(strip_catalogue(str(x.get("content") or ""))
                           for x in msgs[:i] if x["role"] == "user")
        # The value the user supplies AFTER the question must not already be
        # in what they said BEFORE it.
        if len(asked.strip()) >= 2 and _said_value(asked.strip(), earlier):
            return "clarify-asks-for-what-was-said"
    # A clarifying question is answered with the value, not with another
    # question: "Hvilken ovn drejer det sig om?" -> "Hvad er måltemperaturen
    # for ovn OVN-A123?" leaves the missing value supplied by nobody.
    for i, m in enumerate(msgs):
        if m["role"] != "user" or not i:
            continue
        prev = msgs[i - 1]
        if prev["role"] != "assistant" or prev.get("tool_calls"):
            continue
        if not str(prev.get("content") or "").rstrip().endswith("?") \
                and i - 1 >= first_call:
            continue
        if strip_catalogue(str(m.get("content") or "")).rstrip().endswith("?"):
            return "clarify-answered-with-question"
    for m in msgs:
        if m["role"] not in ("user", "assistant") or not m.get("content"):
            continue
        # A backslash is not Danish. One row came back with every å written
        # as a broken escape: "Hvor mange elever er der p\' lige nu".
        if "\\" in str(m["content"]):
            return "prose-contains-backslash"
        if english_turn(strip_catalogue(str(m["content"]))):
            return "turn-in-english"
    names = {n.lower() for n in names if "_" in n}
    if not names:
        return None
    for m in msgs:
        if m["role"] not in ("user", "assistant") or not m.get("content"):
            continue
        low = strip_catalogue(str(m["content"])).lower()
        for n in names:
            if re.search(r"(?<![a-z0-9_])" + re.escape(n) + r"(?![a-z0-9_])",
                         low):
                return f"schema-token-in-prose:{n}"
    return None


# A SPECIFIC time the user pins down: a date, a clock time, a month with a
# year, a bare year. Not "i dag" or "lige nu" -- those are answered perfectly
# well by a field called `access_attempts_today`, and rejecting them would
# cost good rows. This is a Danish time lexicon, like UPPER_PHRASE; it is the
# language the constraint is written in, not a domain vocabulary.
MONTH_YEAR = re.compile(r"(" + MONTHS + r")\s+(19|20)\d{2}", re.I)


def _carries_time(args):
    """Does this call send anything that could represent a time at all?"""
    for k, v in (args or {}).items():
        if TIME_FIELD.search(str(k)):
            return True
        if isinstance(v, bool):
            continue
        if isinstance(v, int) and (1_000_000_000 <= v <= 2_000_000_000
                                   or 1900 <= v <= 2100):
            return True
        s = str(v)
        if DATEISH.search(s) or MONTH_YEAR.search(s):
            return True
    return False


def gate_call(row, tool):
    """The call, against the conversation that is supposed to motivate it."""
    msgs = row["da"]["conversations"]
    said = ""
    for i, m in enumerate(msgs):
        if m["role"] != "user":
            continue
        said += " " + strip_catalogue(m.get("content") or "")
        # A number in the user's mouth that reaches no call is a value the
        # model is being taught to drop: "for plante ID 123" against a call
        # with no arguments at all.
        nxt = next((x for x in msgs[i + 1:] if x.get("tool_calls")), None)
        if nxt is None:
            continue
        pool = _spoken_pool(msgs, i,
                            [(c.get("function") or {}).get("arguments") or {}
                             for c in nxt["tool_calls"]])
        scan = LIST_ENUM.sub(lambda x: " " * len(x.group()),
                             strip_catalogue(m.get("content") or ""))
        scan = DATEISH.sub(lambda x: " " * len(x.group()), scan)
        # A digit glued to letters is a unit or a label, not a value the user
        # is quoting: "trævolumen i m3" is not the number 3.
        scan = re.sub(r"(?<=[a-zæøåA-ZÆØÅ])\d+",
                      lambda x: " " * len(x.group()), scan)
        for t in NUM.finditer(scan):
            if not _traces_to(t.group(), pool):
                return f"user-states-unused-value:{t.group()}"
        # A window the call cannot express. The dresser invents one freely --
        # dates and clock times are masked from the scan above, so they cost
        # nothing -- and then the ANSWER restates it: "mellem 26. oktober kl.
        # 00:00 og 25. oktober kl. 16:30" against a call carrying only a
        # server id, and a backwards window at that. Rather than match the
        # user's phrasing to a value, ask the weaker question: does the call
        # send ANY time at all? If not, the constraint reached nothing.
        said_now = strip_catalogue(m.get("content") or "")
        if (DATEISH.search(said_now) or MONTH_YEAR.search(said_now)) \
                and not any(_carries_time((c.get("function") or {})
                                          .get("arguments") or {})
                            for c in nxt["tool_calls"]):
            return "user-states-unused-time"
        for c in nxt["tool_calls"]:
            a = (c.get("function") or {}).get("arguments") or {}
            bounds = [k for k in a if BOUND_LO.search(k) or BOUND_HI.search(k)]
            if len(bounds) != 1:
                continue
            k = bounds[0]
            lo = bool(BOUND_LO.search(k))
            if lo and LOWER_PHRASE.search(said) \
                    and not UPPER_PHRASE.search(said):
                return f"user-inverts-bound:{k}"
            if not lo and UPPER_PHRASE.search(said) \
                    and not LOWER_PHRASE.search(said):
                return f"user-inverts-bound:{k}"
    for k, req in unspoken_arguments(row, tool):
        return f"call-invents-argument:{k}"
    return None


ANSWER_HINTS = {
    "turn-in-english":
        "Alle replikker skal være på DANSK. Kun værktøjs-, parameter- og "
        "feltnavne er engelske.",
    "clarify-asks-for-what-was-said":
        "Brugerens FØRSTE replik må ikke nævne den oplysning, assistenten "
        "bagefter spørger om -- se `udelad`.",
    "answer-withholds-present-value":
        "Står værdien i `resultat`, SKAL svaret oplyse den. Afvis aldrig og "
        "henvis aldrig til manglende adgang, når feltet står der.",
    "clarify-answered-with-question":
        "Brugeren SVARER på assistentens spørgsmål med værdien -- kort, og "
        "aldrig som et nyt spørgsmål.",
    "question-asks-when-of-a-timeless-field":
        "Spørg ikke \"hvornår\", når svarfeltet ikke er et tidspunkt. Spørg "
        "efter det, feltet faktisk indeholder.",
    "question-asks-how-many-of-one-thing":
        "Spørg ikke \"hvor mange\", når svarfeltet er ÉN navngiven ting -- et "
        "nummer, et id eller en status. Spørg hvad den er.",
    "prose-contains-backslash":
        "Skriv æ, ø og å direkte. Ingen backslash og ingen escape-sekvenser "
        "i replikkerne.",
    "turn-repeated-by-other-speaker":
        "To replikker må ikke være ens. Assistenten gentager ikke brugerens "
        "sætning.",
    "schema-token-in-prose":
        "Skriv aldrig felt-, parameter- eller valgnavne i replikkerne -- "
        "sig på almindeligt dansk, hvad de betyder.",
    "question-leaks-answer":
        "Brugeren må ikke citere svaret i sit spørgsmål. Spørg efter "
        "oplysningen, gengiv den ikke.",
    "call-invents-argument":
        "Brugeren SKAL selv nævne hver værdi, kaldet sender -- id'er, navne, "
        "datoer og tal. Skriv dem ind i brugerens replik.",
    "user-states-unused-time":
        "Nævn kun et tidsrum, hvis kaldet faktisk sender det. Har værktøjet "
        "ingen dato- eller tidsparameter, så spørg uden tidsangivelse.",
    "user-states-unused-value":
        "Brugeren må kun nævne værdier, der står i argumenterne. Opfind "
        "ikke id'er eller tal, kaldet ikke bruger.",
    "user-inverts-bound":
        "Parameteren er en NEDRE grænse, hvis navnet siger min, og en ØVRE, "
        "hvis det siger max. Brugerens ord skal vende samme vej.",
    "answer-cites-unasked-field":
        "Svar KUN på det, brugeren spurgte om. Nævn intet andet felt.",
    "answer-writes-english-decimal":
        "Decimaltal skrives med komma på dansk: 3,43 -- aldrig 3.43.",
}


def spoken_args(tool, args):
    """The arguments the user is supposed to say out loud.

    A SELECTOR's value is code's choice of which field to report, not a fact
    the user supplies -- and telling the model to name it produced "Hvad er
    bevaringsniveauet for bevaringsstatus?", the field asked for itself. The
    call still sends it; the question just stops quoting it. Same exemption
    the invented-argument gate already makes.
    """
    params = {p["name"]: p for p in (tool.get("parameters") or [])}
    def keep(a):
        return {k: v for k, v in (a or {}).items()
                if not (params.get(k, {}).get("enum")
                        or is_lookup_param(k) or is_meta_param(k)
                        or governs_field(params.get(k, {}), tool))}
    if isinstance(args, list):
        out = [keep(a) for a in args]
        return out if any(out) else None
    out = keep(args)
    return out or None


def dress_prompt(tool, beats, idx):
    """What the model is asked to write -- and only that.

    Returns (prompt, mapping) where mapping[nr] is the beat index. The numbers
    the model sees are CONTIGUOUS: numbering every beat but emitting only the
    prose ones produced gaps like [1, 3, 4, 6], and the model answered 2 and 5
    -- renumbering into its own sequence and landing precisely on the numbers
    I had skipped. 9 of 10 missing-text rejections were this.
    """
    items, mapping, n = [], {}, 0
    for bi, beat in enumerate(beats):
        if beat.get("kald"):
            continue                      # code writes this turn, not the model
        n += 1
        mapping[n] = bi
        it = {"nr": n, "rolle": beat["rolle"]}
        if beat.get("bruger_beder_om"):
            it["bruger_beder_om"] = beat["bruger_beder_om"]
            if beat.get("args"):
                spoken = spoken_args(tool, beat["args"])
                if spoken:
                    it["argumenter"] = spoken
            if beat.get("udelad"):
                it["udelad"] = beat["udelad"]
        elif beat.get("bruger_oplyser"):
            it["bruger_oplyser"] = beat["bruger_oplyser"]
            it["argumenter"] = beat.get("args") or {}
        elif beat.get("assistent_spoerger_om"):
            it["assistent_spoerger_om"] = beat["assistent_spoerger_om"]
        elif beat.get("assistent_afslaar"):
            it["assistent_afslaar"] = True
        elif beat.get("assistent_svarer"):
            it["assistent_svarer"] = True
            # REDACTED. The dressing model is shown only the fields it may
            # use. It cannot cite the competitor it never sees, and it cannot
            # write a user turn asking for it -- which is how "og den
            # planlagte hugstvolumen" ended up in a question the assistant was
            # then forbidden to answer. The competitor stays in the emitted
            # payload, where the TRAINEE must learn to pass it over; only the
            # generator is blind to it.
            #
            # Widened from the competitor to EVERY field but the answer. 17 of
            # 176 read rows volunteered a third field the user never asked for
            # -- "Huslejen er 8602 kr. og derudover et forbrugstillæg på 1665
            # kr." -- which the competitor-only redaction cannot reach and an
            # instruction not to do it did not stop. Status fields no longer
            # survive either; see `_visible`.
            red = [{k: v for k, v in p.items()
                    if k in _visible(p, tool["answer_field"])}
                   for p in beat["_pays"]]
            it["resultat"] = red[0] if len(red) == 1 else red
            # The same redaction as the question. A selector value is code's
            # choice, so an answer that states it -- "Temperaturen i
            # modningsrummet for Rococo-osten" -- is asserting an attribute
            # of a subject the user never named.
            args = spoken_args(tool, beat["_args"]) \
                or [{} for _ in beat["_args"]]
            it["argumenter"] = args[0] if len(args) == 1 else args
            it["svar_felt"] = tool["answer_field"]
        items.append(it)
    return json.dumps({
        "vaerktoej": {"navn": tool["name"], "beskrivelse": tool["description"],
                      "parametre": [{"navn": p["name"],
                                     "beskrivelse": p.get("description")}
                                    for p in tool.get("parameters") or []]},
        "beats": items}, ensure_ascii=False, indent=1), mapping


def assemble(idx, tool, beats, texts, catalogue, mapping):
    """Beats + written text -> the pipeline's conversation shape."""
    by_beat = {bi: texts.get(nr) for nr, bi in mapping.items()}
    # Every decimal this row can legitimately quote, so prose can be rewritten
    # to Danish without touching dates, ids, urls or thousands separators.
    reprs = set()
    for b in beats:
        _floats_in(b.get("_pays") or [], reprs)
        _floats_in(b.get("_args") or b.get("kald") or [], reprs)
    msgs = []
    for bi, beat in enumerate(beats):
        if beat.get("kald"):
            # The BEAT's tool when it has one: a chain calls two different
            # tools in one row, and rendering both under the anchor's name
            # would put a call to a tool that was never made.
            nm = (beat.get("_tool") or tool)["name"]
            parsed = [{"function": {"name": nm, "arguments": a}}
                      for a in beat["kald"]]
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": parsed})
            for p in beat["_pays"]:
                msgs.append({"role": "tool",
                             "content": json.dumps(p, ensure_ascii=False)})
            continue
        t = (by_beat.get(bi) or "").strip()
        if not t:
            return None
        msgs.append({"role": beat["rolle"].replace("bruger", "user")
                     .replace("assistent", "assistant"),
                     "content": da_decimals(t, reprs)})
    return {"idx": idx, "da": {"tools": catalogue, "conversations": msgs}}


# `string`, `integer`, `boolean` are what a schema says a value IS, not a
# value. One tool offered `query_params` with examples "string" and "integer",
# so a row varied on it and the user was made to ask "...hvis antal bifamilier
# er en streng?" -- and, in the parallel row, the same question twice.
TYPE_WORDS = {"string", "integer", "boolean", "number", "float", "double",
              "array", "object", "null", "none", "int", "str", "bool",
              "streng", "heltal", "tal"}


# A name that promises a collection, a log or a set of identifiers. Holding a
# bare number, it lies about its own value -- and the question written from it
# lies too: `temperature_log` with examples 300/350/400 was asked as "hvad var
# temperaturen under auktionen?" and answered "416". `_entries` and `_count`
# are absent on purpose: those names say they hold a number.
COLLECTION_NAME = re.compile(r"(_log|_list|_ids|_urls|_times|_notes|_details|"
                             r"_records|_history)$", re.I)
NUMERIC_TYPE = {"number", "integer", "int", "float", "double", "tal", "heltal"}


def gate_tool_examples(t):
    """Example values that name a type instead of being one."""
    for kind in ("parameters", "returns"):
        for f in t.get(kind) or []:
            for e in f.get("examples") or []:
                if _unquote(e).strip().lower() in TYPE_WORDS:
                    return f"example-is-a-type-name:{f.get('name')}={e}"
    for f in t.get("returns") or []:
        n = str(f.get("name") or "")
        if not COLLECTION_NAME.search(n):
            continue
        vals = [_unquote(e).strip() for e in (f.get("examples") or [])]
        if str(f.get("type") or "").lower() in NUMERIC_TYPE or (
                vals and all(_as_number(v) is not None for v in vals)):
            return f"collection-field-holds-a-number:{n}"
    return None


def catalogue_faults(t):
    """Schema defects that every dialogue built on this tool would inherit.

    The single definition of a FAIL: `audit_tool_catalogue.py` reports these,
    `repair_tool_catalogue.py` fixes them, and `--tools-from` refuses a
    catalogue that still carries them. Written twice, they would drift, and a
    catalogue could pass the audit while the generator still choked on it.

    Returns [(check, field), ...] -- empty for a clean tool.
    """
    rets, params = t.get("returns") or [], t.get("parameters") or []
    out = []

    def all_numeric(ex):
        return bool(ex) and all(example_number(e) is not None for e in ex)

    numeric = {str(r.get("name")) for r in rets
               if str(r.get("type") or "").lower() in NUMERIC_TYPE
               or all_numeric(r.get("examples"))}
    w = token_weights([str(r.get("name")) for r in rets]
                      + [str(p.get("name")) for p in params])
    declared = t.get("_bounds") or {}
    for r in rets:
        n = str(r.get("name") or "")
        if COLLECTION_NAME.search(n) and n in numeric:
            out.append(("collection-field-holds-a-number", n))
        if example_envelope(r) is not None and n not in declared:
            out.append(("numeric-return-without-its-envelope", n))
    subject = [p for p in params
               if IDENTIFYING.search(str(p.get("name") or ""))]
    if subject and not any(p.get("required") for p in params):
        out.append(("subject-param-but-nothing-required",
                    str(subject[0].get("name"))))
    for p in params:
        n, d = str(p.get("name") or ""), str(p.get("description") or "")
        is_num = str(p.get("type") or "").lower() in NUMERIC_TYPE \
            or all_numeric(p.get("examples"))
        if str(p.get("type") or "").lower() == "boolean" and any(
                _same_subject(n, str(r.get("name") or ""), w) for r in rets):
            out.append(("boolean-param-restates-a-return", n))
        if BOUND_ANY.search(n) and is_num \
                and not (BOUND_LO.search(n) or BOUND_HI.search(n)) \
                and bool(UPPER_PHRASE.search(d)) == bool(LOWER_PHRASE.search(d)):
            out.append(("threshold-without-direction", n))
        if p.get("required") and not IDENTIFYING.search(n) \
                and not is_lookup_param(n) and not is_filter_param(n) \
                and not governs_field(p, t) \
                and n not in (t.get("_keep_required") or ()) \
                and str(p.get("type") or "").lower() != "boolean":
            # `_keep_required` marks a LOOKUP tool's input. This rule says a
            # required parameter must be an id or a selector, which is right for
            # an ordinary tool and exactly inverted for a lookup: its whole job
            # is to accept the human-readable thing -- a name, an address --
            # and return the id. Without the exemption the assembler's output
            # is condemned by the fault check it was built to satisfy.
            out.append(("required-param-is-neither-id-nor-selector", n))
    dec = t.get("_selectors") or {}
    for pname in {k.split(chr(0))[0] for k in dec}:
        p = next((x for x in params if x.get("name") == pname), None)
        if not p:
            continue
        pairs = [(k.split(chr(0))[1], v) for k, v in dec.items()
                 if k.split(chr(0))[0] == pname]
        vals = {str(v).lower() for _f, v in pairs}
        if str(p.get("type") or "").lower() == "boolean" \
                or vals <= {"true", "false", "ja", "nej"}:
            out.append(("boolean-declared-as-a-selector", pname))
        elif not any(set(_subject_tokens(str(v))) & set(_subject_tokens(f))
                     for f, v in pairs):
            out.append(("selector-values-name-content", pname))
    return out


TOOL_HINTS = {
    "example-is-a-type-name":
        "Eksempelværdier skal være RIGTIGE værdier -- aldrig typenavne som "
        "\"string\", \"integer\" eller \"boolean\".",
    "collection-field-holds-a-number":
        "Et felt, der hedder noget med _log, _list, _ids eller _times, skal "
        "indeholde selve listen. Skal værdien være et tal, så navngiv det "
        "efter det, tallet tæller -- fx `temperature_reading_count`.",
    "competitor-type-differs":
        "competitor_field SKAL have NØJAGTIG samme type som answer_field -- "
        "er svaret et heltal, skal konkurrenten også være et heltal.",
    "return-echoes-parameter":
        "Ingen returfelter må hedde det samme som en parameter.",
    "return-examples-not-varied":
        "Eksempelværdierne for hvert felt skal være FORSKELLIGE fra hinanden.",
    "return-field-named-after-contract":
        "Returfelterne skal have rigtige domænenavne -- aldrig answer_field "
        "eller competitor_field.",
    "answer-field-not-in-returns":
        "answer_field og competitor_field skal begge stå i returns.",
    "competitor-field-not-in-returns":
        "answer_field og competitor_field skal begge stå i returns.",
}


async def invent_tool(session, scenario, tpl, hint=None, temp=0.9):
    prompt = (f"SCENARIE: {scenario['id']} -- {scenario['beskrivelse']}\n\n"
              f"PARAMETRENES FORM:\n{render_template(tpl)}\n"
              + (f"\nEKSTRA KRAV: {hint}\n" if hint else ""))
    return await _ask(session, TOOL_SYS, prompt, TOOL_SCHEMA, "tool", temp=temp)


# ── sibling invention ──────────────────────────────────────────────────────
#
# The anchor tool is given, and the sibling must PRODUCE a handle the anchor
# REQUIRES. Stated that way round because the anchor is the consumer: the row's
# question is what the anchor reports, and the sibling exists to hand it an id.
#
# The one thing this prompt cannot be trusted to get right is the link itself --
# a model asked for "a tool that returns X" will happily return something named
# almost-X -- so `gate_sibling` checks it against the anchor's own schema rather
# than believing the answer.
# The producer is ASSEMBLED, not invented. Code already knows everything about
# the link -- the consumer's required parameter IS the contract -- so the return
# field is COPIED from it rather than described to a model and then policed.
# Two whole defect classes stop existing rather than being gated:
#
#   format mismatch  impossible: one spec, used at both ends
#   circularity      impossible: code refuses an input drawn from the same values
#
# What is left is the only part that needs language and world knowledge: what a
# PERSON would know about this domain -- a name, an address, a product
# description -- and the prose around it. That is the split the rest of this
# generator uses, and siblings were the one place it had been abandoned.
LOOKUP_SYS = """Du opfinder ET OPSLAGSVÆRKTØJ til et dansk arbejdsområde.

Du får et værktøj, der kræver et HÅNDTAG (et id, et nummer, en reference).
Brugeren kender IKKE håndtaget. Dit opslagsværktøj findes for at finde det.

DU SKAL KUN BESKRIVE ÉN TING: hvad brugeren selv kan sige, som entydigt udpeger
den ting, håndtaget hører til.

DET SKAL VÆRE NOGET ET MENNESKE VED UDEN AT SPØRGE SYSTEMET: et personnavn, et
firmanavn, en adresse, en e-mail, et telefonnummer, en titel, en varebeskrivelse,
et registreringsnummer fra et brev.

DET SKAL UDPEGE ÉN TING. `floor_number: 1/2/3` er ikke nok til at finde ét rum --
en etage har mange rum. `country_code: DK/DE` er ikke nok til at finde én
varekode. Vælg det, der gør opslaget entydigt, og sig i beskrivelsen hvordan man
kender det.

DET MÅ ALDRIG VÆRE HÅNDTAGET SELV eller et andet internt id. Kunne brugeren sige
håndtaget, var opslaget overflødigt.

NAVNE PÅ ENGELSK i snake_case, BESKRIVELSER PÅ DANSK. Giv 3 realistiske og
FORSKELLIGE eksempelværdier.

Giv også en dansk beskrivelse af opslaget og et værktøjsnavn, der er FORSKELLIGT
fra det værktøj, du får udleveret -- fx `lookup_`/`find_` foran det, der slås op.

Og to ekstra returfelter af SAMME FORM som håndtaget -- andre referencer i samme
system, som er lette at forveksle med det."""

LOOKUP_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["name", "description", "input", "extra_returns"],
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "input": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "type", "description", "examples"],
            "properties": {
                "name": {"type": "string"}, "type": {"type": "string"},
                "description": {"type": "string"},
                "examples": {"type": "array", "items": {"type": "string"}}}},
        "extra_returns": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "description", "examples"],
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "examples": {"type": "array", "items": {"type": "string"}}}}}}}


async def invent_lookup(session, anchor, key, temp=0.9):
    """The human-sayable half of a producer. Code supplies the rest."""
    kp = next((p for p in (anchor.get("parameters") or [])
               if p.get("name") == key), None) or {}
    ex = [str(_unquote(e)) for e in (kp.get("examples") or [])][:3]
    prompt = (f"VÆRKTØJET DER MANGLER HÅNDTAGET:\n"
              f"  navn: {anchor['name']}\n"
              f"  beskrivelse: {anchor.get('description', '')}\n\n"
              f"HÅNDTAGET: {key} -- {kp.get('description', '')}\n"
              f"  eksempler: {', '.join(ex)}\n")
    return await _ask(session, LOOKUP_SYS, prompt, LOOKUP_SCHEMA, "lookup",
                      temp=temp)


def build_producer(anchor, key, spec):
    """Assemble the producer. The link half is COPIED from the consumer."""
    kp = next((p for p in (anchor.get("parameters") or [])
               if p.get("name") == key), None)
    if not kp or not spec:
        return None
    inp = spec.get("input") or {}
    if not inp.get("name") or not (inp.get("examples") or []):
        return None
    # Enforced here rather than gated afterwards: the prompt asks for an input
    # that is not the handle, and the model supplies the handle anyway -- by
    # name in 5 of 15 and by example values in 3 more. Returning None costs a
    # retry instead of a discarded tool.
    kex = {str(_unquote(e)).strip() for e in (kp.get("examples") or [])}
    iex = {str(x).strip() for x in (inp.get("examples") or [])}
    if inp["name"] == key or (kex and iex & kex):
        return None
    # The link return field IS the consumer's parameter spec. Same name, same
    # type, same examples -- so the value the chain carries is by construction
    # one the consumer declares.
    link_ret = {"name": key, "type": kp.get("type") or "string",
                "description": kp.get("description") or "",
                "examples": [str(_unquote(e)) for e in (kp.get("examples") or [])]}
    rets = [link_ret]
    for r in (spec.get("extra_returns") or [])[:2]:
        if r.get("name") and r["name"] != key and (r.get("examples") or []):
            rets.append({"name": r["name"], "type": link_ret["type"],
                         "description": r.get("description") or "",
                         "examples": [str(x) for x in r["examples"]][:3]})
    if len(rets) < 2:
        return None
    # The NAME is code's too. Asked for "a short name for a tool that looks up
    # X", the model returns X's own name -- 29 of 30 attempts in the first
    # smoke of this design. A lookup's name is mechanical anyway, and deriving
    # it guarantees it differs from the anchor.
    nm = (spec.get("name") or "").strip()
    if not nm or nm == anchor.get("name"):
        nm = f"lookup_{key}" if not key.startswith("lookup") else f"find_{key}"
    return {
        "name": nm,
        "description": spec.get("description") or "",
        "parameters": [{"name": inp["name"],
                        "type": inp.get("type") or "string",
                        "description": inp.get("description") or "",
                        "required": True,
                        "examples": [str(x) for x in inp["examples"]][:3]}],
        "returns": rets,
        "answer_field": key,
        "competitor_field": rets[1]["name"],
        "confusable_fields": [r["name"] for r in rets],
        "selectors": [],
        "user_goal": spec.get("description") or "",
        # The repair demotes a required parameter it judges unaskable, and for a
        # LOOKUP tool that is precisely backwards: accepting the thing a person
        # can say is the entire job. Pinned so the repair leaves it alone.
        "_keep_required": [inp["name"]],
    }


# ── selector resolution ────────────────────────────────────────────────────
#
# `governs_field` trusts `_selectors` when the inventor filled it and falls back
# to a NAME REGEX when it did not. The fallback is why 431 tools are unusable:
# `election_type: [Folketingsvalg, Kommunalvalg]` is an ordinary domain
# parameter condemned by its `_type` suffix, so every one of its return fields
# reads as unreachable and every row dies as `dlg:no-args-for-answer-field`.
#
# The fallback cannot simply be removed. Half the affected tools have a REAL
# selector whose options are Danish and whose fields are English --
# `data_type: [stammer, inkubation]` against `strain_name,
# incubation_temperature_celsius` -- a true 1:1 mapping no token match can see.
# Switching the regex off would fix the first half and let the second half ask
# for purity while sending `stammer`.
#
# So ask, once, per tool: which option names which field, or none. That turns a
# guess into a declaration, and `governs_field` already prefers the declaration.
SELECTOR_SYS = """Du får ét værktøj: en parameter med faste valgmuligheder, og
værktøjets returfelter.

Spørgsmålet er ÉT: vælger parameteren HVILKET RETURFELT der rapporteres?

JA -- hvis hver valgmulighed peger på et bestemt returfelt. Valgmulighederne er
på dansk og felterne på engelsk, så sammenhængen kan ikke ses af navnene alene;
den skal læses af BETYDNINGEN. Fx "stammer" -> `strain_name`, "renhed" ->
`purity_percentage`.

NEJ -- hvis valgmulighederne er egenskaber ved emnet i stedet for navne på
oplysninger. Fx `election_type: Folketingsvalg/Kommunalvalg` siger HVILKET VALG
der spørges om, ikke hvilket tal der returneres. Det samme for materialer,
racer, modeller, diæter, sorter og lignende.

Er svaret NEJ, så sæt `is_field_selector: false` og lad `mappings` være tom.
Er svaret JA, så giv én mapping per valgmulighed. Peger en valgmulighed ikke på
noget returfelt, så sæt `selects_field` til tom streng."""

SELECTOR_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["is_field_selector", "mappings"],
    "properties": {
        "is_field_selector": {"type": "boolean"},
        "mappings": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["option", "selects_field"],
            "properties": {"option": {"type": "string"},
                           "selects_field": {"type": "string"}}}}}}


def selector_candidates(tools):
    """(tool, param) pairs whose selector status is currently GUESSED.

    EVERY guessed parameter, not the first. Stopping at the first one left 18
    tools dead after a full pass: `SELECTOR_NAME` matches the substring
    `report`, so `report_date_after` claimed the tool's only slot and
    `report_type` -- the parameter actually blocking every answer field -- was
    never asked about. Same shape for `crop_type` ahead of
    `soil_analysis_report_id`, and `plant_category` ahead of
    `information_type`.

    Only where `_selectors` is empty: a declared mapping already wins in
    `governs_field`, so asking again would pay for an answer that is ignored.
    """
    out = []
    for t in tools:
        if t.get("_selectors"):
            continue
        for p in (t.get("parameters") or []):
            if governs_field(p, {**t, "answer_field": None}):
                out.append((t, p))
    return out


async def resolve_selector(session, tool, param, temp=0.3):
    """Ask whether `param` picks a return field, and which option picks what."""
    opts = [str(o) for o in (param.get("enum") or [])] or \
           [str(_unquote(e)) for e in (param.get("examples") or [])]
    rets = "\n".join(f"  - {r['name']}: {r.get('description', '')[:90]}"
                     for r in (tool.get("returns") or []))
    prompt = (f"VÆRKTØJ: {tool.get('name')} -- {tool.get('description', '')}\n\n"
              f"PARAMETER: {param.get('name')} -- "
              f"{param.get('description', '')}\n"
              f"VALGMULIGHEDER: {', '.join(opts)}\n\n"
              f"RETURFELTER:\n{rets}\n")
    # Low temperature: this is a reading task with one right answer, not an
    # invention. The tool-invention calls run at 0.9 for variety; variety here
    # would just be inconsistency.
    return await _ask(session, SELECTOR_SYS, prompt, SELECTOR_SCHEMA,
                      "selector", temp=temp)


# An option set carrying no readable content -- bare integers, single
# characters -- cannot be mapped by reading, only by pairing positions. 7 of 788
# declared selectors looked like this and 5 mapped in exact declaration order,
# including `get_field_data` whose options are 1, 5, 12: not even sequential,
# yet paired onto the return fields in order. There is no reading that produces
# that.
#
# Neither verdict is safe for these. Trusting the guess ships a call saying
# `category: "2"` whose answer reports a field that option may not select;
# marking it not-a-selector leaves the parameter required, so the call still
# sends "2" and nothing polices the pairing. The SCHEMA is the defect -- an
# option set of bare integers cannot tell a caller what it selects -- so the
# tool is dropped, which is what the repair pass does with every other
# unrepairable fault.
OPAQUE_OPTION = re.compile(r"^[\d.,\-_/]+$")


def opaque_options(param):
    opts = [str(o).strip() for o in (param.get("enum") or [])] or \
           [str(_unquote(e)).strip() for e in (param.get("examples") or [])]
    return bool(opts) and all(OPAQUE_OPTION.match(o) or len(o) <= 2
                              for o in opts)


def apply_selector_verdict(tool, param, verdict):
    """Write the verdict onto a COPY of the tool. Returns (tool, what-changed).

    A NO becomes `_not_selectors`, which `governs_field` consults before its
    regex -- the parameter stays in the schema and keeps being sampled as an
    ordinary value, it simply stops being read as a choice of field.
    """
    name = param.get("name")
    if not verdict or not verdict.get("is_field_selector"):
        prev = list(tool.get("_not_selectors") or [])
        if name in prev:
            return tool, None
        return {**tool, "_not_selectors": prev + [name]}, "not-a-selector"
    fields = {r["name"] for r in (tool.get("returns") or [])}
    sel = {}
    for m in (verdict.get("mappings") or []):
        f = (m.get("selects_field") or "").strip()
        if f and f in fields and (m.get("option") or "").strip():
            sel[f"{name}\u0000{f}"] = m["option"]
    if not sel:
        # Claimed to be a selector but mapped nothing onto a real field. Trust
        # the mappings, not the claim: an unmapped "selector" is exactly the
        # state that kills the tool.
        prev = list(tool.get("_not_selectors") or [])
        return ({**tool, "_not_selectors": prev + [name]} if name not in prev
                else tool), "claimed-but-unmapped"
    return {**tool, "_selectors": {**(tool.get("_selectors") or {}), **sel}}, \
        "declared"


def gate_sibling(sib, anchor, key):
    """Why this sibling cannot feed `anchor`, or None to keep it.

    Checked against the anchor's schema, never against what the model claims.
    """
    if not sib:
        return "no-proposal"
    if sib.get("name") == anchor.get("name"):
        return "same-name-as-anchor"
    rets = {r["name"] for r in (sib.get("returns") or [])}
    if key not in rets:
        return "does-not-return-the-key"
    # A sibling that REQUIRES the handle it is supposed to produce is a
    # circular lookup: there would be nothing to call it with.
    for p in (sib.get("parameters") or []):
        if p.get("name") == key and p.get("required"):
            return "requires-the-key-it-produces"
    # The same circularity under a DIFFERENT NAME, which the test above cannot
    # see. `biogas_motor_lookup` required `motor_identifier` with examples
    # BM-2023-001, BM-2024-042 and returned `motor_id` with exactly those
    # values -- the caller must already hold the handle to obtain the handle,
    # so the chain teaches nothing. Caught on VALUES, not on shape: a producer
    # taking `zip_code` 2100/8000 and returning `polling_station_id` 101/205
    # matches on shape (all digits) and is perfectly sound, because a postcode
    # is something a person knows and a station id is not.
    ret = next((r for r in (sib.get("returns") or [])
                if r.get("name") == key), None)
    rex = {str(_unquote(e)).strip() for e in ((ret or {}).get("examples") or [])}
    for p in (sib.get("parameters") or []):
        if not p.get("required") or not rex:
            continue
        pex = {str(_unquote(e)).strip() for e in (p.get("examples") or [])}
        if rex & pex:
            return "input-examples-are-the-output"
    # THE TWO ENDS MUST SPEAK THE SAME FORMAT. The link matches by NAME, but a
    # producer returning `item_id` PDS-2023-001 into a consumer whose `item_id`
    # is SK12345 hands a tool a value its own contract calls malformed --
    # `payload_for` builds the producer's payload from the producer's examples,
    # so that is what the chain actually passes. Seen in 3 of 10 siblings
    # despite the prompt asking for the same format in words.
    #
    # SHAPE, not values -- the opposite of the circularity test above, and for
    # the opposite reason: there we needed the two to DIFFER and only equal
    # values proved a problem; here we need them to AGREE and equal values are
    # unnecessary, since two ends of one namespace share a form without sharing
    # instances.
    cons_p = next((p for p in (anchor.get("parameters") or [])
                   if p.get("name") == key), None)
    cex = {str(_unquote(e)).strip() for e in ((cons_p or {}).get("examples") or [])}
    if rex and cex:
        shape = lambda v: re.sub(r"\d+", "#", re.sub(r"[^\W\d_]+", "A", v))
        if not ({shape(x) for x in rex} & {shape(x) for x in cex}):
            return "link-format-differs-from-consumer"
    if not [p for p in (sib.get("parameters") or []) if p.get("required")]:
        return "nothing-required"
    return None


def judge_items(row):
    """(question, arguments, payload, answer) per answer turn.

    All calls and payloads that share an answer are passed together. Pairing
    an answer with only the LAST of them made every correct parallel row look
    like fabrication and cost 14 points of measured quality.
    """
    out, lu, pc, pp = [], "", [], []
    for m in row["da"]["conversations"]:
        if m["role"] == "user":
            lu = strip_catalogue(m.get("content")) or lu
            pc, pp = [], []
        elif m.get("tool_calls"):
            pc += [(c.get("function") or {}).get("arguments") or {}
                   for c in m["tool_calls"]]
        elif m["role"] == "tool":
            pp.append(json.loads(m["content"]))
        elif m["role"] == "assistant" and str(m.get("content") or "").strip() \
                and pp:
            out.append({"spoergsmaal": lu[:300],
                        "argumenter": pc[0] if len(pc) == 1 else pc,
                        "resultat": pp[0] if len(pp) == 1 else pp,
                        "svar": m["content"][:400]})
            pc, pp = [], []
    return out


async def main_async(args):
    import aiohttp
    rng = random.Random(args.seed)
    scenarios = [json.loads(x) for x in args.scenarios.open() if x.strip()]
    rng.shuffle(scenarios)
    args.out.mkdir(parents=True, exist_ok=True)
    stats, tok = Counter(), Counter()
    sem = asyncio.Semaphore(args.concurrency)

    if not LANGDETECT:
        print("  note: langdetect not installed -- turn-in-english is only "
              "the word-list check", flush=True)

    import time as _t
    _t0 = _t.monotonic()

    # RESUME. Rows were held in memory and written once at the end, so a crash
    # at minute 13 of a 14-minute build lost all of it and a rerun started
    # over. Everything is now appended as it completes and keyed by
    # (scenario, k) so a rerun skips what exists.
    done_keys, done_scen, resumed_tools = set(), set(), {}
    tfile = args.out / "tools.jsonl"
    rfile = args.out / "translated.jsonl"
    lfile = args.out / "roles.jsonl"
    if args.resume and tfile.exists():
        for line in tfile.open():
            if line.strip():
                t = json.loads(line)
                if t.get("_scenario"):
                    # list, not scalar: a family may have several members and
                    # the last one read must not silently replace the others.
                    resumed_tools.setdefault(t["_scenario"], []).append(t)
    if args.resume and rfile.exists():
        for line in rfile.open():
            if line.strip():
                r = json.loads(line)
                if r.get("_key"):
                    done_keys.add(r["_key"])
                    done_scen.add(str(r["_key"]).split("#")[0])
    # EVERY prior row is carried forward. Loading only the unjudged ones made
    # the final rewrite emit just the new work and delete 39 of 40 finished
    # rows -- a resume that destroys the corpus is worse than none.
    resumed_rows = []
    if args.resume and rfile.exists():
        for line in rfile.open():
            if line.strip():
                resumed_rows.append(json.loads(line))
    if done_keys or resumed_tools:
        n_t = sum(len(v) for v in resumed_tools.values())
        print(f"resume: {len(done_keys)} rows and {n_t} tools in "
              f"{len(resumed_tools)} families already on disk", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    fh_t = tfile.open(mode)
    fh_r = rfile.open(mode)
    fh_l = lfile.open(mode)

    def _emit(row, af, comp):
        """Append one accepted row, flushed, so a crash keeps what is done."""
        row["idx"] = _emit.n
        _emit.n += 1
        fh_r.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh_l.write(json.dumps({"idx": row["idx"], "answer_field": af,
                               "competitor_field": comp},
                              ensure_ascii=False) + "\n")
        fh_r.flush()
        fh_l.flush()
    _emit.n = len(done_keys)

    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}"}) as session:
        want = (args.n if getattr(args, "tools_only", False)
                else max(args.n // args.dialogues_per_tool, 1))
        tools, seen_names, seen_sigs = [], set(), set()
        # Families known BEFORE dialogue generation, so a row can see its
        # siblings. Filled from the frozen catalogue below (and from anything
        # resumed), because `tools` is still being appended to while dialogues
        # run and a row built early would otherwise see a half-filled family.
        fam_all = {sc: list(ms) for sc, ms in resumed_tools.items()}
        rows, missing = list(resumed_rows), []
        tools_by_name = {}

        async def one_tool(sc):
            # A frozen catalogue: audited and repaired once, then reused. The
            # tool used to be reinvented per run, which meant a new naming
            # space and a fresh sample of schema defects every time.
            if sc.get("_tool") is not None:
                stats["tool:frozen"] += 1
                return sc["_tool"]
            if resumed_tools.get(sc["id"]):
                stats["tool:resumed"] += 1
                # The ANCHOR, i.e. the member written first. Dialogue
                # generation still builds one row per tool; which member a row
                # uses is recorded on the row, not decided here.
                return resumed_tools[sc["id"]][0]
            async with sem:
                tpl = sample_template(rng)
                hint, why, t = None, None, None
                # 49 of ~120 inventions died on competitor-type-differs with no
                # retry -- the most expensive single rejection in the run.
                for attempt in range(2):
                    t, u = await invent_tool(session, sc, tpl, hint,
                                             0.9 if not attempt else 0.6)
                    tok["in"] += u.get("prompt_tokens", 0)
                    tok["out"] += u.get("completion_tokens", 0)
                    if not t:
                        why = "no-proposal"
                        continue
                    why = gate_tool(t) or gate_tool_examples(t)
                    if not why:
                        if attempt:
                            stats["tool:ok-on-retry"] += 1
                        break
                    hint = TOOL_HINTS.get(why.split(":")[0])
                    if not hint:
                        break
                if why:
                    stats[f"tool:{why.split(':')[0]}"] += 1
                    return None
                if not any(p.get("examples") for p in t.get("parameters") or []) \
                        and (t.get("parameters") or []):
                    stats["tool:no-parameter-examples"] += 1
                    return None
                # The confusable SET is the contract; the roles are a property
                # of the question, not the tool. Fixing them at invention made
                # the assistant report `registration_data_summary` to a user
                # who asked for coordinates -- and let a model learn "for this
                # tool, always answer field X".
                by = {r["name"]: r for r in t["returns"]}
                conf = [f for f in (t.get("confusable_fields") or [])
                        if f in by]
                if len(conf) >= 2:
                    typ = by[conf[0]].get("type")
                    conf = [f for f in conf if by[f].get("type") == typ]
                if len(conf) < 2:
                    conf = [t["answer_field"], t["competitor_field"]]
                t["_confusable"] = conf
                # A DECLARED mapping, validated against the schema. The
                # heuristic it replaces matched option strings to field names
                # by prefix, which fails whenever the options are Danish and
                # the fields English -- 64 of 90 selector-shaped parameters,
                # so `emergency_plan_versions` was asked for with
                # `data_category: "valgmateriale-log"`.
                pnames = {x["name"]: x for x in t.get("parameters") or []}
                sel = {}
                for d in t.get("selectors") or []:
                    par, opt, fld = (d.get("parameter"), d.get("option"),
                                     d.get("selects_field"))
                    if par not in pnames or fld not in by:
                        continue
                    opts = {str(o) for o in
                            (pnames[par].get("enum") or [])} | \
                           {_unquote(e) for e in
                            (pnames[par].get("examples") or [])}
                    if str(opt) not in opts:
                        continue
                    sel[(par, fld)] = opt
                t["_selectors"] = {f"{k[0]}\u0000{k[1]}": v
                                   for k, v in sel.items()}
                if (t.get("selectors") or []) and not sel:
                    stats["tool:selectors-unusable"] += 1
                t["_hash"] = template_hash(tpl)
                t["_scenario"] = sc["id"]
                stats["tool:ok"] += 1
                fh_t.write(json.dumps(t, ensure_ascii=False) + "\n")
                fh_t.flush()
                return t

        async def one_dialogue(idx, tool, key=None):
            if key and key in done_keys:
                stats["dlg:resumed"] += 1
                return None
            async with sem:
                conf = tool.get("_confusable") or [tool["answer_field"],
                                                   tool["competitor_field"]]
                # Rotate only over fields a call can ASK for. A required
                # selector with no option for this field makes the row
                # unbuildable, and picking it anyway costs a whole dialogue.
                ok = [f for f in conf if answerable(tool, f)]
                # With exactly one askable field there is still a row here.
                # The COMPETITOR is never asked for -- it is the sibling in
                # the payload that the answer has to not drift onto -- so it
                # does not need to be selectable. Falling back to the
                # unfiltered list instead, which is what this did, put an
                # unaskable field in the answer slot and threw the dialogue
                # away: `dlg:no-args-for-answer-field` was 2,116 of 19,976
                # attempts, half of all yield loss, and 255 tools lose rows
                # this way while still having something to ask about.
                if len(ok) >= 2:
                    conf = ok
                elif len(ok) == 1:
                    conf = ok + [f for f in conf if f != ok[0]]
                af = conf[_hash("role", tool["name"], idx) % len(conf)] \
                    if len(ok) != 1 else conf[0]
                comp = next(f for f in conf if f != af)
                tool = {**tool, "answer_field": af, "competitor_field": comp}
                stats[f"role:{'default' if af == conf[0] else 'rotated'}"] += 1
                members = fam_all.get(tool.get("_scenario")) or [tool]
                plan = pick_plan(tool, idx, rng, members)
                beats = beats_for(plan, tool, idx, members)
                if beats is None:
                    stats["dlg:no-args-for-answer-field"] += 1
                    return None
                seen = {}
                row_args = [a for b in beats for a in (b.get("kald") or [])]
                for bi, b in enumerate(beats):
                    if not b.get("kald"):
                        continue
                    # A chain calls a DIFFERENT tool per beat, so the payload
                    # must be synthesised from that beat's contract rather than
                    # the row's anchor -- otherwise the consumer's result comes
                    # back shaped like the producer.
                    btool = b.get("_tool") or tool
                    # Resolve the link: the id the producer actually returned.
                    # Done here because payloads do not exist when beats are
                    # written. If the producer never emitted the key the row is
                    # dropped rather than shipped with a sentinel in the call.
                    lk = b.get("_link")
                    if lk:
                        src = beats[lk["from"]].get("_pays") or []
                        val = next((p0.get(lk["key"]) for p0 in src
                                    if p0.get(lk["key"]) is not None), None)
                        if val is None:
                            stats["dlg:chain-link-missing"] += 1
                            return None
                        b["kald"] = [{kk: (val if vv == LINK else vv)
                                      for kk, vv in a0.items()}
                                     for a0 in b["kald"]]
                    pays = []
                    for a in b["kald"]:
                        # keyed on WHAT was asked for, not how it was looked
                        # up -- and not on how it was FORMATTED. A report
                        # format, a verbosity, a correlation id: these change
                        # the presentation, never the thing. In the key they
                        # redrew the payload, so one player in one match came
                        # back weighing 79 kg for `feedback_type: physical`
                        # and 86 kg for `technical`.
                        subj = {kk: vv for kk, vv in a.items()
                                if not is_lookup_param(kk)
                                and not is_meta_param(kk)}
                        k = json.dumps(subj, sort_keys=True,
                                       ensure_ascii=False)
                        if k not in seen:
                            # The RECORD belongs to the entity: two calls
                            # about one recipe describe one recipe. Only the
                            # asked-for field is redrawn per query, so a
                            # filter can move the answer without rewriting
                            # what the thing is.
                            pay = dict(payload_for(btool, idx, subj))
                            for f in list(pay):
                                fk = field_key(f, subj)
                                if fk == subj:
                                    continue
                                v = payload_for(btool, idx, fk).get(f)
                                if v is not None:
                                    pay[f] = v
                            used = {str(v.get(af)) for v in seen.values()}
                            for bump in range(1, 10):
                                sibs = {str(v) for kk, v in pay.items()
                                        if kk != af}
                                if str(pay.get(af)) not in used | sibs:
                                    break
                                alt = payload_for(btool, idx + bump * 7919,
                                                  subj).get(af)
                                if alt is None:
                                    break
                                pay[af] = alt
                            # The FULL argument set constrains the result,
                            # including the lookup parameters `subj` drops.
                            seen[k] = respect_constraints(btool, pay, a, idx)
                        pays.append(seen[k])
                    b["_pays"] = key_payloads(btool, pays, b["kald"], idx,
                                              row_args)
                    b["_args"] = b["kald"]
                    # `seen` hands the same object to a repeated subject; the
                    # re-keyed copies must go back so a later beat answering
                    # this call reads what was actually emitted.
                    for a, p in zip(b["kald"], b["_pays"]):
                        seen[json.dumps({kk: vv for kk, vv in a.items()
                                         if not is_lookup_param(kk)
                                         and not is_meta_param(kk)},
                                        sort_keys=True, ensure_ascii=False)] = p
                for i, b in enumerate(beats):
                    if b.get("assistent_svarer"):
                        prev = next(x for x in reversed(beats[:i])
                                    if x.get("kald"))
                        b["_pays"], b["_args"] = prev["_pays"], prev["_args"]
                varied = varied_params(beats)
                prompt, mapping = dress_prompt(tool, beats, idx)
                # A minimal catalogue: the real catalogue is rebuilt once every
                # tool exists, but the gate needs the called tool present now.
                cat = [to_spec(tool)]
                texts, row, why, hint = {}, None, None, None
                # Three attempts, but only the ANSWER-consistency classes buy a
                # retry: they are things the dresser can be told to do
                # differently. A structural rejection is the same on a rerun.
                for attempt in range(3):
                    r, u = await _ask(
                        session,
                        DRESS_SYS + (f"\n\nEKSTRA KRAV: {hint}" if hint else ""),
                        prompt, DRESS_SCHEMA, "dress",
                        temp=0.8 if not attempt else 1.0)
                    tok["in"] += u.get("prompt_tokens", 0)
                    tok["out"] += u.get("completion_tokens", 0)
                    if not r:
                        continue
                    texts = {int(x["nr"]): x.get("tekst")
                             for x in r.get("ture") or []}
                    row, why = None, None
                    if not all((texts.get(n) or "").strip() for n in mapping):
                        continue
                    row = assemble(idx, tool, beats, texts, cat, mapping)
                    if row is None:
                        continue
                    row, dropped = drop_unspoken_optionals(row, tool, varied)
                    for d in dropped:
                        stats[f"drop:{d}"] += 1
                    why = (gate_dialogue(row, tool, plan)
                           or gate_call(row, tool)
                           or gate_prose(row, tool)
                           or gate_kind(row, tool)
                           or gate_answers(row, tool))
                    if not why:
                        if attempt:
                            stats["dlg:ok-on-retry"] += 1
                        break
                    hint = ANSWER_HINTS.get(why.split(":")[0])
                    row = None
                    if not hint:
                        break
                    stats[f"redress:{why.split(':')[0]}"] += 1
                if not texts:
                    stats["dlg:no-output"] += 1
                    return None
                if why:
                    stats[f"dlg:{why.split(':')[0]}"] += 1
                    return None
                if row is None:
                    stats["dlg:missing-text"] += 1
                    missing.append({"plan": plan, "tool": tool["name"],
                                    "answer_field": af,
                                    "wanted_nrs": list(mapping),
                                    "got_nrs": sorted(texts),
                                    "texts": {str(k): (v or "")[:120]
                                              for k, v in texts.items()}})
                    return None
                row["_plan"] = plan
                row["_key"] = key
                # WHICH member of the family this row used. A scenario may hold
                # several tools now, and `_key` names only the family. Rows
                # written before families existed carry no `_tool`; `row_tool`
                # falls back to the single member, which is unambiguous for
                # every catalogue built so far.
                row["_tool"] = tool.get("name")
                row["_answer_field"] = af
                row["_competitor_field"] = comp
                stats["dlg:ok"] += 1
                stats[f"plan:{plan}"] += 1
                _emit(row, af, comp)
                return row

        # PIPELINED. Tool invention finished at ~9s of a 13.4s run with every
        # dialogue waiting behind it. Each tool now starts its own dialogues
        # the moment it passes the gate, so the two phases overlap.
        if getattr(args, "tools_from", None):
            frozen = [json.loads(x) for x in args.tools_from.open()
                      if x.strip()]
            # The repaired catalogue and the raw one differ by a filename
            # suffix. Pointed at the raw one this ran happily and rebuilt
            # every defect the repair exists to remove -- no error, just
            # worse dialogues. Refuse, rather than repair silently: a repair
            # deletes parameters and selector claims, and doing that behind
            # the caller's back hides what changed.
            faults = Counter(why for t in frozen
                             for why, _f in catalogue_faults(t))
            if faults:
                bad = len({t["name"] for t in frozen if catalogue_faults(t)})
                raise SystemExit(
                    f"\n{args.tools_from} is not repaired: {bad} of "
                    f"{len(frozen)} tools carry schema defects that every "
                    f"dialogue built on them would inherit.\n"
                    + "\n".join(f"  {w:42s} {n}"
                                for w, n in faults.most_common())
                    + f"\n\nRun:  python scripts/repair_tool_catalogue.py "
                      f"{args.tools_from} REPAIRED.jsonl\n")
            rng.shuffle(frozen)
            # No 1.8x oversampling: a frozen tool needs no gate and cannot be
            # rejected, so one scenario slot is one tool.
            pool = [{"id": t.get("_scenario") or t["name"],
                     "beskrivelse": t.get("description") or "",
                     "_tool": t} for t in frozen[:want]]
            # The WHOLE frozen file, not the sampled pool: a scenario's
            # siblings must be visible even when only one member was drawn.
            for t in frozen:
                sc = t.get("_scenario")
                if sc and t not in fam_all.get(sc, []):
                    fam_all.setdefault(sc, []).append(t)
        elif getattr(args, "tools_only", False) and resumed_tools:
            # tools.jsonl is REWRITTEN from what this run claimed, so a tool
            # whose scenario falls outside the pool is deleted from the
            # catalogue. The shuffle is over the scenario FILE, so growing
            # that file reorders everything and a plain `scenarios[:n]` would
            # silently drop thousands of already-paid tools. Everything
            # already built is claimed first; new work fills the rest.
            have = [s for s in scenarios if resumed_tools.get(s["id"])]
            rest = [s for s in scenarios if not resumed_tools.get(s["id"])]
            need = max(0, want - len(have))
            pool = have + rest[:int(need * 1.15) + 4]
        elif getattr(args, "tools_only", False):
            # 1.8x oversampling exists because a scenario can fail to yield a
            # DIALOGUE. Inventing tools alone fails ~5% of the time, and every
            # tool past `want` is paid for and thrown away -- half the cost of
            # the first 1000-tool build.
            pool = scenarios[:int(want * 1.15) + 4]
        else:
            pool = scenarios[:int(want * 1.8) + 4]
        claimed = []

        async def tool_then_dialogues(sc):
            # Tools only: build a catalogue to audit, with no dialogues. The
            # tool set is the thing worth freezing -- it is regenerated from
            # scratch every run today, so every run meets a new naming space
            # and a new crop of schema defects.
            if getattr(args, "tools_only", False):
                if len(claimed) >= want:
                    return []
                t = await one_tool(sc)
                if not t or len(claimed) >= want:
                    return []
                if tool_signature(t) in seen_sigs:
                    stats["tool:duplicate-dropped"] += 1
                    return []
                seen_sigs.add(tool_signature(t))
                # A duplicate NAME is kept -- discarding it threw away 23%
                # of paid inventions in the 1000->2766 pass, and a real corpus
                # collides too. A duplicate SHAPE is not: same name, same
                # parameters, same returns is the same tool, and its dialogues
                # would repeat ones already written. Nothing resolves a schema
                # by name; see the catalogue assembly below.
                if t["name"] in seen_names:
                    stats["tool:name-collision-kept"] += 1
                seen_names.add(t["name"])
                claimed.append(t)
                tools.append(t)
                return []
            # The cap must count what is ALREADY on disk. `claimed` starts
            # empty on resume, so a finished run claimed a fresh `want` tools
            # on top of the existing corpus and grew 40 rows into 78.
            if len(done_keys) + len(claimed) * args.dialogues_per_tool \
                    >= args.n:
                return []
            # A scenario whose dialogues are all on disk needs no tool call at
            # all. Without this, resuming a FINISHED run still re-invented
            # tools for every candidate scenario and appended them to
            # tools.jsonl -- paid work with nothing to show.
            keys = {f"{sc['id']}#{k}" for k in range(args.dialogues_per_tool)}
            if keys and keys <= done_keys:
                stats["scenario:complete"] += 1
                return []
            t = await one_tool(sc)
            if not t or len(claimed) >= want:
                return []
            if tool_signature(t) in seen_sigs:
                stats["tool:duplicate-dropped"] += 1
                return []
            seen_sigs.add(tool_signature(t))
            if t["name"] in seen_names:
                stats["tool:name-collision-kept"] += 1
            seen_names.add(t["name"])
            claimed.append(t)
            tools.append(t)
            base = (len(claimed) - 1) * args.dialogues_per_tool
            out = await asyncio.gather(*[
                one_dialogue(base + k, t, f"{sc['id']}#{k}")
                for k in range(args.dialogues_per_tool)])
            return [r for r in out if r]

        got = await asyncio.gather(*[tool_then_dialogues(x) for x in pool])
        for batch in got:
            rows.extend(batch)
        if not tools:
            raise SystemExit("no tools survived the gate")
        (args.out / "tools.jsonl").write_text("\n".join(
            json.dumps(t, ensure_ascii=False) for t in tools) + "\n")
        print(f"  tools {len(claimed)}/{want} from {len(pool)} candidates, "
              f"{len(rows)} dialogues", flush=True)

        # Catalogues are rebuilt once every tool is known: a dialogue that ran
        # early would otherwise draw its distractors from a half-filled pool.
        #
        # Two tools MAY share a name -- the naming convention is narrow
        # (`_data`, `_status`, `_api` end 40% of them) and a real corpus has
        # collisions too. What must never happen is two DIFFERENT schemas
        # under one name inside a single prompt, or a lookup resolving a row
        # to the other tool's schema. Rows carry `_key` = "<scenario>#<k>" and
        # each tool owns its scenario, so every lookup goes through that.
        fam = family_index(tools)
        tools_by_name = {t["name"]: t for t in tools}   # fallback only

        def own_scenario(r):
            return str(r.get("_key") or "").split("#")[0].split("/")[0]

        for r in rows:
            # The row's OWN member, then its spec. Resolving by name would pick
            # the wrong schema whenever two tools share a name, which the
            # catalogue permits on purpose.
            mine_t = row_tool(r, fam, tools_by_name)
            if mine_t is None:
                continue
            mine = to_spec(mine_t)
            # A distractor that shares a name with the called tool -- or with
            # another distractor -- would put two schemas under one name in
            # front of the model. Unique names WITHIN the catalogue.
            #
            # And never a HELD-OUT name. `push_tool_dialogues_hf` drops any row
            # carrying one it does not call, because the model would read that
            # tool's name and description in a training prompt and
            # `eval_unseen_tools` would stop meaning unseen. Drawing from the
            # whole catalogue cost 2,103 of 14,333 rows (14.7%) in the last
            # build -- thrown away at push time, after they were paid for.
            taken = {mine["function"]["name"]}
            # The row's whole FAMILY is excluded, not just the member it calls.
            # A sibling in the catalogue is a legitimate alternative rather than
            # a distractor, and once families chain it is the NEXT call -- so
            # counting it as a distractor would make `right-tool` score a
            # choice the row never asked the model to make.
            mine_fam = own_scenario(r)
            others = [to_spec(x) for sc, ms in fam.items() if sc != mine_fam
                      for x in ms if not is_heldout_tool(x["name"])]
            rng.shuffle(others)
            cat, n_extra = [mine], rng.randint(1, 5)
            for x in others:
                if len(cat) > n_extra:
                    break
                if x["function"]["name"] in taken:
                    continue
                taken.add(x["function"]["name"])
                cat.append(x)
            rng.shuffle(cat)
            r["da"]["tools"] = cat
        print(f"  [phase] dialogues done at {_t.monotonic()-_t0:.1f}s", flush=True)

    # ── judge, with repair ────────────────────────────────────────────────
    if not args.no_judge and rows:
        import aiohttp as _aio
        async def _judging():
            tok2 = Counter()
            async with _aio.ClientSession(
                    headers={"Authorization": f"Bearer {_key()}"}) as sess:
                await check_judge(sess)
                todo = [r for r in rows if not r.get("_judged")]
                print(f"  judging {len(todo)} of {len(rows)} rows "
                      f"({len(rows) - len(todo)} already judged)", flush=True)
                flat = [(r, it) for r in todo for it in judge_items(r)]
                verdict = {}
                B = args.judge_batch
                chunks = [flat[i:i + B] for i in range(0, len(flat), B)]
                for s0 in range(0, len(chunks), args.concurrency):
                    got = await asyncio.gather(*[
                        judge(sess, [it for _r, it in c])
                        for c in chunks[s0:s0 + args.concurrency]])
                    for c, (v, u3) in zip(chunks[s0:s0 + args.concurrency], got):
                        tok2["in"] += u3.get("prompt_tokens", 0)
                        tok2["out"] += u3.get("completion_tokens", 0)
                        if not v:
                            continue
                        for (r, it), d in zip(c, v):
                            if not d.get("ok"):
                                verdict.setdefault(id(r), []).append(
                                    (it, d.get("problem", "")))
                    print(f"  judged {min((s0+args.concurrency)*B, len(flat))}"
                          f"/{len(flat)} turns", flush=True)
                # CONFIRM BEFORE DISCARDING. Judging 10 turns per request is
                # cheap but not neutral: re-judging the same 40 items at
                # batch 1, 5 and 10 flipped 8% of verdicts, so roughly one
                # rejection in twelve is an artefact of what shared its
                # request. Screening stays batched; every DISCARD gets a
                # second look on its own.
                solo = [(r, it, why) for r in rows
                        for it, why in verdict.get(id(r), [])]
                if solo:
                    async def _confirm(r, it, why):
                        async with sem:
                            v, u6 = await judge(sess, [it])
                            tok2["in"] += u6.get("prompt_tokens", 0)
                            tok2["out"] += u6.get("completion_tokens", 0)
                            if v and v[0].get("ok"):
                                verdict[id(r)] = [x for x in verdict.get(id(r), [])
                                                  if x[0] is not it]
                                stats["judge:unconfirmed"] += 1
                    for s2 in range(0, len(solo), args.concurrency):
                        await asyncio.gather(*[_confirm(*x)
                                               for x in solo[s2:s2 + args.concurrency]])
                    print(f"  {stats['judge:unconfirmed']} of {len(solo)} "
                          f"rejections not confirmed alone", flush=True)

                # REPAIR: a verdict is an instruction. Rewrite the offending
                # answer against its own payload plus the complaint, then
                # re-gate and re-judge that one turn.
                bad = [(r, it, why) for r in rows
                       for it, why in verdict.get(id(r), [])]
                if bad:
                    print(f"  repairing {len(bad)} turns", flush=True)

                    async def one(r, it, why):
                        async with sem:
                            # By scenario, not by name: two tools may share a
                            # name, and repairing a turn against the other
                            # one's schema redacts the wrong fields.
                            tl = row_tool(r, fam, tools_by_name)
                            if not tl:
                                return
                            pays = it["resultat"] if isinstance(
                                it["resultat"], list) else [it["resultat"]]
                            comp = r.get("_competitor_field")
                            # The SAME redaction as the first pass. Hiding
                            # only the competitor here let a repaired turn
                            # cite a field the dressing model was never shown
                            # -- "der er 71 dage til næste service" under a
                            # question about coolant level.
                            af_r = r.get("_answer_field")
                            red = [{k: v for k, v in p.items()
                                    if k in _visible(p, af_r)} for p in pays]
                            a, u4 = await _ask(
                                sess, DRESS_SYS, json.dumps({
                                    "beats": [{"nr": 1, "rolle": "assistent",
                                               "assistent_svarer": True,
                                               "spoergsmaal": it["spoergsmaal"],
                                               "argumenter": it["argumenter"],
                                               "resultat": red[0] if len(red) == 1
                                               else red,
                                               "svar_felt": r.get("_answer_field"),
                                               "ret_dette": why}]},
                                    ensure_ascii=False, indent=1),
                                DRESS_SCHEMA, "dress", temp=0.6)
                            tok2["in"] += u4.get("prompt_tokens", 0)
                            tok2["out"] += u4.get("completion_tokens", 0)
                            new = ((a or {}).get("ture") or [{}])[0].get("tekst")
                            if not (new or "").strip():
                                return
                            for m in r["da"]["conversations"]:
                                if m["role"] == "assistant" and \
                                        m.get("content") == it["svar"]:
                                    m["content"] = new.strip()
                                    it["svar"] = new.strip()
                                    break
                            tv, u5 = await judge(sess, [it])
                            tok2["in"] += u5.get("prompt_tokens", 0)
                            tok2["out"] += u5.get("completion_tokens", 0)
                            tlr = {**tl, "answer_field": r.get("_answer_field"),
                                   "competitor_field": comp}
                            if tv and tv[0].get("ok") \
                                    and gate_dialogue(r, tlr, r.get("_plan")) is None \
                                    and gate_prose(r, tlr) is None \
                                    and gate_answers(r, tlr) is None:
                                verdict[id(r)] = [x for x in verdict.get(id(r), [])
                                                  if x[0] is not it]
                                stats["judge:repaired"] += 1

                    for s1 in range(0, len(bad), args.concurrency):
                        await asyncio.gather(*[one(*b)
                                               for b in bad[s1:s1 + args.concurrency]])
            return verdict, tok2

        verdict, tok2 = await _judging()
        keep = [r for r in rows if not verdict.get(id(r))]   # judged ones have none
        stats["judge:rejected"] = len(rows) - len(keep)
        stats["judge:kept"] = len(keep)
        tok["jin"] += tok2["in"]
        tok["jout"] += tok2["out"]
        rows = keep

    for _fh in (fh_t, fh_r, fh_l):
        _fh.close()
    # One consistent file at the end. The append above is the crash guard; the
    # rewrite drops whatever the judge rejected and marks the rest judged, so
    # a later --resume does not re-judge them.
    with rfile.open("w") as f, lfile.open("w") as g:
        for i, r in enumerate(rows):
            r["idx"] = i
            r["_judged"] = True
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            g.write(json.dumps({"idx": i,
                                "answer_field": r.get("_answer_field"),
                                "competitor_field": r.get("_competitor_field")},
                               ensure_ascii=False) + "\n")
    (args.out / "missing_text.jsonl").write_text("\n".join(
        json.dumps(x, ensure_ascii=False) for x in missing) + "\n")
    out = rfile
    print()
    for k, v in sorted(stats.items()):
        print(f"   {v:>5}  {k}")
    jcost = tok["jin"] / 1e6 * 0.10 + tok["jout"] / 1e6 * 0.40
    cost = tok["in"] / 1e6 * 0.10 + tok["out"] / 1e6 * 0.40
    print(f"\ngeneration: in={tok['in']:,} out={tok['out']:,}  ~${cost:.4f}")
    print(f"judge     : in={tok['jin']:,} out={tok['jout']:,}  ~${jcost:.4f}")
    print(f"  per accepted row: ${(cost + jcost) / max(len(rows), 1):.6f}")
    print(f"template shapes: {len({t['_hash'] for t in tools})} over {len(tools)} tools")
    print(f"-> {out}  ({len(rows)} rows)")


async def fix_selectors_run(args):
    """Resolve every GUESSED selector in a catalogue, then rewrite it.

    CACHED, like the tool invention it sits beside: each verdict is appended to
    `<out>/selector_verdicts.jsonl` as it lands, and a rerun reads them back and
    asks only for what is missing. A pass that died at tool 900 of 1,114 used to
    mean paying for 900 answers twice.
    """
    import aiohttp
    tools = [json.loads(l) for l in args.fix_selectors.open() if l.strip()]
    cands = selector_candidates(tools)
    vfile = args.out / "selector_verdicts.jsonl"
    args.out.mkdir(parents=True, exist_ok=True)

    cache = {}
    if vfile.exists():
        for line in vfile.open():
            if line.strip():
                v = json.loads(line)
                cache[(v["tool"], v["scenario"], v["param"])] = v["verdict"]
    todo = [(t, p) for t, p in cands
            if (t.get("name"), t.get("_scenario"), p.get("name")) not in cache]
    print(f"{len(tools):,} tools   {len(cands):,} guessed selectors   "
          f"{len(cache):,} cached   {len(todo):,} to ask", flush=True)

    stats, tok = Counter(), Counter()
    sem = asyncio.Semaphore(args.concurrency)
    fh = vfile.open("a")

    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}"}) as session:
        async def one(t, prm):
            async with sem:
                v, u = await resolve_selector(session, t, prm)
                tok["in"] += u.get("prompt_tokens", 0)
                tok["out"] += u.get("completion_tokens", 0)
                if v is None:
                    stats["no-verdict"] += 1
                    return
                key = (t.get("name"), t.get("_scenario"), prm.get("name"))
                cache[key] = v
                fh.write(json.dumps({"tool": key[0], "scenario": key[1],
                                     "param": key[2], "verdict": v},
                                    ensure_ascii=False) + "\n")
                fh.flush()
                stats["asked"] += 1
                if stats["asked"] % 200 == 0:
                    print(f"  {stats['asked']}/{len(todo)}", flush=True)
        await asyncio.gather(*[one(t, prm) for t, prm in todo])
    fh.close()

    by_param = {}
    for t, prm in cands:
        by_param.setdefault((t.get("name"), t.get("_scenario")), []).append(prm)
    out, before, after = [], 0, 0
    for t in tools:
        prms = by_param.get((t.get("name"), t.get("_scenario"))) or []
        fields = [r["name"] for r in (t.get("returns") or [])]
        n0 = sum(1 for f in fields
                 if sample_args({**t, "answer_field": f}, 1, 0) is not None)
        drop = False
        for prm in prms:
            v = cache.get((t.get("name"), t.get("_scenario"), prm.get("name")))
            if v is None:
                continue
            if v.get("is_field_selector") and opaque_options(prm):
                stats["dropped:opaque-selector-options"] += 1
                drop = True
                break
            t, what = apply_selector_verdict(t, prm, v)
            if what:
                stats[f"verdict:{what}"] += 1
        if drop:
            continue
        n1 = sum(1 for f in fields
                 if sample_args({**t, "answer_field": f}, 1, 0) is not None)
        before += n0
        after += n1
        out.append(t)

    dest = args.fix_selectors.with_name(args.fix_selectors.stem + "_sel.jsonl")
    dest.write_text("\n".join(json.dumps(t, ensure_ascii=False)
                              for t in out) + "\n")
    for k, v in sorted(stats.items()):
        print(f"   {v:>6}  {k}")
    cost = tok["in"] / 1e6 * 0.10 + tok["out"] / 1e6 * 0.40
    print(f"\nreachable answer-field slots: {before:,} -> {after:,} "
          f"(+{after - before:,})")
    print(f"tokens in={tok['in']:,} out={tok['out']:,}  ~${cost:.4f}")
    print(f"-> {dest}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("scratch/proc"))
    ap.add_argument("--scenarios", type=Path,
                    default=Path("data/tool_calls/scenarios_expanded.jsonl"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dialogues-per-tool", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: reuse the tools and "
                         "rows already in --out and generate only what is "
                         "missing")
    ap.add_argument("--tools-from", type=Path, default=None,
                    help="a frozen catalogue (audited and repaired) to build "
                         "dialogues on, instead of inventing tools per run")
    ap.add_argument("--tools-only", action="store_true",
                    help="invent --n TOOLS and stop: no dialogues, no judge. "
                         "For building a catalogue to audit and freeze.")
    ap.add_argument("--fix-selectors", type=Path, default=None,
                    help="resolve every GUESSED selector in this catalogue and "
                         "write <name>_sel.jsonl. Verdicts are cached in "
                         "<out>/selector_verdicts.jsonl, so a rerun asks only "
                         "for what is missing.")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-batch", type=int, default=10)
    a = ap.parse_args()
    try:
        if a.fix_selectors:
            asyncio.run(fix_selectors_run(a))
            return
        asyncio.run(main_async(a))
    except FatalAPIError as e:
        print(f"\nABORTED: {e}\n"
              f"Nothing further was attempted. Whatever completed is in "
              f"{a.out} -- rerun the same command with --resume once the "
              f"account can make calls again.", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
