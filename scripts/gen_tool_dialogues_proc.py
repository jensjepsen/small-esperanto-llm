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

    uv run python scripts/gen_tool_dialogues_proc.py --out scratch/proc --n 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tool_dialogues_da import (  # noqa: E402
    _ask, _hash, _key, _unquote, _numeric_span, _weighted, check_judge,
    gate_dialogue, gate_tool, judge, payload_for, sample_template,
    render_template, strip_catalogue, template_hash, to_spec, PLAN_WEIGHTS)

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

Giv 2-3 realistiske og FORSKELLIGE eksempelværdier for HVER parameter og HVERT
returfelt. Eksemplerne bruges direkte som data, så de skal være rigtige
værdier -- ikke beskrivelser af værdier."""

TOOL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["name", "description", "parameters", "returns",
                 "answer_field", "competitor_field", "confusable_fields",
                 "user_goal"],
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
                              "items": {"type": "string"}}}}

DRESS_SYS = """Du skriver replikkerne i en dansk samtale, hvis FORLØB allerede
ligger fast.

Du får en liste af BEATS. Hvert beat har et nummer og en rolle. Du skriver KUN
tekst til de beats, der beder om det -- du må ikke tilføje, fjerne eller
ombytte beats.

- `bruger_beder_om`: skriv brugerens replik. Den skal naturligt føre til
  præcis de argumenter, der står -- nævn værdierne, men som almindeligt sprog,
  ikke som felter. Feltet beskriver, HVAD brugeren vil vide; gengiv det
  ALDRIG ordret, og skriv aldrig et parameternavn eller en feltbeskrivelse i
  replikken. Spørgsmålet skal handle om netop det, `svar_felt` indeholder --
  ikke om noget andet i resultatet.
- `assistent_spoerger_om`: skriv assistentens spørgsmål efter netop den
  manglende oplysning.
- `assistent_svarer`: skriv assistentens svar. Brug feltet `svar_felt` fra
  `resultat`. Brug kun tal og tekst fra `resultat`, fra argumenterne eller
  fra det brugeren selv sagde. Skriv aldrig feltnavne i teksten; sig hvad
  feltet betyder. Hele sætninger, ikke en rå gengivelse af værdien.
- `assistent_afslaar`: skriv et kort, ligefremt afslag.

Siger `resultat`, at kaldet fejlede (fx `status: "fejl"`), skal svaret sige
det ligeud -- rapporter ALDRIG et tal fra et mislykket kald, som om alt gik
godt.

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


def _selector_for(param, answer_field):
    """A choice this parameter offers that SELECTS the answer field.

    Some parameters pick which aspect the tool reports. Sampling them
    independently of the chosen answer field produced a call for
    `measurement_type: "pump_efficiency"` answering a question about
    `bacteria_count_per_ml` -- the arguments contradicting the question.
    """
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


def sample_args(tool, idx, salt=0, optional_p=0.5, answer_field=None):
    """Required parameters always; optional ones sometimes. Never null."""
    out = {}
    af = answer_field or tool.get("answer_field")
    for p in tool.get("parameters") or []:
        if not p.get("required") and _hash("opt", p.get("name"), idx, salt) % 100 \
                >= optional_p * 100:
            continue
        sel = _selector_for(p, af) if af else None
        v = sel if sel is not None else sample_arg(p, idx, salt)
        if v is not None and v != "":
            out[p["name"]] = v
    return out


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
    cands = [p for p in (tool.get("parameters") or [])
             if p["name"] in args
             and _selector_for(p, af) is None
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
            return out, pick
    return None, None


# ── procedural: the beat sheet ─────────────────────────────────────────────

def can_vary(tool, idx, numeric_ok=True):
    """Can this tool support a second, different call at all?"""
    a = sample_args(tool, idx, 0)
    return vary_one(tool, a, idx, 1, numeric_ok)[0] is not None


def pick_plan(tool, idx, rng):
    """Only plans the TOOL can actually realise.

    A zero-parameter tool cannot be asked for two different things, so
    `parallel` and `multi_turn` silently collapsed to one call while the row
    kept the label -- the distribution said 5 parallel rows and none of them
    had two calls.
    """
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
    """What the user is asking for, in the answer field's own words.

    The role is chosen per row and the question was written from the
    ARGUMENTS, so nothing tied them together: a row whose answer field was
    `wreck_latitude` asked "find fartøjet" and answered with the wreck's
    position, labelled as the vessel's. The question has to name what it wants.
    """
    by = {r["name"]: r for r in tool["returns"]}
    spec = by.get(tool["answer_field"]) or {}
    return (spec.get("description") or tool["answer_field"]).rstrip(".")


def beats_for(plan, tool, idx):
    """The whole dialogue shape, decided in code.

    Every structural defect the gates chase -- opening on the assistant, a
    result nobody answers, a plan that never calls, two answers for one task --
    is a choice made here, correctly, once.
    """
    b = []
    a1 = sample_args(tool, idx, 0)
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
                  "bruger_beder_om": f"dette, men UDEN at oplyse alt: "
                                     f"{_wants(tool)}",
                  "args": {k: v for k, v in a1.items()
                           if not miss or k != miss["name"]}})
        if miss:
            b.append({"rolle": "assistent",
                      "assistent_spoerger_om": miss["description"]})
            b.append({"rolle": "bruger", "bruger_beder_om": "kun den "
                                                            "manglende oplysning",
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


def _called_name_row(row):
    for m in row["da"]["conversations"]:
        for c in (m.get("tool_calls") or []):
            return (c.get("function") or {}).get("name")
    return None


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
                it["argumenter"] = beat["args"]
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
            comp = tool["competitor_field"]
            red = [{k: v for k, v in p.items() if k != comp}
                   for p in beat["_pays"]]
            it["resultat"] = red[0] if len(red) == 1 else red
            it["argumenter"] = beat["_args"][0] if len(beat["_args"]) == 1 \
                else beat["_args"]
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
    msgs = []
    for bi, beat in enumerate(beats):
        if beat.get("kald"):
            parsed = [{"function": {"name": tool["name"], "arguments": a}}
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
                     .replace("assistent", "assistant"), "content": t})
    return {"idx": idx, "da": {"tools": catalogue, "conversations": msgs}}


TOOL_HINTS = {
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

    import time as _t
    _t0 = _t.monotonic()
    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}"}) as session:
        want = max(args.n // args.dialogues_per_tool, 1)
        tools, seen_names = [], set()
        rows, missing = [], []
        specs, tools_by_name = [], {}

        async def one_tool(sc):
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
                    why = gate_tool(t)
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
                t["_hash"] = template_hash(tpl)
                stats["tool:ok"] += 1
                return t

        async def one_dialogue(idx, tool):
            async with sem:
                conf = tool.get("_confusable") or [tool["answer_field"],
                                                   tool["competitor_field"]]
                af = conf[_hash("role", tool["name"], idx) % len(conf)]
                comp = next(f for f in conf if f != af)
                tool = {**tool, "answer_field": af, "competitor_field": comp}
                stats[f"role:{'default' if af == conf[0] else 'rotated'}"] += 1
                plan = pick_plan(tool, idx, rng)
                beats = beats_for(plan, tool, idx)
                seen = {}
                for b in beats:
                    if not b.get("kald"):
                        continue
                    pays = []
                    for a in b["kald"]:
                        k = json.dumps(a, sort_keys=True, ensure_ascii=False)
                        if k not in seen:
                            pay = payload_for(tool, idx, a)
                            used = {str(v.get(af)) for v in seen.values()}
                            for bump in range(1, 10):
                                if str(pay.get(af)) not in used:
                                    break
                                pay = payload_for(tool, idx + bump * 7919, a)
                            seen[k] = pay
                        pays.append(seen[k])
                    b["_pays"], b["_args"] = pays, b["kald"]
                for i, b in enumerate(beats):
                    if b.get("assistent_svarer"):
                        prev = next(x for x in reversed(beats[:i])
                                    if x.get("kald"))
                        b["_pays"], b["_args"] = prev["_pays"], prev["_args"]
                prompt, mapping = dress_prompt(tool, beats, idx)
                texts = {}
                for attempt in range(2):
                    r, u = await _ask(session, DRESS_SYS, prompt, DRESS_SCHEMA,
                                      "dress", temp=0.8 if not attempt else 1.0)
                    tok["in"] += u.get("prompt_tokens", 0)
                    tok["out"] += u.get("completion_tokens", 0)
                    if not r:
                        continue
                    texts = {int(x["nr"]): x.get("tekst")
                             for x in r.get("ture") or []}
                    if all((texts.get(n) or "").strip() for n in mapping):
                        if attempt:
                            stats["dlg:ok-on-retry"] += 1
                        break
                if not texts:
                    stats["dlg:no-output"] += 1
                    return None
                # A minimal catalogue: the real catalogue is rebuilt once every
                # tool exists, but the gate needs the called tool present now.
                cat = [to_spec(tool)]
                row = assemble(idx, tool, beats, texts, cat, mapping)
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
                row["_answer_field"] = af
                row["_competitor_field"] = comp
                why = gate_dialogue(row, tool, plan)
                if why:
                    stats[f"dlg:{why.split(':')[0]}"] += 1
                    return None
                stats["dlg:ok"] += 1
                stats[f"plan:{plan}"] += 1
                return row

        # PIPELINED. Tool invention finished at ~9s of a 13.4s run with every
        # dialogue waiting behind it. Each tool now starts its own dialogues
        # the moment it passes the gate, so the two phases overlap.
        need = int(want * 1.8) + 4
        pool = scenarios[:need]
        claimed = []

        async def tool_then_dialogues(sc):
            t = await one_tool(sc)
            if not t or t["name"] in seen_names or len(claimed) >= want:
                return []
            seen_names.add(t["name"])
            claimed.append(t)
            tools.append(t)
            base = (len(claimed) - 1) * args.dialogues_per_tool
            out = await asyncio.gather(*[
                one_dialogue(base + k, t) for k in range(args.dialogues_per_tool)])
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
        specs = [to_spec(t) for t in tools]
        tools_by_name = {t["name"]: t for t in tools}
        for r in rows:
            called = _called_name_row(r)
            others = [x for x in specs if x["function"]["name"] != called]
            rng.shuffle(others)
            cat = [x for x in specs if x["function"]["name"] == called] + \
                others[:rng.randint(1, 5)]
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
                flat = [(r, it) for r in rows for it in judge_items(r)]
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
                # REPAIR: a verdict is an instruction. Rewrite the offending
                # answer against its own payload plus the complaint, then
                # re-gate and re-judge that one turn.
                bad = [(r, it, why) for r in rows
                       for it, why in verdict.get(id(r), [])]
                if bad:
                    print(f"  repairing {len(bad)} turns", flush=True)

                    async def one(r, it, why):
                        async with sem:
                            tl = tools_by_name.get(_called_name_row(r))
                            if not tl:
                                return
                            pays = it["resultat"] if isinstance(
                                it["resultat"], list) else [it["resultat"]]
                            comp = r.get("_competitor_field")
                            red = [{k: v for k, v in p.items() if k != comp}
                                   for p in pays]
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
                            if tv and tv[0].get("ok") and \
                                    gate_dialogue(r, tlr, r.get("_plan")) is None:
                                verdict[id(r)] = [x for x in verdict.get(id(r), [])
                                                  if x[0] is not it]
                                stats["judge:repaired"] += 1

                    for s1 in range(0, len(bad), args.concurrency):
                        await asyncio.gather(*[one(*b)
                                               for b in bad[s1:s1 + args.concurrency]])
            return verdict, tok2

        verdict, tok2 = await _judging()
        keep = [r for r in rows if not verdict.get(id(r))]
        stats["judge:rejected"] = len(rows) - len(keep)
        stats["judge:kept"] = len(keep)
        tok["jin"] += tok2["in"]
        tok["jout"] += tok2["out"]
        rows = keep

    (args.out / "missing_text.jsonl").write_text("\n".join(
        json.dumps(x, ensure_ascii=False) for x in missing) + "\n")
    out = args.out / "translated.jsonl"
    with out.open("w") as f, (args.out / "roles.jsonl").open("w") as g:
        for i, r in enumerate(rows):
            r["idx"] = i
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            g.write(json.dumps({"idx": i,
                                "answer_field": r.get("_answer_field"),
                                "competitor_field": r.get("_competitor_field")},
                               ensure_ascii=False) + "\n")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("scratch/proc"))
    ap.add_argument("--scenarios", type=Path,
                    default=Path("data/tool_calls/scenarios_expanded.jsonl"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dialogues-per-tool", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-batch", type=int, default=10)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
