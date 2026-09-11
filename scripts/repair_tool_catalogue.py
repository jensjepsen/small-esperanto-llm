"""Repair a tool catalogue offline, without asking the model anything.

Every repair is a deletion or a demotion -- never an invention -- so a
repaired schema says less than it did, and nothing it did not already say.
Over 1000 invented tools this took the failing set from 553 to 22.

Usage:
  uv run --no-project --with aiohttp --with langdetect python \
      scripts/repair_tool_catalogue.py IN.jsonl OUT.jsonl
"""
import json, re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tool_dialogues_proc import (
    BOUND_ANY, BOUND_HI, BOUND_LO, COLLECTION_NAME, LOWER_PHRASE,
    NUMERIC_TYPE, UPPER_PHRASE, IDENTIFYING, _as_number, _same_subject,
    _subject_tokens, _unquote, governs_field, is_filter_param,
    is_lookup_param, token_weights, example_envelope,
    example_number,
)

tools = [json.loads(x) for x in Path(sys.argv[1]).open() if x.strip()]
did, dropped_tools = Counter(), []


def all_numeric(examples):
    """Every example is a number, and there IS an example. `all([])` is True,
    which counted a return with no examples as numeric."""
    return bool(examples) and all(example_number(e) is not None
                                  for e in examples)


out = []
for t in tools:
    rets, params = t.get("returns") or [], t.get("parameters") or []
    numeric = {str(r.get("name")) for r in rets
               if str(r.get("type") or "").lower() in NUMERIC_TYPE
               or all_numeric(r.get("examples"))}
    # A return whose name promises a list and whose value is a number cannot
    # be repaired by deletion -- the name is the lie. Drop the tool.
    if any(COLLECTION_NAME.search(str(r.get("name") or "")) and
           str(r.get("name")) in numeric for r in rets):
        did["dropped:collection-field-holds-a-number"] += 1
        dropped_tools.append(t["name"])
        continue

    w = token_weights([str(r.get("name")) for r in rets] +
                      [str(p.get("name")) for p in params])
    keep = []
    for p in params:
        n = str(p.get("name") or "")
        d = str(p.get("description") or "")
        # A boolean that restates a return field: the call asserts what the
        # tool is being asked to report. "Er skroginspektionen afsluttet?" went
        # out as `is_hull_inspection_complete: false` -- the argument
        # presupposing the answer, and the answer reading it back.
        if str(p.get("type") or "").lower() == "boolean" and any(
                _same_subject(n, str(r.get("name") or ""), w) for r in rets):
            did["dropped-param:boolean-restates-a-return"] += 1
            continue
        is_num = str(p.get("type") or "").lower() in NUMERIC_TYPE \
            or all_numeric(p.get("examples"))
        # A bound whose subject matches no return is LEFT ALONE. Telling a
        # page size from a domain bound needs a vocabulary -- result, record,
        # row, page, limit -- and the next catalogue words it differently. The
        # runtime already refuses to clamp what it cannot target.
        # A numeric threshold with no direction anywhere: unhonourable too.
        if BOUND_ANY.search(n) and is_num \
                and not (BOUND_LO.search(n) or BOUND_HI.search(n)) \
                and bool(UPPER_PHRASE.search(d)) == bool(LOWER_PHRASE.search(d)):
            did["dropped-param:threshold-without-direction"] += 1
            continue
        keep.append(p)
    params = keep

    # A selector declaration is a claim that the parameter picks WHICH field
    # is reported. Where the values name content (a cheese, a substrate) or
    # are true/false, the claim is false: delete it and let the parameter be
    # an ordinary argument the user has to say.
    dec = dict(t.get("_selectors") or {})
    for pname in {k.split(chr(0))[0] for k in dec}:
        p = next((x for x in params if x.get("name") == pname), None)
        pairs = [(k.split(chr(0))[1], v) for k, v in dec.items()
                 if k.split(chr(0))[0] == pname]
        vals = {str(v).lower() for _f, v in pairs}
        # A RESOLVED selector is exempt from the lexical test. That test is a
        # sound check on an INVENTOR's claim -- if the options do not resemble
        # the fields, the claim was probably invented -- and exactly backwards
        # for a mapping that was READ: `liftstatus -> lifts_open_count` shares
        # no tokens because the options are Danish and the fields English, and
        # that disjointness is the whole reason the resolution pass exists.
        # Without this the repair drops all 668 resolved selectors and 235
        # tools go dead again, undoing the pass it runs after.
        resolved = pname in (t.get("_selectors_resolved") or ())
        bad = p is None or str(p.get("type") or "").lower() == "boolean" \
            or vals <= {"true", "false", "ja", "nej"} \
            or (not resolved
                and not any(set(_subject_tokens(str(v)))
                            & set(_subject_tokens(f)) for f, v in pairs))
        if bad:
            for k in [k for k in dec if k.split(chr(0))[0] == pname]:
                dec.pop(k)
            did["dropped-selector-claim"] += 1
    t = {**t, "_selectors": dec} if dec else {k: v for k, v in t.items()
                                              if k != "_selectors"}

    # Required, but nothing in a conversation has to supply it. Demoted, not
    # deleted: the generator may still send it, and the unspoken-optional
    # pass drops it when the user never said it.
    fixed = []
    for p in params:
        n = str(p.get("name") or "")
        # `_keep_required` is set by build_producer on a LOOKUP tool's single
        # input. Demoting it is precisely backwards: accepting the thing a
        # person can say -- a name, an address -- is the whole job of a lookup,
        # and the rule below reads exactly that as "unaskable". It demoted the
        # discriminating parameter on two of ten smoked producers, leaving
        # `floor_number` alone to identify a room and `country_code` alone to
        # identify a customs code.
        if p.get("required") and n in (t.get("_keep_required") or ()):
            fixed.append(p)
            continue
        if p.get("required") and not IDENTIFYING.search(n) \
                and not is_lookup_param(n) and not is_filter_param(n) \
                and not governs_field(p, t) \
                and str(p.get("type") or "").lower() != "boolean":
            p = {**p, "required": False}
            did["demoted-param:required-but-unaskable"] += 1
        fixed.append(p)

    # Deleting parameters can strip a tool of everything it required, leaving
    # an OPTIONAL subject: nothing then obliges a call to name the book the
    # user just asked about, and the answer credits it anyway. Whatever
    # identifies the subject goes back to required.
    if fixed and not any(p.get("required") for p in fixed):
        subj = next((p for p in fixed
                     if IDENTIFYING.search(str(p.get("name") or ""))), None)
        if subj is not None:
            fixed = [{**p, "required": True} if p is subj else p
                     for p in fixed]
            did["restored-required:subject-param"] += 1

    # The range each numeric field's OWN examples describe, written down so
    # the payload cannot invent a magnitude outside it -- 116% CPU on a field
    # whose examples run 5 to 96. No vocabulary: the schema says what the
    # field looks like, and the envelope is that, widened by its own spread.
    #
    # A part under its WHOLE is not declared here. Deriving it from examples
    # catches 7 of 15 real pairs and fires on 154 unrelated ones, and naming
    # it costs a word list -- capacity, kapacitet, rummer, belægning -- that
    # the next catalogue words differently. 15 tools in 1000 is not worth it.
    bounds = {}
    for r in rets:
        env = example_envelope(r)
        if env is not None:
            bounds[str(r.get("name"))] = env
            did["declared-bound:example-envelope"] += 1
    if bounds:
        t = {**t, "_bounds": bounds}

    out.append({**t, "parameters": fixed})

Path(sys.argv[2]).write_text("\n".join(
    json.dumps(t, ensure_ascii=False) for t in out) + "\n")
for k, v in did.most_common():
    print(f"  {k:46s} {v}")
print(f"\n{len(tools)} in -> {len(out)} out "
      f"({len(dropped_tools)} tools dropped)")
