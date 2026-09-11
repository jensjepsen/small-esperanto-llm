"""Audit a tool catalogue against the schema defects that reach dialogues.

Each check is named for the DIALOGUE defect it causes downstream. FAIL means
the schema itself is wrong and every dialogue built on it inherits the error;
RISK means the schema is a shape some gate has to keep watching.

The tool set used to be reinvented from scratch on every run, so every run
met a fresh sample of these and the same dialogue defects kept coming back.
Audit once, repair once, freeze.

Usage:
  uv run --no-project --with aiohttp --with langdetect python \
      scripts/audit_tool_catalogue.py TOOLS.jsonl [failing_names.txt]
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tool_dialogues_proc import (
    BOUND_ANY, BOUND_HI, BOUND_LO, COLLECTION_NAME, IDENTIFYING,
    LOWER_PHRASE, NAMED_THING, NUMERIC_TYPE, TIME_FIELD, UPPER_PHRASE,
    _aggregates_over, _as_number, _same_subject, _subject_tokens, _unquote,
    governs_field, is_filter_param, is_lookup_param, token_weights,
    example_envelope, example_number,
)

DATE_VAL = re.compile(r"^\d{4}-\d{2}-\d{2}")
FAIL, RISK = "FAIL", "RISK"

tools = [json.loads(x) for x in Path(sys.argv[1]).open() if x.strip()]
hits, kind = defaultdict(list), {}


def hit(sev, why, t, f):
    kind[why] = sev
    hits[why].append(f"{t['name']}.{f}")


def all_numeric(examples):
    return bool(examples) and all(example_number(e) is not None
                                  for e in examples)


for t in tools:
    rets, params = t.get("returns") or [], t.get("parameters") or []
    rnames = [str(r.get("name") or "") for r in rets]
    pnames = [str(p.get("name") or "") for p in params]
    numeric = {str(r.get("name")) for r in rets
               if str(r.get("type") or "").lower() in NUMERIC_TYPE
               or all_numeric(r.get("examples"))}
    w = token_weights(rnames + pnames)

    for r in rets:
        n = str(r.get("name") or "")
        # The name promises a list, the value is a scalar: the question
        # written from it asks for a measurement and gets a count.
        if COLLECTION_NAME.search(n) and n in numeric:
            hit(FAIL, "collection-field-holds-a-number", t, n)
        # `_name` is an identifier the sibling-distinctness pass misses, so
        # two subjects in one row can come back with one name.
        if n.lower().endswith(("_name", "_navn")):
            hit(RISK, "name-return-outside-identifier-rules", t, n)
        # Picked as the answer field, this cannot survive "hvor mange".
        if NAMED_THING.search(n) and not TIME_FIELD.search(n):
            hit(RISK, "kind-ambiguous-answer-candidate", t, n)
        # A shape, not an error: `field_key` already refuses to key a count on
        # an identifier of the thing it counts, so the class size stops moving
        # with which pupil you name.
        for p in pnames:
            if IDENTIFYING.search(p) and _aggregates_over(n, p):
                hit(RISK, "aggregate-shares-subject-with-an-id", t, n)

    # Every numeric field whose examples describe a range must carry it, so
    # the payload cannot invent a magnitude the tool never claimed.
    declared = t.get("_bounds") or {}
    for r in rets:
        if example_envelope(r) is not None \
                and str(r.get("name")) not in declared:
            hit(FAIL, "numeric-return-without-its-envelope", t,
                str(r.get("name")))

    # Nothing in the conversation is obliged to name the subject, so a row can
    # ask about "Den store Gatsby" and call the tool with no arguments at all.
    subject = [p for p in params if IDENTIFYING.search(str(p.get("name")))]
    if subject and not any(p.get("required") for p in params):
        hit(FAIL, "subject-param-but-nothing-required", t,
            str(subject[0].get("name")))

    for p in params:
        n = str(p.get("name") or "")
        d = str(p.get("description") or "")
        is_num = str(p.get("type") or "").lower() in NUMERIC_TYPE \
            or all_numeric(p.get("examples"))
        # The call asserting what the tool is asked to report.
        if str(p.get("type") or "").lower() == "boolean" and any(
                _same_subject(n, str(r.get("name") or ""), w) for r in rets):
            hit(FAIL, "boolean-param-restates-a-return", t, n)
        # A numeric threshold whose direction is nowhere stated: the payload
        # cannot be made to obey it and the answer contradicts the question.
        if BOUND_ANY.search(n) and is_num \
                and not (BOUND_LO.search(n) or BOUND_HI.search(n)) \
                and bool(UPPER_PHRASE.search(d)) == bool(LOWER_PHRASE.search(d)):
            hit(FAIL, "threshold-without-direction", t, n)
        # Required, but nothing in a conversation has to supply it: not an
        # identifier the user names, not a selector code picks. It gets
        # invented -- `cheese_type: "Rococo"` under a question about a batch.
        if p.get("required") and not IDENTIFYING.search(n) \
                and not is_lookup_param(n) and not is_filter_param(n) \
                and not governs_field(p, t) \
                and str(p.get("type") or "").lower() != "boolean":
            hit(FAIL, "required-param-is-neither-id-nor-selector", t, n)
        if DATE_VAL.search(str(_unquote((p.get("examples") or [""])[0]))) \
                and re.search(r"(filter|from|efter|since|before|til|until|"
                              r"start|end)", n, re.I) \
                and any(DATE_VAL.search(str(_unquote(
                    (r.get("examples") or [""])[0]))) for r in rets):
            hit(RISK, "date-filter-over-a-date-return", t, n)

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
            # true/false cannot select WHICH of three fields is meant.
            hit(FAIL, "boolean-declared-as-a-selector", t, pname)
        elif not any(set(_subject_tokens(str(v))) & set(_subject_tokens(f))
                     for f, v in pairs):
            # Values name content, not fields: a substrate or a cheese.
            hit(FAIL, "selector-values-name-content", t, pname)

n = len(tools)
failed, risky = set(), set()
print(f"{n} tools audited\n")
for k, v in sorted(hits.items(), key=lambda x: (kind[x[0]] != FAIL, -len(x[1]))):
    tl = {x.split(".")[0] for x in v}
    (failed if kind[k] == FAIL else risky).update(tl)
    print(f"  [{kind[k]}] {k:40s} {len(v):5d} hits  {len(tl):4d} tools "
          f"({100*len(tl)/n:4.1f}%)")
    for s in v[:2]:
        print(f"           {s}")
print(f"\n  tools with a FAIL: {len(failed):4d} ({100*len(failed)/n:.1f}%)")
print(f"  clean of FAILs:    {n-len(failed):4d} ({100*(n-len(failed))/n:.1f}%)")
print(f"  clean of FAIL+RISK:{n-len(failed | risky):4d} "
      f"({100*(n-len(failed|risky))/n:.1f}%)")
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text("\n".join(sorted(failed)) + "\n")
    print(f"\n-> {sys.argv[2]} ({len(failed)} failing tool names)")
