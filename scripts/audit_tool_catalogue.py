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
    example_envelope, example_number, catalogue_faults,
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
    pnames = [str(p.get("name") or "") for p in params]

    # FAILs are defined once, in the generator: the same function gates
    # `--tools-from` and drives the repair, so a catalogue cannot pass the
    # audit and still be refused by the run that uses it.
    for why, field in catalogue_faults(t):
        hit(FAIL, why, t, field)

    # RISKs live here. They are shapes the runtime already handles, reported
    # so a growing catalogue can be watched rather than repaired.
    for r in rets:
        n = str(r.get("name") or "")
        # `_name` is an identifier the sibling-distinctness pass misses, so
        # two subjects in one row can come back with one name.
        if n.lower().endswith(("_name", "_navn")):
            hit(RISK, "name-return-outside-identifier-rules", t, n)
        # Picked as the answer field, this cannot survive "hvor mange".
        if NAMED_THING.search(n) and not TIME_FIELD.search(n):
            hit(RISK, "kind-ambiguous-answer-candidate", t, n)
        # `field_key` already refuses to key a count on an identifier of the
        # thing it counts, so the class size stops moving with which pupil
        # you name.
        for pn in pnames:
            if IDENTIFYING.search(pn) and _aggregates_over(n, pn):
                hit(RISK, "aggregate-shares-subject-with-an-id", t, n)

    # A date filter the payload repair cannot honour: dates are not numbers,
    # so `respect_constraints` leaves them alone.
    for p in params:
        if DATE_VAL.search(str(_unquote((p.get("examples") or [""])[0]))) \
                and re.search(r"(filter|from|efter|since|before|til|until|"
                              r"start|end)", str(p.get("name")), re.I) \
                and any(DATE_VAL.search(str(_unquote(
                    (r.get("examples") or [""])[0]))) for r in rets):
            hit(RISK, "date-filter-over-a-date-return", t, str(p.get("name")))

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
