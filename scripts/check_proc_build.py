"""Every structural check this session established, over one build.

Run after any proc build. Each number has a reference value measured on the
smokes, so a drift shows up as a drift rather than as a number to squint at.

    python scripts/check_proc_build.py data/tool_calls/proc_v2
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_tool_dialogues_proc as G  # noqa: E402
from gen_tool_dialogues_da import _ex_tokens  # noqa: E402

build = Path(sys.argv[1])
rows = [json.loads(l) for l in (build / "translated.jsonl").open() if l.strip()]
_tl = [json.loads(l) for l in (build / "tools.jsonl").open() if l.strip()]
# BY (SCENARIO, NAME). 888 tools in the catalogue share 370 names on purpose,
# so a name-keyed index grades a row against a DIFFERENT tool that happens to
# share its name: `fishing_logistics` in one scenario was checked against the
# other one's returns, and its answer field read as ungoverned.
tools = {t["name"]: t for t in _tl}
tools_sn = {(t.get("_scenario"), t["name"]): t for t in _tl}
n = len(rows)
print(f"{build}   {n:,} rows   {len(tools):,} tools\n")

# ── 1. every gate, over what actually shipped ────────────────────────────
def row_tool(r, name=None):
    """The tool AS THIS ROW USED IT -- roles are rotated per row.

    `gate_answers` reads `tool["answer_field"]`, but ~55% of rows rotate the
    answer/competitor pair. Gating a rotated row against the tool's default
    reads its correct answer as citing the competitor: 337 of 763 rows in the
    first stage-1 sweep, all of them false. `review_generated.py` takes
    `--roles` for exactly this reason and I rebuilt the same bug here.
    """
    sc = str(r.get("_key") or "").split("#")[0] or None
    nm = name or r["_tool"]
    t = tools_sn.get((sc, nm)) or tools.get(nm)
    if not t:
        return None
    if r.get("_answer_field"):
        t = {**t, "answer_field": r["_answer_field"],
             "competitor_field": r.get("_competitor_field")
             or t.get("competitor_field")}
    return t


gates = Counter()
for r in rows:
    t = row_tool(r)
    if not t:
        gates["no-tool-in-build"] += 1
        continue
    why = (G.gate_prose(r, t) or G.gate_kind(r, t) or G.gate_answers(r, t)
           or G.gate_error_answers(r, t) or G.gate_pairing(r, t))
    gates[why or "clean"] += 1
print("GATES (expect all clean -- a shipped row failing one is a gate bug)")
for k, v in gates.most_common():
    print(f"  {v:>6}  {k}")

# ── 2. payload coherence ─────────────────────────────────────────────────
echo = contra = 0
echo_fields = Counter()
for r in rows:
    msgs = r["da"]["conversations"]
    pairs = []
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = [c["function"] for c in m["tool_calls"]]
            res = [json.loads(x["content"])
                   for x in msgs[i + 1:i + 1 + len(calls)] if x["role"] == "tool"]
            pairs += list(zip(calls, res))
    for call, p in pairs:
        if not isinstance(p, dict):
            continue
        for an, av in (call.get("arguments") or {}).items():
            if not isinstance(av, str) or len(av) < 3:
                continue
            for rn, rv in p.items():
                if isinstance(rv, str) and rv.strip() == av.strip() and rn != an:
                    echo += 1
                    echo_fields[f"{an} -> {rn}"] += 1
    for a in range(len(pairs) - 1):
        prod, cons = pairs[a][1], pairs[a + 1][1]
        if not (isinstance(prod, dict) and isinstance(cons, dict)):
            continue
        # A LINK IS A HANDLE, and handles are not bare small integers. Taking
        # any shared value as evidence of a chain read a parallel row about
        # two hospital departments as a producer/consumer pair, because
        # `time_window_hours: 4` happened to equal `transport_requests_pending:
        # 4` in the first payload -- and then called their different bed counts
        # a contradiction, when two departments simply have different beds.
        def handles(vals):
            return {str(v).strip() for v in vals
                    if isinstance(v, str) and len(str(v).strip()) >= 3}
        sent = handles((pairs[a + 1][0].get("arguments") or {}).values())
        if not (sent & handles(prod.values())):
            continue
        for k in [k for k in prod if k in cons]:
            if str(prod[k]).strip() != str(cons[k]).strip():
                contra += 1
print(f"\nPAYLOADS")
print(f"  {contra:>6}  chain self-contradictions        (smokes: 0)")
print(f"  {echo:>6}  arg echoed into another field     (smokes: all legitimate handle->id)")
for k, v in echo_fields.most_common(8):
    print(f"            {v:>4}  {k}")

# ── 3. the selector rule ─────────────────────────────────────────────────
sel_n = sel_omit = 0
for r in rows:
    af, t = r.get("_answer_field"), row_tool(r)
    if not t or not af:
        continue
    # DECLARED mappings only. `_selector_for` falls back to subject scoring,
    # which matched `occupied_spaces_total` to the `available_spaces` option
    # and then demanded a selector the contract never promised for it.
    decl = t.get("_selectors") or {}
    sel = next((p["name"] for p in (t.get("parameters") or [])
                if G.governs_field(p, t)
                and f"{p['name']}\x00{af}" in decl), None)
    if not sel:
        continue
    for m in r["da"]["conversations"]:
        for c in (m.get("tool_calls") or []):
            if c["function"].get("name") == t["name"]:
                sel_n += 1
                sel_omit += sel not in (c["function"].get("arguments") or {})
print(f"\nSELECTOR  {sel_omit}/{sel_n} omitted   (was 15.7% pre-fix, 0/45 on smokes)")

# ── 4. plan mix and error density ────────────────────────────────────────
plan = Counter(r["_plan"] for r in rows)
err = Counter(r["_plan"] for r in rows
              if any('"error"' in (m.get("content") or "")
                     for m in r["da"]["conversations"]))
print(f"\nPLANS")
for k, v in plan.most_common():
    print(f"  {v:>6}  {v/n:6.2%}  {k}")
print(f"\nERRORS   {sum(err.values())}/{n} = {sum(err.values())/n:.2%}   "
      f"(calibration predicts ~3.4%)")
for k, v in err.most_common():
    print(f"  {v:>6}  {k}")

# planned vs shipped, the sensitive within-build yield
cat = [json.loads(l) for l in open("data/tool_calls/tools_v6.jsonl")]
fam = G.family_index(cat)
sub = [t for t in cat if t["name"] in tools]
rng = random.Random(0)
planned = Counter()
for t in sub:
    f = fam.get(t.get("_scenario")) or [t]
    for idx in range(4):
        planned[G.pick_plan(t, idx, rng, f)] += 1
for p in ("error_report", "error_recover"):
    if planned[p]:
        print(f"  {p:<14} planned {planned[p]:>4}  shipped {err[p]:>4}  "
              f"yield {err[p]/planned[p]:.2f}")
