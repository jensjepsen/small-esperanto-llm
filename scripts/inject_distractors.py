"""Put the declared distractor fields into the payloads, with per-row values.

Stage 5.6, after the answers and before the twins.

WHAT THIS FIXES. `gen_return_distractors` extends the `returns` CONTRACT and
the renderer prints it into the catalogue, so by this point every affected
tool advertises fields its payload does not carry. That is the exact
inconsistency the pipeline gates elsewhere (`result-missing-fields`), and
7.5% of v8 rows already trip a version of it. This pass closes the gap.

WHY A SEPARATE PASS, NOT PART OF STAGE 5. Stage 5 only touches DANGLING calls
-- the 67% it answers itself. The other third carry a tool_result straight
from the source and never pass through its splice, so an injection living
there would leave a third of the corpus inconsistent. Every payload needs the
field; only some need an answer.

VALUES ARE PER ROW, NEVER PER SCHEMA. A field that holds the same value in
every row of a tool is the constant beside a varying answer, separable without
reading a description -- a fresh shortcut in place of the one being removed.
`toolmind_distractor_values.generate` derives the value from (row index, field
name) so 400 rows give 400 values; free-text fields draw from a sampled bank
by the same hash.

THE PROPERTY BEING BOUGHT, AND HOW IT IS KEPT. A distractor is only a
distractor if the gold answer does NOT cite it. That holds by construction --
the answers were written before these fields existed and nothing regenerates
them -- with one hole: a synthesised value could coincide with a number the
answer already uses. So every value is checked against the answer text and
resampled on collision, and against the payload's own values, since a
duplicate is not a distractor either. Failures are counted, never silently
shipped.

    uv run python scripts/inject_distractors.py \
        --rows $OUT/sft_v9_answered.jsonl \
        --map scratch/distractors/map.jsonl \
        --banks scratch/distractors/banks.jsonl \
        --out $OUT/sft_v9_dist.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from toolmind_distractor_values import FREE, classify, generate  # noqa: E402

SEP = "Værktøjer:"
DEC = json.JSONDecoder()
MAX_TRIES = 12


def signature(fn) -> str:
    props = ((fn.get("parameters") or {}).get("properties") or {})
    return f"{fn.get('name')}({','.join(sorted(props))})"


def load_map(path: Path):
    out = {}
    for line in path.open():
        if line.strip():
            r = json.loads(line)
            out[r["sig"]] = r["fields"]
    return out


def load_banks(path: Path | None):
    if not path or not path.exists():
        return {}
    out = {}
    for line in path.open():
        if line.strip():
            r = json.loads(line)
            out[(r["felt"], r["beskrivelse"].strip().lower())] = r["vaerdier"]
    return out


def _render(v):
    """How a value would appear in Danish prose, for the collision check."""
    if isinstance(v, float) and v.is_integer():
        return {str(v), str(int(v)), str(v).replace(".", ",")}
    return {str(v), str(v).replace(".", ",")}


def pick(field, exemplars, banks, desc, idx, answer, taken):
    """A value for this row that collides with neither the answer nor the payload.

    Returns (value, why) -- why is None on success. Resampling walks the row
    index rather than the field name so the value stays a pure function of the
    row and reruns reproduce it.
    """
    kind = classify(exemplars)
    bank = banks.get((field, (desc or "").strip().lower())) if kind == FREE else None
    if kind == FREE and not bank:
        return None, "free-no-bank"
    low = (answer or "").lower()
    for t in range(MAX_TRIES):
        if bank:
            from toolmind_distractor_values import _rand
            v = bank[int(_rand(idx + t * 7919, field, "bank") * len(bank))
                     % len(bank)]
        else:
            v = generate(field, exemplars, idx + t * 7919, kind)
        if v is None:
            return None, "no-value"
        forms = _render(v)
        if any(f.lower() in low for f in forms):
            continue                       # the answer already says this
        if any(f in taken for f in forms):
            continue                       # the payload already holds it
        return v, None
    return None, "collision-exhausted"


def inject_row(row, dmap, banks, why):
    """Add every declared-but-absent distractor to this row's payloads."""
    msgs = row["messages"]
    head = msgs[0]["content"]
    if SEP not in head:
        why["no-catalogue"] += 1
        return False
    try:
        cat, rest = DEC.raw_decode(head.split(SEP, 1)[1].lstrip())
    except Exception:
        why["catalogue-unparseable"] += 1
        return False
    by_name = {t["name"]: t for t in cat if "name" in t}
    idx = row.get("idx", 0)
    changed = False

    for i, m in enumerate(msgs):
        if m["role"] != "tool_result":
            continue
        call = None
        for j in range(i - 1, -1, -1):
            if msgs[j]["role"] == "tool_call":
                try:
                    call = json.loads(msgs[j]["content"])
                except Exception:
                    pass
                break
        if not call:
            why["result-without-call"] += 1
            continue
        fn = by_name.get(call.get("name"))
        if not fn:
            why["tool-not-in-catalogue"] += 1
            continue
        fields = dmap.get(signature(fn))
        if not fields:
            continue
        try:
            payload = json.loads(m["content"])
        except Exception:
            why["payload-unparseable"] += 1
            continue
        if not isinstance(payload, dict):
            why["payload-not-object"] += 1
            continue
        answer = (msgs[i + 1]["content"]
                  if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant"
                  else "")
        taken = {f for v in payload.values() for f in _render(v)}
        declared = ((fn.get("returns") or {}).get("properties") or {})
        for f in fields:
            name = f["felt"]
            if name in payload:
                why["already-present"] += 1
                continue
            # ONLY WHAT THE CATALOGUE DECLARES. The contract is merged into the
            # returns map upstream, but ~70 signatures do not pick it up at
            # --annotate, and injecting there would put a key in the payload
            # that the catalogue the model reads never mentions -- the exact
            # inconsistency this stage exists to close. 450 such keys in the
            # first full run.
            if name not in declared:
                why["not-declared-skipped"] += 1
                continue
            v, bad = pick(name, f.get("eksempler"), banks,
                          f.get("beskrivelse"), idx + i, answer, taken)
            if bad:
                why[bad] += 1
                continue
            payload[name] = v
            taken |= _render(v)
            why["injected"] += 1
            changed = True
        m["content"] = json.dumps(payload, ensure_ascii=False)
    return changed


# --- planted controls --------------------------------------------------------
def check():
    """Each guarantee, asserted on a row built to break it."""
    dmap = {"t(a)": [{"felt": "floor", "type": "integer",
                      "beskrivelse": "Etagen",
                      "eksempler": ["2", "3", "5", "11"]}]}
    def row(answer, payload, idx=1, declare=("floor", "author")):
        # The catalogue DECLARES the distractor, as it does after stage 2.5 is
        # merged and --annotate has run. A fixture that does not declare it
        # tests nothing, since injection now (correctly) skips undeclared keys.
        cat = [{"name": "t", "parameters": {"type": "object",
                                            "properties": {"a": {}}},
                "returns": {"type": "object", "properties": {
                    d: {"description": d} for d in declare}}}]
        return {"idx": idx, "messages": [
            {"role": "user", "content": f"{SEP}\n{json.dumps(cat)}\n\nq"},
            {"role": "tool_call", "content": json.dumps({"name": "t"})},
            {"role": "tool_result", "content": json.dumps(payload)},
            {"role": "assistant", "content": answer}]}

    why = Counter()
    r = row("Der er 8 kopper.", {"cups_left": 8})
    inject_row(r, dmap, {}, why)
    p = json.loads(r["messages"][2]["content"])
    assert "floor" in p, "distractor not injected"
    assert p["cups_left"] == 8, "existing value disturbed"

    # the injected value must never be one the answer already states
    for i in range(300):
        r = row(f"Der er {i % 40} kopper.", {"cups_left": 999}, idx=i)
        inject_row(r, dmap, {}, Counter())
        p = json.loads(r["messages"][2]["content"])
        if "floor" in p:
            assert str(p["floor"]) not in f"Der er {i % 40} kopper.", (
                f"row {i}: injected {p['floor']} which the answer cites")

    # nor one the payload already holds
    for i in range(300):
        r = row("Ingen tal her.", {"cups_left": 7}, idx=i)
        inject_row(r, dmap, {}, Counter())
        p = json.loads(r["messages"][2]["content"])
        if "floor" in p:
            assert p["floor"] != 7, f"row {i}: duplicated an existing value"

    # per-row variety, the whole point
    vals = set()
    for i in range(400):
        r = row("Ingen tal.", {"cups_left": 999}, idx=i)
        inject_row(r, dmap, {}, Counter())
        vals.add(json.loads(r["messages"][2]["content"]).get("floor"))
    assert len(vals) >= 10, f"only {len(vals)} distinct values across 400 rows"

    # determinism
    a, b = row("q", {"x": 1}, idx=42), row("q", {"x": 1}, idx=42)
    inject_row(a, dmap, {}, Counter()); inject_row(b, dmap, {}, Counter())
    assert a["messages"][2]["content"] == b["messages"][2]["content"]

    # a free-text field with no bank must be SKIPPED, not invented
    free = {"t(a)": [{"felt": "author", "type": "string",
                      "beskrivelse": "Forfatter",
                      "eksempler": ["Mahatma Gandhi", "Søren Kierkegaard"]}]}
    w = Counter()
    r = row("Ingen navne.", {"quote": "x"})
    inject_row(r, free, {}, w)
    assert "author" not in json.loads(r["messages"][2]["content"])
    assert w["free-no-bank"] == 1, dict(w)

    # a field the CATALOGUE does not declare must never reach the payload
    w = Counter()
    r = row("Ingen tal.", {"cups_left": 8}, declare=())
    inject_row(r, dmap, {}, w)
    assert "floor" not in json.loads(r["messages"][2]["content"]), \
        "injected a key the catalogue never declared"
    assert w["not-declared-skipped"] == 1, dict(w)

    # ...and USED when a bank exists
    w = Counter()
    r = row("Ingen navne.", {"quote": "x"})
    inject_row(r, free, {("author", "forfatter"): [f"Forfatter {i}"
                                                   for i in range(50)]}, w)
    assert "author" in json.loads(r["messages"][2]["content"]), dict(w)
    print(f"inject: {len(vals)} distinct values/400 rows, non-citation + "
          f"non-duplication + determinism + bank routing asserted", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path)
    ap.add_argument("--map", type=Path)
    ap.add_argument("--banks", type=Path, default=None)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check or not args.rows:
        check()
        if not args.rows:
            return
    check()
    dmap, banks = load_map(args.map), load_banks(args.banks)
    print(f"{len(dmap):,} signatures with distractors, "
          f"{len(banks):,} value banks", flush=True)
    rows = [json.loads(l) for l in args.rows.open() if l.strip()]
    why = Counter()
    touched = 0
    for r in rows:
        if inject_row(r, dmap, banks, why):
            touched += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{touched:,}/{len(rows):,} rows changed   {dict(why)}")
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
