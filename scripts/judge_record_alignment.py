"""Are a tool's example lists one record table, or several unrelated ones?

CALIBRATION, not a repair. It answers one question with a number: of the
tools where the generator has to guess the pairing positionally, how many are
merely MIS-ORDERED (a permutation fixes them) versus about DIFFERENT RECORDS
(only a rewrite fixes them). That decides which repair is worth buying.

The population is the one measured off tools_v6: a tool with a subject
parameter and at least one return field whose value is CHOSEN FROM its
examples, where no return example shares a discriminating token with any
subject example. Numeric return fields are excluded -- their values come from
the band the examples imply, so the order of the list never mattered.

PLANTED CONTROLS. A judge's rate is not reportable until the judge is shown
to be right about cases whose answer is known, so every run mixes in three
HAND-BUILT tools -- known by construction, never sampled from the catalogue:

    ALIGNED    already a record table, untouched       -> A (identity) or C
    SHUFFLED   the same, first return list rotated     -> A
    FOREIGN    the same, first list swapped for one
               sharing no token, place or domain       -> B

Both earlier designs failed here and both were the control's fault, not the
judge's. Sampling controls from the content-decided bucket assumed the
heuristic under test: `PROJ-2023-101` "matches" `TP-2023-005` on the year
alone. Then donating one control's list to another was not foreign enough --
a list of Danish towns containing Aarhus pairs fine with a municipality tool.

Controls are indistinguishable from real items in the prompt and are scored
separately. If they do not pass, the headline number is not reported.

    uv run --no-project --with aiohttp python \\
        scripts/judge_record_alignment.py --tools data/tool_calls/tools_v6.jsonl \\
          --n 40 --out scratch/record_alignment.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_tool_dialogues_da as D  # noqa: E402
import gen_tool_dialogues_proc as G  # noqa: E402
from gen_tool_dialogues_da import FatalAPIError, _ask, _hash, _key  # noqa: E402

SYS = """Du vurderer ét værktøjs EKSEMPLER.

Værktøjet slår en post op ud fra det, brugeren siger. Parameteren har 2-3
eksempler på, hvad brugeren kan sige. Hvert returfelt har 2-3 eksempler på,
hvad værktøjet svarer. Meningen er, at eksempel nr. 1 i hver liste hører til
den SAMME post -- men forfatteren skrev listerne hver for sig, så det passer
ikke altid.

Spørgsmålet er KUN dette: kan eksemplerne parres sammen til hele poster ved at
bytte om på rækkefølgen, eller skal der skrives nye værdier?

A = JA, en omrokering er nok. Værdierne hører til de samme 2-3 poster; de står
    bare i forskellig orden. Angiv i `order` hvilket parameter-eksempel
    (0-baseret) hvert eksempel i FØRSTE returfelt hører til.
    Bemærk: et returfelt må gerne indeholde en HELT ANDEN slags værdi end
    parameteren -- et projekt har et tilladelsesnummer, en bygning har en
    ejer. Det er stadig A, hvis værdierne kan høre til de samme poster.

B = NEJ, værdierne skal skrives om. Det er nok, at ÉT enkelt returfelt ikke
    kan parres -- de øvrige felter redder ikke posten. Feltet handler om nogle
    andre ting end parameterens -- fx nævner parameteren tre steder i Jylland og
    feltet tre steder på Sjælland, eller listerne har forskellig længde, så
    ingen parring kan gå op.

C = RÆKKEFØLGEN ER LIGEGYLDIG. Returfelterne er frit udskiftelige egenskaber
    (en status, en farve, et materiale), hvor enhver parring er lige rigtig.

Kan du ikke pege på en konkret uforenelighed, så er svaret A eller C -- ikke B.
`begrundelse`: én kort sætning."""

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["svar", "order", "begrundelse"],
          "properties": {
              "svar": {"type": "string", "enum": ["A", "B", "C"]},
              "order": {"type": "array", "items": {"type": "integer"}},
              "begrundelse": {"type": "string"}}}


def example_selected(field, ex):
    """Does `_value` pick this field's value FROM its examples?"""
    if not ex:
        return False
    if any(re.match(r"^\s*-?\d+[.,]?\d*\s*[-/:,;]", str(e)) for e in ex):
        return True
    kind = D.classify(ex)
    if D._numeric_span(ex, field) and kind in ("int", "float", "numunit"):
        return False
    return kind in ("free", "numunit") or not kind


def profile(t):
    """(subject param, its examples, [(return name, examples)]) or None."""
    sub = None
    for p in (t.get("parameters") or []):
        if not p.get("required") or G.governs_field(p, t):
            continue
        ex = [D._unquote(e) for e in (p.get("examples") or []) if str(e).strip()]
        if len(ex) >= 2:
            sub = (p, ex)
            break
    if sub is None:
        return None
    p, sex = sub
    fields = []
    for r in (t.get("returns") or []):
        rex = [D._unquote(e) for e in (r.get("examples") or []) if str(e).strip()]
        if len(rex) >= 2 and example_selected(r["name"], rex):
            fields.append((r["name"], rex))
    return (p, sex, fields) if fields else None


def bucket(t):
    """'risky' | 'decided' | None -- the buckets measured off tools_v6."""
    pr = profile(t)
    if pr is None:
        return None
    _p, sex, fields = pr
    dec = sum(1 for _n, rex in fields
              if any(D._content_pick(rex, s) for s in sex))
    if dec == len(fields):
        return "decided"
    return "risky" if dec == 0 else None


def item_text(name, sex, fields, pname):
    lines = [f"værktøj: {name}", f"  BRUGEREN SIGER ({pname}):"]
    lines += [f"    [{i}] {e}" for i, e in enumerate(sex)]
    for n, rex in fields:
        lines.append(f"  VÆRKTØJET SVARER ({n}):")
        lines += [f"    [{i}] {e}" for i, e in enumerate(rex)]
    return "\n".join(lines)


# SYNTHETIC, not sampled. The first version drew controls from the
# content-decided bucket and called them aligned, which assumes the very
# heuristic under test is right: `PROJ-2023-101` "matches" `TP-2023-005` on
# the year alone, so two of three control tools were not known-aligned at all
# and the judge was scored wrong for disagreeing with them. A control has to
# be right BY CONSTRUCTION or it measures nothing.
CONTROL_CASES = [
    ("kommune_lookup", "bynavn", ["Aarhus", "Odense", "Aalborg"],
     [("kommune_navn", ["Aarhus Kommune", "Odense Kommune", "Aalborg Kommune"]),
      ("region_navn", ["Midtjylland", "Syddanmark", "Nordjylland"])]),
    ("museum_lookup", "museumsnavn",
     ["Louisiana", "ARoS", "Statens Museum for Kunst"],
     [("by", ["Humlebæk", "Aarhus", "København"]),
      ("grundlagt_aar", ["1958", "2004", "1896"])]),
    ("bro_lookup", "bronavn",
     ["Storebæltsbroen", "Øresundsbroen", "Lillebæltsbroen"],
     [("forbinder", ["Sjælland og Fyn", "Danmark og Sverige",
                     "Fyn og Jylland"]),
      ("laengde_m", ["6790", "7845", "1700"])]),
]


# Shares no token, no domain and no place with any control case.
FOREIGN_DONOR = ["Natriumklorid", "Kaliumpermanganat", "Eddikesyre"]


def make_controls(_decided=None, k=3):
    """(item_text, expected, tag) for cases whose answer is known by design."""
    out = []
    for i, (name, pname, sex, fields) in enumerate(CONTROL_CASES[:k]):
        # ALIGNED: already a table, so re-ordering is a no-op -- A with the
        # identity order and C are both defensible. B is not.
        out.append((item_text(name, sex, fields, pname), "A|C", "ALIGNED"))
        # SHUFFLED: same records, first list rotated. Only A.
        n0, r0 = fields[0]
        out.append((item_text(name, sex, [(n0, r0[1:] + r0[:1])] + fields[1:],
                              pname), "A", "SHUFFLED"))
        # FOREIGN: first list replaced by one that shares NOTHING. Only B.
        # Donating one control case's list to another was not foreign enough
        # -- handing the municipality tool a list of Danish towns containing
        # Aarhus, the judge paired them and was right to. The donor has to be
        # from a domain with no overlap at all.
        out.append((item_text(name, sex, [(n0, FOREIGN_DONOR)] + fields[1:],
                              pname), "B", "FOREIGN"))
    return out


async def run(a):
    import aiohttp
    tools = [json.loads(l) for l in a.tools.open() if l.strip()]
    risky = [t for t in tools if bucket(t) == "risky"]
    decided = [t for t in tools if bucket(t) == "decided"]
    print(f"{len(tools):,} tools   {len(risky):,} risky   "
          f"{len(decided):,} content-decided (control source)", flush=True)
    # deterministic spread, not the first N: the catalogue is ordered by
    # scenario and the head is all one domain.
    step = max(1, len(risky) // a.n)
    sample = risky[::step][:a.n]
    controls = make_controls(decided, a.controls)
    items = [(item_text(t["name"], *[profile(t)[1], profile(t)[2]],
                        profile(t)[0]["name"]), None, t["name"])
             for t in sample] + controls
    print(f"judging {len(sample)} real + {len(controls)} controls", flush=True)

    sem = asyncio.Semaphore(a.concurrency)
    tok = Counter()

    async def one(session, text):
        async with sem:
            r = await _ask(session, SYS, text, SCHEMA, "alignment",
                           temp=0.0, max_tokens=400)
            if r is None:
                return None
            v, u = r
            tok["in"] += u.get("prompt_tokens", 0)
            tok["out"] += u.get("completion_tokens", 0)
            return v

    async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {_key()}"}) as s:
        res = await asyncio.gather(*[one(s, t) for t, _e, _n in items])

    ctl, real = Counter(), Counter()
    ctl_fail = []
    for (text, exp, tag), v in zip(items, res):
        if v is None:
            (ctl if exp else real)["no-answer"] += 1
            continue
        if exp:
            ok = v["svar"] in exp.split("|")
            ctl[f"{tag}:{'pass' if ok else 'FAIL'}"] += 1
            if not ok:
                ctl_fail.append((tag, exp, v["svar"], v["begrundelse"][:90]))
        else:
            real[v["svar"]] += 1

    print("\ncontrols:")
    for k, v in sorted(ctl.items()):
        print(f"  {v:>3}  {k}")
    for t, e, g, w in ctl_fail:
        print(f"     {t}: expected {e}, got {g} -- {w}")
    passed = sum(v for k, v in ctl.items() if k.endswith(":pass"))
    total = sum(ctl.values())

    print(f"\nreal sample (n={sum(real.values())}):")
    lab = {"A": "A  same records, wrong order   -> permutation repair",
           "B": "B  different records           -> record rewrite",
           "C": "C  pairing does not matter     -> nothing to fix"}
    for k in ("A", "B", "C", "no-answer"):
        if real.get(k):
            print(f"  {real[k]:>3}  {real[k]/sum(real.values()):5.1%}  "
                  f"{lab.get(k, k)}")
    cost = tok['in'] / 1e6 * 0.10 + tok['out'] / 1e6 * 0.40
    print(f"\ntokens in={tok['in']:,} out={tok['out']:,}  ~${cost:.4f}")
    if total and passed != total:
        print("\nCONTROLS DID NOT PASS -- the sample rate above is not "
              "reportable; fix the judge before buying a repair.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"controls": dict(ctl), "sample": dict(real),
         "risky_total": len(risky), "cost_usd": round(cost, 4),
         "verdicts": [{"tool": n, "svar": (v or {}).get("svar"),
                       "order": (v or {}).get("order"),
                       "why": (v or {}).get("begrundelse")}
                      for (t, e, n), v in zip(items, res) if e is None]},
        ensure_ascii=False, indent=1))
    print(f"-> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", type=Path, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--controls", type=int, default=3, help="of EACH kind")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--out", type=Path,
                    default=Path("scratch/record_alignment.json"))
    a = ap.parse_args()
    try:
        asyncio.run(run(a))
    except FatalAPIError as e:
        raise SystemExit(f"ABORTED: {e}")


if __name__ == "__main__":
    main()
