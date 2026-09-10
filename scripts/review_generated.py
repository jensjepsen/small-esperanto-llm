"""Render a tool-dialogue corpus as a flat file for reading in `less`.

Every defect this pipeline has ever shipped was found by READING rows, not by
a metric -- seven passes, seven new shapes. This exists so that reading is
cheap: full text, no truncation, flags on the row header AND marked inline on
the turn that caused them.

    uv run python scripts/review_generated.py --rows scratch/gen_ub2/sft.jsonl \
        --tools scratch/gen_ub2/tools.jsonl --out scratch/gen_ub2/review.txt
    less scratch/gen_ub2/review.txt      #  /^!!  jumps to flagged rows

Detectors are the accumulated battery. They are NOT the point -- they mark
what is already known so your eyes are free for what is not.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# NUM is IMPORTED, never redefined: a local `-?\d+` here read the hyphens in
# `Na8-10Al6Si6O24S2-4` as minus signs while `_nums` read them as separators,
# so verbatim quotes of a payload were marked invented -- 32 false flags.
from gen_tool_answer_turns import (  # noqa: E402
    _nums, _traces_to, LIST_ENUM, NUM)
from gen_tool_dialogues_da import strip_catalogue  # noqa: E402

SEP = "Værktøjer:"
SNAKEY = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
JSONISH = re.compile(r'^\s*[{}\[\]]|"\s*:\s*|[}\]]\s*$')
EN = re.compile(r"\b(the|and|with|from|your|please|here|are|this|that|will|"
                r"would|should|have|been|about|which)\b", re.I)
ASK = re.compile(r"(vil du gerne|kan du oplyse|hvilken .* vil du|hvad vil du|"
                 r"kunne du oplyse|angiv venligst)", re.I)
ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
W = 100


def load(path):
    """Accept both rendered (`messages`) and pipeline (`da.conversations`) rows."""
    out = []
    for line in Path(path).open():
        if not line.strip():
            continue
        r = json.loads(line)
        if "messages" in r:
            out.append((r, r["messages"], "rendered"))
        else:
            conv = r["da"]["conversations"]
            msgs = []
            for m in conv:
                if m.get("tool_calls"):
                    for c in m["tool_calls"]:
                        msgs.append({"role": "tool_call",
                                     "content": json.dumps(c.get("function") or {},
                                                           ensure_ascii=False)})
                elif m["role"] == "tool":
                    msgs.append({"role": "tool_result", "content": m["content"]})
                else:
                    msgs.append(m)
            out.append((r, msgs, "pipeline"))
    return out


def split_head(msgs):
    """(catalogue, opening question). Question text via the shared helper."""
    head = str(msgs[0].get("content") or "")
    if SEP not in head:
        return [], head
    rest = head.split(SEP, 1)[1].lstrip()
    try:
        cat, _end = json.JSONDecoder().raw_decode(rest)
    except Exception:
        cat = []
    return cat, strip_catalogue(head)


def flags_for(msgs, tool):
    """[(turn_index or None, code, detail)] -- everything the battery knows."""
    out = []
    calls = [m for m in msgs if m["role"] == "tool_call"]
    pays = [m for m in msgs if m["role"] == "tool_result"]
    _, opening = split_head(msgs)
    # SAME threshold as the gate. The review used <4 while the gate used <3,
    # so it flagged "Hvad er valgstedsbemanding?" -- a perfectly good question.
    if len(opening.split()) < 3:
        out.append((0, "degenerate-opening", repr(opening[:40])))
    for i, m in enumerate(msgs):
        c = str(m.get("content") or "")
        if m["role"] == "user" and i > 0 and c.rstrip().endswith("?") \
                and ASK.search(c):
            out.append((i, "assistant-question-in-user-turn", c[:50]))
        if m["role"] in ("user", "assistant") and len(EN.findall(c)) >= 4:
            out.append((i, "english-in-turn", c[:50]))
        if m["role"] == "assistant" and c.strip() and JSONISH.search(c.strip()) \
                and not re.search(r"[a-zæøå]{4,}\s+[a-zæøå]{4,}", c.lower()):
            out.append((i, "json-fragment", c[:40]))
        if m["role"] == "tool_result":
            nxt = next((x for x in msgs[i + 1:]
                        if x["role"] != "tool_result"), None)
            if nxt is None or nxt["role"] != "assistant":
                out.append((i, "result-never-answered", ""))
            try:
                p = json.loads(c)
            except Exception:
                p = {}
            if isinstance(p, dict):
                dates = [(k, v) for k, v in p.items()
                         if isinstance(v, str) and ISO.match(v)]
                for (k1, v1) in dates:
                    for (k2, v2) in dates:
                        if k1 < k2 and ("start" in k1 and "end" in k2) \
                                and v1 > v2:
                            out.append((i, "payload-dates-out-of-order",
                                        f"{k1}={v1} > {k2}={v2}"))
                for k, v in p.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) \
                            and ("percent" in k or "percentage" in k) and v > 100:
                        out.append((i, "payload-percent-over-100", f"{k}={v}"))
    for a, b in zip(range(len(msgs) - 1), range(1, len(msgs))):
        if msgs[a]["role"] == msgs[b]["role"] == "assistant" \
                and str(msgs[a].get("content") or "").strip() \
                and str(msgs[b].get("content") or "").strip():
            out.append((b, "consecutive-assistant-turns", ""))
    for i, m in enumerate(calls):
        try:
            args = json.loads(m["content"]).get("arguments") or {}
        except Exception:
            continue
        if any(isinstance(v, str) and not v.strip() for v in args.values()):
            out.append((None, "empty-string-argument", ""))
    bykey = {}
    for c, p in zip(calls, pays):
        try:
            k = json.dumps(json.loads(c["content"]).get("arguments") or {},
                           sort_keys=True)
        except Exception:
            continue
        bykey.setdefault(p["content"], set()).add(k)
    if any(len(v) > 1 for v in bykey.values()):
        out.append((None, "different-args-same-payload", ""))
    # grounding, prefix-scoped
    keys = set()
    for p in pays:
        try:
            keys |= set(json.loads(p["content"]))
        except Exception:
            pass
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or not str(m.get("content") or "").strip():
            continue
        if not any(x["role"] == "tool_result" for x in msgs[:i]):
            continue
        prior = set()
        for x in msgs[:i]:
            if x["role"] in ("tool_result", "tool_call", "user"):
                prior |= _nums(x.get("content") or "")
        scan = LIST_ENUM.sub(lambda z: " " * len(z.group()), m["content"])
        bad = [t.group() for t in NUM.finditer(scan)
               if not _traces_to(t.group(), prior)]
        if bad:
            out.append((i, "invents-number", ",".join(bad[:3])))
        leak = [t for t in SNAKEY.findall(m["content"]) if t in keys]
        if leak:
            out.append((i, "field-name-in-prose", ",".join(leak[:2])))
        if tool:
            for p in pays:
                try:
                    cv = json.loads(p["content"]).get(tool["competitor_field"])
                except Exception:
                    continue
                if cv is None:
                    continue
                s = str(cv).strip()
                hit = False
                try:
                    v = float(s.replace(",", "."))
                    hit = any(abs(float(t.group().replace(",", ".")) - v) < 1e-9
                              for t in NUM.finditer(m["content"]))
                except ValueError:
                    hit = len(s) > 2 and s.lower() in m["content"].lower()
                if hit:
                    out.append((i, "cites-competitor",
                                f"{tool['competitor_field']}={s[:24]}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--tools", type=Path, default=None)
    ap.add_argument("--roles", type=Path, default=None,
                    help="roles.jsonl -- per-ROW answer/competitor fields. "
                         "Without it a rotated row is judged against the "
                         "tool's default roles and its correct answer reads "
                         "as citing the competitor.")
    ap.add_argument("--rejects", type=Path, default=None,
                    help="judge_rejects.jsonl, appended in its own section")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    tools = {}
    if a.tools and a.tools.exists():
        for line in a.tools.open():
            if line.strip():
                t = json.loads(line)
                tools[t["name"]] = t

    roles = {}
    if a.roles and a.roles.exists():
        for line in a.roles.open():
            if line.strip():
                r = json.loads(line)
                roles[r["idx"]] = r

    sections = [("ACCEPTED", load(a.rows))]
    if a.rejects and a.rejects.exists():
        sections.append(("JUDGE-REJECTED", load(a.rejects)))

    body, index, tally = [], [], Counter()
    for label, rows in sections:
        body.append(f"\n\n{'#' * W}\n# {label}  ({len(rows)} rows)\n{'#' * W}")
        for n, (_raw, msgs, _kind) in enumerate(rows):
            cat, opening = split_head(msgs)
            called = set()
            for m in msgs:
                if m["role"] == "tool_call":
                    try:
                        called.add(json.loads(m["content"]).get("name"))
                    except Exception:
                        pass
            tool = next((tools[c] for c in called if c in tools), None)
            rr = roles.get(_raw.get("idx"))
            if tool and rr and rr.get("answer_field"):
                tool = {**tool, "answer_field": rr["answer_field"],
                        "competitor_field": rr["competitor_field"]}
            fl = flags_for(msgs, tool)
            for _, code, _d in fl:
                tally[code] += 1
            tag = f"{label[:3]}{n}"
            # EVERY row is indexed, not just the flagged ones. Listing only
            # flags made the file look like it contained only flags -- and the
            # unflagged rows are the ones worth reading, since every defect
            # this pipeline has shipped was found in a row nothing marked.
            codes = sorted({c for _, c, _ in fl})
            first = (msgs[1] if len(msgs) > 1 else msgs[0]).get("content") or ""
            gist = " ".join(str(split_head(msgs)[1] or first).split())[:52]
            index.append(f"  {tag:<9} {'!!' if fl else '  '} "
                         f"{', '.join(codes) if fl else 'ok':<34} {gist}")
            mark = "!!" if fl else "  "
            body.append(f"\n{'=' * W}\n{mark} ### {tag}"
                        + (f"   FLAGS: {', '.join(sorted({c for _, c, _ in fl}))}"
                           if fl else ""))
            if tool:
                body.append(f"   ANSWER={tool['answer_field']}   "
                            f"COMPETITOR={tool['competitor_field']}")
            if cat:
                body.append("   TOOLS: " + ", ".join(
                    ("*" + t.get("name", "?") + "*" if t.get("name") in called
                     else t.get("name", "?")) for t in cat))
            per_turn = {}
            for idx, code, detail in fl:
                if idx is not None:
                    per_turn.setdefault(idx, []).append(
                        code + (f" [{detail}]" if detail else ""))
            for i, m in enumerate(msgs):
                text = opening if i == 0 else str(m.get("content") or "")
                text = " ".join(str(text).split()) or "(tom)"
                lead = f"  [{m['role']:<11}] "
                body.append(textwrap.fill(text, W, initial_indent=lead,
                                          subsequent_indent=" " * len(lead)))
                for msg in per_turn.get(i, []):
                    body.append(f"  {'':<13}  !! {msg}")

    head = [f"{'=' * W}", "REVIEW  " + str(a.rows),
            f"{'=' * W}",
            f"rows: " + ", ".join(f"{lab} {len(rs)}" for lab, rs in sections),
            "", "flag tally:"]
    head += [f"  {v:>4}  {k}" for k, v in tally.most_common()] or ["  (none)"]
    n_flag = sum(1 for x in index if "!!" in x[:14])
    head += ["", f"ALL {len(index)} rows ({n_flag} flagged, "
                 f"{len(index) - n_flag} clean):"] + index
    head += ["", "in less:   /### <enter>  next row of any kind",
             "           /^!! <enter>   next FLAGGED row",
             "           n  repeat search    G  end    q  quit"]
    a.out.write_text("\n".join(head + body) + "\n")
    print(f"-> {a.out}  ({sum(len(r) for _, r in sections)} rows written, "
          f"{sum(1 for x in index if '!!' in x[:14])} flagged)")


if __name__ == "__main__":
    main()
