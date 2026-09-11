"""Make a contract's declared TYPE agree with its declared EXAMPLES.

Deterministic, no API. A field declaring `integer` whose examples are `A12`,
`B05`, `C21` is a string field with a wrong type, and everything downstream
that trusts the type is wrong with it: arguments go out quoted, links form
between value spaces that cannot meet, and a numeric clamp silently skips.

DEMOTION ONLY. A numeric type is believed unless an example contradicts it.
The reverse -- promoting `string` to `integer` because every example happens
to parse -- is refused: an id like "1001" is a string that looks numeric, and
the declaration is the author's statement about the VALUE SPACE, not a guess
to be overridden by three samples.

ORDER IS PRESERVED EXACTLY and only `type` values change, so the output is a
drop-in for the input: the seeded shuffle in the generator picks the same
tools in the same order, and a corpus already built on the old file can be
resumed onto the new one.

    python3 scripts/repair_tool_types.py IN.jsonl OUT.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

NUMERIC = {"integer", "number", "int", "float"}
# A bare number, optionally signed, with a decimal comma or point. Anything
# else -- a unit, a prefix, a range, a comma-separated list -- means the field
# holds text that contains a number, which is not the same as a number.
LITERAL = re.compile(r"^\s*-?\d+(?:[.,]\d+)?\s*$")


def lies(field):
    if str(field.get("type") or "").lower() not in NUMERIC:
        return False
    ex = [str(e).strip().strip("\"'") for e in (field.get("examples") or [])
          if str(e).strip()]
    return bool(ex) and not all(LITERAL.match(e) for e in ex)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    stats, shown = Counter(), []
    out = []
    for line in src.open():
        if not line.strip():
            continue
        t = json.loads(line)
        for kind in ("parameters", "returns"):
            for f in (t.get(kind) or []):
                if lies(f):
                    stats[f"{kind}: {f.get('type')} -> string"] += 1
                    if len(shown) < 8:
                        shown.append(
                            f"{t['name']}.{f['name']}  {f.get('type')} -> string"
                            f"   examples={[str(e) for e in (f.get('examples') or [])][:3]}")
                    f["type"] = "string"
        out.append(t)
    dst.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in out) + "\n")
    print(f"{len(out):,} tools -> {dst}")
    for k, v in stats.most_common():
        print(f"  {v:>5}  {k}")
    for s in shown:
        print(f"     {s}")


if __name__ == "__main__":
    main()
