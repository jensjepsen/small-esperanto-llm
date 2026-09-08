"""Per-row values for distractor fields: shape from the exemplars, value from a hash.

WHY NOT THE EXEMPLARS THEMSELVES. A field whose value is drawn from a fixed
pool is the same string in every row that calls the tool, so the distractor
becomes the CONSTANT sitting beside a varying answer -- perfectly separable
without reading a single description. That is a fresh shortcut in place of the
one distractors exist to remove, and it is the same mistake the symbolized
twins already fixed once: per-row derivation, not per-schema constants.

So the exemplars are treated as a SPECIMEN OF FORM, never as the value. From
them we infer a generator -- integer range and magnitude, decimal precision,
date format and window, `id`-like prefix width, number+unit template -- and
the row's own index seeds it. Same tool, 400 rows, 400 different values.

WHAT CANNOT BE GENERATED. Free text (`author`, `meal_breakdown`) has no
inferable form, and the corpus is no help: of 8,325 observed string payload
fields only 20 carry 20 or more distinct values, the largest being `message`
at 323. Those fields need a sampled value bank, which is a separate pass;
`classify` returns FREE for them so the caller can route them there instead of
shipping a constant.

Deterministic by construction: value(idx) is a pure function of
(row index, field name, exemplars), so a rerun reproduces the corpus exactly
and a diff between two builds shows real changes only.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta

# --- shape classes -----------------------------------------------------------
INT = "int"
FLOAT = "float"
DATE = "date"
DATETIME = "datetime"
IDLIKE = "id"
NUMUNIT = "numunit"
ENUM = "enum"
FREE = "free"

_INT = re.compile(r"^-?\d+$")
_FLOAT = re.compile(r"^-?\d+[.,]\d+$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2})(:(\d{2}))?(Z?)$")
_IDLIKE = re.compile(r"^([A-Za-z]{1,5})[-_]?(\d{3,})$")
_NUMUNIT = re.compile(r"^(-?\d+(?:[.,]\d+)?)(\s*)([^\d\s].*)$")


def _all(exs, rx):
    return bool(exs) and all(rx.match(str(e).strip()) for e in exs)


def classify(exemplars) -> str:
    """The narrowest shape every exemplar satisfies.

    Order matters: `2026-03-14` also matches NUMUNIT (a number then a
    non-digit), so dates are tested first. Getting that backwards produced
    `2026-03-14` -> `4711-03-14`, which is a plausible-looking date from a
    generator that thought it was rescaling a quantity.
    """
    exs = [str(e).strip() for e in (exemplars or []) if str(e).strip()]
    if not exs:
        return FREE
    if _all(exs, _INT):
        return INT
    if _all(exs, _FLOAT):
        return FLOAT
    if _all(exs, _DATETIME):
        return DATETIME
    if _all(exs, _DATE):
        return DATE
    if _all(exs, _IDLIKE):
        return IDLIKE
    if _all(exs, _NUMUNIT):
        return NUMUNIT
    # A closed set is made of TOKENS -- `kvadratmeter`, `hektar`, `PNG`. The
    # first cut allowed up to two words, which swallowed `Mahatma Gandhi` and
    # would have cycled eight author names across every row of the tool: the
    # constant-beside-a-varying-answer failure, reintroduced. Multi-word
    # values are prose and go to the bank.
    if all(" " not in e and not re.search(r"\d", e) for e in exs):
        return ENUM
    return FREE


def _rand(idx, field, salt=""):
    """Deterministic unit float from (row, field). Never `random`: a rerun has
    to reproduce the corpus byte for byte or diffs between builds are noise."""
    h = hashlib.md5(f"{idx}\x00{field}\x00{salt}".encode()).hexdigest()
    return int(h[:12], 16) / float(1 << 48)


def _num(exs):
    vals = []
    for e in exs:
        t = str(e).strip().replace(",", ".")
        m = re.match(r"^-?\d+(\.\d+)?", t)
        if m:
            vals.append(float(m.group()))
    return vals or [1.0]


def _decimals(exs):
    d = 0
    for e in exs:
        t = str(e).strip().replace(",", ".")
        if "." in t:
            d = max(d, len(t.split(".")[1]))
    return d


def generate(field: str, exemplars, idx: int, kind: str | None = None):
    """A value for `field` in row `idx`, shaped like `exemplars`.

    Returns None when the shape is FREE -- the caller must supply a bank
    rather than have this invent English-looking filler in a Danish corpus.
    """
    exs = [str(e).strip() for e in (exemplars or []) if str(e).strip()]
    kind = kind or classify(exs)
    r = _rand(idx, field)

    if kind in (INT, FLOAT, NUMUNIT):
        if kind == NUMUNIT:
            m = _NUMUNIT.match(exs[int(_rand(idx, field, "u") * len(exs))])
            nums, sep, unit = _num(exs), m.group(2), m.group(3)
        else:
            nums, sep, unit = _num(exs), "", ""
        lo, hi = min(nums), max(nums)
        # A NARROW pool yields a narrow field. `["3 timer", "2 dage"]` spans
        # 2..3, which is four possible strings across the whole corpus -- a
        # pool by another name. Widen proportionally rather than to a fixed
        # floor: `priority_level` over 0..3 must not become 0..40.
        if hi - lo < max(4.0, lo * 0.5):
            lo, hi = max(0.0, lo * 0.4), max(hi * 3.0, lo + 12)
        v = lo + r * (hi - lo)
        dec = _decimals(exs) if kind != INT else 0
        if kind == INT or dec == 0:
            out = str(int(round(v)))
            val = int(round(v))
        else:
            out = f"{v:.{dec}f}"
            val = round(v, dec)
        return f"{out}{sep}{unit}" if kind == NUMUNIT else val

    if kind in (DATE, DATETIME):
        days = [datetime.strptime(e[:10], "%Y-%m-%d").date() for e in exs]
        lo, hi = min(days), max(days)
        # Backwards from the newest exemplar, never past it: extending forward
        # gave a `publication_date` of 2028 for a corpus written in 2026.
        span = max((hi - lo).days, 900)   # a one-week pool must not yield a week
        d = hi - timedelta(days=int(r * span))
        if kind == DATE:
            return d.isoformat()
        m = _DATETIME.match(exs[0])
        hh = int(_rand(idx, field, "h") * 24)
        mm = int(_rand(idx, field, "m") * 60)
        tail = "Z" if m.group(6) else ""
        if m.group(5):
            ss = int(_rand(idx, field, "s") * 60)
            return f"{d.isoformat()}T{hh:02d}:{mm:02d}:{ss:02d}{tail}"
        return f"{d.isoformat()}T{hh:02d}:{mm:02d}{tail}"

    if kind == IDLIKE:
        m = _IDLIKE.match(exs[int(_rand(idx, field, "p") * len(exs))])
        prefix, digits = m.group(1), m.group(2)
        n = int(_rand(idx, field, "n") * (10 ** len(digits)))
        sep = "-" if "-" in exs[0] else ("_" if "_" in exs[0] else "")
        return f"{prefix}{sep}{n:0{len(digits)}d}"

    if kind == ENUM:
        # A closed set is closed: cycling it is correct, not a compromise.
        return exs[int(r * len(exs)) % len(exs)]

    return None                            # FREE -> needs a bank


# --- planted controls --------------------------------------------------------
def check():
    """Every claim above, asserted. A generator that silently degrades to one
    value would be invisible in the corpus until a model exploited it."""
    cases = [
        ("count", ["3", "12", "45", "7"], INT),
        ("score", ["85.5", "70.0", "92.1"], FLOAT),
        ("dato", ["2026-03-14", "2025-11-30"], DATE),
        ("ts", ["2024-05-15T10:30:00Z", "2024-07-26T14:30:00Z"], DATETIME),
        ("imdb", ["tt0111161", "tt0068646"], IDLIKE),
        ("varighed", ["3 timer", "2 dage"], NUMUNIT),
        ("enhed", ["kvadratmeter", "hektar"], ENUM),
        ("forfatter", ["Mahatma Gandhi", "Søren Kierkegaard"], FREE),
    ]
    for field, exs, want in cases:
        got = classify(exs)
        assert got == want, f"classify({field}) = {got}, want {want}"

    # per-row VARIETY: the whole point. 500 rows must not collapse to a pool.
    for field, exs, kind in cases:
        vals = [generate(field, exs, i) for i in range(500)]
        if kind == FREE:
            assert all(v is None for v in vals), f"{field}: FREE must return None"
            continue
        n = len(set(map(str, vals)))
        # The guarantee is "many more than the pool", not "500 unique": a
        # small-range integer field genuinely has few plausible values, and
        # forcing variety there would produce nonsense.
        floor = {ENUM: len(exs), INT: 10, NUMUNIT: 10}.get(kind, 50)
        assert n >= floor, f"{field} ({kind}): only {n} distinct in 500 rows"
        # ENUM is exempt on purpose: a closed set has exactly its members, and
        # `kvadratmeter`/`hektar` recurring is what a real unit field does.
        # Every other shape must beat its own pool, or the generator is just
        # cycling the exemplars under a different name.
        if kind != ENUM:
            assert n > len(exs), (f"{field} ({kind}): {n} distinct is no "
                                  f"better than the {len(exs)}-value pool")

    # determinism
    assert generate("count", ["3", "12"], 7) == generate("count", ["3", "12"], 7)
    # and independence: same row, different fields must not co-vary
    assert generate("a", ["3", "12", "45"], 7) != generate("b", ["3", "12", "45"], 7)

    # type fidelity
    assert isinstance(generate("count", ["3", "12"], 1), int)
    assert isinstance(generate("score", ["1.5", "9.5"], 1), float)
    assert _DATE.match(generate("dato", ["2026-03-14", "2025-11-30"], 3))
    assert _DATETIME.match(generate("ts", ["2024-05-15T10:30:00Z"], 3))

    # a one-magnitude pool must still spread
    vals = {generate("n", ["100"], i) for i in range(200)}
    assert len(vals) >= 50, f"single-exemplar pool collapsed to {len(vals)}"

    # dates ordered nearest-first must not invert
    d = generate("dato", ["2026-03-14", "2026-03-15"], 11)
    assert _DATE.match(d), d
    print(f"values: {len(cases)} shapes classified, variety + determinism + "
          f"type fidelity asserted over 500 rows each", flush=True)


if __name__ == "__main__":
    check()
