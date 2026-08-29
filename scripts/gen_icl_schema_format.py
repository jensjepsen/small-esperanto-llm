"""Convert danish-json-grpo-v1 into in-context-learning rows.

Why this source. Probes on the v31 SFT base showed JSON is the only output
format the model can actually produce -- "type: enhed" lines scored 0 valid
outputs in 60 samples and inline <person> spans 0 grounded, while JSON works
because GRPO drilled it in. Building ICL on JSON therefore starts from a
distribution that already has mass: the model can express the answer, and
only the SCHEMA has to be induced from the demonstrations.

The dataset already carries everything needed, so nothing is synthesised:
  fields       the schema -- the grouping key, and the thing to be induced
  gold_values  the exact answer dict, so exemplars AND target render
               deterministically with no LLM and no verification pass
  passage      the input
  domain       ~80 rows each, so groups are never short

The original `prompt` column is deliberately NEVER used: it names the fields
outright ("Sørg for at inkludere felterne aktivitet, varighed_min, ..."),
which would hand over the schema and make the row a format-following exercise
rather than an inductive one.

Symbol tuning (Wei et al., EMNLP 2023, arXiv:2305.08298) comes almost free
here: gold_values is keyed by field name, so remapping the keys to
meaning-free symbols consistently across a group forces the field->symbol
mapping to be learned from the examples. That paper reports symbol tuning
needs only 1k-2k steps and makes models markedly more robust to prompts
without instructions -- which is exactly the row shape below.

Row shape. Exemplars are packed inside the single user turn, because
train_sft.py's collator masks only up to the FIRST <|assistant|> token; with
multi-turn rows the exemplar answers would themselves become training
targets, teaching schema production without a demonstration.

Held-out split partitions SCHEMAS, not rows. Evaluating on shuffled rows of
seen schemas measures memorisation of the 134 schemas that happen to be in
the data; the eval split uses field-sets that never appear in training.

Usage:
  python scripts/gen_icl_schema_format.py --n 200 --out scratch/icl_json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = "jensjepsen/danish-json-grpo-v1"

# Meaning-free key sets. `foo/bar/baz` is the symbol-tuning paper's own
# choice; the others give lexical variety so the model cannot key on one
# particular symbol vocabulary.
SYMBOLS = {
    "greek": ["alfa", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"],
    "kat":   ["kat_a", "kat_b", "kat_c", "kat_d", "kat_e", "kat_f", "kat_g", "kat_h"],
    "fnum":  ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"],
    "foo":   ["foo", "bar", "baz", "qux", "quux", "corge", "grault", "garply"],
}


def schema_id(fields) -> str:
    return "|".join(fields)


def fmt_heldout(fmt: str, frac: float) -> bool:
    """Formats are partitioned like schemas. Without this there is no split
    that answers "does it generalise to a format it never saw" -- exactly the
    question v1 could not answer and got wrong."""
    if frac <= 0:
        return False
    h = hashlib.sha1(("fmt:" + fmt).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF < frac


def is_heldout(fields, frac: float) -> bool:
    """Partition the schema space deterministically, so re-running with a
    different --n cannot leak a held-out schema into training."""
    h = hashlib.sha1(schema_id(fields).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF < frac


# ---------------------------------------------------------------- formats
#
# Format is the SECOND induction axis. Training v1 on JSON alone produced a
# model that answers a "type: enhed" request with malformed JSON -- schema
# induction transferred to unseen schemas, format did not transfer at all.
# So the renderer is sampled per row like the schema is: constant within a
# row (all exemplars and the target), varying across rows.
#
# Three decisions have to be made per format and held consistent, or a row
# stops being inducible:
#   lists  flat formats repeat the key, one element per line/field
#   null   rendered as an explicit marker, never silently dropped -- the key
#          must still be visible or key-set induction breaks
#   types  flat formats are untyped, so scoring compares post-parse as
#          strings; see parse_fmt
NULL = "-"


def _vals(v):
    """Gold value -> list of rendered strings (the common currency)."""
    if v is None or v == "" or v == []:
        return [NULL]
    if isinstance(v, bool):
        return ["true" if v else "false"]
    if isinstance(v, list):
        return [_vals(x)[0] for x in v] or [NULL]
    if isinstance(v, float) and v.is_integer():
        return [str(int(v))]
    return [str(v)]


def _render_flat(gold, fields, keymap, sep, tmpl, numbered=False):
    out, i = [], 0
    for f in fields:
        for x in _vals(gold.get(f)):
            i += 1
            out.append((f"{i}. " if numbered else "") + tmpl.format(k=keymap[f], v=x))
    return sep.join(out)


def _parse_flat(text, keys, rx, val_first=False):
    hits = rx.findall(text or "")
    if not hits:
        return None
    out = {}
    for a_, b_ in hits:
        v, k = (a_, b_) if val_first else (b_, a_)
        k = k.strip()
        if k not in keys:
            return None
        out.setdefault(k, []).append(v.strip())
    return out


def _kpat(keys):
    return "|".join(re.escape(k) for k in keys)


FORMATS = {
    "json": {
        "render": lambda g, fs, km: json.dumps(
            {km[f]: g.get(f) for f in fs}, ensure_ascii=False),
        "parse": lambda t, keys: _parse_json(t, keys),
    },
    "kv_colon": {
        "render": lambda g, fs, km: _render_flat(g, fs, km, "\n", "{k}: {v}"),
        "parse": lambda t, keys: _parse_flat(
            t, keys, re.compile(rf"^\s*({_kpat(keys)})\s*:\s*(.+?)\s*$", re.M)),
    },
    "kv_eq": {
        "render": lambda g, fs, km: _render_flat(g, fs, km, "\n", "{k}={v}"),
        "parse": lambda t, keys: _parse_flat(
            t, keys, re.compile(rf"^\s*({_kpat(keys)})\s*=\s*(.+?)\s*$", re.M)),
    },
    "kv_bracket": {
        "render": lambda g, fs, km: _render_flat(g, fs, km, "\n", "[{k}] {v}"),
        "parse": lambda t, keys: _parse_flat(
            t, keys, re.compile(rf"^\s*\[({_kpat(keys)})\]\s*(.+?)\s*$", re.M)),
    },
    "kv_arrow": {
        "render": lambda g, fs, km: _render_flat(g, fs, km, "\n", "{v} -> {k}"),
        "parse": lambda t, keys: _parse_flat(
            t, keys, re.compile(rf"^\s*(.+?)\s*->\s*({_kpat(keys)})\s*$", re.M),
            val_first=True),
    },
    "numbered": {
        "render": lambda g, fs, km: _render_flat(
            g, fs, km, "\n", "{k}: {v}", numbered=True),
        "parse": lambda t, keys: _parse_flat(
            t, keys,
            re.compile(rf"^\s*\d+\.\s*({_kpat(keys)})\s*:\s*(.+?)\s*$", re.M)),
    },
    "tsv": {
        "render": lambda g, fs, km: _render_flat(g, fs, km, "\n", "{k}\t{v}"),
        "parse": lambda t, keys: _parse_flat(
            t, keys, re.compile(rf"^\s*({_kpat(keys)})\t(.+?)\s*$", re.M)),
    },
    "tagged": {
        "render": lambda g, fs, km: _render_flat(
            g, fs, km, "\n", "<{k}>{v}</{k}>"),
        "parse": lambda t, keys: _parse_flat(
            t, keys,
            re.compile(rf"<({_kpat(keys)})>(.*?)</\1>", re.S)),
    },
    # Two more PAIRED-delimiter formats. v2 held `tagged` out and it scored
    # 0.5% while the line-based `kv_eq` reached 65.1% -- transfer happened
    # within the line family and not across to paired delimiters. The
    # diagnosis was structural, not semantic: 100% of tagged predictions
    # opened a tag, 62% carried all the right values, but only 2% balanced
    # their opens and closes. These exist so a paired format can be TRAINED
    # while a different paired format is held out, which is the direct test
    # of whether the family is learnable at all.
    "bracket_pair": {                       # [k]v[/k] -- bracket vocabulary,
        "render": lambda g, fs, km: _render_flat(   # shared with kv_bracket
            g, fs, km, "\n", "[{k}]{v}[/{k}]"),
        "parse": lambda t, keys: _parse_flat(
            t, keys,
            re.compile(rf"\[({_kpat(keys)})\](.*?)\[/\1\]", re.S)),
    },
    "brace_pair": {                         # {k}v{/k} -- brace vocabulary,
        "render": lambda g, fs, km: _render_flat(   # collides with JSON's
            g, fs, km, "\n", "{{{k}}}{v}{{/{k}}}"),
        "parse": lambda t, keys: _parse_flat(
            t, keys,
            re.compile(rf"\{{({_kpat(keys)})\}}(.*?)\{{/\1\}}", re.S)),
    },
}


def _parse_json(t, keys):
    a_ = t.find("{")
    b_ = t.rfind("}")
    if a_ < 0 or b_ < a_:
        return None
    try:
        d = json.loads(t[a_:b_ + 1])
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    out = {}
    for k, v in d.items():
        if k not in keys:
            return None
        out[k] = _vals(v)
    return out


def render(gold: dict, fields, keymap, fmt: str = "json") -> str:
    """Answer in `fmt`, keys in schema order and remapped for symbols.

    Fixed key order matters: a varying order would make the demonstrations
    look inconsistent for a reason unrelated to the schema.
    """
    return FORMATS[fmt]["render"](gold, fields, keymap)


def canon(text: str, fmt: str, keys) -> dict | None:
    """Parse to {key: [value-strings]} -- the shape both sides of a
    comparison are reduced to, so an untyped flat format and JSON can be
    scored the same way."""
    d = FORMATS[fmt]["parse"](text, set(keys))
    if d is None:
        return None
    return {k: [re.sub(r"\s+", " ", x).strip().lower() for x in v]
            for k, v in d.items()}


def gate_format(fmt: str) -> None:
    """Constructive control per format: a compliant render must parse back to
    the same thing, and a one-edit break must not."""
    fields = ["a", "b", "c"]
    km = {"a": "alfa", "b": "beta", "c": "gamma"}
    probe = {"a": "Knud Vilby", "b": ["X", "Y"], "c": None}
    r = render(probe, fields, km, fmt)
    got = canon(r, fmt, km.values())
    want = {"alfa": ["knud vilby"], "beta": ["x", "y"], "gamma": [NULL]}
    assert got == want, f"{fmt} round-trip: {r!r} -> {got!r}"
    # Dropping the first character is destructive for EVERY format: it eats
    # the opening delimiter or the first key character, so the first item
    # either fails to parse or parses to something else. An earlier version
    # deleted a hand-listed set of delimiters (":", "=", "->", "[", "<", tab)
    # and silently passed brace_pair, whose "{}" was not on the list -- a
    # per-format mutation list is exactly the kind of thing that rots as
    # formats are added.
    assert canon(r[1:], fmt, km.values()) != want, f"{fmt} break undetected"


_W = re.compile(r"[^\W\d_]+[\w.\-]*", re.UNICODE)
_STOP = {
    "og", "i", "på", "af", "en", "et", "den", "det", "de", "til", "for", "med",
    "er", "var", "som", "at", "har", "kan", "fra", "om", "ved", "der", "blev",
    "sig", "han", "hun", "vi", "du", "jeg", "man", "over", "under", "efter",
    "mellem", "hvor", "når", "da", "men", "eller", "så", "nu", "kl", "d",
}


def boundary_anomalies(groups):
    """Rows whose extraction disagrees with its schema's majority convention
    about WHERE a value starts.

    Verbatim-presence does not catch this: "Jane Doe" is verbatim inside
    "professor Jane Doe", so a row whose `underviser` gold drops the title
    passes that check while contradicting demonstrations whose gold for the
    same field keeps it ("professor Jensen"). Harmless for single-row SFT --
    fatal under few-shot, where the demonstrations establish a convention and
    the target then breaks it.

    Per (schema, field): a word seen as the FIRST word of the value in some
    rows and as the word immediately BEFORE it in others marks a boundary
    that moves. The smaller side is the anomaly. Measured over the full
    source: 235 of 16,712 scalar values (1.41%), 232 of 7,143 rows (3.2%).

    Scalars only. For a list field the word before element 2 IS element 1, so
    every list field would self-report; that is adjacency, not inconsistency.
    """
    bad = set()
    for fields, rows in groups.items():
        seen = {}
        for r in rows:
            passage = r["passage"] or ""
            low = passage.lower()
            try:
                g = json.loads(r["gold_values"])
            except Exception:
                continue
            for f in fields:
                v = g.get(f)
                if isinstance(v, list) or not isinstance(v, str):
                    continue
                xs = v.strip()
                if len(xs) < 3:
                    continue
                i = low.find(xs.lower())
                if i < 0:
                    continue
                wv = _W.findall(xs)
                before = _W.findall(passage[:i])
                seen.setdefault(f, []).append(
                    (r["_i"], wv[0].lower() if wv else "",
                     before[-1].lower() if before else ""))
        for f, lst in seen.items():
            if len(lst) < 4:
                continue
            heads = {h for _, h, _ in lst}
            leads = {l for _, _, l in lst}
            for w in heads & leads:
                if w in _STOP or len(w) <= 2:
                    continue
                inc = [n for n, h, _ in lst if h == w]
                exc = [n for n, _, l in lst if l == w]
                bad.update(inc if len(inc) <= len(exc) else exc)
    return bad


def build_row(rng, group, fields, shots, sym_frac, fmt="json"):
    picks = rng.sample(group, shots + 1)
    demos, target = picks[:-1], picks[-1]

    # Distinct passages. Variants within a schema repeat source text, and a
    # duplicated passage spends a demonstration slot without adding evidence.
    # Containment counts too: one variant is often a prefix of another
    # ("Hvad er hovedstaden i Frankrig? Paris er kendt for..." extended by a
    # further sentence), which exact-match dedup does not see.
    passages = [(p["passage"] or "").strip() for p in picks]
    if len(set(passages)) != len(passages):
        return None
    for i, a_ in enumerate(passages):
        for j, b_ in enumerate(passages):
            if i != j and len(a_) > 20 and a_ in b_:
                return None

    # EXTRACTABILITY. Every value must be recoverable from its own passage,
    # for demonstrations and target alike.
    #
    #   strings  ~3% of fill_template gold is invented rather than extracted
    #            -- an email whose `emne` gold is "Bekræftelse af møde" when
    #            the passage only says "jeg skriver for at bekræfte vores
    #            møde". Demonstrations that quote the subject literally
    #            cannot teach a target that summarises one.
    #   numbers  3.5% of numeric gold (272/7779) is derived, not read:
    #            varighed_min=180 from "3 timer", varighed_sek=355 from
    #            "5:55", vingefang_cm=23 from the range "21-24",
    #            befolkning=5900000 from nothing in the passage at all, and
    #            alder=46 from a birth date plus an unstated current year --
    #            that last one no demonstration could ever convey.
    #
    # Both are the same defect: the answer needs an operation the examples
    # never show. Numbers are checked against the digit forms actually used
    # in Danish text (1.300.000 grouping, 12,5 decimals); a comma/dot swap
    # alone never accounts for a miss, so notation is not the issue here.
    for p in picks:
        pa = (p["passage"] or "")
        for v in json.loads(p["gold_values"]).values():
            for x in (v if isinstance(v, list) else [v]):
                if isinstance(x, str) and len(x.strip()) >= 3:
                    # CASE-SENSITIVE. A case-insensitive match lets gold
                    # re-case a span ("normanniske tropper" ->
                    # "Normanniske tropper"); when no demonstration does the
                    # same the target silently breaks the convention the
                    # examples establish.
                    if x.strip() not in pa:
                        return None
                elif isinstance(x, (int, float)) and not isinstance(x, bool):
                    s = "%g" % x
                    forms = {s, s.replace(".", ",")}
                    if float(x).is_integer():
                        forms.add(f"{int(x):,}".replace(",", "."))
                    if not any(f in pa for f in forms):
                        return None

    # LIST ORDER. Some schemas list elements in passage order and others
    # reorder them; either is learnable, but a prompt mixing the two teaches
    # an ordering rule the answer then contradicts. Require all picks to
    # agree.
    def _inorder(p):
        g = json.loads(p["gold_values"])
        pa = (p["passage"] or "").lower()
        flags = []
        for v in g.values():
            if not isinstance(v, list) or len(v) < 2:
                continue
            pos = [pa.find(str(x).lower()) for x in v if isinstance(x, str)]
            pos = [q for q in pos if q >= 0]
            if len(pos) >= 2:
                flags.append(pos == sorted(pos))
        return flags

    order_flags = [f for p in picks for f in _inorder(p)]
    if order_flags and len(set(order_flags)) > 1:
        return None

    # CONVENTION CLASH. A field whose values switch notation across the
    # prompt teaches no rule at all: `resultat` appearing as "1-0" in one
    # demonstration and "Magnus Carlsen vandt" in another leaves the target
    # form undetermined. Likewise `åbning` gold that sometimes carries its
    # enclosing quote characters and sometimes not. Both pass extractability
    # -- the values ARE verbatim -- and both slip past the boundary check,
    # because what varies is not a single adjacent word.
    def _shape(x):
        if not isinstance(x, str):
            return None
        t = x.strip()
        quoted = len(t) > 1 and t[0] in "\"'" and t[-1] in "\"'"
        return ("q" if quoted else "-",
                "n" if re.fullmatch(r"[\d\s\-–/:.,]+", t) else "t")

    for f in fields:
        shapes = {sh for p in picks
                  if (sh := _shape(json.loads(p["gold_values"]).get(f)))}
        if len(shapes) > 1:
            return None

    # BOOLEANS. true/false never appear literally in a Danish passage, so
    # the text->boolean mapping exists only in the demonstrations. A target
    # of debug=False whose exemplars all show debug=True is a coin flip --
    # the same unanswerability as a symbol whose meaning is never shown, and
    # missed by the checks above because those skip bools by construction.
    tgold = json.loads(target["gold_values"])
    for f in fields:
        tv = tgold.get(f)
        if isinstance(tv, bool):
            shown = {json.loads(d["gold_values"]).get(f) for d in demos}
            if tv not in shown:
                return None

    # ANTI-COPY. Within a schema the same underlying item recurs across
    # variants -- two rows both describing Slaget ved Dybbøl produce
    # byte-identical answers, so copying an exemplar scores perfectly and
    # trains the one habit this data exists to break. (seed_idx does NOT
    # separate them: it indexes the schema, 80 rows each, 1:1 with the
    # field-set.) Reject when any single demonstration already supplies most
    # of the target's values.
    # Booleans COUNT here. Exempting them was tried and loosened the guard:
    # rows survived where a demonstration supplied both the boolean and a
    # content value ("Store Bededag" verbatim, 2 of 3 fields liftable). The
    # deadlock that exemption was meant to solve -- a lone boolean field can
    # satisfy neither this check nor the demonstrated-value check above -- is
    # handled instead by rejecting schemas with nothing but booleans, which
    # are a coin flip rather than an extraction task.
    tg = json.loads(target["gold_values"])
    vals = [f for f in fields if tg.get(f) not in (None, "", [], {})]
    if not [f for f in vals if not isinstance(tg.get(f), bool)]:
        return None
    for d in demos:
        dg = json.loads(d["gold_values"])
        same = sum(1 for f in vals if dg.get(f) == tg.get(f))
        if same > len(vals) / 2:
            return None

    scheme = None
    keymap = {f: f for f in fields}
    if rng.random() < sym_frac:
        # A symbol that also occurs in the passages reads as content rather
        # than as a key ("beta" in a text about beta-testing), so schemes
        # colliding with this row's text are skipped rather than used.
        blob = " ".join(p["passage"] or "" for p in picks).lower()
        opts = [k for k, v in SYMBOLS.items()
                if len(fields) <= len(v)
                and not any(sym in blob for sym in v[:len(fields)])]
        if opts:
            scheme = rng.choice(opts)
            shuffled = list(SYMBOLS[scheme][:len(fields)])
            rng.shuffle(shuffled)      # symbol order must not track field order
            keymap = dict(zip(fields, shuffled))

    # WELL-POSEDNESS: every key the target answer uses must appear in at
    # least one demonstration with a non-null value. A key that is null or
    # missing throughout the exemplars has no demonstrated meaning, so under
    # a symbol scheme there is nothing to induce -- the same defect as an
    # answer type no exemplar shows.
    demoed = set()
    for d in demos:
        g = json.loads(d["gold_values"])
        for f in fields:
            v = g.get(f)
            if v not in (None, "", [], {}):
                demoed.add(f)
    tgold = json.loads(target["gold_values"])
    needed = {f for f in fields
              if tgold.get(f) not in (None, "", [], {})}
    if not needed or needed - demoed:
        return None

    parts = ["Eksempler:"]
    for d in demos:
        parts.append(f'Tekst:\n{d["passage"].strip()}\n'
                     f'Svar: {render(json.loads(d["gold_values"]), fields, keymap, fmt)}')
    parts.append(f'Tekst:\n{target["passage"].strip()}\nSvar:')
    # A value containing a newline or tab is fine in JSON (it escapes) but in
    # any line-based format it splits across lines and the parser recovers
    # only the first fragment. One such value exists in the source (an HTML
    # body); it would silently corrupt every flat-format row that drew it.
    if fmt != "json":
        for p_ in picks:
            for v in json.loads(p_["gold_values"]).values():
                for x in (v if isinstance(v, list) else [v]):
                    if isinstance(x, str) and ("\n" in x or "\t" in x):
                        return None

    answer = render(tgold, fields, keymap, fmt)
    # the target must parse back to itself under this format, or the row is
    # scored against something the renderer cannot express
    if canon(answer, fmt, keymap.values()) is None:
        return None
    return {
        # provenance, stripped before writing -- used to build a val split
        # whose passages appear nowhere in train or eval
        "_src": [p["_i"] for p in picks],
        "messages": [{"role": "user", "content": "\n\n".join(parts)},
                     {"role": "assistant", "content": answer}],
        "meta": {"schema": schema_id(fields), "n_fields": len(fields),
                 "shots": shots, "symbols": scheme or "none", "format": fmt,
                 "task_type": target["task_type"], "domain": target["domain"],
                 "heldout_schema": is_heldout(fields, 0.0)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="scratch/icl_json")
    ap.add_argument("--sym-frac", type=float, default=0.5,
                    help="share of rows whose field names are replaced by "
                         "meaning-free symbols (symbol tuning)")
    ap.add_argument("--heldout-frac", type=float, default=0.2)
    ap.add_argument("--formats", nargs="*", default=["json"],
                    help="output formats to sample from, or 'all'. Format is "
                         "an induction axis: v1 trained on json alone and did "
                         "not transfer to any other format.")
    ap.add_argument("--fmt-heldout-frac", type=float, default=0.0,
                    help="share of FORMATS reserved for the unseen-format "
                         "eval splits (hash-partitioned)")
    ap.add_argument("--held-formats", nargs="*", default=None,
                    help="hold out exactly these formats, overriding the hash "
                         "partition. Needed to place a specific format on a "
                         "specific side -- the hash cannot express 'train on "
                         "tagged, hold out bracket_pair'.")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="share of EACH schema's source rows reserved for a "
                         "val split before train/eval are built")
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-fmt", type=int, default=0,
                    help="rows to generate on the HELD-OUT formats, for the "
                         "eval_format / eval_both splits")
    ap.add_argument("--exclude-src", type=Path, default=None,
                    help="JSON list of source row ids to drop before "
                         "generating; pass the used_src.json of an earlier "
                         "build to get a split that shares no passages "
                         "with it")
    ap.add_argument("--keep-boundary-anomalies", action="store_true",
                    help="keep rows whose extraction boundary contradicts "
                         "their schema's majority convention (default: drop)")
    ap.add_argument("--min-passage", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(SOURCE, split="train")
    groups = defaultdict(list)
    seen_ids = []
    n_nopass = 0
    for r in ds:
        if len((r["passage"] or "").strip()) < args.min_passage:
            n_nopass += 1        # generate-style rows have nothing to read from
            continue
        try:
            json.loads(r["gold_values"])
        except Exception:
            continue
        row = dict(r)
        row["_i"] = len(seen_ids)
        seen_ids.append(row)
        groups[tuple(r["fields"])].append(row)
    print(f"{SOURCE}: {len(ds)} rows, {n_nopass} dropped for no passage")
    if not args.keep_boundary_anomalies:
        bad = boundary_anomalies(groups)
        before = sum(len(v) for v in groups.values())
        groups = {k: [r for r in v if r["_i"] not in bad]
                  for k, v in groups.items()}
        after = sum(len(v) for v in groups.values())
        print(f"boundary filter: dropped {before - after} rows "
              f"({100 * (before - after) / max(1, before):.1f}%) whose "
              f"extraction contradicts their schema's majority convention")
    if args.exclude_src:
        drop = set(json.loads(args.exclude_src.read_text()))
        before = sum(len(v) for v in groups.values())
        groups = {k: [r for r in v if r["_i"] not in drop]
                  for k, v in groups.items()}
        print(f"exclude-src: dropped {before - sum(len(v) for v in groups.values())} "
              f"source rows used by {args.exclude_src}")
    # VAL HOLDOUT. Taking val from whatever train happened not to consume
    # gives a thin, lopsided split -- an earlier attempt covered 34 of 113
    # schemas because most were left with fewer than two spare rows. Reserving
    # a share of each schema's rows FIRST keeps val broad and guarantees the
    # passages never appear in train or eval.
    val_pool = {}
    if args.val_frac > 0:
        # Partition on PASSAGE TEXT, not row id: within a schema several
        # source rows carry the same passage, so holding out a row leaves its
        # text reachable through a sibling. Row-id partitioning leaked 47
        # passages from val into train.
        prng = random.Random(args.seed ^ 0x5A17)
        for k, v in list(groups.items()):
            by_text = defaultdict(list)
            for r in v:
                by_text[(r["passage"] or "").strip()].append(r)
            texts = sorted(by_text)
            if len(texts) < 4:
                continue          # too few distinct passages to give any away
            n_hold = max(2, int(round(len(texts) * args.val_frac)))
            n_hold = min(n_hold, len(texts) - 2)
            held_t = set(prng.sample(texts, n_hold))
            groups[k] = [r for t in texts if t not in held_t for r in by_text[t]]
            val_pool[k] = [r for t in held_t for r in by_text[t]]
        print(f"val holdout: reserved "
              f"{sum(len(v) for v in val_pool.values())} source rows across "
              f"{len(val_pool)} schemas ({args.val_frac:.0%} each)")

    usable = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"{len(groups)} schemas, {len(usable)} with >=2 rows "
          f"({sum(len(v) for v in usable.values())} rows usable)", flush=True)

    def generate(rng, pool, n, seen_rows, schema_filter=None,
                 fmt_pool=None):
        keys = sorted(k for k in pool
                      if schema_filter is None or schema_filter(k))
        rows, tried, n_dupe = [], 0, 0
        if not keys:
            return rows, tried, n_dupe
        while len(rows) < n and tried < n * 120:
            tried += 1
            fields = rng.choice(keys)
            group = pool[fields]
            want = rng.randint(1, 5)
            # Retry AT the drawn shot count. Redrawing everything on
            # rejection biases the output toward short prompts, because a
            # 5-shot row must clear every filter five times over: an earlier
            # run came out 2541 one-shot against 1592 five-shot.
            fmt = rng.choice(fmt_pool or seen_f)
            r = None
            for _ in range(6):
                shots = min(want, len(group) - 1)
                r = build_row(rng, group, list(fields), shots, args.sym_frac,
                              fmt)
                if r:
                    break
            if not r:
                continue
            key = hashlib.md5((r["messages"][0]["content"]
                               + r["messages"][1]["content"]).encode()).hexdigest()
            if key in seen_rows:
                n_dupe += 1
                continue
            seen_rows.add(key)
            r["meta"]["heldout_schema"] = is_heldout(fields, args.heldout_frac)
            r["meta"]["heldout_format"] = r["meta"]["format"] in held_f
            rows.append(r)
        return rows, tried, n_dupe

    fmts = sorted(FORMATS) if args.formats == ["all"] else list(args.formats)
    bad_f = [f for f in fmts if f not in FORMATS]
    if bad_f:
        raise SystemExit(f"unknown format(s): {bad_f}; have {sorted(FORMATS)}")
    for f in fmts:
        gate_format(f)            # raises if render/parse do not round-trip
    if args.held_formats is not None:
        unknown = [f for f in args.held_formats if f not in fmts]
        if unknown:
            raise SystemExit(f"--held-formats not in --formats: {unknown}")
        held_f = [f for f in fmts if f in args.held_formats]
    else:
        held_f = [f for f in fmts if fmt_heldout(f, args.fmt_heldout_frac)]
    seen_f = [f for f in fmts if f not in held_f]
    if not seen_f:
        raise SystemExit("fmt-heldout-frac held out every format")
    print(f"formats: {len(fmts)} gated; train formats={seen_f}"
          + (f"  held out={held_f}" if held_f else ""), flush=True)

    rng = random.Random(args.seed)
    seen_rows = set()
    rows, tried, n_dupe = generate(rng, usable, args.n, seen_rows)
    # A second pass over the HELD-OUT formats. generate() otherwise only ever
    # draws from seen_f, so without this there would be no unseen-format rows
    # to evaluate on -- the gap that let v1 ship a single-format model.
    fmt_rows = []
    if held_f and args.n_fmt:
        fmt_rows, f_tried, f_dupe = generate(
            rng, usable, args.n_fmt, seen_rows, fmt_pool=held_f)
        print(f"unseen-format pass: {len(fmt_rows)} rows from {f_tried} draws",
              flush=True)

    # val draws only from reserved passages, seen schemas, seen formats: it
    # varies ONE thing (the passage) so it isolates the task from both axes
    val_usable = {k: v for k, v in val_pool.items() if len(v) >= 2}
    val, v_tried, v_dupe = generate(
        rng, val_usable, args.n_val, seen_rows,
        schema_filter=lambda k: not is_heldout(k, args.heldout_frac))

    allr = rows + fmt_rows + val
    used_src = sorted({i for r in allr for i in r["_src"]})
    for r in allr:
        del r["_src"]
    main = rows + fmt_rows
    rng.shuffle(main)
    rng.shuffle(val)

    def pick(hs, hf):
        return [r for r in main if r["meta"]["heldout_schema"] == hs
                and r["meta"]["heldout_format"] == hf]

    # The factorial: which axis generalises is only answerable if each is
    # held out independently and together.
    splits = [
        ("train",       pick(False, False)),
        ("eval_schema", pick(True, False)),
        ("eval_format", pick(False, True)),
        ("eval_both",   pick(True, True)),
        ("val",         val),
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for nm, rs in splits:
        if not rs:
            continue
        (out / f"{nm}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))
        print(f"-> {out / (nm + '.jsonl')}  ({len(rs)} rows)")

    (out / "used_src.json").write_text(json.dumps(used_src))
    print(f"-> {out / 'used_src.json'}  ({len(used_src)} source rows consumed)")

    print(f"\n{len(rows)} rows from {tried} draws; {n_dupe} exact duplicates dropped")
    tr = dict(splits)["train"]
    for nm, rs in splits:
        if not rs or nm == "train":
            continue
        so = {r["meta"]["schema"] for r in tr} & {r["meta"]["schema"] for r in rs}
        fo = {r["meta"]["format"] for r in tr} & {r["meta"]["format"] for r in rs}
        exp = {"eval_schema": "0 schema", "eval_format": "0 format",
               "eval_both": "0 schema, 0 format", "val": "shares both"}[nm]
        print(f"  {nm:<12} n={len(rs):<6} shares {len(so)} schemas, "
              f"{len(fo)} formats with train   (expect: {exp})")
    for k in ("format", "shots", "symbols", "task_type", "n_fields"):
        c = Counter(r["meta"][k] for r in main)
        print(f"  {k:<10}" + "  ".join(f"{a}={b}" for a, b in
                                       sorted(c.items(), key=lambda x: -x[1])))
    print(f"  distinct schemas used: {len({r['meta']['schema'] for r in main})}")

    for r in rows[:args.show]:
        m = r["meta"]
        print("\n" + "#" * 76)
        print(f"# schema={m['schema']}  symbols={m['symbols']}  "
              f"shots={m['shots']}  task_type={m['task_type']}  "
              f"heldout={m['heldout_schema']}")
        print("#" * 76)
        print(r["messages"][0]["content"])
        print(">" * 20 + " ASSISTANT (supervised target) " + "<" * 20)
        print(r["messages"][1]["content"])


if __name__ == "__main__":
    main()
