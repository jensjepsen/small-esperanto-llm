"""Danish NER SFT rows from DANSK, in three prompt modes.

Source is `chcaa/dansk-ner` (DANSK): 11,762 train sentences annotated with the
same 18 OntoNotes types in EVERY split, character offsets, and a `dagw_domain`
field. Chosen over dane_plus, whose train/dev use a CoNLL scheme with no DATE
at all while test uses OntoNotes -- a mismatch that makes several types
unlearnable from its train split.

Training reads DANSK train only. EuroEval's `dansk-mini` benchmark samples
each of its splits from the CORRESPONDING source split
(`src/scripts/dataset_creation/create_dansk.py`), so training on train is the
ordinary train/test protocol rather than leakage. dev drives model selection;
test is used only to build eval rows and is never trained or selected on.

Three prompt modes, selectable by fraction:
  icl          demonstrations only, no instruction -- the requested type set
               and the output format must both be induced
  instruction  no demonstrations; the instruction names the types and spells
               out the format
  both         instruction plus demonstrations

Reuses the renderers, parsers and round-trip gate from
gen_icl_schema_format.py, so an NER answer is expressed in the same ten
formats and scored the same way.

Two properties come free from the source that the JSON generator had to
enforce with filters: every span is a character slice of its own passage, so
extractability is guaranteed, and the annotation scheme is uniform, so there
is no boundary-convention drift between rows.

Splits, all eval variants built from DANSK test so each varies one axis on
text that was never trained on:
  train        DANSK train, seen formats, seen types
  val          DANSK dev,   seen formats, seen types   (selection)
  eval         DANSK test,  seen formats, seen types   (in-distribution NER)
  eval_format  DANSK test,  UNSEEN formats
  eval_type    DANSK test,  UNSEEN entity types

Text filters are narrow and deliberately NOT by domain: rows with a URL,
under 5 tokens, or majority non-alphabetic are dropped because there is no
context to extract from. Domain filtering was considered and rejected --
Legal reads as clause fragments but is well annotated, and Web is genuinely
mixed rather than uniformly junk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_icl_schema_format import (  # noqa: E402
    FORMATS, NULL, SYMBOLS, canon, gate_format, render,
)

SOURCE = "chcaa/dansk-ner"

# Danish names for the 18 OntoNotes types. These are the "plain" keys; the
# symbol schemes replace them wholesale.
DA = {
    "PERSON": "person", "ORGANIZATION": "organisation", "GPE": "sted",
    "LOCATION": "lokation", "FACILITY": "bygning", "PRODUCT": "produkt",
    "EVENT": "begivenhed", "WORK OF ART": "værk", "LAW": "lov",
    "LANGUAGE": "sprog", "NORP": "gruppe", "DATE": "dato", "TIME": "tidspunkt",
    "PERCENT": "procent", "MONEY": "beløb", "QUANTITY": "mængde",
    "ORDINAL": "ordenstal", "CARDINAL": "tal",
}
GLOSS = {
    "person": "personnavne", "organisation": "organisationer og virksomheder",
    "sted": "lande, byer og regioner", "lokation": "geografiske steder",
    "bygning": "bygninger og anlæg", "produkt": "produkter",
    "begivenhed": "begivenheder", "værk": "titler på værker",
    "lov": "love og paragraffer", "sprog": "sprog",
    "gruppe": "nationaliteter, religiøse eller politiske grupper",
    "dato": "datoer", "tidspunkt": "klokkeslæt og tidsrum",
    "procent": "procentangivelser", "beløb": "pengebeløb",
    "mængde": "mål og mængder", "ordenstal": "ordenstal", "tal": "tal",
}


def held(name: str, frac: float, salt: str) -> bool:
    if frac <= 0:
        return False
    h = hashlib.sha1(f"{salt}:{name}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF < frac


def load_rows(split, min_len, max_len):
    from datasets import load_dataset
    out = []
    for r in load_dataset(SOURCE, split=split):
        # OFFSETS ARE RELATIVE TO THE UNSTRIPPED TEXT. Stripping first and
        # then slicing shifts every span by the number of leading characters
        # removed -- 1.3% of rows have leading whitespace, and it silently
        # turned '@KarinEAxelsson' into 'KarinEAxelsson '. Strip the leading
        # run explicitly and shift the offsets with it.
        raw = r.get("text") or ""
        lead = len(raw) - len(raw.lstrip())
        t = raw.strip()
        if not (min_len <= len(t) <= max_len):
            continue
        if re.search(r"https?://|www\.", t):
            continue
        if len(t.split()) < 5:
            continue
        if sum(c.isalpha() or c.isspace() for c in t) / len(t) < 0.6:
            continue
        g = defaultdict(list)
        offs = []
        for e in sorted(r["ents"] or [], key=lambda e: e["start"]):
            lab = DA.get(str(e.get("label", "")))
            a0, b0 = e["start"] - lead, e["end"] - lead
            s = t[a0:b0].strip()
            # spans are character slices, so verbatim by construction; the
            # only thing to guard is the flat-format delimiter
            if lab and s and "\n" not in s and "\t" not in s and s not in g[lab]:
                assert s in t, f"span {s!r} not in its own passage {t!r}"
                g[lab].append(s)
            if lab and s and "\n" not in s and "\t" not in s:
                offs.append((a0, b0, lab))
        out.append({"text": t, "gold": dict(g), "offs": offs,
                    "types": set(g), "domain": r.get("dagw_domain") or "ukendt",
                    "surf": {v.lower() for vs in g.values() for v in vs}})
    return out


# Span-wrap is a FAMILY, varying the delimiter pair, for the same reason the
# key-value formats do: format is an induction axis, so one wrapper would
# teach one wrapper. Holding a wrapper out then asks whether span-wrap
# transfers WITHIN its own family -- the question the key-value results left
# open, since there the trained and held-out members differed in kind.
SPAN_WRAPS = {
    "spans_angle":   ("<{k}>", "</{k}>"),
    "spans_bracket": ("[{k}]", "[/{k}]"),
    "spans_brace":   ("{{{k}}}", "{{/{k}}}"),
    "spans_paren":   ("({k})", "(/{k})"),
}


def render_spans(text, offs, types, keymap, wrap):
    """Reproduce the passage with the requested entities wrapped in place.

    Structurally different from every other format: the answer contains the
    whole input rather than only the extracted values, so a faithful answer
    cannot hallucinate. It is also the shape that has scored 0% on every
    checkpoint measured so far -- tags get opened and not closed -- which is
    why it belongs in training rather than only in an eval.

    Absence is silent: a requested type with no entities simply has no tags,
    so the NULL marker the other formats use does not apply.
    """
    o_t, c_t = SPAN_WRAPS[wrap]
    want = set(types)
    keep = sorted((a, b, lab) for a, b, lab in offs if lab in want)
    out, prev = [], 0
    for a, b, lab in keep:
        if a < prev:                 # overlapping annotation: keep the first
            continue
        k = keymap[lab]
        out.append(text[prev:a])
        out.append(o_t.format(k=k) + text[a:b] + c_t.format(k=k))
        prev = b
    out.append(text[prev:])
    return "".join(out)


_SENT = "\x00KEY\x00"


def _span_res(keys, wrap):
    """Regexes for a wrapper, built from the FORMATTED delimiter.

    Deriving them from the raw template is wrong: the templates carry
    .format() escaping, so "{{{k}}}" would produce a pattern matching a
    literal "{{person}}" while the renderer emits "{person}". Formatting with
    a sentinel first sidesteps the escaping entirely.
    """
    o_t, c_t = SPAN_WRAPS[wrap]
    kpat = "|".join(re.escape(k) for k in keys)
    op = re.escape(o_t.format(k=_SENT)).replace(re.escape(_SENT), f"({kpat})")
    cl = re.escape(c_t.format(k=_SENT)).replace(re.escape(_SENT), r"\1")
    any_tag = "|".join(
        re.escape(t.format(k=_SENT)).replace(re.escape(_SENT), f"(?:{kpat})")
        for t in (o_t, c_t))
    return re.compile(op + "(.*?)" + cl, re.S), re.compile(any_tag)


def parse_spans(rendered, keys, wrap):
    """-> ({key: [values]}, stripped). `stripped` is the text with all tags
    removed; equality with the passage is the property that makes this format
    worth having, and is checked by the caller."""
    pair_re, tag_re = _span_res(keys, wrap)
    g = defaultdict(list)
    for k, v in pair_re.findall(rendered):
        g[k].append(re.sub(r"\s+", " ", v).strip().lower())
    return dict(g), tag_re.sub("", rendered)


def _gate_spans(wrap):
    """Round-trip + break control, per wrapper."""
    text = "Anna Hansen bor i Aarhus."
    offs = [(0, 11, "person"), (18, 24, "sted")]
    km = {"person": "person", "sted": "sted"}
    r = render_spans(text, offs, ["person", "sted"], km, wrap)
    got, stripped = parse_spans(r, set(km.values()), wrap)
    assert stripped == text, f"{wrap}: strip gave {stripped!r}"
    assert got == {"person": ["anna hansen"], "sted": ["aarhus"]}, f"{wrap}: {got}"
    # a dropped closing delimiter must stop it parsing as a well-formed pair
    o_t, c_t = SPAN_WRAPS[wrap]
    broken = r.replace(c_t.format(k="person"), "", 1)
    got2, _ = parse_spans(broken, set(km.values()), wrap)
    assert got2.get("person") != ["anna hansen"], f"{wrap}: break undetected"


def describe(types, keymap, fmt, empty=NULL):
    """Spell the task out in words, for instruction-mode rows."""
    named = ", ".join(f"{keymap[t]} ({GLOSS[t]})" for t in types)
    if fmt in SPAN_WRAPS:
        o_t, c_t = SPAN_WRAPS[fmt]
        ex = keymap[types[0]]
        return (f"Markér {named} i teksten. "
                f"Gengiv hele teksten ordret og sæt tags omkring hver enhed "
                f"på formen {o_t.format(k=ex)}...{c_t.format(k=ex)}. "
                f"Er der ingen enheder af en type, så sæt ingen tags for den.")
    shape = render({t: [f"<{keymap[t]}>"] for t in types}, types, keymap, fmt)
    first = shape.split("\n")[0]
    return (f"Find {named} i teksten. "
            f"Svar med én linje per fundet enhed på formen \"{first}\". "
            f"Findes en type ikke i teksten, så skriv \"{empty}\" som værdi. "
            f"Gengiv enhederne præcis som de står i teksten.")


def build_row(rng, target, pool, types, keymap, fmt, mode, shots):
    gold = {t: target["gold"].get(t, []) for t in types}
    if not any(gold.values()):
        return None                      # an all-empty answer teaches nothing

    picks = []
    if mode in ("icl", "both"):
        # anti-leak: an exemplar must not hand over a span the target needs
        tl = target["text"].lower()
        cands = [d for d in pool
                 if d is not target and not (d["surf"] & target["surf"])
                 and not any(s in tl for s in d["surf"])]
        # WELL-POSEDNESS: with no instruction, a requested type whose meaning
        # is never demonstrated is unanswerable -- the model cannot know what
        # `gruppe` or `alfa` denotes. Cover every type the ANSWER uses.
        need = {t for t, v in gold.items() if v}
        covered = set()
        rng.shuffle(cands)
        for d in cands:
            if len(picks) >= shots:
                break
            gain = (d["types"] & need) - covered
            if gain or (len(picks) + len(need - covered) < shots):
                picks.append(d)
                covered |= d["types"]
        if need - covered:
            return None
        # if the answer uses the empty marker, some exemplar must show it.
        # Not applicable to spans, where an absent type simply has no tags.
        if fmt not in SPAN_WRAPS and any(not v for v in gold.values()):
            if not any(any(t not in d["types"] for t in types) for d in picks):
                return None
        if len(picks) != shots:
            return None
        # no duplicate or nested passages
        ps = [d["text"] for d in picks] + [target["text"]]
        if len(set(ps)) != len(ps):
            return None

    parts = []
    if mode in ("instruction", "both"):
        parts.append(describe(types, keymap, fmt))
    if picks:
        parts.append("Eksempler:")
        for d in picks:
            if fmt in SPAN_WRAPS:
                shown = render_spans(d["text"], d["offs"], types, keymap, fmt)
            else:
                dg = {t: d["gold"].get(t, []) for t in types}
                shown = render(dg, types, keymap, fmt)
            parts.append(f'Tekst:\n{d["text"]}\nSvar: {shown}')
    parts.append(f'Tekst:\n{target["text"]}\nSvar:')
    if fmt in SPAN_WRAPS:
        answer = render_spans(target["text"], target["offs"], types, keymap, fmt)
        got, stripped = parse_spans(answer, set(keymap.values()), fmt)
        # the whole point of this format: stripping the tags must give the
        # passage back, exactly
        if stripped != target["text"]:
            return None
    else:
        answer = render(gold, types, keymap, fmt)
        if canon(answer, fmt, set(keymap.values())) is None:
            return None
    return {
        "messages": [{"role": "user", "content": "\n\n".join(parts)},
                     {"role": "assistant", "content": answer}],
        "meta": {"mode": mode, "format": fmt, "shots": len(picks),
                 "types": "|".join(types), "n_types": len(types),
                 "symbols": keymap.get("_scheme", "none"),
                 "domain": target["domain"],
                 "n_ents": sum(len(v) for v in gold.values())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-held", type=int, default=1500,
                    help="rows per held-out axis (format / type / domain)")
    ap.add_argument("--out", default="scratch/ner_sft")
    ap.add_argument("--formats", nargs="*", default=["all"])
    ap.add_argument("--held-formats", nargs="*",
                    default=["bracket_pair", "kv_eq", "spans_brace"])
    ap.add_argument("--mode-frac", nargs=3, type=float, default=[0.5, 0.2, 0.3],
                    metavar=("ICL", "INSTRUCTION", "BOTH"))
    ap.add_argument("--sym-frac", type=float, default=0.4)
    ap.add_argument("--min-len", type=int, default=25)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--exclude-src", default=None,
                    help="HF dataset id or JSONL whose PASSAGES to drop before "
                         "generating. Pass the earlier build to get rows that "
                         "share no text with it. DANSK train holds 11,740 "
                         "passages and the v1 SFT build used 7,054, so ~4,686 "
                         "remain for a genuinely unseen GRPO split -- rows "
                         "the policy cannot have memorised.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show", type=int, default=2)
    args = ap.parse_args()

    fmts = (sorted(FORMATS) + sorted(SPAN_WRAPS)) \
        if args.formats == ["all"] else list(args.formats)
    for f in fmts:
        if f in SPAN_WRAPS:       # needs a passage, so it has its own gate
            _gate_spans(f)
        else:
            gate_format(f)
    held_f = [f for f in fmts if f in args.held_formats]
    seen_f = [f for f in fmts if f not in held_f]
    held_t = set()
    assert seen_f, "every format held out"

    train_rows = load_rows("train", args.min_len, args.max_len)
    if args.exclude_src:
        import json as _json
        import re as _re
        _BL = _re.compile(r"Tekst:\n(.*?)\nSvar:", _re.S)
        if args.exclude_src.endswith(".jsonl"):
            _src = [_json.loads(l) for l in open(args.exclude_src) if l.strip()]
        else:
            from datasets import load_dataset as _ld
            _src = list(_ld(args.exclude_src, "default", split="train"))
        used = {x.strip() for r in _src
                for x in _BL.findall(r["messages"][0]["content"])}
        before = len(train_rows)
        train_rows = [r for r in train_rows if r["text"].strip() not in used]
        print(f"--exclude-src {args.exclude_src}: {len(used)} passages used "
              f"there; train {before} -> {len(train_rows)} unseen", flush=True)
        assert train_rows, "exclusion left no train passages"
    dev_rows = load_rows("dev", args.min_len, args.max_len)
    # Every eval split is built from DANSK's own TEST text. An earlier version
    # drew the held-out-format and held-out-type rows from train passages, so
    # only the axis differed and the text itself had been trained on -- the
    # eval was weaker than it looked. Sourcing them from test means each eval
    # row varies one axis AND sits on unseen text, and `eval` stays comparable
    # to published DANSK numbers. dev is kept for selection so test is never
    # tuned on.
    test_rows = load_rows("test", args.min_len, args.max_len)
    print(f"{SOURCE}: train {len(train_rows)}, dev {len(dev_rows)}, "
          f"test {len(test_rows)} ({args.min_len}-{args.max_len} chars)")
    print(f"formats: train={seen_f} held={held_f}")

    all_types = sorted({t for r in train_rows for t in r["types"]})
    seen_t = [t for t in all_types if t not in held_t]

    def pick_types(rng, target, type_pool):
        present = [t for t in target["types"] if t in type_pool]
        if not present:
            return None
        k = rng.randint(1, min(4, len(present)))
        chosen = rng.sample(present, k)
        # add absent types so the empty marker is exercised, not just implied
        absent = [t for t in type_pool if t not in target["types"]]
        for _ in range(rng.randint(0, 2)):
            if absent:
                chosen.append(rng.choice([a for a in absent
                                          if a not in chosen] or absent))
        return sorted(set(chosen))

    def keymap_for(rng, types):
        if rng.random() < args.sym_frac:
            scheme = rng.choice(list(SYMBOLS))
            if len(types) <= len(SYMBOLS[scheme]):
                syms = list(SYMBOLS[scheme][:len(types)])
                rng.shuffle(syms)
                km = dict(zip(types, syms))
                km["_scheme"] = scheme
                return km
        km = {t: t for t in types}
        km["_scheme"] = "none"
        return km

    MODES = ["icl", "instruction", "both"]
    W = args.mode_frac

    def generate(rng, pool, n, type_pool, fmt_pool, dom_filter, seen_keys):
        rows, tried = [], 0
        cand = [r for r in pool if dom_filter(r["domain"])]
        if not cand:
            return rows
        while len(rows) < n and tried < n * 60:
            tried += 1
            target = rng.choice(cand)
            types = pick_types(rng, target, type_pool)
            if not types:
                continue
            km = keymap_for(rng, types)
            km_clean = {k: v for k, v in km.items() if k != "_scheme"}
            mode = rng.choices(MODES, weights=W)[0]
            shots = 0 if mode == "instruction" else rng.randint(1, 4)
            r = build_row(rng, target, cand, types, km_clean,
                          rng.choice(fmt_pool), mode, shots)
            if not r:
                continue
            r["meta"]["symbols"] = km["_scheme"]
            key = hashlib.md5((r["messages"][0]["content"]
                               + r["messages"][1]["content"]).encode()).hexdigest()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rows.append(r)
        return rows

    rng = random.Random(args.seed)
    seen_keys = set()
    keep = lambda d: True
    main_rows = generate(rng, train_rows, args.n, seen_t, seen_f,
                         keep, seen_keys)
    val_rows = generate(rng, dev_rows, args.n_val, seen_t, seen_f,
                        keep, seen_keys)
    ev_rows = generate(rng, test_rows, args.n_held, seen_t, seen_f,
                       keep, seen_keys)
    ev_fmt = generate(rng, test_rows, args.n_held, seen_t, held_f or seen_f,
                      keep, seen_keys) if held_f else []
    ev_typ = []

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # No held-out TYPE split: DANSK ships dev and test, so unseen TEXT is the
    # generalisation axis its owners intended, and holding types out only cost
    # train two of the eighteen. Format stays held out because the format is
    # our construction, not the dataset's.
    splits = [("train", main_rows), ("val", val_rows), ("eval", ev_rows),
              ("eval_format", ev_fmt)]
    for nm, rs in splits:
        if not rs:
            continue
        (out / f"{nm}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))
        print(f"-> {out / (nm + '.jsonl')}  ({len(rs)} rows)")

    tr = main_rows
    print()
    for nm, rs in splits[1:]:
        if not rs:
            continue
        st = {x for r in rs for x in r["meta"]["types"].split("|")}
        sf = {r["meta"]["format"] for r in rs}
        sd = {r["meta"]["domain"] for r in rs}
        ot = len(st & {x for r in tr for x in r["meta"]["types"].split("|")})
        of = len(sf & {r["meta"]["format"] for r in tr})
        od = len(sd & {r["meta"]["domain"] for r in tr})
        print(f"  {nm:<12} n={len(rs):<6} shares {ot} types, {of} formats, "
              f"{od} domains with train")
    for k in ("mode", "format", "symbols", "shots", "n_types", "domain"):
        c = Counter(r["meta"][k] for r in tr)
        print(f"  {k:<9}" + "  ".join(f"{a}={b}" for a, b in
                                      sorted(c.items(), key=lambda x: -x[1])[:9]))

    for r in tr[:args.show]:
        m = r["meta"]
        print("\n" + "#" * 72)
        print(f"# mode={m['mode']} format={m['format']} symbols={m['symbols']} "
              f"shots={m['shots']} types={m['types']} domain={m['domain']}")
        print("#" * 72)
        print(r["messages"][0]["content"])
        print(">>> ASSISTANT <<<")
        print(r["messages"][1]["content"])


if __name__ == "__main__":
    main()
