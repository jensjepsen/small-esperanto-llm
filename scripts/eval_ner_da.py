"""Danish NER eval on dane_plus (held-out test split).

Entity-level micro P/R/F1 over (surface, type) pairs, case-insensitive, with a
per-row status breakdown and an optional readable markdown dump.

The completion parser is deliberately tolerant. The model answers with its own
key names (`navn`, `placering`, `årstal`), writes scalars where a list was
asked for, and repeats keys — a strict parser scores correct extractions as
zero. The tell for a broken scorer is tp==0 AND fp==0 together: that is a
parser discarding everything, not a model producing nothing.

Sentences with NO gold entities are included on purpose: they measure whether
the model invents entities when there are none, which is the failure mode a
JSON-shape reward cannot see.

Usage:
  python scripts/eval_ner_da.py --ckpt <repo-or-path> [--subfolder step-N]
                                [--md out.md] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"

# dane_plus uses OntoNotes-style labels with some annotation noise
# (ORG/ORGANIZATION, PER/PERSON, WORK OF ART/WORK_OF_ART). Fold to 4 buckets.
CANON = {"PERSON": "person", "PER": "person",
         "ORGANIZATION": "org", "ORG": "org",
         "GPE": "sted", "LOCATION": "sted", "LOC": "sted", "FACILITY": "sted",
         "DATE": "dato"}
TYPES = ["person", "org", "sted", "dato"]
DA_NAME = {"person": "PERSON", "org": "ORGANISATION", "sted": "STED", "dato": "DATO"}

# Must match NER_PROMPT in train_grpo_verifier.py — eval and train share it.
PROMPT = ('Find alle navngivne enheder i denne tekst:\n\n"{t}"\n\n'
          'Svar kun med JSON på formen '
          '{{"person": [], "org": [], "sted": [], "dato": []}} — '
          'personer under "person", organisationer under "org", '
          'steder og lande under "sted", datoer og årstal under "dato". '
          'Er der ingen af en slags, så lad listen være tom. '
          'Skriv enhederne præcis som de står i teksten.')

KEYMAP = {}
for _k in ("person", "personer", "people", "navn", "navne", "name", "names"):
    KEYMAP[_k] = "person"
for _k in ("org", "organisation", "organisationer", "organization",
           "organizations", "virksomhed", "virksomheder"):
    KEYMAP[_k] = "org"
for _k in ("sted", "steder", "places", "place", "placering", "lokation",
           "location", "locations", "land", "lande", "by", "byer"):
    KEYMAP[_k] = "sted"
for _k in ("dato", "datoer", "dates", "date", "aar", "år", "årstal",
           "aarstal", "year", "tid"):
    KEYMAP[_k] = "dato"

_KV_RE = re.compile(r'"([^"]+)"\s*:\s*(\[[^\]]*\]|"[^"]*"|[-\d.]+)')
_STR_RE = re.compile(r'"([^"]*)"')


def gold_of(row):
    text = row["text"]
    out = []
    for e in row["ents"] or []:
        raw = str(e.get("label", "")).upper()
        lab = CANON.get(raw) or CANON.get(raw.replace(" ", "_"))
        if lab is None:
            continue
        surf = text[e["start"]:e["end"]].strip()
        if surf:
            out.append((surf, lab))
    return out


def parse_pred(raw):
    """Duplicate-key, scalar and key-synonym tolerant."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return []
    out = []
    for key_raw, val_raw in _KV_RE.findall(m.group(0)):
        key = KEYMAP.get(key_raw.strip().lower())
        if key is None:
            continue
        v = val_raw.strip()
        if v.startswith("["):
            vals = _STR_RE.findall(v)
        elif v.startswith('"'):
            vals = [v.strip('"')]
        else:
            vals = [v]
        for x in vals:
            x = x.strip()
            if x and x not in ("[]", "[],"):
                out.append((x, key))
    return out


def render(ents):
    if not ents:
        return "_(ingen)_"
    by = defaultdict(list)
    for s, l in ents:
        if s not in by[l]:
            by[l].append(s)
    return "  ·  ".join(f"**{DA_NAME[l]}**: " + ", ".join(f'"{x}"' for x in by[l])
                        for l in TYPES if by[l])


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--md", default=None, help="write a readable markdown dump")
    ap.add_argument("--dump-jsonl", default=None)
    args = ap.parse_args()

    ds = load_dataset("KennethEnevoldsen/dane_plus", split=args.split)
    rows = [{"text": r["text"].strip(), "gold": gold_of(r)}
            for r in ds if r["text"].strip()]
    if args.limit:
        rows = rows[:args.limit]
    n_ent = sum(1 for r in rows if r["gold"])
    print(f"{len(rows)} sentences ({n_ent} with entities, "
          f"{len(rows)-n_ent} entity-free)", flush=True)

    kw = {"subfolder": args.subfolder} if args.subfolder else {}
    dt = {"fp32": torch.float32, "fp16": torch.float16,
          "bf16": torch.bfloat16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.ckpt, **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tag = args.ckpt + (f"/{args.subfolder}" if args.subfolder else "")
    print(f"loading {tag} ({args.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=dt, **kw).cuda().eval()
    eid = tok.convert_tokens_to_ids(END)
    eos = [tok.eos_token_id] + ([eid] if eid != tok.unk_token_id else [])

    prompts = [f"{USER}{PROMPT.format(t=r['text'])}{END}{ASST}" for r in rows]
    outs = []
    for i in range(0, len(prompts), args.batch_size):
        b = prompts[i:i + args.batch_size]
        enc = tok(b, return_tensors="pt", padding=True, add_special_tokens=False,
                  return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            g = model.generate(input_ids=enc["input_ids"],
                               attention_mask=enc["attention_mask"],
                               max_new_tokens=args.max_new, do_sample=False,
                               num_beams=1, pad_token_id=tok.pad_token_id,
                               eos_token_id=eos, repetition_penalty=1.1)
        pl = enc["input_ids"].shape[1]
        outs += [tok.decode(x[pl:], skip_special_tokens=True).strip() for x in g]
        if (i // args.batch_size) % 5 == 0 or i + args.batch_size >= len(prompts):
            print(f"  {min(i+args.batch_size, len(prompts))}/{len(prompts)}", flush=True)

    tp = fp = fn = 0
    per = defaultdict(lambda: [0, 0, 0])
    recs = []
    for r, o in zip(rows, outs):
        gold = {(s.lower(), l) for s, l in r["gold"]}
        pred_r = parse_pred(o)
        pred = {(s.lower(), l) for s, l in pred_r}
        hit, spur, miss = pred & gold, pred - gold, gold - pred
        tp += len(hit); fp += len(spur); fn += len(miss)
        for _, l in hit: per[l][0] += 1
        for _, l in spur: per[l][1] += 1
        for _, l in miss: per[l][2] += 1
        if not gold and not pred:
            st = "korrekt tom"
        elif not gold:
            st = f"opfundet ({len(spur)})"
        elif not miss and not spur:
            st = "eksakt"
        elif hit:
            st = f"delvis (tp={len(hit)} fp={len(spur)} fn={len(miss)})"
        else:
            st = f"forfejlet (fp={len(spur)} fn={len(miss)})"
        recs.append({"text": r["text"], "gold": r["gold"], "pred": pred_r,
                     "raw": o, "status": st, "hit": sorted(hit),
                     "spurious": sorted(spur), "missed": sorted(miss)})

    P, R, F = prf(tp, fp, fn)
    stc = Counter(x["status"].split(" (")[0] for x in recs)
    ent_free = [x for x in recs if not x["gold"]]
    invented = sum(1 for x in ent_free if x["pred"])

    print(f"\n{'':<14}{'P':>8}{'R':>8}{'F1':>8}{'tp':>7}{'fp':>7}{'fn':>7}")
    print(f"{'micro':<14}{100*P:>8.1f}{100*R:>8.1f}{100*F:>8.1f}{tp:>7}{fp:>7}{fn:>7}")
    for t in TYPES:
        a, b, c = per[t]
        p, r, f = prf(a, b, c)
        print(f"{DA_NAME[t]:<14}{100*p:>8.1f}{100*r:>8.1f}{100*f:>8.1f}{a:>7}{b:>7}{c:>7}")
    print(f"\nstatus: {dict(stc)}")
    print(f"invented entities on entity-free sentences: "
          f"{invented}/{len(ent_free)}")

    if args.dump_jsonl:
        with open(args.dump_jsonl, "w", encoding="utf-8") as fh:
            for x in recs:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"-> {args.dump_jsonl}")

    if args.md:
        L = [f"# Dansk NER — dane_plus {args.split}", "",
             f"**Model:** `{tag}`  ",
             f"**Sætninger:** {len(rows)} ({n_ent} med entiteter, "
             f"{len(rows)-n_ent} uden)  ",
             f"**Afkodning:** greedy, repetition_penalty=1.1, "
             f"max_new_tokens={args.max_new}, {args.dtype}", "",
             "| | P | R | F1 | tp | fp | fn |", "|---|---|---|---|---|---|---|",
             f"| **micro** | {100*P:.1f} | {100*R:.1f} | **{100*F:.1f}** | {tp} | {fp} | {fn} |"]
        for t in TYPES:
            a, b, c = per[t]
            p, r, f = prf(a, b, c)
            L.append(f"| {DA_NAME[t]} | {100*p:.1f} | {100*r:.1f} | {100*f:.1f} | {a} | {b} | {c} |")
        L += ["", "| status | antal |", "|---|---|"]
        for k, v in stc.most_common():
            L.append(f"| {k} | {v} |")
        L += ["", f"Opfundne entiteter på entitetsfrie sætninger: "
                  f"**{invented}/{len(ent_free)}**", "", "---", ""]
        for i, x in enumerate(recs, 1):
            L += [f"## {i}. {x['status']}", "", f"**Tekst:** {x['text']}", "",
                  f"**Guld:** {render(x['gold'])}", "",
                  f"**Model:** {render(x['pred'])}", ""]
            if x["missed"]:
                L += ["**Overset:** " + ", ".join(f'"{s}" ({DA_NAME[l]})'
                                                  for s, l in x["missed"]), ""]
            if x["spurious"]:
                L += ["**Falske:** " + ", ".join(f'"{s}" ({DA_NAME[l]})'
                                                 for s, l in x["spurious"][:12]), ""]
            L += ["<details><summary>rå output</summary>", "", "```json",
                  x["raw"][:1200], "```", "</details>", ""]
        Path(args.md).write_text("\n".join(L), encoding="utf-8")
        print(f"-> {args.md}")


main()
