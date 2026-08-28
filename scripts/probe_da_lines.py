"""Can these checkpoints produce "type: enhed" line output?

Companion to probe_spans.py, same 20 sentences (same seed), same k, so the
numbers are directly comparable across formats and across checkpoints.

NO few-shot. The earlier version of this probe put "person: Anna Hansen" in
the prompt and the model copied Anna Hansen into unrelated answers, so it was
partly measuring example-copying. Here the syntax is shown as the literal
shape "type: enhed" with no instance filled in — the same discipline the
spans re-run used.

Reported separately because they fail independently:
  FORMAT-VALID  at least one parseable "type: enhed" line, or a bare "ingen"
  GROUNDED      at least one emitted value actually occurs in the source text
                (the spans probe showed a model can learn the syntax token
                while never grounding it in the passage — that has to be
                visible, not hidden inside a format-validity number)
  CORRECT       the extracted (surface, type) set equals gold
"""
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
CANON = {"PERSON": "person", "PER": "person", "ORGANIZATION": "org", "ORG": "org",
         "GPE": "sted", "LOCATION": "sted", "LOC": "sted", "FACILITY": "sted",
         "DATE": "dato"}
INV = {"person": "person", "organisation": "org", "sted": "sted", "dato": "dato"}

PROMPT = ('Find alle personer, organisationer, steder og datoer i denne tekst:\n\n'
          '"{t}"\n\n'
          'Skriv én enhed per linje på formen "type: enhed", hvor type er '
          'person, organisation, sted eller dato. '
          'Findes der ingen enheder, så skriv kun "ingen". '
          'Skriv enhederne præcis som de står i teksten.')

_LINE = re.compile(r"^\s*[-*\d.\s]*(person|organisation|sted|dato)\s*:\s*(.+?)\s*$",
                   re.I | re.M)


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip().strip('"').lower()


def parse(out):
    """-> (pred_set | None, n_lines). None means no recognisable shape."""
    hits = _LINE.findall(out or "")
    if not hits:
        if re.search(r"\bingen\b", (out or "").lower()):
            return set(), 0          # explicit empty marker — valid
        return None, 0
    pred = set()
    for k, v in hits:
        v = v.strip().strip('"').strip("*").strip()
        if v and v.lower() not in ("ingen", "[]", "-"):
            pred.add((v.lower(), INV[k.lower()]))
    return pred, len(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    ds = load_dataset("KennethEnevoldsen/dane_plus", split="test")
    rows = []
    for r in ds:
        t = (r.get("text") or "").strip()
        if not t or len(t) > 200:
            continue
        g = set()
        for e in r["ents"] or []:
            lab = CANON.get(str(e.get("label", "")).upper())
            if lab:
                s = t[e["start"]:e["end"]].strip()
                if s:
                    g.add((s.lower(), lab))
        rows.append({"text": t, "gold": g})
    # identical seed/split logic to probe_spans.py -> identical 20 sentences
    rng = random.Random(5)
    withe = [r for r in rows if r["gold"]]
    noe = [r for r in rows if not r["gold"]]
    sub = rng.sample(withe, int(args.n * 0.7)) + rng.sample(noe, args.n - int(args.n * 0.7))
    print(f"{args.label or args.ckpt}\n{len(sub)} sentences "
          f"({sum(1 for r in sub if r['gold'])} with entities), k={args.k}, "
          f"temp={args.temperature}, NO few-shot\n", flush=True)

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.float32).cuda().eval()
    eid = tok.convert_tokens_to_ids(END)
    eos = [tok.eos_token_id] + ([eid] if eid != tok.unk_token_id else [])

    R = {"valid": 0, "grounded": 0, "correct": 0}
    S = {"valid": 0, "grounded": 0, "tot": 0}
    dump = []
    for i, r in enumerate(sub, 1):
        p = f"{USER}{PROMPT.format(t=r['text'])}{END}{ASST}"
        enc = tok([p] * args.k, return_tensors="pt", padding=True,
                  add_special_tokens=False, return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            g = model.generate(input_ids=enc["input_ids"],
                               attention_mask=enc["attention_mask"],
                               max_new_tokens=args.max_new, do_sample=True,
                               temperature=args.temperature, top_p=0.95,
                               pad_token_id=tok.pad_token_id, eos_token_id=eos,
                               repetition_penalty=1.1)
        pl = enc["input_ids"].shape[1]
        outs = [tok.decode(x[pl:], skip_special_tokens=True).strip() for x in g]
        low = r["text"].lower()
        a_v = a_g = a_c = False
        for o in outs:
            pred, _ = parse(o)
            S["tot"] += 1
            if pred is None:
                continue
            S["valid"] += 1
            a_v = True
            grounded = any(s in low for s, _ in pred)
            S["grounded"] += grounded
            a_g |= grounded
            a_c |= (pred == r["gold"])
        R["valid"] += a_v; R["grounded"] += a_g; R["correct"] += a_c
        dump.append({"text": r["text"], "gold": sorted(r["gold"]),
                     "samples": outs[:3]})
        if i % 5 == 0:
            print(f"  {i}/{len(sub)}", flush=True)

    n = len(sub)
    print(f"\n{'metric':<28}{'pass@k (rows)':>16}{'per-sample':>13}")
    for k_, lab in (("valid", "FORMAT-VALID"), ("grounded", "GROUNDED")):
        print(f"{lab:<28}{R[k_]:>8}/{n:<4} {100*R[k_]/n:>4.0f}%"
              f"{100*S[k_]/max(1,S['tot']):>11.0f}%")
    print(f"{'CORRECT (set==gold)':<28}{R['correct']:>8}/{n:<4} "
          f"{100*R['correct']/n:>4.0f}%")
    if args.dump:
        Path(args.dump).write_text(json.dumps(dump, ensure_ascii=False, indent=2))
        print(f"\n-> {args.dump}")


if __name__ == "__main__":
    main()
