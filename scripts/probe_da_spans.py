"""Can the v31 base ever produce inline span-tagged output?

Gate before building spans as a training format: GRPO needs reward spread
inside a group, so if the base never emits the shape, every sample scores zero
and there is no gradient.

No few-shot example. The previous version showed the model
"<person>Anna Hansen</person> bor i <sted>Aarhus</sted>" and it copied those
entities into answers about unrelated sentences, so the measurement was partly
counting example-copying. The format is described in words only.

Three separate things are reported, because they fail independently:
  HAS-TAGS   emitted at least one well-formed <type>...</type>
  FAITHFUL   stripping the tags recovers the source sentence
  VALID      both of the above (or, for an entity-free sentence, a faithful
             echo with no tags — the documented empty case)
FAITHFUL is the one that matters for the format's whole rationale: spans make
hallucination structurally impossible only if the model reproduces the text.
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

# Shows the literal tag syntax with "..." as the placeholder — the same
# structure the JSON prompt uses (show the shape, supply no instance). The
# previous version described the syntax in prose only ("skrevet i
# vinkelparenteser") and never put the characters <person> on screen, which
# conflated "can it emit spans" with "can it derive XML syntax from a Danish
# description". The version before THAT showed a real example and the model
# copied "Anna Hansen" into unrelated sentences.
PROMPT = ('Markér alle personer, organisationer, steder og datoer i denne tekst:\n\n'
          '"{t}"\n\n'
          'Gengiv hele teksten ordret, men sæt tags omkring hver enhed på formen '
          '<person>...</person>, <organisation>...</organisation>, '
          '<sted>...</sted>, <dato>...</dato>. '
          'Er der ingen enheder, så gengiv teksten uændret uden tags.')

_TAG = re.compile(r"<(person|organisation|sted|dato)>(.*?)</\1>", re.S)
_ANYTAG = re.compile(r"</?(person|organisation|sted|dato)>")


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip().strip('"').lower()


def analyse(out, text):
    tags = _TAG.findall(out or "")
    stripped = _ANYTAG.sub("", out or "")
    faithful = norm(stripped) == norm(text)
    ents = {(v.strip().lower(), INV[k]) for k, v in tags}
    return bool(tags), faithful, ents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
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
    rng = random.Random(5)
    withe = [r for r in rows if r["gold"]]
    noe = [r for r in rows if not r["gold"]]
    sub = rng.sample(withe, int(args.n * 0.7)) + rng.sample(noe, args.n - int(args.n * 0.7))
    print(f"{len(sub)} sentences ({sum(1 for r in sub if r['gold'])} with entities), "
          f"k={args.k}, temp={args.temperature}, NO few-shot\n", flush=True)

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.float32).cuda().eval()
    eid = tok.convert_tokens_to_ids(END)
    eos = [tok.eos_token_id] + ([eid] if eid != tok.unk_token_id else [])

    R = {"tags": 0, "faith": 0, "valid": 0, "correct": 0}
    S = {"tags": 0, "faith": 0, "valid": 0, "tot": 0}
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
        a_t = a_f = a_v = a_c = False
        for o in outs:
            has, faith, ents = analyse(o, r["text"])
            S["tot"] += 1
            S["tags"] += has; S["faith"] += faith
            valid = faith and (has or not r["gold"])
            S["valid"] += valid
            a_t |= has; a_f |= faith; a_v |= valid
            a_c |= (valid and ents == r["gold"])
        R["tags"] += a_t; R["faith"] += a_f; R["valid"] += a_v; R["correct"] += a_c
        dump.append({"text": r["text"], "gold": sorted(r["gold"]),
                     "samples": outs[:3]})
        if i % 5 == 0:
            print(f"  {i}/{len(sub)}", flush=True)

    n = len(sub)
    print(f"\n{'metric':<28}{'pass@k (rows)':>16}{'per-sample':>13}")
    for k_, lab in (("tags", "HAS-TAGS"), ("faith", "FAITHFUL"), ("valid", "VALID")):
        print(f"{lab:<28}{R[k_]:>8}/{n:<4} {100*R[k_]/n:>4.0f}%"
              f"{100*S[k_]/max(1,S['tot']):>11.0f}%")
    print(f"{'CORRECT (spans==gold)':<28}{R['correct']:>8}/{n:<4} "
          f"{100*R['correct']/n:>4.0f}%")
    if args.dump:
        Path(args.dump).write_text(json.dumps(dump, ensure_ascii=False, indent=2))
        print(f"\n-> {args.dump}")


if __name__ == "__main__":
    main()
