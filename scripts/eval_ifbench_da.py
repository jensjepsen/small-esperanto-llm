"""IFBench-DA — runs the Allen AI IFBench (2025) verifier chain against
`jensjepsen/ifbench-da-v1` (Danish translations of the 300 IFBench test
prompts, English anchor tokens preserved).

Reports two metrics per constraint level (loose / strict):
  - prompt-level accuracy (all constraints in a prompt pass)
  - instruction-level accuracy (individual constraints)

Mirrors `scripts/eval_ifeval_da.py` — same loose-transform variant set
and per-family breakdown, but wired to the vendored `ifbench_ai2`
package instead of `ifeval_google`. IFBench verifiers do not use a
`build_description(**kwargs)` interface; they take `**kwargs` directly
in `check_following` in most cases (see `instructions.py`). The lookup
loop below defers to the class's own `build_description` if it exists,
otherwise just instantiates with `is_random=False` and passes kwargs
straight to `check_following`.

Usage:
    uv run python scripts/eval_ifbench_da.py --ckpt CKPT [--batch-size 32]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ifbench_ai2 import instructions_registry as reg  # noqa: E402

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"


def _clean_kwargs(raw):
    """Drop keys whose value is None — IFBench kwargs schema is a giant
    union (every possible arg for every verifier), most are null per row."""
    if not raw:
        return {}
    return {k: v for k, v in raw.items() if v is not None}


def build_instructions(row):
    """Instantiate every constraint checker for this row.
    Same pattern as `eval_ifeval_da.py`: `build_description(**kwargs)`
    then `check_following(value)`."""
    out = []
    for i, name in enumerate(row["instruction_id_list"]):
        cls = reg.INSTRUCTION_DICT.get(name)
        if cls is None:
            print(f"  WARN: unregistered verifier id: {name}", file=sys.stderr)
            continue
        raw = row["kwargs"][i] if i < len(row["kwargs"]) else {}
        kwargs = _clean_kwargs(raw)
        try:
            inst = cls(name)
        except TypeError:
            inst = cls()
        try:
            inst.build_description(**kwargs)
        except TypeError:
            # some verifiers take no kwargs; call plain
            try:
                inst.build_description()
            except Exception as e:
                print(f"  build_description skipped [{name}: {type(e).__name__}: {str(e)[:80]}]",
                      file=sys.stderr)
                continue
        except KeyError as e:
            # unsupported language etc.
            print(f"  build_description KeyError [{name}: {str(e)[:80]}]", file=sys.stderr)
            continue
        out.append((name, inst))
    return out


def _try_check(inst, resp):
    try:
        return bool(inst.check_following(resp))
    except Exception as e:
        print(f"  check_following crash [{type(e).__name__}: {str(e)[:100]}]",
              file=sys.stderr)
        return False


def score_row(response: str, insts):
    """Return (strict_flags, loose_flags) matched to insts order."""
    strict_flags = []
    loose_flags = []
    for _, inst in insts:
        strict_flags.append(_try_check(inst, response))
        variants = [
            response,
            response.strip(),
            "\n".join(response.split("\n")[1:]).strip() if "\n" in response else response,
            response.replace("*", ""),
        ]
        loose = False
        for v in variants:
            if _try_check(inst, v):
                loose = True
                break
        loose_flags.append(loose)
    return strict_flags, loose_flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--dataset", default="jensjepsen/ifbench-da-v1",
                    help="HF dataset id or `load_from_disk` path")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None,
                    help="alias for -n / --n")
    ap.add_argument("-n", "--n", dest="n", type=int, default=None)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--dump-jsonl", default=None,
                    help="If set, per-row (prompt, kwargs, gen, verdicts) go here.")
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32,
             "bf16": torch.bfloat16}[args.dtype]
    n_cap = args.n if args.n is not None else args.limit

    tok_kw = {"subfolder": args.subfolder} if args.subfolder else {}
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.ckpt, **tok_kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tag = args.ckpt + (f"/{args.subfolder}" if args.subfolder else "")
    print(f"loading {tag}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=dtype, **tok_kw).cuda().eval()
    end_id = tok.convert_tokens_to_ids(END)
    eos_ids = [tok.eos_token_id] + ([end_id] if end_id != tok.unk_token_id else [])

    # Load DA dataset — try HF hub, fall back to local dir
    if Path(args.dataset).exists():
        from datasets import load_from_disk
        ds = load_from_disk(args.dataset)
    else:
        ds = load_dataset(args.dataset, split="train")
    if n_cap:
        ds = ds.select(range(min(n_cap, len(ds))))
    print(f"rows: {len(ds)}", flush=True)

    prompts = [f"{USER}{r['prompt']}{END}{ASST}" for r in ds]
    all_insts = [build_instructions(r) for r in ds]
    n_missing = sum(1 for insts in all_insts if not insts)
    print(f"prompts w/ 0 valid verifiers: {n_missing}", flush=True)

    prompt_strict_ok = 0; prompt_loose_ok = 0
    inst_strict_ok = 0; inst_loose_ok = 0
    inst_strict_tot = 0
    prompts_scored = 0
    per_constraint = defaultdict(lambda: [0, 0])
    per_subtype = defaultdict(lambda: [0, 0])
    dump_fh = None
    if args.dump_jsonl:
        Path(args.dump_jsonl).parent.mkdir(parents=True, exist_ok=True)
        dump_fh = open(args.dump_jsonl, "w")

    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=False,
                  return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False, num_beams=1,
                pad_token_id=tok.pad_token_id, eos_token_id=eos_ids,
                repetition_penalty=1.1,
            )
        plen = enc["input_ids"].shape[1]
        for j, row in enumerate(gen):
            row_ix = i + j
            resp = tok.decode(row[plen:], skip_special_tokens=True).strip()
            insts = all_insts[row_ix]
            if not insts:
                continue
            s, l = score_row(resp, insts)
            prompt_strict_ok += all(s)
            prompt_loose_ok += all(l)
            inst_strict_ok += sum(s)
            inst_loose_ok += sum(l)
            inst_strict_tot += len(s)
            prompts_scored += 1
            for (name, _), ok_s in zip(insts, s):
                fam = name.split(":")[0]
                per_constraint[fam][0] += ok_s
                per_constraint[fam][1] += 1
                per_subtype[name][0] += ok_s
                per_subtype[name][1] += 1
            if dump_fh is not None:
                src = ds[row_ix]
                dump_fh.write(json.dumps({
                    "key": src.get("key"),
                    "prompt": src["prompt"],
                    "instruction_id_list": src["instruction_id_list"],
                    "kwargs": src["kwargs"],
                    "generation": resp,
                    "strict_flags": s,
                    "loose_flags": l,
                    "prompt_strict": all(s),
                    "prompt_loose": all(l),
                }, ensure_ascii=False) + "\n")
                dump_fh.flush()
        done = i + len(batch)
        if prompts_scored:
            print(f"  {done}/{len(prompts)}  "
                  f"prompt-strict={100*prompt_strict_ok/prompts_scored:.1f}%  "
                  f"prompt-loose={100*prompt_loose_ok/prompts_scored:.1f}%  "
                  f"inst-strict={100*inst_strict_ok/inst_strict_tot:.1f}%  "
                  f"inst-loose={100*inst_loose_ok/inst_strict_tot:.1f}%",
                  flush=True)

    if dump_fh is not None:
        dump_fh.close()
        print(f"per-row dump → {args.dump_jsonl}", flush=True)

    n = prompts_scored
    print(f"\n=== ifbench-da  n={n} ===")
    if n == 0:
        print("(no prompts scored — bailing)")
        return
    print(f"  prompt-strict:  {prompt_strict_ok}/{n} = {100*prompt_strict_ok/n:.1f}%")
    print(f"  prompt-loose:   {prompt_loose_ok}/{n} = {100*prompt_loose_ok/n:.1f}%")
    print(f"  inst-strict:    {inst_strict_ok}/{inst_strict_tot} = {100*inst_strict_ok/inst_strict_tot:.1f}%")
    print(f"  inst-loose:     {inst_loose_ok}/{inst_strict_tot} = {100*inst_loose_ok/inst_strict_tot:.1f}%")
    agg = (prompt_strict_ok + prompt_loose_ok) / (2 * n) * 100
    print(f"  aggregate (avg of two prompt metrics): {agg:.2f}%")
    print(f"\nPer-family (strict):")
    for fam in sorted(per_constraint, key=lambda f: -per_constraint[f][1]):
        p, t = per_constraint[fam]
        print(f"  {100*p/t:5.1f}%  {p:3d}/{t:3d}  {fam}")
    print(f"\nPer-subtype (strict):")
    for name in sorted(per_subtype, key=lambda n: -per_subtype[n][1]):
        p, t = per_subtype[name]
        print(f"  {100*p/t:5.1f}%  {p:3d}/{t:3d}  {name}")


if __name__ == "__main__":
    main()
