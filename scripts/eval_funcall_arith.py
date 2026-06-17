"""Evaluate a model on funcall-arith eval data.

Loads a checkpoint, runs each user prompt, extracts `<<expr>>` calls
from the assistant output, grades per-call against the expected calls
recorded in the JSONL.

Reports:
  - overall:           record-level exact match (all calls correct + nothing extra),
                       call-level exact / op-only / missing / extra
  - by pattern:        accuracy per category
  - by step count:     1-step vs 2-step vs 3-step
  - by op type:        +, -, *, /

Eval data format: same JSONL as `generate_funcall_arith.py` emits.
Each record must have:
  messages:        [{role: user, content: ...}, {role: assistant, content: ...}]
  expected_calls:  ["<<5+3>>", "<<#1-2>>", ...]
  category:        "funcall_arith:<pattern>"
  n_steps:         int

Usage:
    uv run python scripts/eval_funcall_arith.py \\
        --checkpoint runs/large/checkpoint-44000-sft-v67-funcall/checkpoint-NNNN \\
        --eval data/sft/funcall_arith_eval.jsonl \\
        [--limit 200] [--max-new-tokens 64] [--show-misses]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esperanto_lm.data import _morpheme_preprocess

# Reuse the grader from the generator so eval and training are
# guaranteed to use the same call-extraction logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_funcall_arith import extract_calls, grade, execute_calls  # noqa: E402


USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
SPECIAL = (USER, ASST, END)


def preprocess_chat(text):
    """Mirror `train_sft.py:preprocess_and_tokenize` (and `eval_icl.py`):
    split on chat special tokens, `.strip()` and morpheme-preprocess
    non-empty content parts, preserve whitespace-only parts, rejoin
    with spaces. Required so the eval-time token sequence matches
    what the trainer fed the model."""
    pat = "(" + "|".join(re.escape(t) for t in SPECIAL) + ")"
    parts = re.split(pat, text)
    out = []
    for p in parts:
        if p in SPECIAL:
            out.append(p)
        elif p.strip():
            out.append(_morpheme_preprocess(p.strip()))
        else:
            out.append(p)
    return " ".join(out)


def build_prompt(tok, user_msg):
    chat = preprocess_chat(f"{USER} {user_msg} {ASST}")
    return tok(chat, return_tensors="pt", return_token_type_ids=False)


def decode_completion(tok, completion_ids, end_id):
    toks = tok.convert_ids_to_tokens(completion_ids)
    cleaned = []
    for t in toks:
        if t == END:
            break
        if t in ("<s>", "</s>", "<pad>", "<unk>", USER, ASST):
            continue
        cleaned.append(t)
    return "".join(" " if t == "<w>" else t for t in cleaned).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--tokenizer", default="tokenizer_morpheme")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate at most N records (0 = all)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--show-misses", action="store_true",
                    help="print every record where the model output differs from expected")
    ap.add_argument("--show-n-misses", type=int, default=20,
                    help="cap on miss printouts when --show-misses is on")
    args = ap.parse_args()

    print(f"Loading {args.checkpoint} ...")
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    tok.add_special_tokens({"additional_special_tokens": list(SPECIAL)})
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.float16).cuda().eval()
    model.resize_token_embeddings(len(tok))
    end_id = tok.convert_tokens_to_ids(END)

    records = [json.loads(l) for l in args.eval.open()]
    if args.limit:
        records = records[:args.limit]
    print(f"Evaluating {len(records)} records.\n")

    # Aggregates
    rec_exact = 0  # entire record correct: all calls right, no extras
    call_exact = 0
    call_op_only = 0
    call_structural = 0
    call_missing = 0
    call_extra = 0
    total_expected_calls = 0
    total_actual_calls = 0

    # Value-level: did the call's computed answer match expected?
    # (lenient — model may emit a structurally-different call that
    #  happens to compute the same value, e.g. 5+3 vs 3+5)
    call_val_exact = 0     # per-call value match (computed)
    final_exact = 0        # final answer match (last call's value)
    final_attempted = 0    # records where model emitted at least one parseable call

    by_pattern = defaultdict(lambda: {"n": 0, "rec_exact": 0,
                                      "call_exact": 0, "call_total": 0,
                                      "call_val_exact": 0,
                                      "final_exact": 0})
    by_nsteps = defaultdict(lambda: {"n": 0, "rec_exact": 0,
                                     "final_exact": 0})
    by_op = defaultdict(lambda: {"expected": 0, "exact": 0,
                                 "op_match": 0, "operands_match": 0})

    misses_shown = 0

    for i, rec in enumerate(records):
        user = rec["messages"][0]["content"]
        expected_calls = rec["expected_calls"]
        expected_text = " ".join(expected_calls)
        category = rec.get("category", "?")
        n_steps = rec.get("n_steps", len(expected_calls))

        inputs = build_prompt(tok, user).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=end_id,
                pad_token_id=tok.pad_token_id or end_id)
        gen = out[0][inputs["input_ids"].shape[-1]:].tolist()
        pred_text = decode_completion(tok, gen, end_id)

        g = grade(expected_text, pred_text)

        # Record-level exact: all expected matched + no missing + no extra
        rec_ok = (g["exact"] == g["n_expected"] and g["missing"] == 0
                  and g["extra"] == 0)
        if rec_ok:
            rec_exact += 1

        call_exact      += g["exact"]
        call_op_only    += g["op_only"]
        call_structural += g["structural"]
        call_missing    += g["missing"]
        call_extra      += g["extra"]
        total_expected_calls += g["n_expected"]
        total_actual_calls   += g["n_actual"]

        # Execute both sides to value-grade.
        exp_calls = extract_calls(expected_text)
        act_calls = extract_calls(pred_text)
        exp_vals = execute_calls(exp_calls)
        act_vals = execute_calls(act_calls)

        # Per-call value match across the aligned prefix.
        per_call_val_matches = 0
        for i in range(min(len(exp_vals), len(act_vals))):
            if exp_vals[i] is not None and act_vals[i] == exp_vals[i]:
                per_call_val_matches += 1
        call_val_exact += per_call_val_matches

        # Final answer: the LAST expected call's value must match the
        # value the model would produce at the same step index — i.e.
        # the actual call at position n_expected-1, NOT the model's
        # last call (which may be a spurious extra).
        final_ok = False
        if exp_vals and exp_vals[-1] is not None:
            final_attempted += 1
            target_idx = len(exp_vals) - 1
            if target_idx < len(act_vals) and act_vals[target_idx] == exp_vals[-1]:
                final_ok = True
                final_exact += 1

        # Per-pattern
        bp = by_pattern[category]
        bp["n"] += 1
        bp["rec_exact"] += int(rec_ok)
        bp["call_exact"] += g["exact"]
        bp["call_total"] += g["n_expected"]
        bp["call_val_exact"] += per_call_val_matches
        bp["final_exact"] += int(final_ok)

        # Per-step-count
        bn = by_nsteps[n_steps]
        bn["n"] += 1
        bn["rec_exact"] += int(rec_ok)
        bn["final_exact"] += int(final_ok)

        # Per-op (using expected calls only; that's the denominator).
        # For multi-op calls (e.g. `[[100-50-30-15]]`), each op in the
        # chain counts separately toward its operator's totals — so the
        # by_op tallies measure per-operator-occurrence accuracy across
        # the whole expression, not just the first op.
        for pc, (_, e_ops) in zip(g["per_call"], extract_calls(expected_text)):
            for op in e_ops:
                by_op[op]["expected"] += 1
                if pc["op_match"] and pc["operands_match"]:
                    by_op[op]["exact"] += 1
                if pc["op_match"]:
                    by_op[op]["op_match"] += 1
                if pc["operands_match"]:
                    by_op[op]["operands_match"] += 1

        if args.show_misses and not rec_ok and misses_shown < args.show_n_misses:
            print(f"--- miss #{misses_shown+1} [{category}] ---")
            print(f"  Q: {user[:160]}{'...' if len(user) > 160 else ''}")
            print(f"  expected: {expected_text}")
            print(f"  pred:     {pred_text[:200]}")
            misses_shown += 1

    print("\n=== Overall ===")
    print(f"  records:       {len(records)}")
    print(f"  record-exact:  {rec_exact}/{len(records)} = "
          f"{100*rec_exact/len(records):.1f}%")
    print(f"  final-answer:  {final_exact}/{len(records)} = "
          f"{100*final_exact/len(records):.1f}%   "
          f"(last-call's computed value matches expected)")
    print(f"  calls:         expected={total_expected_calls}  "
          f"actual={total_actual_calls}")
    print(f"  call-exact:    {call_exact}/{total_expected_calls} = "
          f"{100*call_exact/max(1,total_expected_calls):.1f}%   "
          f"(op + operand structure match)")
    print(f"  call-value:    {call_val_exact}/{total_expected_calls} = "
          f"{100*call_val_exact/max(1,total_expected_calls):.1f}%   "
          f"(executor produces the same value at each step)")
    print(f"  call-op-only:  {call_op_only} (op right, operands wrong)")
    print(f"  call-structural: {call_structural} (operands right, op wrong)")
    print(f"  call-missing:  {call_missing}")
    print(f"  call-extra:    {call_extra}")

    print("\n=== By pattern ===")
    for cat in sorted(by_pattern):
        b = by_pattern[cat]
        rec_pct = 100 * b["rec_exact"] / b["n"]
        call_pct = 100 * b["call_exact"] / max(1, b["call_total"])
        val_pct = 100 * b["call_val_exact"] / max(1, b["call_total"])
        final_pct = 100 * b["final_exact"] / b["n"]
        short = cat.replace("funcall_arith:", "")
        print(f"  {short:18s} rec={rec_pct:5.1f}%  final={final_pct:5.1f}%  "
              f"call-exact={call_pct:5.1f}%  call-val={val_pct:5.1f}%")

    print("\n=== By step count ===")
    for k in sorted(by_nsteps):
        b = by_nsteps[k]
        rec_pct = 100 * b["rec_exact"] / b["n"]
        final_pct = 100 * b["final_exact"] / b["n"]
        print(f"  n_steps={k}  rec={rec_pct:.1f}%  final={final_pct:.1f}%  "
              f"({b['n']} records)")

    print("\n=== By operator ===")
    for op in ("+", "-", "*", "/"):
        b = by_op.get(op, {"expected": 0, "exact": 0, "op_match": 0, "operands_match": 0})
        if b["expected"] == 0:
            continue
        e = b["expected"]
        print(f"  {op}  exact={b['exact']}/{e} = {100*b['exact']/e:5.1f}%   "
              f"op_match={b['op_match']}/{e} = {100*b['op_match']/e:5.1f}%   "
              f"operands_match={b['operands_match']}/{e} = {100*b['operands_match']/e:5.1f}%")


if __name__ == "__main__":
    main()
