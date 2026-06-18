"""LFM tool-use distillation: capture multi-turn LFM ↔ calculator traces.

For each English question:
  1. Render the apply_chat_template with the calculator tool spec.
  2. Generate until LFM emits `<|tool_call_end|>` (a tool call) or `<|im_end|>`
     (a final answer).
  3. On tool call: parse the expression, safely eval with a calculator,
     inject a tool-response message, then continue from the assistant turn.
  4. Loop until LFM gives a final answer or max_turns is reached.
  5. Persist the full multi-turn trace as JSONL.

The output JSONL has tool calls/responses in English; a separate script (TBD)
will translate the surrounding prose en→eo via v5b for SFT.

The model never has to compute arithmetic — Python does the math, LFM just
learns when to delegate and how to use the result.
"""
from __future__ import annotations

import argparse
import ast
import json
import operator
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm


# ---------- safe calculator ----------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval(expr: str) -> float | int | None:
    """Evaluate an arithmetic expression with only +-*/^()% and numbers.

    Returns None on parse error, unsupported AST nodes, or runtime errors.
    Strips currency markers ($, €, £) and thousands commas before parsing.
    """
    expr = expr.replace("$", "").replace("€", "").replace("£", "")
    expr = expr.replace(",", "")
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _walk(node):
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_walk(node.left), _walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_walk(node.operand))
        raise ValueError(f"unsupported node: {ast.dump(node)}")

    try:
        result = _walk(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if isinstance(result, float) and result.is_integer():
        return int(result)
    if isinstance(result, float):
        # Round trailing-noise floats (e.g. 0.6 * 5 → 2.9999...)
        return round(result, 6)
    return result


# ---------- tool spec + token markers ----------

CALCULATOR_TOOL = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a single arithmetic expression and return the numeric result. Use for any non-trivial arithmetic in word problems.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression with +, -, *, /, parentheses, and decimal numbers. No variables, no units.",
                }
            },
            "required": ["expression"],
        },
    },
}]

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_RESPONSE = "<|tool_response|>"
TOOL_RESPONSE_END = "<|tool_response_end|>"
IM_END = "<|im_end|>"

# LFM2.5's tool call syntax: [function_name(arg=value, ...)]
# Expressions can themselves contain parens, e.g. expression="(48-3)*2".
# Use a balanced-paren / quoted-string scan rather than a naive [^)]* regex.
FUNCNAME_RE = re.compile(r"\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
ARG_RE = re.compile(r'(?P<k>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"(?P<v>[^"]*)"')


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract (name, kwargs) from a `[name(k="v", ...)]` body.

    Handles parens nested inside argument string literals, which a flat
    [^)]* regex would mishandle.
    """
    m = FUNCNAME_RE.search(text)
    if not m:
        return None
    name = m.group(1)
    # Walk from after `(` to the matching `)`, respecting "..." string literals.
    i = m.end()
    depth = 1
    in_str = False
    start = i
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\" and i + 1 < len(text):
                i += 2; continue
            if ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
        i += 1
    if depth != 0:
        return None
    args_str = text[start:i]
    args = {k: v for k, v in ARG_RE.findall(args_str)}
    return name, args


# ---------- source loaders ----------

def load_gsm8k(n: int, skip: int, split: str = "train") -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    out = []
    end = min(skip + n, len(ds))
    for i in range(skip, end):
        ans = ds[i]["answer"]
        m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", ans)
        gold = m.group(1).replace(",", "").rstrip(".") if m else None
        out.append({"i": i, "q_en": ds[i]["question"].strip(), "gold": gold})
    return out


SOURCE_LOADERS = {"gsm8k": load_gsm8k}


# ---------- the LFM ↔ tool loop ----------

def run_trace(model, tok, q: str, gold: str | None, max_turns: int,
              max_new_per_turn: int) -> dict:
    """Run a multi-turn LFM ↔ calculator loop for one question.

    Returns a dict with `messages` (the full chat trace including tool turns),
    `final_answer` (best extraction from the final assistant message), and
    `tool_calls` (list of (expr, result) tuples that fired).
    """
    pad_id = tok.pad_token_id or tok.eos_token_id
    # Resolve stop ids for tool-call-end and im-end. These may multi-token,
    # so we'll detect them by string match on the decoded text.
    system_prompt = (
        "You are a math problem solver with access to a calculator. For ANY "
        "arithmetic — even trivial — you MUST call the calculator tool. Do NOT "
        "compute in your head and do NOT write the result inline. "
        "Never combine multiple operations into one expression: call once per "
        "individual operation (use '5 + 3' or '8 / 2', not '(5 + 3) / 2'). "
        "After the final tool result, give a one-sentence answer."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": q},
    ]
    live_start = len(messages) - 1
    tool_calls = []

    for turn in range(max_turns):
        prompt = tok.apply_chat_template(
            messages,
            tools=CALCULATOR_TOOL,
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_per_turn,
                do_sample=False,
                pad_token_id=pad_id,
            )
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
        # Cut at the first im_end so we don't get the next turn's prompt
        if IM_END in gen:
            gen = gen.split(IM_END)[0]

        # Did LFM emit a tool call?
        if TOOL_CALL_START in gen and TOOL_CALL_END in gen:
            # Strip the wrapper, parse it, dispatch.
            pre, _, after_start = gen.partition(TOOL_CALL_START)
            body, _, after_end = after_start.partition(TOOL_CALL_END)
            # The assistant message we save includes the call wrapper so it
            # round-trips through apply_chat_template cleanly on next turn.
            assistant_content = pre + TOOL_CALL_START + body + TOOL_CALL_END
            messages.append({"role": "assistant", "content": assistant_content})

            parsed = parse_tool_call(body)
            if not parsed:
                tool_result = "ERROR: could not parse tool call"
            else:
                name, args = parsed
                if name != "calculator":
                    tool_result = f"ERROR: unknown tool '{name}'"
                else:
                    expr = args.get("expression", "").strip()
                    val = safe_eval(expr)
                    tool_result = str(val) if val is not None else "ERROR: invalid expression"
                    if val is not None:
                        tool_calls.append((expr, val))
            messages.append({"role": "tool", "content": tool_result})
            continue

        # No tool call → treat as final answer
        # Strip any leading whitespace / stray template tokens
        final = gen.strip()
        messages.append({"role": "assistant", "content": final})
        break
    else:
        # Hit max_turns without a final answer
        pass

    # Extract a final numeric answer from the last assistant message.
    # Heuristic order: #### N > \boxed{N} > the last tool call's result
    # (since the model usually delegates the final calc) > last currency
    # number ($X) > last number anywhere.
    last_assist = next((m for m in reversed(messages) if m["role"] == "assistant"), None)
    final_text = last_assist["content"] if last_assist else ""
    final_num = None
    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", final_text)
    if m:
        final_num = m.group(1).replace(",", "").rstrip(".")
    if final_num is None:
        m = re.search(r"\\boxed\{\s*\$?\s*([-+]?\d[\d,]*\.?\d*)\s*\}", final_text)
        if m:
            final_num = m.group(1).replace(",", "").rstrip(".")
    if final_num is None and tool_calls:
        # The last tool call's result is usually the final numeric answer.
        final_num = str(tool_calls[-1][1])
    if final_num is None:
        # Prefer currency-anchored numbers ($X) over bare numbers.
        m = re.findall(r"\$\s*([-+]?\d[\d,]*\.?\d*)", final_text)
        if m:
            final_num = m[-1].replace(",", "").rstrip(".")
    if final_num is None:
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", final_text)
        final_num = nums[-1].replace(",", "").rstrip(".") if nums else None

    # Strip the few-shot scaffolding so saved traces contain only the live Q
    # and its derived turns. Prepend the system prompt so the trace is
    # self-contained for SFT.
    saved_messages = [messages[0]] + messages[live_start:]
    return {
        "messages": saved_messages,
        "final_answer": final_num,
        "tool_calls": tool_calls,
        "answer_matches_gold": (gold is not None and final_num == gold),
    }


# ---------- driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfm-model", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--source", default="gsm8k", choices=list(SOURCE_LOADERS))
    ap.add_argument("--source-split", default="train")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=8,
                    help="Max LFM↔tool round-trips per question.")
    ap.add_argument("--max-new-per-turn", type=int, default=300)
    ap.add_argument("--gold-filter", action="store_true",
                    help="Drop traces whose final_answer != gold (gsm8k only).")
    ap.add_argument("--out", default="mt/runs/distill_lfm_tool.jsonl")
    ap.add_argument("--hf-cache", default="/tmp/hf-cache")
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HOME", args.hf_cache)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading source: {args.source} (n={args.n}, skip={args.skip})…")
    rows = SOURCE_LOADERS[args.source](args.n, args.skip, args.source_split)
    print(f"  {len(rows)} questions")

    print(f"Loading {args.lfm_model}…")
    tok = AutoTokenizer.from_pretrained(args.lfm_model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.lfm_model, dtype=torch.float16
    ).to("cuda").eval()
    print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume: skip rows whose i is already written
    done_indices: set[int] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done_indices.add(json.loads(line)["i"])
                except Exception:
                    pass
        print(f"Resume: {len(done_indices)} rows already in {out_path}")

    todo = [r for r in rows if r["i"] not in done_indices]
    if not todo:
        print("Nothing to do.")
        return
    print(f"  {len(todo)} new questions to run")

    t_start = time.perf_counter()
    written = matched = total_tool_calls = no_tool = 0
    with out_path.open("a") as fout:
        for row in tqdm(todo, desc="trace"):
            trace = run_trace(
                model, tok, row["q_en"], row.get("gold"),
                max_turns=args.max_turns,
                max_new_per_turn=args.max_new_per_turn,
            )
            keep = True
            if args.gold_filter and row.get("gold") is not None:
                keep = trace["answer_matches_gold"]
            if not keep:
                continue
            out_row = {
                "i": row["i"],
                "q_en": row["q_en"],
                "gold": row.get("gold"),
                "messages": trace["messages"],
                "final_answer": trace["final_answer"],
                "tool_calls": trace["tool_calls"],
                "answer_matches_gold": trace["answer_matches_gold"],
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            if trace["answer_matches_gold"]:
                matched += 1
            total_tool_calls += len(trace["tool_calls"])
            if not trace["tool_calls"]:
                no_tool += 1

    dt = time.perf_counter() - t_start
    print(f"\nWrote {written} rows in {dt:.0f}s ({written/max(1,dt):.2f} rows/s)")
    print(f"  gold-matched: {matched}/{written} ({100*matched/max(1,written):.1f}%)")
    print(f"  avg tool calls per row: {total_tool_calls/max(1,written):.2f}")
    print(f"  rows with zero tool calls: {no_tool}/{written}")


if __name__ == "__main__":
    main()
