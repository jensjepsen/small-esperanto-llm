"""Full-turn probe on a calculator tool with GSM8K-style word problems.

Sibling of probe_fun_tools_fullturn.py. Same two conditions (free vs forced),
same prompt construction through the trainer's own renderer, same rule that
the tool is not in the corpus so nothing is memorised.

WHAT THIS ASKS THAT THE OTHER PROBE DOES NOT. The four fun tools are
one-argument lookups: the question names a value, the call carries it, the
answer reads a field back. A word problem has no value to carry. The model has
to turn prose into an EXPRESSION -- decide the operations and their order --
and the numbers in the question are deliberately not the answer. `3 poser à 12
æbler, spiser 5` puts 3, 12 and 5 in front of the model and wants 31.

So each item has two independent failure modes, and the probe separates them:

    the CALL     is the expression right?      (arithmetic reasoning)
    the ANSWER   did it read the result back?  (grounding)

A model can get the second right and the first wrong -- faithfully reporting
the answer to the wrong sum -- which is why `expected` is checked against the
tool's own output rather than against the reply text.

SINGLE CALL, COMPOUND EXPRESSION. The catalogue offers one `calculate`, so a
two-step problem must become one expression rather than a chain. That is the
harder formulation and the one that fails visibly; chaining is worth a
separate probe.
"""
import ast
import json
import operator
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sft_packed import format_conversation  # noqa: E402

CALL = re.compile(r"<\|tool_call\|>(.*?)(?:<\|/tool_call\|>|$)", re.S)

# ast, not eval(). The model writes this string and the probe runs it; eval()
# on model output is a hole, and the restricted walk also makes a malformed
# expression a readable error instead of a traceback.
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
        ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}


def _ev(node):
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("non-numeric constant")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.operand))
    raise ValueError(f"unsupported: {ast.dump(node)[:40]}")


def calculate(expression):
    """Danish prose writes 2,5; Python needs 2.5. Accept the comma so a
    correct expression is not marked wrong for a decimal separator."""
    expr = str(expression).replace("−", "-").replace("×", "*").replace("÷", "/")
    expr = re.sub(r"(?<=\d),(?=\d)", ".", expr)
    try:
        val = _ev(ast.parse(expr, mode="eval").body)
    except Exception as e:
        return {"expression": expression, "error": f"kunne ikke beregne: {e}"}
    if isinstance(val, float) and val == int(val):
        val = int(val)
    return {"expression": expression, "result": val}


IMPL = {"calculate": calculate}

CATALOG = [
    {"name": "calculate",
     "description": "Beregn et matematisk udtryk og få resultatet",
     "parameters": {"type": "object", "properties": {
         "expression": {"type": "string",
                        "description": "Udtrykket der skal beregnes, "
                                       "fx '12 * 3 - 5'"}},
         "required": ["expression"]},
     "returns": {"type": "object", "properties": {
         "expression": {"description": "Udtrykket der blev beregnet"},
         "result": {"description": "Resultatet af udtrykket"}}}},
]

# (question, expected result). Every one is multi-step, and in every one the
# numbers the question states are NOT the answer -- so an answer turn that
# echoes an argument is visibly wrong, the same trap the fun-tools probe uses.
QUESTIONS = [
    ("Anna køber 3 poser æbler med 12 æbler i hver. Hun spiser 5 af dem. "
     "Hvor mange æbler har hun tilbage?", 31),
    ("En bog koster 149 kr. Jens køber 4 bøger og får 50 kr i rabat. "
     "Hvor meget betaler han i alt?", 546),
    ("Et tog kører 80 km/t i 2,5 timer. Hvor mange kilometer kører det?", 200),
    ("Der er 24 elever i klassen. En tredjedel er syge. "
     "Hvor mange elever er i skole?", 16),
]

SENTINEL = "\x00"


def user_msg(q):
    cat = json.dumps(CATALOG, ensure_ascii=False)
    return {"role": "user", "content": f"Værktøjer:\n{cat}\n\n{q}"}


def free_prompt(msgs):
    return format_conversation(msgs) + " <|assistant|>"


def forced_prompt(msgs):
    return format_conversation(
        msgs + [{"role": "assistant", "content": ""},
                {"role": "tool_call", "content": SENTINEL}]).split(SENTINEL)[0]


def answer_prompt(msgs, reasoning, call, result):
    return free_prompt(msgs + [
        {"role": "assistant", "content": reasoning},
        {"role": "tool_call", "content": call},
        {"role": "tool_result", "content": result}])


def run_call(raw):
    """Parse a generated call and execute it. Returns (call_json, result_json)."""
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw.strip())
    except Exception:
        return None, None
    name = obj.get("name")
    args = obj.get("arguments") or {}
    if name not in IMPL or not isinstance(args, dict):
        return json.dumps(obj, ensure_ascii=False), None
    try:
        res = IMPL[name](**args)
    except Exception as e:
        res = {"error": f"{type(e).__name__}: {e}"}
    return (json.dumps(obj, ensure_ascii=False),
            json.dumps(res, ensure_ascii=False))


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/ckpts/toolmix_1362"
    # Optional SUBFOLDER, so a checkpoint can be read straight off the Hub
    # instead of copied down first. The run's watcher already uploads every
    # best snapshot, so the transfer is paid for either way -- and scp gives
    # no integrity check, which is how a probe ended up pointed at a 0-byte
    # tokenizer.json that would have loaded and probed something anyway.
    #   probe.py jensjepsen/danish-lm-400m-sft-toolmix-mid step-1816-agg-0.806
    sub = sys.argv[2] if len(sys.argv) > 2 else None
    kw = {"subfolder": sub} if sub else {}
    print(f"ckpt: {ckpt}" + (f" [{sub}]" if sub else "") + "\n", flush=True)
    tok = AutoTokenizer.from_pretrained(ckpt, **kw)
    # fp32: see probe_fun_tools_fullturn. The masters are fp32 and this reads
    # behaviour, so a load-time downcast would sit between the two.
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.float32, **kw).cuda().eval()
    eos = [i for i in (tok.eos_token_id,
                       tok.convert_tokens_to_ids("<|end|>")) if i is not None]

    def gen(p, n=420):
        e = tok(p, return_tensors="pt", add_special_tokens=False,
                return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=n, do_sample=False,
                               num_beams=1, eos_token_id=eos,
                               pad_token_id=tok.pad_token_id or eos[0],
                               repetition_penalty=1.1)
        return tok.decode(o[0][e["input_ids"].shape[1]:],
                          skip_special_tokens=False)

    right_expr = right_final = 0
    for q, expected in QUESTIONS:
        msgs = [user_msg(q)]
        print("=" * 78)
        print(f"USER: {q}")
        print(f"      (forventet resultat: {expected})")
        print("-" * 78)

        for mode in ("free", "forced"):
            raw = gen(free_prompt(msgs) if mode == "free"
                      else forced_prompt(msgs))
            if mode == "free":
                m = CALL.search(raw)
                reasoning = (raw.split("<|tool_call|>")[0]
                             if m else raw).replace("<|end|>", "").strip()
                body = m.group(1) if m else ""
                print(f"[{mode:6}] reasoning ({len(reasoning.split())} words): "
                      f"{reasoning or '(none)'}")
            else:
                body = raw.split("<|/tool_call|>")[0]
                reasoning = ""
            call, result = run_call(body)
            if call is None:
                print(f"[{mode:6}] CALL: UNPARSEABLE -> {body.strip()[:90]}")
                continue
            print(f"[{mode:6}] CALL: {call}")
            if result is None:
                print(f"[{mode:6}] TOOL: (no such tool)")
                continue
            got = json.loads(result).get("result")
            ok = got == expected
            print(f"[{mode:6}] TOOL: {result}   "
                  f"{'✓ korrekt udtryk' if ok else f'✗ giver {got}, ikke {expected}'}")
            ans = gen(answer_prompt(msgs, reasoning, call, result), 200)
            ans = ans.replace("<|end|>", "").strip()
            print(f"[{mode:6}] ANSWER: {ans}")
            if mode == "free":
                right_expr += ok
                # The reply must carry the tool's OWN number, not the
                # question's. str() on the int is enough here because every
                # expected value is an integer the prose never states.
                right_final += ok and str(expected) in ans
            print("-" * 78)
        print()

    n = len(QUESTIONS)
    print(f"free-condition totals: expression right {right_expr}/{n}   "
          f"answer carries the result {right_final}/{n}")


if __name__ == "__main__":
    main()
