"""Abstention probe on the four invented tools.

v40 is the first run with abstention data in the mix (694 rows,
`danish-tool-dialogues-v6:abstention`), and `tool_refusal` reads it as a single
balanced-accuracy number. This shows the actual replies, on tools that are not
in the corpus, in the three situations the corpus distinguishes:

  B  no capable tool   nothing in the catalogue can serve the request. The
                       trained behaviour is to decline WITHOUT calling.

  A  absent field      a tool fits and should be called, but the result does
                       not carry the field the question asked for. The trained
                       behaviour is call -> read -> decline on that field.
                       Two-step, and the harder of the two.

  -  control           an ordinary answerable question. Must NOT be refused.
                       Without this half the probe is passed by a model that
                       refuses everything, which is worse than one that never
                       refuses.

The control questions are the four from probe_fun_tools_fullturn, imported
rather than copied so the two probes cannot drift apart.

Prompts are rendered by the trainer's own format_conversation -- see the note
in that module. Note that the tool_refusal EVAL does not do this for its
positive half: `_load_tool_refusal` builds `f"{USER}{content}{END}{ASST}"` by
hand while its negative half goes through format_conversation, so the two sides
of that metric are prompted in two different formats.
"""
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_fun_tools_fullturn import (  # noqa: E402
    CATALOG, IMPL, QUESTIONS, CALL, answer_prompt, free_prompt, user_msg)

# The same regex the eval scores with, so a disagreement between this probe
# and the metric is a real disagreement and not two different definitions.
DECLINE = re.compile(
    r"\b(kan (jeg )?ikke|har ikke|ingen af|ikke i stand|ikke mulighed|"
    r"desværre|beklager|fremgår ikke|findes ikke|mangler)\b", re.I)

# B: no tool in the catalogue can serve these. Deliberately adjacent to what
# the catalogue DOES cover -- a weather question next to a coffee-machine tool
# is a harder negative than one about astrophysics, because the catalogue is
# full of plausible-looking office utilities.
NO_TOOL = [
    "Hvad bliver vejret i København i morgen?",
    "Kan du bestille en taxa til klokken 18?",
    "Hvor mange kalorier er der i en banan?",
    "Kan du sende en mail til Mette om mødet?",
]

# A: the tool fits, the call is right, the field is not in the result.
#    (question, tool that should be called, field that is missing)
ABSENT_FIELD = [
    ("Hvilket mærke er kaffemaskinen på 4. etage?",
     "coffee_machine_status", "mærke/brand"),
    ("Hvad koster det at leje mødelokalet til 8 personer kl. 14?",
     "find_meeting_room", "pris"),
    ("Er terningerne røde eller hvide, hvis jeg kaster 3 med 6 sider?",
     "roll_dice", "farve"),
]


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data2/ckpts/v38_33993"
    print(f"ckpt: {ckpt}\n", flush=True)
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.float16).cuda().eval()
    eos = [i for i in (tok.eos_token_id,
                       tok.convert_tokens_to_ids("<|end|>")) if i is not None]

    def gen(p, n=220):
        e = tok(p, return_tensors="pt", add_special_tokens=False,
                return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=n, do_sample=False,
                               num_beams=1, eos_token_id=eos,
                               pad_token_id=tok.pad_token_id or 0,
                               repetition_penalty=1.1)
        return tok.decode(o[0][e["input_ids"].shape[1]:],
                          skip_special_tokens=False).strip()

    def call_of(text):
        m = CALL.search(text)
        if not m:
            return None
        try:
            return json.JSONDecoder().raw_decode(m.group(1).strip())[0]
        except Exception:
            return None

    tally = []

    # ---- B: no capable tool. Correct = decline, and do NOT call. ----------
    print("=" * 78)
    print("B — NO CAPABLE TOOL   (correct: decline, no call)")
    for q in NO_TOOL:
        out = gen(free_prompt([user_msg(q)]))
        call = call_of(out)
        said = out.split("<|tool_call|>")[0].strip() or out.strip()
        ok = call is None and bool(DECLINE.search(said))
        tally.append(("B", ok))
        print("-" * 78)
        print(f"  Q: {q}")
        if call:
            print(f"  CALLED: {json.dumps(call, ensure_ascii=False)}  <-- "
                  f"should not have called")
        print(f"  {'OK  ' if ok else 'FAIL'}: {said[:400]}")

    # ---- A: field absent from the result. Correct = call, then decline. ---
    print()
    print("=" * 78)
    print("A — FIELD ABSENT FROM RESULT   (correct: call, then decline)")
    for q, want_tool, missing in ABSENT_FIELD:
        msgs = [user_msg(q)]
        out = gen(free_prompt(msgs))
        call = call_of(out)
        print("-" * 78)
        print(f"  Q: {q}   [missing field: {missing}]")
        if not call:
            tally.append(("A", False))
            print(f"  FAIL: no call -> {out.strip()[:300]}")
            continue
        name = call.get("name") or ""
        print(f"  CALL: {json.dumps(call, ensure_ascii=False)}"
              f"{'' if name == want_tool else f'  <-- expected {want_tool}'}")
        fn = IMPL.get(name)
        if not fn:
            tally.append(("A", False))
            print(f"  FAIL: tool '{name}' does not exist")
            continue
        try:
            result = fn(**(call.get("arguments") or {}))
        except Exception as ex:
            tally.append(("A", False))
            print(f"  FAIL: bad args: {ex}")
            continue
        res = json.dumps(result, ensure_ascii=False)
        print(f"  TOOL: {res}")
        reasoning = out.split("<|tool_call|>")[0].strip()
        ans = gen(answer_prompt(msgs, reasoning,
                                json.dumps(call, ensure_ascii=False), res))
        ok = bool(DECLINE.search(ans))
        tally.append(("A", ok))
        print(f"  {'OK  ' if ok else 'FAIL'}: {ans[:400]}")

    # ---- control: answerable. Correct = do NOT decline. -------------------
    print()
    print("=" * 78)
    print("CONTROL — ANSWERABLE   (correct: do NOT refuse)")
    for q in QUESTIONS:
        out = gen(free_prompt([user_msg(q)]))
        call = call_of(out)
        said = out.split("<|tool_call|>")[0].strip()
        refused = call is None and bool(DECLINE.search(out))
        tally.append(("ctl", not refused))
        print("-" * 78)
        print(f"  Q: {q}")
        print(f"  {'FAIL (over-refusal)' if refused else 'OK  '}: "
              f"{(said or json.dumps(call, ensure_ascii=False))[:300]}")

    print()
    print("=" * 78)
    for kind, label in (("B", "no-capable-tool  declined"),
                        ("A", "absent-field     declined"),
                        ("ctl", "answerable       not refused")):
        got = [ok for k, ok in tally if k == kind]
        print(f"  {label:<32} {sum(got)}/{len(got)}")


if __name__ == "__main__":
    main()
