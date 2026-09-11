"""A tau-bench-shaped retail task, in Danish, run as a real agent loop.

NOT a verbatim tau-bench item. The tool set and task shape follow tau-bench's
retail domain (Yao et al., Sierra) -- user lookup, order lookup, then an action
gated on what the lookup returned -- and the prose is written in Danish here.
Calling it a translation would overclaim; it is the same SHAPE.

WHY THIS AND NOT ANOTHER FUN PROBE. probe_fun_tools_fullturn asks one question,
gets one call, feeds one result, reads one answer. Every argument it needs is
stated in the user's turn. tau-bench's difficulty is elsewhere:

    the order id is NOT in the conversation -- it comes back from a call
    the cancel CANNOT succeed until the lookup has happened
    the tool refuses, in-band, if the chain is skipped

So this measures output-dependent chaining, which the corpus never teaches:
`sample_args` fills every argument from the user turn or from schema examples,
and `gate_call` REJECTS a row whose call carries a value nobody said. A corpus
built on that rule cannot contain a call whose argument came from a previous
tool result.

WHAT IT FOUND, on both arms of the toolmix/symonly A/B:

  toolmix step-1816   get_order_details {order_id: "kaffemaskine",
                                         request_id: "bestillingsnummer"}
  symonly step-1561   cancel_pending_order {order_id: "kaffemaskine",
                                            reason: "no longer needed"}

Both put the Danish word for "coffee machine" in `order_id`, and neither used
the email -- the one key the user actually supplied and the one the correct
first call needs. The argument has to come from a lookup, nothing in the
conversation supplies it, so both reach for the most order-shaped token in the
prose. Independent of training data: the two corpora differ in everything
except this.

TWO SEPARABLE GAPS, neither addressed by any current corpus:

1. DERIVED ARGUMENTS. A call whose input came from a previous tool result.
   `gate_call` actively forbids that shape, so it cannot be learned from data
   built by this generator.

2. ERROR RESULTS. Every tool_result in both corpora is a successful payload,
   so an `error` key is just another field to read values out of. Given
   {"error": "ukendt order_id kaffemaskine; slå ordren op først"} the model
   replied "Din ordre er nu annulleret med ID ukendt og ordrenummer slået op
   først" -- it parsed the error STRING into fields and reported success.
   tau-bench's recovery loop is unreachable not because chaining is hard but
   because nothing signals that a call failed.

The loop runs up to --max-steps assistant turns, feeding each tool result back
through format_conversation, and stops when the model answers without calling.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sft_packed import format_conversation  # noqa: E402

CALL = re.compile(r"<\|tool_call\|>(.*?)(?:<\|/tool_call\|>|$)", re.S)

# ── the world ────────────────────────────────────────────────────────────
# One user, two orders. The pending one is cancellable; the delivered one is
# not, and saying so is the correct refusal if the model reaches for it.
USERS = {"mette.hansen@email.dk": "usr_8842"}
USER_DETAILS = {"usr_8842": {"user_id": "usr_8842", "name": "Mette Hansen",
                             "zip": "8000", "orders": ["ord_5517", "ord_4102"]}}
ORDERS = {
    "ord_5517": {"order_id": "ord_5517", "status": "pending",
                 "items": ["Kaffemaskine KM-200"], "total_dkk": 1299},
    "ord_4102": {"order_id": "ord_4102", "status": "delivered",
                 "items": ["Termokande 1L"], "total_dkk": 249},
}
CANCELLED: set[str] = set()


def find_user_id_by_email(email):
    uid = USERS.get(str(email).strip().lower())
    return {"user_id": uid} if uid else {"error": f"ingen bruger med e-mail {email}"}


def get_user_details(user_id):
    d = USER_DETAILS.get(str(user_id))
    return d or {"error": f"ukendt user_id {user_id}"}


def get_order_details(order_id):
    o = ORDERS.get(str(order_id))
    return dict(o) if o else {"error": f"ukendt order_id {order_id}"}


def cancel_pending_order(order_id, reason):
    """Refuses IN-BAND when the chain was skipped or the order is not pending.

    An invented order id is the failure this probe most expects, so the tool
    has to answer it rather than raise -- the model should read the refusal
    and recover, which is itself part of what tau-bench measures.
    """
    o = ORDERS.get(str(order_id))
    if not o:
        return {"error": f"ukendt order_id {order_id}; slå ordren op først"}
    if o["status"] != "pending":
        return {"error": f"ordre {order_id} har status {o['status']} "
                         f"og kan ikke annulleres"}
    if str(reason) not in ("no longer needed", "ordered by mistake"):
        return {"error": "reason skal være 'no longer needed' eller "
                         "'ordered by mistake'"}
    CANCELLED.add(str(order_id))
    return {"order_id": order_id, "status": "cancelled", "refund_dkk": o["total_dkk"]}


IMPL = {"find_user_id_by_email": find_user_id_by_email,
        "get_user_details": get_user_details,
        "get_order_details": get_order_details,
        "cancel_pending_order": cancel_pending_order}

CATALOG = [
    {"name": "find_user_id_by_email",
     "description": "Find en brugers id ud fra e-mailadresse",
     "parameters": {"type": "object", "properties": {
         "email": {"type": "string", "description": "Brugerens e-mailadresse"}},
         "required": ["email"]},
     "returns": {"type": "object", "properties": {
         "user_id": {"description": "Brugerens id"}}}},
    {"name": "get_user_details",
     "description": "Hent oplysninger om en bruger, herunder ordrenumre",
     "parameters": {"type": "object", "properties": {
         "user_id": {"type": "string", "description": "Brugerens id"}},
         "required": ["user_id"]},
     "returns": {"type": "object", "properties": {
         "name": {"description": "Brugerens navn"},
         "zip": {"description": "Postnummer"},
         "orders": {"description": "Liste af brugerens ordrenumre"}}}},
    {"name": "get_order_details",
     "description": "Hent oplysninger om en ordre",
     "parameters": {"type": "object", "properties": {
         "order_id": {"type": "string", "description": "Ordrenummeret"}},
         "required": ["order_id"]},
     "returns": {"type": "object", "properties": {
         "status": {"description": "Ordrens status, fx pending eller delivered"},
         "items": {"description": "Varer i ordren"},
         "total_dkk": {"description": "Ordrens samlede beløb i kroner"}}}},
    {"name": "cancel_pending_order",
     "description": "Annullér en ordre der endnu ikke er afsendt",
     "parameters": {"type": "object", "properties": {
         "order_id": {"type": "string", "description": "Ordrenummeret"},
         "reason": {"type": "string",
                    "description": "Årsag: 'no longer needed' eller "
                                   "'ordered by mistake'"}},
         "required": ["order_id", "reason"]},
     "returns": {"type": "object", "properties": {
         "status": {"description": "Ordrens nye status"},
         "refund_dkk": {"description": "Beløb der refunderes i kroner"}}}},
]

TASK = ("Hej. Jeg vil gerne annullere min ordre på kaffemaskinen - jeg kom til "
        "at bestille den ved en fejl. Min e-mail er mette.hansen@email.dk.")


def user_msg(q):
    return {"role": "user",
            "content": f"Værktøjer:\n{json.dumps(CATALOG, ensure_ascii=False)}\n\n{q}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--max-steps", type=int, default=6)
    args = ap.parse_args()

    kw = {"subfolder": args.subfolder} if args.subfolder else {}
    print(f"ckpt: {args.ckpt}" + (f" [{args.subfolder}]" if args.subfolder else ""))
    tok = AutoTokenizer.from_pretrained(args.ckpt, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.float32, **kw).cuda().eval()
    eos = [i for i in (tok.eos_token_id,
                       tok.convert_tokens_to_ids("<|end|>")) if i is not None]

    def gen(msgs, n=320):
        p = format_conversation(msgs) + " <|assistant|>"
        e = tok(p, return_tensors="pt", add_special_tokens=False,
                return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=n, do_sample=False,
                               num_beams=1, eos_token_id=eos,
                               pad_token_id=tok.pad_token_id or eos[0],
                               repetition_penalty=1.1)
        return tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=False)

    msgs = [user_msg(TASK)]
    print("=" * 86)
    print(f"USER: {TASK}\n" + "-" * 86)
    called = []
    for step in range(1, args.max_steps + 1):
        raw = gen(msgs)
        m = CALL.search(raw)
        prose = (raw.split("<|tool_call|>")[0] if m else raw)
        prose = prose.replace("<|end|>", "").strip()
        if not m:
            print(f"[{step}] ANSWER (no call): {prose[:300]}")
            break
        if prose:
            print(f"[{step}] prose: {prose[:160]}")
        body = m.group(1).strip()
        try:
            call, _ = json.JSONDecoder().raw_decode(body)
        except Exception:
            print(f"[{step}] CALL: UNPARSEABLE -> {body[:160]}")
            break
        name, a = call.get("name"), call.get("arguments") or {}
        print(f"[{step}] CALL: {json.dumps(call, ensure_ascii=False)[:200]}")
        called.append(name)
        if name not in IMPL:
            res = {"error": f"intet værktøj ved navn {name}"}
        else:
            try:
                res = IMPL[name](**a)
            except TypeError as e:
                res = {"error": f"forkerte parametre: {e}"}
        rj = json.dumps(res, ensure_ascii=False)
        print(f"[{step}] RESULT: {rj[:220]}")
        msgs += [{"role": "assistant", "content": prose},
                 {"role": "tool_call", "content": json.dumps(call, ensure_ascii=False)},
                 {"role": "tool_result", "content": rj}]
    else:
        print(f"(stopped at --max-steps {args.max_steps} without a final answer)")

    print("-" * 86)
    print(f"tools called in order : {called}")
    print(f"ord_5517 cancelled    : {'YES' if 'ord_5517' in CANCELLED else 'NO'}")
    print("expected chain        : find_user_id_by_email -> get_user_details "
          "-> get_order_details -> cancel_pending_order")


if __name__ == "__main__":
    main()
