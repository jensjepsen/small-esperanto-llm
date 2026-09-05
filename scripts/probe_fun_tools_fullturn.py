"""Full-turn probe on four invented tools, free vs forced reasoning.

The tools are not in the training corpus and never were, so nothing here is
memorised: the model must read the catalogue, build a call, receive a real
result and answer from it.

Two conditions per question:

  free   -- generate from <|assistant|>; the model reasons, then calls
  forced -- prefill <|assistant|> <|tool_call|>; straight to JSON

This does NOT answer whether training without reasoning would work -- only a
retrain does. It shows what the reasoning is doing at inference on unseen
tools, and what the answer turn looks like once a real result comes back.
"""
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CALL = re.compile(r"<\|tool_call\|>(.*?)(?:<\|/tool_call\|>|$)", re.S)


# English identifiers, Danish descriptions -- the shape the corpus actually
# trains. The first cut of this probe used Danish tool and parameter names
# (`beregn_hundeår`, `menneskeår`) and the model answered with a tool name
# that did not exist; nothing in training looks like that, so the failure said
# more about the probe than the model.
def roll_dice(count, sides):
    rolls = [((i * 7 + 3) % sides) + 1 for i in range(count)]
    return {"rolls": rolls, "total": sum(rolls)}


def coffee_machine_status(floor):
    return {"floor": floor, "cups_left": 12 - floor, "working": True,
            "last_service": "2026-08-14"}


def dog_years(human_years):
    return {"human_years": human_years, "dog_years": human_years * 7}


def find_meeting_room(people, time):
    big = people > 6
    return {"room": "Bjergtoppen" if big else "Fjorden",
            "capacity": 10 if big else 6, "time": time, "floor": 3}


IMPL = {"roll_dice": roll_dice,
        "coffee_machine_status": coffee_machine_status,
        "dog_years": dog_years,
        "find_meeting_room": find_meeting_room}

CATALOG = [
    {"name": "roll_dice",
     "description": "Kast et antal terninger og få resultatet",
     "parameters": {"type": "object", "properties": {
         "count": {"type": "integer", "description": "Antal terninger"},
         "sides": {"type": "integer",
                   "description": "Antal sider på hver terning"}},
         "required": ["count", "sides"]},
     "returns": {"type": "object", "properties": {
         "rolls": {"type": "array",
                   "items": {"description": "Et enkelt kast"}},
         "total": {"description": "Summen af alle kast"}}}},
    {"name": "coffee_machine_status",
     "description": "Hent status for kaffemaskinen på en etage",
     "parameters": {"type": "object", "properties": {
         "floor": {"type": "integer",
                   "description": "Etagen kaffemaskinen står på"}},
         "required": ["floor"]},
     "returns": {"type": "object", "properties": {
         "cups_left": {"description": "Antal kopper kaffe tilbage"},
         "working": {"description": "Om maskinen virker"},
         "last_service": {"description": "Dato for sidste service"}}}},
    {"name": "dog_years",
     "description": "Omregn menneskeår til hundeår",
     "parameters": {"type": "object", "properties": {
         "human_years": {"type": "integer",
                         "description": "Alder i menneskeår"}},
         "required": ["human_years"]},
     "returns": {"type": "object", "properties": {
         "dog_years": {"description": "Alderen omregnet til hundeår"}}}},
    {"name": "find_meeting_room",
     "description": "Find et ledigt mødelokale",
     "parameters": {"type": "object", "properties": {
         "people": {"type": "integer", "description": "Antal personer"},
         "time": {"type": "string", "description": "Ønsket tidspunkt"}},
         "required": ["people", "time"]},
     "returns": {"type": "object", "properties": {
         "room": {"description": "Navnet på lokalet"},
         "capacity": {"description": "Hvor mange lokalet kan rumme"},
         "floor": {"description": "Etagen lokalet ligger på"}}}},
]

QUESTIONS = [
    "Hej! Kan du kaste 3 terninger med 6 sider for mig?",
    "Er der kaffe tilbage på 4. etage?",
    "Min hund er 4 menneskeår gammel. Hvor gammel er den i hundeår?",
    "Vi er 8 personer og skal mødes kl. 14. Kan du finde et lokale?",
]


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data2/ckpts/v38_33993"
    print(f"ckpt: {ckpt}\n", flush=True)
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.float16).cuda().eval()
    eos = [i for i in (tok.eos_token_id,
                       tok.convert_tokens_to_ids("<|end|>")) if i is not None]

    def gen(p, n=420):
        e = tok(p, return_tensors="pt", add_special_tokens=False,
                return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=n, do_sample=False,
                               num_beams=1, eos_token_id=eos,
                               pad_token_id=tok.pad_token_id or 0,
                               repetition_penalty=1.1)
        return tok.decode(o[0][e["input_ids"].shape[1]:],
                          skip_special_tokens=False).strip()

    cat = json.dumps(CATALOG, ensure_ascii=False)
    for q in QUESTIONS:
        base = f"<|user|>Værktøjer:\n{cat}\n\n{q}<|end|><|assistant|>"
        print("=" * 78)
        print(f"USER: {q}")
        for label, prompt, prefixed in (
                ("free  ", base, False),
                ("forced", base + "<|tool_call|>", True)):
            print("-" * 78)
            out = gen(prompt)
            text = ("<|tool_call|>" + out) if prefixed else out
            if not prefixed:
                think = text.split("<|tool_call|>")[0].strip()
                print(f"[{label}] reasoning ({len(think.split())} words): "
                      f"{think if think else '(none)'}")
            m = CALL.search(text)
            if not m:
                print(f"[{label}] NO CALL -> {out[:300]}")
                continue
            try:
                call, _ = json.JSONDecoder().raw_decode(m.group(1).strip())
            except Exception:
                print(f"[{label}] UNPARSEABLE -> {m.group(1)[:200]}")
                continue
            print(f"[{label}] CALL: {json.dumps(call, ensure_ascii=False)}")
            name = call.get("name") or ""
            fn = IMPL.get(name)
            if not fn:
                # Case-insensitive retry: the model emits `Beregn_hundeår` for
                # `beregn_hundeår`. Resolving it anyway lets the rest of the
                # turn run, so a casing slip does not hide the answer step.
                fn = next((f for k, f in IMPL.items()
                           if k.lower() == name.lower()), None)
                if fn:
                    print(f"[{label}]   (casing slip: '{name}' -> resolved)")
            if not fn:
                print(f"[{label}]   !! tool '{name}' does not exist")
                continue
            try:
                result = fn(**(call.get("arguments") or {}))
            except Exception as ex:
                print(f"[{label}]   !! bad args: {ex}")
                continue
            res = json.dumps(result, ensure_ascii=False)
            print(f"[{label}] TOOL: {res}")
            # feed the real result back; the model must answer from it
            p2 = (prompt + text.split("<|/tool_call|>")[0]
                  + "<|/tool_call|><|end|><|tool_result|>" + res
                  + "<|/tool_result|><|assistant|>")
            print(f"[{label}] ANSWER: {gen(p2, 200)}")
        print()


if __name__ == "__main__":
    main()
