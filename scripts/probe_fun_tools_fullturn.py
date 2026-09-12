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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sft_packed import format_conversation  # noqa: E402

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


# A CHAIN. `bike_service_status` needs an id the user never says, so the only
# route is the lookup first. It refuses IN-BAND when handed something that is
# not an id, because a skipped chain must be visible: the failure this probe
# is looking for is a confident answer built from the owner's name.
BIKES = {"mette nielsen": "BIKE-4471", "jonas berg": "BIKE-1180"}
SERVICE = {
    "BIKE-4471": {"bike_id": "BIKE-4471", "last_service_date": "2026-03-02",
                  "brake_wear_pct": 41, "chain_wear_pct": 17},
    "BIKE-1180": {"bike_id": "BIKE-1180", "last_service_date": "2025-11-19",
                  "brake_wear_pct": 68, "chain_wear_pct": 55},
}


def find_bike_by_owner(owner_name):
    bid = BIKES.get(str(owner_name).strip().lower())
    if not bid:
        return {"error": f"ingen cykel registreret på {owner_name}"}
    return {"bike_id": bid, "owner_name": owner_name, "frame_size_cm": 54}


def bike_service_status(bike_id):
    s = SERVICE.get(str(bike_id).strip())
    if not s:
        return {"error": f"ukendt bike_id {bike_id}; slå cyklen op "
                         f"på ejerens navn først"}
    return dict(s)


IMPL = {"roll_dice": roll_dice,
        "find_bike_by_owner": find_bike_by_owner,
        "bike_service_status": bike_service_status,
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
    {"name": "find_bike_by_owner",
     "description": "Slå en cykel op ud fra ejerens navn",
     "parameters": {"type": "object", "properties": {
         "owner_name": {"type": "string", "description": "Ejerens fulde navn"}},
         "required": ["owner_name"]},
     "returns": {"type": "object", "properties": {
         "bike_id": {"description": "Cyklens unikke id"},
         "frame_size_cm": {"description": "Stelstørrelse i cm"}}}},
    {"name": "bike_service_status",
     "description": "Hent servicestatus for en cykel ud fra dens id",
     "parameters": {"type": "object", "properties": {
         "bike_id": {"type": "string", "description": "Cyklens unikke id"}},
         "required": ["bike_id"]},
     "returns": {"type": "object", "properties": {
         "last_service_date": {"description": "Dato for sidste service"},
         "brake_wear_pct": {"description": "Bremseslid i procent"},
         "chain_wear_pct": {"description": "Kædeslid i procent"}}}},
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
    # CHAINED: the id is not in the question, so this needs two calls.
    "Hvornår blev Mette Nielsens cykel sidst serviceret?",
]


# --- prompt construction ------------------------------------------------
# Built by the TRAINER'S OWN renderer, never by hand. The first version of
# this probe hand-wrote `<|user|>{q}<|end|><|assistant|>`, which differs from
# training in two ways: format_conversation joins turns with a space, and it
# emits <|end|> only after MODEL turns -- so an <|end|> after the user turn
# occurs nowhere in the corpus. The downstream eval renders through
# format_conversation and appends " <|assistant|>"; anything else measures
# the probe rather than the model.
SENTINEL = "\x00"
# Enough for lookup -> consumer -> answer, with one spare.
MAX_STEPS = 4


def user_msg(q):
    cat = json.dumps(CATALOG, ensure_ascii=False)
    return {"role": "user", "content": f"Værktøjer:\n{cat}\n\n{q}"}


def free_prompt(msgs):
    """`<|assistant|>` is supplied by every inference path, so the probe
    supplies it too -- identical to `f"{q} {ASST}"` in the eval."""
    return format_conversation(msgs) + " <|assistant|>"


def forced_prompt(msgs):
    """Prefill straight into the call. Rendered rather than concatenated: an
    empty assistant turn before a call yields TWO spaces
    (`<|assistant|>  <|tool_call|>`), one from the empty content and one from
    the join, which is what the corpus contains and is not guessable."""
    return format_conversation(
        msgs + [{"role": "assistant", "content": ""},
                {"role": "tool_call", "content": SENTINEL}]).split(SENTINEL)[0]


def answer_prompt(msgs, reasoning, call, result):
    return free_prompt(msgs + [
        {"role": "assistant", "content": reasoning},
        {"role": "tool_call", "content": call},
        {"role": "tool_result", "content": result}])


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data2/ckpts/v38_33993"
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
    # fp32, not fp16. The checkpoint's master weights are fp32 and the probe is
    # reading behaviour, not measuring throughput: downcasting at load puts a
    # rounding step between the trained weights and what is probed, so a
    # surprising output cannot be attributed to the model rather than to the
    # cast. A 400M model is 1.6GB in fp32 and fits anywhere this runs.
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
                               pad_token_id=tok.pad_token_id or 0,
                               repetition_penalty=1.1)
        return tok.decode(o[0][e["input_ids"].shape[1]:],
                          skip_special_tokens=False).strip()

    for q in QUESTIONS:
        msgs = [user_msg(q)]
        print("=" * 78)
        print(f"USER: {q}")
        for label, prompt, prefixed in (
                ("free  ", free_prompt(msgs), False),
                ("forced", forced_prompt(msgs), True)):
            print("-" * 78)
            # MULTI-STEP. This used to call once and answer, which cannot
            # complete a chain: `bike_service_status` needs an id that only
            # the lookup produces, so a single-call loop would show the first
            # call and stop, and the interesting turn -- what the model does
            # with a handle it just received -- never happened.
            hist, answered = list(msgs), False
            for step in range(1, MAX_STEPS + 1):
                out = gen(prompt)
                text = ("<|tool_call|>" + out) if (prefixed and step == 1) else out
                if not prefixed or step > 1:
                    think = text.split("<|tool_call|>")[0].strip()
                    if step == 1:
                        print(f"[{label}] reasoning ({len(think.split())} words): "
                              f"{think if think else '(none)'}")
                m = CALL.search(text)
                if not m:
                    print(f"[{label}] ANSWER: {text.strip()[:300]}")
                    answered = True
                    break
                try:
                    call, _ = json.JSONDecoder().raw_decode(m.group(1).strip())
                except Exception:
                    print(f"[{label}] UNPARSEABLE -> {m.group(1)[:200]}")
                    break
                print(f"[{label}] CALL {step}: {json.dumps(call, ensure_ascii=False)}")
                name = call.get("name") or ""
                fn = IMPL.get(name)
                if not fn:
                    # Case-insensitive retry: the model emits `Beregn_hundeår`
                    # for `beregn_hundeår`. Resolving it anyway lets the rest
                    # of the turn run, so a casing slip does not hide the
                    # answer step.
                    fn = next((f for k, f in IMPL.items()
                               if k.lower() == name.lower()), None)
                    if fn:
                        print(f"[{label}]   (casing slip: '{name}' -> resolved)")
                if not fn:
                    print(f"[{label}]   !! tool '{name}' does not exist")
                    break
                try:
                    result = fn(**(call.get("arguments") or {}))
                except Exception as ex:
                    print(f"[{label}]   !! bad args: {ex}")
                    break
                res = json.dumps(result, ensure_ascii=False)
                print(f"[{label}] TOOL {step}: {res}")
                reasoning = text.split("<|tool_call|>")[0].strip()
                hist = hist + [{"role": "assistant", "content": reasoning},
                               {"role": "tool_call",
                                "content": json.dumps(call, ensure_ascii=False)},
                               {"role": "tool_result", "content": res}]
                prompt = free_prompt(hist)
            if not answered:
                print(f"[{label}]   (no answer within {MAX_STEPS} steps)")
        print()


if __name__ == "__main__":
    main()
