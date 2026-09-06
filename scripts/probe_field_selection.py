"""Which field does the model quote when several numbers compete?

The v39 probes converged on one residual failure. Given a payload with a single
number the model answers correctly; given `{"room": "Bjergtoppen", "capacity":
10, "floor": 3}` it reports "10 værelser" and "3 timer" and never names the
room. That is field selection under competition, not missing grounding, and
`tool_answer` sitting flat at ~72 while calling climbed 14 points is consistent
with it.

Each case below asks about ONE field of a multi-field payload. Scoring is
three-way, because "did it say the right number" and "did it say a wrong one"
are different failures:

  target-cited      the asked-for value appears
  distractor-cited  some OTHER payload number appears
  clean             target and no distractor

A model that recites the whole payload scores target-cited but not clean, and
that is the right verdict: quoting everything is not selecting.

Tools are English identifiers with Danish descriptions, matching the corpus --
Danish tool names are off-distribution and produced hallucinated names when
tried.
"""
import argparse
import json
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CALL_RE = re.compile(r"<\|tool_call\|>(.*?)(?:<\|/tool_call\|>|$)", re.S)
NUM = re.compile(r"\d+(?:[.,]\d+)?")

# (tool spec, user question, payload, target field, why this case exists)
CASES = [
    ({"name": "coffee_machine_status",
      "description": "Hent status for kaffemaskinen på en etage",
      "parameters": {"type": "object", "properties": {
          "floor": {"type": "integer", "description": "Etagen"}},
          "required": ["floor"]},
      "returns": {"type": "object", "properties": {
          "cups_left": {"description": "Antal kopper tilbage"},
          "days_since_service": {"description": "Dage siden sidste service"},
          "floor": {"description": "Etagen"}}}},
     "Hvor mange kopper kaffe er der tilbage på 4. etage?",
     {"cups_left": 8, "days_since_service": 12, "floor": 4},
     "cups_left", "3 integers, one asked for"),

    ({"name": "get_flight",
      "description": "Hent oplysninger om en flyafgang",
      "parameters": {"type": "object", "properties": {
          "flight_number": {"type": "string", "description": "Flynummeret"}},
          "required": ["flight_number"]},
      "returns": {"type": "object", "properties": {
          "price": {"description": "Prisen i kroner"},
          "duration_minutes": {"description": "Flyvetid i minutter"},
          "seats_left": {"description": "Ledige sæder"}}}},
     "Hvad koster afgang SK451?",
     {"price": 1200, "duration_minutes": 95, "seats_left": 3},
     "price", "price vs duration vs seats"),

    ({"name": "get_weather",
      "description": "Hent vejret for en by",
      "parameters": {"type": "object", "properties": {
          "city": {"type": "string", "description": "Byen"}},
          "required": ["city"]},
      "returns": {"type": "object", "properties": {
          "temperature": {"description": "Temperatur i grader"},
          "wind_speed": {"description": "Vindhastighed i m/s"},
          "humidity": {"description": "Luftfugtighed i procent"}}}},
     "Hvor varmt er der i Aarhus?",
     {"temperature": 7, "wind_speed": 12, "humidity": 81},
     "temperature", "all three plausibly 'weather numbers'"),

    ({"name": "find_hotel",
      "description": "Find et hotel på en destination",
      "parameters": {"type": "object", "properties": {
          "city": {"type": "string", "description": "Byen"}},
          "required": ["city"]},
      "returns": {"type": "object", "properties": {
          "name": {"description": "Hotellets navn"},
          "price_per_night": {"description": "Pris per nat i kroner"},
          "rooms_available": {"description": "Ledige værelser"},
          "rating": {"description": "Bedømmelse"}}}},
     "Hvad koster det per nat på et hotel i Odense?",
     {"name": "Hotel Fjorden", "price_per_night": 950,
      "rooms_available": 4, "rating": 4.2},
     "price_per_night", "the Bjergtoppen shape: name + 3 numbers"),

    ({"name": "track_package",
      "description": "Følg en pakke",
      "parameters": {"type": "object", "properties": {
          "tracking_id": {"type": "string", "description": "Sporingsnummeret"}},
          "required": ["tracking_id"]},
      "returns": {"type": "object", "properties": {
          "days_in_transit": {"description": "Dage undervejs"},
          "weight_kg": {"description": "Vægt i kilo"},
          "items": {"description": "Antal varer i pakken"}}}},
     "Hvor mange dage har min pakke DK9912 været undervejs?",
     {"days_in_transit": 3, "weight_kg": 2, "items": 5},
     "days_in_transit", "small integers, easily swapped"),

    ({"name": "get_recipe",
      "description": "Hent en opskrift",
      "parameters": {"type": "object", "properties": {
          "dish": {"type": "string", "description": "Retten"}},
          "required": ["dish"]},
      "returns": {"type": "object", "properties": {
          "calories": {"description": "Kalorier per portion"},
          "servings": {"description": "Antal portioner"},
          "minutes": {"description": "Tilberedningstid i minutter"}}}},
     "Hvor lang tid tager det at lave frikadeller?",
     {"calories": 450, "servings": 4, "minutes": 25},
     "minutes", "time asked, calories is the biggest number"),

    ({"name": "get_stock",
      "description": "Hent aktieoplysninger for et selskab",
      "parameters": {"type": "object", "properties": {
          "symbol": {"type": "string", "description": "Aktiesymbolet"}},
          "required": ["symbol"]},
      "returns": {"type": "object", "properties": {
          "price": {"description": "Aktiekursen"},
          "change_percent": {"description": "Ændring i procent"},
          "volume": {"description": "Handlet volumen"}}}},
     "Hvad er kursen på NOVO?",
     {"price": 305, "change_percent": 2.1, "volume": 1200000},
     "price", "volume is a huge distractor"),

    ({"name": "get_gym_status",
      "description": "Hent status for et fitnesscenter",
      "parameters": {"type": "object", "properties": {
          "gym": {"type": "string", "description": "Centeret"}},
          "required": ["gym"]},
      "returns": {"type": "object", "properties": {
          "members_present": {"description": "Medlemmer til stede"},
          "capacity": {"description": "Kapacitet"},
          "free_spots": {"description": "Ledige pladser"}}}},
     "Hvor mange ledige pladser er der i centret?",
     {"members_present": 120, "capacity": 200, "free_spots": 80},
     "free_spots", "free_spots = capacity - members, arithmetic temptation"),
]


def nums_in(text):
    out = set()
    for m in NUM.finditer(text):
        try:
            f = float(m.group().replace(",", "."))
            out.add(int(f) if f.is_integer() else f)
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--show", action="store_true", help="print every answer")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.float16).cuda().eval()
    eos = [i for i in (tok.eos_token_id,
                       tok.convert_tokens_to_ids("<|end|>")) if i is not None]

    def gen(p, n=200):
        e = tok(p, return_tensors="pt", add_special_tokens=False,
                return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=n, do_sample=False,
                               eos_token_id=eos,
                               pad_token_id=tok.pad_token_id or 0,
                               repetition_penalty=1.1)
        return tok.decode(o[0][e["input_ids"].shape[1]:],
                          skip_special_tokens=False).strip()

    n_call = n_target = n_distractor = n_clean = 0
    rows = []
    for spec, question, payload, field, note in CASES:
        cat = json.dumps([spec], ensure_ascii=False)
        p = f"<|user|>Værktøjer:\n{cat}\n\n{question}<|end|><|assistant|>"
        out = gen(p)
        m = CALL_RE.search(out)
        if not m:
            rows.append((spec["name"], field, "NO CALL", out[:120], note))
            continue
        try:
            call = json.loads(m.group(1).strip())
        except Exception:
            rows.append((spec["name"], field, "BAD JSON", m.group(1)[:120], note))
            continue
        if call.get("name") != spec["name"]:
            rows.append((spec["name"], field, "WRONG TOOL",
                         str(call.get("name")), note))
            continue
        n_call += 1
        res = json.dumps(payload, ensure_ascii=False)
        p2 = (p + out.split("<|/tool_call|>")[0] + "<|/tool_call|><|end|>"
              + f"<|tool_result|>{res}<|/tool_result|><|assistant|>")
        ans = gen(p2, 160)
        said = nums_in(ans)
        target = payload[field]
        tgt = (int(target) if float(target).is_integer() else target) \
            if isinstance(target, (int, float)) else target
        hit = tgt in said if isinstance(tgt, (int, float)) \
            else str(tgt).lower() in ans.lower()
        distract = {v for k, v in payload.items()
                    if k != field and isinstance(v, (int, float))}
        distract = {int(v) if float(v).is_integer() else v for v in distract}
        bad = bool(said & distract)
        n_target += hit
        n_distractor += bad
        n_clean += hit and not bad
        verdict = ("clean" if hit and not bad else
                   "target+distractor" if hit else
                   "distractor only" if bad else "neither")
        rows.append((spec["name"], field, verdict,
                     ans.replace("<|end|>", "").strip()[:150], note))

    n = len(CASES)
    print(f"\n{'tool':<24}{'asked':<20}{'verdict':<20}answer")
    print("-" * 110)
    for name, field, verdict, ans, _ in rows:
        print(f"{name:<24}{field:<20}{verdict:<20}{ans}")
    print("-" * 110)
    print(f"cases                  : {n}")
    print(f"right tool called      : {n_call}/{n}")
    print(f"target value cited     : {n_target}/{n}")
    print(f"a distractor cited     : {n_distractor}/{n}")
    print(f"CLEAN (target, no dist): {n_clean}/{n}  ({100*n_clean/n:.0f}%)")


if __name__ == "__main__":
    main()
