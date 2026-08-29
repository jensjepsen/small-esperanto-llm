"""Test whether v6's procedural ratio capability generalizes along several
axes. All variants have the SAME underlying math (X+Y divide 30 in 2:3,
ask X's share = 12), only the surface form changes.

If the model has learned the SOLVER, all should pass.
If it's pattern-matching, only the surface forms seen in training pass.
"""
import re
import sys
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from esperanto_lm.data import _morpheme_preprocess
from probe_algebra import has_answer

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/sft/sft_v6/checkpoint-24000"
USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
SPECIAL = (USER, ASST, END)
SKIP = {"<s>", "</s>", "<pad>", "<unk>", USER, ASST, END}

def pp(s):
    pat = "(" + "|".join(re.escape(t) for t in SPECIAL) + ")"
    return " ".join(p if p in SPECIAL else _morpheme_preprocess(p)
                    for p in re.split(pat, s))

def decode(tok, ids):
    toks = tok.convert_ids_to_tokens(ids)
    toks = [t for t in toks if t not in SKIP]
    return "".join(t if t != "<w>" else " " for t in toks).strip()


# All variants: 2 people split N items in ratio a:b, ask for one share. SAME math.
# Axes: surface form / names / items / numbers / ratios / quantifier
VARIANTS = [
    # CONTROL — exact training form
    ("control: trained form",
     "Anna kaj Bert dividas 30 bombonojn en proporcio 2:3. Kiom da bombonoj ricevas Bert?", "18"),

    # Axis 1: NAMES not in training
    ("name swap: foreign names",
     "Aliyah kaj Tariq dividas 30 bombonojn en proporcio 2:3. Kiom da bombonoj ricevas Tariq?", "18"),
    ("name swap: very short",
     "A kaj B dividas 30 bombonojn en proporcio 2:3. Kiom da bombonoj ricevas B?", "18"),
    ("name swap: 3-person with 2-share ask",
     "Anna, Bert kaj Klara dividas 30 bombonojn en proporcio 2:2:2 (egale). Kiom ricevas Bert?", "10"),

    # Axis 2: ITEMS not in training ontology
    ("item: unfamiliar",
     "Anna kaj Bert dividas 30 zegrojn en proporcio 2:3. Kiom da zegroj ricevas Bert?", "18"),
    ("item: abstract concept",
     "Anna kaj Bert dividas 30 punktojn en proporcio 2:3. Kiom da punktoj ricevas Bert?", "18"),

    # Axis 3: NUMBERS outside training range
    ("number: very large",
     "Anna kaj Bert dividas 30000 bombonojn en proporcio 2:3. Kiom da bombonoj ricevas Bert?", "18000"),
    ("number: very small",
     "Anna kaj Bert dividas 5 bombonojn en proporcio 2:3.", "3"),  # may give non-integer; test it

    # Axis 4: UNUSUAL RATIOS
    ("ratio: prime denominator",
     "Anna kaj Bert dividas 33 bombonojn en proporcio 4:7. Kiom da bombonoj ricevas Bert?", "21"),
    ("ratio: 1:9",
     "Anna kaj Bert dividas 30 bombonojn en proporcio 1:9. Kiom da bombonoj ricevas Bert?", "27"),

    # Axis 5: QUESTION REFORMULATION
    ("question: nominative ask",
     "Anna kaj Bert dividas 30 bombonojn en proporcio 2:3. Kio estas la parto de Bert?", "18"),
    ("question: imperative",
     "Anna kaj Bert dividas 30 bombonojn en proporcio 2:3. Kalkulu la kvanton de Bert.", "18"),
    ("question: passive",
     "30 bombonoj estas dividitaj inter Anna kaj Bert en proporcio 2:3. Kiom havos Bert?", "18"),

    # Axis 6: REVERSED ORDER (B kaj A instead of A kaj B)
    ("order: reverse name/ratio",
     "Bert kaj Anna dividas 30 bombonojn en proporcio 3:2. Kiom da bombonoj ricevas Bert?", "18"),

    # Axis 7: PHRASING SHIFT (no "en proporcio" cue)
    ("phrasing: no 'proporcio' cue",
     "Bert ricevas tri partojn por ĉiu du partoj de Anna. Ili dividas 30 bombonojn entute. Kiom ricevas Bert?", "18"),
    ("phrasing: ratio in words",
     "Anna kaj Bert dividas 30 bombonojn tiel ke Bert ricevas trifoje pli da bombonoj por ĉiu duo de Anna. Kiom ricevas Bert?", "18"),

    # Axis 8: ABSOLUTE NUMERIC EQUIVALENT (no "ratio" word at all)
    ("phrasing: fraction-style",
     "Anna ricevas du kvinonon de 30 bombonoj kaj Bert tri kvinonon. Kiom ricevas Bert?", "18"),
    ("phrasing: percentages",
     "Anna ricevas 40% de 30 bombonoj kaj Bert ricevas la reston. Kiom ricevas Bert?", "18"),
]

print(f"loading {CKPT}", flush=True)
tok = PreTrainedTokenizerFast.from_pretrained("tokenizer_morpheme")
tok.add_special_tokens({"additional_special_tokens": list(SPECIAL)})
model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float16).cuda().eval()
model.resize_token_embeddings(len(tok))


def ask(text):
    p = pp(f"{USER} {text} {ASST} ")
    ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=400, do_sample=False, num_beams=1,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id,
                             repetition_penalty=1.1,
                             eos_token_id=tok.convert_tokens_to_ids(END))
    return decode(tok, out[0][ids.shape[1]:].tolist())


wins = 0
for axis, q, gold in VARIANTS:
    pred = ask(q)
    ok = has_answer(pred, gold)
    flag = "✓" if ok else "✗"
    wins += ok
    print(f"\n[{flag}] {axis}  gold={gold}")
    print(f"    Q: {q}")
    print(f"    pred: {pred[:240]}")

print(f"\n=== TOTAL: {wins}/{len(VARIANTS)} ===")
