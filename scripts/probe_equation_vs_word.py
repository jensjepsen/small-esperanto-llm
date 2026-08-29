"""Probe whether the model solves the same equation differently when
framed as bare 'Solvu: ax+b=c' vs embedded in a word problem."""
import re
import sys
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, "src")
from esperanto_lm.data import _morpheme_preprocess

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


# Pairs: (bare_equation, word_problem_requiring_same_equation, gold)
PAIRS = [
    ("Solvu: 4x - 6 = 18",
     "Petro havas kvarfoje pli da pomoj ol Maria, minus 6. Petro havas 18 pomojn. Kiom da pomoj havas Maria?",
     "6"),
    ("Solvu: 2x + 5 = 11",
     "La aĝo de Hugo estas duoble plus 5 jaroj la aĝo de lia fratino Sara. Hugo havas 11 jarojn. Kiom da jaroj havas Sara?",
     "3"),
    ("Solvu: 3x = 21",
     "Anna kaj du amikoj dividas 21 bombonojn egale. Kiom da bombonoj ricevas Anna?",
     "7"),
    ("Solvu: x/4 = 6",
     "Klara dividas siajn pomojn egale inter 4 amikoj. Ĉiu amiko ricevas 6 pomojn. Kiom da pomoj havis Klara entute?",
     "24"),
    ("Solvu: 2(x + 3) = 14",
     "Du knaboj havas po x + 3 librojn. Entute ili havas 14 librojn. Kio estas x?",
     "4"),
    ("Solvu: x + 8 = 5",
     "Lukas ŝuldas 5 eŭrojn. Se li prunteprenas 8 eŭrojn pli, li havos x eŭrojn. Kio estas x?",
     "-3"),
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


sys.path.insert(0, "scripts")
from probe_algebra import has_answer

for bare, word, gold in PAIRS:
    pred_bare = ask(bare)
    pred_word = ask(word)
    ok_bare = has_answer(pred_bare, gold)
    ok_word = has_answer(pred_word, gold)
    print(f"\n=== gold={gold} ===")
    print(f"BARE   ({'✓' if ok_bare else '✗'}) Q: {bare}")
    print(f"           pred: {pred_bare[:280]}")
    print(f"WORD   ({'✓' if ok_word else '✗'}) Q: {word}")
    print(f"           pred: {pred_word[:280]}")
