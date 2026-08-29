"""More word problems on SFT-16k — wider variety, harder."""
import re
import sys
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, "src")
from esperanto_lm.data import _morpheme_preprocess

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/sft/v10_sftv4/checkpoint-16000"
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


PROBLEMS = [
    # (tag, question, expected_answer)
    ("ratio",
     "Anna kaj Bert dividas 30 bombonojn en rilato 2:3. Kiom da bombonoj ricevas Bert?",
     "18"),
    ("percent",
     "Ĉemizo kostas 40 eŭrojn. Ĝi estas rabatita je 25%. Kiom kostas la ĉemizo nun?",
     "30"),
    ("age",
     "Patrino estas trifoje pli aĝa ol sia filino. Post 10 jaroj, patrino estos dufoje pli aĝa ol la filino. Kiom da jaroj havas la filino nun?",
     "10"),
    ("rate-inverse",
     "Tri laboristoj farbas muron en 6 horoj. Kiom da horoj bezonas ses laboristoj por farbi la saman muron?",
     "3"),
    ("mixture",
     "Vi havas 200 ml-on da 10%-a salakva solvaĵo. Kiom da ml-oj da pura akvo vi devas aldoni por akiri 5%-an solvaĵon?",
     "200"),
    ("rectangle",
     "Rektangulo havas perimetron de 24 cm kaj longon, kiu estas duoble pli granda ol la larĝo. Kio estas la larĝo?",
     "4"),
    ("coin",
     "Ema havas 12 monerojn, ĉiuj aŭ kvinpencaj aŭ dekpencaj. La totala valoro estas 95 pencoj. Kiom da dekpencaj moneroj ŝi havas?",
     "7"),
    ("travel",
     "Trajno A foriras de stacio je 80 km/h. Du horojn poste, trajno B foriras de la sama stacio je 120 km/h en la sama direkto. Kiom da horoj post la foriro de trajno B ĝi atingos trajnon A?",
     "4"),
    ("simple-int",
     "Vi deponas 1000 eŭrojn ĉe 5% jara interezo. Kiom da eŭroj vi havos post 2 jaroj?",
     "1100"),
    ("digits",
     "Du-cifera nombro havas la sumon de la ciferoj egala al 9. Se vi turnas la ciferojn, la nombro pliiĝas je 27. Kio estas la origina nombro?",
     "36"),
    ("consec",
     "La sumo de tri sinsekvaj entjeroj estas 51. Kio estas la plej granda el ili?",
     "18"),
    ("split",
     "Mia patro estas 30 jarojn pli aĝa ol mi. Kune ni havas 50 jarojn. Kiom da jaroj mi havas?",
     "10"),
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


for tag, q, gold in PROBLEMS:
    pred = ask(q)
    print(f"\n[{tag}]  gold={gold}")
    print(f"  Q:    {q}")
    print(f"  PRED: {pred}")
