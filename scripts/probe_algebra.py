"""Probe algebra capability on SFT-16k across difficulty levels."""
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


PROBES = [
    # Level 1: trivial linear
    ("trivial",     "Solvu: x + 3 = 7",            "4"),
    ("trivial",     "Solvu: x - 2 = 5",            "7"),
    ("trivial",     "Solvu: 2x = 10",              "5"),
    # Level 2: one-step neg, multi-step coef
    ("one-step",    "Solvu: 3x = 21",              "7"),
    ("one-step",    "Solvu: x + 8 = 5",            "-3"),
    ("one-step",    "Solvu: x/4 = 6",              "24"),
    # Level 3: two-step
    ("two-step",    "Solvu: 4x - 6 = 18",          "6"),
    ("two-step",    "Solvu: 2x + 5 = 11",          "3"),
    ("two-step",    "Solvu: 5x - 3 = 22",          "5"),
    # Level 4: distribution / parens
    ("distrib",     "Solvu: 2(x + 3) = 14",        "4"),
    ("distrib",     "Solvu: 3(x - 1) = 12",        "5"),
    # Level 5: x both sides
    ("both-sides",  "Solvu: 3x + 2 = x + 8",       "3"),
    ("both-sides",  "Solvu: 5x - 4 = 2x + 11",     "5"),
    # Level 6: fractions
    ("fraction",    "Solvu: x/2 + 3 = 7",          "8"),
    ("fraction",    "Solvu: (x+4)/3 = 5",          "11"),
    # Level 7: word problems
    ("word",        "Petro havas trifoje pli da pomoj ol Maria. Kune ili havas 16 pomojn. Kiom da pomoj havas Maria?", "4"),
    ("word",        "La sumo de du numeroj estas 20, kaj ilia diferenco estas 4. Kio estas la pli granda numero?", "12"),
    ("word",        "Aŭto veturas je 60 km/h. Kiom da kilometroj ĝi veturos en 3 horoj?", "180"),
    # Level 8: quadratic / nonlinear
    ("quad",        "Solvu: x^2 = 16",             "4"),
    ("quad",        "Solvu: x^2 - 9 = 0",          "3"),
]

def has_answer(pred, gold):
    """Strict-ish: gold number must appear AFTER an `=` or `####` marker (i.e.
    as a stated answer), not just anywhere in the chain. Avoids spurious matches
    on the gold digit appearing as a coefficient in the question (e.g. gold=6
    matching "subtraho de 6" or "4x − 6")."""
    # find all positions of `=` or `####` and look at digits in the next ~25 chars
    nums_after_marker = []
    for m in re.finditer(r"=\s*|####\s*", pred):
        tail = pred[m.end() : m.end() + 25]
        nums_after_marker.extend(re.findall(r"-?\d+", tail))
    return gold in nums_after_marker


def main():
    print(f"loading {CKPT}", flush=True)
    tok = PreTrainedTokenizerFast.from_pretrained("tokenizer_morpheme")
    tok.add_special_tokens({"additional_special_tokens": list(SPECIAL)})
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float16).cuda().eval()
    model.resize_token_embeddings(len(tok))

    def ask(text):
        p = pp(f"{USER} {text} {ASST} ")
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=300, do_sample=False, num_beams=1,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                 repetition_penalty=1.1,
                                 eos_token_id=tok.convert_tokens_to_ids(END))
        return decode(tok, out[0][ids.shape[1]:].tolist())

    from collections import Counter
    hit_by = Counter()
    tot_by = Counter()
    print()
    for level, q, gold in PROBES:
        pred = ask(q)
        ok = has_answer(pred, gold)
        tot_by[level] += 1
        if ok:
            hit_by[level] += 1
        flag = "✓" if ok else "✗"
        print(f"[{level:11s}] {flag} gold={gold:>5}  Q: {q}")
        print(f"              pred: {pred}")
        print()

    print(f"\n{'='*55}")
    for level in ["trivial","one-step","two-step","distrib","both-sides","fraction","word","quad"]:
        if tot_by[level]:
            print(f"  {level:11s} {hit_by[level]}/{tot_by[level]}")
    total = sum(hit_by.values())
    totsum = sum(tot_by.values())
    print(f"  {'TOTAL':11s} {total}/{totsum} = {100*total/totsum:.0f}%")


if __name__ == "__main__":
    main()
