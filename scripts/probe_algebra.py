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

def extract_final_answer(pred: str) -> str | None:
    """Pull the model's stated final answer. Preference order:
      1. Number immediately after the LAST `####`
      2. Otherwise: the LAST `x = N` where N is a SOLO whole number / decimal
         (not a numerator of a fraction like `1/3` and not glued to other digits)

    No further fallback — if the model never wrote `####` or a clean `x = N`,
    it didn't actually answer, and we say so (returns None → grades as ✗).
    This trades some real wins (model that says "estas 4" but never writes
    `#### 4` or `x = 4`) for zero spurious matches.
    """
    # 1) `#### N` — the morpheme tokenizer splits `#` chars with spaces so the
    #    decoded chain reads `# # # # 42`, not `#### 42`. Match either form.
    last_hash = None
    for m in re.finditer(r"(?:#\s*){2,}\s*(-?\d+(?:\.\d+)?)", pred):
        last_hash = m.group(1)
    if last_hash is not None:
        return last_hash
    # 2) `x = N` where N is followed by whitespace, punctuation, or end-of-string
    #    (rejects `x = 1/3` matching "1", and `x = 24/4` matching "24").
    #    Allow optional minus (with possible space) e.g. `x = -3` or `x = - 3`.
    last_x = None
    for m in re.finditer(
        r"\bx\s*=\s*(-\s*)?(\d+(?:\.\d+)?)(?![\d./])",
        pred,
    ):
        sign, num = m.group(1), m.group(2)
        last_x = ("-" + num) if sign else num
    return last_x


def has_answer(pred, gold):
    """Strict: the model's STATED final answer must equal `gold`."""
    final = extract_final_answer(pred)
    if final is None:
        return False
    try:
        return float(final) == float(gold)
    except ValueError:
        return final == gold


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
