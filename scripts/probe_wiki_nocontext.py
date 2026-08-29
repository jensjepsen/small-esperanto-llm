"""Probe wiki Qs from eval_handcrafted_v31 on SFT-16k both WITH and WITHOUT context.
Measures whether the model knows these facts from pretraining vs reads them off."""
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, "src")
from esperanto_lm.data import _morpheme_preprocess

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/sft/v10_sftv4/checkpoint-16000"
EVAL_FILE = "/mnt/data/espllm/data/causal_corpus/eval_handcrafted_v31.jsonl"

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
SPECIAL = (USER, ASST, END)
SKIP = {"<s>", "</s>", "<pad>", "<unk>", USER, ASST, END}

ANCHORS = ('ĉefurbo','profesio de zamenhof','naskiĝis ŝekspiro','naskiĝis einsteino',
           'konsistas akvo','orbitas ĉirkaŭ la suno','plej granda surtera besto',
           'unua mondmilito','mortis mozarto','amazona pluvarbo','fluas la nilo',
           'malkovris marie curie','natura satelito de la tero','kreis linukson',
           'granda muro','unue trinkis kafon','strukturon de dna','plej granda planedo')


def pp(s):
    pat = "(" + "|".join(re.escape(t) for t in SPECIAL) + ")"
    return " ".join(p if p in SPECIAL else _morpheme_preprocess(p)
                    for p in re.split(pat, s))


def decode(tok, ids):
    toks = tok.convert_ids_to_tokens(ids)
    toks = [t for t in toks if t not in SKIP]
    return "".join(t if t != "<w>" else " " for t in toks).strip()


print(f"loading {CKPT}", flush=True)
tok = PreTrainedTokenizerFast.from_pretrained("tokenizer_morpheme")
tok.add_special_tokens({"additional_special_tokens": list(SPECIAL)})
model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float16).cuda().eval()
model.resize_token_embeddings(len(tok))

# strip context: everything before "Demando:"
def split_ctx_q(prompt):
    if "Demando:" in prompt:
        ctx, q = prompt.rsplit("Demando:", 1)
        return ctx.strip(), "Demando:" + q.strip()
    return "", prompt


def ask(text):
    p = pp(f"{USER} {text} {ASST} ")
    ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=40, do_sample=False, num_beams=1,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id,
                             repetition_penalty=1.1,
                             eos_token_id=tok.convert_tokens_to_ids(END))
    return decode(tok, out[0][ids.shape[1]:].tolist())


wiki_qs = []
with open(EVAL_FILE) as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        q = r["messages"][0]["content"]
        if any(a in q.lower() for a in ANCHORS):
            wiki_qs.append((i, q, r["messages"][1]["content"], r["accepted_answers"]))

print(f"\n{len(wiki_qs)} wiki questions\n")

n_with_ctx = n_no_ctx = 0
for idx, prompt, gold, accepted in wiki_qs:
    ctx, q_only = split_ctx_q(prompt)
    pred_full = ask(prompt)
    pred_cold = ask(q_only)

    def hit(p):
        pl = p.lower()
        return any(a.lower() in pl for a in accepted) or gold.lower() in pl

    ok_full = hit(pred_full)
    ok_cold = hit(pred_cold)
    n_with_ctx += ok_full
    n_no_ctx += ok_cold
    flag_full = "✓" if ok_full else "✗"
    flag_cold = "✓" if ok_cold else "✗"
    print(f"[{idx}] gold={gold!r}")
    print(f"  Q: {q_only}")
    print(f"  {flag_full} WITH ctx: {pred_full!r}")
    print(f"  {flag_cold} NO   ctx: {pred_cold!r}")
    print()

print(f"\n=== with ctx: {n_with_ctx}/{len(wiki_qs)} = {100*n_with_ctx/len(wiki_qs):.0f}% ===")
print(f"=== no  ctx: {n_no_ctx}/{len(wiki_qs)} = {100*n_no_ctx/len(wiki_qs):.0f}% ===")
