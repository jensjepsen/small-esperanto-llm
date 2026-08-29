"""Quick base-LM probes: raw text completions + per-passage perplexity."""
import re
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM

sys.path.insert(0, "src")
from esperanto_lm.data import load_tokenizer
from esperanto_lm.morphology import decompose

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data2/checkpoints/lm/v11_h100/checkpoint-5000"


def morph_pp(text, tok):
    has_w = "<w>" in tok.get_vocab()
    words = re.findall(r"[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+|[^\s]", text)
    parts = []
    for w in words:
        if parts and has_w: parts.append("<w>")
        if w and w[0].isalpha(): parts.extend(decompose(w))
        else: parts.append(w)
    return parts


def decode_ids(tok, ids, skip=None):
    skip = skip or {"<s>", "</s>", "<pad>", "<unk>"}
    toks = tok.convert_ids_to_tokens(ids)
    toks = [t for t in toks if t not in skip]
    return "".join(t if t != "<w>" else " " for t in toks).strip()


def main():
    print(f"loading {CKPT}", flush=True)
    tok = load_tokenizer(Path("tokenizer_morpheme"))
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"params: {sum(p.numel() for p in model.parameters()):,}\n")

    # ── PROBE 2: raw text completions ──
    print("=" * 70)
    print("PROBE 2: raw text completions (greedy, +50 tokens)")
    print("=" * 70)
    prompts = [
        "En la jaro 1969, homo unuafoje paŝis sur la",
        "Esperanto estas internacia lingvo, kreita de",
        "La akvo bolas je",
        "Por solvi la ekvacion 2x + 3 = 11, ni unue",
        "Albert Einstein estis fama pro sia teorio de",
        "La planedo Marso estas",
        "Iam estis princino kiu loĝis en",  # narrative start
        "La domo havis tri ĉambrojn:",
    ]
    for p in prompts:
        ids = tok(" ".join(morph_pp(p, tok)), return_tensors="pt",
                  add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=50, do_sample=False,
                                 num_beams=1, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                 repetition_penalty=1.1)
        cont = decode_ids(tok, out[0][ids.shape[1]:].tolist())
        print(f"\nPROMPT: {p!r}")
        print(f"  →  {cont}")

    # ── PROBE 3: per-passage perplexity ──
    print("\n" + "=" * 70)
    print("PROBE 3: per-passage perplexity on held-out text")
    print("=" * 70)
    passages = {
        "wiki-style":
            "Esperanto estas la plej parolata internacia helplingvo en la mondo. "
            "Ĝi estis kreita en la malfrua 19-a jarcento de la pollanda kuracisto "
            "L. L. Zamenhof, kiu volis ke ĝi servu kiel neŭtrala dua lingvo.",
        "news-style":
            "Hieraŭ la registaro anoncis novan programon por subteni malgrandajn "
            "entreprenojn. La ministro klarigis ke la celo estas helpi al firmaoj "
            "transiri al pli verda ekonomio en la sekvaj kvin jaroj.",
        "narrative":
            "La maljunulino malfermis la pordon kaj rigardis al la nokto. Ekstere "
            "neĝis silente, kaj ŝi povis vidi nur la flagrojn de stratlampoj tra "
            "la blanka kurteno.",
        "technical":
            "La teorio de relativeco priskribas la rilaton inter spaco kaj tempo. "
            "Speciala relativeco montras ke la lumrapideco estas konstanta en ĉiuj "
            "inertaj referenckadroj.",
        "algebra-chain":
            "Solvu: 5x + 3 = 18. Subtrahu 3: 5x = 15. Dividu per 5: x = 3.",
    }
    for name, text in passages.items():
        ids = tok(" ".join(morph_pp(text, tok)), return_tensors="pt",
                  add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            out = model(ids, labels=ids)
        loss = out.loss.item()
        ppl = torch.exp(out.loss).item()
        n_tok = ids.shape[1]
        print(f"\n  {name:14s}  loss={loss:.3f}  ppl={ppl:.2f}  ({n_tok} tok)")


if __name__ == "__main__":
    main()
