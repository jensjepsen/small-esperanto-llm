"""Eval TriviaQA-EO with greedy generation + alias-aware substring match."""
import re
import sys
import unicodedata
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, "src")
from esperanto_lm.data import _morpheme_preprocess

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
SPECIAL = (USER, ASST, END)
SKIP = {"<s>", "</s>", "<pad>", "<unk>", USER, ASST, END}

ARTICLES_EO = {"la", "kaj", "de", "en", "al", "el", "je"}
ARTICLES_EN = {"a", "an", "the", "and", "of", "in", "to"}


def pp(s):
    pat = "(" + "|".join(re.escape(t) for t in SPECIAL) + ")"
    return " ".join(p if p in SPECIAL else _morpheme_preprocess(p)
                    for p in re.split(pat, s))


def decode(tok, ids):
    toks = tok.convert_ids_to_tokens(ids)
    toks = [t for t in toks if t not in SKIP]
    return "".join(t if t != "<w>" else " " for t in toks).strip()


def normalize(s: str) -> str:
    s = s.lower()
    # strip accents/diacritics for fuzzy match (keeps EO ĉ → c etc.)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # drop articles
    words = [w for w in s.split() if w not in ARTICLES_EO and w not in ARTICLES_EN]
    return " ".join(words)


def matches(pred: str, golds: list[str]) -> bool:
    np_ = normalize(pred)
    for g in golds:
        ng = normalize(g)
        if ng and ng in np_:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("n", type=int, default=100, nargs="?")
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--prompt-format", choices=["chat", "demando"], default="chat",
                    help="chat=<|user|>…<|assistant|> (SFT). "
                         "demando='Demando: …\\nRespondo:' (base-LM pretrain format).")
    args = ap.parse_args()

    print(f"loading {args.ckpt}  n={args.n}", flush=True)
    tok = PreTrainedTokenizerFast.from_pretrained("tokenizer_morpheme")
    tok.add_special_tokens({"additional_special_tokens": list(SPECIAL)})
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.float16).cuda().eval()
    model.resize_token_embeddings(len(tok))

    ds = load_dataset("jensjepsen/esperanto-triviaqa", split="validation").select(range(args.n))

    n_ok = 0
    for i, row in enumerate(ds, 1):
        q = row["question"]
        gold_answers = [row["answer"]]
        # also accept English aliases (proper nouns often verbatim)
        for a in row.get("en_aliases", []) or []:
            if a not in gold_answers:
                gold_answers.append(a)

        if args.prompt_format == "demando":
            prompt = pp(f"Demando: {q}\nRespondo: ")
            eos_id = tok.eos_token_id  # base LM doesn't know END
        else:
            prompt = pp(f"{USER} {q} {ASST} ")
            eos_id = tok.convert_tokens_to_ids(END)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=args.max_new, do_sample=False, num_beams=1,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                repetition_penalty=1.1,
                eos_token_id=eos_id,
            )
        pred = decode(tok, out[0][ids.shape[1]:].tolist())
        ok = matches(pred, gold_answers)
        n_ok += ok
        flag = "✓" if ok else "✗"
        print(f"[{i}/{args.n}] {flag} gold={row['answer']!r}", flush=True)
        if i <= 8 or not ok:
            print(f"   Q: {q}")
            print(f"   pred: {pred[:180]}")

    print(f"\n=== triviaqa {n_ok}/{args.n} = {100*n_ok/args.n:.1f}% ===")


if __name__ == "__main__":
    main()
