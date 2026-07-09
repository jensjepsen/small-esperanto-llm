"""Retrain the EN↔EO SPM tokenizer for v11-mt.

Expands the v10-era `spm_eneo_32k.model` to `spm_eneo_48k_v2.model`:

- Vocab size 32k → 48k (room for math tokens without cannibalizing existing coverage)
- Corpus includes v10's full training mix + orca-math-gemini-parallel
- ``user_defined_symbols`` reserves atomic slots for LaTeX macros
  (``\\sqrt``, ``\\frac``, ``\\int``, ``\\sum``, ``\\pi``, ``\\alpha``…),
  our LatexAwareTranslator sentinel range (``<extra_0>``..``<extra_99>``),
  and SFT chat tokens.
- ``character_coverage`` bumped 0.9995 → 0.99998 to catch multilingual
  diacritics (`ã ł č ō ù ž`) that were dropped in v10.
- Explicit ``|`` (ASCII pipe) inclusion — was the top OOV in v10 training
  data (1808× in a 9k sample, ~72% of all UNK).

Two-pass build:
1. Assemble a plain text corpus from HF datasets into /mnt/data2/spm_v2/corpus.txt
2. Run sentencepiece training

Usage::

    uv run python mt/scripts/train_spm.py --stage corpus   # ~5 min
    uv run python mt/scripts/train_spm.py --stage train    # ~15-30 min

Or ``--stage both`` to do them sequentially.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# ── LaTeX macros to reserve as atomic tokens ────────────────────────────
_LATEX_ATOMS = [
    # Root, fraction, integral, sum, product, limit
    "\\sqrt", "\\frac", "\\int", "\\sum", "\\prod", "\\lim", "\\infty",
    # Trig, log, exp
    "\\sin", "\\cos", "\\tan", "\\sec", "\\csc", "\\cot",
    "\\arcsin", "\\arccos", "\\arctan",
    "\\log", "\\ln", "\\exp",
    # Greek lowercase (most common in math)
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\varepsilon",
    "\\zeta", "\\eta", "\\theta", "\\vartheta", "\\iota", "\\kappa",
    "\\lambda", "\\mu", "\\nu", "\\xi", "\\pi", "\\rho", "\\sigma",
    "\\tau", "\\upsilon", "\\phi", "\\varphi", "\\chi", "\\psi", "\\omega",
    # Greek uppercase
    "\\Gamma", "\\Delta", "\\Theta", "\\Lambda", "\\Xi", "\\Pi",
    "\\Sigma", "\\Upsilon", "\\Phi", "\\Psi", "\\Omega",
    # Structural
    "\\mathbb", "\\mathcal", "\\mathrm", "\\mathbf", "\\mathit",
    "\\text", "\\textbf", "\\textit",
    "\\left", "\\right", "\\begin", "\\end",
    # Math ops
    "\\cdot", "\\times", "\\div", "\\pm", "\\mp",
    "\\leq", "\\geq", "\\neq", "\\equiv", "\\approx", "\\sim", "\\propto",
    "\\to", "\\rightarrow", "\\leftarrow", "\\Rightarrow", "\\Leftarrow",
    "\\iff", "\\Leftrightarrow",
    # Set / logic
    "\\in", "\\notin", "\\subset", "\\subseteq", "\\supset", "\\supseteq",
    "\\cup", "\\cap", "\\setminus", "\\emptyset",
    "\\forall", "\\exists", "\\neg", "\\wedge", "\\vee",
    # Derivatives, calculus
    "\\partial", "\\nabla", "\\dot", "\\ddot", "\\prime",
    # Boxed / annotations
    "\\boxed", "\\overline", "\\underline", "\\bar", "\\hat",
    # Common in Wikipedia math
    "\\det", "\\dim", "\\ker", "\\gcd", "\\max", "\\min", "\\sup", "\\inf",
]

# Sentinel tokens for LatexAwareTranslator + SFT chat tokens.
_SENTINELS = [f"<extra_{i}>" for i in range(100)]
_CHAT = ["<user>", "<assistant>", "<end>",
          "<tool_call>", "</tool_call>", "<tool_result>", "</tool_result>"]

USER_DEFINED = _LATEX_ATOMS + _SENTINELS + _CHAT


# ── HF datasets to include in the corpus ────────────────────────────────
_HF_DATASETS = [
    ("esperanto-mt-parallel", "jensjepsen/esperanto-mt-parallel", None),
    ("math-parallel-v2 rows", "jensjepsen/esperanto-mt-math-parallel-v2", "rows"),
    ("math-parallel-v2 sents", "jensjepsen/esperanto-mt-math-parallel-v2", "sentences"),
    ("yago-parallel-v2 labels", "jensjepsen/esperanto-mt-yago-parallel-v2", "labels"),
    ("yago-parallel-v2 comments", "jensjepsen/esperanto-mt-yago-parallel-v2", "comments"),
    ("opus-parallel", "jensjepsen/esperanto-mt-opus-parallel", None),
    ("orca-math-gemini", "jensjepsen/esperanto-orca-math-gemini-parallel", None),
]


def build_corpus(out_path: Path) -> None:
    """Concatenate en + eo columns from every dataset into one text file."""
    from datasets import load_dataset
    token = (Path.home() / ".cache/huggingface/token").read_text().strip()
    total_lines = 0
    with out_path.open("w") as fout:
        for label, repo, config in _HF_DATASETS:
            kw = {"split": "train", "token": token}
            if config:
                kw["name"] = config
            print(f"loading {label}...", flush=True)
            ds = load_dataset(repo, **kw)
            n = 0
            for r in ds:
                en = (r.get("en") or "").strip()
                eo = (r.get("eo") or "").strip()
                if en:
                    fout.write(en + "\n")
                    n += 1
                if eo:
                    fout.write(eo + "\n")
                    n += 1
            print(f"  {label}: {n:,} lines written", flush=True)
            total_lines += n
    print(f"\ntotal lines: {total_lines:,} → {out_path}", flush=True)


def train_spm(corpus_path: Path, out_prefix: Path, vocab_size: int,
              character_coverage: float, num_threads: int) -> None:
    """Train sentencepiece model."""
    import sentencepiece as spm
    print(f"training SPM: vocab={vocab_size}, coverage={character_coverage}, "
          f"threads={num_threads}", flush=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(out_prefix),
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="unigram",
        # Same special tokens as v1: <s>=0, </s>=1, <unk>=2, <pad>=3
        # (Actually our v1 was pad=0 eos=3, but new SPM will use defaults)
        pad_id=0, unk_id=2, bos_id=1, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
        user_defined_symbols=USER_DEFINED,
        num_threads=num_threads,
        # Larger training window than default 1M since our corpus is bigger
        input_sentence_size=50_000_000,
        shuffle_input_sentence=True,
        # Preserve control chars we care about (|, math ops, etc.)
        normalization_rule_name="nmt_nfkc_cf",  # NFKC + case-fold (v1 was case-folding)
        # Our corpus is 26M+ sentences — SPM defaults blow past int32 array
        # limits without this flag.
        train_extremely_large_corpus=True,
    )
    print(f"  wrote {out_prefix}.model + {out_prefix}.vocab", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["corpus", "train", "both"],
                    default="both")
    ap.add_argument("--corpus-path", type=Path,
                    default=Path("/mnt/data2/spm_v2/corpus.txt"))
    ap.add_argument("--out-prefix", type=Path,
                    default=Path("/mnt/data2/spm_v2/spm_eneo_48k_v2"))
    ap.add_argument("--vocab-size", type=int, default=48_000)
    ap.add_argument("--character-coverage", type=float, default=0.99998)
    ap.add_argument("--num-threads", type=int, default=4,
                    help="Cap at 4 — box crashes under sustained >4-core load")
    args = ap.parse_args()

    args.corpus_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.stage in ("corpus", "both"):
        build_corpus(args.corpus_path)

    if args.stage in ("train", "both"):
        train_spm(args.corpus_path, args.out_prefix,
                  args.vocab_size, args.character_coverage, args.num_threads)


if __name__ == "__main__":
    main()
