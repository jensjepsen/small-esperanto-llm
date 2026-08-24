# ifbench_ai2 — vendored IFBench verifier chain

Allen AI's IFBench (Pyatkin et al., 2025 NeurIPS D&B) verifier package,
vendored under `scripts/ifbench_ai2/` so the whole `check_following`
chain is a plain Python import — no pip dep, no version drift.

Paper: https://arxiv.org/abs/2507.02833  
Upstream: https://github.com/allenai/IFBench  
License: Apache-2.0 (see `LICENSE`).

## Files

- `instructions.py` — 58 verifier classes (`WordCountRangeChecker`,
  `SpecialBulletPointsChecker`, `EmojiSentenceChecker`, `PalindromeChecker`,
  …). Verbatim from upstream.
- `instructions_registry.py` — `INSTRUCTION_DICT` mapping the 58 IFBench
  IDs (`count:*`, `ratio:*`, `words:*`, `sentence:*`, `format:*`,
  `custom:*`, `repeat:*`) to their classes. Verbatim.
- `instructions_util.py` — tokenization / stopword / syllable helpers
  used by the checkers. Verbatim; auto-downloads `punkt`, `punkt_tab`,
  `stopwords`, `averaged_perceptron_tagger_eng` from NLTK on first
  call.
- `__init__.py` — **our shim**. Vendored upstream files use bare
  `import instructions` / `import instructions_util` (top-level, not
  package-relative). Rather than patch the vendored code, this
  `__init__` prepends the package's own directory to `sys.path` at
  import time so those bare names resolve to the sibling files.
- `LICENSE` — upstream Apache-2.0.

## Pip deps introduced

Added to `pyproject.toml` under `dependencies`:

- `emoji>=2.0` — for `EmojiSentenceChecker`.
- `syllapy>=0.7` — for `AlternateParitySyllablesChecker`.

`nltk`, `absl-py`, `langdetect` were already present for
`scripts/ifeval_google/`.

## Consumers

- `scripts/eval_ifbench_da.py` — standalone benchmark eval against
  `jensjepsen/ifbench-da-v1` (300 rows; 4-way strict/loose ×
  prompt/instruction plus per-family breakdown).

## Sync policy

If a real bug needs picking up from upstream:

1. Diff upstream `instructions.py`, `instructions_registry.py`,
   `instructions_util.py` against our copies.
2. Replace verbatim (no local patches to preserve — the sys.path
   shim lives in `__init__.py`, not in the vendored files).
3. Run `uv run python scripts/eval_ifbench_da.py --ckpt CKPT --n 20`
   and confirm no crashes + per-family lines populate.

## Note vs `scripts/ifeval_google/`

`ifeval_google` was patched with Danish acceptance in
`CapitalLettersEnglishChecker` and DFM aliases in
`INSTRUCTION_DICT`. IFBench needs no such patches — its 58 verifiers
are already language-agnostic (counts, ratios, punctuation, format
tokens, letter positions). If a Danish-specific crash surfaces, patch
`scripts/eval_ifbench_da.py` (our wrapper), not the vendored files.
