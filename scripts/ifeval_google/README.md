# ifeval_google — local fork

Google Research's IFEval verifier code (`instructions.py`, `instructions_registry.py`,
`instructions_util.py`) with **local patches** required to evaluate against the
Danish benchmark `danish-foundation-models/ifeval-da` and to run our combined
GRPO reward. **Do not re-vendor blindly from upstream** — the patches below
must be re-applied.

## Provenance

- Base: `google-research/google-research/instruction_following_eval/` (Apache-2.0)
- Vendored under `scripts/ifeval_google/` so the whole `check_following` chain
  is a plain Python import — no pip dep, no version drift.

## Local diffs vs upstream

Three intentional additions:

### 1. `instructions.py` — Danish acceptance on English-suffixed classes

`CapitalLettersEnglishChecker.check_following` and
`LowercaseLettersEnglishChecker.check_following` now accept
`langdetect.detect(value) in ("en", "da")` (was `== "en"`). Prevents 0-scoring
of correctly-cased Danish responses to `change_case:english_*` rules.

### 2. `instructions.py` — new language-agnostic case classes

`CapitalLettersChecker` and `LowercaseLettersChecker` (right above
`CommaChecker`). Pure `str.isupper()` / `str.islower()` over alpha chars,
**no language filter**. Reason: `langdetect` mis-classifies ALL-CAPS text
as German (`de`) because it profiles on lowercase word N-grams — so
even the Danish-patched English classes fail on the very output the
constraint asks for.

### 3. `instructions_registry.py` — DFM alias entries

`danish-foundation-models/ifeval-da` uses the constraint IDs
`change_case:capital_letters` (n=25 rows) and `change_case:lowercase_letters`
(n=39 rows). Google upstream never shipped these names; DFM renamed the
keys without publishing matching verifier code. Aliases in `INSTRUCTION_DICT`
route them to the new language-agnostic classes; the same aliases are
mirrored into `INSTRUCTION_CONFLICTS[_LANGUAGE + "response_language"]`.

Without those aliases, `build_instructions` silently skips 64 of 89
change_case rules in ifeval-da (72%) and the per-family score reads
0/25 — a verifier-registry bug, not a model deficiency. Post-patch, the
same rules score ~65-70% pass on the same model.

## Consumers

- `scripts/eval_ifeval_da.py` — standalone benchmark eval (541 rows,
  4-way strict/loose × prompt/instruction).
- `src/esperanto_lm/rl_rewards.py::reward_ifeval_combined` — GRPO reward
  that dispatches by `google:` name prefix.
- `scripts/train_grpo_verifier.py::IFEvalDACallback` — in-training
  eval callback used for `--task combined` runs.

## Sync policy

If a real bug needs picking up from upstream:

1. Diff upstream `instructions.py` and `instructions_registry.py` against our
   copies (only carry over the specific fixes; do not replace whole files).
2. Re-apply the three diffs above.
3. Run `uv run python scripts/eval_ifeval_da.py <ckpt> --limit 20` and
   spot-check the per-family output includes non-zero `change_case`.
