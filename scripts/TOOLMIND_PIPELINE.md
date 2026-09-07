# Danish tool-dialogue corpus (ToolMind → `danish-tool-dialogues-vN`)

Source: `Nanbeige/ToolMind`, file `open_datasets/glaive-function-calling-v2-query.jsonl`
(~20k rows, the smallest of eight files in that repo).

Run the stages in order. Each writes into one `--out` directory and is
resumable: rerunning skips whatever is already cached, so a failed stage costs
only what it had not finished.

```bash
OUT=scratch/toolmind_da_v6fresh

# 1. translate. --drop-reasoning FIRST: the reasoning before a call is 64.8% of
#    the translatable text and the renderer discards it anyway, so translating
#    it is money spent on output that never reaches the model (~$5.19 -> ~$2.40).
#    Also builds spec_map, value_map and returns_map, then gates every row.
uv run python scripts/translate_toolmind_da.py --out $OUT \
    --drop-reasoning --n 25000 --concurrency 48

# 2. contracts for signatures that were never observed returning anything.
#    --merge-into is part of this stage, not a shell step: it appends the
#    proposals to the returns map and SKIPS any signature that already has
#    observation-derived keys.
uv run python scripts/gen_missing_returns.py --src $OUT \
    --batch 8 --concurrency 24 \
    --out $OUT/proposed_returns.jsonl --merge-into $OUT/returns_map.jsonl

# 3. attach the contracts to each row's own tools
uv run python scripts/translate_toolmind_da.py --out $OUT --n 25000 --annotate

# 4. render to the trainer's chat format
uv run python scripts/render_toolmind_sft.py --in $OUT --clean-only \
    --catalogue-size 6 --catalogue-min 2 --no-reasoning --out $OUT/sft.jsonl

# 5. answer the dangling terminal calls (97% of rows end at one)
uv run python scripts/gen_tool_answer_turns.py --rows $OUT/sft.jsonl \
    --out $OUT/sft_answered.jsonl --cache scratch/tool_answers_v6/answers.jsonl \
    --rejects scratch/tool_answers_v6/rejects.jsonl --concurrency 24

# 6. abstention rows: result lacks the asked-for field / no capable tool
uv run python scripts/gen_abstention_rows.py --rows $OUT/sft_answered.jsonl \
    --n 2500 --concurrency 24 --out scratch/abstention/final.jsonl

# 7. push. --abstention goes through THIS script, see "Gotchas".
uv run python scripts/push_tool_dialogues_hf.py --src $OUT \
    --repo jensjepsen/danish-tool-dialogues-vN \
    --catalogue-size 6 --catalogue-min 2 --no-reasoning \
    --answers scratch/tool_answers_v6/answers.jsonl \
    --abstention scratch/abstention/final.jsonl
```

Optional, after stage 1 and before stage 3 — a second look at lexicon entries
the translator left unchanged, which is where ordinary words hide among proper
nouns (`Italian`, `chess`, `laptop`). Re-run stage 1 with `--respec` afterwards
so the corpus picks the changes up:

```bash
uv run python scripts/revisit_unchanged_values.py --map $OUT/value_map.jsonl --apply
uv run python scripts/translate_toolmind_da.py --out $OUT --n 25000 --respec
```

## Acceptance checks

Each stage has a number that says whether it worked. Run them; the structural
checks are all green in the failure modes below.

| After | Check | Expect |
|---|---|---|
| 1 | `GATE: n/m clean` in the log | ~99%, and all 8 planted controls FAIL |
| 2,3 | `scripts/build_returns_groups.py` | declared-vs-present F1 ~97% (v5: 62%) |
| 4 | `trainer check over 400 rows` | CLEAN |
| 5 | accept rate in the log | ~67%; `answer-cites-irrelevant` should be non-zero at scale |
| 6 | opening distribution printed by the script | no opening above ~10% |
| 7 | `get_dataset_config_names(repo)` | includes `abstention` |

## Why the corpus is shaped this way

**A tool NAME is not a function.** 379 of 875 names carry more than one
parameter schema; `search_movies` carries 63, because glaive invented every
dialogue independently. Keying anything on the name alone merges unrelated
functions — that is how `search_quotes` acquired an AAPL stock ticker. The
identity is `(name, sorted parameter names)`, and `_signature()` computes it in
three places that must agree: the returns map, the missing-returns proposals,
and the eval-split check.

**Contracts come from observation where observation exists.** Fields are kept
when they appear in ≥90% of a signature's real payloads; a signature whose
fields all fall below that keeps its most frequent ones anyway. Proposals fill
only signatures never observed at all. Getting this backwards scored 2,005
observed payloads against invented contracts at F1 25.4%, where
observation-derived contracts score 97.4%. Map entries carry
`src: observed|proposed` so the rule stays checkable.

**Answers are scored on precision, not just grounding.** Requiring only "cites
some payload value" makes reciting the whole payload optimal: 43% of v5's
answers cited every field, and the eval — recall over all payload values —
ranked a padded model answer above the reference reply (47.8% vs 26.1%). The
generator now declares `relevante_felter`, the gate rejects answers that miss
them or drag in others, and `tool_answer` is F1 against the fields gold cites.

**Catalogue position carries no signal.** The source lists the called tool
first in 98.4% of multi-tool rows, so "call tool #1" scored 99.2% right-tool —
better than the trained model. Catalogues are shuffled and padded to a size
drawn per row.

## Gotchas

- **`--n` defaults to 25.** A full run needs `--n 25000`. Without it you get a
  25-row corpus and the log looks normal.
- **The push script rewrites the dataset card**, which de-registers any config
  it did not create. `abstention` must be pushed *after* the card, which is why
  it goes through `--abstention` rather than a separate script. It silently
  vanished twice; the push log said "pushed abstention (694)" both times, so
  `get_dataset_config_names` is the only check that means anything.
- **Changing a value desynchronises the prose that quoted it.** `--respec`
  rewrites assistant prose to match; without that, `stale-prose-echo` fires
  (72 rows, the first time the lexicon moved under an already-translated
  corpus).
- **Unit slots are pinned.** `miles -> mil` is not a rendering choice: Danish
  *mil* is 10 km, so it changes what the call means against a result computed
  in miles. Same reason the palindrome and word-count tools pin their values.
- **The answer cache keys on the tool's `returns` contract**, canonicalised.
  Edit a contract and exactly the affected rows regenerate; edit nothing and
  the cache stays warm. This is what keeps a re-run from re-buying ~$5.50 of
  answers.

## Costs (v6, ~19.5k rows)

| Stage | ~USD |
|---|---:|
| translate, reasoning dropped | 2.40 |
| returns descriptions | 1.35 |
| missing-returns proposals | 0.75 |
| answers | 5.50 |
| value revisit + abstention | 1.50 |
| **total** | **~11.50** |

## Known limitations

- 3,239 of 6,389 contract fields are **proposed, not observed** — invented from
  the tool's name and description for signatures that never returned anything.
  ToolMind's `APIGen-MT-5k` and `tau-train` files carry real results for 85% and
  80% of their calls and would replace these.
- Generated payloads are independent per call, so a multi-turn dialogue can
  imply two different "today" (age 31 for a 1990 birth, 23 for a 2000 birth).
- Abstention is 3.6% of rows, below the 10–15% intended: flattening the opening
  distribution discards more than half the generated candidates.
- No "should have called" counter-metric yet, so refusal spreading to
  answerable questions would not be detected.
