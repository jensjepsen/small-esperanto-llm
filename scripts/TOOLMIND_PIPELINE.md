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

# 4. render to the trainer's chat format. The pedagogy filters run here by
#    default (--keep-defective reproduces v6 and earlier, which lacked them).
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
| 1 | `GATE: n/m clean` in the log | ~96%, and all 9 planted controls FAIL |
| 2,3 | `scripts/build_returns_groups.py` | declared-vs-present F1 ~97% (v5: 62%) |
| 4 | `trainer check over 400 rows` | CLEAN |
| 4 | `pedagogy filters:` line in the log | ~665 dropped; rendered corpus has 0 rows without a `tool_call` |
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

**A faithful translation of a bad conversation is still bad.** The gate checks
translation fidelity and format validity; nothing asked whether a row teaches
the behaviour we want, so two populations passed every check while teaching the
opposite. Both come from ONE upstream batch — rows 19258..20016 of
`glaive-function-calling-v2-query.jsonl`, the last 759 of 20,017 — and order
survives translation, so it stayed contiguous all the way to sft rows
17004..17666 in v6.

| | body (0–19257) | tail (19258–end) |
|---|---|---|
| first assistant turn is prose, no call | 8.78% | 96.05% |
| row never calls a tool at all | 0.04% (8 rows) | 72.20% (548 rows) |

`no-tool-call` answers a tool-shaped question in prose. `deflection-retry` is
worse: the assistant declines a task its catalogue covers, a canned user turn
says more functions are available, and it then calls correctly — v6 row 17283
refuses `check_flight_status` and invents referral URLs first. Those rows
CONTAIN a call, so a no-call rule misses all 186 of them, and the trained
target is a competent-sounding refusal.

Matched on the canned turn, not on structure: structure would also catch
"clarify, then call", which is 1,684 legitimate body rows. Checked in the
English original AND the Danish, because `orig` matches 735 rows and `da` only
593. Measured on v6's data: 665 dropped, **zero** outside the batch except the
8 body no-call rows, and the collateral defects fall with it — reasoning-as-answer
394 → 25, LaTeX residue 300 → 24.

**A do-not-translate list is not a do-insert list.** `BEVAR UÆNDRET ... også
midt i en dansk sætning` was read as an obligation to INSERT the pinned name:
the source says "an annual interest rate of 5%" and the translation says "en
årlig interest_rate på 5%"; "my user id is 12345" becomes "mit user_id er
12345". Only 1 of 163 sampled had the identifier in the English at all.

Measured on v6: 634 rows inject an identifier, across 114 distinct keys. This
is worse than the failure it mirrors -- a user who says `interest_rate` hands
the model the schema mapping in the prompt, so it never has to read the
description, which also makes symbolizing keys pointless while it is
uncorrected. The forward check (`dnt-token-lost`) is structurally blind to it:
a translation that ADDS an identifier loses nothing.

Fixed in three places: the prompt now says to keep pinned names only where
they already occur; `identifier-injected` is a gate check with its own planted
control; and `_pinned_names()` no longer pins ordinary-word parameter keys.
That last one closed an instruction/enforcement mismatch -- 244 of 898 pinned
tokens were ordinary English words (`title`, `location`, `year`, `subject`)
that the gate never enforced, because `dnt-token-lost` filters to
`IDENT_STRICT`. The list is now 687 tokens, of which the 33 non-identifiers
are tool names and character-dependent values (`racecar`, `Hello, World!`).

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
- The pedagogy filters landed AFTER v6 was built and pushed, so
  `danish-tool-dialogues-v6` still carries all 665 rows — 593 of them in
  `sft:train` at indices 17004..17666, plus 22/722 in `eval_seen_tools` and
  13/779 in `eval_unseen_tools`. `abstention` is clean. v40 trains on them.
- v6 and earlier were rendered with `make_catalogue` seeded on the row's
  POSITION, so their catalogues cannot be reproduced by the current code. A
  v7 rebuild redraws every catalogue once, on the stable source `idx`. That is
  a one-time break, and it is the last one: after it, dropping rows leaves
  every survivor byte-identical (verified at 100.000% over 14,535 rows).
- 25 rows still ship a reasoning trace as the answer and 24 carry LaTeX
  residue (`\text{` → TAB + `ext{`, `\boxed` → BS + `oxed`), a 0.13% body-rate
  background the batch filters do not reach.
