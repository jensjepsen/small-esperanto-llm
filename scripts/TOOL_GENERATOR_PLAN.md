# Generating Danish tool-call data (plan)

Replaces `gen_tool_call_sft.py`, which predates the tool-dialogue pipeline and
cannot feed it. Written 2026-09-10 off the ToolACE smoke; every number cited
here was measured this session unless marked otherwise.

## Why generate rather than translate

Translation inherits the source's payload shape, and payload shape is the
lever we have not pulled. Measured:

    v8   87.9% of answer turns carry exactly ONE number
    ToolACE 56.0% "one number, nothing to discriminate"

"Emit the number you just saw" is a perfect policy on most of the corpus, and
answer-turn payload values cost CE 0.020 against 0.197 for the Danish around
them -- they are copied, not selected. `gen_return_distractors` can only
RETROFIT a choice onto payloads that already exist, and it is capped by
`no-type-overlap`: 76.9% of tools on v8, 46.1% on ToolACE, whose returns are
prose and arrays.

A generator invents the tool, so it can mint competing fields that are the
same type and similar magnitude BY CONSTRUCTION, at 100%. That is the whole
argument. Everything else below is bookkeeping.

Secondary: our gates only ever see turns we generate. On ToolACE the source
turns arrive ungated and carry 3.8% unanswered mid-dialogue calls, 5.1%
`prostring` corruption, 1.9% blanked assistant turns, and a role-swapped row.
Generating puts every turn inside the gates.

## Output contract -- generate INTO the pipeline, not beside it

The generator emits exactly the shape `translated.jsonl` has:

    {"idx": N, "da": {"tools": [{"type":"function","function":{...}}],
                      "conversations": [{"role":..., "content":..., "tool_calls":[...]}]}}

Then stages 4 (render), 5.6 (inject), 6 (abstention), 7 (twins), 8 (push) and
`audit_distractors.py` run UNCHANGED. Stages 1-3 (translate, contracts,
annotate) are skipped -- the generator is their replacement. Stage 5
(`gen_tool_answer_turns`) is used only where we deliberately leave a call
dangling.

This is the same requirement the ToolACE parser was held to. Do not invent a
second format.

## Conventions

* Tool names and parameter names in ENGLISH (`get_stock_quote`,
  `phone_number`). Matches v9, the eval splits, the scorers, and
  `symbolize_twins`, which maps identifiers to `p`/`r` symbols.
* Every description and every dialogue turn in DANISH.
* JSON-schema vocabulary: `name`/`description`/`parameters`/`properties`,
  types `string`/`integer`/`number`/`boolean`/`object`/`array`.
* OpenRouter only. Delete the Gemini path.

The old generator's all-Danish schema vocabulary (`navn`, `parametre`,
`argumenter`, type `tekst`) is why its 37k rows have never been usable
downstream. Changing it means the new corpus does not match published
`danish-tool-calls-v1`; that is intended, not a migration.

## What survives from `gen_tool_call_sft.py`

Keep (~150-200 of 879 lines), structurally language-neutral:

* `sample_template` + `N_PARAM_WEIGHTS`/`TYPE_WEIGHTS` -- the structural
  grammar over n_params, types, required/optional, decorators, nesting.
* `template_hash` -- lets coverage over schema shapes be MEASURED, not hoped
  for.
* `check_tool_schema_shape`, `DistractorPool` (scenario-aware catalogue
  distractors), the difficulty buckets.
* The expanded scenario pool from `expand_tool_scenarios.py`. Scenarios are
  domains, not identifiers -- unaffected by the vocabulary change.

Discard: every prompt (Danish schema vocabulary is baked in), the emitter
(`build_sft_row`, `format_catalog`), the Gemini path, and `validate_example`
-- it validates the old shape and has no planted controls.

## Schema source is a TRAIN/EVAL split decision

    train      harvest the 2,907 schemas from parse_toolace_original.py
               (names/descriptions/parameters only -- NOT the payloads, which
               are what capped distractors), descriptions translated to Danish
    held-out   INVENT via the template grammar

Rationale: a fixed pool has to serve both splits, and the recurring failure in
this project is eval splits the model has effectively memorised -- the ICL
work needed freshly minted disjoint schemas for exactly this, and
`tool_unseen` already overstates by holding out only the NAME (symbolized A/B
1.000 -> 0.600). An invented-tool generator is an unlimited source of schemas
the model provably has not seen. That capability is the reason to keep the
grammar alive even though ToolACE supplies the training vocabulary.

## Stages

### A. Tool + contract (one LLM call per tool)

Input: a scenario, plus either a harvested schema or a sampled template.
Output: `{name, description, parameters, returns}` with the contract designed
alongside the tool, never bolted on.

The `returns` block must be built to pose a CHOICE:

* at least two fields of the SAME TYPE whose values will be of SIMILAR
  MAGNITUDE -- `{"cups_left": 8, "floor": 4}` is the template;
* the question decides which is correct, so the description is the only
  discriminator;
* no field that merely echoes a parameter (the old generator's payloads echo
  the call arguments; only 79% carry >=2 non-echoed fields);
* instrumentation (`status`, `request_id`, `latency_ms`) is not a distractor.

Reuse the gates already written in `gen_return_distractors.py`:
`echoes-parameter`, `instrumentation-not-domain`, `no-type-overlap`,
`no-numeric-distractor`, `examples-not-varied`, `placeholder-example`. They
are calibrated and have 16 planted controls.

EVERY tool gets a contract, including catalogue padding. Skipping padding is
what made a `returns` block identify the tool to call in 66.7% of ToolACE
rows.

### B. Dialogue plan (no LLM)

Sample turn structure and difficulty. Target the rates the corpus lacks
rather than the old generator's distribution (`multi_chain` 23/37,032 =
0.06%):

    single call            ~40%
    parallel calls          ~25%   (ToolACE runs 38.2%)
    multi-turn, >=2 calls   ~20%   (ToolACE 22.2%)
    clarify / ask-for-arg   ~10%   (bucket already exists)
    refuse / no capable     ~5%    (bucket already exists)

Catalogue size 2-6, drawn from `DistractorPool`. Single-tool catalogues teach
no selection; the old generator had 21% of them.

### C. Turns (one LLM call per dialogue)

Generate the Danish user utterances and assistant turns against the tool,
the contract, and the plan. No reasoning text -- the renderer discards it and
`--drop-reasoning` exists precisely because it was 64.8% of translatable text
and never reached the model.

### D. Payload synthesis (no LLM)

Values from the contract, per row, derived from a hash of `(idx, field)` so
they are deterministic and varied -- reuse `toolmind_distractor_values.py`,
which already classifies INT/FLOAT/DATE/DATETIME/IDLIKE/NUMUNIT/ENUM/FREE and
has controls asserting >=50 distinct values per 500 rows. `FREE` fields draw
from the LLM-sampled banks (`gen_distractor_value_banks.py`, draw-count
sized).

The gold answer must cite the asked-for field and NOT its same-type
competitor. That is the training signal.

### E. Gates and controls

Non-negotiable, and the reason to write this rather than patch the old
script: every stage carries planted defects that must fail, in the style the
pipeline already uses. Current batteries for reference: answer turns 12
defects / 10 clean, distractors 18/2, missing-returns 7/2, translate 11.

Minimum new controls:
* a payload whose competitor is cited -> rejected
* a tool whose contract has no same-type pair -> rejected
* a catalogue where only the called tool has `returns` -> rejected
* a dialogue whose call is not in its catalogue -> rejected
* a turn containing English prose -> rejected
* role-ordering: no user-after-user, no call without result except a
  deliberate terminal dangling call

Then reuse `audit_distractors.py` unchanged -- it already asserts no cited
distractor, no undeclared payload key, no twin carrying a distractor under
its English name, and computes the payload-composition table.

### F. Acceptance targets

Read against the ToolACE smoke and v9:

    "one number, nothing to discriminate"   56.0%  ->  <20%
    "several, gold cites SOME"              24.2%  ->  >60%
    tools with a distractor contract        46.1%  ->  ~100%
    called tool is the ONLY one w/ returns   1.3%  ->  <2%   (hold)
    corruption / role-swap / stub answers    5.1%  ->  0%    (by construction)
    parallel calls                            n/a  ->  ~25%
    multi-turn                                n/a  ->  ~20%

## Cost

Two LLM calls per dialogue (tool+contract, turns), amortised because one tool
serves several dialogues. Measured neighbours: distractor proposals
$0.000358/tool, value banks $0.00014/field, ToolACE 200-row translation ~$1
including waste. A 20k-row corpus plausibly lands in the low tens of dollars,
but that is an extrapolation, not a measurement -- smoke 200 rows and read the
printed cost before committing to a build.

## Order of work

1. Emitter + conventions: template grammar -> English identifiers, Danish
   descriptions, `translated.jsonl` shape. Verify by running stage 4 on the
   output and getting `trainer check CLEAN`.
2. Stage A with the distractor-aware contract, reusing the existing gates.
3. Stages B-D, then the controls battery.
4. Smoke 200 rows end-to-end through render -> inject -> twins -> audit, and
   read the acceptance table above.
5. Only then decide corpus size.

Do not skip 4. Every finding in this document came from smoking 200 rows.
