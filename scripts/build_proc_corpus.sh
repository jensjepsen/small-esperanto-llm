#!/usr/bin/env bash
# Build the procedural tool-dialogue corpus, in resumable stages.
#
#   scripts/build_proc_corpus.sh 2000        # first or next stage
#   scripts/build_proc_corpus.sh 35892       # the whole catalogue
#
# --n is a count of DIALOGUES PLANNED, not rows kept: ~76% survive the gates
# and the judge, so 35,892 planned is ~27,300 rows. It is also the total, not
# an increment -- every run after the first resumes into the same directory
# and generates only the difference.
#
# STAGING IS NOT FREE AT THE SAME --n. The cap tops up toward --n against the
# rows that survived, so re-running an already-satisfied --n still generates
# replacements for judged-out rows. Raise --n to add work; re-run the same --n
# only if you want the shortfall refilled.
#
# The seed fixes which tools are drawn: `frozen[:n/dialogues_per_tool]` over a
# seeded shuffle, so a larger --n keeps every earlier tool and appends. CHANGE
# THE SEED AND A RESUME BECOMES INCOHERENT -- it would draw a different
# prefix and strand the rows already on disk.
set -euo pipefail

N="${1:?usage: build_proc_corpus.sh <total dialogues planned>}"
TOOLS="${TOOLS:-data/tool_calls/tools_v6.jsonl}"
OUT="${OUT:-data/tool_calls/proc_v2}"
SEED="${SEED:-0}"
DPT="${DPT:-4}"

# tools_v6 = tools_v2_selr (audited, repaired, selectors resolved; all 4,984
# byte-identical) + 3,989 invented siblings that make families chainable.
# Pointing this at a raw catalogue rebuilds every defect the repair removes,
# silently -- the generator refuses to run without --tools-from for that
# reason, but it cannot tell a repaired catalogue from an unrepaired one.
[ -f "$TOOLS" ] || { echo "no catalogue at $TOOLS" >&2; exit 1; }

RESUME=""
[ -f "$OUT/translated.jsonl" ] && RESUME="--resume"
echo "building $OUT to n=$N (seed $SEED, $DPT/tool) ${RESUME:-fresh}"

uv run --no-project --with aiohttp --with langdetect \
  python scripts/gen_tool_dialogues_proc.py \
    --tools-from "$TOOLS" --out "$OUT" \
    --n "$N" --dialogues-per-tool "$DPT" --seed "$SEED" $RESUME \
  2>&1 | tee -a "scratch/$(basename "$OUT")_build.log"

python3 scripts/check_proc_build.py "$OUT"
