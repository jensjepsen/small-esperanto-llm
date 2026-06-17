# SFT Data

Files under `data/sft/` and the HF Hub datasets they map to.

## Files

| File | HF dataset | Convs | Format | Teaches |
|---|---|---:|---|---|
| `sft_factoid.jsonl` | `jensjepsen/esperanto-sft-factoid` | ~370K | multi-turn QA | Wikidata factual recall |
| `sft_creative.jsonl` | `jensjepsen/esperanto-sft-creative` | ~22K | single-turn | Descriptive/creative prose |
| `gsm8k/train.jsonl` | `jensjepsen/esperanto-gsm8k` | ~7.4K | single-turn CoT | Word-problem reasoning |
| `arithmetic_cot/train.jsonl` | `jensjepsen/esperanto-arithmetic-cot` | ~60K | single-turn CoT | Digit-by-digit math |
| `sft_atomic_icl.jsonl` | `jensjepsen/esperanto-sft-atomic-icl` | ~12K | single-turn ICL | Format-following + commonsense pattern inference |
| `sft_atomic_qa.jsonl` | `jensjepsen/esperanto-sft-atomic-qa` | ~15K | multi-turn | Commonsense reasoning with context carry |

Current default `--sft-data` list in `scripts/train_sft.py` pulls all six from the Hub.

## Distribution caveat

Wikidata factoid is ~76% of conversations and ~75% of text. Smaller sources risk being drowned out. When rebalancing, either grow the minority sources (especially ATOMIC, creative) or downsample factoid before training.

## ATOMIC translation (`scripts/translate_atomic.py`)

Translates ATOMIC 2020 commonsense KG components (heads + tails) individually via Gemini, then reassembles Esperanto triples. Translation cache at `data/atomic_eo/atomic_dict.json` is reusable across runs.

**Head quality is asymmetric** — matters when choosing scale flags:

- **PersonX heads** (social events, ~21.5K unique): stay clean deep into the ranks. Rank 10K still has freq~43 per head and reads naturally ("PersonX loves PersonY anyway"). Safe to scale to 4K+ heads.
- **Object heads** (physical/concept, ~15.4K unique): degrade around rank 2–3K. Below rank 5K, typos appear ("champagn") and per-head frequency drops under 10. Keep ≤ 1–2K heads for quality.

Prefer the asymmetric flags over symmetric `--max-heads`:

```bash
uv run python scripts/translate_atomic.py \
    --personx-heads 4000 --object-heads 1000 \
    --max-tails-per-relation 3 --parallel 10
```

~119K Esperanto triples, ~$0.15 incremental API cost with the existing cache.

## Regeneration

ATOMIC ICL and QA generators read from `data/atomic_eo/atomic_eo.jsonl` and are cheap to rerun (seconds):

```bash
uv run python scripts/generate_atomic_icl.py --n 30000
uv run python scripts/generate_atomic_qa.py  --n 30000
```

Format and shot-count variety come from the generators, not the source data — regenerating with the same source but a higher `--n` mostly produces new combinations.
