# eo-mt-v5b training

Bidirectional EN↔EO MarianMT (~61M params), initialized from `runs/mt/eneo_v5/final`.

## Training mix — 4,149,324 rows

| corpus | rows |
|---|---:|
| ccmatrix_filtered_dedup | 1,944,726 |
| xlent_dedup | 1,204,545 |
| tatoeba_train_dedup | 426,558 |
| wikimatrix_dedup | 294,142 |
| opus100_train_dedup | 144,549 |
| bible_uedin_dedup | 61,274 |
| opensubtitles_v2024_dedup | 40,196 |
| wikimedia_dedup | 21,180 |
| ted2020_dedup | 10,656 |
| opusbooks_train_dedup | 1,498 |

All files live in `mt/data/parallel/`. Each row is `{en, eo, src}`.

## Validation

- `opus100_validation.jsonl` (400 sampled, en→eo)
- `flores_devtest.jsonl` (400 sampled, en→eo)

## Reference scores (en→eo, our sacrebleu pipeline)

- v5b: BLEU 24.52 / chrF 58.02 / chrF++ 55.28
- NLLB-200-distilled-600M: same numbers via same pipeline (reference baseline)
- NLLB-3.3B / 54B (published): chrF++ 60.8 / 61.4

## Scripts

- `scripts/train.py` — bidirectional seq2seq training
- `scripts/dataset.py` — `ParallelDataset`, `Seq2SeqCollator`
- `scripts/download_*.py` — corpus pulls
- `scripts/dedup_parallel.py` — dedup step
- `scripts/eval_bleu.py`, `scripts/eval_nllb.py` — eval drivers

Training log: `mt/logs/eneo_v5b.log`.
