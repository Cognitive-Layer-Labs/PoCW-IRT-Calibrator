# Local Dataset Notes

Place your manually downloaded training file here if HF gated access is unavailable.

Recommended filename:

- `cross_difficulty_train.csv`

Minimum required columns:

- `question`
- `irt_difficulty`

Optional columns (ignored by training unless mapped explicitly):

- `category`
- `domain` / `config` (used for domain-specific models)
- `bloom_level` (numeric 1-6 or text labels; optional future signal)

Run training from repo root:

```bash
python train_pipeline.py --data-path data/cross_difficulty_train.csv
```

If column names differ:

```bash
python train_pipeline.py \
  --data-path data/cross_difficulty_train.csv \
  --question-column your_question_col \
  --target-column your_difficulty_col
```
