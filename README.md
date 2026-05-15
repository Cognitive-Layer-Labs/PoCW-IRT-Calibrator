# IRT Difficulty Predictor (Question-Only, 2PL-Compatible API)

This project trains a local ML model that predicts IRT item difficulty (`b`) from question text only.
The inference API is 2PL-compatible by returning both parameters:

- `b`: predicted difficulty
- `a`: fixed to `1.0` for now

That means the model behaves like 1PL/Rasch today, while keeping a stable interface for future 2PL calibration.

## Required Libraries

The project uses:

- `datasets`
- `pandas`, `numpy`
- `scikit-learn`
- `textstat`
- `nltk`
- `spacy`
- `matplotlib`, `seaborn`
- `joblib`
- `transformers` (included per requested environment/tooling)
- `tqdm`

All are listed in `requirements.txt`.

## Local Setup (macOS / Python 3.11+)

Use any available Python interpreter that is 3.11 or newer.

```bash
python3 -V
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python3` is not found, install Python first:

```bash
brew install python
```

If you specifically want Python 3.11:

```bash
brew install python@3.11
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download NLP assets once:

```bash
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger stopwords wordnet omw-1.4
python -m spacy download en_core_web_sm
```

## Training

The CLI is intentionally minimal.

Core command (recommended):

```bash
python train_pipeline.py --data-path data/cross_difficulty_train.csv
```

Optional:

- `--verbose` for detailed logs

Examples:

```bash
python train_pipeline.py --data-path data/cross_difficulty_train.csv --verbose
```

The trainer now runs in **max-power mode by default** (largest candidate set).
No profile flag is required.

What happens by default:

- The trainer tests multiple candidate pipelines automatically (linguistic, embedding, hybrid).
- In `max` profile it also evaluates strong transformer embeddings (`all-mpnet-base-v2`, `bge-base-en-v1.5`) if dependencies are installed.
- It winsorizes extreme target values before training (outlier-robust target).
- It evaluates each with cross-validation.
- It keeps and saves the **best** model automatically based on held-out performance:
  highest **domain-aware** `acc_within_2.5` (fallback `acc_within_2`), then lowest domain-aware `test_mae`.
- It applies isotonic calibration when calibration helps on OOF predictions.
- It also fits per-domain models (when enough samples exist) and enables per-domain calibrators only if they improve held-out objective, with global fallback at inference.
- It computes a weighted blend of top models for comparison (`blend_result` in metrics).
- Full logs go to `reports/train.log`.

Important: to enable transformer candidates you must install PyTorch:

```bash
pip install -r requirements.txt
```

Expected columns in local data:

- `question`
- `irt_difficulty`

If `--data-path` is missing, the script attempts HF fallback. If HF access is gated,
set `HF_TOKEN` or keep using local file mode.

## What the Pipeline Does

1. Loads either:
   - Hugging Face dataset `BatsResearch/Cross-Difficulty` (train split), or
   - a local file/folder from `--data-path`.
2. Uses only `question` text (no answer choices) and `irt_difficulty`.
3. Winsorizes extreme target values to reduce outlier sensitivity.
4. Trains multiple candidate pipelines (linguistic, embedding, hybrid) with tuned regressors (`RF`, `ExtraTrees`, `Ridge`, `HGB`, `MLP`, optional `XGBoost`).
5. Selects the best candidate automatically using domain-aware held-out tolerance + MAE objective.
6. Fits isotonic calibration and evaluates calibrated predictions.
7. Trains per-domain models using the best feature space and keeps global fallback.
8. Evaluates on test set (MSE, MAE, R2), including domain-aware fallback metrics.
9. Saves model bundle + feature names + metrics + diagnostic plots.

## Output Artifacts

After training, you should get:

- `artifacts/difficulty_predictor.pkl`
- `artifacts/feature_names.json`
- `reports/metrics.json`
- `reports/model_leaderboard.csv`
- `reports/test_predictions.csv`
- `reports/figures/cv_results.png`
- `reports/figures/pred_vs_actual.png`
- `reports/figures/feature_importance.png`
- `reports/figures/learning_curve.png`
- `reports/figures/residuals.png`

## Plot Interpretation

- `cv_results.png`: compares CV train/test MSE across sampled hyperparameter configurations.
  - Useful for seeing if some configurations overfit (large train-test gap).
- `pred_vs_actual.png`: predicted vs real `b` values on test set.
  - Closer to the diagonal line means better calibration quality.
- `feature_importance.png`: top feature importances/weights of the selected best model.
  - Shows which features contributed most to `b` prediction.
- `learning_curve.png`: training and validation MSE vs training set size.
  - Indicates whether more data may improve generalization.

For exact numbers and top ranked features, check `reports/metrics.json`.

`reports/test_predictions.csv` includes both:
- `predicted_b_global` (single global model path)
- `predicted_b_domain_aware` (domain override + fallback path)

### Meaning of `acc_within_X`

`acc_within_X` = percent of questions where:

`|predicted_b - true_b| <= X`

Examples:

- `acc_within_0.5 = 0.30` means 30% of predictions are within +/-0.5 logits.
- `acc_within_2.0 = 0.80` means 80% are within +/-2 logits.

## Inference Usage

```python
from src.predictor import load_predictor

predictor = load_predictor("artifacts/difficulty_predictor.pkl")
params = predictor.predict_item_params("What is the capital of France?")
print(params)  # {'b': 0.23, 'a': 1.0}

# Optional: provide domain if available (falls back to global if missing)
params_domain = predictor.predict_item_params("What is the capital of France?", domain="arc")
print(params_domain)

# Optional: provide Bloom level (1-6 or text like "analyze")
params_bloom = predictor.predict_item_params(
    "What is the capital of France?",
    domain="arc",
    bloom_level=2
)
print(params_bloom)
```

CLI example:

```bash
python example_inference.py --question "Explain how photosynthesis converts light into chemical energy."
```

## Fine-tune a Transformer Regressor

Use this when you want a dedicated regression head on top of a transformer:

```bash
python finetune_transformer_regressor.py \
  --data-path data/cross_difficulty_train.csv \
  --model-name sentence-transformers/all-mpnet-base-v2 \
  --output-dir artifacts/finetuned-transformer-regressor \
  --epochs 3 \
  --batch-size 16
```

This saves:

- fine-tuned checkpoint in `artifacts/finetuned-transformer-regressor`
- encoder-only export in `artifacts/finetuned-transformer-regressor/encoder`
- metrics in `artifacts/finetuned-transformer-regressor/finetune_metrics.json`

You can then point embedding candidates to the exported encoder path by replacing
`embedding_model` in candidate configs if desired.

