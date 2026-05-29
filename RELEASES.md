# IRT Predictor — Model Releases

Trained on **11,270 questions** from MMLU + BoolQ + TriviaQA, benchmarked against **12 open-source language models** (TinyLlama 1.1B → Llama 3.1 8B).

Embeddings: `BAAI/bge-small-en-v1.5` (384-dim).
Predictor: XGBoost regressors, 5-fold CV evaluation.

---

## v1.0 — 4PL Model (`20260525_2339_12000q_12m_4pl_mmlu-boolq-triviaqa`)

**Active release. Used by the PoCW oracle service.**

The full 4-Parameter Logistic IRT model. Each question gets individual `a`, `b`, `c`, `d` estimates from py-irt, then XGBoost is trained to predict those parameters from text features + embeddings.

```
P(correct | θ) = c + (d − c) / (1 + exp(−a · (θ − b)))
```

### Prediction metrics (5-fold CV, n=11,270)

| Parameter | Meaning | RMSE | R² | Within ±20% margin |
|---|---|---|---|---|
| `a` — discrimination | How well the item separates ability levels | 0.324 | 0.063 | **90.7%** |
| `b` — difficulty | IRT logit difficulty (range ≈ −3 to +3) | 2.708 | 0.086 | 39.6% |
| `c` — guessing | Lower asymptote (type-based rule) | 0.008 | 0.998 | **99.9%** |
| `d` — upper asymptote | Max P(correct) for high-ability student | 0.205 | 0.096 | 27.9% |
| `p_correct` | Empirical difficulty [0=hard, 1=easy] | 0.264 | 0.229 | 52.8% |
| `item_discrimination` | Point-biserial correlation | 0.368 | 0.115 | 62.9% |

### What these numbers mean

- **`a` (discrimination):** 90.7% of predictions fall within ±0.49 of the true value. R²=0.06 means the model captures very little linear variance — but for the PoCW oracle, `a` is used only to scale the IRT update, not to rank items. Even weak discrimination signal adds value.
- **`b` (difficulty):** R²=0.086, RMSE=2.71 — the model is essentially a poor predictor of `b` on a logit scale. This is caused by Bayesian shrinkage in py-irt with only 12 examiners and a 4-parameter model. The PoCW oracle compensates with a weighted blend: `b_used = 0.85 × b_llm + 0.15 × b_pred`.
- **`c` (guessing):** Near-perfect prediction (R²=0.998) because `c` is type-determined: TF=0.5, MCQ=0.25, open=0.0. The XGBoost model learns this rule trivially.
- **`d` (upper asymptote):** Similar problem to `b` — Bayesian shrinkage compresses `d` values. R²=0.10 means limited predictive power, but clamping to [0.75, 1.0] in the oracle limits the damage.

### Model accuracy by LLM examiner

| Model | Overall | Open | MC | T/F |
|---|---|---|---|---|
| tinyllama:1.1b | 34.8% | 27.8% | 23.5% | 57.1% |
| qwen2.5:1.5b | 53.9% | 46.3% | 55.2% | 61.6% |
| llama3.2:1b | 45.0% | 51.7% | 24.3% | 62.2% |
| smollm2:1.7b | 50.5% | 56.0% | 35.8% | 61.7% |
| gemma2:2b | 58.5% | 57.9% | 53.9% | 64.9% |
| phi3.5 | 58.3% | 54.7% | 60.0% | 60.7% |
| qwen2.5:3b | 55.6% | 53.6% | 61.7% | 50.5% |
| llama3.2:3b | 58.7% | 64.0% | 49.2% | 63.7% |
| mistral:7b | 63.9% | 76.7% | 52.2% | 62.4% |
| qwen2.5:7b | 65.0% | 66.6% | 69.3% | 57.8% |
| llama3.1:8b | 63.5% | 80.1% | 44.5% | 66.6% |
| phi4-mini:3.8b | 59.4% | 55.1% | 60.2% | 63.7% |
| **Average** | **55.6%** | **57.5%** | **49.1%** | **61.1%** |

---

## v0.1 — 2PL Model (`20260526_1726_12000q_12m_2pl_mmlu-boolq-triviaqa`)

**Stable alternative with better `b` and `a` scale precision.**

The 2-Parameter Logistic model fixes `c = type_based` and `d = 1.0`, reducing IRT to:

```
P(correct | θ) = c + (1 − c) / (1 + exp(−a · (θ − b)))
```

With fewer free parameters, py-irt converges more reliably and the `b` estimates are far less affected by Bayesian shrinkage.

### Prediction metrics (5-fold CV, n=11,270)

| Parameter | Meaning | RMSE | R² | Within ±20% margin |
|---|---|---|---|---|
| `a` — discrimination | IRT discrimination | 0.061 | −0.088 | **88.5%** |
| `b` — difficulty | IRT logit difficulty (range ≈ −1 to +1) | 0.054 | 0.032 | **86.1%** |
| `c` — guessing | Fixed type-based rule | 0.008 | 0.998 | **99.9%** |
| `d` — upper asymptote | Fixed = 1.0 (2PL assumption) | 0.000 | 1.000 | **100%** |
| `p_correct` | Empirical difficulty | 0.264 | 0.229 | 52.8% |
| `item_discrimination` | Point-biserial correlation | 0.368 | 0.115 | 62.9% |

### Comparison: 4PL vs 2PL

| Parameter | 4PL RMSE | 4PL R² | 4PL ±20% | 2PL RMSE | 2PL R² | 2PL ±20% |
|---|---|---|---|---|---|---|
| `a` | 0.324 | 0.063 | 90.7% | 0.061 | −0.09 | 88.5% |
| `b` | 2.708 | 0.086 | 39.6% | 0.054 | 0.032 | **86.1%** |
| `c` | 0.008 | 0.998 | 99.9% | 0.008 | 0.998 | 99.9% |
| `d` | 0.205 | 0.096 | 27.9% | 0.000 | 1.000 | 100% |
| `p_correct` | 0.264 | 0.229 | 52.8% | 0.264 | 0.229 | 52.8% |

### Interpreting the comparison

**2PL `b` is dramatically more precise (RMSE 0.054 vs 2.708)** — but note the scales differ. 4PL `b` is fit on a logit scale spanning roughly [−5, +5], while 2PL `b` is fit on a compressed scale of roughly [−1, +1] due to fewer parameters and less shrinkage. The ±20% margin threshold reflects this: 4PL's threshold is ±1.6 logits, 2PL's is ±0.08 logits.

**In absolute terms,** 86.1% of 2PL `b` predictions fall within ±0.08 of their IRT estimate. For `b` usage in PoCW (selecting question difficulty), this is much more useful than 4PL's ±1.6 tolerance.

**`a` R² is negative for 2PL (−0.09)** — the model predicts `a` worse than predicting the mean. However, the RMSE is tiny (0.061) because 2PL `a` values are tightly clustered near 1.0. For practical use, `a` from both models is roughly "the model learned the mean."

**`d` is trivially perfect in 2PL** because it is fixed at 1.0 by assumption — not a prediction.

### When to use 2PL vs 4PL

| Use case | Recommendation |
|---|---|
| PoCW oracle (current) | **4PL** — captures upper asymptote, oracle compensates for weak `b` with LLM blend |
| Research: rank items by difficulty | **2PL** — far better `b` precision on its own scale |
| Few examiners (<8 models) | **2PL** — 4PL will not converge reliably |
| Full IRT fidelity (theta estimation, simulation) | **4PL** with caution on `b` |

---

## Retrain

To reproduce either release:

```bash
# 4PL (v1.0)
python3 train_irt_predictor.py --irt-model 4pl --questions 12000

# 2PL (v0.1)
python3 train_irt_predictor.py --irt-model 2pl --questions 12000
```

Requires ~11–14 hours on Apple Silicon M4 Max with all 12 Ollama models downloaded.
See the main [README](README.md) for setup instructions.
