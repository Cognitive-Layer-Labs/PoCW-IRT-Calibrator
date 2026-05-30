# PoCW-IRT-Calibrator

**Model releases and metrics:** [RELEASES.md](RELEASES.md) — 2PL vs 4PL comparison with RMSE / R² / within-margin rates for all parameters.

---

Automatically calibrates **IRT** (Item Response Theory) parameters for multiple-choice, true/false, and open-ended questions, using small LLM models as "synthetic examiners". The result is a set of XGBoost regressors that predict IRT parameters and empirical statistics directly from question text.

---

## Table of Contents

- [How it works](#how-it-works)
- [Dataset](#dataset)
- [Ollama models](#ollama-models)
- [IRT — supported models](#irt--supported-models)
- [Training process](#training-process)
- [Current performance](#current-performance)
- [Known limitations and improvements](#known-limitations-and-improvements)
- [Input / Output](#input--output)
- [Setup](#setup)
- [Running](#running)
- [CLI — full options](#cli--full-options)
- [Generated artifacts](#generated-artifacts)

---

## How it works

```
Questions (MMLU + BoolQ + TriviaQA)
           │
           ▼
┌──────────────────────┐
│  Stage 1             │  Download and shuffle the 3 sources from HuggingFace
│  Load dataset        │  → 11,270 questions (MC + TF + open-ended)
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  Stage 2             │  12 Ollama models answer every question
│  Response matrix     │  async, greedy (temp=0, top_k=1)
│                      │  Open-ended: validated by qwen2.5:14b (external model)
│                      │  → binary matrix (12 models × 11,270 questions)
│                      │  Per-model checkpointing — can be stopped and resumed
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  Stage 3             │  py-irt (Pyro/PyTorch) fits IRT on the response matrix
│  Fit IRT             │  Supports 1PL / 2PL / 4PL
│                      │  Automatic fallback: MLE scipy (2PL per item)
│                      │  → irt_params.json (a, b, [c, d] per question)
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  Stage 4             │  BAAI/bge-small-en-v1.5 (384 dim, runs on MPS)
│  Embed questions     │  text = "[subject] question A) ... B) ..."
│                      │  → embeddings.npy (11,270 × 384)
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  Stage 5             │  XGBoost per parameter: embedding (384d) +
│  Train regressors    │  10 text features → a, b, c, d (IRT)
│                      │                  → p_correct, item_discrimination (empirical)
│                      │  Evaluation: 5-fold CV (OOF); final model on full dataset
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  Stage 6 + 7         │  irt_predictor.py — standalone API
│  Export + Plots      │  training_dataset.parquet / .csv
│                      │  plots/ — accuracy, IRT distributions, CV metrics
└──────────────────────┘
```

---

## Dataset

Three sources downloaded automatically from HuggingFace on first run (**no API key required**). Local cache in `~/.cache/huggingface/`.

| Source | HuggingFace ID | Type | Effective rows | Size |
|---|---|---|---|---|
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | `cais/mmlu` | MC 4 choices | 4,000 | ~50 MB |
| [BoolQ](https://huggingface.co/datasets/google/boolq) | `google/boolq` | True / False | 3,270 (cap) | ~5 MB |
| [TriviaQA](https://huggingface.co/datasets/trivia_qa) | `trivia_qa rc.nocontext` | Open-ended | 4,000 | ~80 MB |

`--questions N` distributes equally across sources (N/3 each). BoolQ is capped at 3,270:

```
--questions 900    →  mmlu=300,  boolq=300,  triviaqa=300   →  900 effective
--questions 12000  →  mmlu=4000, boolq=3270, triviaqa=4000  →  11,270 effective
```

### Prompt formats

```
# MC (MMLU)
Q: What is the chemical formula for water?
A) CO2   B) H2O   C) NaCl   D) O2
Answer with only the letter (A/B/C/D):

# True/False (BoolQ)
Q: Is the Eiffel Tower located in Paris?
A) True   B) False
Answer with only the letter (A/B):

# Open-ended (TriviaQA)
Q: What is the capital of Australia?
Answer briefly in 1-2 sentences:
```

---

## Ollama models

Models act as **"synthetic examiners"** — their collective responses allow estimating difficulty and discrimination via IRT. Architectural diversity is essential: models from different families make different mistakes, which stabilises IRT estimates.

### Tier 1 — Must-have (~5.6 GB total)

| Model | Family | Size | Accuracy (current run) |
|---|---|---|---|
| [tinyllama:1.1b](https://ollama.com/library/tinyllama) | LLaMA | 0.6 GB | 34.8% |
| [qwen2.5:1.5b](https://ollama.com/library/qwen2.5) | Qwen | 1.0 GB | 53.9% |
| [llama3.2:1b](https://ollama.com/library/llama3.2) | LLaMA | 1.3 GB | 45.0% |
| [smollm2:1.7b](https://ollama.com/library/smollm2) | SmolLM | 1.0 GB | 50.5% |
| [gemma2:2b](https://ollama.com/library/gemma2) | Gemma | 1.6 GB | 58.5% |

### Tier 2 — Recommended (~6.2 GB total)

| Model | Family | Size | Accuracy (current run) |
|---|---|---|---|
| [phi3.5](https://ollama.com/library/phi3.5) | Phi | 2.2 GB | 58.3% |
| [qwen2.5:3b](https://ollama.com/library/qwen2.5) | Qwen | 2.0 GB | 55.6% |
| [llama3.2:3b](https://ollama.com/library/llama3.2) | LLaMA | 2.0 GB | 58.7% |

### Tier 3 — Optional (~16.2 GB total)

| Model | Family | Size | Accuracy (current run) |
|---|---|---|---|
| [mistral:7b](https://ollama.com/library/mistral) | Mistral | 4.1 GB | 63.9% |
| [qwen2.5:7b](https://ollama.com/library/qwen2.5) | Qwen | 4.7 GB | 65.0% |
| [llama3.1:8b](https://ollama.com/library/llama3.1) | LLaMA | 4.9 GB | 63.5% |
| [phi4-mini:3.8b](https://ollama.com/library/phi4-mini) | Phi | 2.5 GB | 59.4% |

> **Validator note:** `qwen2.5:14b` (~8.2 GB) is used **exclusively as a validator** for open-ended questions and does **not** enter the response matrix. It can be replaced in `models_config.json` with `gemma2:9b` or `phi4:14b`.

> **Minimum recommended for stable IRT:** at least 8 models from Tier 1+2+3. Below 5 models, the `a` parameter becomes unstable. Below 10 models with 4PL, the `b` parameter suffers from severe Bayesian shrinkage (see [Limitations](#known-limitations-and-improvements)).

> **Reasoning models** (deepseek-r1, qwq, :thinking): generate long `<think>` blocks that frequently exceed the token limit and cause parsing errors (~23%). Not recommended as examiners.

---

## IRT — supported models

| Model | Params per item | Equation | When to use |
|---|---|---|---|
| **1PL** (Rasch) | 1 (`b`) | `P = 1/(1+exp(-(θ-b)))` | Few examiners (<5), maximum consistency |
| **2PL** | 2 (`a`, `b`) | `P = 1/(1+exp(-a(θ-b)))` | Recommended with 8–15 examiners |
| **4PL** | 4 (`a`, `b`, `c`, `d`) | `P = c + (d-c)/(1+exp(-a(θ-b)))` | Requires >15 examiners for stability |

Configurable with `--irt-model 1pl|2pl|4pl`. Current default: `4pl` (can be changed in `CFG["irt_model"]`).

**Empirical statistics** (added automatically, independent of IRT model):

| Statistic | Formula | Advantage over IRT |
|---|---|---|
| `p_correct` | `mean(column from response matrix)` | No Bayesian shrinkage; more predictable from text |
| `item_discrimination` | Point-biserial correlation item vs. rest-score | More stable than `a` with few examiners |

---

## Training process

### Stage 2: Response matrix

Each model answers greedily (`temperature=0`, `top_k=1`) via the [Ollama API](https://github.com/ollama/ollama), async with 12 parallel requests.

- **MC / TF:** first valid letter from the response, compared with the correct answer → `1/0`
- **Open-ended:** model answers freely (max 80 tokens), `qwen2.5:14b` validates with `yes/no`; exact aliases (e.g. "canberra" in the response) short-circuit the LLM call

Per-model checkpointing in `irt_checkpoints/{run_id}/responses_{model}.json` — the process can be stopped and resumed with `--resume`.

### Stage 3: IRT fitting

[py-irt](https://github.com/nd-ball/py-irt) (Pyro + PyTorch) fits the IRT model on the full matrix simultaneously (NUTS or SVI, 3000 epochs). If py-irt fails (timeout, convergence error), automatic fallback to MLE scipy (L-BFGS-B per item, 2PL).

### Stage 5: XGBoost with 5-fold CV

Input: embeddings (384d) + 10 text features (question length, word count, total answer characters, presence of negations and interrogative words).

For each parameter (`a`, `b`, `c`, `d`, `p_correct`, `item_discrimination`):
1. **5-fold CV** for evaluation (OOF predictions → RMSE, R², % within 20% margin)
2. **Final model trained on the full dataset** — this is saved for inference

---

## Current performance

Run: `20260525_2339` — 11,270 questions × 12 models, 4PL IRT, bge-small-en-v1.5 embeddings.

| Parameter | Meaning | CV RMSE | CV R² | Within 20% margin |
|---|---|---|---|---|
| `a` (discrimination) | IRT discriminability | 0.324 | 0.063 | **90.7%** ✓ |
| `b` (difficulty) | IRT difficulty (logit) | 2.708 | 0.086 | 39.6% |
| `c` (guessing) | Lower asymptote | 0.008 | 0.998 | **99.9%** ✓ |
| `d` (upper asymptote) | Upper asymptote | 0.205 | 0.096 | 27.9% |
| `p_correct` | Empirical difficulty [0,1] | 0.264 | 0.229 | 52.8% |
| `item_discrimination` | Empirical discrimination [-1,1] | 0.368 | 0.116 | 62.9% |

**Note on 20% margin:** the threshold is `0.20 × (max - min)` per parameter (e.g. `b` with range=8 → threshold ±1.6; `p_correct` with range=1 → threshold ±0.20).

---

## Known limitations and improvements

### Bayesian shrinkage on `b` and `d`

With 12 small models (35–65% accuracy) and the 4PL model, py-irt applies a strong Bayesian prior that **compresses** all parameters toward the mean. Visible effect: `b` has R²=0.086 (practically unpredictable from text), and `d` has R²=0.095.

**Solutions in order of cost:**

| Solution | Cost | Estimated gain on `b` |
|---|---|---|
| Switch to 2PL (`--irt-model 2pl`) | Re-run stage 3 (~45 min) | Moderate — fewer parameters, less shrinkage |
| Larger embeddings (bge-large, 1024d) | Re-run stage 4 (~15 min) | ~20–30% RMSE reduction |
| Add 14B+ models (phi4:14b, qwen2.5:32b) | Download + full re-inference (~12h) | Large — more stable IRT estimates |
| All combined | ~14h total | Potentially 60–75% within 20% for `b` |

### `p_correct` vs. IRT `b` as difficulty proxy

`p_correct` (column mean from the response matrix) is **more predictable** from text (R²=0.229 vs. 0.086) because it does not suffer from Bayesian shrinkage. If the goal is to estimate relative question difficulty, `p_correct` is recommended as the primary target.

### deepseek-r1 and reasoning models

deepseek-r1:7b generates `<think>...</think>` blocks that frequently exceed 1024 tokens (~23% parsing errors) and cannot be used as a reliable examiner. Exclude them via `reasoning_model_patterns` in CFG or through `models_config.json`.

---

## Input / Output

### Inference with standalone predictor

Artifacts in `irt_runs/{run_id}/` are portable. Copy the directory and import:

```python
import sys
sys.path.insert(0, "irt_runs/20260525_2339_12000q_12m_4pl_mmlu-boolq-triviaqa")
from irt_predictor import IRTPredictor

p = IRTPredictor()

result = p.predict(
    question="Which of the following best describes osmosis?",
    choices=[
        "Movement of solutes from high to low concentration",
        "Movement of water across a semipermeable membrane",
        "Active transport requiring ATP",
        "Diffusion of gases in the lungs",
    ]
)
```

### Output

```python
{
    "a":                   1.43,    # IRT discrimination
    "b":                   0.21,    # IRT difficulty (logit, scale ~[-3, 3])
    "c":                   0.25,    # guessing (lower asymptote)
    "d":                   0.97,    # upper asymptote
    "p_correct":           0.61,    # empirical difficulty [0=hard, 1=easy]
    "item_discrimination": 0.48,    # point-biserial correlation [-1, 1]
    "difficulty":          "medium",  # semantic label for b
    "discrimination":      "good discriminability",
}
```

### Semantic labels

| `b` | Difficulty |
|---|---|
| `b < -1.5` | Very easy |
| `-1.5 ≤ b < -0.5` | Easy |
| `-0.5 ≤ b < 0.5` | Medium |
| `0.5 ≤ b < 1.5` | Hard |
| `b ≥ 1.5` | Very hard |

| `a` | Discrimination |
|---|---|
| `a < 0.5` | Poor discriminability |
| `0.5 ≤ a < 1.0` | Moderate discriminability |
| `1.0 ≤ a < 2.0` | Good discriminability |
| `a ≥ 2.0` | Excellent discriminability |

---

## Setup

```bash
git clone <repo-url>
cd PoCW-IRT-Calibrator

# py-irt requires Python <3.12
/opt/homebrew/bin/python3.10 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# macOS: OpenMP for XGBoost
brew install libomp

# Start Ollama (must be running in the background)
ollama serve
```

### Download models

```bash
# Check what is already installed
python3 download_models.py --check

# Tier 1 — minimum viable (~5.6 GB)
python3 download_models.py --fast

# Tier 1+2 — recommended (~11.8 GB)
python3 download_models.py --tier 2

# Tier 1+2+3 — full (~28 GB)
python3 download_models.py --tier 3

# Functional test after download
python3 download_models.py --verify
```

---

## Running

```bash
source .venv/bin/activate

# Quick test with 5 Tier 1 models (~3-4 hours)
python3 train_irt_predictor.py --questions 300 --irt-model 2pl --fast

# Standard run — 12 models, ~11,270 questions (~11-14 hours)
python3 train_irt_predictor.py --irt-model 4pl

# Resume from where it stopped (per-model checkpointing)
python3 train_irt_predictor.py --resume

# Re-run only stage 5 (regressors) with existing data
python3 train_irt_predictor.py --resume --skip-stage 1 --skip-stage 2 --skip-stage 3 --skip-stage 4

# Re-run stages 4+5 (re-embed + regressors) with existing response matrix
python3 train_irt_predictor.py --resume --skip-stage 1 --skip-stage 2 --skip-stage 3
```

### Estimated durations (M4 Max, 36 GB RAM)

| Stage | ~11,270 questions × 12 models | `--fast` (5 models) |
|---|---|---|
| Dataset download (first time) | ~5 min | ~5 min |
| Response matrix MC/TF | ~7–9 hours | ~2–3 hours |
| Open-ended validation (qwen2.5:14b) | ~2–3 hours | ~1 hour |
| IRT fitting (py-irt, 3000 epochs) | ~15 min | ~10 min |
| Embedding + XGBoost (5-fold CV) | ~10 min | ~5 min |
| **Total** | **~11–14 hours** | **~3–4 hours** |

---

## CLI — full options

```
python3 train_irt_predictor.py [options]

  --questions N         Total number of questions (default: 12000; effective: 11,270)
  --irt-model MODEL     IRT model: 1pl, 2pl, 4pl (default: 4pl)
  --fast                Limit to 5 Tier 1 models (tinyllama, qwen2.5:1.5b,
                        llama3.2:1b, smollm2:1.7b, gemma2:2b)
  --resume              Resume from the most recent run (matching checkpoints)
  --skip-stage N        Skip stage N (1-5); can be repeated
                        e.g.: --skip-stage 2 --skip-stage 3
```

`--resume` automatically searches `irt_checkpoints/` for a run with the same number of models, IRT model, and datasets, and resumes from existing checkpoints.

---

## Generated artifacts

```
irt_runs/{timestamp}_{N}q_{M}m_{irt}_{datasets}/
├── response_matrix.npy       ← (M models × N questions) binary
├── embeddings.npy            ← (N, 384) float32
├── irt/
│   ├── irt_params.json       ← arrays a, b, [c, d] per question
│   └── responses.jsonlines   ← raw input for py-irt
├── xgb_regressors.pkl        ← 6 serialised XGBoost regressors
├── irt_predictor.py          ← standalone predictor (portable)
├── training_dataset.parquet  ← N questions + all IRT + empirical parameters
├── training_dataset.csv      ← same dataset in CSV format
├── metrics.json              ← CV RMSE, R², within_20pct_margin per param
├── training_config.json      ← full config + timestamp + model accuracies
├── training.log              ← full run log
└── plots/
    ├── accuracy_per_model.png
    ├── irt_distributions.png
    ├── xgb_metrics.png
    └── response_matrix_heatmap.png

irt_checkpoints/{timestamp}_{N}q_{M}m_{irt}_{datasets}/
├── questions.json                    ← questions (avoids re-downloading)
└── responses_{model}.json            ← per-model responses (12 files)
```

The run name (`{timestamp}_{N}q_{M}m_{irt}_{datasets}`) uniquely identifies the configuration and allows `--resume` to find the correct checkpoints.
