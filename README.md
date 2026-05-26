# PoCW-IRT-Calibrator

Calibrează automat parametrii **IRT** (Item Response Theory) pentru întrebări de tip multiple-choice, true/false și open-ended, folosind răspunsurile unor modele LLM mici ca „examinatori sintetici". Rezultatul este un set de regresorii XGBoost care prezic parametrii IRT și statistici empirice direct din textul întrebării.

---

## Cuprins

- [Cum funcționează](#cum-funcționează)
- [Dataset](#dataset)
- [Modele Ollama](#modele-ollama)
- [IRT — modele suportate](#irt--modele-suportate)
- [Procesul de training](#procesul-de-training)
- [Performanță curentă](#performanță-curentă)
- [Limitări cunoscute și îmbunătățiri](#limitări-cunoscute-și-îmbunătățiri)
- [Input / Output](#input--output)
- [Setup](#setup)
- [Rulare](#rulare)
- [CLI — opțiuni complete](#cli--opțiuni-complete)
- [Artefacte generate](#artefacte-generate)

---

## Cum funcționează

```
Întrebări (MMLU + BoolQ + TriviaQA)
           │
           ▼
┌──────────────────────┐
│  Etapa 1             │  Descarcă și amestecă cele 3 surse de pe HuggingFace
│  Load dataset        │  → 11 270 întrebări (MC + TF + open-ended)
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  Etapa 2             │  12 modele Ollama răspund la fiecare întrebare
│  Response matrix     │  async, greedy (temp=0, top_k=1)
│                      │  Open-ended: validat de qwen2.5:14b (model extern)
│                      │  → matrice binară (12 modele × 11 270 întrebări)
│                      │  Checkpointing per model — se poate opri și relua
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  Etapa 3             │  py-irt (Pyro/PyTorch) fittează IRT pe response matrix
│  Fit IRT             │  Suportă 1PL / 2PL / 4PL
│                      │  Fallback automat: MLE scipy (2PL per item)
│                      │  → irt_params.json (a, b, [c, d] per întrebare)
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  Etapa 4             │  BAAI/bge-small-en-v1.5 (384 dim, rulat pe MPS)
│  Embed întrebări     │  text = "[subject] întrebare A) ... B) ..."
│                      │  → embeddings.npy (11 270, 384)
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  Etapa 5             │  XGBoost per parametru: embedding (384d) +
│  Train regresorii    │  10 features text → a, b, c, d (IRT)
│                      │            → p_correct, item_discrimination (empirice)
│                      │  Evaluare: 5-fold CV (OOF); model final pe tot setul
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  Etapa 6 + 7         │  irt_predictor.py — API standalone
│  Export + Plots      │  training_dataset.parquet / .csv
│                      │  plots/ — acuratețe, distribuții IRT, metrici CV
└──────────────────────┘
```

---

## Dataset

Trei surse descărcate automat de pe HuggingFace la prima rulare (**fără API key**). Cache local în `~/.cache/huggingface/`.

| Sursă | HuggingFace ID | Tip | Rows efective | Dimensiune |
|---|---|---|---|---|
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | `cais/mmlu` | MC 4 variante | 4 000 | ~50 MB |
| [BoolQ](https://huggingface.co/datasets/google/boolq) | `google/boolq` | True / False | 3 270 (cap) | ~5 MB |
| [TriviaQA](https://huggingface.co/datasets/trivia_qa) | `trivia_qa rc.nocontext` | Open-ended | 4 000 | ~80 MB |

`--questions N` distribuie egal între surse (N/3 fiecare). BoolQ are cap la 3 270:

```
--questions 900    →  mmlu=300,  boolq=300,  triviaqa=300   →  900 efective
--questions 12000  →  mmlu=4000, boolq=3270, triviaqa=4000  →  11 270 efective
```

### Formate de prompt

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

## Modele Ollama

Modelele joacă rolul de **„examinatori sintetici"** — răspunsurile lor colective permit estimarea dificultății și discriminabilității prin IRT. Diversitatea arhitecturală este esențială: modele din familii diferite fac greșeli diferite, ceea ce stabilizează estimările IRT.

### Tier 1 — Must-have (~5.6 GB total)

| Model | Familie | Dim | Acuratețe (run curent) |
|---|---|---|---|
| [tinyllama:1.1b](https://ollama.com/library/tinyllama) | LLaMA | 0.6 GB | 34.8% |
| [qwen2.5:1.5b](https://ollama.com/library/qwen2.5) | Qwen | 1.0 GB | 53.9% |
| [llama3.2:1b](https://ollama.com/library/llama3.2) | LLaMA | 1.3 GB | 45.0% |
| [smollm2:1.7b](https://ollama.com/library/smollm2) | SmolLM | 1.0 GB | 50.5% |
| [gemma2:2b](https://ollama.com/library/gemma2) | Gemma | 1.6 GB | 58.5% |

### Tier 2 — Recomandat (~6.2 GB total)

| Model | Familie | Dim | Acuratețe (run curent) |
|---|---|---|---|
| [phi3.5](https://ollama.com/library/phi3.5) | Phi | 2.2 GB | 58.3% |
| [qwen2.5:3b](https://ollama.com/library/qwen2.5) | Qwen | 2.0 GB | 55.6% |
| [llama3.2:3b](https://ollama.com/library/llama3.2) | LLaMA | 2.0 GB | 58.7% |

### Tier 3 — Opțional (~16.2 GB total)

| Model | Familie | Dim | Acuratețe (run curent) |
|---|---|---|---|
| [mistral:7b](https://ollama.com/library/mistral) | Mistral | 4.1 GB | 63.9% |
| [qwen2.5:7b](https://ollama.com/library/qwen2.5) | Qwen | 4.7 GB | 65.0% |
| [llama3.1:8b](https://ollama.com/library/llama3.1) | LLaMA | 4.9 GB | 63.5% |
| [phi4-mini:3.8b](https://ollama.com/library/phi4-mini) | Phi | 2.5 GB | 59.4% |

> **Notă validator:** `qwen2.5:14b` (~8.2 GB) este folosit **exclusiv ca validator** pentru open-ended și **nu** intră în matricea de răspunsuri. Poate fi înlocuit în `models_config.json` cu `gemma2:9b` sau `phi4:14b`.

> **Minimum recomandat pentru IRT stabil:** cel puțin 8 modele din Tier 1+2+3. Sub 5 modele, parametrul `a` devine instabil. Sub 10 modele cu 4PL, parametrul `b` suferă de shrinkage Bayesian sever (vezi [Limitări](#limitări-cunoscute-și-îmbunătățiri)).

> **Modele de tip reasoning** (deepseek-r1, qwq, :thinking): generate blocuri `<think>` lungi care depășesc frecvent limita de tokeni și cauzează erori de parsing (~23%). Nu sunt recomandate ca examinatori.

---

## IRT — modele suportate

| Model | Params per item | Ecuație | Când să folosești |
|---|---|---|---|
| **1PL** (Rasch) | 1 (`b`) | `P = 1/(1+exp(-(θ-b)))` | Puțini examinatori (<5), consistență maximă |
| **2PL** | 2 (`a`, `b`) | `P = 1/(1+exp(-a(θ-b)))` | Recomandat cu 8-15 examinatori |
| **4PL** | 4 (`a`, `b`, `c`, `d`) | `P = c + (d-c)/(1+exp(-a(θ-b)))` | Necesită >15 examinatori pentru stabilitate |

Configurabil cu `--irt-model 1pl|2pl|4pl`. Default curent: `4pl` (poate fi schimbat în `CFG["irt_model"]`).

**Statistici empirice** (adăugate automat, independent de modelul IRT):

| Statistică | Formulă | Avantaj față de IRT |
|---|---|---|
| `p_correct` | `mean(coloană din response matrix)` | Fără shrinkage Bayesian; mai predictibil din text |
| `item_discrimination` | Corelație punct-biserială item vs. scor-rest | Mai stabil decât `a` cu puțini examinatori |

---

## Procesul de training

### Etapa 2: Response matrix

Fiecare model răspunde greedy (`temperature=0`, `top_k=1`) via [Ollama API](https://github.com/ollama/ollama), async cu 12 cereri paralele.

- **MC / TF:** prima literă validă din răspuns, comparată cu răspunsul corect → `1/0`
- **Open-ended:** modelul răspunde liber (max 80 tokeni), `qwen2.5:14b` validează cu `yes/no`; alias-urile exacte (ex. "canberra" în răspuns) scurtcircuitează apelul LLM

Checkpointing per model în `irt_checkpoints/{run_id}/responses_{model}.json` — procesul poate fi oprit și reluat cu `--resume`.

### Etapa 3: IRT fitting

[py-irt](https://github.com/nd-ball/py-irt) (Pyro + PyTorch) fittează modelul IRT pe toată matricea simultan (NUTS sau SVI, 3000 epoch-uri). Dacă py-irt eșuează (timeout, eroare de convergență), fallback automat la MLE scipy (L-BFGS-B per item, 2PL).

### Etapa 5: XGBoost cu 5-fold CV

Input: embeddings (384d) + 10 features din text (lungime întrebare, număr cuvinte, total caractere răspunsuri, prezența negațiilor și a cuvintelor interogative).

Pentru fiecare parametru (`a`, `b`, `c`, `d`, `p_correct`, `item_discrimination`):
1. **5-fold CV** pentru evaluare (OOF predictions → RMSE, R², % în margine 20%)
2. **Model final antrenat pe tot setul** — acesta este salvat pentru inferență

---

## Performanță curentă

Run: `20260525_2339` — 11 270 întrebări × 12 modele, 4PL IRT, bge-small-en-v1.5 embeddings.

| Parametru | Semnificație | CV RMSE | CV R² | În margine 20% |
|---|---|---|---|---|
| `a` (discrimination) | Discriminabilitate IRT | 0.324 | 0.063 | **90.7%** ✓ |
| `b` (difficulty) | Dificultate IRT (logit) | 2.708 | 0.086 | 39.6% |
| `c` (guessing) | Lower asymptote | 0.008 | 0.998 | **99.9%** ✓ |
| `d` (upper asymptote) | Upper asymptote | 0.205 | 0.096 | 27.9% |
| `p_correct` | Dificultate empirică [0,1] | 0.264 | 0.229 | 52.8% |
| `item_discrimination` | Discriminare empirică [-1,1] | 0.368 | 0.116 | 62.9% |

**Notă margine 20%:** pragul este `0.20 × (max - min)` per parametru (ex. `b` cu range=8 → prag ±1.6; `p_correct` cu range=1 → prag ±0.20).

---

## Limitări cunoscute și îmbunătățiri

### Shrinkage Bayesian pe `b` și `d`

Cu 12 modele mici (35–65% acuratețe) și modelul 4PL, py-irt aplică un prior Bayesian puternic care **comprimă** toți parametrii spre medie. Efectul vizibil: `b` are R²=0.086 (practic nepredicibil din text), iar `d` are R²=0.095.

**Soluții în ordinea costului:**

| Soluție | Cost | Câștig estimat pe `b` |
|---|---|---|
| Schimbă la 2PL (`--irt-model 2pl`) | Re-run etapa 3 (~45 min) | Moderat — mai puțini parametri, mai puțin shrinkage |
| Embeddings mai mari (bge-large, 1024d) | Re-run etapa 4 (~15 min) | ~20–30% RMSE reduction |
| Adaugă modele 14B+ (phi4:14b, qwen2.5:32b) | Download + full re-inference (~12h) | Mare — estimări IRT mai stabile |
| Toate combinat | ~14h total | Potențial 60–75% within 20% pentru `b` |

### `p_correct` vs. IRT `b` ca proxy de dificultate

`p_correct` (media coloanei din response matrix) este **mai predictibil** din text (R²=0.229 vs. 0.086) deoarece nu suferă de shrinkage Bayesian. Dacă scopul este să estimezi dificultatea relativă a întrebărilor, `p_correct` este recomandată ca target primar.

### deepseek-r1 și modele de tip reasoning

deepseek-r1:7b generează blocuri `<think>...</think>` care depășesc frecvent 1024 tokeni (~23% erori de parsing) și nu poate fi folosit ca examinator fiabil. Exclude-le prin `reasoning_model_patterns` în CFG sau prin `models_config.json`.

---

## Input / Output

### Inferență cu predictor standalone

Artefactele din `irt_runs/{run_id}/` sunt portabile. Copiază directorul și importă:

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
    "a":                   1.43,    # discrimination IRT
    "b":                   0.21,    # difficulty IRT (logit, scala ~[-3, 3])
    "c":                   0.25,    # guessing (lower asymptote)
    "d":                   0.97,    # upper asymptote
    "p_correct":           0.61,    # dificultate empirică [0=greu, 1=ușor]
    "item_discrimination": 0.48,    # corelație punct-biserială [-1, 1]
    "difficulty":          "medie", # etichetă semantică pentru b
    "discrimination":      "bun discriminativă",
}
```

### Etichete semantice

| `b` | Dificultate |
|---|---|
| `b < -1.5` | Foarte ușoară |
| `-1.5 ≤ b < -0.5` | Ușoară |
| `-0.5 ≤ b < 0.5` | Medie |
| `0.5 ≤ b < 1.5` | Grea |
| `b ≥ 1.5` | Foarte grea |

| `a` | Discriminare |
|---|---|
| `a < 0.5` | Slab discriminativă |
| `0.5 ≤ a < 1.0` | Moderat discriminativă |
| `1.0 ≤ a < 2.0` | Bun discriminativă |
| `a ≥ 2.0` | Excelent discriminativă |

---

## Setup

```bash
git clone <repo-url>
cd PoCW-IRT-Calibrator

# py-irt necesită Python <3.12
/opt/homebrew/bin/python3.10 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# macOS: OpenMP pentru XGBoost
brew install libomp

# Pornește Ollama (trebuie să ruleze în fundal)
ollama serve
```

### Descarcă modelele

```bash
# Verifică ce e instalat deja
python3 download_models.py --check

# Tier 1 — minim viabil (~5.6 GB)
python3 download_models.py --fast

# Tier 1+2 — recomandat (~11.8 GB)
python3 download_models.py --tier 2

# Tier 1+2+3 — complet (~28 GB)
python3 download_models.py --tier 3

# Test funcțional după download
python3 download_models.py --verify
```

---

## Rulare

```bash
source .venv/bin/activate

# Test rapid cu 5 modele Tier 1 (~3-4 ore)
python3 train_irt_predictor.py --questions 300 --irt-model 2pl --fast

# Rulare standard — 12 modele, ~11 270 întrebări (~11-14 ore)
python3 train_irt_predictor.py --irt-model 4pl

# Reia de unde s-a oprit (checkpointing per model)
python3 train_irt_predictor.py --resume

# Re-rulează doar etapa 5 (regresorii) cu date existente
python3 train_irt_predictor.py --resume --skip-stage 1 --skip-stage 2 --skip-stage 3 --skip-stage 4

# Re-rulează etapele 4+5 (reembedding + regresorii) cu response matrix existentă
python3 train_irt_predictor.py --resume --skip-stage 1 --skip-stage 2 --skip-stage 3
```

### Durate estimate (M4 Max, 36 GB RAM)

| Etapă | ~11 270 întrebări × 12 modele | `--fast` (5 modele) |
|---|---|---|
| Download datasets (prima dată) | ~5 min | ~5 min |
| Response matrix MC/TF | ~7–9 ore | ~2–3 ore |
| Validare open-ended (qwen2.5:14b) | ~2–3 ore | ~1 oră |
| IRT fitting (py-irt, 3000 epoch-uri) | ~15 min | ~10 min |
| Embedding + XGBoost (5-fold CV) | ~10 min | ~5 min |
| **Total** | **~11–14 ore** | **~3–4 ore** |

---

## CLI — opțiuni complete

```
python3 train_irt_predictor.py [opțiuni]

  --questions N         Număr total de întrebări (default: 12000; efective: 11270)
  --irt-model MODEL     Modelul IRT: 1pl, 2pl, 4pl (default: 4pl)
  --fast                Limitează la 5 modele Tier 1 (tinyllama, qwen2.5:1.5b,
                        llama3.2:1b, smollm2:1.7b, gemma2:2b)
  --resume              Reia din cel mai recent run (matching checkpoints)
  --skip-stage N        Sare etapa N (1-5); poate fi repetat
                        ex: --skip-stage 2 --skip-stage 3
```

`--resume` caută automat în `irt_checkpoints/` după un run cu același număr de modele, model IRT și datasets, și reia din checkpointurile existente.

---

## Artefacte generate

```
irt_runs/{timestamp}_{N}q_{M}m_{irt}_{datasets}/
├── response_matrix.npy       ← (M modele × N întrebări) binară
├── embeddings.npy            ← (N, 384) float32
├── irt/
│   ├── irt_params.json       ← arrays a, b, [c, d] per întrebare
│   └── responses.jsonlines   ← input brut pentru py-irt
├── xgb_regressors.pkl        ← 6 regresorii XGBoost serializați
├── irt_predictor.py          ← predictor standalone (copiabil)
├── training_dataset.parquet  ← N întrebări + toți parametrii IRT + empirici
├── training_dataset.csv      ← același dataset în format CSV
├── metrics.json              ← CV RMSE, R², within_20pct_margin per param
├── training_config.json      ← config complet + timestamp + acuratețe modele
├── training.log              ← log complet al run-ului
└── plots/
    ├── accuracy_per_model.png
    ├── irt_distributions.png
    ├── xgb_metrics.png
    └── response_matrix_heatmap.png

irt_checkpoints/{timestamp}_{N}q_{M}m_{irt}_{datasets}/
├── questions.json                    ← întrebările (evită re-descărcarea)
└── responses_{model}.json            ← răspunsurile per model (12 fișiere)
```

Numele run-ului (`{timestamp}_{N}q_{M}m_{irt}_{datasets}`) identifică unic configurația și permite `--resume` să găsească checkpointurile corecte.
