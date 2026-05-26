#!/usr/bin/env python3
"""
train_irt_predictor.py
──────────────────────
Pipeline complet: dataset → response matrix → IRT params → regressor text→IRT

Etape:
  1. Descarcă dataset MMLU (MC, 14K întrebări, 57 subiecte) de pe HuggingFace
  2. Generează response matrix: fiecare model Ollama răspunde la fiecare întrebare
  3. Antrenează IRT (2PL sau 4PL via py-irt) pe response matrix
  4. Embed-uiește întrebările cu sentence-transformers
  5. Antrenează XGBoost regressor: embedding → [a, b, c, d]
  6. Salvează modelul și evaluează

Durată estimată pe M4 Max:
  · 5 000 întrebări, 8 modele: ~1–2h
  · 14 000 întrebări, 8 modele: ~3–5h

Usage:
    python train_irt_predictor.py
    python train_irt_predictor.py --questions 5000 --irt-model 2pl --skip-stage 2
    python train_irt_predictor.py --resume  # reia de unde s-a oprit

Dependențe:
    pip install datasets sentence-transformers xgboost scikit-learn \
                httpx tqdm numpy pandas py-irt torch
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════
#  CONFIG — modifică aici
# ══════════════════════════════════════════════════════════════

CFG = {
    # ── Dataset mix ───────────────────────────────────────────
    # Fiecare sursă poate fi activată/dezactivată; n = nr. întrebări eșantionate
    "datasets": ["mmlu", "boolq", "triviaqa"],
    "dataset_mix": {
        "mmlu":     {"n": 4000},   # MC 4 variante, 57 subiecte academice
        "boolq":    {"n": 4000},   # True/False (da/nu)
        "triviaqa": {"n": 4000},   # Open-ended, răspuns factual scurt
    },
    "random_seed": 42,

    # ── Modele Ollama ─────────────────────────────────────────
    "models_config_path": "models_config.json",
    "fallback_models": [
        "tinyllama:1.1b",
        "qwen2.5:1.5b",
        "llama3.2:1b",
        "smollm2:1.7b",
        "gemma2:2b",
        "phi3.5",
        "qwen2.5:3b",
        "llama3.2:3b",
    ],
    # Modele folosite cu --fast (Tier 1, sub 2GB, rapide)
    "fast_models": [
        "tinyllama:1.1b",
        "qwen2.5:1.5b",
        "llama3.2:1b",
        "smollm2:1.7b",
        "gemma2:2b",
    ],

    # ── Ollama API ────────────────────────────────────────────
    "ollama_base_url":  "http://localhost:11434",
    "concurrency":      12,
    "num_predict":      5,      # tokeni pentru MC/TF (doar litera)
    "open_num_predict": 80,     # tokeni pentru open-ended
    "ollama_timeout":   45.0,

    # ── Validator open-ended ──────────────────────────────────
    # Trebuie să fie UN MODEL DIN AFARA listei de examinatori (evită bias circular).
    # qwen2.5:14b: ~8.2GB, excelent la yes/no, nu e în lista de examinatori.
    # Alternativă: gemma2:9b (~5.4GB), phi4:14b (~8.9GB)
    "validator_model":        "qwen2.5:14b",
    "validator_concurrency":  6,
    "validator_timeout":      90.0,
    "validator_num_predict":  10,

    # ── Modele de reasoning (chain-of-thought) ────────────────
    # Pentru acestea se dezactivează thinking în API (think=false)
    # și se mărește num_predict ca fallback
    "reasoning_model_patterns": ["deepseek-r1", "qwq", ":thinking"],
    "reasoning_num_predict":    1024,  # suficient pentru thinking + răspuns

    # ── IRT ───────────────────────────────────────────────────
    "irt_model":        "2pl",  # "1pl", "2pl", "4pl"
                                # 4PL necesită mai mulți examinatori (>15) pentru stabilitate
    "irt_epochs":       3000,
    "irt_device":       "cpu",  # py-irt suportă doar "cpu" și "cuda" (nu MPS)

    # ── Embedding ─────────────────────────────────────────────
    # bge-small: 33M params, 384 dim, rapid, bun calitativ
    # bge-m3: multilingv, 1.2B, mai bun dar mai lent
    "embedder_model":   "BAAI/bge-small-en-v1.5",
    "embed_batch_size": 256,

    # ── XGBoost ───────────────────────────────────────────────
    "xgb_n_estimators": 600,
    "xgb_max_depth":    6,
    "xgb_lr":           0.05,
    "xgb_subsample":    0.8,
    "xgb_colsample":    0.8,
    "test_split":       0.15,

    # ── Output ────────────────────────────────────────────────
    "output_dir":       "irt_output",
    "checkpoint_dir":   "irt_checkpoints",
}

# ══════════════════════════════════════════════════════════════

# Configurare logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  UTILITĂȚI
# ─────────────────────────────────────────────────────────────

def ensure_dirs():
    for d in [CFG["output_dir"], CFG["checkpoint_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_models_config() -> list[str]:
    """Citește modelele din fișierul generat de download_models.py."""
    if CFG["models_config_path"] == "__fast__":
        log.info(f"--fast: folosesc doar modelele Tier 1: {CFG['fast_models']}")
        return CFG["fast_models"]
    cfg_path = Path(CFG["models_config_path"])
    if cfg_path.exists():
        with open(cfg_path) as f:
            data = json.load(f)
        models = data.get("available_models", [])
        # Preia validatorul din config dacă nu e setat explicit
        if "validator_model" in data and CFG["validator_model"] == "qwen2.5:14b":
            CFG["validator_model"] = data["validator_model"]
        if models:
            log.info(f"Modele din {cfg_path}: {models}")
            return models
    log.warning(f"{cfg_path} nu există sau e gol — folosesc fallback_models din config.")
    log.warning("Rulează download_models.py pentru management automat al modelelor.")
    return CFG["fallback_models"]


def format_question(question: str, choices: list[str]) -> str:
    """Formatează o întrebare ca prompt pentru Ollama (0–4 variante)."""
    n = min(len(choices), 4)
    if n == 0:
        return f"Q: {question}\nAnswer briefly in 1-2 sentences:"
    letters = ["A", "B", "C", "D"]
    choice_str = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(n))
    valid = "/".join(letters[:n])
    return f"Q: {question}\n{choice_str}\nAnswer with only the letter ({valid}):"


def _strip_thinking(text: str) -> str:
    """Elimină blocurile <think>...</think> generate de modele reasoning (DeepSeek-R1 etc.)."""
    import re
    # Elimină blocuri complete <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Dacă thinking-ul nu e închis (tăiat de num_predict), ia tot ce e după >
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def extract_answer_letter(response: str, valid_letters: str = "ABCD") -> Optional[str]:
    """
    Extrage prima literă validă din output-ul modelului.
    valid_letters limitează ce se acceptă (ex: "AB" pentru True/False).
    """
    if not response:
        return None
    response = _strip_thinking(response)
    if not response:
        return None
    valid = set(valid_letters.upper())
    for char in response.strip()[:20]:
        if char.upper() in valid:
            return char.upper()
    for char in response:
        if char.upper() in valid:
            return char.upper()
    return None


# ─────────────────────────────────────────────────────────────
#  ETAPA 1: Încarcă dataset
# ─────────────────────────────────────────────────────────────

def _load_mmlu(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    questions = []
    for i in sorted(idx):
        item = ds[int(i)]
        choices = item.get("choices", [])
        answer_idx = int(item.get("answer", 0))
        n_ch = min(len(choices), 4)
        if n_ch < 2:
            continue
        choices = choices[:n_ch]
        questions.append({
            "id": f"mmlu_{i}",
            "question": item["question"],
            "choices": choices,
            "n_choices": n_ch,
            "answer_idx": answer_idx,
            "answer_letter": "ABCD"[answer_idx],
            "correct_answer": choices[answer_idx],
            "correct_aliases": [],
            "subject": item.get("subject", "unknown"),
            "question_type": "mc",
            "prompt": format_question(item["question"], choices),
        })
    return questions


def _load_boolq(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="validation", trust_remote_code=True)
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    questions = []
    for i in sorted(idx):
        item = ds[int(i)]
        answer = bool(item["answer"])
        answer_idx = 0 if answer else 1
        choices = ["True", "False"]
        questions.append({
            "id": f"boolq_{i}",
            "question": item["question"],
            "choices": choices,
            "n_choices": 2,
            "answer_idx": answer_idx,
            "answer_letter": "AB"[answer_idx],
            "correct_answer": str(answer),
            "correct_aliases": [],
            "subject": "boolean",
            "question_type": "tf",
            "prompt": format_question(item["question"], choices),
        })
    return questions


def _load_triviaqa(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    # rc.nocontext: întrebări fără pasaj de context, răspunsuri scurte
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation", trust_remote_code=True)
    rng = np.random.default_rng(seed + 2)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    questions = []
    for i in sorted(idx):
        item = ds[int(i)]
        answer_val = item["answer"]["value"]
        aliases = [a.lower().strip() for a in item["answer"].get("aliases", [answer_val])]
        questions.append({
            "id": f"trivia_{i}",
            "question": item["question"],
            "choices": [],
            "n_choices": 0,
            "answer_idx": None,
            "answer_letter": None,
            "correct_answer": answer_val,
            "correct_aliases": aliases,
            "subject": item.get("type", "trivia"),
            "question_type": "open",
            "prompt": format_question(item["question"], []),
        })
    return questions


def stage1_load_dataset() -> list[dict]:
    """
    Încarcă și combină MMLU (MC), BoolQ (T/F) și TriviaQA (open-ended).
    Returnează lista de întrebări cu format standardizat.
    """
    log.info("═" * 60)
    log.info("ETAPA 1: Încărcare dataset mix (MMLU + BoolQ + TriviaQA)")
    log.info("═" * 60)

    checkpoint_path = Path(CFG["checkpoint_dir"]) / "questions.json"
    expected_total = sum(v["n"] for v in CFG["dataset_mix"].values())
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            questions = json.load(f)
        if len(questions) == expected_total:
            log.info(f"Checkpoint găsit: {checkpoint_path}")
            log.info(f"→ {len(questions)} întrebări din checkpoint")
            return questions
        else:
            log.info(f"Checkpoint găsit cu {len(questions)} întrebări, dar se cer {expected_total} — reincarc dataset.")
            checkpoint_path.unlink()

    try:
        from datasets import load_dataset  # noqa — verifică că e instalat
    except ImportError:
        log.error("Lipsă pachet: pip install datasets")
        sys.exit(1)

    loaders = {"mmlu": _load_mmlu, "boolq": _load_boolq, "triviaqa": _load_triviaqa}
    seed = CFG["random_seed"]
    questions: list[dict] = []

    for source in CFG["datasets"]:
        if source not in loaders:
            log.warning(f"  Sursă necunoscută '{source}' — skip")
            continue
        n = CFG["dataset_mix"].get(source, {}).get("n", 1000)
        log.info(f"  Încarc {source} (n={n}) ...")
        try:
            batch = loaders[source](n, seed)
            log.info(f"  → {len(batch)} întrebări [{source}]")
            questions.extend(batch)
        except Exception as e:
            log.error(f"  Eroare la {source}: {e}")
            log.error("  Continuă fără această sursă.")

    # Amestecă (seed fix pentru reproductibilitate)
    rng = np.random.default_rng(seed)
    rng.shuffle(questions)

    # Statistici
    by_type: dict[str, int] = {}
    by_subj: dict[str, int] = {}
    for q in questions:
        by_type[q["question_type"]] = by_type.get(q["question_type"], 0) + 1
        by_subj[q["subject"]]       = by_subj.get(q["subject"], 0) + 1

    log.info(f"Total: {len(questions)} întrebări")
    for qtype, cnt in sorted(by_type.items()):
        log.info(f"  {qtype}: {cnt}")
    top5 = sorted(by_subj.items(), key=lambda x: -x[1])[:5]
    log.info(f"Top subiecte: {top5}")

    with open(checkpoint_path, "w") as f:
        json.dump(questions, f, indent=2)
    log.info(f"Checkpoint salvat: {checkpoint_path}")

    return questions


# ─────────────────────────────────────────────────────────────
#  ETAPA 2: Response matrix — inferență Ollama
# ─────────────────────────────────────────────────────────────

def _is_reasoning_model(model: str) -> bool:
    """Returnează True dacă modelul folosește chain-of-thought (DeepSeek-R1 etc.)."""
    model_lower = model.lower()
    return any(p in model_lower for p in CFG.get("reasoning_model_patterns", []))


async def query_ollama_async(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    model: str,
    question: dict,
) -> tuple[str, Optional[str]]:
    """
    Trimite o întrebare la un model Ollama.
    MC/TF  → (question_id, litera "A"/"B"/...)
    Open   → (question_id, raw_text) pentru validare ulterioară
    """
    is_open = question.get("question_type") == "open"
    is_reasoning = _is_reasoning_model(model)

    if is_open:
        num_predict = CFG["open_num_predict"]
    elif is_reasoning:
        num_predict = CFG["reasoning_num_predict"]
    else:
        num_predict = CFG["num_predict"]

    payload: dict = {
        "model": model,
        "prompt": question["prompt"],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "top_p": 1.0,
            "top_k": 1,
            "repeat_penalty": 1.0,
        },
    }
    if is_reasoning and not is_open:
        payload["think"] = False

    async with semaphore:
        try:
            r = await client.post(
                "/api/generate",
                json=payload,
                timeout=CFG["ollama_timeout"],
            )
            r.raise_for_status()
            response_text = r.json().get("response", "").strip()
            if is_open:
                return question["id"], response_text or None
            valid = "ABCD"[:question.get("n_choices", 4)]
            return question["id"], extract_answer_letter(response_text, valid)
        except httpx.TimeoutException:
            log.debug(f"Timeout pentru {model} pe {question['id']}")
            return question["id"], None
        except httpx.HTTPStatusError as e:
            log.debug(f"HTTP {e.response.status_code} pentru {model}: {e}")
            return question["id"], None
        except Exception as e:
            log.debug(f"Eroare neașteptată {model}/{question['id']}: {e}")
            return question["id"], None


async def validate_open_responses_batch(
    pairs: list[tuple[dict, str]],
    base_url: str,
) -> dict[str, int]:
    """
    Judecă răspunsuri open-ended via CFG['validator_model'].
    Verifică mai întâi alias-urile exacte (fără LLM), apoi apelează LLM.
    Returnează {question_id: 1/0}.
    """
    validator = CFG["validator_model"]
    semaphore = asyncio.Semaphore(CFG["validator_concurrency"])

    async def _judge(client: httpx.AsyncClient, q: dict, response: str) -> tuple[str, int]:
        resp_norm = response.lower().strip()
        # Verificare rapidă prin alias-uri (evită apelul LLM pentru răspunsuri clare)
        aliases = q.get("correct_aliases") or [q["correct_answer"].lower().strip()]
        if any(alias and alias in resp_norm for alias in aliases if len(alias) > 2):
            return q["id"], 1

        prompt = (
            f"Question: {q['question']}\n"
            f"Correct answer: {q['correct_answer']}\n"
            f"Student answer: {response.strip()[:300]}\n\n"
            "Is the student answer correct? Reply with only 'yes' or 'no':"
        )
        async with semaphore:
            try:
                r = await client.post(
                    "/api/generate",
                    json={
                        "model": validator,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0,
                            "num_predict": CFG["validator_num_predict"],
                            "top_k": 1,
                        },
                    },
                    timeout=CFG["validator_timeout"],
                )
                r.raise_for_status()
                verdict = r.json().get("response", "").strip().lower()
                return q["id"], 1 if verdict.startswith("yes") else 0
            except Exception:
                return q["id"], 0

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(CFG["validator_timeout"]),
        limits=httpx.Limits(max_connections=CFG["validator_concurrency"] + 2),
    ) as client:
        tasks = [_judge(client, q, resp) for q, resp in pairs]
        results: dict[str, int] = {}
        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks),
            desc=f"  validator ({validator})", leave=False,
        ):
            qid, score = await coro
            results[qid] = score
    return results


async def run_model_inference(
    model: str,
    questions: list[dict],
    base_url: str,
) -> dict[str, int]:
    """
    Rulează un model pe toate întrebările în paralel.
    MC/TF: scor imediat prin compararea literei.
    Open:  colectează raw text, validează în batch cu validator_model.
    Returnează {question_id: 1/0}.
    """
    semaphore = asyncio.Semaphore(CFG["concurrency"])
    q_map = {q["id"]: q for q in questions}

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(CFG["ollama_timeout"]),
        limits=httpx.Limits(max_connections=CFG["concurrency"] + 4),
    ) as client:
        tasks = [query_ollama_async(client, semaphore, model, q) for q in questions]

        results: dict[str, int] = {}
        open_pending: list[tuple[dict, str]] = []  # (question, raw_response)
        none_count = 0
        correct_count = 0

        with tqdm(
            total=len(questions), desc=f"  {model:<22}", unit="q", dynamic_ncols=True,
        ) as pbar:
            for coro in asyncio.as_completed(tasks):
                qid, response = await coro
                q = q_map.get(qid)

                if q is None or response is None:
                    results[qid] = 0
                    none_count += 1
                elif q.get("question_type") == "open":
                    open_pending.append((q, response))
                else:
                    correct = int(response == q["answer_letter"])
                    results[qid] = correct
                    correct_count += correct

                pbar.update(1)
                pbar.set_postfix(
                    acc=f"{correct_count / max(1, len(results)):.1%}",
                    open=len(open_pending),
                    err=none_count,
                )

    # Validare open-ended după ce toate răspunsurile sunt colectate
    if open_pending:
        log.info(f"  → {len(open_pending)} răspunsuri open de validat cu {CFG['validator_model']} ...")
        validated = await validate_open_responses_batch(open_pending, base_url)
        results.update(validated)
        correct_count += sum(validated.values())

    # Completează cu 0 pentru eventuale lipsuri
    for q in questions:
        results.setdefault(q["id"], 0)

    acc = sum(results.values()) / max(1, len(questions))
    log.info(f"  {model}: accuracy={acc:.3f}, errors={none_count}, open_validated={len(open_pending)}")
    return results


def check_ollama_has_model(model: str, base_url: str) -> bool:
    """Verifică rapid că modelul e disponibil."""
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        installed = {m["name"] for m in r.json().get("models", [])}
        # Verifică exact sau cu :latest suffix
        return (
            model in installed
            or f"{model}:latest" in installed
            or model.split(":")[0] in installed
        )
    except Exception:
        return False


def stage2_response_matrix(
    questions: list[dict],
    models: list[str],
) -> np.ndarray:
    """
    Generează response matrix: shape (n_models, n_questions), valori 0/1.
    Checkpointing per model — dacă se oprește, reia de unde a rămas.
    """
    log.info("═" * 60)
    log.info("ETAPA 2: Generare response matrix")
    log.info(f"  {len(models)} modele × {len(questions)} întrebări")
    log.info(f"  Concurență: {CFG['concurrency']} requests paralele")
    log.info("═" * 60)

    n_models = len(models)
    n_questions = len(questions)
    q_ids = [q["id"] for q in questions]
    base_url = CFG["ollama_base_url"]

    # Inițializare matrice
    matrix = np.full((n_models, n_questions), -1, dtype=np.int8)  # -1 = nedeterminat

    # Verifică checkpoints existente
    ckpt_dir = Path(CFG["checkpoint_dir"])
    models_processed = []

    for mi, model in enumerate(models):
        ckpt_path = ckpt_dir / f"responses_{model.replace(':', '_').replace('/', '_')}.json"

        if ckpt_path.exists():
            with open(ckpt_path) as f:
                saved = json.load(f)
            if saved.get("model") == model and saved.get("n_questions") == n_questions:
                responses = saved["responses"]
                for qi, qid in enumerate(q_ids):
                    matrix[mi, qi] = responses.get(qid, 0)
                acc = matrix[mi].mean()
                log.info(f"  Checkpoint: {model} (acc={acc:.3f}) ← din {ckpt_path.name}")
                models_processed.append(model)
                continue

        # Verifică că modelul e instalat
        if not check_ollama_has_model(model, base_url):
            log.warning(f"  ⚠ Model {model} nu e instalat în Ollama — skip")
            matrix[mi] = 0  # tratează ca incapabil
            continue

        log.info(f"\n  Procesez model {mi + 1}/{n_models}: {model}")
        t0 = time.time()

        try:
            responses = asyncio.run(
                run_model_inference(model, questions, base_url)
            )
        except KeyboardInterrupt:
            log.warning("\n  Întrerupt de utilizator. Salvez progresul ...")
            # Salvează ce s-a procesat până acum
            _save_matrix_partial(matrix, models, q_ids, models_processed)
            sys.exit(0)
        except Exception as e:
            log.error(f"  Eroare critică la {model}: {e}")
            log.error("  Tratez ca zero responses și continui.")
            responses = {qid: 0 for qid in q_ids}

        # Umple rândul din matrice
        for qi, qid in enumerate(q_ids):
            matrix[mi, qi] = responses.get(qid, 0)

        elapsed = time.time() - t0
        acc = matrix[mi].mean()
        log.info(f"  → {model}: accuracy={acc:.3f}, timp={elapsed:.0f}s "
                 f"({elapsed/len(questions)*1000:.0f}ms/q)")

        # Checkpoint pentru model
        with open(ckpt_path, "w") as f:
            json.dump({
                "model": model,
                "n_questions": n_questions,
                "responses": responses,
                "accuracy": float(acc),
                "elapsed_s": elapsed,
            }, f)

        models_processed.append(model)

    # Înlocuiește -1 cu 0 (modele skip-uite)
    matrix = np.where(matrix == -1, 0, matrix)

    # Salvează matricea completă
    matrix_path = Path(CFG["output_dir"]) / "response_matrix.npy"
    np.save(matrix_path, matrix)
    log.info(f"\n  Response matrix salvată: {matrix_path}")
    log.info(f"  Shape: {matrix.shape} — {matrix.sum()} răspunsuri corecte din {matrix.size}")

    # Statistici pe model
    log.info("\n  Statistici per model:")
    for mi, model in enumerate(models):
        valid_rows = matrix[mi][matrix[mi] >= 0]
        acc = valid_rows.mean() if len(valid_rows) > 0 else 0
        log.info(f"    {model:<25} acc={acc:.3f}")

    return matrix


def _save_matrix_partial(matrix, models, q_ids, models_processed):
    """Salvează starea parțială în caz de întrerupere."""
    path = Path(CFG["output_dir"]) / "response_matrix_partial.npy"
    np.save(path, matrix)
    log.info(f"Matrice parțială salvată: {path}")
    log.info(f"Modele procesate: {models_processed}")


# ─────────────────────────────────────────────────────────────
#  ETAPA 3: Fit IRT
# ─────────────────────────────────────────────────────────────

def _write_pyirt_jsonlines(
    matrix: np.ndarray,
    models: list[str],
    question_ids: list[str],
    path: Path,
) -> None:
    """Scrie datele în formatul jsonlines cerut de py-irt.
    Fiecare linie = un subiect (model) cu toate răspunsurile sale:
    {"subject_id": "model_0", "responses": {"q_mc_0": 1, "q_mc_1": 0, ...}}
    """
    with open(path, "w") as f:
        for mi, model in enumerate(models):
            responses = {qid: int(matrix[mi, qi]) for qi, qid in enumerate(question_ids)}
            row = {"subject_id": model, "responses": responses}
            f.write(json.dumps(row) + "\n")
    log.info(f"  pyirt jsonlines scris: {path} ({len(models)} rânduri)")


def _parse_pyirt_output(output_dir: Path, question_ids: list[str], questions: list[dict] = None) -> dict[str, np.ndarray]:
    """
    Parsează fișierul best_parameters.json generat de py-irt.
    Returnează dict cu arrays per parametru, indexate ca question_ids.
    """
    params_file = output_dir / "best_parameters.json"
    if not params_file.exists():
        # încearcă și alte locații posibile
        for fname in ["parameters.json", "item_parameters.json"]:
            alt = output_dir / fname
            if alt.exists():
                params_file = alt
                break
        else:
            raise FileNotFoundError(
                f"Nu găsesc fișierul de parametri în {output_dir}. "
                f"Conținut: {list(output_dir.iterdir())}"
            )

    with open(params_file) as f:
        data = json.load(f)

    log.info(f"  Parametri py-irt disponibili: {list(data.keys())}")

    # py-irt stochează parametrii ca liste, cu item_ids corespunzătoare
    item_ids_order = data.get("item_ids", [])

    # Construiește mapping qid → index în output
    id_to_idx = {iid: i for i, iid in enumerate(item_ids_order)}

    def get_param(key: str, default: float = 0.0) -> np.ndarray:
        raw = data.get(key, [])
        if not raw:
            return np.full(len(question_ids), default)
        arr = np.array(raw, dtype=np.float32)
        # Reordonează după question_ids originale
        result = np.full(len(question_ids), default, dtype=np.float32)
        for qi, qid in enumerate(question_ids):
            idx = id_to_idx.get(qid)
            if idx is not None and idx < len(arr):
                result[qi] = arr[idx]
        return result

    # c default per item: 1/n_choices (0.5 T/F, 0.33 3-opt, 0.25 4-opt)
    if questions:
        q_nc = {q["id"]: q.get("n_choices", 4) for q in questions}
        c_defaults = np.array([1.0 / q_nc.get(qid, 4) for qid in question_ids], dtype=np.float32)
    else:
        c_defaults = np.full(len(question_ids), 0.25, dtype=np.float32)

    # Mapare câmpuri py-irt → semantică IRT standard
    raw_c = data.get("gammas", [])
    params = {
        "a": get_param("discriminations", default=1.0),
        "b": get_param("difficulties", default=0.0),
        "c": get_param("gammas") if raw_c else c_defaults,
        "d": get_param("lambdas", default=1.0),
    }

    # Validare și clipare la range-uri rezonabile IRT
    params["a"] = np.clip(params["a"], 0.01, 3.0)     # discrimination: (0, 3] — valori > 3 = overfitting
    params["b"] = np.clip(params["b"], -4.0, 4.0)     # difficulty: logit scale, practic [-3, +3]
    params["c"] = np.clip(params["c"], 0.0, 0.5)      # guessing: max 50%
    params["d"] = np.clip(params["d"], 0.5, 1.0)      # upper asymptote

    log.info(f"  Statistici parametri:")
    for pname, arr in params.items():
        log.info(f"    {pname}: mean={arr.mean():.3f}, std={arr.std():.3f}, "
                 f"range=[{arr.min():.3f}, {arr.max():.3f}]")

    return params


def stage3_fit_irt(
    matrix: np.ndarray,
    models: list[str],
    questions: list[dict],
) -> dict[str, np.ndarray]:
    """
    Antrenează modelul IRT via Python API py-irt (evită CLI-ul care are bug typer/click).
    Returnează parametrii per întrebare.
    """
    log.info("═" * 60)
    log.info(f"ETAPA 3: Fit IRT ({CFG['irt_model'].upper()})")
    log.info(f"  {len(models)} modele (examinatori) × {len(questions)} întrebări (items)")
    log.info("═" * 60)

    irt_dir = Path(CFG["output_dir"]) / "irt"
    irt_dir.mkdir(exist_ok=True)
    params_cache = irt_dir / "irt_params.json"

    if params_cache.exists():
        log.info(f"  Checkpoint IRT găsit: {params_cache}")
        with open(params_cache) as f:
            saved = json.load(f)
        return {k: np.array(v) for k, v in saved.items()}

    q_ids = [q["id"] for q in questions]

    try:
        from py_irt.training import IrtModelTrainer
        from py_irt.config import IrtConfig
    except ImportError:
        log.warning("  py-irt nu e instalat. Fallback la scipy MLE.")
        return _stage3_fallback_scipy(matrix, q_ids, questions)

    jsonlines_path = irt_dir / "responses.jsonlines"
    _write_pyirt_jsonlines(matrix, models, q_ids, jsonlines_path)

    log.info(f"  IRT model: {CFG['irt_model'].upper()}, epochs={CFG['irt_epochs']}, device={CFG['irt_device']}")
    log.info("  (poate dura 5–20 minute în funcție de date)")

    t0 = time.time()
    try:
        config = IrtConfig(
            model_type=CFG["irt_model"],
            epochs=CFG["irt_epochs"],
        )
        trainer = IrtModelTrainer(data_path=jsonlines_path, config=config)
        trainer.train(device=CFG["irt_device"])
        elapsed = time.time() - t0
        log.info(f"  IRT antrenat în {elapsed:.0f}s")
    except KeyboardInterrupt:
        log.warning("  Întrerupt. Fallback la scipy.")
        return _stage3_fallback_scipy(matrix, q_ids, questions)
    except Exception as e:
        log.error(f"  Eroare py-irt: {e}. Fallback la scipy.")
        return _stage3_fallback_scipy(matrix, q_ids, questions)

    try:
        raw = trainer.export(items=list(q_ids))
        params = _parse_pyirt_results(raw, q_ids, questions)
    except Exception as e:
        log.error(f"  Eroare export py-irt: {e}. Fallback la scipy.")
        return _stage3_fallback_scipy(matrix, q_ids, questions)

    with open(params_cache, "w") as f:
        json.dump({k: v.tolist() for k, v in params.items()}, f)
    log.info(f"  Parametri IRT salvați: {params_cache}")

    return params


def _parse_pyirt_results(
    raw: dict,
    question_ids: list[str],
    questions: list[dict] = None,
) -> dict[str, np.ndarray]:
    """Parsează rezultatul trainer.export() în arrays indexate după question_ids.

    py-irt export() returnează:
      "item_ids":  {idx: item_id}  (dict index→id)
      "disc":      [float, ...]    (discrimination, 2PL/4PL)
      "diff":      [float, ...]    (difficulty)
      "lambdas":   [float, ...]    (upper asymptote, 4PL only)
    """
    # ix_to_item_id e dict {0: 'q_mc_0', 1: 'q_mc_1', ...}
    ix_to_item_id = raw.get("item_ids", {})
    item_id_to_ix = {v: int(k) for k, v in ix_to_item_id.items()}

    def get_param(key: str, default: float = 0.0) -> np.ndarray:
        raw_list = raw.get(key, [])
        result = np.full(len(question_ids), default, dtype=np.float32)
        if raw_list:
            arr = np.array(raw_list, dtype=np.float32)
            for qi, qid in enumerate(question_ids):
                idx = item_id_to_ix.get(qid)
                if idx is not None and idx < len(arr):
                    result[qi] = arr[idx]
        return result

    if questions:
        q_nc = {q["id"]: q.get("n_choices", 4) for q in questions}
        c_defaults = np.array(
            [1.0 / nc if (nc := q_nc.get(qid, 4)) > 0 else 0.0 for qid in question_ids],
            dtype=np.float32,
        )
    else:
        c_defaults = np.full(len(question_ids), 0.25, dtype=np.float32)

    params = {
        "a": np.clip(get_param("disc", 1.0), 0.01, 3.0),
        "b": np.clip(get_param("diff", 0.0), -4.0, 4.0),
        "c": np.clip(get_param("gammas") if raw.get("gammas") else c_defaults, 0.0, 0.5),
        "d": np.clip(get_param("lambdas", 1.0), 0.5, 1.0),
    }

    log.info("  Statistici parametri IRT:")
    for pname, arr in params.items():
        log.info(f"    {pname}: mean={arr.mean():.3f}, std={arr.std():.3f}, "
                 f"range=[{arr.min():.3f}, {arr.max():.3f}]")
    return params


def _stage3_fallback_scipy(
    matrix: np.ndarray,
    q_ids: list[str],
    questions: list[dict] = None,
) -> dict[str, np.ndarray]:
    """
    Estimare 2PL simplificată cu scipy.
    Folosită când py-irt nu e disponibil.
    Marginally maximum likelihood estimation per item.
    """
    log.info("  Scipy fallback: estimare 2PL per item ...")
    from scipy.special import expit
    from scipy.optimize import minimize

    n_models, n_items = matrix.shape

    # Ability estimation simplă: proporția de răspunsuri corecte → logit
    accuracy_per_model = matrix.mean(axis=1)
    accuracy_per_model = np.clip(accuracy_per_model, 0.01, 0.99)
    thetas = np.log(accuracy_per_model / (1 - accuracy_per_model))

    difficulties = np.zeros(n_items)
    discriminations = np.ones(n_items)

    def neg_log_lik_2pl(params, responses, thetas):
        a, b = params[0], params[1]
        a = max(a, 0.01)
        p = expit(a * (thetas - b))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll = np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))
        return -ll

    log.info(f"  Estimez 2PL pentru {n_items} items ...")
    for i in tqdm(range(n_items), desc="  Items IRT", unit="item"):
        responses = matrix[:, i].astype(float)
        try:
            result = minimize(
                neg_log_lik_2pl,
                x0=[1.0, 0.0],
                args=(responses, thetas),
                method="L-BFGS-B",
                bounds=[(0.01, 5.0), (-5.0, 5.0)],
                options={"maxiter": 200},
            )
            if result.success:
                discriminations[i] = result.x[0]
                difficulties[i] = result.x[1]
        except Exception:
            pass

    # c = 1/n_choices per item (0.5 T/F, 0.33 pentru 3 variante, 0.25 pentru 4)
    if questions:
        q_nc = {q["id"]: q.get("n_choices", 4) for q in questions}
        # open-ended (n_choices=0) → c=0 (nu există guessing fără variante)
        guessing = np.array(
            [1.0 / nc if (nc := q_nc.get(qid, 4)) > 0 else 0.0 for qid in q_ids],
            dtype=np.float32,
        )
    else:
        guessing = np.full(n_items, 0.25, dtype=np.float32)

    params = {
        "a": discriminations.astype(np.float32),
        "b": difficulties.astype(np.float32),
        "c": guessing,
        "d": np.ones(n_items, dtype=np.float32),
    }

    # Salvează
    params_cache = Path(CFG["output_dir"]) / "irt" / "irt_params.json"
    params_cache.parent.mkdir(exist_ok=True)
    with open(params_cache, "w") as f:
        json.dump({k: v.tolist() for k, v in params.items()}, f)

    log.info(f"  Parametri scipy salvați: {params_cache}")
    log.info(f"  a (disc): mean={discriminations.mean():.3f}, std={discriminations.std():.3f}")
    log.info(f"  b (diff): mean={difficulties.mean():.3f}, std={difficulties.std():.3f}")

    return params


# ─────────────────────────────────────────────────────────────
#  ETAPA 4: Embed întrebările
# ─────────────────────────────────────────────────────────────

def stage4_embed(questions: list[dict]) -> np.ndarray:
    """
    Generează embeddings pentru toate întrebările.
    Returnează array (n_questions, embedding_dim).
    """
    log.info("═" * 60)
    log.info("ETAPA 4: Embedding întrebări")
    log.info(f"  Model: {CFG['embedder_model']}")
    log.info("═" * 60)

    embed_cache = Path(CFG["output_dir"]) / "embeddings.npy"
    if embed_cache.exists():
        embeddings = np.load(embed_cache)
        log.info(f"  Checkpoint: {embed_cache} — shape={embeddings.shape}")
        return embeddings

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("Lipsă: pip install sentence-transformers")
        sys.exit(1)

    log.info(f"  Descărcare/încărcare {CFG['embedder_model']} ...")
    embedder = SentenceTransformer(CFG["embedder_model"])

    # Formatare text: întrebare + choices concatenate
    # Prefixul "query: " e specific bge-small pentru retrieval
    texts = []
    for q in questions:
        choices_str = " | ".join(
            f"{letter}) {choice}"
            for letter, choice in zip("ABCD", q["choices"])
        )
        # Include subiectul pentru context domenial
        subject = q.get("subject", "")
        text = f"query: [{subject}] {q['question']} {choices_str}"
        texts.append(text)

    log.info(f"  Embed {len(texts)} texte cu batch_size={CFG['embed_batch_size']} ...")
    t0 = time.time()

    embeddings = embedder.encode(
        texts,
        batch_size=CFG["embed_batch_size"],
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity ready
        device="mps" if sys.platform == "darwin" else None,
    )

    elapsed = time.time() - t0
    log.info(f"  Embeddings generate: shape={embeddings.shape}, timp={elapsed:.0f}s")

    np.save(embed_cache, embeddings)
    log.info(f"  Salvat: {embed_cache}")

    return embeddings


# ─────────────────────────────────────────────────────────────
#  Statistici empirice din matricea de răspunsuri
# ─────────────────────────────────────────────────────────────

def _compute_empirical_stats(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """
    Statistici empirice directe din response matrix (fără shrinkage Bayesian).

    p_correct: proporția modelelor care au răspuns corect per întrebare
               (echivalent dificultăție empirice; 1=ușor, 0=greu)
    item_discrimination: corelație punct-biserială între scorul itemului și
                         scorul rest (sum fără itemul curent) — proxy pentru
                         IRT discrimination, mai stabil cu puțini examinatori
    """
    m = matrix.astype(float)
    n_models, n_questions = m.shape

    p_correct = m.mean(axis=0)

    total_scores = m.sum(axis=1)
    item_discrimination = np.empty(n_questions)
    for j in range(n_questions):
        col = m[:, j]
        rest = total_scores - col
        if col.std() < 1e-10 or rest.std() < 1e-10:
            item_discrimination[j] = 0.0
            continue
        item_discrimination[j] = float(np.corrcoef(col, rest)[0, 1])

    item_discrimination = np.where(
        np.isfinite(item_discrimination), item_discrimination, 0.0
    )
    return {"p_correct": p_correct, "item_discrimination": item_discrimination}


# ─────────────────────────────────────────────────────────────
#  ETAPA 5: Antrenează regresorii XGBoost
# ─────────────────────────────────────────────────────────────

def stage5_train_regressors(
    embeddings: np.ndarray,
    irt_params: dict[str, np.ndarray],
    questions: list[dict],
) -> dict:
    """
    Antrenează câte un regressor XGBoost per parametru IRT.
    Evaluare prin 5-fold cross-validation; model final antrenat pe tot setul.
    Returnează dict cu modelele antrenate și metricile de evaluare.
    """
    log.info("═" * 60)
    log.info("ETAPA 5: Antrenare regresor XGBoost (5-fold CV)")
    log.info(f"  X: embeddings shape={embeddings.shape}")
    log.info(f"  y: {list(irt_params.keys())} (IRT params + statistici empirice)")
    log.info("═" * 60)

    try:
        import xgboost as xgb
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error, r2_score
        import joblib
    except ImportError as e:
        log.error(f"Lipsă pachete: {e}")
        log.error("pip install xgboost scikit-learn joblib")
        sys.exit(1)

    X = embeddings
    models_trained = {}
    metrics = {}

    # Features suplimentare din text (simple, dar ajută XGBoost)
    text_features = _extract_text_features(questions)
    X_augmented = np.hstack([X, text_features])
    log.info(f"  X augmentat cu {text_features.shape[1]} features text: shape={X_augmented.shape}")

    kf = KFold(n_splits=5, shuffle=True, random_state=CFG["random_seed"])

    xgb_params = dict(
        n_estimators=CFG["xgb_n_estimators"],
        max_depth=CFG["xgb_max_depth"],
        learning_rate=CFG["xgb_lr"],
        subsample=CFG["xgb_subsample"],
        colsample_bytree=CFG["xgb_colsample"],
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=CFG["random_seed"],
        n_jobs=-1,
        verbosity=0,
    )

    for param_name, y in irt_params.items():
        y = np.asarray(y, dtype=np.float64)
        log.info(f"\n  → Parametru: {param_name}")
        log.info(f"    Distribuție: mean={y.mean():.3f}, std={y.std():.3f}, "
                 f"range=[{y.min():.3f}, {y.max():.3f}]")

        if y.std() < 0.001:
            log.warning(f"    Variație quasi-zero pentru {param_name} — skip training")
            models_trained[param_name] = {"type": "constant", "value": float(y.mean())}
            metrics[param_name] = {"cv_rmse": 0.0, "cv_r2": 1.0, "within_20pct_margin": 1.0, "note": "constant"}
            continue

        # ── 5-fold CV pentru evaluare ──────────────────────────
        oof_preds = np.zeros(len(y))
        for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_augmented), start=1):
            fold_model = xgb.XGBRegressor(**xgb_params)
            fold_model.fit(X_augmented[train_idx], y[train_idx], verbose=False)
            oof_preds[val_idx] = fold_model.predict(X_augmented[val_idx])
            fold_rmse = float(np.sqrt(mean_squared_error(y[val_idx], oof_preds[val_idx])))
            log.info(f"    Fold {fold_i}/5: RMSE={fold_rmse:.4f}")

        cv_rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
        cv_r2 = float(r2_score(y, oof_preds))
        y_range = float(y.max() - y.min())
        margin_thresh = max(0.20 * y_range, 1e-6)
        within_20pct = float(np.mean(np.abs(oof_preds - y) <= margin_thresh))

        log.info(f"    CV (5-fold OOF): RMSE={cv_rmse:.4f}, R²={cv_r2:.4f}")
        log.info(f"    În margine 20% (±{margin_thresh:.4f}): {within_20pct:.1%}")

        # ── Model final pe tot setul (pentru inferență) ────────
        t0 = time.time()
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_augmented, y, verbose=False)
        elapsed = time.time() - t0
        log.info(f"    Model final (tot setul) antrenat în {elapsed:.1f}s")

        # Feature importance top-3 (din features text)
        n_embed = X.shape[1]
        feat_names = [f"emb_{i}" for i in range(n_embed)] + _text_feature_names()
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-3:][::-1]
        top_feats = [(feat_names[i], importances[i]) for i in top_idx if i >= n_embed]
        if top_feats:
            log.info(f"    Top text features: {top_feats}")

        models_trained[param_name] = model
        metrics[param_name] = {
            "cv_rmse": cv_rmse,
            "cv_r2": cv_r2,
            "within_20pct_margin": within_20pct,
            "margin_threshold": margin_thresh,
            "n_samples": len(y),
        }

    # Salvează modelele
    output_dir = Path(CFG["output_dir"])
    import joblib
    models_path = output_dir / "xgb_regressors.pkl"
    joblib.dump(models_trained, models_path)
    log.info(f"\n  Regresorii salvați: {models_path}")

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"  Metrici salvate: {metrics_path}")

    return {"models": models_trained, "metrics": metrics}


def _text_feature_names() -> list[str]:
    return [
        "question_len", "n_words", "n_choices_chars",
        "has_not", "has_always", "has_never",
        "has_which", "has_what", "has_why", "has_how",
    ]


def _extract_text_features(questions: list[dict]) -> np.ndarray:
    """Features simple din text care ajută predicția IRT."""
    feats = []
    for q in questions:
        text = q["question"]
        choices = q.get("choices", [])
        words = text.lower().split()

        feats.append([
            len(text),                                              # lungime caractere
            len(words),                                            # număr cuvinte
            sum(len(c) for c in choices),                         # lungime totală choices
            int("not" in words or "n't" in text.lower()),         # negație
            int("always" in words),
            int("never" in words),
            int("which" in words),
            int("what" in words),
            int("why" in words),
            int("how" in words),
        ])

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
#  ETAPA 6: Salvează artefacte finale + predictor API
# ─────────────────────────────────────────────────────────────

def _accuracy_stats(matrix: np.ndarray, models: list[str], questions: list[dict]) -> dict:
    """Calculează acuratețea per model și per tip de întrebare."""
    q_types = [q.get("question_type", "mc") for q in questions]
    type_indices = {}
    for qi, qt in enumerate(q_types):
        type_indices.setdefault(qt, []).append(qi)

    stats = {"per_model": {}, "per_type": {}, "overall": float(matrix.mean())}

    for mi, model in enumerate(models):
        row = matrix[mi]
        model_stats = {"total": float(row.mean())}
        for qt, idxs in type_indices.items():
            model_stats[qt] = float(row[idxs].mean()) if idxs else 0.0
        stats["per_model"][model] = model_stats

    for qt, idxs in type_indices.items():
        if idxs:
            stats["per_type"][qt] = float(matrix[:, idxs].mean())

    return stats


def stage6_save_and_evaluate(
    questions: list[dict],
    irt_params: dict[str, np.ndarray],
    embedder_model_name: str,
    trained: dict,
    models_used: list[str],
    matrix: np.ndarray,
) -> None:
    """Salvează tot și scrie un predictor de sine stătător."""
    log.info("═" * 60)
    log.info("ETAPA 6: Salvare artefacte finale")
    log.info("═" * 60)

    output_dir = Path(CFG["output_dir"])

    # Salvează questions + params ca dataset complet
    records = []
    for qi, q in enumerate(questions):
        row = {
            "id": q["id"],
            "question": q["question"],
            "choices": q["choices"],
            "n_choices": q.get("n_choices", len(q["choices"])),
            "answer_letter": q.get("answer_letter"),
            "correct_answer": q.get("correct_answer", ""),
            "subject": q.get("subject", ""),
            "question_type": q.get("question_type", "mc"),
        }
        # IRT params (4PL: a, b, c, d; 2PL: a, b; empirice: p_correct, item_discrimination)
        for pname in ["a", "b", "c", "d"]:
            if pname in irt_params:
                row[f"irt_{pname}"] = float(irt_params[pname][qi])
        for pname in ["p_correct", "item_discrimination"]:
            if pname in irt_params:
                row[pname] = float(irt_params[pname][qi])
        records.append(row)

    df = pd.DataFrame(records)
    df.to_parquet(output_dir / "training_dataset.parquet", index=False)
    df.to_csv(output_dir / "training_dataset.csv", index=False)
    log.info(f"  Dataset complet salvat: training_dataset.[parquet|csv]")

    # Calculează statistici acuratețe
    acc_stats = _accuracy_stats(matrix, models_used, questions)

    # Scrie config final
    final_config = {
        "dataset": "+".join(CFG["datasets"]),
        "n_questions": len(questions),
        "models_used": models_used,
        "irt_model": CFG["irt_model"],
        "embedder": embedder_model_name,
        "xgb_metrics": trained["metrics"],
        "accuracy": acc_stats,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(final_config, f, indent=2)

    # Scrie predictor API de sine stătător
    predictor_code = _generate_predictor_code(embedder_model_name)
    predictor_path = output_dir / "irt_predictor.py"
    with open(predictor_path, "w") as f:
        f.write(predictor_code)
    log.info(f"  Predictor API scris: irt_predictor.py")

    # ── Sumar final ───────────────────────────────────────────
    log.info("\n" + "═" * 60)
    log.info("  SUMAR FINAL")
    log.info("═" * 60)
    log.info(f"  Dataset: {len(questions)} întrebări × {len(models_used)} modele")
    log.info(f"  IRT model: {CFG['irt_model'].upper()}")

    log.info("\n  Acuratețe modele (per tip de întrebare):")
    types = sorted(acc_stats["per_type"].keys())
    header = f"    {'Model':<25}" + "".join(f"{t:>9}" for t in types) + f"{'Total':>9}"
    log.info(header)
    log.info("    " + "─" * (25 + 9 * (len(types) + 1)))
    for model, ms in acc_stats["per_model"].items():
        row = f"    {model:<25}"
        for t in types:
            row += f"{ms.get(t, 0):>8.1%} "
        row += f"{ms['total']:>8.1%}"
        log.info(row)
    log.info("    " + "─" * (25 + 9 * (len(types) + 1)))
    avg_row = f"    {'MEDIE':<25}"
    for t in types:
        avg_row += f"{acc_stats['per_type'][t]:>8.1%} "
    avg_row += f"{acc_stats['overall']:>8.1%}"
    log.info(avg_row)

    log.info("\n  Metrici XGBoost (5-fold CV):")
    for param, m in trained["metrics"].items():
        if isinstance(m, dict) and "cv_rmse" in m:
            within = m.get("within_20pct_margin", float("nan"))
            log.info(
                f"    {param}: RMSE={m['cv_rmse']:.4f}, R²={m['cv_r2']:.4f}, "
                f"în margine 20%={within:.1%}"
            )
        elif isinstance(m, dict):
            log.info(f"    {param}: {m.get('note', 'ok')}")

    log.info(f"\n  Artefacte salvate în: {output_dir}/")
    log.info("    ├── xgb_regressors.pkl")
    log.info("    ├── irt_predictor.py")
    log.info("    ├── training_dataset.parquet / .csv")
    log.info("    ├── embeddings.npy")
    log.info("    ├── training_config.json")
    log.info("    ├── plots/")
    log.info("    └── training.log")

    # Generează ploturi
    stage7_plots(df, irt_params, matrix, models_used, trained["metrics"], acc_stats, output_dir)


def stage7_plots(
    df: "pd.DataFrame",
    irt_params: dict[str, np.ndarray],
    matrix: np.ndarray,
    models: list[str],
    xgb_metrics: dict,
    acc_stats: dict,
    output_dir: Path,
) -> None:
    """Generează și salvează ploturile relevante în output_dir/plots/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        log.warning("  matplotlib nu e instalat — ploturi omise. pip install matplotlib")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    types = sorted(df["question_type"].unique())
    type_colors = {"mc": "#4C72B0", "tf": "#55A868", "open": "#C44E52"}

    # ── Plot 1: Acuratețe per model și tip ────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.9), 5))
    x = np.arange(len(models))
    width = 0.8 / max(len(types), 1)
    for ti, qtype in enumerate(types):
        vals = [acc_stats["per_model"][m].get(qtype, 0) for m in models]
        offset = (ti - len(types) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=qtype,
                      color=type_colors.get(qtype, f"C{ti}"), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(":")[0] for m in models], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Acuratețe")
    ax.set_title("Acuratețe per model și tip de întrebare")
    ax.legend(title="Tip")
    ax.set_ylim(0, 1)
    ax.axhline(acc_stats["overall"], color="gray", linestyle="--", linewidth=1,
               label=f"Medie globală {acc_stats['overall']:.1%}")
    ax.legend(title="Tip", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy_per_model.png", dpi=150)
    plt.close(fig)
    log.info("  Plot: accuracy_per_model.png")

    # ── Plot 2: Distribuții parametri IRT per tip ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (param, label) in zip(axes, [("a", "Discrimination (a)"), ("b", "Difficulty (b)")]):
        for qtype in types:
            mask = df["question_type"] == qtype
            vals = irt_params[param][mask.values]
            ax.hist(vals, bins=40, alpha=0.6, label=qtype,
                    color=type_colors.get(qtype, None), density=True)
        ax.set_xlabel(label)
        ax.set_ylabel("Densitate")
        ax.set_title(f"Distribuție {label}")
        ax.legend(title="Tip")
    fig.suptitle("Distribuții parametri IRT")
    fig.tight_layout()
    fig.savefig(plots_dir / "irt_distributions.png", dpi=150)
    plt.close(fig)
    log.info("  Plot: irt_distributions.png")

    # ── Plot 3: XGBoost metrics ───────────────────────────────
    params_with_metrics = [p for p, m in xgb_metrics.items() if isinstance(m, dict)]
    if params_with_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        rmse_vals = [xgb_metrics[p].get("rmse", 0) for p in params_with_metrics]
        r2_vals   = [xgb_metrics[p].get("r2", 0)   for p in params_with_metrics]
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

        axes[0].bar(params_with_metrics, rmse_vals, color=colors[:len(params_with_metrics)])
        axes[0].set_title("RMSE per parametru IRT")
        axes[0].set_ylabel("RMSE")

        axes[1].bar(params_with_metrics, r2_vals, color=colors[:len(params_with_metrics)])
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_title("R² per parametru IRT")
        axes[1].set_ylabel("R²")

        fig.suptitle("Performanță regresor XGBoost")
        fig.tight_layout()
        fig.savefig(plots_dir / "xgb_metrics.png", dpi=150)
        plt.close(fig)
        log.info("  Plot: xgb_metrics.png")

    # ── Plot 4: Heatmap response matrix — toate întrebările, sortate după b ─
    n_q = matrix.shape[1]
    n_m = len(models)
    # Sortează întrebările după dificultate b (ușoare → grele)
    b_vals = irt_params["b"]
    sort_idx = np.argsort(b_vals)
    mat_sorted = matrix[:, sort_idx]
    q_types_sorted = df["question_type"].values[sort_idx]

    # Dimensiunea figurii: 1 pixel per întrebare la 100 DPI
    # width = n_q / 100 inch, height = n_m * 0.55 + 1.5 inch
    fig_w = max(8, n_q / 100)
    fig_h = max(3, n_m * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat_sorted, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([m.split(":")[0] for m in models], fontsize=8)
    ax.set_xlabel(f"Întrebări sortate după dificultate b (n={n_q})")
    ax.set_title("Response matrix — toate întrebările, verde=corect, roșu=greșit")

    # Marchează granițele dintre tipuri de întrebări
    type_boundaries = []
    prev = q_types_sorted[0]
    for i, qt in enumerate(q_types_sorted):
        if qt != prev:
            type_boundaries.append((i, prev))
            prev = qt
    type_boundaries.append((n_q, prev))
    start = 0
    for end, qt in type_boundaries:
        mid = (start + end) / 2
        ax.axvline(end - 0.5, color="white", linewidth=0.5, alpha=0.7)
        ax.text(mid, -0.7, qt, ha="center", va="top", fontsize=7,
                color=type_colors.get(qt, "black"), transform=ax.get_xaxis_transform())
        start = end

    fig.colorbar(im, ax=ax, fraction=0.01, pad=0.01, label="corect")
    fig.tight_layout()
    fig.savefig(plots_dir / "response_matrix_heatmap.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Plot: response_matrix_heatmap.png ({n_q} întrebări × {n_m} modele)")

    log.info(f"  Toate ploturile → {plots_dir}/")


def _generate_predictor_code(embedder_name: str) -> str:
    """Generează codul predictorului standalone."""
    return f'''#!/usr/bin/env python3
"""
irt_predictor.py — IRT Parameter Predictor
Generat automat de train_irt_predictor.py

Usage:
    from irt_predictor import IRTPredictor
    p = IRTPredictor()
    result = p.predict("Ce este fotosinteza?", ["A...", "B...", "C...", "D..."])
    # result: {{"a": 1.2, "b": 0.3, "c": 0.25, "d": 0.95, "p_avg_student": 0.61}}
"""

import math
import numpy as np
from pathlib import Path


class IRTPredictor:
    """
    Predictor IRT 4PL din text.
    Parametri returnați:
      a  — discrimination: cât de bine separă cei care știu de cei care nu știu
      b  — difficulty: pragul theta la care P(corect) = (c+d)/2
      c  — guessing: probabilitate de răspuns corect la abilitate -∞ (lower asymptote)
      d  — upper asymptote: probabilitate maximă de răspuns corect
    """

    def __init__(self, model_dir: str = "."):
        import joblib
        from sentence_transformers import SentenceTransformer

        model_dir = Path(model_dir)
        regressors_path = model_dir / "xgb_regressors.pkl"
        if not regressors_path.exists():
            regressors_path = Path(__file__).parent / "xgb_regressors.pkl"

        self.regressors = joblib.load(regressors_path)
        self.embedder = SentenceTransformer("{embedder_name}")
        self._text_feature_names = [
            "question_len", "n_words", "n_choices_chars",
            "has_not", "has_always", "has_never",
            "has_which", "has_what", "has_why", "has_how",
        ]

    def _text_features(self, question: str, choices: list) -> np.ndarray:
        text = question
        words = text.lower().split()
        return np.array([[
            len(text), len(words), sum(len(c) for c in choices),
            int("not" in words or "n\'t" in text.lower()),
            int("always" in words), int("never" in words),
            int("which" in words), int("what" in words),
            int("why" in words), int("how" in words),
        ]], dtype=np.float32)

    def predict(
        self,
        question: str,
        choices: list[str] = None,
        theta: float = 0.0,
    ) -> dict:
        """
        Prezice parametrii IRT pentru o întrebare.

        Args:
            question: textul întrebării
            choices: lista de variante (opțional, max 4)
            theta: abilitatea examinee-ului (implicit 0.0 = student mediu)

        Returns:
            dict cu a, b, c, d, p_correct, interpretation
        """
        choices = choices or []
        choices_str = " | ".join(
            f"{{l}}) {{c}}" for l, c in zip("ABCD", choices[:4])
        )
        text = f"query: {{question}} {{choices_str}}"

        emb = self.embedder.encode([text], normalize_embeddings=True)
        text_feats = self._text_features(question, choices)
        X = np.hstack([emb, text_feats])

        params = {{}}
        for pname, reg in self.regressors.items():
            if isinstance(reg, dict) and reg.get("type") == "constant":
                params[pname] = reg["value"]
            else:
                params[pname] = float(reg.predict(X)[0])

        a = max(params.get("a", 1.0), 0.01)
        b = params.get("b", 0.0)
        c = max(0.0, min(0.5, params.get("c", 0.25)))
        d = max(0.5, min(1.0, params.get("d", 1.0)))

        # ICC 4PL: P(corect | theta) = c + (d-c) / (1 + exp(-a*(theta-b)))
        p_correct = c + (d - c) / (1 + math.exp(-a * (theta - b)))

        # Interpretare semantică
        difficulty_label = (
            "foarte ușoară" if b < -1.5 else
            "ușoară"        if b < -0.5 else
            "medie"         if b < 0.5  else
            "grea"          if b < 1.5  else
            "foarte grea"
        )
        disc_label = (
            "slab discriminativă" if a < 0.5 else
            "moderat"             if a < 1.0 else
            "bun discriminativă"  if a < 2.0 else
            "excelent discriminativă"
        )

        return {{
            "a":              round(a, 4),   # discrimination
            "b":              round(b, 4),   # difficulty
            "c":              round(c, 4),   # guessing (lower asymptote)
            "d":              round(d, 4),   # upper asymptote
            "p_correct":      round(p_correct, 4),
            "difficulty":     difficulty_label,
            "discrimination": disc_label,
        }}

    def predict_batch(self, items: list[dict]) -> list[dict]:
        """
        Prezice pentru o listă de items.
        Fiecare item: {{"question": "...", "choices": ["A", "B", "C", "D"]}}
        """
        return [
            self.predict(item["question"], item.get("choices", []))
            for item in items
        ]


if __name__ == "__main__":
    import json, sys
    p = IRTPredictor()

    question = sys.argv[1] if len(sys.argv) > 1 else "What is photosynthesis?"
    choices = ["A process", "A chemical", "A plant", "A cell"]
    result = p.predict(question, choices)
    print(json.dumps(result, indent=2, ensure_ascii=False))
'''


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Antrenează regresori text→IRT pe MMLU cu Ollama"
    )
    parser.add_argument(
        "--questions", type=int, default=None,
        help="Total întrebări (distribuite proporțional între surse)"
    )
    parser.add_argument(
        "--irt-model", choices=["1pl", "2pl", "4pl"], default=None,
        help="Tipul modelului IRT (override CFG['irt_model'])"
    )
    parser.add_argument(
        "--skip-stage", type=int, action="append", default=[],
        help="Sare peste etapa N (poate fi specificat de mai multe ori)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reia de unde s-a oprit (folosește checkpoints)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="Cereri paralele Ollama"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Folosește doar modelele Tier 1 (5 modele mici, <2GB) — pentru teste rapide"
    )
    args = parser.parse_args()

    # Override config
    if args.questions:
        # Distribuie proporțional între surse (păstrează raportul)
        total_default = sum(v["n"] for v in CFG["dataset_mix"].values())
        ratio = args.questions / total_default
        for src in CFG["dataset_mix"]:
            CFG["dataset_mix"][src]["n"] = max(1, int(CFG["dataset_mix"][src]["n"] * ratio))
    if args.irt_model:
        CFG["irt_model"] = args.irt_model
    if args.concurrency:
        CFG["concurrency"] = args.concurrency
    if args.fast:
        CFG["models_config_path"] = "__fast__"   # sentinel — override în load_models_config

    total_n = sum(v["n"] for v in CFG["dataset_mix"].values())
    log.info("═" * 60)
    log.info("  IRT PREDICTOR TRAINING PIPELINE")
    log.info(f"  Datasets: {CFG['datasets']} | N≈{total_n}")
    for src, cfg in CFG["dataset_mix"].items():
        log.info(f"    {src}: {cfg['n']}")
    log.info(f"  IRT: {CFG['irt_model'].upper()} | Embedder: {CFG['embedder_model']}")
    log.info("═" * 60)

    # Verifică Ollama
    try:
        r = httpx.get(f"{CFG['ollama_base_url']}/api/tags", timeout=5.0)
        r.raise_for_status()
    except Exception:
        log.error(f"Ollama nu răspunde la {CFG['ollama_base_url']}")
        log.error("Pornește-l cu: ollama serve")
        sys.exit(1)

    # Încarcă lista de modele
    models = load_models_config()
    log.info(f"Modele configurate: {models}")

    # Sanity check: validatorul nu trebuie să fie și examinator (exact match)
    validator = CFG["validator_model"]
    if validator in models:
        log.warning("═" * 60)
        log.warning(f"  BIAS ALERT: validatorul '{validator}' este și examinator!")
        log.warning("  Răspunsurile open-ended vor fi judecate de același model care le-a generat.")
        log.warning("  Schimbă 'validator_model' cu un model din afara listei de examinatori.")
        log.warning("═" * 60)

    # ── Setează output_dir și checkpoint_dir unice per rulare ────
    from datetime import datetime
    n_models = len(models)
    datasets_tag = "-".join(CFG["datasets"])
    run_suffix = f"{total_n}q_{n_models}m_{CFG['irt_model']}_{datasets_tag}"

    if args.resume:
        # Caută cel mai recent checkpoint care se potrivește parametrilor curenți
        ckpt_base = Path("irt_checkpoints")
        matching = sorted(
            [d for d in ckpt_base.glob(f"*_{run_suffix}") if d.is_dir()],
            key=lambda d: d.name, reverse=True
        )
        if matching:
            existing = matching[0]
            CFG["checkpoint_dir"] = str(existing)
            # Derivă output_dir din același run tag
            run_tag = existing.name
            CFG["output_dir"] = str(Path("irt_runs") / run_tag)
            log.info(f"  --resume: continuă din {existing}")
        else:
            log.warning("  --resume: nu există checkpoint potrivit, pornesc de la zero.")
            args.resume = False

    if not args.resume:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        run_tag = f"{ts}_{run_suffix}"
        CFG["output_dir"]     = str(Path("irt_runs") / run_tag)
        CFG["checkpoint_dir"] = str(Path("irt_checkpoints") / run_tag)

    ensure_dirs()
    log.info(f"  Output dir: {CFG['output_dir']}")

    # Adaugă file handler în folderul rulării curente
    log_path = Path(CFG["output_dir"]) / "training.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    log.addHandler(fh)

    t_total = time.time()

    def _find_data_file(*rel_parts) -> Path:
        """Caută un fișier în dirul curent, apoi în cel mai recent run existent."""
        primary = Path(CFG["output_dir"]).joinpath(*rel_parts)
        if primary.exists():
            return primary
        # Caută în irt_runs/ (runs noi cu subdirectoare)
        candidates = sorted(
            [d for d in Path("irt_runs").glob("*") if d.is_dir()],
            key=lambda p: p.name, reverse=True
        )
        for run in candidates:
            alt = run.joinpath(*rel_parts)
            if alt.exists():
                log.info(f"  Datele găsite în: {run.name}")
                return alt
        # Fallback: irt_output/ (format vechi)
        legacy = Path("irt_output").joinpath(*rel_parts)
        if legacy.exists():
            log.info("  Date găsite (format vechi: irt_output/)")
            return legacy
        return primary

    def _find_ckpt_file(*rel_parts) -> Path:
        primary = Path(CFG["checkpoint_dir"]).joinpath(*rel_parts)
        if primary.exists():
            return primary
        # Caută în subdirectoarele irt_checkpoints/ (runs noi)
        candidates = sorted(
            [d for d in Path("irt_checkpoints").glob("*") if d.is_dir()],
            key=lambda p: p.name, reverse=True
        )
        for run in candidates:
            alt = run.joinpath(*rel_parts)
            if alt.exists():
                log.info(f"  Checkpoint găsit în: {run.name}")
                return alt
        # Fallback: root-ul irt_checkpoints/ (format vechi)
        legacy = Path("irt_checkpoints").joinpath(*rel_parts)
        if legacy.exists():
            log.info("  Checkpoint găsit (format vechi: irt_checkpoints/)")
            return legacy
        return primary

    # ── Etapa 1 ──────────────────────────────────────────────
    if 1 not in args.skip_stage:
        questions = stage1_load_dataset()
    else:
        log.info("Etapa 1 skip — încarc din checkpoint ...")
        ckpt = _find_ckpt_file("questions.json")
        with open(ckpt) as f:
            questions = json.load(f)
        log.info(f"  {len(questions)} întrebări din checkpoint")

    # ── Etapa 2 ──────────────────────────────────────────────
    if 2 not in args.skip_stage:
        matrix = stage2_response_matrix(questions, models)
    else:
        log.info("Etapa 2 skip — încarc response matrix ...")
        matrix_path = _find_data_file("response_matrix.npy")
        if not matrix_path.exists():
            log.error(f"Response matrix nu există: {matrix_path}")
            sys.exit(1)
        matrix = np.load(matrix_path)
        log.info(f"  Matrix shape: {matrix.shape}")

    # ── Etapa 3 ──────────────────────────────────────────────
    if 3 not in args.skip_stage:
        irt_params = stage3_fit_irt(matrix, models, questions)
    else:
        log.info("Etapa 3 skip — încarc parametri IRT ...")
        params_path = _find_data_file("irt", "irt_params.json")
        with open(params_path) as f:
            saved = json.load(f)
        irt_params = {k: np.array(v) for k, v in saved.items()}

    # Adaugă statistici empirice (nu depind de etapa 3, ci de matricea de răspunsuri)
    empirical = _compute_empirical_stats(matrix)
    irt_params.update(empirical)
    log.info(
        f"  Statistici empirice: p_correct mean={empirical['p_correct'].mean():.3f} "
        f"std={empirical['p_correct'].std():.3f}, "
        f"item_discrimination mean={empirical['item_discrimination'].mean():.3f} "
        f"std={empirical['item_discrimination'].std():.3f}"
    )

    # ── Etapa 4 ──────────────────────────────────────────────
    if 4 not in args.skip_stage:
        embeddings = stage4_embed(questions)
    else:
        log.info("Etapa 4 skip — încarc embeddings ...")
        embed_cache = _find_data_file("embeddings.npy")
        embeddings = np.load(embed_cache)
        log.info(f"  Embeddings shape: {embeddings.shape}")

    # ── Etapa 5 ──────────────────────────────────────────────
    if 5 not in args.skip_stage:
        trained = stage5_train_regressors(embeddings, irt_params, questions)
    else:
        log.info("Etapa 5 skip — încarc regresorii ...")
        import joblib
        regressors = joblib.load(_find_data_file("xgb_regressors.pkl"))
        trained = {"models": regressors, "metrics": {}}

    # ── Etapa 6 ──────────────────────────────────────────────
    stage6_save_and_evaluate(
        questions, irt_params,
        CFG["embedder_model"],
        trained, models, matrix,
    )

    total_elapsed = time.time() - t_total
    log.info(f"\n  Pipeline complet în {total_elapsed/60:.1f} minute")
    log.info(f"  → Rezultate în: {CFG['output_dir']}")


if __name__ == "__main__":
    main()
