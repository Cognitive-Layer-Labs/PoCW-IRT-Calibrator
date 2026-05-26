#!/usr/bin/env python3
"""
download_models.py
──────────────────
Descarcă și verifică modelele Ollama necesare pentru generarea
matricei de răspunsuri IRT.

Rulează ÎNAINTE de train_irt_predictor.py

Usage:
    python download_models.py              # descarcă toate modelele din config
    python download_models.py --check      # doar verifică, nu descarcă
    python download_models.py --fast       # doar modelele mici (< 2GB)
    python download_models.py --verify     # verificare completă cu test prompt
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

import httpx

# ══════════════════════════════════════════════════════════════
#  CONFIG — modifică după nevoie
# ══════════════════════════════════════════════════════════════

OLLAMA_BASE_URL = "http://localhost:11434"

# Modele ordonate după dimensiune (mici → mari).
# Diversitatea arhitecturală e importantă pentru IRT:
# vrei răspunsuri variate, nu 8 clone ale aceluiași model.
@dataclass
class ModelSpec:
    name: str
    size_gb: float
    family: str
    desc: str
    priority: int  # 1=must-have, 2=recommended, 3=optional

MODELS: list[ModelSpec] = [
    # ── Tier 1: Must-have (mici, rapide, diverse arhitecturi) ──
    ModelSpec("tinyllama:1.1b",   0.64, "llama",   "TinyLlama 1.1B — baseline slab",          1),
    ModelSpec("qwen2.5:1.5b",     0.99, "qwen",    "Qwen 2.5 1.5B — bun pe cunoștințe",       1),
    ModelSpec("llama3.2:1b",      1.32, "llama",   "Llama 3.2 1B — Meta official",            1),
    ModelSpec("smollm2:1.7b",     1.03, "smollm",  "SmolLM2 1.7B — HuggingFace distilat",    1),
    ModelSpec("gemma2:2b",        1.63, "gemma",   "Gemma 2 2B — Google, raționament bun",    1),

    # ── Tier 2: Recommended (3B, mai bune, ~2GB) ──
    ModelSpec("phi3.5",            2.20, "phi",     "Phi-3.5 Mini 3.8B — Microsoft",           2),
    ModelSpec("qwen2.5:3b",       2.04, "qwen",    "Qwen 2.5 3B — upgrade Qwen",              2),
    ModelSpec("llama3.2:3b",      2.02, "llama",   "Llama 3.2 3B — Meta, mai capabil",        2),

    # ── Tier 3: Optional (7B+, mai lente dar discriminative) ──
    ModelSpec("mistral:7b",       4.10, "mistral",    "Mistral 7B — referință clasică",              3),
    ModelSpec("qwen2.5:7b",       4.68, "qwen",       "Qwen 2.5 7B — cel mai capabil Qwen",          3),
    ModelSpec("llama3.1:8b",      4.92, "llama",      "Llama 3.1 8B — Meta flagship mid-size",       3),
    ModelSpec("deepseek-r1:7b",   4.70, "deepseek",   "DeepSeek-R1 7B — reasoning distilat, unic",   3),
    ModelSpec("phi4-mini:3.8b",   2.50, "phi",        "Phi-4 Mini 3.8B — Microsoft gen 4",           3),
]

# Validator pentru răspunsuri open-ended — trebuie să fie SEPARAT de examinatori
# Nu apare în models_config.json și nu participă la response matrix.
VALIDATOR_MODEL = ModelSpec(
    "qwen2.5:14b", 8.19, "qwen",
    "Qwen 2.5 14B — validator open-ended (nu e examinator)", 0
)

# Prompt de test pentru verificare rapidă
TEST_PROMPT = (
    "Q: What is the chemical symbol for water?\n"
    "A) CO2\nB) H2O\nC) NaCl\nD) O2\n"
    "Answer with just the letter:"
)
EXPECTED_ANSWER = "B"
VERIFY_TIMEOUT = 30  # secunde per model

# ══════════════════════════════════════════════════════════════


def check_ollama_running(base_url: str) -> bool:
    """Verifică dacă Ollama daemon rulează."""
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def get_installed_models(base_url: str) -> set[str]:
    """Returnează setul de modele deja instalate."""
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=10.0)
        r.raise_for_status()
        data = r.json()
        return {m["name"] for m in data.get("models", [])}
    except Exception as e:
        print(f"  [WARN] Nu pot citi lista de modele instalate: {e}")
        return set()


def normalize_model_name(name: str, installed: set[str]) -> Optional[str]:
    """
    Ollama poate stoca modelele cu tag exact sau cu :latest.
    Caută variante posibile.
    """
    if name in installed:
        return name
    # încearcă cu :latest adăugat
    if f"{name}:latest" in installed:
        return f"{name}:latest"
    # încearcă fără tag (e.g. "llama3.2:1b" → "llama3.2")
    base = name.split(":")[0]
    if base in installed:
        return base
    return None


def pull_model(model_name: str, base_url: str) -> bool:
    """
    Descarcă un model via `ollama pull`.
    Afișează progresul în timp real.
    Returnează True dacă reușit.
    """
    print(f"\n  ↓ Descarcă {model_name} ...")
    cmd = ["ollama", "pull", model_name]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        last_status = ""
        for line in process.stdout:  # type: ignore
            line = line.strip()
            if line and line != last_status:
                # formatează mai curat output-ul ollama
                if any(kw in line.lower() for kw in ["pulling", "verifying", "writing", "success"]):
                    print(f"    {line}")
                    last_status = line

        process.wait()
        if process.returncode == 0:
            print(f"  ✓ {model_name} descărcat cu succes")
            return True
        else:
            print(f"  ✗ Eroare la descărcare {model_name} (exit code {process.returncode})")
            return False

    except FileNotFoundError:
        print("  ✗ Comanda 'ollama' nu a fost găsită în PATH.")
        print("    Instalează Ollama de la https://ollama.ai")
        return False
    except KeyboardInterrupt:
        print(f"\n  ⚠ Descărcare întreruptă pentru {model_name}")
        process.terminate()
        return False
    except Exception as e:
        print(f"  ✗ Eroare neașteptată la {model_name}: {e}")
        return False


def verify_model(model_name: str, base_url: str) -> tuple[bool, str]:
    """
    Trimite un prompt de test și verifică că modelul răspunde corect.
    Returnează (success, răspuns_primit).
    """
    try:
        r = httpx.post(
            f"{base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": TEST_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 5,
                    "top_p": 1.0,
                },
            },
            timeout=VERIFY_TIMEOUT,
        )
        r.raise_for_status()
        response_text = r.json().get("response", "").strip()
        # Extrage primul caracter alfabetic
        letter = next(
            (c.upper() for c in response_text if c.isalpha()),
            "?"
        )
        success = letter == EXPECTED_ANSWER
        return success, response_text
    except httpx.TimeoutException:
        return False, "[TIMEOUT]"
    except Exception as e:
        return False, f"[ERROR: {e}]"


def print_summary(results: dict[str, dict]) -> None:
    """Afișează un tabel sumar cu statusul tuturor modelelor."""
    print("\n" + "═" * 70)
    print("  SUMAR MODELE")
    print("═" * 70)
    print(f"  {'Model':<22} {'Status':<12} {'Tier':<8} {'Familie':<10} {'GB':>5}")
    print("  " + "─" * 65)

    ok_count = 0
    for spec in MODELS:
        info = results.get(spec.name, {})
        status = info.get("status", "skipped")
        symbol = {
            "installed": "✓",
            "downloaded": "✓",
            "failed": "✗",
            "skipped": "·",
            "verified": "✓✓",
        }.get(status, "?")
        tier_label = {1: "must", 2: "rec.", 3: "opt."}.get(spec.priority, "?")

        print(
            f"  {symbol} {spec.name:<20} {status:<12} {tier_label:<8} "
            f"{spec.family:<10} {spec.size_gb:>4.1f}GB"
        )
        if status in ("installed", "downloaded", "verified"):
            ok_count += 1

    print("═" * 70)
    print(f"  Modele disponibile: {ok_count}/{len(MODELS)}")

    if ok_count < 5:
        print("\n  ⚠ AVERTISMENT: Sunt recomandate cel puțin 5 modele pentru IRT stabil.")
        print("    Cu mai puțini 'examinatori', parametrii IRT sunt instabili.")
    elif ok_count < 8:
        print(f"\n  → {ok_count} modele OK. Suficient pentru IRT, dar cu mai puțin de 8")
        print("    parametrul de discriminare (a) e mai puțin precis.")
    else:
        print(f"\n  → {ok_count} modele OK. Configurație ideală pentru IRT robust.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Descarcă și verifică modele Ollama pentru IRT training"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verifică doar ce e instalat, nu descarcă nimic",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Descarcă doar modelele Tier 1 (< 2GB)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Testează fiecare model cu un prompt real după instalare",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Descarcă doar modelele până la tier-ul specificat (1=mici, 3=toate)",
    )
    parser.add_argument(
        "--validator",
        action="store_true",
        help=f"Descarcă și modelul validator ({VALIDATOR_MODEL.name}, {VALIDATOR_MODEL.size_gb:.1f}GB) — separat de examinatori",
    )
    args = parser.parse_args()

    print("\n" + "═" * 70)
    print("  IRT MODEL DOWNLOADER")
    print("  Ollama model manager pentru train_irt_predictor.py")
    print("═" * 70)

    # ── Verifică Ollama ──────────────────────────────────────────
    print(f"\n  Verific Ollama la {OLLAMA_BASE_URL} ...")
    if not check_ollama_running(OLLAMA_BASE_URL):
        print("  ✗ Ollama nu rulează!")
        print("    Pornește-l cu: ollama serve")
        print("    Sau descarcă de la: https://ollama.ai")
        sys.exit(1)
    print("  ✓ Ollama rulează")

    # ── Ce e deja instalat ───────────────────────────────────────
    installed = get_installed_models(OLLAMA_BASE_URL)
    print(f"\n  Modele deja instalate: {len(installed)}")
    if installed:
        for m in sorted(installed):
            print(f"    · {m}")

    # ── Determină ce trebuie descărcat ──────────────────────────
    max_tier = args.tier or (1 if args.fast else 3)
    target_models = [m for m in MODELS if m.priority <= max_tier]

    total_size = sum(m.size_gb for m in target_models)
    already_size = sum(
        m.size_gb for m in target_models
        if normalize_model_name(m.name, installed) is not None
    )
    to_download = [
        m for m in target_models
        if normalize_model_name(m.name, installed) is None
    ]

    print(f"\n  Target: {len(target_models)} modele (Tier 1–{max_tier})")
    print(f"  Total spațiu necesar: {total_size:.1f} GB")
    print(f"  Deja instalat: {already_size:.1f} GB")
    print(f"  De descărcat: {sum(m.size_gb for m in to_download):.1f} GB "
          f"({len(to_download)} modele)")

    if args.check:
        print("\n  [--check] Mod verificare, nu descărc nimic.")
        results = {}
        for spec in target_models:
            if normalize_model_name(spec.name, installed) is not None:
                results[spec.name] = {"status": "installed"}
            else:
                results[spec.name] = {"status": "missing"}
        print_summary(results)
        sys.exit(0)

    # ── Descarcă modelele lipsă ──────────────────────────────────
    results: dict[str, dict] = {}

    for spec in target_models:
        found_name = normalize_model_name(spec.name, installed)

        if found_name is not None:
            print(f"\n  ✓ {spec.name} — deja instalat ({spec.size_gb:.1f}GB)")
            results[spec.name] = {"status": "installed", "installed_name": found_name}
        else:
            success = pull_model(spec.name, OLLAMA_BASE_URL)
            if success:
                results[spec.name] = {"status": "downloaded"}
                # actualizează lista de instalate
                installed = get_installed_models(OLLAMA_BASE_URL)
            else:
                results[spec.name] = {"status": "failed"}

    # ── Verificare opțională ─────────────────────────────────────
    if args.verify:
        print("\n" + "─" * 70)
        print("  VERIFICARE MODELE (test prompt)")
        print("─" * 70)

        for spec in target_models:
            if results.get(spec.name, {}).get("status") in ("installed", "downloaded"):
                # găsim numele exact instalat
                found = normalize_model_name(spec.name, installed) or spec.name
                print(f"\n  Testing {found} ...", end="", flush=True)

                start = time.time()
                ok, response = verify_model(found, OLLAMA_BASE_URL)
                elapsed = time.time() - start

                symbol = "✓" if ok else "⚠"
                print(f" {symbol}  ({elapsed:.1f}s)")
                print(f"    Răspuns: '{response[:80]}{'...' if len(response) > 80 else ''}'")
                if not ok:
                    print(f"    Așteptat: '{EXPECTED_ANSWER}' — modelul poate fi slab pe MC")

                results[spec.name]["status"] = "verified" if ok else "installed"
                results[spec.name]["verify_time_s"] = round(elapsed, 2)

    # ── Descarcă validatorul dacă e cerut ───────────────────────
    if args.validator:
        print(f"\n  ── Validator: {VALIDATOR_MODEL.name} ({VALIDATOR_MODEL.size_gb:.1f}GB) ──")
        found = normalize_model_name(VALIDATOR_MODEL.name, installed)
        if found:
            print(f"  ✓ {VALIDATOR_MODEL.name} — deja instalat")
        else:
            pull_model(VALIDATOR_MODEL.name, OLLAMA_BASE_URL)
        installed = get_installed_models(OLLAMA_BASE_URL)
    else:
        v_installed = normalize_model_name(VALIDATOR_MODEL.name, installed) is not None
        v_status = "instalat ✓" if v_installed else "LIPSĂ ✗"
        print(f"\n  Validator ({VALIDATOR_MODEL.name}): {v_status}")
        if not v_installed:
            print(f"    → Rulează: python download_models.py --validator")

    # ── Salvează config pentru train script ──────────────────────
    available_models = [
        spec.name
        for spec in target_models
        if results.get(spec.name, {}).get("status") in ("installed", "downloaded", "verified")
    ]

    config_path = "models_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "ollama_base_url": OLLAMA_BASE_URL,
                "available_models": available_models,  # doar examinatorii, fără validator
                "validator_model": VALIDATOR_MODEL.name,
                "model_details": [
                    {
                        "name": spec.name,
                        "size_gb": spec.size_gb,
                        "family": spec.family,
                        "tier": spec.priority,
                    }
                    for spec in target_models
                    if spec.name in available_models
                ],
            },
            f,
            indent=2,
        )
    print(f"\n  ✓ Config salvat la: {config_path}")
    print("    Folosit automat de train_irt_predictor.py")

    print_summary(results)

    # Exit code non-zero dacă prea puține modele disponibile
    if len(available_models) < 4:
        print("  ✗ Sub 4 modele disponibile — IRT va fi instabil.")
        print("    Rulează fără --fast sau verifică conexiunea la internet.")
        sys.exit(1)


if __name__ == "__main__":
    main()
