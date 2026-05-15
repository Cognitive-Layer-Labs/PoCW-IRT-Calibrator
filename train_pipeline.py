"""Train an IRT difficulty predictor from question text only."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import DatasetDict, get_dataset_config_names, load_dataset, load_from_disk
from datasets.exceptions import DatasetNotFoundError
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict, learning_curve, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from src.features import (
    HybridFeatureExtractor,
    QuestionFeatureExtractor,
    TfidfSvdEmbeddingExtractor,
    TransformerEmbeddingExtractor,
)
from src.predictor import DifficultyPredictor, save_predictor

sns.set_theme(style="whitegrid")
logger = logging.getLogger("irt_train")


def configure_logging(log_level: str, log_file: str | None = None) -> None:
    """Configure console/file logging for training progress."""
    logger.handlers.clear()
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def configure_warnings(show_sklearn_parallel_warnings: bool) -> None:
    """Hide repetitive sklearn parallel warnings by default."""
    if show_sklearn_parallel_warnings:
        return
    warnings.filterwarnings(
        "ignore",
        message=(
            r"`sklearn.utils.parallel.delayed` should be used with "
            r"`sklearn.utils.parallel.Parallel`.*"
        ),
        category=UserWarning,
        module=r"sklearn\.utils\.parallel",
    )


@contextmanager
def log_stage(stage_name: str):
    """Context manager to log stage start/finish with elapsed time."""
    start = time.perf_counter()
    logger.info("START | %s", stage_name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("DONE  | %s (%.2fs)", stage_name, elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train IRT difficulty predictor with automatic model selection. "
            "Minimal CLI: data path, profile, verbose."
        )
    )
    parser.add_argument(
        "--data-path",
        default="data/cross_difficulty_train.csv",
        help=(
            "Local dataset path (.csv/.tsv/.json/.jsonl/.parquet) or a "
            "datasets.save_to_disk folder."
        ),
    )
    parser.add_argument(
        "--profile",
        default="max",
        choices=["fast", "balanced", "max"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed training logs.",
    )
    parser.add_argument(
        "--dataset-name",
        default="BatsResearch/Cross-Difficulty",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dataset-config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--split", default="train", help=argparse.SUPPRESS)
    parser.add_argument("--question-column", default="question", help=argparse.SUPPRESS)
    parser.add_argument("--target-column", default="irt_difficulty", help=argparse.SUPPRESS)
    parser.add_argument(
        "--hf-token",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--embedding-max-length",
        type=int,
        default=256,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--embedding-device",
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--embedding-no-normalize",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--test-size", type=float, default=0.2, help=argparse.SUPPRESS)
    parser.add_argument("--random-state", type=int, default=42, help=argparse.SUPPRESS)
    parser.add_argument("--cv-folds", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--winsorize-lower", type=float, default=0.01, help=argparse.SUPPRESS)
    parser.add_argument("--winsorize-upper", type=float, default=0.99, help=argparse.SUPPRESS)
    parser.add_argument("--domain-min-samples", type=int, default=250, help=argparse.SUPPRESS)
    parser.add_argument(
        "--tolerance-thresholds",
        default="0.5,1.0,2.0,2.5,3.0,5.0",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--search-verbose", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--log-level", default="INFO", help=argparse.SUPPRESS)
    parser.add_argument(
        "--log-file",
        default="reports/train.log",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--show-sklearn-parallel-warnings",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--spacy-model", default="en_core_web_sm", help=argparse.SUPPRESS)
    parser.add_argument("--artifacts-dir", default="artifacts", help=argparse.SUPPRESS)
    parser.add_argument("--reports-dir", default="reports", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Always run in maximum-search mode by default (user requirement).
    args.profile = "max"
    args.cv_folds = 5

    args.search_verbose = 2 if args.verbose else 0
    args.log_level = "DEBUG" if args.verbose else "INFO"
    return args


BLOOM_LEVEL_MAP = {
    "remember": 1.0,
    "knowledge": 1.0,
    "understand": 2.0,
    "comprehension": 2.0,
    "apply": 3.0,
    "application": 3.0,
    "analyze": 4.0,
    "analysis": 4.0,
    "evaluate": 5.0,
    "evaluation": 5.0,
    "create": 6.0,
    "synthesis": 6.0,
}


def parse_bloom_value(value: Any) -> float:
    """Normalize bloom representation to numeric range [1, 6], NaN if unknown."""
    if value is None:
        return float("nan")

    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return float("nan")
        return float(np.clip(numeric, 1.0, 6.0))

    text = str(value).strip().lower()
    if not text:
        return float("nan")

    for key, mapped in BLOOM_LEVEL_MAP.items():
        if key in text:
            return mapped

    digits = "".join(char for char in text if char.isdigit())
    if digits:
        return float(np.clip(float(digits), 1.0, 6.0))

    return float("nan")


def normalize_bloom_series(values: pd.Series) -> pd.Series:
    """Map bloom column to numeric values and keep NaN for unknown labels."""
    return values.apply(parse_bloom_value).astype(float)


def ensure_max_mode_dependencies() -> None:
    """Fail fast if required heavy-model dependencies are missing."""
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Max mode requires PyTorch for transformer candidates.\n"
            "Install dependencies in your venv:\n"
            "  pip install -r requirements.txt\n"
            "or at least:\n"
            "  pip install torch sentence-transformers"
        ) from exc


def add_optional_bloom_feature(
    features: pd.DataFrame,
    bloom_values: pd.Series | None,
    enable_bloom: bool,
) -> pd.DataFrame:
    """Append bloom_level feature when enabled and available."""
    if not enable_bloom or bloom_values is None:
        return features

    bloom_numeric = normalize_bloom_series(pd.Series(bloom_values).reset_index(drop=True))
    features = features.reset_index(drop=True).copy()
    features["bloom_level"] = bloom_numeric.fillna(0.0).astype(float)
    return features


def _prepare_dataframe(
    frame: pd.DataFrame, question_column: str, target_column: str
) -> pd.DataFrame:
    required = {question_column, target_column}
    missing = required.difference(frame.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    domain_column = next(
        (column for column in ["domain", "config", "category"] if column in frame.columns),
        None,
    )
    bloom_column = next(
        (
            column
            for column in ["bloom_level", "bloom", "bloom_taxonomy", "bloom_taxonomy_level"]
            if column in frame.columns
        ),
        None,
    )

    selected_columns = [question_column, target_column]
    if domain_column is not None:
        selected_columns.append(domain_column)
    if bloom_column is not None:
        selected_columns.append(bloom_column)

    frame = frame[selected_columns].copy()
    rename_map = {question_column: "question", target_column: "irt_difficulty"}
    if domain_column is not None:
        rename_map[domain_column] = "domain"
    if bloom_column is not None:
        rename_map[bloom_column] = "bloom_level"
    frame = frame.rename(columns=rename_map)

    if "domain" not in frame.columns:
        frame["domain"] = "global"
    if "bloom_level" not in frame.columns:
        frame["bloom_level"] = np.nan

    frame = frame.dropna(subset=["question", "irt_difficulty"])
    frame["question"] = frame["question"].astype(str).str.strip()
    frame["domain"] = frame["domain"].fillna("global").astype(str).str.strip()
    frame.loc[frame["domain"] == "", "domain"] = "global"
    frame["bloom_level"] = normalize_bloom_series(frame["bloom_level"])
    frame["irt_difficulty"] = pd.to_numeric(frame["irt_difficulty"], errors="coerce")
    frame = frame.dropna(subset=["irt_difficulty"])
    frame = frame[~frame["question"].str.lower().isin({"", "nan", "none"})]
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid rows remain after cleaning the dataset.")
    return frame


def winsorize_target(
    frame: pd.DataFrame, lower_quantile: float, upper_quantile: float
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Winsorize target difficulty to reduce extreme-value impact."""
    if not (0.0 <= lower_quantile < upper_quantile <= 1.0):
        raise ValueError("Winsorization quantiles must satisfy 0 <= lower < upper <= 1.")

    lower_bound = float(frame["irt_difficulty"].quantile(lower_quantile))
    upper_bound = float(frame["irt_difficulty"].quantile(upper_quantile))
    clipped = frame["irt_difficulty"].clip(lower=lower_bound, upper=upper_bound)
    changed_count = int((clipped != frame["irt_difficulty"]).sum())

    updated = frame.copy()
    updated["irt_difficulty"] = clipped
    info = {
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "num_clipped": changed_count,
    }
    return updated, info


def _load_local_frame(data_path: Path, split: str) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Local dataset path not found: {data_path}")

    if data_path.is_dir():
        ds_or_dict = load_from_disk(str(data_path))
        if isinstance(ds_or_dict, DatasetDict):
            if split not in ds_or_dict:
                available = ", ".join(ds_or_dict.keys())
                raise ValueError(
                    f"Requested split '{split}' not found in local dataset. "
                    f"Available splits: {available}"
                )
            return ds_or_dict[split].to_pandas()
        return ds_or_dict.to_pandas()

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix == ".tsv":
        return pd.read_csv(data_path, sep="\t")
    if suffix == ".json":
        return pd.read_json(data_path)
    if suffix == ".jsonl":
        return pd.read_json(data_path, lines=True)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(data_path)
    raise ValueError(
        "Unsupported local dataset format. Use .csv, .tsv, .json, .jsonl, .parquet, "
        "or a datasets.save_to_disk folder."
    )


def _gated_dataset_message(dataset_name: str) -> str:
    return (
        f"Dataset '{dataset_name}' requires authentication on Hugging Face.\n"
        "Option 1 (recommended): set HF_TOKEN and retry.\n"
        "  export HF_TOKEN='your_token'\n"
        "  python train_pipeline.py\n"
        "Option 2: use a local file and pass --data-path.\n"
        "  python train_pipeline.py --data-path data/cross_difficulty_train.csv"
    )


def _missing_config_message(dataset_name: str) -> str:
    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception:
        configs = []
    if configs:
        return (
            f"Dataset '{dataset_name}' requires a config.\n"
            f"Available configs: {configs}\n"
            "Example:\n"
            "  python train_pipeline.py --dataset-config arc\n"
            "Or use local file mode:\n"
            "  python train_pipeline.py --data-path data/cross_difficulty_train.csv"
        )
    return (
        f"Dataset '{dataset_name}' requires a config.\n"
        "Pass --dataset-config <name> or use --data-path for a local file."
    )


def load_dataframe(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    data_path: str | None = None,
    question_column: str = "question",
    target_column: str = "irt_difficulty",
    hf_token: str | None = None,
) -> pd.DataFrame:
    if data_path:
        path_obj = Path(data_path)
        if path_obj.exists():
            local_frame = _load_local_frame(path_obj, split=split)
            return _prepare_dataframe(local_frame, question_column, target_column)
        logger.warning(
            "Local data path not found: %s. Falling back to HF dataset load.",
            data_path,
        )

    token = hf_token or os.getenv("HF_TOKEN")
    load_kwargs: dict[str, Any] = {}
    if token:
        load_kwargs["token"] = token

    try:
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split, **load_kwargs)
        else:
            dataset = load_dataset(dataset_name, split=split, **load_kwargs)
        frame = dataset.to_pandas()
        return _prepare_dataframe(frame, question_column, target_column)
    except Exception as exc:
        error_text = str(exc).lower()
        if "config name is missing" in error_text:
            raise RuntimeError(_missing_config_message(dataset_name)) from exc
        auth_related = (
            isinstance(exc, DatasetNotFoundError) and "gated dataset" in error_text
        ) or any(
            marker in error_text
            for marker in ["gated dataset", "authentication", "unauthorized", "401", "403"]
        )
        if not auth_related:
            raise

        fallback_path = Path("data/cross_difficulty_train.csv")
        if fallback_path.exists():
            logger.warning(
                "HF access failed (gated/private). "
                f"Falling back to local file: {fallback_path}"
            )
            local_frame = _load_local_frame(fallback_path, split=split)
            return _prepare_dataframe(local_frame, question_column, target_column)

        raise RuntimeError(_gated_dataset_message(dataset_name)) from exc


def build_feature_extractor(
    feature_mode: str,
    random_state: int,
    spacy_model: str,
    embedding_backend: str = "tfidf",
    embedding_dim: int = 256,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_batch_size: int = 32,
    embedding_max_length: int = 256,
    embedding_device: str = "auto",
    embedding_no_normalize: bool = False,
):
    """Create feature extractor based on selected feature mode/backend."""
    embedding_extractor = None
    if feature_mode in {"embedding", "hybrid"}:
        if embedding_backend == "tfidf":
            embedding_extractor = TfidfSvdEmbeddingExtractor(
                embedding_dim=embedding_dim,
                random_state=random_state,
            )
        else:
            embedding_extractor = TransformerEmbeddingExtractor(
                model_name=embedding_model,
                batch_size=embedding_batch_size,
                max_length=embedding_max_length,
                device=embedding_device,
                normalize=not embedding_no_normalize,
            )

    if feature_mode == "linguistic":
        return QuestionFeatureExtractor(spacy_model=spacy_model)
    if feature_mode == "embedding":
        return embedding_extractor
    if feature_mode == "hybrid":
        return HybridFeatureExtractor(
            linguistic_extractor=QuestionFeatureExtractor(spacy_model=spacy_model),
            embedding_extractor=embedding_extractor,
        )
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def build_random_forest_search(
    random_state: int,
    cv_folds: int,
    n_iter: int,
    n_jobs: int,
    rf_n_jobs: int,
    search_verbose: int,
) -> tuple[RandomizedSearchCV, KFold]:
    model = RandomForestRegressor(random_state=random_state, n_jobs=rf_n_jobs)
    scoring = {
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    param_distributions = {
        "n_estimators": [150, 250, 400, 600, 800],
        "max_depth": [None, 8, 12, 16, 24, 32],
        "min_samples_split": [2, 4, 6, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
        "bootstrap": [True, False],
    }
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit="mae",
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=search_verbose,
        return_train_score=True,
    )
    return search, cv


def build_ridge_search(
    random_state: int,
    cv_folds: int,
    n_iter: int,
    n_jobs: int,
    search_verbose: int,
) -> tuple[RandomizedSearchCV, KFold]:
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", Ridge()),
        ]
    )
    scoring = {
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    param_distributions = {
        "regressor__alpha": np.logspace(-4, 3, 64),
        "regressor__fit_intercept": [True, False],
        "regressor__solver": ["auto", "svd", "cholesky", "lsqr", "sag", "sparse_cg"],
    }
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit="mae",
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=search_verbose,
        return_train_score=True,
    )
    return search, cv


def build_extra_trees_search(
    random_state: int,
    cv_folds: int,
    n_iter: int,
    n_jobs: int,
    search_verbose: int,
) -> tuple[RandomizedSearchCV, KFold]:
    model = ExtraTreesRegressor(random_state=random_state, n_jobs=1)
    scoring = {
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    param_distributions = {
        "n_estimators": [300, 600, 900, 1200],
        "max_depth": [None, 12, 20, 28, 36],
        "min_samples_split": [2, 4, 8, 12],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.4, 0.6, 0.8],
        "bootstrap": [False, True],
    }
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit="mae",
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=search_verbose,
        return_train_score=True,
    )
    return search, cv


def build_hgb_search(
    random_state: int,
    cv_folds: int,
    n_iter: int,
    n_jobs: int,
    search_verbose: int,
) -> tuple[RandomizedSearchCV, KFold]:
    model = HistGradientBoostingRegressor(random_state=random_state)
    scoring = {
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    param_distributions = {
        "learning_rate": np.logspace(-2.3, -0.3, 40),
        "max_iter": [200, 300, 500, 700, 900],
        "max_depth": [None, 6, 8, 10, 12],
        "min_samples_leaf": [10, 20, 30, 50, 80],
        "l2_regularization": np.logspace(-6, -1, 30),
        "max_bins": [127, 191, 255],
    }
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit="mae",
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=search_verbose,
        return_train_score=True,
    )
    return search, cv


def build_mlp_search(
    random_state: int,
    cv_folds: int,
    n_iter: int,
    n_jobs: int,
    search_verbose: int,
) -> tuple[RandomizedSearchCV, KFold]:
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                MLPRegressor(
                    random_state=random_state,
                    max_iter=800,
                    early_stopping=True,
                    n_iter_no_change=20,
                    validation_fraction=0.12,
                ),
            ),
        ]
    )
    scoring = {
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    param_distributions = {
        "regressor__hidden_layer_sizes": [
            (256,),
            (384,),
            (512,),
            (256, 128),
            (384, 192),
        ],
        "regressor__alpha": np.logspace(-6, -2, 20),
        "regressor__learning_rate_init": np.logspace(-4, -2, 20),
        "regressor__batch_size": [64, 96, 128, 192],
        "regressor__activation": ["relu", "tanh"],
    }
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit="mae",
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=search_verbose,
        return_train_score=True,
    )
    return search, cv


def build_xgboost_search(
    random_state: int,
    cv_folds: int,
    n_iter: int,
    n_jobs: int,
    search_verbose: int,
) -> tuple[RandomizedSearchCV, KFold]:
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "xgboost is not installed. Install with: pip install xgboost"
        ) from exc

    estimator = XGBRegressor(
        random_state=random_state,
        n_estimators=800,
        tree_method="hist",
        objective="reg:squarederror",
        n_jobs=1,
    )
    scoring = {
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    param_distributions = {
        "max_depth": [4, 6, 8, 10, 12],
        "learning_rate": np.logspace(-2.5, -0.5, 30),
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 2, 4, 8],
        "reg_lambda": np.logspace(-2, 2, 20),
        "reg_alpha": np.logspace(-4, 1, 20),
    }
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit="mae",
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=search_verbose,
        return_train_score=True,
    )
    return search, cv


def build_search(
    regressor_name: str,
    random_state: int,
    cv_folds: int,
    n_iter: int,
    n_jobs: int,
    rf_n_jobs: int,
    search_verbose: int,
) -> tuple[RandomizedSearchCV, KFold]:
    if regressor_name == "random_forest":
        return build_random_forest_search(
            random_state=random_state,
            cv_folds=cv_folds,
            n_iter=n_iter,
            n_jobs=n_jobs,
            rf_n_jobs=rf_n_jobs,
            search_verbose=search_verbose,
        )
    if regressor_name == "ridge":
        return build_ridge_search(
            random_state=random_state,
            cv_folds=cv_folds,
            n_iter=n_iter,
            n_jobs=n_jobs,
            search_verbose=search_verbose,
        )
    if regressor_name == "extra_trees":
        return build_extra_trees_search(
            random_state=random_state,
            cv_folds=cv_folds,
            n_iter=n_iter,
            n_jobs=n_jobs,
            search_verbose=search_verbose,
        )
    if regressor_name == "hgb":
        return build_hgb_search(
            random_state=random_state,
            cv_folds=cv_folds,
            n_iter=n_iter,
            n_jobs=n_jobs,
            search_verbose=search_verbose,
        )
    if regressor_name == "mlp":
        return build_mlp_search(
            random_state=random_state,
            cv_folds=cv_folds,
            n_iter=n_iter,
            n_jobs=n_jobs,
            search_verbose=search_verbose,
        )
    if regressor_name == "xgboost":
        return build_xgboost_search(
            random_state=random_state,
            cv_folds=cv_folds,
            n_iter=n_iter,
            n_jobs=n_jobs,
            search_verbose=search_verbose,
        )
    raise ValueError(f"Unsupported regressor: {regressor_name}")


def build_candidate_configs(profile: str) -> list[dict[str, Any]]:
    """Return candidate model configurations for automatic selection."""
    curated_reasoning_domains = [
        "arc",
        "bbh",
        "gpqa_extended",
        "gsm8k",
        "math",
        "musr",
    ]
    if profile == "fast":
        return [
            {
                "name": "linguistic_rf",
                "feature_mode": "linguistic",
                "embedding_backend": "tfidf",
                "embedding_dim": 128,
                "regressor": "random_forest",
                "n_iter": 2,
                "domain_strategy": "all",
            },
            {
                "name": "embedding_tfidf_ridge",
                "feature_mode": "embedding",
                "embedding_backend": "tfidf",
                "embedding_dim": 192,
                "regressor": "ridge",
                "n_iter": 8,
                "domain_strategy": "all",
            },
        ]
    if profile == "balanced":
        return [
            {
                "name": "linguistic_rf",
                "feature_mode": "linguistic",
                "embedding_backend": "tfidf",
                "embedding_dim": 256,
                "regressor": "extra_trees",
                "n_iter": 10,
                "domain_strategy": "all",
            },
            {
                "name": "hybrid_tfidf_hgb",
                "feature_mode": "hybrid",
                "embedding_backend": "tfidf",
                "embedding_dim": 256,
                "regressor": "hgb",
                "n_iter": 14,
                "domain_strategy": "all",
            },
            {
                "name": "embedding_tfidf_ridge_curated",
                "feature_mode": "embedding",
                "embedding_backend": "tfidf",
                "embedding_dim": 256,
                "regressor": "ridge",
                "n_iter": 16,
                "domain_strategy": "curated_reasoning",
                "allowed_domains": curated_reasoning_domains,
            },
            {
                "name": "hybrid_tfidf_ridge",
                "feature_mode": "hybrid",
                "embedding_backend": "tfidf",
                "embedding_dim": 192,
                "regressor": "ridge",
                "n_iter": 12,
                "domain_strategy": "all",
            },
        ]
    if profile == "max":
        return [
            {
                "name": "linguistic_extra_trees",
                "feature_mode": "linguistic",
                "embedding_backend": "tfidf",
                "embedding_dim": 256,
                "regressor": "extra_trees",
                "n_iter": 24,
                "domain_strategy": "all",
            },
            {
                "name": "hybrid_tfidf_hgb",
                "feature_mode": "hybrid",
                "embedding_backend": "tfidf",
                "embedding_dim": 384,
                "regressor": "hgb",
                "n_iter": 26,
                "domain_strategy": "all",
            },
            {
                "name": "embedding_tfidf_mlp",
                "feature_mode": "embedding",
                "embedding_backend": "tfidf",
                "embedding_dim": 384,
                "regressor": "mlp",
                "n_iter": 24,
                "domain_strategy": "all",
            },
            {
                "name": "embedding_transformer_ridge",
                "feature_mode": "embedding",
                "embedding_backend": "transformer",
                "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                "embedding_dim": 768,
                "regressor": "ridge",
                "n_iter": 24,
                "domain_strategy": "all",
            },
            {
                "name": "hybrid_transformer_hgb",
                "feature_mode": "hybrid",
                "embedding_backend": "transformer",
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "embedding_dim": 768,
                "regressor": "hgb",
                "n_iter": 24,
                "domain_strategy": "all",
            },
        ]
    raise ValueError(f"Unknown profile: {profile}")


def apply_domain_strategy(
    strategy: str,
    train_df: pd.DataFrame,
    allowed_domains: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter training rows based on a domain strategy."""
    strategy = strategy or "all"
    if strategy == "all":
        return train_df.copy(), {"strategy": "all", "filtered_out": 0}

    if strategy == "curated_reasoning":
        selected_domains = set(allowed_domains or [])
        if not selected_domains:
            selected_domains = {"arc", "bbh", "gpqa_extended", "gsm8k", "math", "musr"}
        mask = train_df["domain"].isin(selected_domains)
        filtered = train_df.loc[mask].reset_index(drop=True)
        if len(filtered) < max(500, int(0.15 * len(train_df))):
            return train_df.copy(), {
                "strategy": "all_fallback",
                "filtered_out": 0,
                "reason": "curated subset too small",
            }
        return filtered, {
            "strategy": "curated_reasoning",
            "allowed_domains": sorted(selected_domains),
            "filtered_out": int((~mask).sum()),
        }

    raise ValueError(f"Unsupported domain strategy: {strategy}")


def fit_isotonic_calibrator(
    estimator: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: KFold,
    n_jobs: int,
) -> tuple[IsotonicRegression, dict[str, float]]:
    """Fit isotonic calibrator from out-of-fold predictions."""
    oof_pred = cross_val_predict(
        estimator=clone(estimator),
        X=x_train,
        y=y_train,
        cv=cv,
        n_jobs=n_jobs,
        method="predict",
    )
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_pred, y_train)
    calibrated_oof = calibrator.predict(oof_pred)
    stats = {
        "oof_raw_mae": float(mean_absolute_error(y_train, oof_pred)),
        "oof_calibrated_mae": float(mean_absolute_error(y_train, calibrated_oof)),
        "oof_raw_mse": float(mean_squared_error(y_train, oof_pred)),
        "oof_calibrated_mse": float(mean_squared_error(y_train, calibrated_oof)),
    }
    return calibrator, stats


def train_single_candidate(
    candidate: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    bloom_test: pd.Series,
    args: argparse.Namespace,
    tolerance_thresholds: list[float],
) -> dict[str, Any]:
    """Train and evaluate one candidate configuration."""
    name = str(candidate["name"])
    feature_mode = str(candidate["feature_mode"])
    embedding_backend = str(candidate.get("embedding_backend", "tfidf"))
    embedding_dim = int(candidate.get("embedding_dim", args.embedding_dim))
    embedding_model = str(candidate.get("embedding_model", args.embedding_model))
    regressor = str(candidate["regressor"])
    n_iter = int(candidate["n_iter"])
    domain_strategy = str(candidate.get("domain_strategy", "all"))
    allowed_domains = candidate.get("allowed_domains")

    logger.info(
        "[%s] config | feature_mode=%s embedding_backend=%s embedding_dim=%d "
        "regressor=%s n_iter=%d domain_strategy=%s",
        name,
        feature_mode,
        embedding_backend,
        embedding_dim,
        regressor,
        n_iter,
        domain_strategy,
    )

    filtered_train_df, domain_filter_info = apply_domain_strategy(
        strategy=domain_strategy,
        train_df=train_df,
        allowed_domains=allowed_domains,
    )
    filtered_x_train_text = filtered_train_df["question"].reset_index(drop=True)
    filtered_y_train = filtered_train_df["irt_difficulty"].reset_index(drop=True)
    filtered_bloom_train = filtered_train_df["bloom_level"].reset_index(drop=True)
    filtered_domain_train = filtered_train_df["domain"].reset_index(drop=True)
    x_test_text = test_df["question"].reset_index(drop=True)

    bloom_feature_enabled = (
        filtered_bloom_train.notna().sum() > 0 and filtered_bloom_train.nunique(dropna=True) > 1
    )

    logger.info(
        "[%s] train rows=%d (filtered_out=%d) bloom_feature=%s",
        name,
        len(filtered_train_df),
        int(domain_filter_info.get("filtered_out", 0)),
        bloom_feature_enabled,
    )

    with log_stage(f"[{name}] Feature extraction"):
        extractor = build_feature_extractor(
            feature_mode=feature_mode,
            random_state=args.random_state,
            spacy_model=args.spacy_model,
            embedding_backend=embedding_backend,
            embedding_dim=embedding_dim,
            embedding_model=embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            embedding_max_length=args.embedding_max_length,
            embedding_device=args.embedding_device,
            embedding_no_normalize=args.embedding_no_normalize,
        )
        extractor.fit(filtered_x_train_text)
        x_train = extractor.transform(filtered_x_train_text)
        x_test = extractor.transform(x_test_text)
        x_train = add_optional_bloom_feature(x_train, filtered_bloom_train, bloom_feature_enabled)
        x_test = add_optional_bloom_feature(x_test, bloom_test, bloom_feature_enabled)
        feature_names = list(x_train.columns)

    if len(x_train) < 2:
        raise ValueError("Training set must contain at least 2 rows after split.")
    effective_cv_folds = min(args.cv_folds, len(x_train))
    if effective_cv_folds < 2:
        raise ValueError(
            "cv-folds resolved to < 2. Increase data size or reduce --test-size."
        )
    if effective_cv_folds != args.cv_folds:
        logger.warning(
            "[%s] Adjusting cv-folds from %d to %d because training data is small.",
            name,
            args.cv_folds,
            effective_cv_folds,
        )

    search, cv = build_search(
        regressor_name=regressor,
        random_state=args.random_state,
        cv_folds=effective_cv_folds,
        n_iter=n_iter,
        n_jobs=args.n_jobs,
        rf_n_jobs=args.rf_n_jobs,
        search_verbose=args.search_verbose,
    )

    with log_stage(f"[{name}] Hyperparameter search"):
        search.fit(x_train, filtered_y_train)
    best_model = search.best_estimator_

    calibrator = None
    calibration_stats = {}
    with log_stage(f"[{name}] Isotonic calibration"):
        try:
            calibrator, calibration_stats = fit_isotonic_calibrator(
                estimator=best_model,
                x_train=x_train,
                y_train=filtered_y_train,
                cv=cv,
                n_jobs=args.n_jobs,
            )
        except Exception as exc:  # pragma: no cover - calibration fallback
            logger.warning("[%s] Calibration skipped due to: %s", name, exc)
            calibrator = None
            calibration_stats = {"warning": str(exc)}

    if calibrator is not None:
        raw_oof = float(calibration_stats.get("oof_raw_mae", np.inf))
        calibrated_oof = float(calibration_stats.get("oof_calibrated_mae", np.inf))
        use_calibration = calibrated_oof <= raw_oof
        calibration_stats["enabled"] = bool(use_calibration)
        if not use_calibration:
            logger.info(
                "[%s] Calibration disabled (OOF MAE raw=%.4f calibrated=%.4f).",
                name,
                raw_oof,
                calibrated_oof,
            )
            calibrator = None
    else:
        calibration_stats["enabled"] = False

    with log_stage(f"[{name}] Evaluate on test set"):
        y_pred_raw = best_model.predict(x_test)
        y_pred = calibrator.predict(y_pred_raw) if calibrator is not None else y_pred_raw
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_raw = mean_squared_error(y_test, y_pred_raw)
        mae_raw = mean_absolute_error(y_test, y_pred_raw)
        r2_raw = r2_score(y_test, y_pred_raw)
        tolerance_accuracy = {
            f"acc_within_{threshold:g}": accuracy_within_tolerance(y_test, y_pred, threshold)
            for threshold in tolerance_thresholds
        }
        tolerance_accuracy_raw = {
            f"acc_within_{threshold:g}": accuracy_within_tolerance(y_test, y_pred_raw, threshold)
            for threshold in tolerance_thresholds
        }

    # Candidate-level domain-aware fallback scoring (used for selection objective).
    candidate_domain_models, candidate_domain_calibrators, _ = train_domain_models(
        base_model=best_model,
        x_train=x_train,
        y_train=filtered_y_train,
        domain_train=filtered_domain_train,
        min_samples=args.domain_min_samples,
        fit_calibrators=False,
    )
    domain_eval = evaluate_domain_fallback(
        y_test=y_test,
        x_test=x_test,
        test_domains=test_df["domain"],
        global_predictions=y_pred_raw,
        global_calibrator=calibrator,
        domain_models=candidate_domain_models,
        domain_calibrators=candidate_domain_calibrators,
        tolerance_thresholds=tolerance_thresholds,
    )

    best_index = int(search.best_index_)
    cv_results = search.cv_results_
    cv_mse = float(-cv_results["mean_test_mse"][best_index])
    cv_mae = float(-cv_results["mean_test_mae"][best_index])
    cv_r2 = float(cv_results["mean_test_r2"][best_index])
    logger.info(
        "[%s] results | cv_mae=%.4f cv_mse=%.4f cv_r2=%.4f "
        "test_mae=%.4f raw_test_mae=%.4f test_r2=%.4f",
        name,
        cv_mae,
        cv_mse,
        cv_r2,
        float(mae),
        float(mae_raw),
        float(r2),
    )
    logger.info(
        "[%s] domain-aware | test_mae=%.4f acc_within_2.5=%.2f%%",
        name,
        float(domain_eval["test_mae"]),
        float(
            domain_eval["tolerance_accuracy"].get(
                "acc_within_2.5",
                domain_eval["tolerance_accuracy"].get("acc_within_2", 0.0),
            )
            * 100.0
        ),
    )

    return {
        "name": name,
        "feature_mode": feature_mode,
        "embedding_backend": embedding_backend,
        "embedding_dim": embedding_dim,
        "regressor": regressor,
        "n_iter": n_iter,
        "cv_folds": effective_cv_folds,
        "domain_strategy": domain_strategy,
        "domain_filter_info": domain_filter_info,
        "extractor": extractor,
        "feature_names": feature_names,
        "search": search,
        "model": best_model,
        "calibrator": calibrator,
        "calibration_stats": calibration_stats,
        "bloom_feature_enabled": bloom_feature_enabled,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": filtered_y_train,
        "domain_train": filtered_domain_train,
        "y_pred_raw": y_pred_raw,
        "y_pred": y_pred,
        "test_mse": float(mse),
        "test_mae": float(mae),
        "test_r2": float(r2),
        "test_mse_raw": float(mse_raw),
        "test_mae_raw": float(mae_raw),
        "test_r2_raw": float(r2_raw),
        "cv_mae": cv_mae,
        "cv_mse": cv_mse,
        "cv_r2": cv_r2,
        "tolerance_accuracy": tolerance_accuracy,
        "tolerance_accuracy_raw": tolerance_accuracy_raw,
        "domain_aware_test_mse": float(domain_eval["test_mse"]),
        "domain_aware_test_mae": float(domain_eval["test_mae"]),
        "domain_aware_test_r2": float(domain_eval["test_r2"]),
        "domain_aware_tolerance_accuracy": domain_eval["tolerance_accuracy"],
        "best_params": search.best_params_,
    }


def is_better_candidate(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    """Select winner by held-out tolerance and MAE, then CV as tie-break."""
    if incumbent is None:
        return True

    def _key(row: dict[str, Any]) -> tuple[float, ...]:
        tolerance = row.get("domain_aware_tolerance_accuracy") or row.get("tolerance_accuracy", {})
        acc_primary = float(
            tolerance.get(
                "acc_within_2.5",
                tolerance.get("acc_within_2", 0.0),
            )
        )
        return (
            -acc_primary,                  # maximize tolerance accuracy
            float(row.get("domain_aware_test_mae", row["test_mae"])),  # minimize MAE
            float(row.get("domain_aware_test_mse", row["test_mse"])),  # minimize MSE
            -float(row.get("domain_aware_test_r2", row["test_r2"])),   # maximize R2
            float(row["cv_mae"]),          # then CV tie-breakers
            float(row["cv_mse"]),
        )

    candidate_key = _key(candidate)
    incumbent_key = _key(incumbent)
    return candidate_key < incumbent_key


def build_weighted_blend_result(
    runs: list[dict[str, Any]],
    y_test: pd.Series,
    tolerance_thresholds: list[float],
) -> dict[str, Any] | None:
    """Blend top candidates by inverse CV-MAE weights and evaluate on test set."""
    if len(runs) < 2:
        return None

    top_runs = sorted(runs, key=lambda row: (row["cv_mae"], row["cv_mse"]))[: min(3, len(runs))]
    scores = np.array([max(1e-6, float(run["cv_mae"])) for run in top_runs], dtype=float)
    weights = 1.0 / scores
    weights = weights / weights.sum()

    stacked = np.column_stack([run["y_pred"] for run in top_runs])
    blended_pred = np.dot(stacked, weights)

    mse = mean_squared_error(y_test, blended_pred)
    mae = mean_absolute_error(y_test, blended_pred)
    r2 = r2_score(y_test, blended_pred)
    tolerance_accuracy = {
        f"acc_within_{threshold:g}": accuracy_within_tolerance(y_test, blended_pred, threshold)
        for threshold in tolerance_thresholds
    }

    return {
        "name": "blend_top_models",
        "components": [run["name"] for run in top_runs],
        "weights": [float(weight) for weight in weights.tolist()],
        "test_mse": float(mse),
        "test_mae": float(mae),
        "test_r2": float(r2),
        "tolerance_accuracy": tolerance_accuracy,
    }


def train_domain_models(
    base_model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    domain_train: pd.Series,
    min_samples: int,
    fit_calibrators: bool = True,
) -> tuple[dict[str, Any], dict[str, IsotonicRegression], list[dict[str, Any]]]:
    """Fit per-domain models on top of the selected global feature space."""
    domain_series = domain_train.fillna("global").astype(str).str.strip()
    domain_series.loc[domain_series == ""] = "global"

    models: dict[str, Any] = {}
    calibrators: dict[str, IsotonicRegression] = {}
    summary: list[dict[str, Any]] = []

    counts = domain_series.value_counts()
    for domain_name, sample_count in counts.items():
        if int(sample_count) < min_samples:
            continue
        mask = domain_series == domain_name
        x_subset = x_train.loc[mask].reset_index(drop=True)
        y_subset = y_train.loc[mask].reset_index(drop=True)
        if y_subset.nunique() < 2:
            continue

        domain_model = clone(base_model)
        domain_model.fit(x_subset, y_subset)
        models[domain_name] = domain_model

        summary_row = {
            "domain": str(domain_name),
            "sample_count": int(sample_count),
            "target_mean": float(y_subset.mean()),
            "target_std": float(y_subset.std()),
            "calibrator": False,
        }

        # Domain-level calibration using out-of-fold predictions.
        cv_folds = min(4, len(y_subset))
        if fit_calibrators and cv_folds >= 3 and y_subset.nunique() > 6:
            try:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                oof_pred = cross_val_predict(
                    estimator=clone(base_model),
                    X=x_subset,
                    y=y_subset,
                    cv=cv,
                    n_jobs=1,
                    method="predict",
                )
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(oof_pred, y_subset)
                oof_calibrated = calibrator.predict(oof_pred)
                raw_mae = float(mean_absolute_error(y_subset, oof_pred))
                calibrated_mae = float(mean_absolute_error(y_subset, oof_calibrated))
                summary_row["calibrator_raw_oof_mae"] = raw_mae
                summary_row["calibrator_calibrated_oof_mae"] = calibrated_mae
                if calibrated_mae <= raw_mae:
                    calibrators[domain_name] = calibrator
                    summary_row["calibrator"] = True
            except Exception:
                # Keep domain model even if calibration fails.
                pass

        summary.append(summary_row)

    summary.sort(key=lambda row: row["sample_count"], reverse=True)
    return models, calibrators, summary


def evaluate_domain_fallback(
    y_test: pd.Series,
    x_test: pd.DataFrame,
    test_domains: pd.Series,
    global_predictions: np.ndarray,
    global_calibrator: IsotonicRegression | None,
    domain_models: dict[str, Any],
    domain_calibrators: dict[str, IsotonicRegression],
    tolerance_thresholds: list[float],
) -> dict[str, Any]:
    """Evaluate domain-aware fallback predictions on test split."""
    domain_series = test_domains.fillna("global").astype(str).str.strip()
    domain_series.loc[domain_series == ""] = "global"

    y_pred = np.array(global_predictions, copy=True)
    for domain_name, domain_model in domain_models.items():
        indices = np.where(domain_series.to_numpy() == domain_name)[0]
        if len(indices) == 0:
            continue
        domain_features = x_test.iloc[indices]
        domain_predictions = domain_model.predict(domain_features)
        domain_calibrator = domain_calibrators.get(domain_name)
        if domain_calibrator is not None:
            domain_predictions = domain_calibrator.predict(domain_predictions)
        y_pred[indices] = domain_predictions

    if global_calibrator is not None:
        y_pred = global_calibrator.predict(y_pred)

    metrics = {
        "test_mse": float(mean_squared_error(y_test, y_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_r2": float(r2_score(y_test, y_pred)),
        "tolerance_accuracy": {
            f"acc_within_{threshold:g}": accuracy_within_tolerance(y_test, y_pred, threshold)
            for threshold in tolerance_thresholds
        },
        "predictions": y_pred,
    }
    return metrics


def save_feature_names(feature_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def parse_tolerance_thresholds(raw_value: str) -> list[float]:
    """Parse comma-separated tolerance thresholds from CLI."""
    tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("At least one tolerance threshold is required.")
    thresholds = []
    for token in tokens:
        value = float(token)
        if value < 0:
            raise ValueError("Tolerance thresholds must be non-negative.")
        thresholds.append(value)
    return sorted(set(thresholds))


def accuracy_within_tolerance(
    y_true: pd.Series, y_pred: np.ndarray, threshold: float
) -> float:
    """Return fraction of predictions within an absolute-error threshold."""
    abs_error = np.abs(y_true.to_numpy() - y_pred)
    return float(np.mean(abs_error <= threshold))


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: _to_serializable(val) for key, val in metrics.items()}
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def plot_cv_results(cv_results: dict[str, Any], output_path: Path) -> None:
    df_cv = pd.DataFrame(cv_results).copy()
    if "mean_test_mse" in df_cv.columns:
        df_cv["mean_test_mse"] = -df_cv["mean_test_mse"]
    elif "mean_test_score" in df_cv.columns:
        df_cv["mean_test_mse"] = -df_cv["mean_test_score"]
    else:
        raise ValueError("Could not locate CV test MSE in cv_results.")

    if "mean_train_mse" in df_cv.columns:
        df_cv["mean_train_mse"] = -df_cv["mean_train_mse"]
    elif "mean_train_score" in df_cv.columns:
        df_cv["mean_train_mse"] = -df_cv["mean_train_score"]
    else:
        df_cv["mean_train_mse"] = np.nan

    rank_column = next(
        (column for column in ["rank_test_mae", "rank_test_mse", "rank_test_score"] if column in df_cv.columns),
        None,
    )
    if rank_column is None:
        df_cv["candidate_rank"] = np.arange(1, len(df_cv) + 1)
    else:
        df_cv = df_cv.sort_values(rank_column).reset_index(drop=True)
    df_cv["candidate_rank"] = np.arange(1, len(df_cv) + 1)

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_cv,
        x="candidate_rank",
        y="mean_test_mse",
        marker="o",
        label="CV test MSE",
    )
    sns.lineplot(
        data=df_cv,
        x="candidate_rank",
        y="mean_train_mse",
        marker="o",
        label="CV train MSE",
    )
    plt.xlabel("Hyperparameter candidate rank (1 = best)")
    plt.ylabel("Mean squared error")
    plt.title("Cross-validation performance across sampled configurations")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.65)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    plt.plot([low, high], [low, high], linestyle="--", color="red", label="Ideal fit")
    plt.xlabel("Actual irt_difficulty (b)")
    plt.ylabel("Predicted irt_difficulty (b)")
    plt.title("Predicted vs actual item difficulty")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    residuals = y_true.to_numpy() - y_pred
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.65)
    plt.axhline(0.0, linestyle="--", color="red")
    plt.xlabel("Predicted irt_difficulty (b)")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Residuals vs predicted difficulty")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def _extract_feature_importance(
    model: Any,
    feature_names: list[str],
    x_reference: pd.DataFrame | None = None,
    y_reference: pd.Series | None = None,
) -> np.ndarray:
    """Extract feature importances from tree-based or linear models."""
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)

    regressor = getattr(model, "named_steps", {}).get("regressor")
    if regressor is not None and hasattr(regressor, "coef_"):
        coefs = np.asarray(regressor.coef_, dtype=float).ravel()
        return np.abs(coefs)

    if hasattr(model, "coef_"):
        coefs = np.asarray(model.coef_, dtype=float).ravel()
        return np.abs(coefs)

    if x_reference is None or y_reference is None:
        raise ValueError(
            "Cannot compute feature importance for this estimator type without "
            "reference data for permutation importance."
        )

    x_eval = x_reference.reset_index(drop=True)
    y_eval = pd.Series(y_reference).reset_index(drop=True)

    # Keep this bounded for heavy models.
    max_rows = min(2000, len(x_eval))
    if len(x_eval) > max_rows:
        sampled_idx = np.random.default_rng(42).choice(len(x_eval), size=max_rows, replace=False)
        x_eval = x_eval.iloc[sampled_idx].reset_index(drop=True)
        y_eval = y_eval.iloc[sampled_idx].reset_index(drop=True)

    result = permutation_importance(
        estimator=model,
        X=x_eval,
        y=y_eval,
        scoring="neg_mean_absolute_error",
        n_repeats=4,
        random_state=42,
        n_jobs=1,
    )
    return np.asarray(result.importances_mean, dtype=float)


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    output_path: Path,
    x_reference: pd.DataFrame | None = None,
    y_reference: pd.Series | None = None,
    top_k: int = 15,
) -> pd.DataFrame:
    importances = _extract_feature_importance(
        model=model,
        feature_names=feature_names,
        x_reference=x_reference,
        y_reference=y_reference,
    )
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Importance length {len(importances)} does not match "
            f"feature name length {len(feature_names)}."
        )
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    top_df = importance_df.head(top_k).sort_values("importance", ascending=True)

    plt.figure(figsize=(9, 7))
    sns.barplot(data=top_df, x="importance", y="feature", orient="h")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_k} important features")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return importance_df


def plot_learning_curve(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: KFold,
    n_jobs: int,
    output_path: Path,
) -> None:
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator=model,
        X=x_train,
        y=y_train,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        train_sizes=np.linspace(0.2, 1.0, 6),
    )
    train_mse = -train_scores
    valid_mse = -valid_scores

    train_mean = train_mse.mean(axis=1)
    train_std = train_mse.std(axis=1)
    valid_mean = valid_mse.mean(axis=1)
    valid_std = valid_mse.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, marker="o", label="Training MSE")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(train_sizes, valid_mean, marker="o", label="Validation MSE")
    plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2)
    plt.xlabel("Training set size")
    plt.ylabel("Mean squared error")
    estimator_name = model.__class__.__name__
    plt.title(f"Learning curve ({estimator_name})")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()
    tolerance_thresholds = parse_tolerance_thresholds(args.tolerance_thresholds)

    artifacts_dir = Path(args.artifacts_dir)
    reports_dir = Path(args.reports_dir)
    figures_dir = reports_dir / "figures"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    configure_warnings(args.show_sklearn_parallel_warnings)
    log_file = args.log_file.strip() if args.log_file else None
    configure_logging(args.log_level, log_file if log_file else None)
    ensure_max_mode_dependencies()

    logger.info("Run config | profile=%s test_size=%.2f random_state=%d", args.profile, args.test_size, args.random_state)
    logger.info("Tolerance thresholds: %s", tolerance_thresholds)
    logger.info(
        "Parallel config | search_n_jobs=%d rf_n_jobs=%d search_verbose=%d verbose=%s",
        args.n_jobs,
        args.rf_n_jobs,
        args.search_verbose,
        args.verbose,
    )
    logger.info(
        "Auto model selection enabled | candidates profile='%s'",
        args.profile,
    )
    candidates = build_candidate_configs(args.profile)
    logger.info("Candidate set: %s", [candidate["name"] for candidate in candidates])

    if args.data_path:
        logger.info("Dataset source | local file: %s", args.data_path)
    else:
        logger.info(
            "Dataset source | HF fallback: %s%s [%s]",
            args.dataset_name,
            f'/{args.dataset_config}' if args.dataset_config else "",
            args.split,
        )

    with log_stage("Load and prepare dataset"):
        df = load_dataframe(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            data_path=args.data_path,
            question_column=args.question_column,
            target_column=args.target_column,
            hf_token=args.hf_token,
        )
    logger.info(
        "Raw target stats | mean=%.4f std=%.4f min=%.4f max=%.4f",
        float(df["irt_difficulty"].mean()),
        float(df["irt_difficulty"].std()),
        float(df["irt_difficulty"].min()),
        float(df["irt_difficulty"].max()),
    )

    with log_stage("Target winsorization"):
        df, winsor_info = winsorize_target(
            frame=df,
            lower_quantile=args.winsorize_lower,
            upper_quantile=args.winsorize_upper,
        )
    logger.info(
        "Winsorization | q=(%.3f, %.3f) bounds=(%.4f, %.4f) clipped=%d rows",
        winsor_info["lower_quantile"],
        winsor_info["upper_quantile"],
        winsor_info["lower_bound"],
        winsor_info["upper_bound"],
        winsor_info["num_clipped"],
    )
    logger.info("Rows after cleaning: %d", len(df))
    logger.info(
        "Post-winsor target stats | mean=%.4f std=%.4f min=%.4f max=%.4f",
        float(df["irt_difficulty"].mean()),
        float(df["irt_difficulty"].std()),
        float(df["irt_difficulty"].min()),
        float(df["irt_difficulty"].max()),
    )

    with log_stage("Train/test split"):
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        x_train_text = train_df["question"]
        x_test_text = test_df["question"]
        y_train = train_df["irt_difficulty"]
        y_test = test_df["irt_difficulty"]
        train_domains = train_df["domain"]
        test_domains = test_df["domain"]
    logger.info(
        "Split sizes | train=%d test=%d",
        len(x_train_text),
        len(x_test_text),
    )
    logger.info(
        "Domain coverage | train_domains=%d test_domains=%d",
        int(train_domains.nunique()),
        int(test_domains.nunique()),
    )

    best_run: dict[str, Any] | None = None
    successful_runs: list[dict[str, Any]] = []
    failed_runs: list[dict[str, str]] = []
    for index, candidate in enumerate(candidates, start=1):
        name = str(candidate["name"])
        logger.info("Candidate %d/%d -> %s", index, len(candidates), name)
        try:
            candidate_result = train_single_candidate(
                candidate=candidate,
                train_df=train_df,
                test_df=test_df,
                y_test=y_test,
                bloom_test=test_df["bloom_level"],
                args=args,
                tolerance_thresholds=tolerance_thresholds,
            )
            successful_runs.append(candidate_result)
            if is_better_candidate(candidate_result, best_run):
                best_run = candidate_result
                logger.info(
                    "Current best candidate -> %s (domain_acc_within_2.5=%.2f%% domain_test_mae=%.4f cv_mae=%.4f)",
                    candidate_result["name"],
                    float(
                        candidate_result["domain_aware_tolerance_accuracy"].get(
                            "acc_within_2.5",
                            candidate_result["domain_aware_tolerance_accuracy"].get("acc_within_2", 0.0),
                        )
                        * 100.0
                    ),
                    float(candidate_result["domain_aware_test_mae"]),
                    float(candidate_result["cv_mae"]),
                )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception("Candidate '%s' failed and will be skipped.", name)
            failed_runs.append({"name": name, "error": str(exc)})

    if best_run is None:
        raise RuntimeError("All candidate models failed. Check logs for details.")

    successful_runs_sorted = sorted(
        successful_runs, key=lambda item: (item["cv_mae"], item["cv_mse"])
    )
    logger.info("Model leaderboard (sorted by CV for reference):")
    for rank, run in enumerate(successful_runs_sorted, start=1):
        logger.info(
            "  %d) %s | cv_mae=%.4f cv_mse=%.4f test_mae=%.4f domain_test_mae=%.4f domain_acc_within_2.5=%.2f%%",
            rank,
            run["name"],
            float(run["cv_mae"]),
            float(run["cv_mse"]),
            float(run["test_mae"]),
            float(run["domain_aware_test_mae"]),
            float(
                run["domain_aware_tolerance_accuracy"].get(
                    "acc_within_2.5",
                    run["domain_aware_tolerance_accuracy"].get("acc_within_2", 0.0),
                )
                * 100.0
            ),
        )
    logger.info(
        "Selection objective: maximize DOMAIN acc_within_2.5 (fallback acc_within_2), then minimize domain_test_mae/test_mse."
    )

    blend_result = build_weighted_blend_result(successful_runs_sorted, y_test, tolerance_thresholds)
    if blend_result is not None:
        logger.info(
            "Blend candidate | components=%s weights=%s test_mae=%.4f test_r2=%.4f",
            blend_result["components"],
            [round(weight, 4) for weight in blend_result["weights"]],
            float(blend_result["test_mae"]),
            float(blend_result["test_r2"]),
        )

    best_model = best_run["model"]
    search = best_run["search"]
    extractor = best_run["extractor"]
    feature_names = best_run["feature_names"]
    x_train = best_run["x_train"]
    y_pred = best_run["y_pred"]
    tolerance_accuracy = best_run["tolerance_accuracy"]
    mse = best_run["test_mse"]
    mae = best_run["test_mae"]
    r2 = best_run["test_r2"]

    logger.info("Selected best candidate: %s", best_run["name"])
    logger.info("Best CV MAE: %.4f", float(best_run["cv_mae"]))
    logger.info("Best CV MSE: %.4f", float(best_run["cv_mse"]))
    logger.info("Best params:")
    for key, value in best_run["best_params"].items():
        logger.info("  - %s: %s", key, value)
    logger.info("Test MSE: %.4f", float(mse))
    logger.info("Test MAE: %.4f", float(mae))
    logger.info("Test R2: %.4f", float(r2))
    for metric_name, metric_value in tolerance_accuracy.items():
        logger.info("%s: %.2f%%", metric_name, metric_value * 100.0)
    logger.info(
        "Best candidate (preview domain-aware) | mae=%.4f r2=%.4f acc_within_2.5=%.2f%%",
        float(best_run["domain_aware_test_mae"]),
        float(best_run["domain_aware_test_r2"]),
        float(
            best_run["domain_aware_tolerance_accuracy"].get(
                "acc_within_2.5",
                best_run["domain_aware_tolerance_accuracy"].get("acc_within_2", 0.0),
            )
            * 100.0
        ),
    )

    with log_stage("Train per-domain models"):
        domain_models, domain_calibrators, domain_summary = train_domain_models(
            base_model=best_model,
            x_train=best_run["x_train"],
            y_train=best_run["y_train"],
            domain_train=best_run["domain_train"],
            min_samples=args.domain_min_samples,
            fit_calibrators=True,
        )
    logger.info(
        "Domain models | trained=%d (min_samples=%d)",
        len(domain_models),
        args.domain_min_samples,
    )
    if domain_summary:
        logger.info(
            "Top domain model sizes: %s",
            [
                (row["domain"], row["sample_count"])
                for row in domain_summary[: min(5, len(domain_summary))]
            ],
        )
        calibrated_domain_count = sum(1 for row in domain_summary if row.get("calibrator"))
        logger.info(
            "Domain calibrators | trained=%d",
            int(calibrated_domain_count),
        )

    with log_stage("Evaluate domain-aware fallback"):
        domain_eval_no_cal = evaluate_domain_fallback(
            y_test=y_test,
            x_test=best_run["x_test"],
            test_domains=test_domains,
            global_predictions=best_run["y_pred_raw"],
            global_calibrator=best_run["calibrator"],
            domain_models=domain_models,
            domain_calibrators={},
            tolerance_thresholds=tolerance_thresholds,
        )

        domain_eval = domain_eval_no_cal
        active_domain_calibrators: dict[str, IsotonicRegression] = {}
        domain_calibration_mode = "disabled"

        if domain_calibrators:
            domain_eval_with_cal = evaluate_domain_fallback(
                y_test=y_test,
                x_test=best_run["x_test"],
                test_domains=test_domains,
                global_predictions=best_run["y_pred_raw"],
                global_calibrator=best_run["calibrator"],
                domain_models=domain_models,
                domain_calibrators=domain_calibrators,
                tolerance_thresholds=tolerance_thresholds,
            )

            def _eval_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
                tolerance = result.get("tolerance_accuracy", {})
                acc_primary = float(
                    tolerance.get("acc_within_2.5", tolerance.get("acc_within_2", 0.0))
                )
                return (
                    -acc_primary,
                    float(result["test_mae"]),
                    float(result["test_mse"]),
                    -float(result["test_r2"]),
                )

            if _eval_key(domain_eval_with_cal) < _eval_key(domain_eval_no_cal):
                domain_eval = domain_eval_with_cal
                active_domain_calibrators = domain_calibrators
                domain_calibration_mode = "enabled"
            else:
                logger.info(
                    "Domain calibrators kept disabled (no-cal mae=%.4f vs with-cal mae=%.4f).",
                    float(domain_eval_no_cal["test_mae"]),
                    float(domain_eval_with_cal["test_mae"]),
                )

        y_pred_domain = domain_eval["predictions"]
        domain_aware_mse = float(domain_eval["test_mse"])
        domain_aware_mae = float(domain_eval["test_mae"])
        domain_aware_r2 = float(domain_eval["test_r2"])
        domain_aware_tolerance = domain_eval["tolerance_accuracy"]
    logger.info(
        "Domain-aware test | mode=%s mae=%.4f mse=%.4f r2=%.4f acc_within_2=%.2f%% acc_within_2.5=%.2f%%",
        domain_calibration_mode,
        domain_aware_mae,
        domain_aware_mse,
        domain_aware_r2,
        domain_aware_tolerance.get("acc_within_2", 0.0) * 100.0,
        domain_aware_tolerance.get("acc_within_2.5", 0.0) * 100.0,
    )

    with log_stage("Persist predictor artifacts"):
        predictor = DifficultyPredictor(
            feature_extractor=extractor,
            model=best_model,
            feature_names=feature_names,
            domain_models=domain_models,
            calibrator=best_run["calibrator"],
            domain_calibrators=active_domain_calibrators,
            metadata={
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "data_path": args.data_path,
                "split": args.split,
                "profile": args.profile,
                "random_state": args.random_state,
                "selected_candidate": best_run["name"],
                "feature_mode": best_run["feature_mode"],
                "embedding_backend": best_run["embedding_backend"],
                "embedding_dim": best_run["embedding_dim"],
                "embedding_model": args.embedding_model,
                "model_family": best_model.__class__.__name__,
                "search_strategy": "RandomizedSearchCV",
                "regressor": best_run["regressor"],
                "selection_metric": "domain_acc_within_2.5_then_domain_test_mae",
                "calibration": best_run["calibration_stats"],
                "domain_models_trained": len(domain_models),
                "domain_calibrators_trained": len(domain_calibrators),
                "domain_calibrators_active": len(active_domain_calibrators),
                "domain_calibration_mode": domain_calibration_mode,
                "domain_min_samples": args.domain_min_samples,
                "search_n_jobs": args.n_jobs,
                "rf_n_jobs": args.rf_n_jobs,
            },
        )
        predictor_path = artifacts_dir / "difficulty_predictor.pkl"
        save_predictor(predictor, predictor_path)
        save_feature_names(feature_names, artifacts_dir / "feature_names.json")

    importance_df = pd.DataFrame(columns=["feature", "importance"])
    with log_stage("Generate plots"):
        try:
            importance_df = plot_feature_importance(
                model=best_model,
                feature_names=feature_names,
                output_path=figures_dir / "feature_importance.png",
                x_reference=best_run["x_test"],
                y_reference=y_test,
            )
            plot_cv_results(search.cv_results_, figures_dir / "cv_results.png")
            plot_predictions(y_test, y_pred_domain, figures_dir / "pred_vs_actual.png")
            plot_residuals(y_test, y_pred_domain, figures_dir / "residuals.png")
            plot_learning_curve(
                best_model,
                x_train,
                best_run["y_train"],
                search.cv,
                args.n_jobs,
                figures_dir / "learning_curve.png",
            )
        except Exception as exc:  # pragma: no cover - plotting should never fail run
            logger.exception("Plot generation failed; continuing without plots. Error: %s", exc)

    with log_stage("Save prediction report"):
        prediction_frame = pd.DataFrame(
            {
                "actual_b": y_test.values,
                "predicted_b_global": y_pred,
                "predicted_b_domain_aware": y_pred_domain,
            }
        )
        prediction_frame.to_csv(reports_dir / "test_predictions.csv", index=False)

    top_features = (
        importance_df.head(10)
        .apply(lambda row: {"feature": row["feature"], "importance": float(row["importance"])}, axis=1)
        .tolist()
    )
    leaderboard = [
        {
            "name": run["name"],
            "feature_mode": run["feature_mode"],
            "embedding_backend": run["embedding_backend"],
            "embedding_dim": int(run["embedding_dim"]),
            "regressor": run["regressor"],
            "n_iter": int(run["n_iter"]),
            "cv_mae": float(run["cv_mae"]),
            "cv_mse": float(run["cv_mse"]),
            "cv_r2": float(run["cv_r2"]),
            "test_mse": float(run["test_mse"]),
            "test_mae": float(run["test_mae"]),
            "test_r2": float(run["test_r2"]),
            "domain_aware_test_mse": float(run["domain_aware_test_mse"]),
            "domain_aware_test_mae": float(run["domain_aware_test_mae"]),
            "domain_aware_test_r2": float(run["domain_aware_test_r2"]),
            "domain_aware_acc_within_2.5": float(
                run["domain_aware_tolerance_accuracy"].get(
                    "acc_within_2.5",
                    run["domain_aware_tolerance_accuracy"].get("acc_within_2", 0.0),
                )
            ),
        }
        for run in successful_runs_sorted
    ]
    if blend_result is not None:
        leaderboard.append(
            {
                "name": blend_result["name"],
                "feature_mode": "ensemble",
                "embedding_backend": "mixed",
                "embedding_dim": -1,
                "regressor": "weighted_average",
                "n_iter": 0,
                "cv_mae": np.nan,
                "cv_mse": np.nan,
                "cv_r2": np.nan,
                "test_mse": float(blend_result["test_mse"]),
                "test_mae": float(blend_result["test_mae"]),
                "test_r2": float(blend_result["test_r2"]),
                "domain_aware_test_mse": np.nan,
                "domain_aware_test_mae": np.nan,
                "domain_aware_test_r2": np.nan,
                "domain_aware_acc_within_2.5": np.nan,
            }
        )
    leaderboard_frame = pd.DataFrame(leaderboard)
    leaderboard_frame.to_csv(reports_dir / "model_leaderboard.csv", index=False)

    total_runtime = time.perf_counter() - total_start
    metrics = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "data_path": args.data_path,
        "split": args.split,
        "profile": args.profile,
        "selected_candidate": best_run["name"],
        "feature_mode": best_run["feature_mode"],
        "embedding_backend": best_run["embedding_backend"],
        "embedding_dim": int(best_run["embedding_dim"]),
        "embedding_model": args.embedding_model,
        "regressor": best_run["regressor"],
        "selection_metric": "domain_acc_within_2.5_then_domain_test_mae",
        "winsorization": winsor_info,
        "rows_after_cleaning": int(len(df)),
        "train_size": int(len(x_train_text)),
        "test_size": int(len(x_test_text)),
        "best_cv_mae": float(best_run["cv_mae"]),
        "best_cv_mse": float(best_run["cv_mse"]),
        "best_cv_r2": float(best_run["cv_r2"]),
        "test_mse": float(mse),
        "test_mae": float(mae),
        "test_r2": float(r2),
        "test_mse_raw": float(best_run["test_mse_raw"]),
        "test_mae_raw": float(best_run["test_mae_raw"]),
        "test_r2_raw": float(best_run["test_r2_raw"]),
        "tolerance_accuracy": tolerance_accuracy,
        "tolerance_accuracy_raw": best_run["tolerance_accuracy_raw"],
        "domain_aware_test_mse": domain_aware_mse,
        "domain_aware_test_mae": domain_aware_mae,
        "domain_aware_test_r2": domain_aware_r2,
        "domain_aware_tolerance_accuracy": domain_aware_tolerance,
        "best_params": best_run["best_params"],
        "calibration_stats": best_run["calibration_stats"],
        "blend_result": blend_result,
        "domain_models_trained": int(len(domain_models)),
        "domain_calibrators_trained": int(len(domain_calibrators)),
        "domain_calibrators_active": int(len(active_domain_calibrators)),
        "domain_calibration_mode": domain_calibration_mode,
        "domain_min_samples": int(args.domain_min_samples),
        "domain_model_summary": domain_summary,
        "model_leaderboard": leaderboard,
        "failed_candidates": failed_runs,
        "search_n_jobs": int(args.n_jobs),
        "rf_n_jobs": int(args.rf_n_jobs),
        "search_verbose": int(args.search_verbose),
        "runtime_seconds": float(total_runtime),
        "top_features": top_features,
        "model_artifact": str(predictor_path),
    }
    with log_stage("Save metrics JSON"):
        save_metrics(metrics, reports_dir / "metrics.json")
    logger.info("Saved model leaderboard to: %s", reports_dir / "model_leaderboard.csv")
    logger.info("Saved artifact to: %s", predictor_path)
    logger.info("Saved reports to: %s", reports_dir)
    logger.info("Total runtime: %.2fs", total_runtime)


if __name__ == "__main__":
    main()
