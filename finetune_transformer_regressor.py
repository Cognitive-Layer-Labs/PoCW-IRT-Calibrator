"""Fine-tune a transformer regression head for IRT difficulty prediction."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from train_pipeline import load_dataframe, winsorize_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune transformer regressor for IRT difficulty."
    )
    parser.add_argument(
        "--data-path",
        default="data/cross_difficulty_train.csv",
        help="Path to local training data.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Base HF model for fine-tuning.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/finetuned-transformer-regressor",
        help="Directory where the fine-tuned model and metrics are saved.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--winsorize-lower", type=float, default=0.01)
    parser.add_argument("--winsorize-upper", type=float, default=0.99)
    parser.add_argument(
        "--domain-strategy",
        default="all",
        choices=["all", "curated_reasoning"],
        help="Optionally restrict training domains.",
    )
    parser.add_argument(
        "--embedding-export-dir",
        default=None,
        help=(
            "Optional folder to export only the base encoder weights for embedding usage. "
            "Defaults to <output-dir>/encoder."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def choose_device() -> str:
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def apply_domain_strategy(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "all":
        return frame
    if strategy == "curated_reasoning":
        keep = {"arc", "bbh", "gpqa_extended", "gsm8k", "math", "musr"}
        filtered = frame[frame["domain"].isin(keep)].reset_index(drop=True)
        if filtered.empty:
            raise ValueError("Domain filtering removed all rows. Use --domain-strategy all.")
        return filtered
    raise ValueError(f"Unsupported domain strategy: {strategy}")


def make_hf_dataset(frame: pd.DataFrame) -> Dataset:
    subset = frame[["question", "irt_difficulty"]].copy()
    subset = subset.rename(columns={"question": "text", "irt_difficulty": "label"})
    return Dataset.from_pandas(subset, preserve_index=False)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = logits.reshape(-1)
    labels = labels.reshape(-1)
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    abs_error = np.abs(labels - predictions)
    return {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "acc_within_0.5": float(np.mean(abs_error <= 0.5)),
        "acc_within_1.0": float(np.mean(abs_error <= 1.0)),
        "acc_within_2.0": float(np.mean(abs_error <= 2.0)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading data from {args.data_path}")
    frame = load_dataframe(
        dataset_name="BatsResearch/Cross-Difficulty",
        dataset_config=None,
        split="train",
        data_path=args.data_path,
        question_column="question",
        target_column="irt_difficulty",
        hf_token=None,
    )
    frame, winsor = winsorize_target(frame, args.winsorize_lower, args.winsorize_upper)
    frame = apply_domain_strategy(frame, args.domain_strategy)
    print(f"[INFO] Rows after curation: {len(frame)}")
    print(
        "[INFO] Winsor bounds:",
        f"{winsor['lower_bound']:.4f} .. {winsor['upper_bound']:.4f}",
        f"(clipped {winsor['num_clipped']} rows)",
    )

    train_frame, val_frame = train_test_split(
        frame,
        test_size=args.val_size,
        random_state=args.seed,
    )
    train_frame = train_frame.reset_index(drop=True)
    val_frame = val_frame.reset_index(drop=True)
    print(f"[INFO] Split sizes train={len(train_frame)} val={len(val_frame)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = make_hf_dataset(train_frame)
    val_ds = make_hf_dataset(val_frame)

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type="regression",
    )

    device = choose_device()
    print(f"[INFO] Using device: {device}")

    use_fp16 = device == "cuda"
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_mae",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=50,
        seed=args.seed,
        dataloader_num_workers=0,
        report_to=[],
        fp16=use_fp16,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print("[INFO] Eval metrics:", eval_metrics)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Export encoder-only weights for embedding extractor usage
    encoder_dir = (
        Path(args.embedding_export_dir)
        if args.embedding_export_dir
        else output_dir / "encoder"
    )
    encoder_dir.mkdir(parents=True, exist_ok=True)
    base_model = getattr(model, model.base_model_prefix, None)
    if base_model is not None:
        base_model.save_pretrained(encoder_dir)
        tokenizer.save_pretrained(encoder_dir)
    else:
        model.save_pretrained(encoder_dir)
        tokenizer.save_pretrained(encoder_dir)

    report = {
        "model_name": args.model_name,
        "output_dir": str(output_dir),
        "encoder_dir": str(encoder_dir),
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "winsorization": winsor,
        "domain_strategy": args.domain_strategy,
        "eval_metrics": eval_metrics,
    }
    (output_dir / "finetune_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[INFO] Saved model to {output_dir}")
    print(f"[INFO] Saved encoder export to {encoder_dir}")


if __name__ == "__main__":
    main()
