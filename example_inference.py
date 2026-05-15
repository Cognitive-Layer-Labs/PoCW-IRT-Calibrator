"""Minimal usage example for the persisted IRT difficulty predictor."""

from __future__ import annotations

import argparse

from src.predictor import load_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with saved predictor.")
    parser.add_argument(
        "--model-path",
        default="artifacts/difficulty_predictor.pkl",
        help="Path to the saved predictor artifact.",
    )
    parser.add_argument(
        "--question",
        default="What is the capital of France?",
        help="Question text to score.",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Optional domain key for domain-specific fallback (e.g., arc, math).",
    )
    parser.add_argument(
        "--bloom-level",
        default=None,
        help="Optional Bloom level (1-6 or text like 'analyze').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = load_predictor(args.model_path)
    params = predictor.predict_item_params(
        args.question,
        domain=args.domain,
        bloom_level=args.bloom_level,
    )
    print(params)
    if predictor.available_domains():
        sample_domain = predictor.available_domains()[0]
        params_domain = predictor.predict_item_params(
            args.question,
            domain=sample_domain,
            bloom_level=args.bloom_level,
        )
        print({"domain": sample_domain, "params": params_domain})


if __name__ == "__main__":
    main()
