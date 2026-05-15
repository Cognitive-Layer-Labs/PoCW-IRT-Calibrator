"""Predictor wrapper that exposes IRT-compatible item parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass
class DifficultyPredictor:
    """Wrap a difficulty regressor with a 2PL-compatible API."""

    feature_extractor: Any
    model: Any
    feature_names: list[str]
    domain_models: dict[str, Any] = field(default_factory=dict)
    calibrator: Any | None = None
    domain_calibrators: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_b(
        self,
        question_text: str,
        domain: str | None = None,
        bloom_level: float | int | str | None = None,
    ) -> float:
        """Predict the IRT difficulty parameter b for one question."""
        text = question_text if question_text is not None else ""
        features = self.feature_extractor.transform([str(text)])
        features = self._align_feature_order(features, bloom_levels=[bloom_level])
        model = self._select_model(domain)
        prediction = model.predict(features)[0]
        prediction = self._apply_calibration(float(prediction), domain)
        return float(prediction)

    def predict_item_params(
        self,
        question_text: str,
        domain: str | None = None,
        bloom_level: float | int | str | None = None,
    ) -> dict[str, float]:
        """Return 2PL-style item params while fixing discrimination to a=1.0."""
        predicted_b = self.predict_b(question_text, domain=domain, bloom_level=bloom_level)
        return {"b": predicted_b, "a": 1.0}

    def predict_batch_item_params(
        self,
        question_texts: list[str],
        domains: list[str | None] | None = None,
        bloom_levels: list[float | int | str | None] | None = None,
    ) -> list[dict[str, float]]:
        """Predict params for a list of questions."""
        if domains is not None and len(domains) != len(question_texts):
            raise ValueError("domains must have the same length as question_texts.")
        if bloom_levels is not None and len(bloom_levels) != len(question_texts):
            raise ValueError("bloom_levels must have the same length as question_texts.")

        if domains is None and not self.domain_models:
            features = self.feature_extractor.transform([str(text) for text in question_texts])
            features = self._align_feature_order(features, bloom_levels=bloom_levels)
            b_values = self.model.predict(features)
            calibrated_values = [
                self._apply_calibration(float(value), domain=None) for value in b_values
            ]
            return [{"b": float(value), "a": 1.0} for value in calibrated_values]

        features = self.feature_extractor.transform([str(text) for text in question_texts])
        features = self._align_feature_order(features, bloom_levels=bloom_levels)
        predictions = self.model.predict(features)

        if domains is not None and self.domain_models:
            normalized_domains = [self._normalize_domain(domain) for domain in domains]
            unique_domains = sorted(set(normalized_domains))
            for domain in unique_domains:
                if domain is None:
                    continue
                domain_model = self.domain_models.get(domain)
                if domain_model is None:
                    continue
                indices = [idx for idx, value in enumerate(normalized_domains) if value == domain]
                if not indices:
                    continue
                domain_features = features.iloc[indices]
                domain_preds = domain_model.predict(domain_features)
                for local_idx, sample_idx in enumerate(indices):
                    predictions[sample_idx] = domain_preds[local_idx]

        calibrated_values = [
            self._apply_calibration(
                float(predictions[idx]),
                domain=domains[idx] if domains is not None else None,
            )
            for idx in range(len(predictions))
        ]
        return [{"b": float(value), "a": 1.0} for value in calibrated_values]

    def available_domains(self) -> list[str]:
        """Return supported domain labels for domain-specific models."""
        return sorted(self.domain_models.keys())

    def _select_model(self, domain: str | None):
        if domain is None:
            return self.model
        key = self._normalize_domain(domain)
        if not key:
            return self.model
        return self.domain_models.get(key, self.model)

    def _normalize_domain(self, domain: str | None) -> str | None:
        if domain is None:
            return None
        key = str(domain).strip()
        return key if key else None

    def _normalize_bloom(self, bloom_level: float | int | str | None) -> float:
        if bloom_level is None:
            return 0.0
        if isinstance(bloom_level, (int, float)):
            value = float(bloom_level)
            if value != value:  # NaN
                return 0.0
            return float(max(0.0, min(6.0, value)))

        text = str(bloom_level).strip().lower()
        if not text:
            return 0.0
        mapping = {
            "remember": 1.0,
            "understand": 2.0,
            "apply": 3.0,
            "analyze": 4.0,
            "evaluate": 5.0,
            "create": 6.0,
        }
        for key, value in mapping.items():
            if key in text:
                return value
        digits = "".join(char for char in text if char.isdigit())
        if digits:
            return float(max(0.0, min(6.0, float(digits))))
        return 0.0

    def _align_feature_order(
        self, features: pd.DataFrame, bloom_levels: list[float | int | str | None] | None = None
    ) -> pd.DataFrame:
        aligned = features.copy()
        if bloom_levels is not None and "bloom_level" in self.feature_names:
            bloom_numeric = [self._normalize_bloom(value) for value in bloom_levels]
            if len(bloom_numeric) == len(aligned):
                aligned["bloom_level"] = bloom_numeric

        if not self.feature_names:
            return aligned
        return aligned.reindex(columns=self.feature_names, fill_value=0.0)

    def _apply_calibration(self, prediction: float, domain: str | None) -> float:
        domain_key = self._normalize_domain(domain)
        if domain_key is not None and domain_key in self.domain_calibrators:
            calibrated = self.domain_calibrators[domain_key].predict([prediction])[0]
            return float(calibrated)
        if self.calibrator is not None:
            calibrated = self.calibrator.predict([prediction])[0]
            return float(calibrated)
        return float(prediction)


def save_predictor(predictor: DifficultyPredictor, output_path: str | Path) -> None:
    """Persist the predictor bundle with joblib."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, output_path)


def load_predictor(model_path: str | Path) -> DifficultyPredictor:
    """Load a persisted predictor bundle."""
    return joblib.load(model_path)
