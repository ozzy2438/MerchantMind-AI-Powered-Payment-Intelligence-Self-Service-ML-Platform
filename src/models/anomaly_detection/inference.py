"""Model inference helper for anomaly detection artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass
class PredictionResult:
    score: float
    model_version: str


class AnomalyInferenceService:
    """Loads model artifacts and serves ensemble score predictions."""

    def __init__(self, artifacts_path: str = "model_artifacts/anomaly_detector.joblib"):
        self.artifacts_path = artifacts_path
        self.artifacts: dict[str, Any] = {}

    def load(self) -> None:
        path = Path(self.artifacts_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifacts not found: {path}")
        self.artifacts = joblib.load(path)

    def predict(self, features: dict[str, float]) -> PredictionResult:
        if not self.artifacts:
            self.load()

        xgb_model = self.artifacts["xgboost"]
        iso_model = self.artifacts["isolation_forest"]
        columns = self.artifacts["feature_columns"]

        vector = np.array([[features.get(col, 0.0) for col in columns]], dtype=float)
        iso_score = -iso_model.score_samples(vector)[0]
        xgb_score = xgb_model.predict_proba(vector)[0][1]

        combined = 0.3 * iso_score + 0.7 * xgb_score
        score = float(max(0.0, min(1.0, combined)))

        return PredictionResult(score=score, model_version="v1")
