"""Train anomaly detection model using Isolation Forest + XGBoost ensemble."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        precision_recall_fscore_support,
        roc_auc_score,
    )
except Exception:  # pragma: no cover
    IsolationForest = None
    average_precision_score = None
    precision_recall_curve = None
    precision_recall_fscore_support = None
    roc_auc_score = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


FEATURE_COLUMNS = [
    "txn_velocity_1h",
    "txn_velocity_24h",
    "avg_amount_7d",
    "stddev_amount_30d",
    "amount_zscore",
    "max_amount_7d",
    "unique_customers_1d",
    "pct_tap_payments_7d",
    "hour_of_day",
    "is_outside_business_hours",
    "pct_debit_7d",
    "off_hours_txn_ratio_7d",
    "state_avg_amount_deviation",
    "is_manual_entry",
    "is_chip_and_pin",
    "is_credit_card",
    "amount_to_merchant_avg_7d",
]


class AnomalyDetectionTrainer:
    """Train and evaluate the anomaly detection ensemble."""

    def __init__(self, feature_store_offline_path: str):
        self.data_path = feature_store_offline_path

    def train(self) -> dict:
        self._ensure_dependencies()

        df = self._load_dataset()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp").reset_index(drop=True)

        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            raise RuntimeError(f"Training dataset missing features: {missing_features}")
        X = df[FEATURE_COLUMNS].fillna(0)
        y = df["is_fraud"].astype(int)

        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        iso_forest = IsolationForest(
            n_estimators=200,
            contamination=0.005,
            random_state=42,
            n_jobs=-1,
        )
        iso_forest.fit(X_train)
        iso_scores = -iso_forest.score_samples(X_test)

        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        xgb_model = xgb.XGBClassifier(
            n_estimators=450,
            max_depth=5,
            learning_rate=0.03,
            min_child_weight=8,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            early_stopping_rounds=30,
            random_state=42,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_scores = xgb_model.predict_proba(X_test)[:, 1]

        iso_normalized = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
        ensemble_scores = 0.3 * iso_normalized + 0.7 * xgb_scores

        threshold = self._select_threshold(y_test.to_numpy(), ensemble_scores)
        metrics = self._evaluate(y_test, ensemble_scores, threshold=threshold)

        explainer = shap.TreeExplainer(xgb_model) if shap else None

        artifacts = {
            "isolation_forest": iso_forest,
            "xgboost": xgb_model,
            "shap_explainer": explainer,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": metrics,
        }

        Path("model_artifacts").mkdir(parents=True, exist_ok=True)
        joblib.dump(artifacts, "model_artifacts/anomaly_detector.joblib")

        return metrics

    def _load_dataset(self) -> pd.DataFrame:
        path = Path(self.data_path)

        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)

        if path.suffix.lower() == ".parquet":
            try:
                return pd.read_parquet(path)
            except Exception:
                fallback = path.with_suffix(".csv")
                if fallback.exists():
                    return pd.read_csv(fallback)
                raise

        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)

    def _evaluate(self, y_true, scores, threshold: float = 0.85) -> dict:
        predictions = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, predictions, average="binary", zero_division=0
        )

        return {
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "avg_precision": float(average_precision_score(y_true, scores)),
            "precision_at_threshold": float(precision),
            "recall_at_threshold": float(recall),
            "f1_at_threshold": float(f1),
            "threshold": threshold,
            "total_samples": int(len(y_true)),
            "fraud_samples": int(y_true.sum()),
        }

    def _select_threshold(self, y_true: np.ndarray, scores: np.ndarray) -> float:
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        if len(thresholds) == 0:
            return 0.85

        f1_scores = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-9)
        best_idx = int(np.argmax(f1_scores))
        selected = float(thresholds[best_idx])

        return max(0.05, min(0.95, selected))

    def _ensure_dependencies(self) -> None:
        missing = []
        if IsolationForest is None:
            missing.append("scikit-learn")
        if xgb is None:
            missing.append("xgboost")
        if (
            precision_recall_fscore_support is None
            or roc_auc_score is None
            or average_precision_score is None
            or precision_recall_curve is None
        ):
            missing.append("scikit-learn.metrics")
        if missing:
            raise RuntimeError(f"Missing dependencies for training: {sorted(set(missing))}")
