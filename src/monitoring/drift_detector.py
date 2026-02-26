"""Continuous model monitoring for data drift."""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    import boto3
except Exception:  # pragma: no cover
    boto3 = None

try:
    import structlog
except Exception:  # pragma: no cover
    import logging

    class _CompatLogger:
        def __init__(self, name: str):
            self._logger = logging.getLogger(name)

        def info(self, event: str, **kwargs):
            self._logger.info("%s %s", event, kwargs if kwargs else "")

        def warning(self, event: str, **kwargs):
            self._logger.warning("%s %s", event, kwargs if kwargs else "")

        def error(self, event: str, **kwargs):
            self._logger.error("%s %s", event, kwargs if kwargs else "")

    class _StructlogFallback:
        @staticmethod
        def get_logger():
            return _CompatLogger("merchantmind")

    structlog = _StructlogFallback()  # type: ignore[assignment]

try:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.report import Report
except Exception:  # pragma: no cover
    ColumnMapping = None
    DataDriftPreset = None
    TargetDriftPreset = None
    Report = None

logger = structlog.get_logger()


class ModelMonitor:
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
    ]

    DRIFT_THRESHOLD = 0.3

    def __init__(self, reference_data_path: str):
        self.reference = pd.read_parquet(reference_data_path)
        self.cloudwatch = boto3.client("cloudwatch", region_name="ap-southeast-2") if boto3 else None
        self.sagemaker = boto3.client("sagemaker", region_name="ap-southeast-2") if boto3 else None

        self.column_mapping = (
            ColumnMapping(
                target="is_fraud",
                prediction="predicted_anomaly",
                numerical_features=self.FEATURE_COLUMNS[:8],
                categorical_features=["merchant_category", "payment_terminal", "state"],
            )
            if ColumnMapping
            else None
        )

    def run_monitoring(self, current_data: pd.DataFrame) -> dict[str, Any]:
        if not (Report and DataDriftPreset and TargetDriftPreset and self.column_mapping):
            drift_score = self._fallback_drift_score(current_data)
            if drift_score > self.DRIFT_THRESHOLD:
                self._trigger_retraining()
            return {
                "drift_detected": drift_score > self.DRIFT_THRESHOLD,
                "drift_score": drift_score,
                "action_taken": "retraining_triggered" if drift_score > self.DRIFT_THRESHOLD else "none",
            }

        drift_report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
        drift_report.run(
            reference_data=self.reference,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )
        results = drift_report.as_dict()

        drift_detected = results["metrics"][0]["result"]["dataset_drift"]
        drift_score = float(results["metrics"][0]["result"]["share_of_drifted_columns"])

        self._publish_metrics(
            {
                "data_drift_score": drift_score,
                "drifted_features_count": results["metrics"][0]["result"]["number_of_drifted_columns"],
                "dataset_drift_detected": 1.0 if drift_detected else 0.0,
            }
        )

        if drift_score > self.DRIFT_THRESHOLD:
            logger.warning("drift_threshold_exceeded", drift_score=drift_score, threshold=self.DRIFT_THRESHOLD)
            self._trigger_retraining()

        return {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "action_taken": "retraining_triggered" if drift_score > self.DRIFT_THRESHOLD else "none",
        }

    def _fallback_drift_score(self, current_data: pd.DataFrame) -> float:
        common = [col for col in self.FEATURE_COLUMNS if col in self.reference.columns and col in current_data.columns]
        if not common:
            return 0.0
        diffs = []
        for col in common:
            ref_mean = float(self.reference[col].fillna(0).mean())
            cur_mean = float(current_data[col].fillna(0).mean())
            denom = max(abs(ref_mean), 1.0)
            diffs.append(min(abs(cur_mean - ref_mean) / denom, 1.0))
        return float(sum(diffs) / len(diffs))

    def _publish_metrics(self, metrics: dict[str, float]) -> None:
        if not self.cloudwatch:
            return
        metric_data = [
            {
                "MetricName": name,
                "Value": value,
                "Unit": "None",
                "Dimensions": [
                    {"Name": "ModelName", "Value": "anomaly-detector"},
                    {"Name": "Environment", "Value": "production"},
                ],
            }
            for name, value in metrics.items()
        ]
        self.cloudwatch.put_metric_data(Namespace="MerchantMind/ModelMonitoring", MetricData=metric_data)

    def _trigger_retraining(self) -> None:
        if not self.sagemaker:
            return
        self.sagemaker.start_pipeline_execution(
            PipelineName="merchantmind-retrain-pipeline",
            PipelineParameters=[
                {"Name": "trigger_reason", "Value": "auto_drift_detection"},
                {"Name": "drift_score", "Value": str(self.DRIFT_THRESHOLD)},
            ],
        )
        logger.info("retraining_pipeline_triggered", reason="data_drift")
