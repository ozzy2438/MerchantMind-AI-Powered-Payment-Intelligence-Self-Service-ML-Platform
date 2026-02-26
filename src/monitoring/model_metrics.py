"""Model metrics helper functions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelMetrics:
    roc_auc: float
    avg_precision: float
    precision_at_threshold: float
    recall_at_threshold: float


def is_healthy(metrics: ModelMetrics, min_roc_auc: float = 0.8) -> bool:
    return metrics.roc_auc >= min_roc_auc
