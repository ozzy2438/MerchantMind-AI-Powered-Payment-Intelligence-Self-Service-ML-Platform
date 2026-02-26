"""Bias detection helper stubs."""

from __future__ import annotations

import pandas as pd


def evaluate_category_bias(df: pd.DataFrame, score_col: str, label_col: str) -> dict:
    """Return simple per-category precision gaps as a placeholder."""
    if df.empty:
        return {"status": "no_data", "max_precision_gap": 0.0}

    if "merchant_category" not in df.columns:
        return {"status": "missing_category", "max_precision_gap": 0.0}

    grouped = df.groupby("merchant_category")
    precision = {}
    for category, frame in grouped:
        tp = ((frame[score_col] >= 0.85) & (frame[label_col] == 1)).sum()
        fp = ((frame[score_col] >= 0.85) & (frame[label_col] == 0)).sum()
        precision[category] = float(tp / max(tp + fp, 1))

    if not precision:
        return {"status": "no_groups", "max_precision_gap": 0.0}

    gap = max(precision.values()) - min(precision.values())
    return {"status": "ok", "precision_by_category": precision, "max_precision_gap": float(gap)}
