"""Build offline training features from generated transaction data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.anomaly_detection.train import FEATURE_COLUMNS


def _build_merchant_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("timestamp").copy()
    idx = group.set_index("timestamp")

    group["txn_velocity_1h"] = idx["transaction_id"].rolling("1h").count().values
    group["txn_velocity_24h"] = idx["transaction_id"].rolling("24h").count().values
    group["avg_amount_7d"] = idx["amount_aud"].rolling("7D").mean().values
    group["stddev_amount_30d"] = idx["amount_aud"].rolling("30D").std().fillna(0).values
    group["max_amount_7d"] = idx["amount_aud"].rolling("7D").max().values

    customer_codes = pd.Series(pd.factorize(group["customer_id"])[0], index=idx.index)
    group["unique_customers_1d"] = (
        customer_codes.rolling("1D").apply(lambda x: len(np.unique(x)), raw=True).fillna(0).values
    )

    group["pct_tap_payments_7d"] = (
        (idx["payment_terminal"] == "tap_and_go").astype(float).rolling("7D").mean().fillna(0).values
    )
    group["pct_debit_7d"] = (
        (idx["card_type"] == "debit").astype(float).rolling("7D").mean().fillna(0).values
    )
    group["off_hours_txn_ratio_7d"] = (
        idx["is_outside_business_hours"].astype(float).rolling("7D").mean().fillna(0).values
    )

    group["hour_of_day"] = group["timestamp"].dt.hour.astype(float)
    group["is_outside_business_hours"] = group["is_outside_business_hours"].astype(float)
    group["is_manual_entry"] = (group["payment_terminal"] == "manual_entry").astype(float)
    group["is_chip_and_pin"] = (group["payment_terminal"] == "chip_and_pin").astype(float)
    group["is_credit_card"] = (group["card_type"] == "credit").astype(float)

    std_safe = group["stddev_amount_30d"].replace(0, 0.01)
    group["amount_zscore"] = (group["amount_aud"] - group["avg_amount_7d"]) / std_safe
    group["amount_to_merchant_avg_7d"] = group["amount_aud"] / np.maximum(group["avg_amount_7d"], 1.0)

    return group


def build_training_features(input_path: str, output_path: str) -> pd.DataFrame:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input dataset not found: {source}")

    df = pd.read_csv(source)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["merchant_id", "timestamp"]).reset_index(drop=True)

    featured = pd.concat(
        [_build_merchant_features(group) for _, group in df.groupby("merchant_id", sort=False)],
        ignore_index=True,
    )

    state_avg_amount = featured.groupby("state")["amount_aud"].transform("mean")
    featured["state_avg_amount_deviation"] = featured["avg_amount_7d"] - state_avg_amount

    featured[FEATURE_COLUMNS] = featured[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.suffix.lower() == ".parquet":
        try:
            featured.to_parquet(target, index=False)
            return featured
        except Exception:
            target = target.with_suffix(".csv")

    featured.to_csv(target, index=False)
    return featured


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training features dataset")
    parser.add_argument(
        "--input",
        default="data/generated/transactions_12months.csv",
        help="Path to generated transactions CSV",
    )
    parser.add_argument(
        "--output",
        default="data/generated/training_features.csv",
        help="Path to output training features file",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    data = build_training_features(args.input, str(output_path))

    fraud_rate = float(data["is_fraud"].mean()) if "is_fraud" in data.columns else 0.0
    print(f"Built training features: {len(data):,} rows")
    print(f"Output: {output_path}")
    print(f"Fraud rate: {fraud_rate:.4%}")


if __name__ == "__main__":
    main()
