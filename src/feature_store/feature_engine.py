"""Central feature computation engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from sagemaker.feature_store.feature_group import FeatureGroup
except Exception:  # pragma: no cover
    FeatureGroup = Any  # type: ignore[assignment]

from src.feature_store.feature_definitions import (
    ADDITIONAL_FEATURES,
    FEATURE_REGISTRY,
    FeatureDefinition,
)
from src.common.duckdb_backend import DuckDBUnavailableError, DuckDBWarehouse


class _NoopFeatureGroup:
    def put_record(self, Record: list[dict[str, str]]) -> None:  # noqa: N802
        _ = Record


class MerchantFeatureEngine:
    """Computes merchant-level real-time and batch features."""

    def __init__(
        self,
        snowflake_conn: Any = None,
        feature_group: FeatureGroup | None = None,
        backend: str = "duckdb",
    ):
        self.conn = snowflake_conn
        self.feature_group = feature_group or _NoopFeatureGroup()
        self.backend = "snowflake" if snowflake_conn is not None else backend
        self.duckdb_warehouse = None
        if self.backend == "duckdb":
            try:
                self.duckdb_warehouse = DuckDBWarehouse()
            except DuckDBUnavailableError:
                self.duckdb_warehouse = None

    def compute_realtime_features(self, merchant_id: str, current_txn: dict) -> Dict[str, float]:
        historical = self._fetch_merchant_history(merchant_id)
        features: Dict[str, float] = {"merchant_id": merchant_id}  # type: ignore[assignment]

        for feat_def in FEATURE_REGISTRY:
            if feat_def.window != "none":
                windowed = self._apply_window(historical, feat_def.window)
                features[feat_def.name] = self._aggregate(windowed, feat_def.computation, current_txn)
            else:
                features[feat_def.name] = self._compute_instant(feat_def.computation, current_txn)

        avg = float(features.get("avg_amount_7d", 0.0))
        std = float(features.get("stddev_amount_30d", 1.0))
        features["amount_zscore"] = (float(current_txn["amount_aud"]) - avg) / max(std, 0.01)

        self._upsert_to_feature_store(features)
        return features

    def compute_batch_features(self, date: str) -> pd.DataFrame:
        if self.conn is None and self.duckdb_warehouse is None:
            raise RuntimeError("No batch backend available. Use snowflake_conn or install duckdb.")

        query = f"""
        SELECT
            merchant_id,
            COUNT(*) OVER (
                PARTITION BY merchant_id
                ORDER BY timestamp
                RANGE BETWEEN INTERVAL '1 HOUR' PRECEDING AND CURRENT ROW
            ) as txn_velocity_1h,
            AVG(amount_aud) OVER (
                PARTITION BY merchant_id
                ORDER BY timestamp
                RANGE BETWEEN INTERVAL '7 DAY' PRECEDING AND CURRENT ROW
            ) as avg_amount_7d,
            STDDEV(amount_aud) OVER (
                PARTITION BY merchant_id
                ORDER BY timestamp
                RANGE BETWEEN INTERVAL '30 DAY' PRECEDING AND CURRENT ROW
            ) as stddev_amount_30d,
            HOUR(timestamp) as hour_of_day,
            is_fraud
        FROM transactions
        WHERE DATE(timestamp) = '{date}'
        """
        if self.conn is not None:
            snowflake_query = query.replace("FROM transactions", "FROM MERCHANTMIND.CURATED.TRANSACTIONS")
            return pd.read_sql(snowflake_query, self.conn)
        return self.duckdb_warehouse.query_df(query)

    def _fetch_merchant_history(self, merchant_id: str) -> pd.DataFrame:
        if self.duckdb_warehouse is not None:
            try:
                history = self.duckdb_warehouse.fetch_merchant_history(merchant_id)
                if not history.empty:
                    return history
            except Exception:
                pass

        # Fallback synthetic history if backend data is unavailable.
        rng = np.random.default_rng(abs(hash(merchant_id)) % (2**32))
        now = datetime.utcnow()
        rows = 300
        timestamps = pd.date_range(end=now, periods=rows, freq="4h")
        data = pd.DataFrame(
            {
                "merchant_id": [merchant_id] * rows,
                "amount_aud": np.clip(rng.normal(loc=95.0, scale=35.0, size=rows), 1.0, None),
                "timestamp": timestamps,
                "customer_id": [f"C{n:08d}" for n in rng.integers(1, 2_000_000, size=rows)],
                "payment_terminal": rng.choice(
                    ["tap_and_go", "chip_and_pin", "eftpos", "manual_entry"],
                    p=[0.95, 0.03, 0.015, 0.005],
                    size=rows,
                ),
                "card_type": rng.choice(["debit", "credit"], p=[0.69, 0.31], size=rows),
                "is_outside_business_hours": rng.choice([False, True], p=[0.88, 0.12], size=rows),
                "state": rng.choice(["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"], size=rows),
            }
        )
        return data

    def _apply_window(self, historical: pd.DataFrame, window: str) -> pd.DataFrame:
        if historical.empty:
            return historical
        anchor = pd.to_datetime(historical["timestamp"]).max()
        if window.endswith("H"):
            hours = int(window[:-1])
            start = anchor - pd.Timedelta(hours=hours)
        elif window.endswith("D"):
            days = int(window[:-1])
            start = anchor - pd.Timedelta(days=days)
        else:
            return historical
        ts = pd.to_datetime(historical["timestamp"])
        return historical.loc[ts >= start]

    def _aggregate(self, windowed: pd.DataFrame, computation: str, current_txn: dict) -> float:
        if windowed.empty:
            return 0.0

        if computation == "count":
            return float(len(windowed))
        if computation == "mean":
            return float(windowed["amount_aud"].mean())
        if computation == "std":
            return float(windowed["amount_aud"].std(ddof=0) or 0.0)
        if computation == "max":
            return float(windowed["amount_aud"].max())
        if computation == "nunique":
            return float(windowed["customer_id"].nunique())
        if computation == "ratio":
            return float((windowed["payment_terminal"] == "tap_and_go").mean())
        if computation == "debit_ratio":
            return float((windowed["card_type"] == "debit").mean())
        if computation == "off_hours_ratio":
            return float(windowed["is_outside_business_hours"].astype(float).mean())
        if computation == "state_deviation":
            merchant_avg = float(windowed["amount_aud"].mean())
            state = current_txn.get("state") or windowed["state"].iloc[-1]
            state_avg = float(windowed.loc[windowed["state"] == state, "amount_aud"].mean())
            if np.isnan(state_avg):
                state_avg = merchant_avg
            return merchant_avg - state_avg
        if computation == "zscore":
            avg = float(windowed["amount_aud"].mean())
            std = float(windowed["amount_aud"].std(ddof=0) or 1.0)
            return (float(current_txn["amount_aud"]) - avg) / max(std, 0.01)
        return 0.0

    def _compute_instant(self, computation: str, current_txn: dict) -> float:
        timestamp = current_txn.get("timestamp")
        dt = pd.to_datetime(timestamp) if timestamp is not None else datetime.utcnow()
        hour = int(dt.hour)
        if computation == "extract_hour":
            return float(hour)
        if computation == "business_hours_check":
            return float(hour < 8 or hour > 20)
        return 0.0

    def _upsert_to_feature_store(self, features: dict) -> None:
        record = [{"FeatureName": k, "ValueAsString": str(v)} for k, v in features.items()]
        self.feature_group.put_record(Record=record)


__all__ = ["FeatureDefinition", "FEATURE_REGISTRY", "ADDITIONAL_FEATURES", "MerchantFeatureEngine"]
