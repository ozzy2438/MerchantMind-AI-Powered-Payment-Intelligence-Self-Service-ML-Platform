"""Calibrated transaction generator using multi-source calibration tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipelines.ingestion.real_data_sources import AustralianDataIngestion


@dataclass
class GeneratorConfig:
    num_transactions: int = 120_000
    num_merchants: int = 3_000
    months: int = 12
    seed: int = 42
    output_path: str = "data/generated/transactions_12months.parquet"


class CalibratedTransactionGenerator:
    """Generate synthetic transactions calibrated to open-data priors."""

    STATES = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]
    STATE_WEIGHTS = np.array([0.33, 0.26, 0.18, 0.10, 0.07, 0.03, 0.02, 0.01])
    CATEGORIES = ["hospitality", "retail", "health", "services"]
    CATEGORY_WEIGHTS = np.array([0.34, 0.31, 0.15, 0.20])
    TERMINALS = ["tap_and_go", "chip_and_pin", "eftpos", "manual_entry"]
    TERMINAL_WEIGHTS = np.array([0.95, 0.03, 0.015, 0.005])
    CARD_TYPES = ["debit", "credit"]
    CARD_WEIGHTS = np.array([0.69, 0.31])
    CATEGORY_HOUR_PROFILES = {
        # Hospitality can run late, but still not uniformly overnight.
        "hospitality": np.array(
            [
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.02,
                0.03,
                0.05,
                0.06,
                0.06,
                0.06,
                0.06,
                0.06,
                0.06,
                0.06,
                0.06,
                0.06,
                0.06,
                0.07,
                0.08,
                0.09,
                0.08,
                0.06,
                0.04,
            ]
        ),
        # Retail peaks in daytime and drops sharply at night.
        "retail": np.array(
            [
                0.002,
                0.002,
                0.002,
                0.002,
                0.002,
                0.003,
                0.01,
                0.03,
                0.05,
                0.08,
                0.09,
                0.09,
                0.09,
                0.09,
                0.08,
                0.08,
                0.08,
                0.07,
                0.06,
                0.04,
                0.02,
                0.01,
                0.005,
                0.003,
            ]
        ),
        # Health should be near-zero midnight to 6 AM.
        "health": np.array(
            [
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.0005,
                0.0008,
                0.002,
                0.01,
                0.08,
                0.10,
                0.10,
                0.10,
                0.10,
                0.09,
                0.08,
                0.07,
                0.06,
                0.05,
                0.03,
                0.02,
                0.01,
                0.003,
                0.0015,
                0.0012,
            ]
        ),
        # Services mostly business hours.
        "services": np.array(
            [
                0.0015,
                0.0015,
                0.0015,
                0.0015,
                0.0015,
                0.002,
                0.01,
                0.03,
                0.07,
                0.09,
                0.10,
                0.10,
                0.10,
                0.09,
                0.08,
                0.07,
                0.06,
                0.05,
                0.04,
                0.03,
                0.015,
                0.008,
                0.004,
                0.002,
            ]
        ),
    }

    def __init__(self, config: GeneratorConfig | None = None):
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def generate_dataset(self) -> pd.DataFrame:
        merchants = self._build_merchants()
        txns = self._build_transactions(merchants)
        return txns

    def save(self, df: pd.DataFrame | None = None) -> Path:
        frame = df if df is not None else self.generate_dataset()
        output = Path(self.config.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            frame.to_parquet(output, index=False)
            return output
        except Exception:
            fallback = output.with_suffix(".csv")
            frame.to_csv(fallback, index=False)
            return fallback

    def run(self, include_ingestion: bool = True) -> Path:
        if include_ingestion:
            AustralianDataIngestion().run_full_ingestion()
        df = self.generate_dataset()
        return self.save(df)

    def _build_merchants(self) -> pd.DataFrame:
        merchant_ids = [f"M{idx:05d}" for idx in range(1, self.config.num_merchants + 1)]
        states = self.rng.choice(self.STATES, size=self.config.num_merchants, p=self.STATE_WEIGHTS)
        categories = self.rng.choice(
            self.CATEGORIES,
            size=self.config.num_merchants,
            p=self.CATEGORY_WEIGHTS,
        )
        abn_numbers = self.rng.integers(10_000_000_000, 99_999_999_999, size=self.config.num_merchants)
        return pd.DataFrame(
            {
                "merchant_id": merchant_ids,
                "merchant_abn": [str(n) for n in abn_numbers],
                "state": states,
                "merchant_category": categories,
            }
        )

    def _build_transactions(self, merchants: pd.DataFrame) -> pd.DataFrame:
        tx_count = self.config.num_transactions
        merchant_idx = self.rng.integers(0, len(merchants), size=tx_count)
        merchant_rows = merchants.iloc[merchant_idx].reset_index(drop=True)

        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(days=30 * self.config.months)
        total_days = max(int((now.date() - start.date()).days), 1)
        day_offsets = self.rng.integers(0, total_days + 1, size=tx_count)
        minutes = self.rng.integers(0, 60, size=tx_count)
        seconds = self.rng.integers(0, 60, size=tx_count)
        sampled_hours = [self._sample_hour_for_category(cat) for cat in merchant_rows["merchant_category"]]
        timestamps = [
            start
            + timedelta(
                days=int(day_offsets[i]),
                hours=int(sampled_hours[i]),
                minutes=int(minutes[i]),
                seconds=int(seconds[i]),
            )
            for i in range(tx_count)
        ]

        category_base = {
            "hospitality": (45.0, 0.7),
            "retail": (78.0, 0.8),
            "health": (120.0, 0.65),
            "services": (95.0, 0.75),
        }
        amounts = []
        category_expected_amount = []
        for category in merchant_rows["merchant_category"]:
            mean, sigma = category_base[category]
            value = float(self.rng.lognormal(mean=np.log(mean), sigma=sigma))
            amounts.append(round(min(max(value, 1.0), 50_000.0), 2))
            category_expected_amount.append(mean)

        terminals = self.rng.choice(self.TERMINALS, size=tx_count, p=self.TERMINAL_WEIGHTS)
        card_types = self.rng.choice(self.CARD_TYPES, size=tx_count, p=self.CARD_WEIGHTS)

        outside_hours = []
        for ts, category in zip(timestamps, merchant_rows["merchant_category"]):
            hour = ts.hour
            if category == "hospitality":
                outside = hour < 7 or hour > 23
            elif category == "retail":
                outside = hour < 9 or hour > 19
            elif category == "health":
                outside = hour < 8 or hour > 18
            else:
                outside = hour < 8 or hour > 20
            outside_hours.append(outside)

        # Build fraud labels with realistic signal patterns rather than pure random noise.
        # This keeps global fraud rate low while making anomalies learnable by the model.
        expected = np.array(category_expected_amount, dtype=float)
        amount_ratio = np.array(amounts, dtype=float) / np.maximum(expected, 1.0)
        outside_arr = np.array(outside_hours, dtype=bool)

        fraud_prob = np.full(tx_count, 0.0003, dtype=float)
        fraud_prob += (terminals == "manual_entry").astype(float) * 0.1174
        fraud_prob += (terminals == "chip_and_pin").astype(float) * 0.0106
        fraud_prob += outside_arr.astype(float) * 0.0024
        fraud_prob += (card_types == "credit").astype(float) * 0.0006
        fraud_prob += (amount_ratio > 4.0).astype(float) * 0.0287
        fraud_prob += (amount_ratio > 8.0).astype(float) * 0.0462
        fraud_prob += (
            (terminals == "manual_entry") & outside_arr & (amount_ratio > 4.0)
        ).astype(float) * 0.3994
        fraud_prob += (
            (terminals == "chip_and_pin") & outside_arr & (amount_ratio > 4.0)
        ).astype(float) * 0.1747

        fraud_prob = np.clip(fraud_prob, 0.0, 0.45)
        fraud = self.rng.random(tx_count) < fraud_prob

        customer_ids = [f"C{n:08d}" for n in self.rng.integers(1, 40_000_000, size=tx_count)]

        frame = pd.DataFrame(
            {
                "transaction_id": [f"T{n:010d}" for n in range(1, tx_count + 1)],
                "merchant_id": merchant_rows["merchant_id"],
                "merchant_abn": merchant_rows["merchant_abn"],
                "customer_id": customer_ids,
                "amount_aud": amounts,
                "currency": ["AUD"] * tx_count,
                "timestamp": [ts.isoformat() for ts in timestamps],
                "merchant_category": merchant_rows["merchant_category"],
                "state": merchant_rows["state"],
                "payment_terminal": terminals,
                "card_type": card_types,
                "is_fraud": fraud,
                "is_outside_business_hours": outside_hours,
            }
        )
        return frame.sort_values("timestamp").reset_index(drop=True)

    def _sample_hour_for_category(self, category: str) -> int:
        profile = self.CATEGORY_HOUR_PROFILES.get(category)
        if profile is None or len(profile) != 24:
            return int(self.rng.integers(0, 24))
        probs = profile / profile.sum()
        return int(self.rng.choice(np.arange(24), p=probs))


if __name__ == "__main__":
    generator = CalibratedTransactionGenerator()
    output = generator.run(include_ingestion=True)
    print(f"Generated dataset at {output}")
