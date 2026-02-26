"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from src.pipelines.ingestion.calibrated_generator import CalibratedTransactionGenerator, GeneratorConfig


class _MockFeatureGroup:
    def __init__(self):
        self.records = []

    def put_record(self, Record):  # noqa: N802
        self.records.append(Record)


@pytest.fixture
def mock_conn():
    return None


@pytest.fixture
def mock_fg():
    return _MockFeatureGroup()


@pytest.fixture
def curated_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["T1", "T2", "T3"],
            "merchant_id": ["M1", "M1", "M2"],
            "amount_aud": [10.0, 25.0, 30.0],
            "currency": ["AUD", "AUD", "AUD"],
            "timestamp": ["2026-01-01T10:00:00", "2026-01-01T11:00:00", "2026-01-01T12:00:00"],
            "merchant_category": ["retail", "retail", "services"],
            "state": ["NSW", "NSW", "VIC"],
            "is_fraud": [0, 0, 1],
        }
    )


@pytest.fixture(scope="session")
def generated_dataset() -> pd.DataFrame:
    cfg = GeneratorConfig(num_transactions=30_000, num_merchants=2_000, seed=42)
    generator = CalibratedTransactionGenerator(config=cfg)
    return generator.generate_dataset()
