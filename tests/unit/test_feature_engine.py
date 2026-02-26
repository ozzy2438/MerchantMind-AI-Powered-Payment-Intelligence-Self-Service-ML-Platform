"""Unit tests for feature computation."""

import math

from src.feature_store.feature_engine import MerchantFeatureEngine


class TestFeatureEngine:
    def test_zscore_normal_transaction(self, mock_conn, mock_fg):
        engine = MerchantFeatureEngine(mock_conn, mock_fg)
        features = engine.compute_realtime_features(
            merchant_id="M001",
            current_txn={"amount_aud": 50.0, "timestamp": "2025-01-01T12:00:00"},
        )
        assert abs(features["amount_zscore"]) < 3.0

    def test_zscore_anomalous_transaction(self, mock_conn, mock_fg):
        engine = MerchantFeatureEngine(mock_conn, mock_fg)
        features = engine.compute_realtime_features(
            merchant_id="M001",
            current_txn={"amount_aud": 50000.0, "timestamp": "2025-01-01T03:00:00"},
        )
        assert features["amount_zscore"] > 3.0

    def test_handles_zero_stddev(self, mock_conn, mock_fg):
        engine = MerchantFeatureEngine(mock_conn, mock_fg)
        features = engine.compute_realtime_features(
            merchant_id="NEW_MERCHANT",
            current_txn={"amount_aud": 100.0, "timestamp": "2025-01-01T12:00:00"},
        )
        assert not math.isinf(features["amount_zscore"])
        assert not math.isnan(features["amount_zscore"])
