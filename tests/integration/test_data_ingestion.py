"""Integration tests for ingestion provenance and calibrated generator outputs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import src.pipelines.ingestion.real_data_sources as ingestion_module
from src.pipelines.ingestion.real_data_sources import AustralianDataIngestion


pytestmark = pytest.mark.integration


def _load_run_script_module():
    script_path = Path("data/scripts/run_full_ingestion.py")
    spec = importlib.util.spec_from_file_location("run_full_ingestion_script", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, url: str):
        self.url = url
        self._json_payload = {"data": {"observations": []}}
        self.text = ""

        if "api.data.abs.gov.au" in url:
            self._json_payload = {
                "data": {
                    "observations": [
                        {"TIME_PERIOD": "2026-01", "OBS_VALUE": 128.0},
                        {"TIME_PERIOD": "2026-02", "OBS_VALUE": 129.1},
                    ]
                }
            }
        elif "tables-d1.csv" in url:
            self.text = (
                "month,contactless_ratio,debit_ratio,fraud_rate\n"
                "2026-01,0.95,0.69,0.0012\n"
                "2026-02,96,70,0.13\n"
            )

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._json_payload


class TestDataIngestion:
    def test_abs_retail_trade_returns_data(self):
        ingestion = AustralianDataIngestion()
        df = ingestion.fetch_abs_retail_trade()
        assert len(df) > 0
        assert "TIME_PERIOD" in df.columns
        assert "OBS_VALUE" in df.columns

    def test_abs_fallback_metadata_when_live_fails(self, monkeypatch, tmp_path):
        if ingestion_module.requests is None:
            pytest.skip("requests dependency missing")

        def fail_get(*args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("network_down")

        monkeypatch.setattr(ingestion_module.requests, "get", fail_get)

        ingestion = AustralianDataIngestion(output_dir=str(tmp_path))
        results = ingestion.run_full_ingestion()

        abs_result = results["abs_retail_trade"]
        assert abs_result["status"] == "success"
        assert abs_result["mode"] == "fallback"
        assert abs_result["live_attempted"] is True
        assert abs_result["live_success"] is False
        assert abs_result["fallback_used"] is True

    def test_rba_live_parse_success_sets_live_mode(self, monkeypatch, tmp_path):
        if ingestion_module.requests is None:
            pytest.skip("requests dependency missing")

        def fake_get(url, timeout):
            _ = timeout
            return _FakeResponse(url)

        monkeypatch.setattr(ingestion_module.requests, "get", fake_get)

        ingestion = AustralianDataIngestion(output_dir=str(tmp_path))
        results = ingestion.run_full_ingestion()

        rba_result = results["rba_card_payments"]
        assert rba_result["status"] == "success"
        assert rba_result["mode"] == "live"
        assert rba_result["live_attempted"] is True
        assert rba_result["live_success"] is True
        assert rba_result["fallback_used"] is False
        assert rba_result["records"] > 0

    def test_ingestion_report_counts_cover_all_7_sources(self, monkeypatch, tmp_path):
        if ingestion_module.requests is None:
            pytest.skip("requests dependency missing")

        def fail_get(*args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("offline")

        monkeypatch.setattr(ingestion_module.requests, "get", fail_get)

        ingestion = AustralianDataIngestion(output_dir=str(tmp_path / "external"))
        results = ingestion.run_full_ingestion()

        script_module = _load_run_script_module()
        report = script_module.build_ingestion_report(results)
        report_path = script_module.write_ingestion_report(report, tmp_path / "external" / "ingestion_report.json")

        assert report["source_count"] == 7
        assert report["live_success_count"] + report["fallback_success_count"] + report["calibrated_count"] == 7
        assert report_path.exists()

        loaded = json.loads(report_path.read_text())
        assert loaded["source_count"] == 7


class TestCalibratedGenerator:
    def test_fraud_rate_matches_rba(self, generated_dataset):
        actual_rate = generated_dataset["is_fraud"].mean()
        assert 0.0005 < actual_rate < 0.005

    def test_contactless_rate_matches_rba(self, generated_dataset):
        tap_rate = (generated_dataset["payment_terminal"] == "tap_and_go").mean()
        assert 0.85 < tap_rate < 0.98

    def test_state_distribution_matches_abs(self, generated_dataset):
        state_counts = generated_dataset.groupby("state")["merchant_id"].nunique()
        assert state_counts["NSW"] > state_counts["TAS"]
        assert state_counts["VIC"] > state_counts["NT"]

    def test_no_negative_amounts(self, generated_dataset):
        assert (generated_dataset["amount_aud"] > 0).all()

    def test_all_required_columns_present(self, generated_dataset):
        required = [
            "transaction_id",
            "merchant_id",
            "merchant_abn",
            "merchant_category",
            "state",
            "amount_aud",
            "currency",
            "timestamp",
            "payment_terminal",
            "card_type",
            "is_fraud",
        ]
        for col in required:
            assert col in generated_dataset.columns, f"Missing: {col}"
