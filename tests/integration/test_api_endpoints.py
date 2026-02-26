"""API integration tests for dashboard and backward-compatible /v1 endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from src.api import main as api_main


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(api_main.app)


def test_api_health_and_overview(client: TestClient):
    health = client.get("/api/health")
    assert health.status_code == 200
    health_payload = health.json()
    assert "status" in health_payload
    assert "transactions_loaded" in health_payload

    overview = client.get("/api/dashboard/overview")
    assert overview.status_code == 200
    payload = overview.json()
    required = {
        "total_transactions",
        "total_merchants",
        "total_volume_aud",
        "avg_transaction_aud",
        "fraud_rate_pct",
        "states_covered",
    }
    assert required.issubset(payload.keys())


def test_api_dashboard_endpoints_return_200(client: TestClient):
    endpoints = [
        "/api/config/public",
        "/api/dashboard/revenue-by-category",
        "/api/dashboard/revenue-trend",
        "/api/dashboard/fraud-by-state",
        "/api/dashboard/hourly-pattern",
        "/api/dashboard/payment-methods",
        "/api/dashboard/top-merchants?limit=5",
        "/api/dashboard/anomaly-summary",
        "/api/dashboard/threat-map?days=30",
        "/api/live-feed?limit=10",
    ]

    for endpoint in endpoints:
        response = client.get(endpoint)
        assert response.status_code == 200, endpoint


def test_public_config_contract(client: TestClient):
    response = client.get("/api/config/public")
    assert response.status_code == 200
    payload = response.json()
    assert "map_provider" in payload
    assert "google_maps_js_api_key" in payload
    assert payload["map_provider"] in {"leaflet", "google"}


def test_live_feed_contract(client: TestClient):
    response = client.get("/api/live-feed?limit=10")
    assert response.status_code == 200
    payload = response.json()
    assert "threshold_used" in payload
    assert "transactions" in payload
    assert isinstance(payload["transactions"], list)
    if payload["transactions"]:
        row = payload["transactions"][0]
        assert {
            "transaction_id",
            "merchant_id",
            "state",
            "amount_aud",
            "anomaly_score",
            "is_anomaly",
            "risk_level",
        }.issubset(row.keys())


def test_agent_query_scope_is_enforced_with_token(client: TestClient):
    response = client.post(
        "/api/agent/query",
        headers={"Authorization": "Bearer merchant:M00001"},
        json={"question": "revenue summary", "merchant_id": "M00002"},
    )
    assert response.status_code == 403


def test_api_score_returns_threshold_and_risk(client: TestClient):
    response = client.post(
        "/api/score",
        json={
            "merchant_id": "M00001",
            "amount_aud": 500.0,
            "payment_terminal": "tap_and_go",
            "hour_of_day": 14,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "anomaly_score" in payload
    assert "threshold_used" in payload
    assert payload["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_v1_contracts_still_work(client: TestClient, monkeypatch):
    class _Pred:
        score = 0.42

    monkeypatch.setattr(
        api_main.feature_engine,
        "compute_realtime_features",
        lambda merchant_id, current_txn: {
            "merchant_id": merchant_id,
            "amount_zscore": 0.7,
            "txn_velocity_1h": 2.0,
            "is_outside_business_hours": 0.0,
        },
    )
    monkeypatch.setattr(api_main.inference_service, "predict", lambda features: _Pred())

    async def _fake_router(query: str, merchant_id: str):
        return {
            "response": f"stub response for {merchant_id}",
            "intent": "sql",
            "confidence": 0.99,
            "request_id": "req_test",
            "sources": ["duckdb"],
        }

    monkeypatch.setattr(api_main.router_agent, "handle_query", _fake_router)

    score_resp = client.post(
        "/v1/transactions/score",
        headers={"Authorization": "Bearer merchant:M00001"},
        json={
            "transaction_id": "T-test",
            "merchant_id": "M00001",
            "amount_aud": 120.0,
            "payment_terminal": "tap_and_go",
            "timestamp": "2026-02-26T10:00:00",
        },
    )
    assert score_resp.status_code == 200
    score_payload = score_resp.json()
    assert set(score_payload.keys()) == {
        "anomaly_score",
        "is_anomaly",
        "risk_level",
        "explanation",
        "recommended_action",
    }

    agent_resp = client.post(
        "/v1/agent/query",
        headers={"Authorization": "Bearer merchant:M00001"},
        json={"question": "how many transactions?", "merchant_id": "M00001"},
    )
    assert agent_resp.status_code == 200
    payload = agent_resp.json()
    assert payload["intent"] == "sql"
    assert payload["request_id"] == "req_test"

    dashboard_resp = client.get(
        "/v1/merchants/M00001/dashboard",
        headers={"Authorization": "Bearer merchant:M00001"},
    )
    assert dashboard_resp.status_code == 200
    dashboard_payload = dashboard_resp.json()
    assert {
        "revenue_trend_30d",
        "anomaly_summary",
        "category_benchmark",
        "top_payment_methods",
        "peak_hours",
    }.issubset(dashboard_payload.keys())
