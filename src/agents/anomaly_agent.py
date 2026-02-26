"""Anomaly Agent for suspicious transaction investigation."""

from __future__ import annotations

import json
from typing import Any

try:
    import boto3
except Exception:  # pragma: no cover
    boto3 = None


class AnomalyAgent:
    def __init__(self):
        self.endpoint_name = "merchantmind-anomaly-detector"
        self.sagemaker_runtime = (
            boto3.client("sagemaker-runtime", region_name="ap-southeast-2") if boto3 else None
        )

    async def execute(self, query: str, merchant_id: str) -> dict[str, Any]:
        recent_txns = await self._fetch_recent_transactions(merchant_id, days=7)

        scored = []
        for txn in recent_txns:
            features = await self._get_features(txn)
            score = self._invoke_model(features)
            if score > 0.7:
                explanation = self._get_shap_explanation(features)
                scored.append(
                    {
                        "transaction_id": txn["transaction_id"],
                        "amount_aud": txn["amount_aud"],
                        "timestamp": txn["timestamp"],
                        "anomaly_score": round(score, 3),
                        "risk_level": self._classify_risk(score),
                        "top_reasons": explanation["top_features"],
                    }
                )

        summary = self._generate_summary(scored, merchant_id, query)

        return {
            "response": summary,
            "flagged_transactions": scored,
            "total_reviewed": len(recent_txns),
            "total_flagged": len(scored),
            "confidence": 0.85,
        }

    def _invoke_model(self, features: dict[str, Any]) -> float:
        if not self.sagemaker_runtime:
            return float(min(max(features.get("amount_zscore", 0.0) / 10.0, 0.0), 1.0))

        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps({"features": features}),
        )
        result = json.loads(response["Body"].read().decode())
        return float(result["anomaly_score"])

    def _classify_risk(self, score: float) -> str:
        if score >= 0.95:
            return "CRITICAL"
        if score >= 0.85:
            return "HIGH"
        if score >= 0.70:
            return "MEDIUM"
        return "LOW"

    def _get_shap_explanation(self, features: dict[str, Any]) -> dict[str, Any]:
        _ = features
        return {
            "top_features": [
                {
                    "feature": "amount_zscore",
                    "impact": "high",
                    "explanation": "Transaction amount exceeds normal merchant range",
                },
                {
                    "feature": "hour_of_day",
                    "impact": "medium",
                    "explanation": "Observed outside common business hours",
                },
            ]
        }

    async def _fetch_recent_transactions(self, merchant_id: str, days: int) -> list[dict[str, Any]]:
        _ = days
        return [
            {
                "transaction_id": f"T-{merchant_id}-001",
                "merchant_id": merchant_id,
                "amount_aud": 4999.0,
                "timestamp": "2026-02-20T03:00:00",
                "amount_zscore": 8.0,
                "hour_of_day": 3,
            },
            {
                "transaction_id": f"T-{merchant_id}-002",
                "merchant_id": merchant_id,
                "amount_aud": 120.0,
                "timestamp": "2026-02-20T12:00:00",
                "amount_zscore": 0.2,
                "hour_of_day": 12,
            },
        ]

    async def _get_features(self, txn: dict[str, Any]) -> dict[str, Any]:
        return {
            "amount_zscore": float(txn.get("amount_zscore", 0.0)),
            "hour_of_day": float(txn.get("hour_of_day", 12)),
        }

    def _generate_summary(self, scored: list[dict[str, Any]], merchant_id: str, query: str) -> str:
        _ = query
        if not scored:
            return f"No high-risk anomalies found for merchant {merchant_id} in the recent window."
        return (
            f"Found {len(scored)} suspicious transactions for merchant {merchant_id}. "
            f"Top risk score: {max(row['anomaly_score'] for row in scored):.3f}."
        )
