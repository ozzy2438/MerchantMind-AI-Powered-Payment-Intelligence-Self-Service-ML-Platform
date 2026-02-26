"""Structured logging and immutable audit trail."""

from __future__ import annotations

import json
from datetime import datetime

try:
    import boto3
except Exception:  # pragma: no cover
    boto3 = None

try:
    import structlog
except Exception:  # pragma: no cover
    import logging

    class _CompatLogger:
        def __init__(self, name: str):
            self._logger = logging.getLogger(name)

        def info(self, event: str, **kwargs):
            self._logger.info("%s %s", event, kwargs if kwargs else "")

        def warning(self, event: str, **kwargs):
            self._logger.warning("%s %s", event, kwargs if kwargs else "")

        def error(self, event: str, **kwargs):
            self._logger.error("%s %s", event, kwargs if kwargs else "")

    class _StructlogFallback:
        class processors:
            @staticmethod
            def TimeStamper(fmt: str = "iso"):  # noqa: N802
                def _processor(logger, method_name, event_dict):
                    _ = (logger, method_name, fmt)
                    return event_dict

                return _processor

            @staticmethod
            def add_log_level(logger, method_name, event_dict):
                _ = logger
                event_dict["level"] = method_name
                return event_dict

            @staticmethod
            def StackInfoRenderer():
                def _processor(logger, method_name, event_dict):
                    _ = (logger, method_name)
                    return event_dict

                return _processor

            @staticmethod
            def JSONRenderer():
                def _processor(logger, method_name, event_dict):
                    _ = (logger, method_name)
                    return event_dict

                return _processor

        class BoundLogger:  # pragma: no cover
            pass

        class PrintLoggerFactory:
            def __call__(self):
                return logging.getLogger("merchantmind")

        @staticmethod
        def configure(**kwargs):
            _ = kwargs

        @staticmethod
        def get_logger():
            return _CompatLogger("merchantmind")

    structlog = _StructlogFallback()  # type: ignore[assignment]


structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


class AuditLogger:
    """Immutable audit trail helper (S3 object-lock capable)."""

    def __init__(self):
        self.s3 = boto3.client("s3", region_name="ap-southeast-2") if boto3 else None
        self.audit_bucket = "merchantmind-audit-production"

    async def log_prediction(
        self,
        transaction_id: str,
        merchant_id: str,
        anomaly_score: float,
        model_version: str,
        features_used: list,
        explanation: dict,
    ) -> None:
        entry = {
            "event_type": "ml_prediction",
            "timestamp": datetime.utcnow().isoformat(),
            "transaction_id": transaction_id,
            "merchant_id": merchant_id,
            "anomaly_score": anomaly_score,
            "model_version": model_version,
            "features_used": features_used,
            "explanation_summary": explanation,
        }
        await self._write_audit_log(entry)

    async def log_agent_interaction(
        self,
        request_id: str,
        merchant_id: str,
        query: str,
        intent: str,
        response: str,
        confidence: float,
    ) -> None:
        entry = {
            "event_type": "agent_interaction",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "merchant_id": merchant_id,
            "query": query,
            "intent_classified": intent,
            "response_length": len(response),
            "confidence": confidence,
        }
        await self._write_audit_log(entry)

    async def log_blocked_query(self, query: str, merchant_id: str, reason: str) -> None:
        entry = {
            "event_type": "query_blocked",
            "timestamp": datetime.utcnow().isoformat(),
            "merchant_id": merchant_id,
            "query_snippet": query[:100],
            "block_reason": reason,
            "severity": "HIGH",
        }
        await self._write_audit_log(entry)
        logger.warning("query_blocked", merchant_id=merchant_id, reason=reason)

    async def _write_audit_log(self, entry: dict) -> None:
        now = datetime.utcnow()
        key = (
            f"audit/{entry['event_type']}/year={now.year}/month={now.month:02d}/day={now.day:02d}/"
            f"{now.strftime('%H%M%S')}_{entry.get('request_id', 'na')}.json"
        )

        if self.s3:
            self.s3.put_object(
                Bucket=self.audit_bucket,
                Key=key,
                Body=json.dumps(entry).encode("utf-8"),
                ContentType="application/json",
            )
            return

        # Local fallback for dev and tests.
        from pathlib import Path

        local_dir = Path("reports/audit")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / key.replace("/", "_")).write_text(json.dumps(entry))
