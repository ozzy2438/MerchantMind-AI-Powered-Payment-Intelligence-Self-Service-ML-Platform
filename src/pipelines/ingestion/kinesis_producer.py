"""Streaming ingestion simulation via Kinesis."""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime
from typing import Any

try:
    import boto3
except Exception:  # pragma: no cover
    boto3 = None


REGION = "ap-southeast-2"
STREAM_NAME = "merchantmind-transactions"

kinesis = boto3.client("kinesis", region_name=REGION) if boto3 else None
s3 = boto3.client("s3", region_name=REGION) if boto3 else None


class TransactionProducer:
    """Publishes payment transactions to a Kinesis stream."""

    def __init__(self, generator: Any):
        self.generator = generator

    def produce(self, transactions_per_second: int = 100) -> None:
        if not kinesis:
            raise RuntimeError("boto3 is required for Kinesis publishing")

        while True:
            batch = []
            for _ in range(transactions_per_second):
                txn = self.generator.generate_transaction()
                batch.append(
                    {
                        "Data": json.dumps(txn).encode("utf-8"),
                        "PartitionKey": txn["merchant_id"],
                    }
                )
            kinesis.put_records(StreamName=STREAM_NAME, Records=batch)
            time.sleep(1)


class TransactionConsumer:
    """Lambda-style consumer for Kinesis records."""

    def handler(self, event: dict[str, Any], context: Any) -> None:
        for record in event.get("Records", []):
            payload = json.loads(base64.b64decode(record["kinesis"]["data"]).decode("utf-8"))
            validated = self._validate_transaction(payload)
            self._write_to_s3_raw(validated)
            features = self._compute_realtime_features(validated)
            score = self._invoke_anomaly_model(features)
            if score > 0.85:
                self._publish_alert(validated, score)

    def _validate_transaction(self, txn: dict[str, Any]) -> dict[str, Any]:
        required = {"transaction_id", "merchant_id", "amount_aud", "timestamp"}
        missing = required - set(txn.keys())
        if missing:
            raise ValueError(f"Missing required keys: {sorted(missing)}")
        return txn

    def _write_to_s3_raw(self, txn: dict[str, Any]) -> None:
        if not s3:
            return
        now = datetime.utcnow()
        key = (
            f"raw/transactions/year={now.year}/month={now.month:02d}/day={now.day:02d}/"
            f"hour={now.hour:02d}/{txn['transaction_id']}.json"
        )
        s3.put_object(
            Bucket="merchantmind-data-lake",
            Key=key,
            Body=json.dumps(txn).encode("utf-8"),
        )

    def _compute_realtime_features(self, txn: dict[str, Any]) -> dict[str, Any]:
        # TODO: wire to MerchantFeatureEngine in runtime deployment.
        return {
            "amount_aud": float(txn["amount_aud"]),
            "hour_of_day": datetime.fromisoformat(txn["timestamp"]).hour,
        }

    def _invoke_anomaly_model(self, features: dict[str, Any]) -> float:
        # TODO: wire to model endpoint.
        return min(max(features.get("amount_aud", 0) / 5000.0, 0.0), 1.0)

    def _publish_alert(self, txn: dict[str, Any], score: float) -> None:
        # TODO: integrate SNS/PagerDuty.
        _ = (txn, score)
