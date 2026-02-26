"""Compatibility wrapper for transaction Kinesis consumer."""

from src.pipelines.ingestion.kinesis_producer import TransactionConsumer

__all__ = ["TransactionConsumer"]
