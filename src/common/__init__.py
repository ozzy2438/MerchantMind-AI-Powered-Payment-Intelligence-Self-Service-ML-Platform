"""Common utilities and backend adapters."""

from src.common.duckdb_backend import DuckDBUnavailableError, DuckDBWarehouse

__all__ = ["DuckDBWarehouse", "DuckDBUnavailableError"]
