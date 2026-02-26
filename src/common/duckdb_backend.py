"""Local DuckDB warehouse backend (default for portfolio mode)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DuckDBUnavailableError(RuntimeError):
    """Raised when DuckDB is not installed but required."""


class DuckDBWarehouse:
    """Lightweight local SQL backend backed by generated CSV artifacts."""

    def __init__(
        self,
        db_path: str = "data/generated/merchantmind.duckdb",
        transactions_csv: str = "data/generated/transactions_12months.csv",
        features_csv: str = "data/generated/training_features.csv",
    ):
        self.db_path = Path(db_path)
        self.transactions_csv = Path(transactions_csv)
        self.features_csv = Path(features_csv)
        self._initialized = False

    def _connect(self, read_only: bool = False):
        try:
            import duckdb
        except Exception as exc:  # pragma: no cover
            raise DuckDBUnavailableError(
                "DuckDB is required. Install with: pip install duckdb"
            ) from exc

        if read_only and self.db_path.exists():
            return duckdb.connect(str(self.db_path), read_only=True)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.db_path))

    def ensure_initialized(self) -> None:
        if self._initialized:
            return

        if self.db_path.exists():
            try:
                check_conn = self._connect(read_only=True)
                try:
                    existing = {
                        row[0]
                        for row in check_conn.execute("SHOW TABLES").fetchall()
                    }
                    has_tx = "transactions" in existing
                    has_feat = (
                        "training_features" in existing
                        or not self.features_csv.exists()
                    )
                    if has_tx and has_feat:
                        self._initialized = True
                        return
                finally:
                    check_conn.close()
            except Exception:
                pass

        conn = self._connect(read_only=False)
        try:
            if self.transactions_csv.exists():
                conn.execute(
                    """
                    CREATE OR REPLACE TABLE transactions AS
                    SELECT * FROM read_csv_auto(?, HEADER=TRUE)
                    """,
                    [str(self.transactions_csv)],
                )

            if self.features_csv.exists():
                conn.execute(
                    """
                    CREATE OR REPLACE TABLE training_features AS
                    SELECT * FROM read_csv_auto(?, HEADER=TRUE)
                    """,
                    [str(self.features_csv)],
                )
        finally:
            conn.close()

        self._initialized = True

    def query_df(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        self.ensure_initialized()
        conn = self._connect(read_only=True)
        try:
            return conn.execute(sql, params or []).df()
        finally:
            conn.close()

    def query_records(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        df = self.query_df(sql, params)
        return df.to_dict(orient="records")

    def fetch_merchant_history(self, merchant_id: str, limit: int = 2000) -> pd.DataFrame:
        self.ensure_initialized()

        query = """
        SELECT
            transaction_id,
            merchant_id,
            amount_aud,
            timestamp,
            customer_id,
            payment_terminal,
            card_type,
            is_outside_business_hours,
            state
        FROM transactions
        WHERE merchant_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """

        df = self.query_df(query, [merchant_id, int(limit)])
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df
