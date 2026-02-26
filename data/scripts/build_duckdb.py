"""Initialize local DuckDB warehouse from generated CSV datasets."""

from __future__ import annotations

from src.common.duckdb_backend import DuckDBWarehouse


def main() -> None:
    warehouse = DuckDBWarehouse()
    warehouse.ensure_initialized()

    transactions = warehouse.query_records("SELECT COUNT(*) AS n FROM transactions")
    features = warehouse.query_records("SELECT COUNT(*) AS n FROM training_features")

    print(f"DuckDB initialized at: {warehouse.db_path}")
    print(f"transactions rows: {transactions[0]['n'] if transactions else 0}")
    print(f"training_features rows: {features[0]['n'] if features else 0}")


if __name__ == "__main__":
    main()
