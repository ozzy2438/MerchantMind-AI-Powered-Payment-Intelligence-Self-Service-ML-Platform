"""Deprecated script kept for compatibility.

This repository now uses multi-source ingestion via:
- src/pipelines/ingestion/real_data_sources.py
- src/pipelines/ingestion/calibrated_generator.py
"""


def main() -> None:
    raise RuntimeError(
        "Deprecated: use 'python data/scripts/run_full_ingestion.py' for the canonical ingestion flow."
    )


if __name__ == "__main__":
    main()
