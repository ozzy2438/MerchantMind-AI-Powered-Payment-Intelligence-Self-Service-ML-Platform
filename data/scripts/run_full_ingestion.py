"""Run external ingestion and calibrated generation end-to-end."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.pipelines.ingestion.calibrated_generator import CalibratedTransactionGenerator
from src.pipelines.ingestion.real_data_sources import AustralianDataIngestion


def build_ingestion_report(results: dict[str, dict]) -> dict:
    success_count = sum(1 for item in results.values() if item.get("status") == "success")
    live_success_count = sum(
        1 for item in results.values() if item.get("status") == "success" and item.get("mode") == "live"
    )
    fallback_success_count = sum(
        1 for item in results.values() if item.get("status") == "success" and item.get("mode") == "fallback"
    )
    calibrated_count = sum(
        1 for item in results.values() if item.get("status") == "success" and item.get("mode") == "calibrated"
    )
    failed_sources = sorted(name for name, item in results.items() if item.get("status") != "success")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_count": len(results),
        "success_count": success_count,
        "live_success_count": live_success_count,
        "fallback_success_count": fallback_success_count,
        "calibrated_count": calibrated_count,
        "failed_sources": failed_sources,
        "sources": results,
    }


def write_ingestion_report(report: dict, report_path: str | Path = "data/external/ingestion_report.json") -> Path:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path


def main() -> None:
    ingestion = AustralianDataIngestion()
    results = ingestion.run_full_ingestion()
    report = build_ingestion_report(results)
    report_path = write_ingestion_report(report)

    print(f"Ingestion completed: success={report['success_count']}/{len(results)}")
    print(
        "Breakdown:"
        f" live_success={report['live_success_count']},"
        f" fallback_success={report['fallback_success_count']},"
        f" calibrated={report['calibrated_count']}"
    )
    print(f"Ingestion report written: {report_path}")

    generator = CalibratedTransactionGenerator()
    output = generator.run(include_ingestion=False)
    print(f"Generated transactions dataset: {output}")


if __name__ == "__main__":
    main()
