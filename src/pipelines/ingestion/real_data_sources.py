"""Multi-source ingestion for Australian payment context data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


@dataclass
class SourceResult:
    status: str
    records: int
    output_path: str
    mode: str
    live_attempted: bool
    live_success: bool
    fallback_used: bool
    error: str | None = None


@dataclass
class _FetchOutcome:
    frame: pd.DataFrame
    mode: str
    live_attempted: bool
    live_success: bool
    fallback_used: bool
    error: str | None = None


class AustralianDataIngestion:
    """Fetches external calibration data from public/open sources."""

    def __init__(self, output_dir: str = "data/external", timeout_seconds: int = 20):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds

    def fetch_abs_retail_trade(self) -> pd.DataFrame:
        """Fetch ABS retail trade style data."""
        return self._fetch_abs_retail_trade_with_meta().frame

    def _fetch_abs_retail_trade_with_meta(self) -> _FetchOutcome:
        """Fetch ABS retail trade with explicit live/fallback provenance."""
        fallback = pd.DataFrame(
            {
                "TIME_PERIOD": ["2025-10", "2025-11", "2025-12", "2026-01"],
                "OBS_VALUE": [124.1, 126.4, 127.2, 128.0],
                "MEASURE": ["index"] * 4,
                "FREQUENCY": ["M"] * 4,
            }
        )
        if requests is None:
            return _FetchOutcome(
                frame=fallback,
                mode="fallback",
                live_attempted=False,
                live_success=False,
                fallback_used=True,
                error="requests_not_available",
            )

        url = "https://api.data.abs.gov.au/data/ABS,RT,1.0.0/.AUS.M/all"
        try:
            response = requests.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            live = self._parse_abs_payload(payload)
            if live.empty:
                raise ValueError("ABS payload parsed empty")
            return _FetchOutcome(
                frame=live,
                mode="live",
                live_attempted=True,
                live_success=True,
                fallback_used=False,
            )
        except Exception as exc:
            return _FetchOutcome(
                frame=fallback,
                mode="fallback",
                live_attempted=True,
                live_success=False,
                fallback_used=True,
                error=str(exc),
            )

    def fetch_rba_card_payments(self) -> pd.DataFrame:
        """Fetch card payment summary aligned with RBA statistics."""
        return self._fetch_rba_card_payments_with_meta().frame

    def _fetch_rba_card_payments_with_meta(self) -> _FetchOutcome:
        """Fetch RBA card summary with explicit live/fallback provenance."""
        fallback = pd.DataFrame(
            {
                "month": ["2025-10", "2025-11", "2025-12", "2026-01"],
                "contactless_ratio": [0.93, 0.94, 0.95, 0.95],
                "debit_ratio": [0.67, 0.68, 0.69, 0.69],
                "fraud_rate": [0.0012, 0.0011, 0.0013, 0.0012],
            }
        )
        if requests is None:
            return _FetchOutcome(
                frame=fallback,
                mode="fallback",
                live_attempted=False,
                live_success=False,
                fallback_used=True,
                error="requests_not_available",
            )

        url = "https://www.rba.gov.au/statistics/tables/csv/tables-d1.csv"
        try:
            response = requests.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
            if not response.text.strip():
                raise ValueError("RBA CSV response is empty")
            csv_path = self.output_dir / "rba_card_payments_raw.csv"
            csv_path.write_text(response.text)
            parsed = pd.read_csv(StringIO(response.text))
            canonical = self._parse_rba_table(parsed)
            if canonical.empty:
                raise ValueError("RBA CSV parse produced empty canonical frame")
            return _FetchOutcome(
                frame=canonical,
                mode="live",
                live_attempted=True,
                live_success=True,
                fallback_used=False,
            )
        except Exception as exc:
            return _FetchOutcome(
                frame=fallback,
                mode="fallback",
                live_attempted=True,
                live_success=False,
                fallback_used=True,
                error=str(exc),
            )

    def fetch_merchant_population(self) -> pd.DataFrame:
        """Synthetic merchant population calibrated to Australian state mix."""
        return pd.DataFrame(
            {
                "state": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "merchant_weight": [0.33, 0.26, 0.18, 0.10, 0.07, 0.03, 0.02, 0.01],
            }
        )

    def fetch_terminal_mix(self) -> pd.DataFrame:
        """Payment terminal mix used by generator calibration."""
        return pd.DataFrame(
            {
                "payment_terminal": [
                    "tap_and_go",
                    "chip_and_pin",
                    "eftpos",
                    "manual_entry",
                ],
                "ratio": [0.95, 0.03, 0.015, 0.005],
            }
        )

    def fetch_card_type_mix(self) -> pd.DataFrame:
        """Debit vs credit mix used by generator calibration."""
        return pd.DataFrame(
            {
                "card_type": ["debit", "credit"],
                "ratio": [0.69, 0.31],
            }
        )

    def fetch_business_hours_profile(self) -> pd.DataFrame:
        """Simple business-hour profile by category."""
        return pd.DataFrame(
            {
                "merchant_category": ["hospitality", "retail", "health", "services"],
                "open_hour": [7, 9, 8, 8],
                "close_hour": [23, 19, 18, 20],
            }
        )

    def run_full_ingestion(self) -> dict[str, dict[str, Any]]:
        """Run all source ingestions and save outputs under data/external."""
        sources = {
            "abs_retail_trade": self._fetch_abs_retail_trade_with_meta,
            "rba_card_payments": self._fetch_rba_card_payments_with_meta,
            "merchant_population": self.fetch_merchant_population,
            "terminal_mix": self.fetch_terminal_mix,
            "card_type_mix": self.fetch_card_type_mix,
            "business_hours_profile": self.fetch_business_hours_profile,
        }

        results: dict[str, dict[str, Any]] = {}
        for name, fn in sources.items():
            try:
                if name in {"abs_retail_trade", "rba_card_payments"}:
                    outcome = fn()
                else:
                    frame = fn()
                    outcome = _FetchOutcome(
                        frame=frame,
                        mode="calibrated",
                        live_attempted=False,
                        live_success=False,
                        fallback_used=False,
                    )
                output_path = self._persist_dataframe(name, outcome.frame)
                results[name] = SourceResult(
                    status="success",
                    records=int(len(outcome.frame)),
                    output_path=str(output_path),
                    mode=outcome.mode,
                    live_attempted=outcome.live_attempted,
                    live_success=outcome.live_success,
                    fallback_used=outcome.fallback_used,
                    error=outcome.error,
                ).__dict__
            except Exception as exc:
                results[name] = SourceResult(
                    status="failed",
                    records=0,
                    output_path="",
                    mode="fallback",
                    live_attempted=False,
                    live_success=False,
                    fallback_used=True,
                    error=str(exc),
                ).__dict__

        try:
            metadata_df = self._build_ingestion_metadata(current_results=results, total_sources=7)
            metadata_path = self._persist_dataframe("ingestion_metadata", metadata_df)
            results["ingestion_metadata"] = SourceResult(
                status="success",
                records=int(len(metadata_df)),
                output_path=str(metadata_path),
                mode="calibrated",
                live_attempted=False,
                live_success=False,
                fallback_used=False,
            ).__dict__
        except Exception as exc:
            results["ingestion_metadata"] = SourceResult(
                status="failed",
                records=0,
                output_path="",
                mode="calibrated",
                live_attempted=False,
                live_success=False,
                fallback_used=False,
                error=str(exc),
            ).__dict__

        return results

    def _build_ingestion_metadata(
        self,
        current_results: dict[str, dict[str, Any]],
        total_sources: int,
    ) -> pd.DataFrame:
        """Build a one-row metadata record for auditing."""
        success_count = sum(1 for item in current_results.values() if item.get("status") == "success")
        live_success_count = sum(1 for item in current_results.values() if item.get("mode") == "live")
        fallback_success_count = sum(1 for item in current_results.values() if item.get("mode") == "fallback")
        calibrated_count = sum(1 for item in current_results.values() if item.get("mode") == "calibrated")
        return pd.DataFrame(
            {
                "ingested_at": [datetime.utcnow().isoformat()],
                "success_count": [success_count],
                "live_success_count": [live_success_count],
                "fallback_success_count": [fallback_success_count],
                "calibrated_count": [calibrated_count],
                "source_count": [total_sources],
            }
        )

    def _persist_dataframe(self, name: str, df: pd.DataFrame) -> Path:
        """Persist as parquet when available; fallback to CSV."""
        parquet_path = self.output_dir / f"{name}.parquet"
        try:
            df.to_parquet(parquet_path, index=False)
            return parquet_path
        except Exception:
            csv_path = self.output_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False)
            return csv_path

    @staticmethod
    def _parse_abs_payload(payload: dict[str, Any]) -> pd.DataFrame:
        """Parse ABS JSON payload into a canonical table with TIME_PERIOD/OBS_VALUE."""
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        observations = data.get("observations", [])

        if isinstance(observations, list):
            frame = pd.DataFrame(observations)
            if not frame.empty and "TIME_PERIOD" in frame.columns and "OBS_VALUE" in frame.columns:
                frame = frame[["TIME_PERIOD", "OBS_VALUE"]].copy()
                frame["MEASURE"] = "index"
                frame["FREQUENCY"] = "M"
                return frame

        if isinstance(observations, dict):
            rows: list[dict[str, Any]] = []
            for key, value in observations.items():
                if isinstance(value, dict):
                    period = value.get("TIME_PERIOD") or value.get("time_period") or key
                    obs_value = value.get("OBS_VALUE") or value.get("obs_value")
                else:
                    period = key
                    obs_value = value
                rows.append(
                    {
                        "TIME_PERIOD": str(period),
                        "OBS_VALUE": obs_value,
                        "MEASURE": "index",
                        "FREQUENCY": "M",
                    }
                )
            frame = pd.DataFrame(rows)
            if not frame.empty:
                frame["OBS_VALUE"] = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
                frame = frame.dropna(subset=["OBS_VALUE"]).reset_index(drop=True)
                return frame

        return pd.DataFrame(columns=["TIME_PERIOD", "OBS_VALUE", "MEASURE", "FREQUENCY"])

    def _parse_rba_table(self, parsed: pd.DataFrame) -> pd.DataFrame:
        """Map arbitrary RBA CSV columns into canonical monthly ratios."""
        if parsed.empty:
            return pd.DataFrame(columns=["month", "contactless_ratio", "debit_ratio", "fraud_rate"])

        normalized = {str(col).strip().lower(): col for col in parsed.columns}
        month_col = self._pick_column(normalized, ["month", "date", "period"])
        contactless_col = self._pick_column(normalized, ["contactless"])
        debit_col = self._pick_column(normalized, ["debit"])
        fraud_col = self._pick_column(normalized, ["fraud"])

        if not month_col:
            month_col = parsed.columns[0]
        if not (contactless_col and debit_col and fraud_col):
            return pd.DataFrame(columns=["month", "contactless_ratio", "debit_ratio", "fraud_rate"])

        frame = pd.DataFrame(
            {
                "month": parsed[month_col].astype(str),
                "contactless_ratio": self._coerce_ratio(parsed[contactless_col]),
                "debit_ratio": self._coerce_ratio(parsed[debit_col]),
                "fraud_rate": self._coerce_fraud_rate(parsed[fraud_col]),
            }
        )

        month_ts = pd.to_datetime(frame["month"], errors="coerce")
        frame["month"] = month_ts.dt.strftime("%Y-%m")
        frame = frame.dropna(subset=["month", "contactless_ratio", "debit_ratio", "fraud_rate"])
        frame = frame.drop_duplicates(subset=["month"]).sort_values("month")
        return frame.reset_index(drop=True)

    @staticmethod
    def _pick_column(normalized: dict[str, Any], tokens: list[str]) -> str | None:
        for col_lower, raw_col in normalized.items():
            if all(token in col_lower for token in tokens):
                return raw_col
        return None

    @staticmethod
    def _coerce_ratio(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(
            series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce",
        )
        max_value = values.max(skipna=True)
        if pd.notna(max_value) and max_value > 1.0:
            values = values / 100.0
        return values.clip(lower=0.0, upper=1.0)

    @staticmethod
    def _coerce_fraud_rate(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(
            series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce",
        )
        max_value = values.max(skipna=True)
        if pd.notna(max_value):
            if max_value > 100:
                values = values / 10_000.0
            elif max_value > 1:
                values = values / 100.0
        return values.clip(lower=0.0, upper=1.0)
