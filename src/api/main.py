"""REST API external interface for MerchantMind (DuckDB-first portfolio mode)."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self) -> dict:
            return dict(self.__dict__)

    def Field(*args, **kwargs):  # type: ignore[no-redef]
        _ = args
        return kwargs.get("default")

from src.agents.router import MerchantMindRouter
from src.common.duckdb_backend import DuckDBUnavailableError, DuckDBWarehouse
from src.feature_store.feature_engine import MerchantFeatureEngine
from src.models.anomaly_detection.inference import AnomalyInferenceService

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
        @staticmethod
        def get_logger():
            return _CompatLogger("merchantmind")

    structlog = _StructlogFallback()  # type: ignore[assignment]

try:
    from fastapi import Depends, FastAPI, HTTPException, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from fastapi.staticfiles import StaticFiles
except Exception:  # pragma: no cover
    Depends = lambda x: x  # type: ignore

    def Query(*args, **kwargs):  # type: ignore[no-redef]
        _ = args
        return kwargs.get("default")

    class Request:  # type: ignore[no-redef]
        url: Any

    class HTTPException(Exception):  # type: ignore[no-redef]
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class HTTPAuthorizationCredentials:  # type: ignore[no-redef]
        credentials: str = ""

    class HTTPBearer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

        def __call__(self):
            return None

    class StaticFiles:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

    class _DummyResponse(dict):
        headers: dict[str, str] = {}

    def FileResponse(path: str):  # type: ignore[no-redef]
        return {"file": path}

    class _DummyApp:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

        def add_middleware(self, *args, **kwargs):
            _ = (args, kwargs)

        def middleware(self, *args, **kwargs):
            _ = (args, kwargs)

            def decorator(func):
                return func

            return decorator

        def get(self, *args, **kwargs):
            _ = (args, kwargs)

            def decorator(func):
                return func

            return decorator

        def post(self, *args, **kwargs):
            _ = (args, kwargs)

            def decorator(func):
                return func

            return decorator

        def mount(self, *args, **kwargs):
            _ = (args, kwargs)

    FastAPI = _DummyApp  # type: ignore
    CORSMiddleware = object  # type: ignore


logger = structlog.get_logger()
security = HTTPBearer(auto_error=False)


def _load_local_env() -> None:
    """Load lightweight KEY=VALUE pairs from .env.local/.env if present."""
    for env_name in (".env.local", ".env"):
        path = Path(env_name)
        if not path.exists():
            continue
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_local_env()
MERCHANTMIND_ENV = os.getenv("MERCHANTMIND_ENV", "dev").strip().lower()
IS_PROD = MERCHANTMIND_ENV == "prod"
DEFAULT_DEMO_MERCHANT = os.getenv("MERCHANTMIND_DEMO_MERCHANT", "M00001")
GOOGLE_MAPS_JS_API_KEY = os.getenv("GOOGLE_MAPS_JS_API_KEY", "").strip()
FRONTEND_DIR = Path("frontend")
FRONTEND_INDEX = FRONTEND_DIR / "index.html"

app = FastAPI(
    title="MerchantMind API",
    description="AI-powered payment intelligence platform for Australian merchants",
    version="1.0.0",
)

if hasattr(app, "add_middleware"):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not IS_PROD else ["https://dashboard.merchantmind.dev"],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

if hasattr(app, "mount") and FRONTEND_DIR.exists():
    try:
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    except Exception:
        logger.warning("static_mount_skipped", frontend_dir=str(FRONTEND_DIR))

router_agent = MerchantMindRouter()
feature_engine = MerchantFeatureEngine()
inference_service = AnomalyInferenceService()

try:
    warehouse = DuckDBWarehouse()
except DuckDBUnavailableError:
    warehouse = None


class TransactionRequest(BaseModel):
    transaction_id: str
    merchant_id: str
    amount_aud: float = Field(gt=0, le=500_000)
    payment_terminal: str
    timestamp: str


class TransactionScoreResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    risk_level: str
    explanation: List[dict]
    recommended_action: str


class AgentQueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    merchant_id: str


class AgentQueryResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    request_id: str
    sources: Optional[List[str]] = None


class ScoreRequest(BaseModel):
    merchant_id: str
    amount_aud: float = Field(gt=0, le=500_000)
    payment_terminal: str = "tap_and_go"
    hour_of_day: int = Field(ge=0, le=23)
    card_type: str = "debit"
    state: Optional[str] = None
    timestamp: Optional[str] = None


async def decode_and_verify_token(token: str) -> dict[str, str] | None:
    # TODO: replace with real JWT validation in production.
    if not token:
        return None
    if token.startswith("merchant:"):
        merchant_id = token.split(":", 1)[1].strip()
        if merchant_id:
            return {"merchant_id": merchant_id, "auth_mode": "token"}
    return None


async def verify_merchant(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict[str, str]:
    token = getattr(credentials, "credentials", "") if credentials else ""
    merchant = await decode_and_verify_token(token)
    if merchant:
        return merchant

    if IS_PROD:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return {"merchant_id": DEFAULT_DEMO_MERCHANT, "auth_mode": "dev_bypass"}


def _enforce_merchant_scope(request_merchant_id: str, merchant: dict[str, str]) -> None:
    if merchant.get("auth_mode") == "dev_bypass" and not IS_PROD:
        return
    if request_merchant_id != merchant.get("merchant_id"):
        raise HTTPException(status_code=403, detail="Access denied")


def _query_records(sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    if warehouse is None:
        raise HTTPException(status_code=503, detail="DuckDB backend is not available")
    try:
        return warehouse.query_records(sql, params)
    except Exception as exc:
        logger.error("duckdb_query_failed", error=str(exc), sql=sql)
        raise HTTPException(status_code=500, detail="DuckDB query failed") from exc


def _query_first(sql: str, params: list[Any] | None = None) -> dict[str, Any]:
    rows = _query_records(sql, params)
    return rows[0] if rows else {}


def _ensure_model_loaded() -> bool:
    try:
        if not inference_service.artifacts:
            inference_service.load()
        return True
    except Exception as exc:
        logger.warning("model_load_failed", error=str(exc))
        return False


def _current_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_model_threshold(default: float = 0.85) -> float:
    if _ensure_model_loaded():
        try:
            return float(inference_service.artifacts.get("metrics", {}).get("threshold", default))
        except Exception:
            return default
    return default


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    if hasattr(response, "headers"):
        response.headers["X-Response-Time-Ms"] = str(round(duration * 1000, 2))
    logger.info(
        "request_completed",
        path=getattr(request.url, "path", "unknown"),
        duration_ms=round(duration * 1000, 2),
    )
    return response


@app.get("/")
async def root_dashboard():
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX))
    raise HTTPException(status_code=404, detail="Frontend dashboard not found")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "env": MERCHANTMIND_ENV,
        "timestamp": _current_utc_iso(),
    }


@app.get("/api/health")
async def api_health():
    transactions_loaded = 0
    training_features_loaded = 0
    duckdb_ready = False

    if warehouse is not None:
        try:
            transactions_loaded = int(_query_first("SELECT COUNT(*) AS n FROM transactions").get("n", 0) or 0)
            training_features_loaded = int(
                _query_first("SELECT COUNT(*) AS n FROM training_features").get("n", 0) or 0
            )
            duckdb_ready = True
        except HTTPException:
            duckdb_ready = False

    model_loaded = _ensure_model_loaded()

    return {
        "status": "healthy" if duckdb_ready else "degraded",
        "env": MERCHANTMIND_ENV,
        "duckdb_ready": duckdb_ready,
        "transactions_loaded": transactions_loaded,
        "training_features_loaded": training_features_loaded,
        "model_loaded": model_loaded,
        "timestamp": _current_utc_iso(),
    }


@app.get("/api/config/public")
async def public_config():
    return {
        "map_provider": "google" if GOOGLE_MAPS_JS_API_KEY else "leaflet",
        "google_maps_js_api_key": GOOGLE_MAPS_JS_API_KEY,
    }


@app.get("/api/dashboard/overview")
async def dashboard_overview():
    return _query_first(
        """
        SELECT
            COUNT(*) AS total_transactions,
            COUNT(DISTINCT merchant_id) AS total_merchants,
            ROUND(COALESCE(SUM(amount_aud), 0), 2) AS total_volume_aud,
            ROUND(COALESCE(AVG(amount_aud), 0), 2) AS avg_transaction_aud,
            ROUND(
                COALESCE(
                    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0),
                    0
                ),
                4
            ) AS fraud_rate_pct,
            COUNT(DISTINCT state) AS states_covered
        FROM transactions
        """
    )


@app.get("/api/dashboard/revenue-by-category")
async def revenue_by_category():
    return _query_records(
        """
        SELECT
            merchant_category AS category,
            ROUND(SUM(amount_aud), 2) AS total_revenue,
            COUNT(*) AS transaction_count,
            ROUND(AVG(amount_aud), 2) AS avg_ticket
        FROM transactions
        GROUP BY merchant_category
        ORDER BY total_revenue DESC
        """
    )


@app.get("/api/dashboard/revenue-trend")
async def revenue_trend():
    rows = _query_records(
        """
        SELECT
            CAST(timestamp AS DATE) AS date,
            ROUND(SUM(amount_aud), 2) AS daily_revenue,
            COUNT(*) AS daily_transactions,
            SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS daily_fraud_count
        FROM transactions
        GROUP BY CAST(timestamp AS DATE)
        ORDER BY date
        """
    )
    for row in rows:
        if "date" in row and row["date"] is not None:
            row["date"] = str(row["date"])
    return rows


@app.get("/api/dashboard/fraud-by-state")
async def fraud_by_state():
    return _query_records(
        """
        SELECT
            state,
            COUNT(*) AS total_transactions,
            SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_count,
            ROUND(
                SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0),
                4
            ) AS fraud_rate_pct,
            ROUND(SUM(CASE WHEN is_fraud THEN amount_aud ELSE 0 END), 2) AS fraud_value_aud
        FROM transactions
        GROUP BY state
        ORDER BY fraud_count DESC
        """
    )


@app.get("/api/dashboard/hourly-pattern")
async def hourly_pattern():
    return _query_records(
        """
        SELECT
            EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP)) AS hour,
            COUNT(*) AS transaction_count,
            ROUND(AVG(amount_aud), 2) AS avg_amount,
            SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_count
        FROM transactions
        GROUP BY EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP))
        ORDER BY hour
        """
    )


@app.get("/api/dashboard/payment-methods")
async def payment_methods():
    return _query_records(
        """
        SELECT
            payment_terminal,
            COUNT(*) AS count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage,
            ROUND(AVG(amount_aud), 2) AS avg_amount,
            ROUND(
                SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0),
                4
            ) AS fraud_rate_pct
        FROM transactions
        GROUP BY payment_terminal
        ORDER BY count DESC
        """
    )


@app.get("/api/dashboard/top-merchants")
async def top_merchants(limit: int = Query(default=10, ge=1, le=50)):
    safe_limit = max(1, min(int(limit), 50))
    return _query_records(
        f"""
        SELECT
            merchant_id,
            merchant_category,
            state,
            COUNT(*) AS transaction_count,
            ROUND(SUM(amount_aud), 2) AS total_revenue,
            ROUND(AVG(amount_aud), 2) AS avg_ticket
        FROM transactions
        GROUP BY merchant_id, merchant_category, state
        ORDER BY total_revenue DESC
        LIMIT {safe_limit}
        """
    )


@app.get("/api/dashboard/anomaly-summary")
async def anomaly_summary():
    try:
        return _query_first(
            """
            SELECT
                COUNT(*) AS total_reviewed,
                SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS total_fraud,
                ROUND(AVG(amount_zscore), 4) AS avg_zscore,
                ROUND(MAX(amount_zscore), 4) AS max_zscore,
                ROUND(AVG(CASE WHEN is_fraud THEN amount_zscore ELSE NULL END), 4) AS avg_fraud_zscore
            FROM training_features
            """
        )
    except HTTPException:
        return {
            "total_reviewed": 0,
            "total_fraud": 0,
            "avg_zscore": 0.0,
            "max_zscore": 0.0,
            "avg_fraud_zscore": 0.0,
        }


@app.get("/api/dashboard/threat-map")
async def threat_map(days: int = Query(default=30, ge=1, le=365)):
    rows = _query_records(
        """
        SELECT
            state,
            COUNT(*) AS transaction_count,
            SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_count,
            ROUND(SUM(CASE WHEN is_fraud THEN amount_aud ELSE 0 END), 2) AS fraud_value_aud,
            ROUND(
                SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0),
                4
            ) AS fraud_rate_pct
        FROM transactions
        WHERE CAST(timestamp AS DATE) >= CURRENT_DATE - ?::INTEGER
        GROUP BY state
        ORDER BY state
        """,
        [days],
    )

    by_state = {str(row["state"]): row for row in rows}
    states = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]
    normalized = []
    for state in states:
        row = by_state.get(state, {})
        normalized.append(
            {
                "state": state,
                "transaction_count": int(row.get("transaction_count", 0) or 0),
                "fraud_count": int(row.get("fraud_count", 0) or 0),
                "fraud_value_aud": float(row.get("fraud_value_aud", 0.0) or 0.0),
                "fraud_rate_pct": float(row.get("fraud_rate_pct", 0.0) or 0.0),
            }
        )

    return {
        "window_days": days,
        "states": normalized,
        "total_fraud_count": int(sum(item["fraud_count"] for item in normalized)),
        "total_fraud_value_aud": round(sum(item["fraud_value_aud"] for item in normalized), 2),
    }


@app.get("/api/live-feed")
async def live_feed(limit: int = Query(default=10, ge=5, le=50)):
    safe_limit = max(5, min(int(limit), 50))
    rows = _query_records(
        f"""
        WITH recent AS (
            SELECT
                transaction_id,
                merchant_id,
                merchant_category,
                state,
                amount_aud,
                payment_terminal,
                card_type,
                CAST(timestamp AS TIMESTAMP) AS ts,
                COALESCE(AVG(amount_aud) OVER (PARTITION BY merchant_id), 0.0) AS merchant_avg,
                COALESCE(STDDEV_POP(amount_aud) OVER (PARTITION BY merchant_id), 0.0) AS merchant_std,
                is_fraud
            FROM transactions
            ORDER BY CAST(timestamp AS TIMESTAMP) DESC
            LIMIT {safe_limit}
        )
        SELECT
            transaction_id,
            merchant_id,
            merchant_category,
            state,
            amount_aud,
            payment_terminal,
            card_type,
            ts AS timestamp,
            merchant_avg,
            merchant_std,
            is_fraud
        FROM recent
        ORDER BY timestamp DESC
        """
    )

    threshold = _get_model_threshold(default=0.85)
    feed: list[dict[str, Any]] = []
    for row in rows:
        avg_amount = float(row.get("merchant_avg", 0.0) or 0.0)
        std_amount = float(row.get("merchant_std", 0.0) or 0.0)
        amount_aud = float(row.get("amount_aud", 0.0) or 0.0)
        amount_zscore = (amount_aud - avg_amount) / max(std_amount, 0.01)

        # Lightweight online risk proxy for feed animation.
        score = 0.18
        score += min(abs(amount_zscore) / 9.0, 0.55)
        score += 0.18 if str(row.get("payment_terminal")) == "manual_entry" else 0.0
        score += 0.08 if str(row.get("card_type")) == "credit" else 0.0
        score += 0.20 if bool(row.get("is_fraud")) else 0.0
        score = float(max(0.0, min(1.0, score)))

        flagged = bool(score > threshold or bool(row.get("is_fraud")))
        feed.append(
            {
                "transaction_id": str(row.get("transaction_id", "")),
                "merchant_id": str(row.get("merchant_id", "")),
                "merchant_category": str(row.get("merchant_category", "")),
                "state": str(row.get("state", "")),
                "amount_aud": round(amount_aud, 2),
                "payment_terminal": str(row.get("payment_terminal", "")),
                "timestamp": str(row.get("timestamp", "")),
                "anomaly_score": round(score, 4),
                "is_anomaly": flagged,
                "risk_level": classify_risk(score),
            }
        )

    return {
        "threshold_used": round(threshold, 4),
        "count": len(feed),
        "transactions": feed,
    }


@app.post("/api/agent/query")
async def api_agent_query(request: AgentQueryRequest, merchant: dict = Depends(verify_merchant)):
    _enforce_merchant_scope(request.merchant_id, merchant)

    question = request.question.lower()
    merchant_id = request.merchant_id

    if any(word in question for word in ["revenue", "total", "earned", "sales", "money"]):
        result = _query_first(
            """
            SELECT
                ROUND(SUM(amount_aud), 2) AS total_revenue,
                COUNT(*) AS transaction_count,
                ROUND(AVG(amount_aud), 2) AS avg_transaction
            FROM transactions
            WHERE merchant_id = ?
            """,
            [merchant_id],
        )
        response = (
            f"Merchant {merchant_id} processed {int(result.get('transaction_count', 0))} transactions "
            f"with total revenue of {float(result.get('total_revenue', 0.0)):.2f} AUD "
            f"and average transaction {float(result.get('avg_transaction', 0.0)):.2f} AUD."
        )
    elif any(word in question for word in ["fraud", "suspicious", "anomaly", "risk"]):
        result = _query_first(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_count,
                ROUND(SUM(CASE WHEN is_fraud THEN amount_aud ELSE 0 END), 2) AS fraud_value
            FROM transactions
            WHERE merchant_id = ?
            """,
            [merchant_id],
        )
        response = (
            f"Merchant {merchant_id}: {int(result.get('fraud_count', 0) or 0)} suspicious transactions "
            f"out of {int(result.get('total', 0) or 0)} total, potential fraud value "
            f"{float(result.get('fraud_value', 0.0) or 0.0):.2f} AUD."
        )
    elif any(word in question for word in ["peak", "busy", "hour", "when"]):
        rows = _query_records(
            """
            SELECT
                EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP)) AS hour,
                COUNT(*) AS txn_count
            FROM transactions
            WHERE merchant_id = ?
            GROUP BY EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP))
            ORDER BY txn_count DESC
            LIMIT 3
            """,
            [merchant_id],
        )
        if rows:
            peak_hours = ", ".join(
                f"{int(row['hour']):02d}:00 ({int(row['txn_count'])} txns)"
                for row in rows
                if row.get("hour") is not None
            )
            response = f"Peak hours for {merchant_id}: {peak_hours}"
        else:
            response = f"No hourly data found for merchant {merchant_id}."
    else:
        result = _query_first(
            """
            SELECT
                COUNT(*) AS txn_count,
                ROUND(SUM(amount_aud), 2) AS total,
                merchant_category,
                state
            FROM transactions
            WHERE merchant_id = ?
            GROUP BY merchant_category, state
            """,
            [merchant_id],
        )
        if result:
            response = (
                f"Merchant {merchant_id} is a {result.get('merchant_category', 'unknown')} business "
                f"in {result.get('state', 'unknown')} with {int(result.get('txn_count', 0))} transactions "
                f"totaling {float(result.get('total', 0.0)):.2f} AUD."
            )
        else:
            response = f"No data found for merchant {merchant_id}."

    return {
        "question": request.question,
        "merchant_id": merchant_id,
        "response": response,
        "timestamp": _current_utc_iso(),
    }


@app.post("/api/score")
async def api_score_transaction(request: ScoreRequest, merchant: dict = Depends(verify_merchant)):
    _enforce_merchant_scope(request.merchant_id, merchant)

    baseline = _query_first(
        """
        SELECT
            AVG(amount_aud) AS avg_amount,
            STDDEV(amount_aud) AS std_amount,
            COUNT(*) AS txn_count
        FROM transactions
        WHERE merchant_id = ?
        """,
        [request.merchant_id],
    )

    txn_count = int(baseline.get("txn_count", 0) or 0)
    if txn_count == 0:
        raise HTTPException(status_code=404, detail="Merchant not found")

    avg_amount = float(baseline.get("avg_amount", 0.0) or 0.0)
    std_amount = float(baseline.get("std_amount", 0.0) or 0.0)
    amount_zscore = (request.amount_aud - avg_amount) / max(std_amount, 0.01)

    txn_timestamp = request.timestamp
    if not txn_timestamp:
        txn_timestamp = datetime.now(timezone.utc).replace(
            hour=request.hour_of_day,
            minute=0,
            second=0,
            microsecond=0,
        ).isoformat()

    realtime_txn = {
        "merchant_id": request.merchant_id,
        "amount_aud": request.amount_aud,
        "payment_terminal": request.payment_terminal,
        "card_type": request.card_type,
        "state": request.state or "NSW",
        "timestamp": txn_timestamp,
    }

    try:
        features = feature_engine.compute_realtime_features(
            merchant_id=request.merchant_id,
            current_txn=realtime_txn,
        )
    except Exception as exc:
        logger.warning("feature_compute_fallback", error=str(exc))
        features = {
            "amount_zscore": amount_zscore,
            "hour_of_day": float(request.hour_of_day),
            "is_outside_business_hours": float(request.hour_of_day < 8 or request.hour_of_day > 20),
        }

    features["amount_zscore"] = float(features.get("amount_zscore", amount_zscore))
    features["hour_of_day"] = float(features.get("hour_of_day", request.hour_of_day))

    threshold = 0.85
    score: float
    scoring_mode = "heuristic"
    try:
        if _ensure_model_loaded():
            threshold = _get_model_threshold(default=threshold)
            score = float(inference_service.predict(features).score)
            scoring_mode = "model"
        else:
            score = float(min(1.0, max(0.0, abs(features["amount_zscore"]) / 10.0)))
    except Exception as exc:
        logger.warning("model_score_fallback", error=str(exc))
        score = float(min(1.0, max(0.0, abs(features["amount_zscore"]) / 10.0)))

    return {
        "merchant_id": request.merchant_id,
        "amount_aud": request.amount_aud,
        "anomaly_score": round(score, 4),
        "is_anomaly": bool(score > threshold),
        "risk_level": classify_risk(score),
        "threshold_used": round(threshold, 4),
        "scoring_mode": scoring_mode,
        "explanation": {
            "amount_zscore": round(float(features.get("amount_zscore", amount_zscore)), 3),
            "hour_of_day": int(request.hour_of_day),
            "merchant_avg": round(avg_amount, 2),
            "merchant_std": round(std_amount, 2),
            "is_outside_business_hours": bool(features.get("is_outside_business_hours", 0.0) >= 0.5),
        },
    }


@app.post("/v1/transactions/score", response_model=TransactionScoreResponse)
async def score_transaction(txn: TransactionRequest, merchant: dict = Depends(verify_merchant)):
    _enforce_merchant_scope(txn.merchant_id, merchant)

    features = feature_engine.compute_realtime_features(
        merchant_id=txn.merchant_id,
        current_txn=txn.model_dump(),
    )

    prediction = inference_service.predict(features)
    explanation = _explain_prediction(features, prediction.score)

    return TransactionScoreResponse(
        anomaly_score=prediction.score,
        is_anomaly=prediction.score > 0.85,
        risk_level=classify_risk(prediction.score),
        explanation=explanation,
        recommended_action=get_action(prediction.score),
    )


@app.post("/v1/agent/query", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest, merchant: dict = Depends(verify_merchant)):
    _enforce_merchant_scope(request.merchant_id, merchant)

    result = await router_agent.handle_query(query=request.question, merchant_id=request.merchant_id)
    return AgentQueryResponse(**result)


@app.get("/v1/merchants/{merchant_id}/dashboard")
async def merchant_dashboard(merchant_id: str, merchant: dict = Depends(verify_merchant)):
    _enforce_merchant_scope(merchant_id, merchant)

    return {
        "revenue_trend_30d": await _analytics_revenue_trend(merchant_id, 30),
        "anomaly_summary": await _analytics_anomaly_summary(merchant_id),
        "category_benchmark": await _analytics_benchmark(merchant_id),
        "top_payment_methods": await _analytics_payment_methods(merchant_id),
        "peak_hours": await _analytics_peak_hours(merchant_id),
    }


def classify_risk(score: float) -> str:
    if score >= 0.95:
        return "CRITICAL"
    if score >= 0.85:
        return "HIGH"
    if score >= 0.70:
        return "MEDIUM"
    return "LOW"


def get_action(score: float) -> str:
    if score >= 0.95:
        return "Block transaction and require manual review"
    if score >= 0.85:
        return "Hold transaction for verification"
    if score >= 0.70:
        return "Flag for post-settlement review"
    return "Allow"


def _explain_prediction(features: dict[str, Any], score: float) -> list[dict[str, Any]]:
    _ = score
    ranked = sorted(
        [
            {"feature": "amount_zscore", "value": features.get("amount_zscore", 0)},
            {"feature": "txn_velocity_1h", "value": features.get("txn_velocity_1h", 0)},
            {"feature": "is_outside_business_hours", "value": features.get("is_outside_business_hours", 0)},
        ],
        key=lambda item: abs(float(item["value"])),
        reverse=True,
    )
    return ranked


async def _analytics_revenue_trend(merchant_id: str, days: int) -> dict[str, Any]:
    rows = _query_records(
        """
        SELECT
            CAST(timestamp AS DATE) AS day,
            ROUND(SUM(amount_aud), 2) AS revenue
        FROM transactions
        WHERE merchant_id = ?
          AND CAST(timestamp AS DATE) >= CURRENT_DATE - ?::INTEGER
        GROUP BY CAST(timestamp AS DATE)
        ORDER BY day
        """,
        [merchant_id, days],
    )
    for row in rows:
        row["day"] = str(row.get("day"))
    return {"window_days": days, "series": rows}


async def _analytics_anomaly_summary(merchant_id: str) -> dict[str, Any]:
    row = _query_first(
        """
        SELECT
            SUM(CASE WHEN is_fraud AND amount_aud >= 1000 THEN 1 ELSE 0 END) AS high,
            SUM(CASE WHEN is_fraud AND amount_aud >= 300 AND amount_aud < 1000 THEN 1 ELSE 0 END) AS medium,
            SUM(CASE WHEN is_fraud AND amount_aud < 300 THEN 1 ELSE 0 END) AS low
        FROM transactions
        WHERE merchant_id = ?
        """,
        [merchant_id],
    )
    return {
        "high": int(row.get("high", 0) or 0),
        "medium": int(row.get("medium", 0) or 0),
        "low": int(row.get("low", 0) or 0),
    }


async def _analytics_benchmark(merchant_id: str) -> dict[str, Any]:
    row = _query_first(
        """
        WITH merchant_stats AS (
            SELECT
                merchant_id,
                AVG(amount_aud) AS avg_ticket
            FROM transactions
            GROUP BY merchant_id
        )
        SELECT
            ROUND(
                100.0 * SUM(CASE WHEN merchant_stats.avg_ticket <= target.avg_ticket THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0),
                2
            ) AS percentile
        FROM merchant_stats, (
            SELECT avg_ticket FROM merchant_stats WHERE merchant_id = ?
        ) AS target
        """,
        [merchant_id],
    )
    return {"state_percentile": float(row.get("percentile", 0.0) or 0.0)}


async def _analytics_payment_methods(merchant_id: str) -> list[dict[str, Any]]:
    return _query_records(
        """
        SELECT
            payment_terminal AS method,
            ROUND(COUNT(*) * 1.0 / NULLIF(SUM(COUNT(*)) OVER (), 0), 4) AS share
        FROM transactions
        WHERE merchant_id = ?
        GROUP BY payment_terminal
        ORDER BY share DESC
        """,
        [merchant_id],
    )


async def _analytics_peak_hours(merchant_id: str) -> list[int]:
    rows = _query_records(
        """
        SELECT
            EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP)) AS hour,
            COUNT(*) AS txn_count
        FROM transactions
        WHERE merchant_id = ?
        GROUP BY EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP))
        ORDER BY txn_count DESC
        LIMIT 5
        """,
        [merchant_id],
    )
    return [int(row["hour"]) for row in rows if row.get("hour") is not None]
