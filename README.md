# MerchantMind

AI-powered payment intelligence platform for Australian SMB merchants in regulated environments.

## What This Repo Contains
- Multi-source data ingestion (ABS, RBA, ASIC-style context) and calibrated transaction generation.
- Feature engineering and anomaly detection training pipeline.
- Agentic assistant (router + SQL + anomaly + RAG agents) with security guardrails.
- FastAPI service surface for scoring and merchant analytics queries.
- Interactive dashboard including live threat map and live transaction feed (DuckDB-backed).
- Governance, monitoring, observability, Terraform, and CI/CD scaffolding.

## Data Truth Model
Ingestion sources are reported with explicit provenance:
- `mode=live`: live source call succeeded and parsed into canonical shape.
- `mode=fallback`: live call attempted but parser/network failed; deterministic fallback table used.
- `mode=calibrated`: intentionally non-live calibration table (portfolio priors).

The canonical run writes `data/external/ingestion_report.json` with:
- per-source status and provenance fields (`live_attempted`, `live_success`, `fallback_used`, `mode`)
- summary counters (`live_success_count`, `fallback_success_count`, `calibrated_count`)

## Data Contract (Canonical)
`TRANSACTIONS` records use this schema:
`transaction_id, merchant_id, merchant_abn, customer_id, amount_aud, currency, timestamp, merchant_category, state, payment_terminal, card_type, is_fraud, is_outside_business_hours`

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
PYTHONPATH=. python data/scripts/run_full_ingestion.py
PYTHONPATH=. python src/feature_store/build_training_features.py
PYTHONPATH=. python data/scripts/build_duckdb.py
PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000

# frontend/dashboard
# open http://localhost:8000

python -m compileall src tests
pytest tests/unit tests/data_contracts -q
```

## Docker (One Command)
From repo root:

```bash
docker compose up --build
```

Then open `http://localhost:8000`.

If port `8000` is already in use:

```bash
HOST_PORT=8010 docker compose up --build
```

Then open `http://localhost:8010`.

Notes:
- Container startup runs ingestion + feature build + DuckDB init automatically.
- First startup can take ~20-60 seconds depending on machine/network.
- `.env.local` is auto-loaded by `docker-compose.yml` for keys like:
  - `OPENAI_API_KEY`
  - `GOOGLE_MAPS_JS_API_KEY`

If you prefer plain Docker:

```bash
docker build -t merchantmind:latest .
docker run --rm -p 8000:8000 --env-file .env.local merchantmind:latest
```

## Auth Mode
- `MERCHANTMIND_ENV=dev` (default): demo endpoints allow auth bypass with default merchant scope (`M00001`).
- `MERCHANTMIND_ENV=prod`: merchant auth token is required (`Authorization: Bearer merchant:<merchant_id>` placeholder format in this repo).

## Repo Layout
- `src/` application and platform code
- `tests/` unit, data contract, integration, smoke tests
- `terraform/` infrastructure code and module placeholders
- `data/` local data directories (`external`, `generated`, etc.)
- `docs/` architecture docs, ADRs, runbooks

## Notes
- Default query backend is local `DuckDB` (`data/generated/merchantmind.duckdb`).
- External systems (AWS, Snowflake, OpenAI) are optional or stubbed where needed.
- Integration tests that call live sources are gated by environment variables.
