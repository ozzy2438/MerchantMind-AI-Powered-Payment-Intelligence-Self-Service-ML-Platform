# MerchantMind

MerchantMind is a portfolio-grade, end-to-end AI payment intelligence platform for Australian SMB merchants.

## Objective
Build a practical, production-grade system that demonstrates how fraud detection, merchant analytics, and AI-assisted querying can run on a single stack without expensive cloud lock-in.  
The project targets three outcomes:
- Detect suspicious transactions with explainable ML signals.
- Give merchants self-service answers through SQL/AI agents and a dashboard.
- Keep ingestion claims technically honest with explicit `live`, `fallback`, and `calibrated` provenance.

## 🎬 Demo

[![🎥 Watch Demo](https://img.shields.io/badge/🎥_Watch_Demo-MerchantMind_Platform_Walkthrough-green?style=for-the-badge)](https://www.canva.com/design/DAHCZTAOlSo/7kvJBRSHHObnwr1oXxrSlg/watch?utm_content=DAHCZTAOlSo&utm_campaign=designshare&utm_medium=embeds&utm_source=link)

## How It Works
1. **Ingestion + truth model**
   - Runs 7 sources through `run_full_ingestion.py`.
   - Every source is labeled as `mode=live`, `mode=fallback`, or `mode=calibrated`.
   - Writes a full provenance report to `data/external/ingestion_report.json`.

2. **Calibrated transaction generation**
   - Uses ingestion outputs to generate a canonical 12-month transaction dataset.
   - Canonical schema:
   `transaction_id, merchant_id, merchant_abn, customer_id, amount_aud, currency, timestamp, merchant_category, state, payment_terminal, card_type, is_fraud, is_outside_business_hours`.

3. **Feature engineering + model training**
   - Builds training features (`training_features.csv`) with behavioural, velocity, temporal, and risk signals.
   - Trains an ensemble anomaly model (`Isolation Forest + XGBoost`) and persists artifacts to `model_artifacts/anomaly_detector.joblib`.

4. **Serving layer (FastAPI + DuckDB)**
   - Loads `transactions` and `training_features` into local DuckDB.
   - Exposes dashboard, agent query, and real-time scoring endpoints from one API app.
   - Keeps `/v1/*` routes for backward compatibility.

5. **Portfolio UI**
   - Vanilla JS + Chart.js dashboard (KPIs, category/hour/state/payment charts, top merchants, anomaly summary).
   - Includes live threat map and live transaction feed.

## What It Achieved
- **Data pipeline coverage:** `7/7` sources succeeded with explicit breakdown: `live_success=0`, `fallback_success=2`, `calibrated=5`.
- **Scale:** `120,000` transactions, `3,000` merchants, `120,000` training feature rows.
- **Data realism target:** fraud rate at `0.355%` in generated data.
- **Model performance (saved artifact):**
  - ROC AUC: `0.8324`
  - Average Precision: `0.0212`
  - Threshold: `0.6165`
  - Test split samples: `24,000` (`65` fraud cases)
- **Product outcome:** one runnable platform delivering ingestion reporting, ML scoring, merchant Q&A, and interactive analytics in a single local deployment.

## API Surface
- Health: `GET /health`, `GET /api/health`
- Dashboard: `GET /api/dashboard/*`
- Agent: `POST /api/agent/query`
- Real-time scoring: `POST /api/score`
- Backward-compatible routes:
  - `POST /v1/transactions/score`
  - `POST /v1/agent/query`
  - `GET /v1/merchants/{merchant_id}/dashboard`

## Quickstart (Local)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
PYTHONPATH=. python data/scripts/run_full_ingestion.py
PYTHONPATH=. python src/feature_store/build_training_features.py
PYTHONPATH=. python data/scripts/build_duckdb.py
PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000
```
Open `http://localhost:8000`.

## Docker (One Command)
```bash
docker compose up --build
```
Open `http://localhost:8000`.

If port `8000` is occupied:
```bash
HOST_PORT=8010 docker compose up --build
```
Open `http://localhost:8010`.

Container startup automatically runs ingestion, feature build, and DuckDB initialisation.

## Environment and Auth
- `MERCHANTMIND_ENV=dev` (default): demo auth bypass with default merchant scope (`M00001`).
- `MERCHANTMIND_ENV=prod`: merchant token required (`Authorization: Bearer merchant:<merchant_id>` placeholder format).

## Repo Layout
- `src/`: application code (agents, API, features, models, governance, monitoring)
- `data/`: external/generated datasets and scripts
- `frontend/`: dashboard UI
- `tests/`: unit, data contract, integration, smoke tests
- `terraform/`: IaC scaffolding
- `docs/`: architecture, ADRs, runbooks
