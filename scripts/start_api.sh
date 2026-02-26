#!/usr/bin/env sh
set -eu

export PYTHONPATH="${PYTHONPATH:-/app}"
export MERCHANTMIND_ENV="${MERCHANTMIND_ENV:-dev}"
export BOOTSTRAP_DATA_ON_START="${BOOTSTRAP_DATA_ON_START:-1}"
export TRAIN_MODEL_IF_MISSING="${TRAIN_MODEL_IF_MISSING:-0}"
export UVICORN_HOST="${UVICORN_HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

echo "[merchantmind] Environment: ${MERCHANTMIND_ENV}"

if [ "${BOOTSTRAP_DATA_ON_START}" = "1" ]; then
  echo "[merchantmind] Bootstrapping data artifacts..."
  python data/scripts/run_full_ingestion.py
  python src/feature_store/build_training_features.py
  python data/scripts/build_duckdb.py
fi

if [ ! -f "model_artifacts/anomaly_detector.joblib" ]; then
  echo "[merchantmind] WARNING: model_artifacts/anomaly_detector.joblib not found."
  if [ "${TRAIN_MODEL_IF_MISSING}" = "1" ]; then
    echo "[merchantmind] TRAIN_MODEL_IF_MISSING=1 set, training model..."
    python -c "from src.models.anomaly_detection.train import AnomalyDetectionTrainer; print(AnomalyDetectionTrainer('data/generated/training_features.csv').train())"
  else
    echo "[merchantmind] Using heuristic scoring fallback until model artifact is provided."
  fi
fi

echo "[merchantmind] Starting API on ${UVICORN_HOST}:${PORT}"
exec python -m uvicorn src.api.main:app --host "${UVICORN_HOST}" --port "${PORT}"
