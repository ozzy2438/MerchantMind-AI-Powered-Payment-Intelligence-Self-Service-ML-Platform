FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    MERCHANTMIND_ENV=dev \
    BOOTSTRAP_DATA_ON_START=1 \
    TRAIN_MODEL_IF_MISSING=0 \
    UVICORN_HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

COPY requirements.txt requirements-docker.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY src ./src
COPY data ./data
COPY frontend ./frontend
COPY model_artifacts ./model_artifacts
COPY scripts ./scripts
COPY README.md pyproject.toml ./

RUN chmod +x ./scripts/start_api.sh \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

ENTRYPOINT ["./scripts/start_api.sh"]
