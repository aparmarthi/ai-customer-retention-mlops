# ── FastAPI churn inference service ──────────────────────────────────────────
# Slim Python base keeps the image ~400 MB instead of ~1.5 GB.
# PyTorch / SHAP / training deps are intentionally excluded (requirements-serve.txt).

FROM python:3.11-slim

# Prevents .pyc files and enables real-time log output in Cloud Run / Render
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install OS deps needed by LightGBM, pandas, and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer — only rebuilds when requirements change)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy source and champion artifacts
COPY src/ ./src/
COPY artifacts/champion/ ./artifacts/champion/

# Non-root user — security baseline for any production deployment
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Health check — lets Docker and orchestrators detect unresponsive containers
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# PORT env var is injected by Cloud Run and Render at runtime
CMD ["sh", "-c", "uvicorn src.serving.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
