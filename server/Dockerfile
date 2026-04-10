# TriageOps — AI Customer Support Ops OpenEnv Environment
# Single-stage build using python:3.11-slim. Stays well under the 8GB / 2 vCPU
# hackathon constraints. Port 8000 (HF Space routes 7860 → 8000 via README app_port).

FROM python:3.11-slim

# System deps: curl for healthcheck, git for any VCS deps in pyproject
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first for better layer caching
COPY pyproject.toml requirements.txt ./
COPY uv.lock ./

# Install uv (fast Python package manager) and project dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r requirements.txt && \
    pip install --no-cache-dir openenv-core>=0.2.2

# Copy the rest of the source code
COPY . .

# Expose the OpenEnv standard port
EXPOSE 8000

# Healthcheck via the standard /health endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server (created via openenv create_app() factory)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
