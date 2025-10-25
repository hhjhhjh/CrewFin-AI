# syntax=docker/dockerfile:1.7-labs
FROM python:3.11-slim AS base


# --- system deps (node for MCP, curl for health/debug) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
nodejs npm curl ca-certificates \
&& rm -rf /var/lib/apt/lists/*


ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1 \
PIP_NO_CACHE_DIR=1


WORKDIR /app


# install deps first (better layer caching)
COPY requirements.txt ./
RUN pip install -r requirements.txt


# copy project
COPY . .


# security: create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser


# MCP helper script is optional; container can run without it
# default command runs FastAPI server; override via docker run ...
EXPOSE 8000
CMD ["uvicorn", "server.app_server:app", "--host", "0.0.0.0", "--port", "8000"]