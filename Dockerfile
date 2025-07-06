# Optimized Multi-stage Dockerfile for Coqui TTS Web Interface
# This version is optimized for CPU usage and Docker Desktop testing
FROM python:3.10-slim AS builder

# Set build environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    wget \
    curl \
    pkg-config \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy and install Python dependencies
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy TTS source files
COPY TTS/ ./TTS/
COPY setup.py pyproject.toml MANIFEST.in ./

# Copy additional requirements files that setup.py expects
COPY requirements*.txt ./

# Install TTS
RUN pip install --no-cache-dir -e .

# Stage 2: Runtime Environment
FROM python:3.10-slim

# Set runtime environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TTS_HOME=/app/models \
    NUMBA_CACHE_DIR=/tmp/numba_cache

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    espeak-data \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app directory structure
WORKDIR /app
RUN mkdir -p \
    /app/models \
    /app/voices \
    /app/outputs \
    /app/uploads \
    /app/static/css \
    /app/static/js \
    /app/static/audio \
    /app/templates \
    /tmp/numba_cache

# Copy TTS library
COPY --from=builder /build/TTS ./TTS

# Copy web server application
COPY web_server/ ./web_server/

# Set Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Create non-root user
RUN groupadd -r ttsuser && \
    useradd -r -g ttsuser -d /app -s /sbin/nologin ttsuser && \
    chown -R ttsuser:ttsuser /app /tmp/numba_cache

# Switch to non-root user
USER ttsuser

# Expose port 2201
EXPOSE 2201

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/web_server/health_check.py || exit 1

# Default command
CMD ["python", "-m", "web_server.app"]