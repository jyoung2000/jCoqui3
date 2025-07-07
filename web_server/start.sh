#!/bin/bash
set -e

echo "🐸 Starting Coqui TTS Web Server..."

# Ensure directories exist and have correct permissions
mkdir -p /app/models /app/voices /app/outputs /app/uploads /tmp/numba_cache

# Set environment variables
export NUMBA_CACHE_DIR=/tmp/numba_cache
export PYTHONPATH="/app:${PYTHONPATH}"
export TTS_HOME=/app/models

echo "📁 Directory structure:"
ls -la /app/

echo "🐍 Python path:"
echo $PYTHONPATH

echo "🔧 Environment variables:"
env | grep -E "(TTS_|NUMBA_|PYTHON)" || true

echo "🚀 Starting web server on port 2201..."
cd /app
exec python /app/web_server/app.py