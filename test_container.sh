#!/bin/bash
set -e

echo "🐳 Testing Coqui TTS Web Container"
echo "=================================="

# Build the container
echo "📦 Building container..."
docker build -t coqui-tts-web . --no-cache

echo "🚀 Starting container..."
# Remove any existing container
docker rm -f coqui-tts-web-test 2>/dev/null || true

# Start container in detached mode
docker run -d \
    --name coqui-tts-web-test \
    -p 2201:2201 \
    -v "$(pwd)/test_outputs:/app/outputs" \
    coqui-tts-web

echo "⏳ Waiting for container to start (30 seconds)..."
sleep 30

echo "🔍 Checking container logs..."
docker logs coqui-tts-web-test

echo "🌐 Testing web interface..."
# Test health endpoint
if curl -f http://localhost:2201/health; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    docker logs coqui-tts-web-test
    exit 1
fi

# Test main page
if curl -f http://localhost:2201/ >/dev/null 2>&1; then
    echo "✅ Main page accessible"
else
    echo "❌ Main page not accessible"
    docker logs coqui-tts-web-test
    exit 1
fi

# Test API endpoint
if curl -f http://localhost:2201/api/models >/dev/null 2>&1; then
    echo "✅ API endpoints accessible"
else
    echo "❌ API endpoints not accessible"
    docker logs coqui-tts-web-test
    exit 1
fi

echo "🎉 All tests passed!"
echo "🌍 Web GUI is accessible at: http://localhost:2201"
echo ""
echo "To stop the test container:"
echo "docker stop coqui-tts-web-test"
echo "docker rm coqui-tts-web-test"