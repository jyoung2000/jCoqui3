#!/bin/bash

echo "🐸 Verifying Coqui TTS Container on Port 2201"
echo "============================================"

# Check if container is running
echo -n "1. Checking if container is running... "
if docker-compose ps | grep -q "coqui-tts-web"; then
    echo "✓ Container is running"
else
    echo "✗ Container is not running"
    echo "   Run: docker-compose up -d"
    exit 1
fi

# Check port 2201
echo -n "2. Checking if port 2201 is accessible... "
if curl -s -o /dev/null -w "%{http_code}" http://localhost:2201 | grep -q "200"; then
    echo "✓ Port 2201 is responding"
else
    echo "✗ Port 2201 is not responding"
    echo "   Container may still be starting up"
fi

# Check health endpoint
echo -n "3. Checking health endpoint... "
HEALTH=$(curl -s http://localhost:2201/health 2>/dev/null)
if echo $HEALTH | grep -q "healthy"; then
    echo "✓ Health check passed"
    echo "   Response: $HEALTH"
else
    echo "✗ Health check failed"
fi

# Check API endpoints
echo -n "4. Checking API endpoints... "
if curl -s http://localhost:2201/api/models > /dev/null 2>&1; then
    echo "✓ API endpoints are accessible"
else
    echo "✗ API endpoints not accessible"
fi

echo ""
echo "Container verification complete!"
echo "Access the web interface at: http://localhost:2201"