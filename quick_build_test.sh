#!/bin/bash
set -e

echo "🚀 Quick Build and Test Script"
echo "=============================="

# Build with cache to speed up
echo "📦 Building container (using cache)..."
docker build -t coqui-tts-web . 

# Remove any existing test container
docker rm -f coqui-test 2>/dev/null || true

echo "🏃 Starting container..."
# Start container
docker run -d \
    --name coqui-test \
    -p 2201:2201 \
    coqui-tts-web

echo "⏳ Waiting 15 seconds for startup..."
sleep 15

echo "🔍 Checking container status..."
docker ps | grep coqui-test

echo "📋 Container logs:"
docker logs --tail 20 coqui-test

echo "🌐 Testing endpoints..."
echo "Testing health endpoint..."
curl -s http://localhost:2201/health | python3 -m json.tool || echo "Health endpoint failed"

echo ""
echo "Testing models endpoint..."
curl -s http://localhost:2201/api/models | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    total_tts = data.get('total_tts_models', 0)
    total_vocoder = data.get('total_vocoder_models', 0)
    print(f'✅ TTS Models: {total_tts}')
    print(f'✅ Vocoder Models: {total_vocoder}')
    if total_tts > 0:
        print('🎯 SUCCESS: Models are loading!')
    else:
        print('❌ ISSUE: No models loaded')
except Exception as e:
    print(f'❌ Error parsing models response: {e}')
"

echo ""
echo "Testing speakers endpoint..."
curl -s http://localhost:2201/api/speakers | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    coqui_count = len(data.get('coqui_speakers', {}))
    custom_count = len(data.get('custom_voices', {}))
    print(f'✅ Coqui Speakers: {coqui_count}')
    print(f'✅ Custom Voices: {custom_count}')
    if coqui_count > 0:
        print('🎯 SUCCESS: Speakers are loading!')
    else:
        print('❌ ISSUE: No speakers loaded')
except Exception as e:
    print(f'❌ Error parsing speakers response: {e}')
"

echo ""
echo "🌍 Web interface available at: http://localhost:2201"
echo ""
echo "To view logs: docker logs coqui-test"
echo "To stop: docker stop coqui-test && docker rm coqui-test"