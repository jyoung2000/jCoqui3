#!/bin/bash
# Test what files are included in Docker build context

echo "🐳 Testing Docker build context..."

# Create a simple test Dockerfile that just lists files
cat > Dockerfile.test << 'EOF'
FROM alpine:latest
COPY . /test/
WORKDIR /test
RUN echo "=== Files in build context ===" && \
    ls -la && \
    echo "=== README.md exists? ===" && \
    ls -la README.md 2>/dev/null && echo "✅ README.md found" || echo "❌ README.md missing" && \
    echo "=== Setup files exist? ===" && \
    ls -la setup.py pyproject.toml MANIFEST.in 2>/dev/null || echo "❌ Some setup files missing"
EOF

echo "📦 Building test image to check build context..."
docker build -f Dockerfile.test -t test-context . 2>&1 | grep -A 20 "Files in build context"

# Clean up
rm -f Dockerfile.test
echo "✅ Test complete!"