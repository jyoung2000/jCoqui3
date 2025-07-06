@echo off
REM Build script to force rebuild without cache

echo 🐸 Building Coqui TTS Container (no cache)...
docker-compose build --no-cache

echo ✅ Build complete! Run with:
echo docker-compose up -d
pause