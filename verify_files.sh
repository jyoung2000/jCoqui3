#!/bin/bash
# Verify all required files exist before building

echo "🔍 Verifying required files for Docker build..."

# Check requirements files
echo "📄 Requirements files:"
for file in requirements.txt requirements.dev.txt requirements.notebooks.txt requirements.ja.txt requirements.web.txt; do
    if [ -f "$file" ]; then
        echo "  ✅ $file ($(wc -l < "$file") lines)"
    else
        echo "  ❌ $file - MISSING!"
    fi
done

echo ""
echo "📂 Core files:"
for file in setup.py pyproject.toml MANIFEST.in README.md; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file - MISSING!"
    fi
done

echo ""
echo "📁 Directories:"
for dir in TTS web_server; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/ ($(find "$dir" -name "*.py" | wc -l) Python files)"
    else
        echo "  ❌ $dir/ - MISSING!"
    fi
done

echo ""
echo "🐳 Docker files:"
for file in Dockerfile docker-compose.yml; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file - MISSING!"
    fi
done

echo ""
echo "✅ File verification complete!"
echo "Now you can run: docker-compose build --no-cache"