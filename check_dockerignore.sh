#!/bin/bash
# Check if .dockerignore is preventing README.md from being copied

echo "🔍 Checking .dockerignore configuration..."

if [ -f ".dockerignore" ]; then
    echo "📄 .dockerignore exists"
    
    if grep -q "^\*.md$" .dockerignore; then
        echo "⚠️  Found '*.md' in .dockerignore"
        
        if grep -q "^!README.md$" .dockerignore; then
            echo "✅ Found '!README.md' exception - README.md should be included"
        else
            echo "❌ No '!README.md' exception found - README.md will be excluded!"
            echo "💡 Add '!README.md' after '*.md' in .dockerignore to fix this"
        fi
    else
        echo "✅ No '*.md' exclusion found"
    fi
    
    echo ""
    echo "📋 Current .dockerignore rules affecting markdown files:"
    grep -n "\.md" .dockerignore || echo "  (none found)"
    
else
    echo "❌ .dockerignore not found"
fi

echo ""
echo "📁 Available README files:"
ls -la README*.md 2>/dev/null || echo "  (no README files found)"