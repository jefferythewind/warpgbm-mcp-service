#!/bin/bash

echo "🧪 Running WarpGBM MCP Test Suite"
echo "=================================="
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Install test dependencies if needed
if ! python -c "import pytest" 2>/dev/null; then
    echo "📥 Installing pytest..."
    pip install -q pytest pytest-asyncio httpx
fi

echo ""
echo "🔍 Running tests file by file (avoids GPU contamination)..."
echo ""

# Run each test file separately to avoid GPU kernel contamination
total_passed=0
total_failed=0

for test_file in tests/test_*.py; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📝 Running: $(basename $test_file)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Run pytest and capture results
    pytest "$test_file" -v --tb=short
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $(basename $test_file) - ALL PASSED"
    else
        echo "❌ $(basename $test_file) - SOME FAILURES"
        ((total_failed++))
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Test Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $total_failed -eq 0 ]; then
    echo "🎉 ALL TEST FILES PASSED!"
    exit 0
else
    echo "⚠️  $total_failed test file(s) had failures"
    exit 1
fi

echo ""
echo "✅ Test suite complete!"

