#!/bin/bash
# Test the service locally

set -e

echo "🧪 Testing WarpGBM MCP Service locally..."

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run tests
echo "Running pytest..."
pytest tests/ -v --tb=short

# Test health endpoint
echo ""
echo "Testing health endpoint..."
curl -s http://localhost:4000/healthz | jq .

echo ""
echo "✅ All tests passed!"




