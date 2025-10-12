#!/bin/bash
# Run the service locally

set -e

echo "🚀 Starting WarpGBM MCP Service locally..."

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set environment
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run uvicorn
echo "Starting server on http://0.0.0.0:4000"
echo "Docs available at http://0.0.0.0:4000/docs"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 4000 --reload




