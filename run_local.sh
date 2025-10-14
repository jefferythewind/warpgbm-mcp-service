#!/bin/bash
# Run local development server on your GPU

echo "🚀 Starting local WarpGBM dev server..."
echo "💻 Using your local GPU (free!)"
echo "📍 Server will be at http://localhost:8000"
echo "📚 API docs at http://localhost:8000/docs"
echo "🔌 MCP endpoint at http://localhost:8000/mcp/sse"
echo ""

source .venv/bin/activate
uvicorn local_dev:app --reload --host 0.0.0.0 --port 8000


