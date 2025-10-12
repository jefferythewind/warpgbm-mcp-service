#!/bin/bash
# Setup development environment

set -e

echo "🔧 Setting up WarpGBM MCP development environment..."

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements-dev.txt

# Make scripts executable
chmod +x scripts/*.sh

echo ""
echo "✅ Development environment ready!"
echo ""
echo "Next steps:"
echo "  1. Activate venv: source .venv/bin/activate"
echo "  2. Run locally: ./scripts/run_local.sh"
echo "  3. Run tests: pytest tests/ -v"
echo "  4. Deploy to Modal: ./scripts/deploy_modal.sh"




