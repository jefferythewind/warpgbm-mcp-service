#!/bin/bash
# Deploy to Modal

set -e

echo "🚀 Deploying WarpGBM MCP Service to Modal..."

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Install with: pip install modal"
    exit 1
fi

# Check if authenticated
if ! modal token show &> /dev/null; then
    echo "❌ Not authenticated with Modal. Run: modal token new"
    exit 1
fi

# Deploy
echo "Deploying..."
modal deploy modal_app.py

echo ""
echo "✅ Deployment complete!"
echo "Your service is now live on Modal."
echo ""
echo "Next steps:"
echo "1. Check Modal dashboard for your service URL"
echo "2. Update .well-known/mcp.json with your production URL"
echo "3. Test the endpoints with the provided URLs"




