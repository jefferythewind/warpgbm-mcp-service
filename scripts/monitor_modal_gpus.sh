#!/bin/bash
# Monitor Modal GPU usage in real-time

echo "🔍 Checking Modal GPU usage for warpgbm-mcp..."
echo ""

# Check if modal CLI is available
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Install with: pip install modal"
    exit 1
fi

# Get app stats
echo "📊 App Stats:"
modal app list | grep warpgbm-mcp || echo "   No active deployments found"
echo ""

# Show recent GPU function invocations
echo "🚀 Recent GPU Function Calls (train_warpgbm_gpu):"
modal function logs warpgbm-mcp::train_warpgbm_gpu --tail 20 2>/dev/null || echo "   No GPU function calls yet"
echo ""

# Show active containers
echo "🐳 Active Containers:"
modal container list --app warpgbm-mcp 2>/dev/null || echo "   No active containers"
echo ""

echo "💡 Tips:"
echo "   - GPU containers shut down after 60 seconds idle"
echo "   - CPU containers shut down after 5 minutes idle"
echo "   - Check billing: https://modal.com/tdelise/settings/billing"
echo ""
echo "🔄 Run this script periodically to monitor usage"

