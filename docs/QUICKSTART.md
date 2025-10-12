# 🚀 Quick Start Guide

Get your multi-model MCP service running in 5 minutes!

## Step 1: Setup Development Environment

```bash
# Clone or navigate to the project
cd mcp-warpgmb

# Run setup script
./scripts/setup_dev.sh

# Activate virtual environment
source .venv/bin/activate
```

## Step 2: Start the Service Locally

```bash
# Start FastAPI server
./scripts/run_local.sh

# Server will be available at:
# http://localhost:4000
# Docs at: http://localhost:4000/docs
```

## Step 3: Test with Example Requests

### List available models

```bash
curl http://localhost:4000/models
```

Response:
```json
{
  "models": ["warpgbm", "lightgbm"],
  "default": "warpgbm"
}
```

### Train a WarpGBM model

```bash
curl -X POST http://localhost:4000/train \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[1,2],[3,4],[5,6],[7,8]],
    "y": [0,1,0,1],
    "model_type": "warpgbm",
    "objective": "binary",
    "num_trees": 10,
    "max_depth": 3
  }'
```

### Train a LightGBM model

```bash
curl -X POST http://localhost:4000/train \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[1,2],[3,4],[5,6],[7,8]],
    "y": [0,1,0,1],
    "model_type": "lightgbm",
    "objective": "binary",
    "num_trees": 10,
    "num_leaves": 15
  }'
```

### Run the comparison example

```bash
python examples/compare_models.py
```

## Step 4: Deploy to Your Network (Tailscale)

If you're on Tailscale, teammates can access your service immediately:

```bash
# Find your Tailscale IP
tailscale ip -4

# Share with team
echo "Service available at: http://$(tailscale ip -4):4000"
```

No firewall configuration needed! 🎉

## Step 5: Deploy to Production (Modal)

```bash
# Install Modal
pip install modal

# Authenticate
modal token new

# Deploy
./scripts/deploy_modal.sh
```

Modal will give you public HTTPS URLs like:
```
https://warpgbm-mcp--fastapi-app.modal.run
```

Update `.well-known/mcp.json` with your production URL and you're live!

## Step 6: Expose via Cloudflare Tunnel (Optional)

For temporary public access without Modal:

```bash
# Install cloudflared
sudo apt install cloudflared  # or brew install cloudflared

# Create tunnel
cloudflared tunnel --url http://localhost:4000

# Outputs: https://random-name.trycloudflare.com
```

## Next Steps

### Add More Models

See `app/model_registry.py` to add new model backends:

```python
class XGBoostAdapter(ModelAdapter):
    # Implement adapter methods
    pass

# Register it
registry.register("xgboost", XGBoostAdapter())
```

### Enable X402 Payments

1. Update `.well-known/x402` with your wallet address
2. Implement real verification in `app/x402.py`
3. Change `verify_payment_optional` to `require_payment` in endpoints

### Monitor Usage

```bash
# Check logs
tail -f /var/log/warpgbm-mcp.log

# Or with Modal
modal logs warpgbm-mcp
```

## Troubleshooting

### Import Error: No module named 'warpgbm'

WarpGBM falls back to sklearn for testing. To use real WarpGBM:

```bash
# Install from your WarpGBM repo
pip install -e /path/to/warpgbm
```

### GPU Not Available

Check CUDA:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If no GPU, LightGBM will work fine on CPU!

### Port Already in Use

```bash
# Kill process on port 4000
sudo lsof -t -i:4000 | xargs kill -9

# Or use a different port
uvicorn app.main:app --port 5000
```

## Full API Documentation

Visit `http://localhost:4000/docs` for interactive Swagger docs.

## Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model_registry.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## Questions?

- **Model Support**: See [docs/MODEL_SUPPORT.md](docs/MODEL_SUPPORT.md)
- **Architecture**: See [README.md](README.md)
- **Issues**: Open an issue on GitHub

---

**You're all set! 🎉**

Your multi-model MCP service is ready to accept training requests from AI agents and developers.




