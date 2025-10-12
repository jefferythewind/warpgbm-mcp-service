# WarpGBM MCP Service

> Multi-model gradient boosting as a stateless MCP + X402 service

## 🎯 Overview

WarpGBM-MCP is a **universal gradient boosting service** that exposes multiple GBDT backends through a unified Model Context Protocol (MCP) interface. AI agents and developers can choose the best model for their task and get portable artifacts for inference.

**Available Models:**
- 🚀 **[WarpGBM](https://github.com/jefferythewind/warpgbm)** - GPU-accelerated GBDT with era-aware splitting
- ⚡ **LightGBM** - Microsoft's fast, distributed gradient boosting

**Key Features:**
- 🧩 **Multi-Model**: Choose between WarpGBM, LightGBM, and more
- 🚀 **Stateless**: Train → return portable model artifact → no session management
- 💰 **Monetized**: X402 micropayment support for pay-per-use
- 🔒 **Secure**: Sandboxed execution, no filesystem access
- 🌐 **Discoverable**: MCP manifest for automatic agent discovery
- ⚡ **Fast**: GPU-accelerated (WarpGBM) or CPU-optimized (LightGBM)
- 🔁 **Portable**: Returns CPU-compatible models (joblib/ONNX)
- 🔌 **Extensible**: Easy to add new model backends

## 🏗️ Architecture

```
Client (Agent, Script, Human)
      │
      ▼
HTTPS / JSON (MCP + X402)
      │
WarpGBM-MCP FastAPI Service
      │
      ├─ GPU Training (CUDA)
      ├─ CPU Inference
      └─ Portable Artifacts (joblib/ONNX)
```

## 🚀 Quick Start

### Local Development (Ubuntu)

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run locally
uvicorn app.main:app --host 0.0.0.0 --port 4000 --reload

# 4. Test
curl http://localhost:4000/healthz
```

### Expose via Tailscale (for team testing)

Your Ubuntu box is already on your company's Tailscale network. Once running:

```bash
# Find your Tailscale IP
tailscale ip -4

# Service will be available at:
# http://<tailscale-ip>:4000
# or http://<hostname>.tail<tailnet-id>.ts.net:4000
```

Teammates can access it with zero setup if they're on the same Tailnet.

### Expose via Cloudflare Tunnel (for public testing)

```bash
# Install cloudflared (if not already)
brew install cloudflared  # or apt install cloudflared

# Create tunnel
cloudflared tunnel --url http://localhost:4000

# Outputs: https://random-name.trycloudflare.com
```

### Deploy to Modal (production)

```bash
# Install Modal
pip install modal

# Authenticate
modal token new

# Deploy
modal deploy modal_app.py

# Outputs public HTTPS URLs for each endpoint
```

## 📡 API Endpoints

### List Available Models

```bash
GET /models

Response:
{
  "models": ["warpgbm", "lightgbm"],
  "default": "warpgbm"
}
```

### Training

```bash
POST /train
Content-Type: application/json

{
  "X": [[1.0, 2.0], [3.0, 4.0], ...],
  "y": [0, 1, 0, ...],
  "model_type": "warpgbm",  # or "lightgbm"
  "objective": "multiclass",
  "num_class": 3,
  "max_depth": 6,
  "num_trees": 100,
  "learning_rate": 0.1
}

Response:
{
  "model_type": "warpgbm",
  "model_artifact_joblib": "<base64>",
  "model_artifact_onnx": "<base64>",
  "training_time_seconds": 1.234
}
```

### Inference from Artifact

```bash
POST /predict_from_artifact
Content-Type: application/json

{
  "model_artifact": "<base64-encoded-model>",
  "X": [[1.0, 2.0], [3.0, 4.0]],
  "format": "joblib"  # or "onnx"
}

Response:
{
  "predictions": [0, 1]
}
```

### Probability Predictions

```bash
POST /predict_proba_from_artifact
Content-Type: application/json

{
  "model_artifact": "<base64-encoded-model>",
  "X": [[1.0, 2.0]],
  "format": "joblib"
}

Response:
{
  "probabilities": [[0.1, 0.7, 0.2]]
}
```

### Health Check

```bash
GET /healthz

Response:
{
  "status": "ok",
  "gpu_available": true,
  "gpu_name": "NVIDIA A100",
  "version": "1.0.0"
}
```

## 🪙 X402 Payment Flow (Optional)

For monetized deployments:

1. Client fetches pricing: `GET /.well-known/x402`
2. Client pays (on-chain or via gateway)
3. Client verifies payment: `POST /x402/verify` with tx_hash
4. Server returns JWT token
5. Client includes token in subsequent requests: `Authorization: Bearer <token>`

## 🔐 Security

- ✅ Stateless: no user data persisted
- ✅ Sandboxed: runs in temporary directories
- ✅ Size limits: max 50 MB request payload
- ✅ No code execution: only structured JSON parameters
- ✅ Read-only filesystem (when deployed to Modal)
- ✅ Rate limiting: configurable per-IP throttling

## 📦 Project Structure

```
mcp-warpgmb/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app + endpoints
│   ├── x402.py              # Payment verification
│   ├── models.py            # Pydantic schemas
│   └── utils.py             # Helpers (serialization, etc.)
├── .well-known/
│   ├── mcp.json             # MCP capability manifest
│   └── x402                 # X402 pricing manifest
├── tests/
│   ├── test_train.py
│   ├── test_predict.py
│   └── test_integration.py
├── Dockerfile               # Local + cloud container
├── modal_app.py             # Modal deployment
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Test training locally
python tests/test_train.py

# Test full workflow
python tests/test_integration.py

# Compare models
python examples/compare_models.py
```

## 📚 Model Documentation

See [docs/MODEL_SUPPORT.md](docs/MODEL_SUPPORT.md) for detailed information on:
- Available models and their parameters
- Performance comparisons
- When to use each model
- Adding new model backends

## 🌍 Deployment Options

| Environment | Use Case | Command |
|-------------|----------|---------|
| **Local** | Development | `uvicorn app.main:app` |
| **Tailscale** | Team testing | Already accessible at Tailscale IP |
| **Cloudflare Tunnel** | Public demo | `cloudflared tunnel --url http://localhost:4000` |
| **Modal** | Production GPU | `modal deploy modal_app.py` |
| **Docker** | Custom cloud | `docker build -t warpgbm-mcp . && docker run --gpus all -p 4000:4000 warpgbm-mcp` |

## 💰 Pricing (X402)

| Endpoint | Description | Suggested Price |
|----------|-------------|-----------------|
| `/train` | Train model, return artifact | $0.01 |
| `/predict_from_artifact` | Single inference batch | $0.001 |
| `/predict_proba_from_artifact` | Probability inference | $0.001 |

## 🔮 Roadmap

- [x] Core training + inference endpoints
- [x] ONNX export support
- [x] X402 payment verification
- [x] Modal deployment config
- [ ] Async job queue for large datasets
- [ ] S3/IPFS dataset URL support
- [ ] Dynamic pricing based on GPU load
- [ ] Python client library (`WarpGBMClassifierRemote`)
- [ ] Marketplace listing (Cursor, OpenAI)

## 📚 Learn More

- [WarpGBM GitHub](https://github.com/jefferythewind/warpgbm)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [X402 Specification](https://x402.org)
- [Modal Documentation](https://modal.com/docs)

## 📄 License

GPL-3.0 (same as WarpGBM core)

---

Built with ❤️ for the open agent economy

