# WarpGBM MCP Service

> Cloud GPU gradient boosting service with portable artifacts and smart caching

🌐 **Production Service**: https://warpgbm.ai  
📡 **MCP Endpoint**: https://warpgbm.ai/mcp/sse  
📖 **API Documentation**: https://warpgbm.ai/docs

## ⚡ What is This?

**Outsource your GBDT workload to the world's fastest GPU implementation.**

WarpGBM MCP is a cloud service that gives AI agents instant access to [WarpGBM's](https://github.com/jefferythewind/warpgbm) GPU-accelerated training. Train models on our A10G GPUs, receive portable artifacts, and cache them for millisecond inference. No GPU required on your end.

### 🏗️ How It Works

1. **Train**: POST your data → Train on our A10G GPUs → Get portable model artifact
2. **Cache**: `artifact_id` cached for 5 minutes → Blazing fast predictions
3. **Inference**: Online (via cache) or offline (download artifact for local use)

**Architecture**: Stateless service. No model storage. You own your artifacts. Use them in production, locally, or via our caching layer for fast online serving.

## 🔗 About the WarpGBM Python Package

**This MCP service is built on the [WarpGBM Python package](https://github.com/jefferythewind/warpgbm).**

### 🐍 WarpGBM Python Package (Recommended for Production)

For production ML workflows, use **WarpGBM directly**:

- **GitHub**: https://github.com/jefferythewind/warpgbm ⭐ 91+
- **Installation**: `pip install git+https://github.com/jefferythewind/warpgbm.git`
- **Agent Guide**: https://github.com/jefferythewind/warpgbm/blob/main/AGENT_GUIDE.md
- **License**: GPL-3.0

### MCP Service vs Python Package

| Feature | MCP Service (This Repo) | Python Package |
|---------|------------------------|----------------|
| **Installation** | None needed | `pip install git+...` |
| **GPU** | Cloud (pay-per-use) | Your GPU (free) |
| **Control** | REST API parameters | Full Python API |
| **Features** | Train, predict, upload | + Cross-validation, feature importance, callbacks |
| **Best For** | Quick experiments, demos | Production pipelines, research |
| **Cost** | $0.01 per training | Free (your hardware) |

**Use this MCP service for**: Quick tests, prototyping, agents without local GPU  
**Use Python package for**: Production ML, research, cost savings, full control

---

## 🎯 MCP Service Features

**Available Models:**
- 🚀 **[WarpGBM](https://github.com/jefferythewind/warpgbm)** - 13× faster than LightGBM. Custom CUDA kernels. Invariant learning.
- ⚡ **LightGBM** - Microsoft's distributed gradient boosting. Battle-tested.

**Key Features:**
- 🧩 **Multi-Model**: Choose between WarpGBM (GPU) and LightGBM (CPU)
- 🚀 **Artifact-Based**: Train → Get portable artifact → Use anywhere
- ⚡ **Smart Caching**: `artifact_id` → 5min TTL → Sub-100ms inference
- 💰 **Pay-per-use**: X402 micropayment support (optional)
- 🤖 **MCP Native**: Direct tool integration for AI agents
- 🔁 **Portable Artifacts**: Download and use locally or in production
- 🌐 **Stateless**: No model storage. You own your artifacts.
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

# The service will be available at https://warpgbm.ai
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

## 🎯 Complete Example: Iris Dataset

Train a multiclass classifier on Iris dataset (**Note**: Use 60+ samples for proper binning with WarpGBM):

```bash
# 1. Train WarpGBM model (60 samples - 3× the base 20)
curl -X POST https://warpgbm.ai/train \
  -H "Content-Type: application/json" \
  -d '{
  "X": [[5.1,3.5,1.4,0.2], [4.9,3,1.4,0.2], [4.7,3.2,1.3,0.2], [4.6,3.1,1.5,0.2], [5,3.6,1.4,0.2],
        [7,3.2,4.7,1.4], [6.4,3.2,4.5,1.5], [6.9,3.1,4.9,1.5], [5.5,2.3,4,1.3], [6.5,2.8,4.6,1.5],
        [6.3,3.3,6,2.5], [5.8,2.7,5.1,1.9], [7.1,3,5.9,2.1], [6.3,2.9,5.6,1.8], [6.5,3,5.8,2.2],
        [7.6,3,6.6,2.1], [4.9,2.5,4.5,1.7], [7.3,2.9,6.3,1.8], [6.7,2.5,5.8,1.8], [7.2,3.6,6.1,2.5],
        [5.1,3.5,1.4,0.2], [4.9,3,1.4,0.2], [4.7,3.2,1.3,0.2], [4.6,3.1,1.5,0.2], [5,3.6,1.4,0.2],
        [7,3.2,4.7,1.4], [6.4,3.2,4.5,1.5], [6.9,3.1,4.9,1.5], [5.5,2.3,4,1.3], [6.5,2.8,4.6,1.5],
        [6.3,3.3,6,2.5], [5.8,2.7,5.1,1.9], [7.1,3,5.9,2.1], [6.3,2.9,5.6,1.8], [6.5,3,5.8,2.2],
        [7.6,3,6.6,2.1], [4.9,2.5,4.5,1.7], [7.3,2.9,6.3,1.8], [6.7,2.5,5.8,1.8], [7.2,3.6,6.1,2.5],
        [5.1,3.5,1.4,0.2], [4.9,3,1.4,0.2], [4.7,3.2,1.3,0.2], [4.6,3.1,1.5,0.2], [5,3.6,1.4,0.2],
        [7,3.2,4.7,1.4], [6.4,3.2,4.5,1.5], [6.9,3.1,4.9,1.5], [5.5,2.3,4,1.3], [6.5,2.8,4.6,1.5],
        [6.3,3.3,6,2.5], [5.8,2.7,5.1,1.9], [7.1,3,5.9,2.1], [6.3,2.9,5.6,1.8], [6.5,3,5.8,2.2],
        [7.6,3,6.6,2.1], [4.9,2.5,4.5,1.7], [7.3,2.9,6.3,1.8], [6.7,2.5,5.8,1.8], [7.2,3.6,6.1,2.5]],
  "y": [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2,2,2,2,2,2,
        0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2,2,2,2,2,2,
        0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2,2,2,2,2,2],
  "model_type": "warpgbm",
  "objective": "multiclass",
  "n_estimators": 100
}'

# Response:
{
  "artifact_id": "abc123-def456-...",
  "model_artifact_joblib": "H4sIA...",
  "training_time_seconds": 0.0
}

# 2. Fast inference using cached artifact_id (< 100ms)
curl -X POST https://warpgbm.ai/predict_from_artifact \
  -H "Content-Type: application/json" \
  -d '{
  "artifact_id": "abc123-def456-...",
  "X": [[5,3.4,1.5,0.2], [6.7,3.1,4.4,1.4], [7.7,3.8,6.7,2.2]]
}'

# Predictions: [0, 1, 2] ← Perfect classification!
```

**Why 60 samples?** WarpGBM uses quantile binning which needs sufficient data per class. With only 20 samples (~6-7 per class), the model can't learn proper decision boundaries. **Always use 60+ samples for robust training.**

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

