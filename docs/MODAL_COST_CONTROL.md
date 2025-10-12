# Modal Cost Control Guide

## 🏗️ Architecture: Two-Function Design

We use **TWO separate Modal functions** to control costs:

### 1. **Main Service** (CPU-only) - Always Running
```python
@app.function(
    cpu=2.0,              # 2 vCPUs (explicit)
    memory=2048,          # 2GB RAM
    concurrency_limit=10, # Max 10 concurrent containers
    scaledown_window=300  # Keep warm 5 min, then shut down
)
@modal.asgi_app()
def serve():
    # Handles: healthchecks, MCP endpoints, LightGBM, inference
```

**Handles:**
- Health checks
- MCP/SSE endpoints
- LightGBM training (CPU-only, fast)
- Model inference (all models)

**Cost:** ~$0.0001/second = ~$0.36/hour

---

### 2. **GPU Function** - Only for WarpGBM
```python
@app.function(
    gpu="A10G",           # GPU ONLY for WarpGBM
    cpu=4.0,
    memory=16384,
    timeout=600,          # Max 10 minutes
    scaledown_window=60,  # Shut down after 60s idle 🔥
    concurrency_limit=2,  # Max 2 GPUs at once
)
def train_warpgbm_gpu(X, y, **params):
    # Handles: WarpGBM training ONLY
```

**Handles:**
- WarpGBM training ONLY (when `model_type="warpgbm"`)

**Cost:** ~$0.0006/second = ~$2.16/hour

**Auto-shutdown:** After 60 seconds idle, GPU container **automatically shuts down** → $0 cost

---

## ⚠️ What Happened Before

Your `serve` function was somehow running with **10 GPUs**, which is expensive!

**Actions taken:**
1. ✅ Stopped the app immediately (`modal app stop warpgbm-mcp`)
2. ✅ Separated CPU and GPU functions
3. ✅ GPU function only called for WarpGBM (future)
4. ✅ Added strict limits and fast scaledown

## Modal Pricing (as of 2024)

**CPU Pricing:**
- **Free tier**: 30 CPU-hours/month
- **Paid**: ~$0.0001/CPU-second (~$0.36/CPU-hour)

**GPU Pricing:**
- **A10G**: ~$0.0006/second (~$2.16/hour) 💸
- **A100**: ~$0.0024/second (~$8.64/hour) 💸💸
- **H100**: ~$0.0096/second (~$34.56/hour) 💸💸💸

**10 GPUs simultaneously = $21.60-$345.60 per hour!** 😱

## Cost Control Commands

### 1. Check what's running
```bash
modal app list                # See deployed apps
modal container list          # See active containers
```

### 2. Stop everything immediately
```bash
modal app stop warpgbm-mcp    # Stop your app
modal app stop --all          # Stop ALL apps (nuclear option)
```

### 3. Monitor usage
```bash
# Check Modal dashboard
https://modal.com/home

# View usage page
https://modal.com/usage
```

### 4. Set up billing alerts
1. Go to https://modal.com/settings/billing
2. Set up email alerts for usage thresholds
3. Set a monthly budget limit

## Before Each Deployment

### Pre-flight Checklist:
```bash
# 1. Review the modal_app.py config
cat modal_app.py | grep -A 5 "@app.function"

# 2. Verify NO gpu= parameter (unless intentional)
cat modal_app.py | grep "gpu="

# 3. Check for explicit CPU limits
cat modal_app.py | grep "cpu="

# 4. Deploy
./deploy.sh

# 5. IMMEDIATELY verify it's CPU-only
modal container list

# 6. If you see unexpected containers, STOP IMMEDIATELY
modal app stop warpgbm-mcp
```

## Safe Deployment Workflow

```bash
# Stop any existing deployment first
modal app stop warpgbm-mcp

# Deploy new version
./deploy.sh

# Monitor for 30 seconds
watch -n 5 'modal container list'

# Check costs
modal usage
```

## How GPU Billing Works

**Example WarpGBM training request:**
1. User calls `/train` with `model_type="warpgbm"`
2. Main CPU service receives request
3. CPU service calls `train_warpgbm_gpu.remote()` → **GPU container starts** (cold start ~10-30s)
4. GPU training runs (you pay for this time)
5. GPU function returns model artifact
6. GPU container stays **idle for 60 seconds** (warm, but idle = still billed)
7. After 60s with no requests → **GPU shuts down automatically** → $0 cost

**Total GPU cost per request:**
- Training time (e.g., 30 seconds) + 60 seconds idle = 90 seconds max
- 90 seconds × $0.0006/second = **$0.054 per training request**

**No WarpGBM requests = No GPU costs!** The GPU function doesn't exist until called.

---

## GPU Cost Control Features

✅ **Separate function**: GPU only runs when explicitly called for WarpGBM  
✅ **Fast scaledown**: 60 seconds idle (vs 5 minutes for CPU)  
✅ **Concurrency limit**: Max 2 GPUs at once (prevents runaway costs)  
✅ **Timeout**: 10 minutes max per training job  
✅ **Specific GPU**: A10G only (not the expensive H100s)  

---

## When to Use GPUs (Current Status)

**Now:** GPU function exists but raises `NotImplementedError` until WarpGBM is pip-installable

**Future:** When WarpGBM is ready:
1. Uncomment WarpGBM in `modal_app.py`
2. Deploy
3. GPU will auto-scale based on demand
4. You only pay when someone trains WarpGBM models

## Emergency: Stop All Charges

If you get another email about GPU usage:

```bash
# 1. Immediate stop
source .venv/bin/activate
modal app stop --all

# 2. Verify nothing running
modal container list

# 3. Check dashboard
# https://modal.com/usage

# 4. Contact me or Modal support if needed
```

## Current Status

✅ **App stopped**  
✅ **Config updated to CPU-only**  
✅ **Limits added**  

**Safe to redeploy when ready!**

## Testing Before Redeployment

```bash
# 1. Test locally first
./run_local.sh

# 2. When ready for Modal, deploy with monitoring
./deploy.sh && watch -n 5 'source .venv/bin/activate && modal container list'

# 3. Make ONE test request
curl https://tdelise--warpgbm-mcp-serve.modal.run/healthz

# 4. Verify CPU-only in dashboard
# https://modal.com/apps/tdelise/main/deployed/warpgbm-mcp
```

## Summary

- **Before**: Somehow had 10 GPUs running 😱
- **Now**: Explicitly CPU-only with limits ✅
- **Cost**: CPU is ~100x cheaper than GPU
- **Free tier**: 30 CPU-hours/month is plenty for testing

**You're safe to redeploy now with the updated config!**

