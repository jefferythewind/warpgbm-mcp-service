# GPU Limits & Safety

## ✅ Current Configuration

**GPU Function: `train_warpgbm_gpu`**
- **Max GPUs**: 1 (hardcoded via `concurrency_limit=1`)
- **GPU Type**: A10G only
- **Auto-shutdown**: 60 seconds after idle
- **Max runtime**: 10 minutes per training job
- **Trigger**: Only when `model_type="warpgbm"` is requested

**Main Service: `serve`**
- **CPU-only**: No GPU access
- **Handles**: Everything except WarpGBM training

---

## 💰 Cost Control

### Maximum Possible GPU Cost
With `concurrency_limit=1`:
- **Absolute max**: 1 GPU × $2.16/hour = **$2.16/hour**
- Even if 1000 people request WarpGBM training simultaneously, only **1** will run on GPU
- The rest queue up and wait (or fail if timeout is reached)

### Typical Request Cost
- Training time: 10-120 seconds
- Idle time: 60 seconds
- **Total**: 70-180 seconds = **$0.042 - $0.108 per request**

### No Activity Cost
- **$0** when no one is using WarpGBM
- GPU container doesn't exist until called
- CPU service stays on (~$0.36/hour, covered by free tier)

---

## 🔒 Safety Features

### 1. **Concurrency Limit** (Most Important!)
```python
concurrency_limit=1  # Maximum 1 GPU at ANY time
```
- **Prevents**: Runaway scaling (10 GPUs incident)
- **Effect**: Queue or reject excess requests

### 2. **Fast Scaledown**
```python
scaledown_window=60  # Shut down after 60 seconds idle
```
- **Prevents**: Paying for idle GPUs
- **Effect**: GPU shuts down ~1 minute after training completes

### 3. **Timeout**
```python
timeout=600  # Kill after 10 minutes
```
- **Prevents**: Infinite training jobs
- **Effect**: Maximum charge per request = 10 min × $0.036/min = $0.36

### 4. **Specific GPU Type**
```python
gpu="A10G"  # Cheapest ML-capable GPU
```
- **Prevents**: Accidentally using expensive H100s
- **Effect**: $2.16/hour instead of $34.56/hour

### 5. **Separate Function**
- WarpGBM GPU training is isolated from main service
- Main service **cannot** accidentally allocate GPUs
- You must explicitly call `train_warpgbm_gpu.remote()` to use GPU

---

## 🧪 Testing GPU Limits

### Check before deploying:
```bash
python scripts/check_modal_config.py
```

### After deploying:
```bash
# Check active containers
source .venv/bin/activate
modal container list --app warpgbm-mcp

# Should show:
# - 1-2 CPU containers (serve function)
# - 0-1 GPU containers (only when training WarpGBM)
```

### Monitor GPU usage:
```bash
./scripts/monitor_modal_gpus.sh
```

---

## 📊 Comparison: Before vs After

| Scenario | Before | After |
|----------|--------|-------|
| **Main service** | 10 GPUs (bug) | 0 GPUs (CPU-only) |
| **WarpGBM training** | N/A | 1 GPU max |
| **Max concurrent GPUs** | 10+ | 1 |
| **Idle cost** | $21.60+/hour | $0.36/hour (CPU only) |
| **Max GPU cost** | $216/hour | $2.16/hour |
| **Auto-shutdown** | ❌ | ✅ 60s |

---

## 🚦 When GPU is Used

```
User Request:
POST /train
{
  "model_type": "warpgbm",  ← Triggers GPU
  "X": [...],
  "y": [...]
}

Flow:
1. CPU service receives request
2. CPU service calls train_warpgbm_gpu.remote()
3. Modal spins up GPU container (10-30s cold start)
4. GPU trains model
5. GPU returns artifact
6. GPU stays idle for 60s
7. GPU shuts down → $0

User Request:
POST /train
{
  "model_type": "lightgbm",  ← NO GPU
  "X": [...],
  "y": [...]
}

Flow:
1. CPU service receives request
2. CPU trains locally (no GPU)
3. CPU returns artifact
```

---

## 🎛️ Changing GPU Limits

### To allow 2 concurrent GPUs:
```python
# In modal_app.py
concurrency_limit=2  # Max 2 GPUs at once
```

### To disable GPU entirely:
```python
# Comment out the entire train_warpgbm_gpu function
# Or change in app/main.py:
GPU_MODELS = set()  # Empty set = no GPU models
```

---

## ⚠️ Emergency Stop

If you see unexpected GPU usage:
```bash
# Stop ALL Modal apps immediately
source .venv/bin/activate
modal app stop warpgbm-mcp

# Verify nothing running
modal container list

# Check billing
# https://modal.com/settings/billing
```

---

## ✅ Summary

- **Max GPUs**: 1 (hardcoded, cannot exceed)
- **Auto-shutdown**: 60 seconds idle
- **Max cost**: $2.16/hour even if bombarded
- **Idle cost**: $0 (GPU doesn't exist when not used)
- **Safety**: Pre-deployment check script
- **Status**: Ready for production 🚀

