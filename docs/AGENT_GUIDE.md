# WarpGBM MCP Service - Agent Guide

## 🎯 What is This Service?

**Outsource your GBDT workload to the world's fastest GPU implementation.**

WarpGBM MCP is a **cloud GPU gradient boosting service** that gives AI agents instant access to GPU-accelerated training. Train models on our A10G GPUs, receive portable artifacts, and cache them for millisecond inference. No GPU required on your end.

### 🏗️ How It Works

1. **Train**: POST your data → Train on our A10G GPUs → Get portable model artifact
2. **Cache**: `artifact_id` cached for 5 minutes → Blazing fast predictions
3. **Inference**: Online (via cache) or offline (download artifact for local use)

**Architecture**: Stateless service. No model storage. You own your artifacts. Use them in production, locally, or via our caching layer for fast online serving.

### Available Models

- **WarpGBM**: GPU-accelerated, 13× faster than LightGBM, custom CUDA kernels, invariant learning
- **LightGBM**: CPU-optimized, Microsoft's distributed gradient boosting, battle-tested

---

## 🔗 About the WarpGBM Python Package

**This MCP service is a cloud API wrapper around the WarpGBM Python package.**

### Want More Control? Use WarpGBM Directly!

For production ML workflows, consider using the **WarpGBM Python package** directly:

- **GitHub**: https://github.com/jefferythewind/warpgbm (91+ ⭐)
- **Full Agent Guide**: https://github.com/jefferythewind/warpgbm/blob/main/AGENT_GUIDE.md
- **License**: GPL-3.0

### MCP Service vs Python Package

| Feature | MCP Service (This) | Python Package |
|---------|-------------------|----------------|
| Installation | None needed | `pip install git+https://github.com/jefferythewind/warpgbm.git` |
| GPU Access | Cloud (pay-per-use) | Your local GPU (free) |
| API | REST + MCP tools | Full Python API |
| Control | Limited parameters | Full control + custom losses |
| Features | Train, predict, upload | + Cross-validation, feature importance, era analysis |
| Best For | Quick experiments | Production ML pipelines |

**Use this MCP service for**: Quick tests, prototyping, no local GPU  
**Use Python package for**: Production, research, full control, cost savings

### Installation (Python Package)
```bash
# Standard
pip install git+https://github.com/jefferythewind/warpgbm.git

# Colab
!pip install warpgbm --no-build-isolation
```

See [WARPGBM_PYTHON_GUIDE.md](./WARPGBM_PYTHON_GUIDE.md) for complete Python package documentation.

---

## 📊 Available Models

### WarpGBM
- **Acceleration**: GPU (CUDA)
- **Best for**: Time-series data, financial modeling, temporal datasets with era/time structure
- **Special features**: Era-aware splitting, GPU-accelerated training
- **Status**: CPU training available now, GPU coming soon

### LightGBM  
- **Acceleration**: CPU (highly optimized)
- **Best for**: General-purpose ML, tabular data, large datasets (10K-10M+ rows)
- **Special features**: Fast training, low memory usage, handles categorical features
- **Performance**: 10-100x faster than basic sklearn for large datasets

---

## 🛠️ Available Tools

### 1. `list_models`
**Purpose**: List all available model backends

**Parameters**: None

**Returns**:
```json
{
  "models": ["warpgbm", "lightgbm"],
  "default": "warpgbm"
}
```

**When to use**: To show users what models are available

---

### 2. `train`
**Purpose**: Train a gradient boosting model and get a portable artifact

**Required Parameters**:
- `X`: Feature matrix (2D array of floats) - e.g., `[[1.0, 2.0], [3.0, 4.0]]`
- `y`: Target labels (floats for regression, integers for classification)

**Optional Parameters**:
- `model_type`: `"warpgbm"` or `"lightgbm"` (default: `"warpgbm"`)
- `objective`: `"regression"`, `"binary"`, or `"multiclass"` (default: `"multiclass"`)
- `n_estimators`: Number of trees (default: 100)
- `learning_rate`: Learning rate (default: 0.1)
- `max_depth`: Maximum tree depth (default: 6)
- `num_class`: Number of classes for multiclass (auto-detected if not provided)
- `export_joblib`: Return joblib artifact (default: true)
- `export_onnx`: Return ONNX artifact (default: false, not yet implemented)

**Returns**:
```json
{
  "model_type": "lightgbm",
  "model_artifact_joblib": "base64_encoded_model...",
  "model_artifact_onnx": null,
  "num_samples": 100,
  "num_features": 5,
  "training_time_seconds": 0.234
}
```

**Important Notes**:
- **Minimum samples**: Need at least 2 samples
- **Binary classification**: Must have exactly 2 unique classes
- **Multiclass**: Must have 2+ unique classes
- **Regression**: Use float values in `y`
- **Classification**: Use integer values in `y` (will be cast to int automatically)

**Common Errors & Solutions**:
- ❌ `"Binary classification requires exactly 2 classes, found 3"`
  - ✅ Use `"objective": "multiclass"` instead
  
- ❌ `"Training requires at least 2 samples"`
  - ✅ Provide more data (minimum 2 rows)
  
- ❌ `"X and y shape mismatch"`
  - ✅ Ensure X has same number of rows as y has elements

---

### 3. `predict_from_artifact`
**Purpose**: Run inference using a trained model artifact

**Required Parameters**:
- `X`: Feature matrix for prediction (2D array)
- `model_artifact_joblib`: Base64-encoded joblib model (from `train` response)

**Optional Parameters**:
- `model_artifact_onnx`: Base64-encoded ONNX model (not yet implemented)

**Returns**:
```json
{
  "predictions": [0, 1, 0, 1],
  "num_samples": 4,
  "inference_time_seconds": 0.012
}
```

---

## 💡 Usage Examples

### Example 1: Regression
```json
{
  "X": [[1, 2], [3, 4], [5, 6], [7, 8]],
  "y": [1.5, 3.2, 5.1, 7.3],
  "model_type": "lightgbm",
  "objective": "regression",
  "n_estimators": 50,
  "learning_rate": 0.1
}
```

### Example 2: Binary Classification
```json
{
  "X": [[1, 2], [3, 4], [5, 6], [7, 8]],
  "y": [0, 1, 0, 1],
  "model_type": "lightgbm",
  "objective": "binary",
  "n_estimators": 100
}
```

### Example 3: Multiclass Classification
```json
{
  "X": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
  "y": [0, 1, 2, 0, 1, 2],
  "model_type": "lightgbm",
  "objective": "multiclass",
  "num_class": 3,
  "n_estimators": 100
}
```

### Example 4: Full Training + Inference Workflow
```json
// Step 1: Train
{
  "X": [[1, 2], [3, 4], [5, 6]],
  "y": [0, 1, 0],
  "objective": "binary"
}

// Response includes: model_artifact_joblib = "gASV..."

// Step 2: Predict
{
  "X": [[2, 3], [4, 5]],
  "model_artifact_joblib": "gASV..."
}

// Response: {"predictions": [0, 1]}
```

### Example 5: Iris Dataset (Proper Sample Size)

⚠️ **Important**: WarpGBM uses quantile binning which requires **60+ samples** (not 20!) for robust training. With insufficient data, the model can't learn proper decision boundaries.

```json
// Train on Iris with 60 samples (3× the base 20)
{
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
}

// Response includes artifact_id for smart caching
{
  "artifact_id": "abc123-def456-...",
  "model_artifact_joblib": "H4sIA...",
  "training_time_seconds": 0.0
}

// Fast inference with cached artifact_id (< 100ms)
{
  "artifact_id": "abc123-def456-...",
  "X": [[5,3.4,1.5,0.2], [6.7,3.1,4.4,1.4], [7.7,3.8,6.7,2.2]]
}

// Predictions: [0, 1, 2] ← Perfect classification!
```

---

## 🚀 Best Practices

### For Users (Humans)
1. **Use sufficient training data**: WarpGBM needs **60+ samples** for proper binning (20 samples = poor results!)
2. **Use LightGBM** for general-purpose tasks (it's fast and reliable)
3. **Save the artifact_id** for fast cached predictions (5min TTL, < 100ms inference)
4. **Download the model_artifact_joblib** for offline/production use
5. **Match the objective** to your task (regression vs classification)

### For Agents (AI)
1. **Always validate input shapes** before calling `train`
2. **Use artifact_id for repeated predictions** - it's cached for 5 minutes and much faster
3. **Store model artifacts** from training responses for long-term use
4. **Handle errors gracefully** - parse the error message for actionable feedback
5. **Choose the right objective**:
   - Continuous output → `"regression"`
   - Two classes → `"binary"`
   - 3+ classes → `"multiclass"`
6. **Ensure sufficient data**: Minimum 60+ samples for WarpGBM, 20+ for LightGBM
7. **Start with defaults** (100 trees, 0.1 learning rate, depth 6)

---

## 💰 Pricing (X402)

The service supports X402 micropayments on Base network:

- **Training**: 0.01 USDC per request
- **Inference**: 0.001 USDC per request
- **Model Listing**: Free

*Note: Payment enforcement is currently optional (demo mode)*

---

## 🔗 Service Info

- **Base URL**: `https://warpgbm.ai`
- **MCP Endpoint**: `https://warpgbm.ai/mcp/sse`
- **Health Check**: `GET https://warpgbm.ai/healthz`
- **API Docs**: `GET https://warpgbm.ai/docs`
- **MCP Manifest**: `GET https://warpgbm.ai/.well-known/mcp.json`
- **X402 Pricing**: `GET https://warpgbm.ai/.well-known/x402`

---

## 📝 Common Workflows

### Workflow 1: Quick Classification
1. Call `list_models` to see options
2. Call `train` with your data (X, y) and `objective: "binary"` or `"multiclass"`
3. Save the `model_artifact_joblib` from the response
4. Call `predict_from_artifact` with new data and the saved artifact

### Workflow 2: Compare Models
1. Train the same data with `model_type: "lightgbm"`
2. Train the same data with `model_type: "warpgbm"` 
3. Compare `training_time_seconds` and evaluate predictions
4. Choose the best model for your use case

### Workflow 3: Hyperparameter Tuning
1. Train with different `n_estimators` (50, 100, 200)
2. Train with different `learning_rate` (0.01, 0.1, 0.3)
3. Train with different `max_depth` (3, 6, 10)
4. Evaluate predictions on held-out test data
5. Use the best hyperparameters

---

## ⚠️ Limitations

- **Max data size**: 50 MB per request
- **Cold start**: First request may take 5-15 seconds (CPU container spin-up)
- **Timeout**: 15 minutes max per training request
- **GPU training**: WarpGBM currently trains on CPU (GPU coming soon)
- **ONNX export**: Not yet implemented

---

## 🐛 Troubleshooting

### "Loading tools" forever in Cursor
- **Cause**: MCP server connection issue
- **Fix**: Restart Cursor completely, check URL in settings.json

### Training returns empty response
- **Cause**: Container cold start timeout
- **Fix**: Retry after 10-15 seconds, container will be warm

### "Invalid training data" error
- **Cause**: Wrong objective for data type (e.g., regression with integer labels)
- **Fix**: Match objective to your data type

### "Module not found" errors
- **Cause**: Service deployment issue
- **Fix**: Report to service owner, temporary outage

---

## 📞 Support

- **Service Owner**: jefferythewind
- **Project**: https://github.com/jefferythewind/warpgbm
- **Modal Dashboard**: https://modal.com/apps/tdelise/main/deployed/warpgbm-mcp

---

**Last Updated**: 2025-10-11
**Version**: 1.0.0
**Protocol**: MCP 2024-11-05



