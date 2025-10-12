# WarpGBM MCP Service - Agent Guide

## 🎯 What is This Service?

WarpGBM MCP is a **multi-model gradient boosting training and inference service** that provides:

- **WarpGBM**: GPU-accelerated gradient boosting with era-aware splitting (ideal for temporal/time-series data)
- **LightGBM**: Microsoft's fast, distributed gradient boosting framework

The service returns **portable model artifacts** (joblib format) that can be reused for inference without keeping models in memory.

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

---

## 🚀 Best Practices

### For Users (Humans)
1. **Start with small datasets** to test (< 1000 rows)
2. **Use LightGBM** for general-purpose tasks (it's fast and reliable)
3. **Save the model artifact** from training to reuse for predictions
4. **Match the objective** to your task (regression vs classification)

### For Agents (AI)
1. **Always validate input shapes** before calling `train`
2. **Store model artifacts** from training responses for later use
3. **Handle errors gracefully** - parse the error message for actionable feedback
4. **Choose the right objective**:
   - Continuous output → `"regression"`
   - Two classes → `"binary"`
   - 3+ classes → `"multiclass"`
5. **Start with defaults** (100 trees, 0.1 learning rate, depth 6)

---

## 💰 Pricing (X402)

The service supports X402 micropayments on Base network:

- **Training**: 0.01 USDC per request
- **Inference**: 0.001 USDC per request
- **Model Listing**: Free

*Note: Payment enforcement is currently optional (demo mode)*

---

## 🔗 Service Info

- **Base URL**: `https://tdelise--warpgbm-mcp-serve.modal.run`
- **Health Check**: `GET /healthz`
- **API Docs**: `GET /docs`
- **MCP Manifest**: `GET /.well-known/mcp.json`
- **X402 Pricing**: `GET /.well-known/x402`

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



