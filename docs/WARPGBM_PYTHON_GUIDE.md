# WarpGBM Python Package Guide

## 🚀 About WarpGBM

**WarpGBM** is a high-performance, GPU-accelerated Gradient Boosted Decision Tree (GBDT) library built from the ground up with PyTorch and custom CUDA kernels.

- **GitHub Repository**: https://github.com/jefferythewind/warpgbm
- **License**: GPL-3.0
- **Current Version**: v2.1.1
- **Stars**: 91+ ⭐

---

## 🤔 MCP Service vs Python Package

### **This MCP Service** (What You're Using Now)
- ✅ **Quick experimentation** - No installation needed
- ✅ **Cloud GPU training** - Pay-per-use Modal GPUs
- ✅ **Stateless** - Get portable model artifacts
- ✅ **API-based** - Works from any language/agent
- ❌ Limited to MCP tools (train, predict, upload)
- ❌ Can't customize loss functions or callbacks

### **WarpGBM Python Package** (Recommended for Serious Work)
- ✅ **Full control** - All parameters, custom losses, callbacks
- ✅ **Free local GPU** - Use your own hardware
- ✅ **Richer API** - Feature importance, per-era analysis, SHAP
- ✅ **Better performance** - Direct GPU access, no serialization overhead
- ✅ **Reproducibility** - `random_state` parameter for consistent results
- ✅ **Integration** - Works with scikit-learn pipelines, pandas, numpy

**Bottom line**: Use this MCP service for quick tests and demos. For production ML workflows, install the Python package directly.

---

## 📦 Installation

### Standard Installation (Linux/macOS)
```bash
pip install git+https://github.com/jefferythewind/warpgbm.git
```

### Google Colab / Jupyter
```bash
!pip install warpgbm --no-build-isolation
```

### Requirements
- Python 3.8+
- PyTorch with CUDA support
- NVIDIA GPU with CUDA 11.0+

---

## 🎯 Quick Start Examples

### Basic Regression
```python
import numpy as np
from warpgbm import WarpGBM

# Your data
X_train = np.random.randn(1000, 10)
y_train = np.random.randn(1000)

# Train on GPU
model = WarpGBM(
    objective='regression',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    device='cuda'  # Use GPU
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Multiclass Classification
```python
from warpgbm import WarpGBM
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)

# Train with automatic label encoding
model = WarpGBM(
    objective='multiclass',
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1
)

model.fit(X, y)

# Get class probabilities
probabilities = model.predict_proba(X_test)

# Get predicted classes
labels = model.predict(X_test)
```

### Invariant Learning (WarpGBM's Unique Feature!)
```python
import pandas as pd
from warpgbm import WarpGBM

# Financial data with time-based eras
df = pd.read_csv('stock_data.csv')
X = df[[f'feature_{i}' for i in range(50)]].values
y = df['returns'].values
eras = df['era'].values  # Time periods or market regimes

# Train with Directional Era-Splitting (DES)
# Only learns signals that work across ALL eras
model = WarpGBM(
    objective='regression',
    n_estimators=200,
    max_depth=5,
    learning_rate=0.01
)

model.fit(X, y, era_id=eras)

# Identify features that are robust across all eras
per_era_importance = model.get_per_era_feature_importance()
invariant_features = per_era_importance.min(axis=0) > threshold
```

### Feature Importance Analysis
```python
# Get overall feature importance
importance = model.get_feature_importance(normalize=True)

# Get per-era importance (when trained with era_id)
per_era_imp = model.get_per_era_feature_importance(normalize=True)

# Find features that work in ALL environments
stable_features = per_era_imp.min(axis=0) > 0.01
```

---

## ⚡ Key Features

### 1. **GPU-Accelerated Everything**
Custom CUDA kernels for:
- Histogram binning
- Split finding
- Gradient computation
- Inference

**Result**: 10-50x faster than CPU-based GBDTs

### 2. **Invariant Learning (DES Algorithm)**
The only open-source GBDT with native **Directional Era-Splitting**:
- Learns signals stable across distribution shifts
- Perfect for financial ML, time series, scientific data
- Prevents overfitting to spurious correlations

### 3. **Unified Regression + Classification**
- Regression with MSE or custom losses
- Binary classification with log loss
- Multiclass with softmax (automatic label encoding)

### 4. **Blazing Fast Pre-binned Data**
For datasets like Numerai (already quantized):
```python
# WarpGBM auto-detects pre-binned data
X = df[features].astype('int8').values  # Already binned to 0-19
model.fit(X, y)  # Skips binning, 13x faster!
```

### 5. **Production-Ready**
- Early stopping with validation sets
- Feature subsampling (`colsample_bytree`)
- Reproducible results (`random_state`)
- L2 regularization
- Min child weight / split gain constraints

---

## 📊 Full API Reference

### Constructor Parameters
```python
WarpGBM(
    objective='regression',        # 'regression', 'binary', 'multiclass'
    num_bins=10,                   # Histogram bins
    max_depth=3,                   # Tree depth
    learning_rate=0.1,             # Shrinkage
    n_estimators=100,              # Number of trees
    min_child_weight=20,           # Min sum of instance weights
    min_split_gain=0.0,            # Min gain to split
    L2_reg=1e-6,                   # L2 regularization
    colsample_bytree=1.0,          # Feature subsampling
    random_state=None,             # Reproducibility
    device='cuda'                  # 'cuda' or 'cpu'
)
```

### Training Methods
```python
model.fit(
    X,                              # Features (numpy array)
    y,                              # Target (numpy array)
    era_id=None,                    # Era labels for invariant learning
    X_eval=None,                    # Validation features
    y_eval=None,                    # Validation targets
    eval_every_n_trees=10,          # Eval frequency
    early_stopping_rounds=20,       # Early stopping
    eval_metric='mse'               # 'mse', 'rmsle', 'corr', 'logloss', 'accuracy'
)
```

### Prediction Methods
```python
predictions = model.predict(X)              # Regression values or class labels
probabilities = model.predict_proba(X)      # Class probabilities (classification)
importance = model.get_feature_importance() # Feature importance
```

---

## 🔬 Use Cases

### Financial Machine Learning
Learn signals that work across market regimes:
```python
model.fit(X, y, era_id=month_ids)  # Each month = one era
```

### Time Series Forecasting
Robust predictions across distribution shifts.

### Scientific Research
Models that generalize across experimental batches.

### Kaggle Competitions
GPU-accelerated hyperparameter tuning.

### High-Speed Inference
Production systems with millisecond SLAs.

---

## 🆚 Benchmarks

From the WarpGBM README:

**Numerai Dataset** (500k samples, 310 features):
- **WarpGBM**: 49 seconds
- **LightGBM**: 643 seconds
- **Speedup**: 13x faster

**Regression (10k samples, 100 features)**:
- **WarpGBM**: 0.8 seconds
- **XGBoost (CPU)**: 12.3 seconds
- **LightGBM (CPU)**: 8.7 seconds
- **Speedup**: 10-15x faster

---

## 🔗 Resources

- **GitHub Repository**: https://github.com/jefferythewind/warpgbm
- **Issues & Support**: https://github.com/jefferythewind/warpgbm/issues
- **Agent Guide**: https://github.com/jefferythewind/warpgbm/blob/main/AGENT_GUIDE.md
- **Examples**: https://github.com/jefferythewind/warpgbm/tree/main/examples

---

## 🤝 When to Use This MCP vs Python Package

| Scenario | Use MCP Service | Use Python Package |
|----------|----------------|-------------------|
| Quick experiment | ✅ | ✅ |
| No GPU available locally | ✅ | ❌ |
| Need custom loss functions | ❌ | ✅ |
| Production ML pipeline | ❌ | ✅ |
| Feature engineering workflow | ❌ | ✅ |
| Hyperparameter tuning | ❌ | ✅ |
| Cross-validation | ❌ | ✅ |
| Era-aware training | ⚠️ Limited | ✅ Full support |
| Cost-sensitive | ⚠️ Pay per use | ✅ Free (your GPU) |

---

## 💡 Example: Transitioning from MCP to Python

### Using MCP (Quick Test)
```python
import requests
import base64

response = requests.post("https://mcp-service.com/train", json={
    "X": X.tolist(),
    "y": y.tolist(),
    "model_type": "warpgbm",
    "objective": "regression"
})
model_artifact = response.json()["model_artifact_joblib"]
```

### Using Python Package (Production)
```python
from warpgbm import WarpGBM
from sklearn.model_selection import cross_val_score

# Full scikit-learn compatibility
model = WarpGBM(objective='regression', n_estimators=100)

# Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Feature engineering
model.fit(X_train, y_train, 
          X_eval=X_val, y_eval=y_val,
          early_stopping_rounds=20)

# Detailed analysis
importance = model.get_feature_importance()
top_features = X_columns[importance.argsort()[-10:]]
```

---

## 🚧 Roadmap

Upcoming features in WarpGBM Python package:
- Multi-GPU training
- SHAP value computation on GPU
- Feature interaction constraints
- Monotonic constraints
- Custom loss functions
- ONNX export

---

## 📄 License

WarpGBM is licensed under GPL-3.0.  
See: https://github.com/jefferythewind/warpgbm/blob/main/LICENSE

---

## 🙌 Contributing

Pull requests welcome!  
See: https://github.com/jefferythewind/warpgbm

---

**Built with 🔥 by @jefferythewind**  
_"Train smarter. Predict faster. Generalize better."_



