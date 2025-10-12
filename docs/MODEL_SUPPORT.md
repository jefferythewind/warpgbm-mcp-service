# Model Support

WarpGBM MCP Service supports multiple gradient boosting backends, each with their own strengths and parameter sets.

## Available Models

### 🚀 WarpGBM (default)

**GPU-accelerated GBDT with era-aware splitting**

```json
{
  "model_type": "warpgbm",
  "objective": "multiclass",
  "num_class": 3,
  "max_depth": 6,
  "num_trees": 100,
  "learning_rate": 0.1,
  "num_bins": 256,
  "min_child_weight": 1.0,
  "min_split_gain": 0.0,
  "colsample_bytree": 1.0
}
```

**Strengths:**
- ⚡ CUDA-accelerated training
- 🧠 Invariant learning with era-aware splitting
- 🎯 Optimized for temporal data with distribution shift

**Parameters:**
- `num_bins` (2-256): Number of bins for histogram construction
- `min_child_weight` (≥0): Minimum sum of instance weight in child
- `min_split_gain` (≥0): Minimum loss reduction for split
- `colsample_bytree` (0-1): Feature sampling ratio per tree

---

### ⚡ LightGBM

**Microsoft's fast, distributed GBDT**

```json
{
  "model_type": "lightgbm",
  "objective": "multiclass",
  "num_class": 3,
  "max_depth": -1,
  "num_trees": 100,
  "learning_rate": 0.1,
  "num_leaves": 31,
  "min_data_in_leaf": 20,
  "feature_fraction": 1.0,
  "bagging_fraction": 1.0,
  "lambda_l1": 0.0,
  "lambda_l2": 0.0
}
```

**Strengths:**
- 🚄 Extremely fast training on large datasets
- 📊 Industry-standard, battle-tested
- 🎛️ Rich set of hyperparameters and objectives
- 💾 Memory efficient

**Parameters:**
- `num_leaves` (2-131072): Maximum number of leaves in one tree
- `min_data_in_leaf` (≥1): Minimum number of data points in a leaf
- `feature_fraction` (0-1): Feature sampling ratio (like colsample_bytree)
- `bagging_fraction` (0-1): Data sampling ratio per iteration
- `bagging_freq` (≥0): Frequency for bagging (0 = disabled)
- `lambda_l1` (≥0): L1 regularization term
- `lambda_l2` (≥0): L2 regularization term
- `max_depth` (-1 or ≥1): Max tree depth (-1 = no limit)

---

## Common Parameters

These work across all models:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `objective` | string | "multiclass" | "regression", "binary", or "multiclass" |
| `num_class` | int | None | Number of classes (required for multiclass) |
| `max_depth` | int | Model-specific | Maximum tree depth |
| `num_trees` | int | 100 | Number of boosting rounds |
| `learning_rate` | float | 0.1 | Step size shrinkage |

## Choosing a Model

### Use **WarpGBM** when:
- You have temporal/financial data with distribution shift
- You need era-aware robustness
- You have GPU available
- Dataset size: 10K - 10M rows

### Use **LightGBM** when:
- You need maximum training speed
- You have very large datasets (>10M rows)
- You want industry-standard behavior
- CPU-only deployment

## Adding New Models

To add a new model backend:

1. Create an adapter in `app/model_registry.py`:

```python
class MyModelAdapter(ModelAdapter):
    def create_model(self, config: ModelConfig) -> Any:
        # Create your model instance
        pass
    
    def get_config_class(self) -> Type[ModelConfig]:
        # Return your config class
        pass
    
    def supports_early_stopping(self) -> bool:
        return True  # or False
    
    def to_cpu(self, model: Any) -> Any:
        # Convert to CPU if needed
        return model
```

2. Register it:

```python
registry.register("mymodel", MyModelAdapter())
```

3. Add it to the `model_type` enum in `app/models.py`

## Example Requests

### WarpGBM multiclass:

```bash
curl -X POST http://localhost:4000/train \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[1,2],[3,4],[5,6]],
    "y": [0,1,2],
    "model_type": "warpgbm",
    "objective": "multiclass",
    "num_class": 3,
    "num_trees": 50
  }'
```

### LightGBM with regularization:

```bash
curl -X POST http://localhost:4000/train \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[1,2],[3,4],[5,6]],
    "y": [0,1,0],
    "model_type": "lightgbm",
    "objective": "binary",
    "num_trees": 100,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1
  }'
```

## Performance Comparison

Typical training time on 1M rows × 100 features:

| Model | GPU | CPU | Notes |
|-------|-----|-----|-------|
| WarpGBM | ~15s | N/A | Requires CUDA |
| LightGBM | N/A | ~25s | CPU-optimized |

Memory usage (approximate):

| Model | 1M rows | 10M rows |
|-------|---------|----------|
| WarpGBM | ~2GB GPU | ~20GB GPU |
| LightGBM | ~500MB | ~5GB |




