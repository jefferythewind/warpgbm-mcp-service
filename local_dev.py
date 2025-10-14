"""
Local development server - runs WarpGBM directly on your GPU.
Use this for free local development. Deploy to Modal for production.

This module:
1. Defines local GPU training/prediction functions
2. Injects them into app.main
3. Uses app.main.app directly (no code duplication!)

Run with: uvicorn local_dev:app --reload --host 0.0.0.0 --port 8000
"""

import numpy as np
import torch

from app.utils import serialize_model_joblib, deserialize_model_joblib
from app.model_registry import get_registry


def _clear_gpu_cache():
    """Helper to clear GPU cache safely"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_warpgbm_local(X, y, **params):
    """
    Local GPU training function for WarpGBM.
    Equivalent to modal_app.py's train_warpgbm_gpu but runs on your local GPU.
    
    Args:
        X: Feature matrix (list of lists)
        y: Target labels (list)
        **params: WarpGBM hyperparameters
    
    Returns:
        str: Base64-encoded gzipped joblib model artifact
    """
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32 if params.get("objective") == "regression" else np.int32)
    
    # Get model registry and create model
    registry = get_registry()
    ConfigClass = registry.get_config_class("warpgbm")
    config = ConfigClass(**params)
    model = registry.create_model(config)
    
    # Train on local GPU
    model.fit(X, y)
    
    # IMPORTANT: Synchronize CUDA before serialization to ensure all GPU ops are complete
    # This prevents the model from holding invalid GPU memory references
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Serialize model (will pickle GPU tensors, but they're stable after synchronize)
    artifact = serialize_model_joblib(model)
    
    # DON'T delete model or clear cache here - let Python GC handle it naturally
    # Aggressive cleanup causes "illegal memory access" when deserializing later
    
    return artifact


def _load_warpgbm_model_gpu(model_artifact):
    """Helper to load WarpGBM model on GPU"""
    import base64
    import gzip
    import io
    import joblib
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for WarpGBM")
    
    # Deserialize model with GPU support (DON'T force CPU!)
    compressed_bytes = base64.b64decode(model_artifact)
    model_bytes = gzip.decompress(compressed_bytes)
    buf = io.BytesIO(model_bytes)
    
    # Monkey-patch torch.load to force GPU (like Modal does)
    original_load = torch.load
    def gpu_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = 'cuda'  # Force GPU for WarpGBM
        return original_load(*args, **kwargs)
    
    torch.load = gpu_load
    try:
        model = joblib.load(buf)
    finally:
        torch.load = original_load
    
    return model


def predict_warpgbm_local(model_artifact, X):
    """
    Local GPU prediction function for WarpGBM.
    Equivalent to modal_app.py's warpgbm_gpu_predict but runs on your local GPU.
    
    Args:
        model_artifact: Base64-encoded gzipped joblib model
        X: Feature matrix (list of lists)
    
    Returns:
        list: Predictions
    """
    model = _load_warpgbm_model_gpu(model_artifact)
    
    # Predict
    X = np.array(X, dtype=np.float32)
    preds = model.predict(X)
    
    result = preds.tolist() if hasattr(preds, 'tolist') else list(preds)
    
    return result


def predict_proba_warpgbm_local(model_artifact, X):
    """
    Local GPU probability prediction function for WarpGBM.
    
    Args:
        model_artifact: Base64-encoded gzipped joblib model
        X: Feature matrix (list of lists)
    
    Returns:
        list: Probability predictions (n_samples, n_classes)
    """
    model = _load_warpgbm_model_gpu(model_artifact)
    
    # Predict probabilities
    X = np.array(X, dtype=np.float32)
    probs = model.predict_proba(X)
    
    result = probs.tolist() if hasattr(probs, 'tolist') else list(probs)
    
    return result


# =============================================================================
# INJECT GPU FUNCTIONS INTO APP.MAIN
# =============================================================================
# This makes app.main use our local GPU functions instead of Modal's remote GPU
from app import main as main_module

main_module._gpu_training_function = train_warpgbm_local
main_module._gpu_predict_function = predict_warpgbm_local
main_module._gpu_predict_proba_function = predict_proba_warpgbm_local

# Import the app (it will now use our GPU functions)
app = main_module.app

print("🚀 Local dev mode: Using app.main with local GPU functions injected")
print(f"📍 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA'}")
print(f"📍 Functions: train={train_warpgbm_local.__name__}, predict={predict_warpgbm_local.__name__}, predict_proba={predict_proba_warpgbm_local.__name__}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("local_dev:app", host="0.0.0.0", port=8000, reload=True)
