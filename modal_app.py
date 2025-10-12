"""
Modal deployment configuration for WarpGBM MCP Service

Deploys the FastAPI app to Modal with GPU support.

Usage:
    modal deploy modal_app.py
"""

import modal

# Create Modal app
app = modal.App("warpgbm-mcp")

# Create Modal Dict for artifact caching (persists across container instances)
artifact_cache_dict = modal.Dict.from_name("artifact-cache", create_if_missing=True)

# Define container image with dependencies
# Use Modal's GPU image which includes CUDA toolkit for WarpGBM's JIT compilation
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    # Install build tools (gcc, g++, clang, make) needed for compiling CUDA extensions
    .apt_install("build-essential", "clang")
    .pip_install(
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "joblib>=1.3.0",
        "PyJWT>=2.8.0",
        "python-multipart>=0.0.6",
        "onnx>=1.15.0",
        "onnxruntime>=1.16.0",
        "skl2onnx>=1.16.0",
        "slowapi>=0.1.9",
        "httpx>=0.24.0",
        # Model backends
        "lightgbm>=4.0.0",
        "scikit-learn>=1.3.0",
    )
    # Install torch first (WarpGBM needs it to detect CUDA)
    .pip_install("torch", "wheel")
    # Set CUDA architectures for WarpGBM compilation (A10G = compute_86, also add common ones)
    .env({"TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6;8.9"})  # V100, A100, A10G, H100
    # Now install WarpGBM with --no-build-isolation so it can compile CUDA kernels
    .pip_install("warpgbm", extra_options="--no-build-isolation")
    .add_local_dir("app", remote_path="/root/app")
    .add_local_dir(".well-known", remote_path="/root/.well-known")
    .add_local_file("AGENT_GUIDE.md", remote_path="/root/AGENT_GUIDE.md")
)


# GPU function for WarpGBM training (when implemented)
# GPU functions for WarpGBM (training and prediction)
@app.function(
    image=image,
    gpu="A10G",  # Specific GPU type for WarpGBM
    cpu=4.0,
    memory=16384,  # 16GB RAM
    timeout=600,  # 10 minutes max
    scaledown_window=60,  # Shut down after 1 minute idle (save $$$)
    max_containers=1,  # Max 1 GPU at a time (safest for cost control)
)
def warpgbm_gpu_predict(model_artifact: str, X):
    """
    GPU-accelerated WarpGBM prediction.
    WarpGBM is GPU-only, so predictions must also run on GPU.
    
    Args:
        model_artifact: Base64-encoded gzipped joblib model
        X: Feature matrix (list of lists)
    
    Returns:
        list: Predictions
    """
    import sys
    sys.path.insert(0, "/root")
    
    import numpy as np
    import torch
    import joblib
    import base64
    import gzip
    import io
    
    # Deserialize model with GPU support (DON'T force CPU!)
    compressed_bytes = base64.b64decode(model_artifact)
    model_bytes = gzip.decompress(compressed_bytes)
    buf = io.BytesIO(model_bytes)
    
    # Load with GPU mapping (WarpGBM needs CUDA tensors)
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
    
    # Convert input to numpy
    X = np.array(X, dtype=np.float32)
    
    # Predict on GPU
    preds = model.predict(X)
    
    return preds.tolist() if hasattr(preds, 'tolist') else list(preds)


@app.function(
    image=image,
    gpu="A10G",  # Specific GPU type for WarpGBM
    cpu=4.0,
    memory=16384,  # 16GB RAM
    timeout=600,  # 10 minutes max
    scaledown_window=60,  # Shut down after 1 minute idle (save $$$)
    max_containers=1,  # Max 1 GPU at a time (safest for cost control)
)
def train_warpgbm_gpu(X, y, **params):
    """
    GPU-accelerated WarpGBM training.
    
    This function is separate from the main service to:
    1. Control GPU costs (only runs when explicitly called)
    2. Fast scaledown (60s vs 5min)
    3. Limited concurrency (MAX 1 GPU container - safest)
    4. Automatic shutdown after 60s idle
    
    Cost: ~$0.0006/second = ~$2.16/hour (A10G)
    Max cost: $2.16/hour × 1 container = $2.16/hour (even if bombarded with requests)
    
    Args:
        X: Feature matrix (list of lists)
        y: Target labels (list)
        **params: WarpGBM hyperparameters
    
    Returns:
        str: Base64-encoded gzipped joblib model artifact
    """
    import sys
    sys.path.insert(0, "/root")
    
    import numpy as np
    from app.utils import serialize_model_joblib
    from app.model_registry import registry
    
    # Convert lists to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32 if params.get("objective") == "regression" else np.int32)
    
    try:
        # Import WarpGBM (GPU-accelerated)
        from warpgbm.core import WarpGBM
        
        # Create config and model using the registry
        ConfigClass = registry.get_config_class("warpgbm")
        config = ConfigClass(**params)
        model = registry.create_model(config)
        
        # Train on GPU
        model.fit(X, y)
        
        # Keep model on GPU - WarpGBM is GPU-only!
        # Serialize with GPU tensors (will be loaded back on GPU for predictions)
        return serialize_model_joblib(model)
        
    except ImportError as e:
        raise NotImplementedError(
            "WarpGBM library is not installed in this Modal container. "
            "Please add WarpGBM to the Modal image in modal_app.py. "
            f"Error: {e}"
        )


@app.function(
    image=image,
    # CPU-ONLY for main service (healthchecks, LightGBM, inference)
    cpu=2.0,  # 2 vCPUs
    memory=2048,  # 2GB RAM
    timeout=900,  # 15 minutes max per request
    scaledown_window=300,  # Keep warm for 5 minutes
    max_containers=10,  # Max 10 concurrent requests
)
@modal.asgi_app()
def serve():
    """
    Serve the FastAPI app (CPU-only for cost efficiency).
    
    - Healthchecks, MCP endpoints: CPU
    - LightGBM training: CPU (fast enough)
    - WarpGBM training: Delegates to GPU function
    - Inference: CPU (models are already trained)
    
    Cost: ~$0.0001/second = ~$0.36/hour (CPU)
    """
    import sys
    sys.path.insert(0, "/root")
    
    # Initialize the cache with Modal Dict backend
    from app.utils import artifact_cache
    # Modal Dicts are automatically available in all functions within the app
    artifact_cache.__init__(default_ttl_seconds=300, backend=artifact_cache_dict)
    
    # Inject GPU functions into main module
    import app.main
    app.main._gpu_training_function = train_warpgbm_gpu
    app.main._gpu_predict_function = warpgbm_gpu_predict
    
    from app.main import app as fastapi_app
    return fastapi_app

