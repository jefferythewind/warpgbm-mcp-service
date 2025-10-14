"""
Utility functions for serialization, ONNX export, etc.
"""

import base64
import io
import time
import tempfile
import os
import gzip
from typing import Any, Tuple, Optional
import numpy as np
import joblib


def serialize_model_joblib(model: Any) -> str:
    """
    Serialize a model to base64-encoded, gzip-compressed joblib format.
    
    Args:
        model: Any sklearn-API model (should have .to_cpu() method for GPU models)
        
    Returns:
        Base64-encoded, gzip-compressed string (typically 60-80% smaller)
    """
    # Convert to CPU if needed
    if hasattr(model, "to_cpu"):
        model_cpu = model.to_cpu()
    else:
        model_cpu = model
    
    # Serialize to bytes
    buf = io.BytesIO()
    joblib.dump(model_cpu, buf)
    buf.seek(0)
    
    # Compress with gzip
    compressed = gzip.compress(buf.getvalue())
    
    # Encode to base64
    return base64.b64encode(compressed).decode("utf-8")


def deserialize_model_joblib(artifact_b64: str) -> Any:
    """
    Deserialize a base64-encoded, gzip-compressed joblib model.
    
    WARNING: This should ONLY be used for CPU models (LightGBM).
    WarpGBM models must be loaded with GPU-specific deserializers!
    
    Args:
        artifact_b64: Base64-encoded, gzip-compressed model string
        
    Returns:
        Deserialized model object
    """
    # Decode base64
    compressed_bytes = base64.b64decode(artifact_b64)
    
    # Decompress
    model_bytes = gzip.decompress(compressed_bytes)
    
    # Load from bytes
    buf = io.BytesIO(model_bytes)
    model = joblib.load(buf)
    
    return model


def serialize_model_onnx(model: Any, X_sample: np.ndarray) -> str:
    """
    Convert model to ONNX format and serialize to base64.
    
    Args:
        model: WarpGBM model
        X_sample: Sample input for shape inference
        
    Returns:
        Base64-encoded ONNX model
    """
    try:
        # Check if model has to_onnx method
        if hasattr(model, "to_onnx"):
            onnx_model = model.to_onnx(X_sample)
        else:
            # Fallback: use skl2onnx if model is sklearn-compatible
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        # Serialize ONNX model to bytes
        onnx_bytes = onnx_model.SerializeToString()
        
        # Encode to base64
        return base64.b64encode(onnx_bytes).decode("utf-8")
        
    except Exception as e:
        # ONNX export might not be supported yet
        raise NotImplementedError(f"ONNX export not yet implemented for this model: {e}")


def deserialize_model_onnx(artifact_b64: str):
    """
    Deserialize a base64-encoded ONNX model.
    
    Args:
        artifact_b64: Base64-encoded ONNX model
        
    Returns:
        ONNX InferenceSession
    """
    import onnxruntime as ort
    
    # Decode base64
    onnx_bytes = base64.b64decode(artifact_b64)
    
    # Create inference session
    return ort.InferenceSession(onnx_bytes)


def predict_onnx(session, X: np.ndarray) -> np.ndarray:
    """
    Run inference with ONNX model.
    
    Args:
        session: ONNX InferenceSession
        X: Input features
        
    Returns:
        Predictions
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    result = session.run([output_name], {input_name: X.astype(np.float32)})
    return result[0]


def check_gpu_availability() -> Tuple[bool, Optional[str]]:
    """
    Check if GPU is available for WarpGBM.
    
    Returns:
        Tuple of (is_available, gpu_name)
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return True, gpu_name
        return False, None
    except ImportError:
        # PyTorch not installed, assume no GPU
        return False, None


class Timer:
    """Simple context manager for timing operations"""
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start


def create_temp_workspace():
    """
    Create a temporary directory for job isolation.
    
    Returns:
        TemporaryDirectory context manager
    """
    return tempfile.TemporaryDirectory(prefix="warpgbm_job_")


def validate_array_size(X: list, y: Optional[list] = None, max_mb: int = 50):
    """
    Validate that input arrays are within size limits.
    
    Args:
        X: Feature matrix
        y: Target vector (optional)
        max_mb: Maximum size in megabytes
        
    Raises:
        ValueError if arrays are too large
    """
    import sys
    
    X_size = sys.getsizeof(X) / (1024 * 1024)  # Convert to MB
    
    if y is not None:
        y_size = sys.getsizeof(y) / (1024 * 1024)
        total_size = X_size + y_size
    else:
        total_size = X_size
    
    if total_size > max_mb:
        raise ValueError(
            f"Input data size ({total_size:.2f} MB) exceeds limit ({max_mb} MB). "
            "Consider using dataset URLs or chunking for large datasets."
        )


class ArtifactCache:
    """
    Cache for model artifacts with TTL.
    Supports both in-memory (local) and Modal Dict (production) backends.
    Thread-safe for concurrent requests.
    """
    
    def __init__(self, default_ttl_seconds: int = 300, backend=None):
        """
        Args:
            default_ttl_seconds: Default time-to-live in seconds (default: 5 minutes)
            backend: Optional backend storage (e.g., Modal Dict). If None, uses in-memory dict.
        """
        self.cache = backend if backend is not None else {}
        self.default_ttl = default_ttl_seconds
        self.use_modal = backend is not None
        if not self.use_modal:
            import threading
            self.lock = threading.Lock()
    
    def set(self, artifact_id: str, data: dict, ttl: Optional[int] = None):
        """
        Store an artifact in cache.
        
        Args:
            artifact_id: Unique identifier for the artifact
            data: Dict with keys: artifact, format, model_type
            ttl: Time-to-live in seconds (uses default if None)
        """
        expires_at = time.time() + (ttl or self.default_ttl)
        
        entry = {
            "artifact": data["artifact"],
            "format": data.get("format", "joblib"),
            "model_type": data.get("model_type"),
            "expires_at": expires_at,
            "created_at": time.time()
        }
        
        if self.use_modal:
            # Modal Dict is already thread-safe
            self.cache[artifact_id] = entry
        else:
            with self.lock:
                self.cache[artifact_id] = entry
    
    def get(self, artifact_id: str) -> Optional[dict]:
        """
        Retrieve an artifact from cache.
        
        Args:
            artifact_id: Unique identifier for the artifact
            
        Returns:
            Dict with keys: artifact, format, model_type (if found and not expired), None otherwise
        """
        if self.use_modal:
            # Modal Dict access
            entry = self.cache.get(artifact_id)
            if entry is None:
                return None
            
            # Check if expired
            if time.time() > entry["expires_at"]:
                try:
                    del self.cache[artifact_id]
                except KeyError:
                    pass
                return None
            
            # Return dict
            return {
                "artifact": entry["artifact"],
                "format": entry["format"],
                "model_type": entry.get("model_type")
            }
        else:
            # In-memory access
            with self.lock:
                if artifact_id not in self.cache:
                    return None
                
                entry = self.cache[artifact_id]
                
                # Check if expired
                if time.time() > entry["expires_at"]:
                    del self.cache[artifact_id]
                    return None
                
                # Return dict
                return {
                    "artifact": entry["artifact"],
                    "format": entry["format"],
                    "model_type": entry.get("model_type")
                }
    
    def cleanup_expired(self):
        """Remove all expired entries from cache."""
        current_time = time.time()
        
        if self.use_modal:
            # Modal Dict cleanup (be careful with concurrent access)
            expired_keys = [
                key for key, entry in self.cache.items()
                if current_time > entry["expires_at"]
            ]
            for key in expired_keys:
                try:
                    del self.cache[key]
                except KeyError:
                    pass
        else:
            with self.lock:
                expired_keys = [
                    key for key, entry in self.cache.items()
                    if current_time > entry["expires_at"]
                ]
                
                for key in expired_keys:
                    del self.cache[key]
    
    def size(self) -> int:
        """Return number of items in cache."""
        if self.use_modal:
            return len(self.cache)
        else:
            with self.lock:
                return len(self.cache)


# Global cache instance
artifact_cache = ArtifactCache(default_ttl_seconds=300)  # 5 minutes


