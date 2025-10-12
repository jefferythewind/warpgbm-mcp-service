"""
Main FastAPI application for WarpGBM MCP Service
"""

import os
import uuid
import time
from contextlib import asynccontextmanager

# Global references to GPU functions (injected by modal_app.py)
_gpu_training_function = None
_gpu_predict_function = None
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import numpy as np

from app.models import (
    TrainRequest,
    TrainResponse,
    PredictRequest,
    PredictResponse,
    PredictProbaResponse,
    HealthResponse,
)
from app.utils import (
    serialize_model_joblib,
    serialize_model_onnx,
    deserialize_model_joblib,
    deserialize_model_onnx,
    predict_onnx,
    check_gpu_availability,
    Timer,
    create_temp_workspace,
    validate_array_size,
    artifact_cache,
)
from app.x402 import router as x402_router, verify_payment_optional
from app.model_registry import get_registry
from app.mcp_sse import router as mcp_sse_router
from app import __version__


# Rate limiting
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown logic"""
    # Startup
    gpu_available, gpu_name = check_gpu_availability()
    print(f"🚀 WarpGBM MCP Service v{__version__} starting...")
    print(f"   GPU Available: {gpu_available}")
    if gpu_available:
        print(f"   GPU: {gpu_name}")
    
    yield
    
    # Shutdown
    print("👋 WarpGBM MCP Service shutting down...")


# Create FastAPI app
app = FastAPI(
    title="WarpGBM MCP Service",
    description="GPU-accelerated gradient boosting as a stateless MCP + X402 service",
    version=__version__,
    lifespan=lifespan,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include X402 payment routes
app.include_router(x402_router)
app.include_router(mcp_sse_router)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs"""
    registry = get_registry()
    return {
        "service": "WarpGBM MCP",
        "version": __version__,
        "docs": "/docs",
        "mcp_manifest": "/.well-known/mcp.json",
        "x402_manifest": "/.well-known/x402",
        "available_models": registry.list_models(),
    }


@app.get("/models", tags=["models"])
async def list_models():
    """List all available model backends"""
    registry = get_registry()
    return {
        "models": registry.list_models(),
        "default": "warpgbm"
    }


@app.get("/healthz", response_model=HealthResponse, tags=["health"])
@limiter.limit("10/minute")
async def health_check(request: Request):
    """
    Health check endpoint with GPU status.
    """
    gpu_available, gpu_name = check_gpu_availability()
    
    return HealthResponse(
        status="ok",
        version=__version__,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
    )


@app.post("/train", response_model=TrainResponse, tags=["training"])
@limiter.limit("10/minute")
async def train_model(
    request: Request,
    train_request: TrainRequest,
    paid: bool = Depends(verify_payment_optional),
):
    """
    Train a WarpGBM model and return serialized artifact(s).
    
    **Payment**: Optional for demo, can be enforced by changing dependency to `require_payment`.
    
    **Returns**:
    - Joblib artifact (CPU-compatible)
    - ONNX artifact (optional, platform-independent)
    """
    # Validate input size
    try:
        validate_array_size(train_request.X, train_request.y, max_mb=50)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    
    # Convert to numpy
    try:
        X = np.array(train_request.X, dtype=np.float32)
        y = np.array(train_request.y)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data format: {str(e)}"
        )
    
    # Validate shapes
    if X.shape[0] != len(y):
        raise HTTPException(
            status_code=400,
            detail=f"X and y shape mismatch: X has {X.shape[0]} samples but y has {len(y)} samples"
        )
    
    # Validate minimum samples
    if X.shape[0] < 2:
        raise HTTPException(
            status_code=400,
            detail="Training requires at least 2 samples"
        )
    
    # Validate objective and data compatibility
    if train_request.objective == "regression":
        # For regression, ensure y has continuous values
        pass
    elif train_request.objective == "binary":
        unique_classes = len(np.unique(y))
        if unique_classes != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Binary classification requires exactly 2 classes, found {unique_classes}. "
                       f"Either provide binary data or use 'multiclass' objective."
            )
    elif train_request.objective == "multiclass":
        unique_classes = len(np.unique(y))
        if unique_classes < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Multiclass classification requires at least 2 classes, found {unique_classes}"
            )
        # Ensure num_class is set for multiclass
        if train_request.num_class is None:
            train_request.num_class = unique_classes
    
    # Get model registry
    registry = get_registry()
    
    # Build model config from request
    config_data = {
        "model_type": train_request.model_type,
        "objective": train_request.objective,
        "num_class": train_request.num_class,
    }
    
    # Add all non-None parameters
    for field_name, field_value in train_request.dict(exclude_unset=True).items():
        if field_value is not None and field_name not in ["X", "y", "export_onnx", "export_joblib"]:
            config_data[field_name] = field_value
    
    # Convert num_trees to num_iterations for LightGBM
    if train_request.model_type == "lightgbm" and "num_trees" in config_data:
        config_data["num_iterations"] = config_data.pop("num_trees")
    
    # Check if model requires GPU
    GPU_MODELS = {"warpgbm"}  # Add more GPU models here (e.g., "pytorch", "tensorflow")
    requires_gpu = train_request.model_type in GPU_MODELS
    
    # If GPU required, delegate to GPU function or fail
    if requires_gpu:
        # Check if Modal GPU function is available
        global _gpu_training_function
        gpu_func = _gpu_training_function
        
        if gpu_func is not None:
            try:
                # Call Modal GPU function remotely
                # Use .remote.aio() for async or .remote() for sync within Modal
                if hasattr(gpu_func, 'remote'):
                    model_joblib = gpu_func.remote(X.tolist(), y.tolist(), **config_data)
                else:
                    # Direct call (for local testing)
                    model_joblib = gpu_func(X.tolist(), y.tolist(), **config_data)
                
                # Generate artifact ID and cache (with metadata)
                artifact_id = str(uuid.uuid4())
                artifact_cache.set(
                    artifact_id, 
                    {
                        "artifact": model_joblib,
                        "model_type": train_request.model_type,
                        "format": "joblib"
                    }
                )
                
                return TrainResponse(
                    model_type=train_request.model_type,
                    artifact_id=artifact_id,
                    model_artifact_joblib=model_joblib,
                    num_samples=X.shape[0],
                    num_features=X.shape[1],
                    training_time_seconds=0.0,  # GPU function tracks its own time
                )
            except Exception as e:
                if "NotImplementedError" in str(e):
                    raise HTTPException(
                        status_code=501,
                        detail=f"GPU training for {train_request.model_type} is not yet implemented. "
                               "Coming soon! For now, use model_type='lightgbm' for CPU training."
                    )
                raise HTTPException(
                    status_code=500,
                    detail=f"GPU training failed: {str(e)}"
                )
        else:
            # GPU function not available (probably running locally without Modal)
            raise HTTPException(
                status_code=503,
                detail=f"GPU training for {train_request.model_type} requires Modal deployment. "
                       f"This service is running locally or GPU function is not available. "
                       f"Use model_type='lightgbm' for CPU training, or deploy to Modal for GPU support."
            )
    
    # CPU training path (LightGBM and other CPU models)
    with create_temp_workspace():
        try:
            with Timer() as timer:
                # Get the appropriate config class
                ConfigClass = registry.get_config_class(train_request.model_type)
                config = ConfigClass(**config_data)
                
                # Create and train model
                model = registry.create_model(config)
                model.fit(X, y)
            
            training_time = timer.elapsed
            
            # Convert to CPU if needed
            model_cpu = registry.to_cpu(model, train_request.model_type)
            
            # Serialize artifacts
            model_joblib = None
            
            if train_request.export_joblib:
                model_joblib = serialize_model_joblib(model_cpu)
            
            # Generate artifact ID and cache for fast predictions
            artifact_id = str(uuid.uuid4())
            
            # Cache the artifact for fast predictions (with metadata)
            if model_joblib:
                artifact_cache.set(
                    artifact_id,
                    {
                        "artifact": model_joblib,
                        "model_type": train_request.model_type,
                        "format": "joblib"
                    }
                )
            
            return TrainResponse(
                model_type=train_request.model_type,
                artifact_id=artifact_id,
                model_artifact_joblib=model_joblib,
                num_samples=X.shape[0],
                num_features=X.shape[1],
                training_time_seconds=round(training_time, 4),
            )
            
        except ValueError as e:
            # Validation/data errors - client's fault
            error_msg = str(e).lower()
            if "label type" in error_msg or "classes" in error_msg or "shape" in error_msg:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid training data: {str(e)}"
                )
            raise HTTPException(status_code=400, detail=f"Training validation error: {str(e)}")
        except Exception as e:
            # Unexpected errors - server's fault
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict_from_artifact", response_model=PredictResponse, tags=["inference"])
@limiter.limit("50/minute")
async def predict_from_artifact(
    request: Request,
    predict_request: PredictRequest,
    paid: bool = Depends(verify_payment_optional),
):
    """
    Run inference using a serialized model artifact or cached artifact_id.
    
    **Supports**:
    - Artifact ID (fast, from recent training, valid for 5 minutes)
    - Joblib format (CPU-based, gzip-compressed)
    - ONNX format (cross-platform)
    """
    # Validate input size
    try:
        validate_array_size(predict_request.X, max_mb=50)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    
    X = np.array(predict_request.X, dtype=np.float32)
    
    # Determine artifact source
    model_artifact = predict_request.model_artifact
    artifact_format = predict_request.format
    model_type = None
    
    # Try to get from cache if artifact_id provided
    if predict_request.artifact_id:
        cached = artifact_cache.get(predict_request.artifact_id)
        if cached is None:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact ID '{predict_request.artifact_id}' not found or expired. "
                       "Artifact IDs are valid for 5 minutes. Please use the full model_artifact instead."
            )
        # Cache now always returns dict format
        model_artifact = cached["artifact"]
        artifact_format = cached.get("format", "joblib")
        model_type = cached.get("model_type")
    
    if not model_artifact:
        raise HTTPException(
            status_code=400,
            detail="Either artifact_id or model_artifact must be provided"
        )
    
    # Route WarpGBM to GPU (GPU-only model)
    if model_type == "warpgbm":
        global _gpu_predict_function
        if _gpu_predict_function:
            try:
                start = time.time()
                if hasattr(_gpu_predict_function, 'remote'):
                    preds = _gpu_predict_function.remote(model_artifact, predict_request.X)
                else:
                    preds = _gpu_predict_function(model_artifact, predict_request.X)
                return PredictResponse(
                    predictions=preds,
                    num_samples=len(preds),
                    inference_time_seconds=round(time.time() - start, 4)
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"GPU prediction failed: {str(e)}")
        else:
            raise HTTPException(status_code=503, detail="WarpGBM requires GPU (Modal deployment)")
    
    # CPU predictions for LightGBM and others
    try:
        with Timer() as timer:
            if artifact_format == "joblib":
                model = deserialize_model_joblib(model_artifact)
                predictions = model.predict(X)
            elif artifact_format == "onnx":
                session = deserialize_model_onnx(model_artifact)
                predictions = predict_onnx(session, X)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown format: {artifact_format}")
        
        inference_time = timer.elapsed
        
        # Convert to list
        predictions_list = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)
        
        return PredictResponse(
            predictions=predictions_list,
            num_samples=X.shape[0],
            inference_time_seconds=round(inference_time, 4),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_proba_from_artifact", response_model=PredictProbaResponse, tags=["inference"])
@limiter.limit("50/minute")
async def predict_proba_from_artifact(
    request: Request,
    predict_request: PredictRequest,
    paid: bool = Depends(verify_payment_optional),
):
    """
    Run probability inference using a serialized model artifact or cached artifact_id.
    
    Only works for classification models.
    """
    # Validate input size
    try:
        validate_array_size(predict_request.X, max_mb=50)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    
    X = np.array(predict_request.X, dtype=np.float32)
    
    # Determine artifact source
    model_artifact = predict_request.model_artifact
    artifact_format = predict_request.format
    
    # Try to get from cache if artifact_id provided
    if predict_request.artifact_id:
        cached = artifact_cache.get(predict_request.artifact_id)
        if cached is None:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact ID '{predict_request.artifact_id}' not found or expired. "
                       "Artifact IDs are valid for 5 minutes. Please use the full model_artifact instead."
            )
        model_artifact, artifact_format = cached
    
    if not model_artifact:
        raise HTTPException(
            status_code=400,
            detail="Either artifact_id or model_artifact must be provided"
        )
    
    try:
        with Timer() as timer:
            if artifact_format == "joblib":
                model = deserialize_model_joblib(model_artifact)
                if not hasattr(model, "predict_proba"):
                    raise HTTPException(
                        status_code=400,
                        detail="Model does not support probability prediction"
                    )
                probabilities = model.predict_proba(X)
            elif artifact_format == "onnx":
                # ONNX probability prediction
                session = deserialize_model_onnx(model_artifact)
                # Assume output is probabilities for ONNX
                probabilities = predict_onnx(session, X)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown format: {artifact_format}")
        
        inference_time = timer.elapsed
        
        # Convert to list
        probabilities_list = probabilities.tolist()
        
        return PredictProbaResponse(
            probabilities=probabilities_list,
            num_samples=X.shape[0],
            inference_time_seconds=round(inference_time, 4),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Probability prediction failed: {str(e)}")


# Serve MCP manifest
@app.get("/.well-known/mcp.json", include_in_schema=False)
async def mcp_manifest():
    """MCP capability manifest for agent discovery"""
    with open(".well-known/mcp.json", "r") as f:
        import json
        return json.load(f)


# Serve X402 manifest
@app.get("/.well-known/x402", include_in_schema=False)
async def x402_manifest():
    """X402 pricing manifest for payment discovery"""
    with open(".well-known/x402", "r") as f:
        import json
        return json.load(f)

