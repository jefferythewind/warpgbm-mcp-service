"""
Main FastAPI application for WarpGBM MCP Service
"""

import os
import uuid
import time
from contextlib import asynccontextmanager

# Global references to GPU functions (injected by modal_app.py or local_dev.py)
_gpu_training_function = None
_gpu_predict_function = None
_gpu_predict_proba_function = None
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import numpy as np
import pandas as pd
import base64
import io

from app.models import (
    TrainRequest,
    TrainResponse,
    PredictRequest,
    PredictResponse,
    PredictProbaResponse,
    HealthResponse,
    DataUploadRequest,
    DataUploadResponse,
    FeedbackRequest,
    FeedbackResponse,
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
from app.feedback_storage import get_feedback_storage
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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    """Landing page with service information"""
    # Check if this is a browser request
    accept_header = request.headers.get("accept", "")
    
    # Return HTML for browsers, JSON for API clients
    if "text/html" in accept_header:
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>WarpGBM MCP Service</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #333;
                    line-height: 1.6;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 2rem;
                }
                header {
                    background: white;
                    padding: 2rem;
                    border-radius: 12px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                    margin-bottom: 2rem;
                    text-align: center;
                }
                h1 {
                    font-size: 3rem;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    margin-bottom: 1rem;
                }
                .subtitle {
                    font-size: 1.25rem;
                    color: #666;
                    margin-bottom: 2rem;
                }
                .grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 2rem;
                }
                .card {
                    background: white;
                    padding: 2rem;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    transition: transform 0.2s, box-shadow 0.2s;
                }
                .card:hover {
                    transform: translateY(-4px);
                    box-shadow: 0 8px 12px rgba(0,0,0,0.15);
                }
                .card h2 {
                    color: #667eea;
                    margin-bottom: 1rem;
                    font-size: 1.5rem;
                }
                .card p {
                    color: #666;
                    margin-bottom: 1rem;
                }
                .card ul {
                    list-style: none;
                    padding-left: 0;
                }
                .card li {
                    padding: 0.5rem 0;
                    border-bottom: 1px solid #f0f0f0;
                }
                .card li:last-child {
                    border-bottom: none;
                }
                a {
                    color: #667eea;
                    text-decoration: none;
                    font-weight: 500;
                }
                a:hover {
                    text-decoration: underline;
                }
                .button {
                    display: inline-block;
                    padding: 0.75rem 1.5rem;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 8px;
                    text-decoration: none;
                    font-weight: 600;
                    transition: opacity 0.2s;
                    margin-right: 1rem;
                }
                .button:hover {
                    opacity: 0.9;
                    text-decoration: none;
                }
                .code {
                    background: #f6f8fa;
                    padding: 1rem;
                    border-radius: 6px;
                    font-family: 'Courier New', monospace;
                    font-size: 0.9rem;
                    overflow-x: auto;
                    margin: 1rem 0;
                }
                .badge {
                    display: inline-block;
                    padding: 0.25rem 0.75rem;
                    background: #667eea;
                    color: white;
                    border-radius: 12px;
                    font-size: 0.85rem;
                    font-weight: 600;
                    margin-right: 0.5rem;
                }
                .badge.gpu {
                    background: #48bb78;
                }
                .badge.cpu {
                    background: #4299e1;
                }
                footer {
                    text-align: center;
                    color: white;
                    padding: 2rem;
                    margin-top: 2rem;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>🚀 WarpGBM MCP</h1>
                    <p class="subtitle">GPU-Accelerated Gradient Boosting as a Universal MCP Service</p>
                    <p style="margin: 1rem 0; padding: 1rem; background: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107;">
                        <strong>💡 Looking for the Python package?</strong><br>
                        This is a cloud API service. For production ML workflows, use the 
                        <a href="https://github.com/jefferythewind/warpgbm" target="_blank" style="color: #667eea; font-weight: 600;">WarpGBM Python package</a> 
                        directly (91+ ⭐, free GPU on your hardware)
                    </p>
                    <div>
                        <a href="/docs" class="button">📚 API Docs</a>
                        <a href="/guide" class="button">📖 Usage Guide</a>
                        <a href="/.well-known/mcp.json" class="button">🔧 MCP Manifest</a>
                    </div>
                </header>

                <div class="grid">
                    <div class="card">
                        <h2>🤖 For AI Agents</h2>
                        <p>Connect via MCP protocol for seamless ML integration:</p>
                        <ul>
                            <li><strong>MCP SSE Endpoint:</strong> <code>/mcp/sse</code></li>
                            <li><strong>Manifest:</strong> <a href="/.well-known/mcp.json">/.well-known/mcp.json</a></li>
                            <li><strong>Pricing:</strong> <a href="/.well-known/x402">/.well-known/x402</a></li>
                        </ul>
                        <p style="margin-top: 1rem;">Supports direct tool calls for training, prediction, and feedback.</p>
                    </div>

                    <div class="card">
                        <h2>👨‍💻 For Developers</h2>
                        <p>Use our REST API for programmatic access:</p>
                        <ul>
                            <li><strong>Health Check:</strong> <a href="/healthz">/healthz</a></li>
                            <li><strong>List Models:</strong> <a href="/models">/models</a></li>
                            <li><strong>Train:</strong> POST <code>/train</code></li>
                            <li><strong>Predict:</strong> POST <code>/predict_from_artifact</code></li>
                            <li><strong>Upload Data:</strong> POST <code>/upload_data</code></li>
                        </ul>
                    </div>

                    <div class="card">
                        <h2>⚡ Available Models</h2>
                        <p>Choose from multiple gradient boosting backends:</p>
                        <div style="margin: 1rem 0;">
                            <span class="badge gpu">GPU</span>
                            <strong>WarpGBM</strong> - GPU-accelerated GBDT with era-aware splitting
                        </div>
                        <div>
                            <span class="badge cpu">CPU</span>
                            <strong>LightGBM</strong> - Microsoft's fast, distributed gradient boosting
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h2>🎯 Quick Start Example</h2>
                    <p>Train a model using curl:</p>
                    <div class="code">
curl -X POST https://warpgbm.ai/train \\<br>
  -H "Content-Type: application/json" \\<br>
  -d '{<br>
    "X": [[1,2,3], [4,5,6], [7,8,9]],<br>
    "y": [0, 1, 2],<br>
    "model_type": "lightgbm",<br>
    "objective": "multiclass",<br>
    "num_class": 3<br>
  }'
                    </div>
                </div>

                <div class="card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                    <h2 style="color: white;">🐍 WarpGBM Python Package</h2>
                    <p>For production ML workflows, install the full Python package:</p>
                    <div class="code" style="background: rgba(0,0,0,0.2); color: #fff;">
pip install git+https://github.com/jefferythewind/warpgbm.git
                    </div>
                    <p style="margin-top: 1rem;"><strong>Why use Python directly?</strong></p>
                    <ul style="list-style: disc; padding-left: 1.5rem; margin-top: 0.5rem;">
                        <li>✅ Free - Use your own GPU</li>
                        <li>✅ Full control - Custom losses, callbacks, cross-validation</li>
                        <li>✅ Richer API - Feature importance, era analysis, SHAP</li>
                        <li>✅ Faster - No serialization overhead</li>
                        <li>✅ 91+ GitHub stars, GPL-3.0 license</li>
                    </ul>
                    <div style="margin-top: 1rem;">
                        <a href="https://github.com/jefferythewind/warpgbm" target="_blank" style="color: white; text-decoration: underline;">📦 View on GitHub</a>
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        <a href="https://github.com/jefferythewind/warpgbm/blob/main/AGENT_GUIDE.md" target="_blank" style="color: white; text-decoration: underline;">📘 Python Agent Guide</a>
                    </div>
                </div>

                <div class="card">
                    <h2>💡 MCP Service Features</h2>
                    <p><em>Use this cloud API for quick experiments and prototyping:</em></p>
                    <div class="grid" style="margin-top: 1rem;">
                        <div>
                            <strong>🔒 Stateless Design</strong>
                            <p style="font-size: 0.9rem; color: #666;">No model storage - get portable artifacts</p>
                        </div>
                        <div>
                            <strong>⚡ GPU-Accelerated</strong>
                            <p style="font-size: 0.9rem; color: #666;">WarpGBM trains on NVIDIA A10G GPUs</p>
                        </div>
                        <div>
                            <strong>💰 Pay-per-use</strong>
                            <p style="font-size: 0.9rem; color: #666;">X402 micropayments on Base network</p>
                        </div>
                        <div>
                            <strong>📊 Data Formats</strong>
                            <p style="font-size: 0.9rem; color: #666;">CSV, Parquet, JSON arrays</p>
                        </div>
                    </div>
                </div>

                <footer>
                    <p>WarpGBM MCP Service | <a href="https://github.com/jefferythewind/warpgbm" style="color: white;">GitHub</a></p>
                    <p style="opacity: 0.8; margin-top: 0.5rem;">Built with FastAPI, Modal, and MCP protocol</p>
                </footer>
            </div>
        </body>
        </html>
        """
    else:
        # Return JSON for API clients
        registry = get_registry()
        return JSONResponse({
            "service": "WarpGBM MCP",
            "version": __version__,
            "status": "online",
            "endpoints": {
                "docs": "/docs",
                "guide": "/guide",
                "health": "/healthz",
                "models": "/models",
                "mcp_manifest": "/.well-known/mcp.json",
                "x402_manifest": "/.well-known/x402",
            },
            "available_models": registry.list_models(),
        })


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
        global _gpu_predict_proba_function
        if _gpu_predict_proba_function:
            try:
                start = time.time()
                if hasattr(_gpu_predict_proba_function, 'remote'):
                    probs = _gpu_predict_proba_function.remote(model_artifact, predict_request.X)
                else:
                    probs = _gpu_predict_proba_function(model_artifact, predict_request.X)
                return PredictProbaResponse(
                    probabilities=probs,
                    num_samples=len(probs),
                    inference_time_seconds=round(time.time() - start, 4)
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"GPU probability prediction failed: {str(e)}")
        else:
            raise HTTPException(status_code=503, detail="WarpGBM requires GPU (Modal deployment)")
    
    # CPU predictions for LightGBM and others
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


@app.get("/robots.txt", response_class=PlainTextResponse, include_in_schema=False)
async def robots_txt():
    """Robots.txt for search engine crawlers"""
    return """User-agent: *
Disallow: /mcp/sse
Disallow: /train
Disallow: /predict_from_artifact
Disallow: /predict_proba_from_artifact
Disallow: /upload_data
Allow: /
Allow: /healthz
Allow: /models
Allow: /guide
Allow: /docs
Allow: /.well-known/

Sitemap: /.well-known/mcp.json
"""


@app.get("/guide", response_class=HTMLResponse, tags=["documentation"])
async def guide():
    """Interactive usage guide"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>WarpGBM MCP - Usage Guide</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                background: #f5f7fa;
                color: #333;
                line-height: 1.6;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                padding: 2rem;
            }
            header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 3rem 2rem;
                border-radius: 12px;
                margin-bottom: 2rem;
                text-align: center;
            }
            h1 { font-size: 2.5rem; margin-bottom: 1rem; }
            h2 {
                color: #667eea;
                margin-top: 2rem;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #667eea;
            }
            h3 { color: #555; margin-top: 1.5rem; margin-bottom: 0.5rem; }
            .card {
                background: white;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin-bottom: 2rem;
            }
            code {
                background: #f6f8fa;
                padding: 0.2rem 0.4rem;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }
            pre {
                background: #282c34;
                color: #abb2bf;
                padding: 1.5rem;
                border-radius: 8px;
                overflow-x: auto;
                margin: 1rem 0;
            }
            pre code {
                background: none;
                color: inherit;
                padding: 0;
            }
            .endpoint {
                display: inline-block;
                background: #667eea;
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 4px;
                font-weight: 600;
                margin-right: 0.5rem;
            }
            .method {
                display: inline-block;
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
                font-weight: 600;
                font-size: 0.85rem;
            }
            .method.get { background: #48bb78; color: white; }
            .method.post { background: #4299e1; color: white; }
            ul { padding-left: 1.5rem; margin: 1rem 0; }
            li { margin: 0.5rem 0; }
            a { color: #667eea; text-decoration: none; font-weight: 500; }
            a:hover { text-decoration: underline; }
            .back-link {
                display: inline-block;
                margin-bottom: 1rem;
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <a href="/" class="back-link">← Back to Home</a>
                <h1>📖 WarpGBM MCP Usage Guide</h1>
                <p>Complete guide for developers and AI agents</p>
                <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid #ffc107;">
                    <strong>💡 Want the Python package?</strong>
                    For production ML workflows, use 
                    <a href="https://github.com/jefferythewind/warpgbm" target="_blank" style="color: #667eea;">WarpGBM directly</a> 
                    (91+ ⭐) | 
                    <a href="https://github.com/jefferythewind/warpgbm/blob/main/AGENT_GUIDE.md" target="_blank" style="color: #667eea;">Python Agent Guide</a>
                </div>
            </header>

            <div class="card">
                <h2>🚀 Quick Start</h2>
                <p>Train a model in 3 steps:</p>
                
                <h3>1. Check Service Health</h3>
                <pre><code>curl https://warpgbm.ai/healthz</code></pre>
                
                <h3>2. Train a Model</h3>
                <pre><code>curl -X POST https://warpgbm.ai/train \\
  -H "Content-Type: application/json" \\
  -d '{
    "X": [[1,2,3], [4,5,6], [7,8,9]],
    "y": [0, 1, 2],
    "model_type": "lightgbm",
    "objective": "multiclass",
    "num_class": 3,
    "num_trees": 100
  }'</code></pre>
                
                <h3>3. Make Predictions</h3>
                <pre><code>curl -X POST https://warpgbm.ai/predict_from_artifact \\
  -H "Content-Type: application/json" \\
  -d '{
    "artifact_id": "&lt;artifact_id_from_training&gt;",
    "X": [[2,3,4], [5,6,7]]
  }'</code></pre>
            </div>

            <div class="card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                <h2 style="color: white;">🐍 MCP Service vs Python Package</h2>
                <table style="width: 100%; border-collapse: collapse; margin-top: 1rem;">
                    <tr style="background: rgba(255,255,255,0.1);">
                        <th style="padding: 0.5rem; text-align: left;">Feature</th>
                        <th style="padding: 0.5rem; text-align: left;">MCP Service</th>
                        <th style="padding: 0.5rem; text-align: left;">Python Package</th>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem;">Install</td>
                        <td style="padding: 0.5rem;">None needed</td>
                        <td style="padding: 0.5rem;"><code style="background: rgba(0,0,0,0.3); padding: 0.2rem;">pip install git+...</code></td>
                    </tr>
                    <tr style="background: rgba(255,255,255,0.05);">
                        <td style="padding: 0.5rem;">GPU</td>
                        <td style="padding: 0.5rem;">Cloud (pay-per-use)</td>
                        <td style="padding: 0.5rem;">Your GPU (free)</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem;">Features</td>
                        <td style="padding: 0.5rem;">Train, predict, upload</td>
                        <td style="padding: 0.5rem;">Full API + cross-val + feature importance</td>
                    </tr>
                    <tr style="background: rgba(255,255,255,0.05);">
                        <td style="padding: 0.5rem;">Best For</td>
                        <td style="padding: 0.5rem;">Quick experiments</td>
                        <td style="padding: 0.5rem;">Production pipelines</td>
                    </tr>
                </table>
                <p style="margin-top: 1rem;">
                    <a href="https://github.com/jefferythewind/warpgbm" target="_blank" style="color: white; text-decoration: underline;">View WarpGBM on GitHub →</a>
                </p>
            </div>

            <div class="card">
                <h2>🔧 Available Endpoints</h2>
                
                <h3><span class="method get">GET</span> /healthz</h3>
                <p>Check service health and GPU availability</p>
                
                <h3><span class="method get">GET</span> /models</h3>
                <p>List all available model backends (warpgbm, lightgbm)</p>
                
                <h3><span class="method post">POST</span> /train</h3>
                <p>Train a gradient boosting model. Returns a portable model artifact.</p>
                <p><strong>Required:</strong> <code>X</code>, <code>y</code></p>
                <p><strong>Optional:</strong> <code>model_type</code>, <code>objective</code>, <code>num_trees</code>, <code>learning_rate</code>, etc.</p>
                
                <h3><span class="method post">POST</span> /predict_from_artifact</h3>
                <p>Make predictions using a trained model artifact.</p>
                <p><strong>Required:</strong> <code>X</code>, and either <code>artifact_id</code> or <code>model_artifact</code></p>
                
                <h3><span class="method post">POST</span> /upload_data</h3>
                <p>Upload CSV or Parquet files for training. Returns structured X and y arrays.</p>
                <p><strong>Required:</strong> <code>file_content</code> (base64), <code>file_format</code>, <code>target_column</code></p>
                
                <h3><span class="method post">POST</span> /feedback</h3>
                <p>Submit feedback about the service (bugs, feature requests, etc.)</p>
                <p><strong>Required:</strong> <code>feedback_type</code>, <code>message</code></p>
            </div>

            <div class="card">
                <h2>🤖 Model Types</h2>
                
                <h3>WarpGBM (GPU)</h3>
                <ul>
                    <li>GPU-accelerated gradient boosting</li>
                    <li>Ideal for: Time-series data with era structure</li>
                    <li>Trains on NVIDIA A10G GPUs</li>
                    <li>Parameters: <code>num_trees</code>, <code>max_depth</code>, <code>learning_rate</code>, <code>num_bins</code></li>
                </ul>
                
                <h3>LightGBM (CPU)</h3>
                <ul>
                    <li>Microsoft's fast gradient boosting framework</li>
                    <li>Ideal for: General-purpose ML, large datasets</li>
                    <li>Highly optimized for CPU</li>
                    <li>Parameters: <code>num_trees</code>, <code>max_depth</code>, <code>learning_rate</code>, <code>num_leaves</code></li>
                </ul>
            </div>

            <div class="card">
                <h2>📊 Working with Large Datasets</h2>
                <p>For datasets too large to send as JSON arrays, use the data upload endpoint:</p>
                
                <h3>Upload CSV Example</h3>
                <pre><code>import base64
import requests

# Read your CSV file
with open('data.csv', 'rb') as f:
    file_content = base64.b64encode(f.read()).decode()

response = requests.post(
    'https://warpgbm.ai/upload_data',
    json={
        'file_content': file_content,
        'file_format': 'csv',
        'target_column': 'target',
        'feature_columns': ['feat1', 'feat2', 'feat3']  # optional
    }
)

# Response includes X, y arrays ready for training
data = response.json()
print(f"Loaded {data['num_samples']} samples, {data['num_features']} features")</code></pre>
            </div>

            <div class="card">
                <h2>💰 X402 Payment Protocol</h2>
                <p>This service supports X402 micropayments on Base network:</p>
                <ul>
                    <li><strong>Training:</strong> 0.01 USDC per request</li>
                    <li><strong>Inference:</strong> 0.001 USDC per request</li>
                    <li><strong>Data Upload:</strong> 0.005 USDC per request</li>
                </ul>
                <p>Payment is optional for demo/testing. See <a href="/.well-known/x402">/.well-known/x402</a> for details.</p>
            </div>

            <div class="card">
                <h2>🔗 MCP Integration</h2>
                <p>For AI agents using the MCP protocol:</p>
                <ul>
                    <li><strong>SSE Endpoint:</strong> <code>/mcp/sse</code></li>
                    <li><strong>Manifest:</strong> <a href="/.well-known/mcp.json">/.well-known/mcp.json</a></li>
                    <li><strong>Tools:</strong> <code>train</code>, <code>predict_from_artifact</code>, <code>list_models</code>, <code>get_agent_guide</code></li>
                </ul>
            </div>

            <div class="card">
                <h2>❓ Need Help?</h2>
                <p>Resources:</p>
                <ul>
                    <li><a href="/docs">OpenAPI Documentation</a></li>
                    <li><a href="https://github.com/jefferythewind/warpgbm">GitHub Repository</a></li>
                    <li>Submit feedback via <code>POST /feedback</code></li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """


@app.post("/upload_data", response_model=DataUploadResponse, tags=["data"])
@limiter.limit("20/minute")
async def upload_data(
    request: Request,
    upload_request: DataUploadRequest,
    paid: bool = Depends(verify_payment_optional),
):
    """
    Upload CSV or Parquet files for training.
    
    Returns structured X (features) and y (target) arrays ready for /train endpoint.
    """
    try:
        # Decode base64 content
        file_bytes = base64.b64decode(upload_request.file_content)
        
        # Parse based on format
        if upload_request.file_format == "csv":
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif upload_request.file_format == "parquet":
            df = pd.read_parquet(io.BytesIO(file_bytes))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {upload_request.file_format}")
        
        # Validate target column
        if upload_request.target_column and upload_request.target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{upload_request.target_column}' not found in data. "
                       f"Available columns: {list(df.columns)}"
            )
        
        # Extract features and target
        if upload_request.target_column:
            y = df[upload_request.target_column].values
            if upload_request.feature_columns:
                # Use specified feature columns
                missing = set(upload_request.feature_columns) - set(df.columns)
                if missing:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Feature columns not found: {list(missing)}"
                    )
                X = df[upload_request.feature_columns].values
                feature_names = upload_request.feature_columns
            else:
                # Use all columns except target
                feature_cols = [col for col in df.columns if col != upload_request.target_column]
                X = df[feature_cols].values
                feature_names = feature_cols
        else:
            # No target specified - return all data as features
            X = df.values
            y = None
            feature_names = list(df.columns)
        
        # Convert to lists for JSON serialization
        X_list = X.tolist()
        y_list = y.tolist() if y is not None else []
        
        # Create preview
        preview = df.head(5).to_dict(orient='records')
        
        return DataUploadResponse(
            num_samples=len(df),
            num_features=X.shape[1],
            feature_names=feature_names,
            target_name=upload_request.target_column or "not_specified",
            preview=preview,
        )
        
    except HTTPException:
        # Re-raise HTTPException as-is (don't wrap in 500)
        raise
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse {upload_request.file_format}: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data upload failed: {str(e)}")


@app.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
@limiter.limit("10/minute")
async def submit_feedback(
    request: Request,
    feedback_request: FeedbackRequest,
):
    """
    Submit feedback about the service.
    
    Agents and users can report bugs, request features, or provide general feedback.
    Feedback is logged for service improvement.
    """
    feedback_id = str(uuid.uuid4())
    timestamp = time.time()
    
    # Prepare feedback data
    feedback_log = {
        "id": feedback_id,
        "type": feedback_request.feedback_type,
        "message": feedback_request.message,
        "endpoint": feedback_request.endpoint,
        "model_type": feedback_request.model_type,
        "severity": feedback_request.severity,
        "agent_info": str(feedback_request.agent_info) if feedback_request.agent_info else None,
        "timestamp": timestamp,
    }
    
    # Save to parquet file using storage adapter
    storage = get_feedback_storage()
    success = storage.save_feedback(feedback_log)
    
    if success:
        print(f"📣 FEEDBACK SAVED ({storage.environment}): {feedback_log['type']} - {feedback_log['message'][:50]}... (ID: {feedback_id})")
    
    return FeedbackResponse(
        feedback_id=feedback_id,
        status="received",
        message=f"Thank you for your feedback! ID: {feedback_id}",
        timestamp=timestamp,
    )


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

