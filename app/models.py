"""
Pydantic models for request/response validation
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class TrainRequest(BaseModel):
    """Request schema for training endpoint"""
    
    X: List[List[float]] = Field(..., description="Feature matrix (2D array)")
    y: List[float] = Field(..., description="Target labels (integers for classification, floats for regression)")
    
    # Model selection
    model_type: Literal["warpgbm", "lightgbm"] = Field(
        default="warpgbm",
        description="Model backend to use"
    )
    
    # Common hyperparameters
    objective: Literal["regression", "binary", "multiclass"] = Field(
        default="multiclass",
        description="Training objective"
    )
    num_class: Optional[int] = Field(
        default=None,
        description="Number of classes for multiclass classification"
    )
    
    # Model-specific parameters (stored as extra fields)
    # WarpGBM parameters:
    max_depth: Optional[int] = Field(default=None, ge=-1, le=20, description="Maximum tree depth")
    num_trees: Optional[int] = Field(default=None, ge=1, le=5000, description="Number of boosting rounds")
    learning_rate: Optional[float] = Field(default=None, gt=0, le=1.0, description="Learning rate")
    
    # WarpGBM specific:
    num_bins: Optional[int] = Field(default=None, ge=2, le=256, description="Number of bins (WarpGBM)")
    min_child_weight: Optional[float] = Field(default=None, ge=0, description="Min child weight (WarpGBM)")
    min_split_gain: Optional[float] = Field(default=None, ge=0, description="Min split gain (WarpGBM)")
    colsample_bytree: Optional[float] = Field(default=None, gt=0, le=1.0, description="Feature sampling ratio")
    
    # LightGBM specific:
    num_leaves: Optional[int] = Field(default=None, ge=2, description="Number of leaves (LightGBM)")
    min_data_in_leaf: Optional[int] = Field(default=None, ge=1, description="Min data in leaf (LightGBM)")
    feature_fraction: Optional[float] = Field(default=None, gt=0, le=1.0, description="Feature fraction (LightGBM)")
    bagging_fraction: Optional[float] = Field(default=None, gt=0, le=1.0, description="Bagging fraction (LightGBM)")
    bagging_freq: Optional[int] = Field(default=None, ge=0, description="Bagging frequency (LightGBM)")
    lambda_l1: Optional[float] = Field(default=None, ge=0, description="L1 regularization (LightGBM)")
    lambda_l2: Optional[float] = Field(default=None, ge=0, description="L2 regularization (LightGBM)")
    
    # Export options
    export_onnx: bool = Field(default=True, description="Whether to export ONNX model")
    export_joblib: bool = Field(default=True, description="Whether to export joblib model")
    
    @field_validator("num_class")
    @classmethod
    def validate_num_class(cls, v, info):
        if info.data.get("objective") == "multiclass" and v is None:
            raise ValueError("num_class is required for multiclass objective")
        return v
    
    class Config:
        extra = "allow"  # Allow additional model-specific parameters
        json_schema_extra = {
            "example": {
                "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                "y": [0, 1, 2],
                "model_type": "warpgbm",
                "objective": "multiclass",
                "num_class": 3,
                "max_depth": 6,
                "num_trees": 100,
                "learning_rate": 0.1
            }
        }


class TrainResponse(BaseModel):
    """Response schema for training endpoint"""
    
    model_type: str = Field(..., description="Model backend used")
    artifact_id: str = Field(..., description="Temporary artifact ID for fast predictions (valid for 5 minutes)")
    model_artifact_joblib: Optional[str] = Field(
        default=None,
        description="Base64-encoded, gzip-compressed joblib serialized model"
    )
    num_samples: int = Field(..., description="Number of training samples")
    num_features: int = Field(..., description="Number of features")
    training_time_seconds: float = Field(..., description="Training duration")
    
    class Config:
        # Exclude None values from JSON response
        json_encoders = {type(None): lambda v: None}
        exclude_none = True


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint"""
    
    artifact_id: Optional[str] = Field(
        default=None,
        description="Temporary artifact ID from training (fast, valid for 5 minutes)"
    )
    model_artifact: Optional[str] = Field(
        default=None,
        description="Base64-encoded, gzip-compressed model artifact"
    )
    X: List[List[float]] = Field(..., description="Feature matrix for inference")
    format: Literal["joblib", "onnx"] = Field(
        default="joblib",
        description="Model format"
    )
    
    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_source(cls, v, info):
        """Ensure either artifact_id or model_artifact is provided"""
        artifact_id = v
        model_artifact = info.data.get("model_artifact")
        
        if not artifact_id and not model_artifact:
            raise ValueError("Either artifact_id or model_artifact must be provided")
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_artifact": "<base64-encoded-model>",
                "X": [[1.0, 2.0], [3.0, 4.0]],
                "format": "joblib"
            }
        }


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint"""
    
    predictions: List[float] = Field(..., description="Model predictions")
    num_samples: int = Field(..., description="Number of samples predicted")
    inference_time_seconds: float = Field(..., description="Inference duration")


class PredictProbaResponse(BaseModel):
    """Response schema for probability prediction endpoint"""
    
    probabilities: List[List[float]] = Field(..., description="Class probabilities")
    num_samples: int = Field(..., description="Number of samples predicted")
    inference_time_seconds: float = Field(..., description="Inference duration")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint"""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_name: Optional[str] = Field(default=None, description="GPU device name")


class ErrorResponse(BaseModel):
    """Structured error response for better agent understanding"""
    error: str = Field(..., description="Human-readable error message")
    error_type: Literal["validation", "training", "inference", "server"] = Field(
        ..., 
        description="Category of error"
    )
    suggestion: Optional[str] = Field(
        default=None, 
        description="Suggested fix or next steps"
    )


class X402VerifyRequest(BaseModel):
    """Request schema for X402 payment verification"""
    
    tx_hash: str = Field(..., description="Transaction hash or payment proof")


class X402VerifyResponse(BaseModel):
    """Response schema for X402 verification"""
    
    status: Literal["paid", "unpaid", "pending"] = Field(..., description="Payment status")
    token: Optional[str] = Field(default=None, description="Access token if paid")
    expires_in: Optional[int] = Field(default=None, description="Token expiry in seconds")
    message: Optional[str] = Field(default=None, description="Additional information about verification status")

