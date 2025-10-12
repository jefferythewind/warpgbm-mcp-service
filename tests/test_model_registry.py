"""
Test model registry and multi-model support
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.model_registry import get_registry

client = TestClient(app)


def test_list_models():
    """Test listing available models"""
    response = client.get("/models")
    assert response.status_code == 200
    
    data = response.json()
    assert "models" in data
    assert "warpgbm" in data["models"]
    assert "lightgbm" in data["models"]
    assert data["default"] == "warpgbm"


def test_model_registry():
    """Test model registry functionality"""
    registry = get_registry()
    
    # Should have both models
    models = registry.list_models()
    assert "warpgbm" in models
    assert "lightgbm" in models
    
    # Should be able to get adapters
    warp_adapter = registry.get_adapter("warpgbm")
    assert warp_adapter is not None
    
    lgb_adapter = registry.get_adapter("lightgbm")
    assert lgb_adapter is not None


def test_train_warpgbm():
    """Test training with WarpGBM"""
    request_data = {
        "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        "y": [0, 1, 0, 1],
        "model_type": "warpgbm",
        "objective": "binary",
        "max_depth": 3,
        "num_trees": 10,
        "export_joblib": True,
        "export_onnx": False,
    }
    
    response = client.post("/train", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["model_type"] == "warpgbm"
    assert "model_artifact_joblib" in data


def test_train_lightgbm():
    """Test training with LightGBM"""
    request_data = {
        "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        "y": [0, 1, 0, 1],
        "model_type": "lightgbm",
        "objective": "binary",
        "max_depth": 3,
        "num_trees": 10,
        "num_leaves": 7,
        "export_joblib": True,
        "export_onnx": False,
    }
    
    response = client.post("/train", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["model_type"] == "lightgbm"
    assert "model_artifact_joblib" in data


def test_train_unknown_model():
    """Test training with unknown model type"""
    request_data = {
        "X": [[1.0, 2.0]],
        "y": [0],
        "model_type": "unknown_model",
        "objective": "binary",
    }
    
    # Should fail validation (not in Literal enum)
    response = client.post("/train", json=request_data)
    assert response.status_code == 422  # Validation error


def test_compare_warpgbm_lightgbm():
    """Test that both models can train and predict on same data"""
    train_data = {
        "X": [[i, i+1] for i in range(10)],
        "y": [0, 1] * 5,
        "objective": "binary",
        "max_depth": 3,
        "num_trees": 10,
        "export_joblib": True,
        "export_onnx": False,
    }
    
    test_data = [[5.0, 6.0], [3.0, 4.0]]
    
    models = ["warpgbm", "lightgbm"]
    results = {}
    
    for model_type in models:
        train_request = {**train_data, "model_type": model_type}
        
        # Train
        train_response = client.post("/train", json=train_request)
        assert train_response.status_code == 200
        
        train_result = train_response.json()
        assert train_result["model_type"] == model_type
        
        # Predict
        predict_request = {
            "model_artifact": train_result["model_artifact_joblib"],
            "X": test_data,
            "format": "joblib"
        }
        
        predict_response = client.post("/predict_from_artifact", json=predict_request)
        assert predict_response.status_code == 200
        
        predict_result = predict_response.json()
        results[model_type] = predict_result["predictions"]
    
    # Both models should make predictions (values may differ)
    assert len(results["warpgbm"]) == 2
    assert len(results["lightgbm"]) == 2




