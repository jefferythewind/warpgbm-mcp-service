"""
Integration tests for end-to-end workflows
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_complete_multiclass_workflow():
    """Test complete workflow: train -> predict -> predict_proba"""
    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.rand(100, 5).tolist()
    y_train = np.random.randint(0, 3, 100).tolist()
    X_test = np.random.rand(10, 5).tolist()
    
    # Train
    train_request = {
        "X": X_train,
        "y": y_train,
        "objective": "multiclass",
        "num_class": 3,
        "max_depth": 4,
        "num_trees": 20,
        "learning_rate": 0.1,
        "export_joblib": True,
        "export_onnx": False,
    }
    
    train_response = client.post("/train", json=train_request)
    assert train_response.status_code == 200
    
    train_data = train_response.json()
    model_artifact = train_data["model_artifact_joblib"]
    assert model_artifact is not None
    
    # Predict
    predict_request = {
        "model_artifact": model_artifact,
        "X": X_test,
        "format": "joblib",
    }
    
    predict_response = client.post("/predict_from_artifact", json=predict_request)
    assert predict_response.status_code == 200
    
    predict_data = predict_response.json()
    predictions = predict_data["predictions"]
    assert len(predictions) == 10
    assert all(0 <= p <= 2 for p in predictions)
    
    # Predict probabilities
    proba_response = client.post("/predict_proba_from_artifact", json=predict_request)
    assert proba_response.status_code == 200
    
    proba_data = proba_response.json()
    probabilities = proba_data["probabilities"]
    assert len(probabilities) == 10
    assert all(len(p) == 3 for p in probabilities)
    
    # Check probabilities sum to ~1
    for probs in probabilities:
        assert 0.99 <= sum(probs) <= 1.01


def test_mcp_manifest():
    """Test MCP manifest is accessible"""
    response = client.get("/.well-known/mcp.json")
    assert response.status_code == 200
    
    data = response.json()
    assert data["name"] == "warpgbm"
    assert "capabilities" in data
    assert "train" in data["capabilities"]
    assert "predict_from_artifact" in data["capabilities"]


def test_x402_manifest():
    """Test X402 manifest is accessible"""
    response = client.get("/.well-known/x402")
    assert response.status_code == 200
    
    data = response.json()
    assert data["name"] == "WarpGBM MCP Service"
    assert "pricing" in data
    assert "train" in data["pricing"]




