"""
Test training endpoint
"""

import pytest


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "gpu_available" in data


def test_train_multiclass(client):
    """Test multiclass training"""
    request_data = {
        "X": [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ],
        "y": [0, 1, 2, 0, 1],
        "model_type": "warpgbm",
        "objective": "multiclass",
        "num_class": 3,
        "max_depth": 3,
        "num_trees": 10,
        "learning_rate": 0.1,
        "export_joblib": True,
        "export_onnx": False,  # Skip ONNX for now
    }
    
    response = client.post("/train", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["model_type"] == "warpgbm"
    assert "model_artifact_joblib" in data
    assert data["model_artifact_joblib"] is not None
    assert data["num_samples"] == 5
    assert data["num_features"] == 2
    assert data["training_time_seconds"] >= 0  # Can be 0 for very fast training


def test_train_binary(client):
    """Test binary classification training"""
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


def test_train_regression(client):
    """Test regression training"""
    request_data = {
        "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "y": [10, 20, 30],
        "model_type": "warpgbm",
        "objective": "regression",
        "max_depth": 3,
        "num_trees": 10,
        "export_joblib": True,
        "export_onnx": False,
    }
    
    response = client.post("/train", json=request_data)
    assert response.status_code == 200


def test_train_invalid_shape(client):
    """Test training with mismatched X and y shapes"""
    request_data = {
        "X": [[1.0, 2.0], [3.0, 4.0]],
        "y": [0, 1, 2],  # Wrong length
        "model_type": "warpgbm",
        "objective": "multiclass",
        "num_class": 3,
    }
    
    response = client.post("/train", json=request_data)
    assert response.status_code == 400


def test_train_missing_num_class(client):
    """Test multiclass without num_class - should auto-infer but fail on insufficient samples"""
    request_data = {
        "X": [[1.0, 2.0]],
        "y": [0],
        "model_type": "warpgbm",
        "objective": "multiclass",
        # Missing num_class - will be auto-inferred
    }
    
    response = client.post("/train", json=request_data)
    # Gets 400 for insufficient classes (only 1 class), not 422 for missing num_class
    assert response.status_code == 400


def test_train_lightgbm_specific_params(client):
    """Test LightGBM with its specific parameters"""
    request_data = {
        "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        "y": [0, 1, 0, 1],
        "model_type": "lightgbm",
        "objective": "binary",
        "num_trees": 20,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "export_joblib": True,
        "export_onnx": False,
    }
    
    response = client.post("/train", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["model_type"] == "lightgbm"

