"""
Test prediction endpoints
"""

import pytest


def test_predict_from_artifact(client):
    """Test full train -> predict workflow using artifact_id"""
    # First, train a model
    train_request = {
        "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        "y": [0, 1, 0, 1],
        "objective": "binary",
        "max_depth": 3,
        "num_trees": 10,
        "export_joblib": True,
        "export_onnx": False,
    }
    
    train_response = client.post("/train", json=train_request)
    assert train_response.status_code == 200
    
    train_data = train_response.json()
    artifact_id = train_data["artifact_id"]  # Use artifact_id for caching
    
    # Now predict using artifact_id (preserves model_type metadata)
    predict_request = {
        "artifact_id": artifact_id,  # Use cached artifact with metadata
        "X": [[2.0, 3.0], [6.0, 7.0]],
    }
    
    predict_response = client.post("/predict_from_artifact", json=predict_request)
    assert predict_response.status_code == 200
    
    predict_data = predict_response.json()
    assert "predictions" in predict_data
    assert len(predict_data["predictions"]) == 2
    assert predict_data["num_samples"] == 2


def test_predict_proba_from_artifact(client):
    """Test probability prediction using artifact_id (WarpGBM GPU)"""
    # Train a model
    train_request = {
        "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        "y": [0, 1, 0, 1],
        "objective": "binary",
        "max_depth": 3,
        "num_trees": 10,
        "export_joblib": True,
        "export_onnx": False,
    }
    
    train_response = client.post("/train", json=train_request)
    assert train_response.status_code == 200
    artifact_id = train_response.json()["artifact_id"]
    
    # Predict probabilities using artifact_id
    predict_request = {
        "artifact_id": artifact_id,  # Use cached artifact with metadata
        "X": [[2.0, 3.0]],
    }
    
    predict_response = client.post("/predict_proba_from_artifact", json=predict_request)
    assert predict_response.status_code == 200
    
    predict_data = predict_response.json()
    assert "probabilities" in predict_data
    assert len(predict_data["probabilities"]) == 1
    assert len(predict_data["probabilities"][0]) == 2  # Binary classification


def test_predict_invalid_artifact(client):
    """Test prediction with invalid artifact"""
    predict_request = {
        "model_artifact": "invalid_base64",
        "X": [[1.0, 2.0]],
        "format": "joblib",
    }
    
    predict_response = client.post("/predict_from_artifact", json=predict_request)
    assert predict_response.status_code == 500




