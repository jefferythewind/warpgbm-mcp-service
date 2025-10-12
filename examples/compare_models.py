"""
Example: Compare WarpGBM and LightGBM on the same dataset
"""

import requests
import numpy as np
import json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Configuration
API_URL = "http://localhost:4000"

# Generate synthetic dataset
print("🎲 Generating synthetic dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test:  {X_test.shape[0]} samples")
print()

# Models to compare
models_config = [
    {
        "name": "WarpGBM",
        "config": {
            "model_type": "warpgbm",
            "objective": "multiclass",
            "num_class": 3,
            "max_depth": 6,
            "num_trees": 100,
            "learning_rate": 0.1,
            "export_joblib": True,
            "export_onnx": False,
        }
    },
    {
        "name": "LightGBM",
        "config": {
            "model_type": "lightgbm",
            "objective": "multiclass",
            "num_class": 3,
            "max_depth": 6,
            "num_trees": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "export_joblib": True,
            "export_onnx": False,
        }
    }
]

results = []

for model_spec in models_config:
    model_name = model_spec["name"]
    print(f"🚀 Training {model_name}...")
    
    # Train
    train_request = {
        "X": X_train.tolist(),
        "y": y_train.tolist(),
        **model_spec["config"]
    }
    
    train_response = requests.post(
        f"{API_URL}/train",
        json=train_request
    )
    
    if train_response.status_code != 200:
        print(f"   ❌ Training failed: {train_response.text}")
        continue
    
    train_data = train_response.json()
    print(f"   ✅ Training completed in {train_data['training_time_seconds']:.3f}s")
    
    # Get artifact
    model_artifact = train_data["model_artifact_joblib"]
    
    # Predict
    predict_request = {
        "model_artifact": model_artifact,
        "X": X_test.tolist(),
        "format": "joblib"
    }
    
    predict_response = requests.post(
        f"{API_URL}/predict_from_artifact",
        json=predict_request
    )
    
    if predict_response.status_code != 200:
        print(f"   ❌ Prediction failed: {predict_response.text}")
        continue
    
    predict_data = predict_response.json()
    predictions = np.array(predict_data["predictions"])
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    
    print(f"   📊 Test Accuracy: {accuracy:.3f}")
    print(f"   ⏱️  Inference time: {predict_data['inference_time_seconds']:.4f}s")
    print()
    
    results.append({
        "model": model_name,
        "train_time": train_data["training_time_seconds"],
        "inference_time": predict_data["inference_time_seconds"],
        "accuracy": float(accuracy)
    })

# Summary
print("=" * 60)
print("📊 COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Model':<15} {'Train (s)':<12} {'Inference (s)':<15} {'Accuracy':<10}")
print("-" * 60)

for r in results:
    print(f"{r['model']:<15} {r['train_time']:<12.3f} {r['inference_time']:<15.4f} {r['accuracy']:<10.3f}")

print("=" * 60)

# Find best model
if results:
    best_by_accuracy = max(results, key=lambda x: x["accuracy"])
    best_by_speed = min(results, key=lambda x: x["train_time"])
    
    print()
    print(f"🏆 Best Accuracy:      {best_by_accuracy['model']} ({best_by_accuracy['accuracy']:.3f})")
    print(f"⚡ Fastest Training:   {best_by_speed['model']} ({best_by_speed['train_time']:.3f}s)")




