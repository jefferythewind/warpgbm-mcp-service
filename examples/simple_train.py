"""
Simple example: Train a model and make predictions
"""

import requests
import json

# Configuration
API_URL = "http://localhost:4000"

print("🤖 WarpGBM MCP Service - Simple Example\n")

# Generate some simple data
X_train = [
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0],
    [5.0, 6.0],
    [6.0, 7.0],
]
y_train = [0, 0, 1, 1, 2, 2]

X_test = [
    [1.5, 2.5],
    [4.5, 5.5],
]

print("📊 Training data:")
print(f"   {len(X_train)} samples, {len(X_train[0])} features")
print(f"   Classes: {set(y_train)}")
print()

# Train model
print("🚀 Training WarpGBM model...")
train_request = {
    "X": X_train,
    "y": y_train,
    "model_type": "warpgbm",  # Try "lightgbm" too!
    "objective": "multiclass",
    "num_class": 3,
    "max_depth": 3,
    "num_trees": 20,
    "learning_rate": 0.1,
    "export_joblib": True,
    "export_onnx": False,
}

response = requests.post(f"{API_URL}/train", json=train_request)

if response.status_code != 200:
    print(f"❌ Training failed: {response.text}")
    exit(1)

train_data = response.json()
print(f"✅ Training completed in {train_data['training_time_seconds']:.3f}s")
print(f"   Model type: {train_data['model_type']}")
print()

# Get model artifact
model_artifact = train_data["model_artifact_joblib"]
print(f"💾 Model artifact size: ~{len(model_artifact) / 1024:.1f} KB")
print()

# Predict
print("🔮 Making predictions...")
predict_request = {
    "model_artifact": model_artifact,
    "X": X_test,
    "format": "joblib"
}

response = requests.post(f"{API_URL}/predict_from_artifact", json=predict_request)

if response.status_code != 200:
    print(f"❌ Prediction failed: {response.text}")
    exit(1)

predict_data = response.json()
predictions = predict_data["predictions"]

print("✅ Predictions:")
for i, (x, pred) in enumerate(zip(X_test, predictions)):
    print(f"   Sample {i+1}: {x} → Class {int(pred)}")

print()
print(f"⏱️  Inference time: {predict_data['inference_time_seconds']:.4f}s")
print()
print("🎉 Done!")




