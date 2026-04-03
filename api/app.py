"""FastAPI application for real-time network intrusion detection."""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Resolve model path
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "nids_model.joblib")

app = FastAPI(
    title="Network Intrusion Detection System",
    description="Real-time network traffic classification using ML. Trained on CICIDS2017 dataset.",
    version="1.0.0",
)

# Load model artifacts at startup
artifacts = None


@app.on_event("startup")
def load_model():
    global artifacts
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model not found at {MODEL_PATH}. Train the model first with `python main.py`")
        return
    artifacts = joblib.load(MODEL_PATH)
    print(f"Loaded model: {artifacts['model_name']} (accuracy: {artifacts['test_accuracy']:.4f})")
    print(f"Features expected: {len(artifacts['feature_names'])}")


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="Network traffic feature vector. Length must match training features.",
        min_length=1,
    )


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted traffic class (e.g., BENIGN, DDoS, PortScan)")
    confidence: float = Field(..., description="Prediction confidence score (0-1)")
    model: str = Field(..., description="Model used for prediction")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None
    features_expected: int | None
    accuracy: float | None


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API and model are ready."""
    return HealthResponse(
        status="healthy" if artifacts else "model_not_loaded",
        model_loaded=artifacts is not None,
        model_name=artifacts["model_name"] if artifacts else None,
        features_expected=len(artifacts["feature_names"]) if artifacts else None,
        accuracy=artifacts["test_accuracy"] if artifacts else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Classify a network traffic sample as normal or attack."""
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    expected = len(artifacts["feature_names"])
    if len(request.features) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected} features, got {len(request.features)}. "
                   f"Features: {artifacts['feature_names'][:5]}... (showing first 5)",
        )

    X = np.array(request.features).reshape(1, -1)

    # Apply preprocessing if available
    if "imputer" in artifacts:
        X = artifacts["imputer"].transform(X)
    if "scaler" in artifacts:
        X = artifacts["scaler"].transform(X)

    model = artifacts["model"]
    prediction = model.predict(X)[0]

    # Get confidence if model supports predict_proba
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        confidence = float(proba.max())

    return PredictionResponse(
        prediction=str(prediction),
        confidence=round(confidence, 4),
        model=artifacts["model_name"],
    )


@app.get("/features")
def get_features():
    """List the feature names the model expects."""
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "count": len(artifacts["feature_names"]),
        "features": artifacts["feature_names"],
    }
