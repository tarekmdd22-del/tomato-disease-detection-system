"""
FastAPI inference service for the Tomato Disease Detection System.

Run:
    uvicorn src.api:app --reload --port 8000

Authentication:
    All /predict endpoints require an API key header: X-API-Key
    Default demo key below - override with the MHAS_API_KEY env var (or set
    TOMATO_API_KEY here) before real deployment.

Endpoints:
    GET  /              -> service info (no auth)
    GET  /health          -> service status (no auth)
    POST /predict          -> upload one leaf image, get prediction (requires API key)
"""
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Security, UploadFile, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from preprocessing import preprocess_single_image, IMG_SIZE

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
API_KEY = os.environ.get("TOMATO_API_KEY", "demo-key-2026")
API_VERSION = "1.0.0"

HEALTHY_LABELS = {"healthy", "healthy tomato leaf", "healthy_tomato_leaf"}

app = FastAPI(
    title="Tomato Disease Detection API",
    description=(
        "Detects tomato leaf diseases from an uploaded image. All /predict "
        "endpoints require an `X-API-Key` header."
    ),
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: Optional[str] = Security(api_key_header)):
    if not key or key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key. Send it in the 'X-API-Key' header.")
    return key


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "status_code": exc.status_code, "detail": exc.detail,
                 "timestamp": datetime.now(timezone.utc).isoformat()},
    )


class ModelState:
    model = None
    class_names = []
    loaded_at = None


state = ModelState()


@app.on_event("startup")
def load_model():
    model_path = os.path.join(MODELS_DIR, "tomato_disease_model.keras")
    classes_path = os.path.join(MODELS_DIR, "class_names.json")
    if os.path.exists(model_path) and os.path.exists(classes_path):
        state.model = tf.keras.models.load_model(model_path)
        with open(classes_path) as f:
            state.class_names = json.load(f)
        state.loaded_at = datetime.now(timezone.utc).isoformat()
    else:
        print("WARNING: no trained model found in models/. Train it first with src/train_model.py.")


class PredictionOut(BaseModel):
    request_id: str
    predicted_disease: str
    confidence: float
    disease_status: str  # "Healthy" or "Diseased"
    top_3: list
    processed_at: str


@app.get("/")
def root():
    return {
        "service": "Tomato Disease Detection API",
        "version": API_VERSION,
        "docs": "/docs",
        "authentication": "Send your API key in the 'X-API-Key' header for /predict",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": API_VERSION,
        "model_loaded": state.model is not None,
        "classes": state.class_names,
        "model_loaded_at": state.loaded_at,
    }


@app.post("/predict", response_model=PredictionOut)
async def predict(file: UploadFile = File(...), api_key: str = Security(require_api_key)):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not available. Train it first.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    try:
        image_batch = preprocess_single_image(image_bytes, img_size=IMG_SIZE)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Please upload a valid JPG/PNG file.")

    proba = state.model.predict(image_batch, verbose=0)[0]
    top_idx = int(np.argmax(proba))
    label = state.class_names[top_idx]
    confidence = float(proba[top_idx])

    top3_idx = np.argsort(proba)[::-1][:3]
    top_3 = [{"disease": state.class_names[i], "confidence": round(float(proba[i]), 4)} for i in top3_idx]

    status = "Healthy" if label.strip().lower() in HEALTHY_LABELS else "Diseased"

    return PredictionOut(
        request_id=str(uuid.uuid4()),
        predicted_disease=label,
        confidence=round(confidence, 4),
        disease_status=status,
        top_3=top_3,
        processed_at=datetime.now(timezone.utc).isoformat(),
    )
