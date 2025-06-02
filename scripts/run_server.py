#!/usr/bin/env python
"""Run an inference server for mushroom classification."""

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from uvicorn import run

from mushroom_classifier.infer import MushroomInferenceModel


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    class_index: int
    class_name: str
    confidence: float
    probabilities: Optional[dict] = None


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str
    model_loaded: bool
    model_type: Optional[str] = None


app = FastAPI(
    title="Mushroom Classification API",
    description="API for mushroom classification using ML models",
    version="0.1.0",
)

# Global model instance
model = None


def load_model(
    model_path: str, class_mapping_path: Optional[str] = None, device: str = "cpu"
):
    """Load the model."""
    global model
    try:
        model = MushroomInferenceModel(
            model_path=model_path, class_mapping_path=class_mapping_path, device=device
        )
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    global model
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": model.model_type if model is not None else None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), return_probs: bool = False):
    """Prediction endpoint."""
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Read and process the image
        content = await file.read()
        img = Image.open(BytesIO(content)).convert("RGB")

        # Make prediction
        result = model.predict(img, return_probabilities=return_probs)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the inference server")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model file"
    )
    parser.add_argument(
        "--class-mapping", type=str, help="Path to the class mapping file"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda)"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Load the model
    try:
        load_model(args.model_path, args.class_mapping, args.device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Run the server
    print(f"Starting server at http://{args.host}:{args.port}")
    run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
