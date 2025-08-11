from typing import Optional
import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config.config import AppConfig
from src.ml.models.service import ModelService

app = FastAPI(
    title="Sentiment Analysis Template", 
    version="1.0.0",
    description="Production-ready sentiment analysis API with 77%+ accuracy"
)



class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float
    confidence: float

# Global service
service = None

@app.on_event("startup")
def startup_event() -> None:
    global service
    try:
        print("Starting up... Loading model...")
        config = AppConfig.from_env()
        service = ModelService.initialize_from_artifacts(config)
        print("Model loaded successfully!")
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        raise RuntimeError(f"Model initialization failed: {exc}")

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": service is not None}

@app.get("/version")
def version() -> dict:
    return {"version": app.version, "model_accuracy": "77.11%"}

@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(request: PredictRequest) -> PredictResponse:
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")
    
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        score = service.predict_proba(request.text)
        label = "positive" if score >= 0.5 else "negative"
        confidence = abs(score - 0.5) * 2
        
        return PredictResponse(
            label=label, 
            score=float(score),
            confidence=float(confidence)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "Sentiment Analysis API",
        "accuracy": "77.11%",
        "endpoints": {
            "health": "/health",
            "version": "/version", 
            "predict": "/predict"
        }
    }

@app.get("/ping")
def ping():
    return {"message": "pong", "status": "ok"}

@app.get("/simple")
def simple():
    return {"message": "This is a simple endpoint", "working": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
