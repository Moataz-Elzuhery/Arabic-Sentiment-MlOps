import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("api")
MODEL_DIR = os.getenv("MODEL_DIR", "models/v1")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None
    try:
        from .inference import Predictor
        app.state.predictor = Predictor(MODEL_DIR)
        log.info("Model loaded from %s", MODEL_DIR)
    except Exception:
        log.exception("Could not load model from %s", MODEL_DIR)
    yield


app = FastAPI(title="Arabic Sentiment API", lifespan=lifespan)


@app.get("/health")
def health():
    if getattr(app.state, "predictor", None) is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be blank")
    return predictor.predict([req.text])[0]
