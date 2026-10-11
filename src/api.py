import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from . import monitoring

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
        monitoring.configure_from_env()
    except Exception:
        log.exception("Could not configure monitoring")
    try:
        from .inference import Predictor
        app.state.predictor = Predictor(MODEL_DIR)
        log.info("Model loaded from %s", MODEL_DIR)
    except Exception:
        log.exception("Could not load model from %s", MODEL_DIR)
    yield


app = FastAPI(title="Arabic Sentiment API", lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = route.path if route else "unmatched"
    if path != "/metrics":
        monitoring.record_http(request.method, path, response.status_code,
                               time.perf_counter() - start)
    return response


@app.get("/metrics", include_in_schema=False)
def metrics():
    body, content_type = monitoring.render_metrics()
    return Response(content=body, media_type=content_type)


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
    result = predictor.predict([req.text])[0]
    monitoring.record_prediction(len(req.text), result)
    return result
