"""BentoML serving with server-side adaptive batching.

Run locally:   bentoml serve service:SentimentService --port 3000
Request:       POST /predict  {"texts": ["المنتج ممتاز"]}   ->  [{"label": ..., "confidence": ..., "model_version": ...}]

Many concurrent single-item requests are grouped by BentoML into one batch before
they reach the model, which is what improves throughput under load.
"""
import os

import bentoml

from src.inference import Predictor

MAX_BATCH = int(os.getenv("MAX_BATCH_SIZE", "32"))
# Max time a request may wait in the batching queue. If the model cannot keep up within this
# budget BentoML answers 503 (load shedding). 50 ms is far too tight for a BERT on CPU.
MAX_LATENCY_MS = int(os.getenv("MAX_LATENCY_MS", "2000"))


@bentoml.service(workers=1, traffic={"timeout": 60})
class SentimentService:
    def __init__(self) -> None:
        self.predictor = Predictor(os.getenv("MODEL_DIR", "models/v1"))

    @bentoml.api(batchable=True, max_batch_size=MAX_BATCH, max_latency_ms=MAX_LATENCY_MS)
    def predict(self, texts: list[str]) -> list[dict]:
        return self.predictor.predict(texts)
