import json
import os
from pathlib import Path

from .preprocessing import ID2LABEL, normalize_arabic


class Predictor:
    def __init__(self, model_dir: str, max_length: int = 128):
        import torch  # lazy: keeps tests/imports light

        from .model import load_model

        self._torch = torch
        self.tokenizer, self.model = load_model(model_dir)
        self.model.eval()
        self.max_length = max_length
        meta = Path(model_dir) / "metrics.json"
        self.model_version = (
            json.loads(meta.read_text()).get("model_version", "unknown")
            if meta.exists() else os.getenv("MODEL_VERSION", "unknown")
        )

    def predict(self, texts: list[str]) -> list[dict]:
        torch = self._torch
        clean = [normalize_arabic(t) for t in texts]
        enc = self.tokenizer(clean, truncation=True, max_length=self.max_length,
                             padding=True, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        return [
            {"label": ID2LABEL[i.item()], "confidence": round(c.item(), 4),
             "model_version": self.model_version}
            for c, i in zip(conf, idx)
        ]
