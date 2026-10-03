from fastapi.testclient import TestClient

from src.api import app


class FakePredictor:
    def predict(self, texts):
        return [{"label": "positive", "confidence": 0.94, "model_version": "v1"} for _ in texts]


def test_health_without_model_is_503():
    app.state.predictor = None
    assert TestClient(app).get("/health").status_code == 503


def test_health_ok_and_predict():
    app.state.predictor = FakePredictor()
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "healthy"}
    r = c.post("/predict", json={"text": "المنتج ممتاز جدًا"})
    assert r.status_code == 200
    assert r.json() == {"label": "positive", "confidence": 0.94, "model_version": "v1"}


def test_predict_validation():
    app.state.predictor = FakePredictor()
    c = TestClient(app)
    assert c.post("/predict", json={"text": ""}).status_code == 422
    assert c.post("/predict", json={"text": "   "}).status_code == 422
    assert c.post("/predict", json={}).status_code == 422
