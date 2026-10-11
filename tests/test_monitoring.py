import json
import random

from fastapi.testclient import TestClient

from src import monitoring
from src.api import app
from src.monitoring import DriftMonitor, psi


def _lengths(rng, n, lo, hi):
    return [rng.randint(lo, hi) for _ in range(n)]


def test_psi_same_distribution_is_small_and_shift_is_large():
    rng = random.Random(0)
    ref = _lengths(rng, 2000, 20, 300)
    same = _lengths(rng, 500, 20, 300)
    shifted = _lengths(rng, 500, 3, 25)
    assert psi(ref, same) < 0.1
    assert psi(ref, shifted) > 0.25


def _reference(rng, n=1000):
    return {"text_length": _lengths(rng, n, 20, 300),
            "confidence": [min(0.999, rng.betavariate(8, 1.5)) for _ in range(n)]}


def test_drift_monitor_waits_for_enough_samples():
    rng = random.Random(1)
    mon = DriftMonitor(_reference(rng), window=300, min_samples=100)
    for _ in range(50):
        mon.observe(100, 0.9)
    size, values = mon.snapshot()
    assert size == 50 and values == {}


def test_drift_monitor_flags_shifted_traffic():
    rng = random.Random(2)
    ref = _reference(rng)
    mon = DriftMonitor(ref, window=300, min_samples=100)
    for _ in range(300):
        mon.observe(rng.randint(3, 15), 0.55)
    size, values = mon.snapshot()
    assert size == 300
    assert values["text_length"] > 0.25 and values["confidence"] > 0.25


def test_drift_monitor_quiet_for_normal_traffic():
    rng = random.Random(3)
    ref = _reference(rng)
    mon = DriftMonitor(ref, window=300, min_samples=100)
    live = _reference(rng, 300)
    for length, conf in zip(live["text_length"], live["confidence"]):
        mon.observe(length, conf)
    _, values = mon.snapshot()
    assert values["text_length"] < 0.1 and values["confidence"] < 0.1


def test_from_csv(tmp_path):
    f = tmp_path / "ref.csv"
    f.write_text("text_length,confidence,label\n" + "\n".join(
        f"{20 + i},0.{50 + i % 49},positive" for i in range(200)) + "\n")
    mon = DriftMonitor.from_csv(str(f))
    assert mon.edges["text_length"]


class FakePredictor:
    def predict(self, texts):
        return [{"label": "positive", "confidence": 0.9, "model_version": "v1"} for _ in texts]


def test_metrics_endpoint_and_prediction_log(tmp_path):
    log_file = tmp_path / "pred.jsonl"
    monitoring.configure(reference_path=None, log_path=str(log_file))
    app.state.predictor = FakePredictor()
    client = TestClient(app)
    secret = "نص سري لا يجب تسجيله"
    assert client.post("/predict", json={"text": secret}).status_code == 200

    body = client.get("/metrics").text
    for name in ("sentiment_predictions_total", "sentiment_prediction_confidence_bucket",
                 "sentiment_input_text_length_chars_bucket", "http_requests_total",
                 "sentiment_drift_window_size", "sentiment_drift_enabled"):
        assert name in body

    line = json.loads(log_file.read_text().splitlines()[0])
    assert line["text_length"] == len(secret) and line["label"] == "positive"
    assert secret not in log_file.read_text()          # privacy: no raw text on disk
    monitoring.configure()                              # reset for other tests


def test_build_rows_stores_lengths_not_text():
    from scripts.build_reference import build_rows

    def fake(batch):
        return [{"label": "positive", "confidence": 0.91234} for _ in batch]

    rows = build_rows(["abc", "defgh", "x"], fake, batch_size=2)
    assert [r["text_length"] for r in rows] == [3, 5, 1]
    assert all(set(r) == {"text_length", "confidence", "label"} for r in rows)


def test_simulate_traffic_modes():
    import random

    from scripts.simulate_traffic import make_text

    base = ["كلمة أولى ثانية ثالثة رابعة", "نص آخر طويل نسبيا بكلمات كثيرة"]
    rng = random.Random(0)
    assert all(len(make_text(base, "short", rng).split()) <= 3 for _ in range(20))
    assert len(make_text(base, "long", rng).split()) > 8


def test_load_log_last_n(tmp_path):
    from scripts.drift_report import load_log

    f = tmp_path / "p.jsonl"
    f.write_text("\n".join(json.dumps({"text_length": i, "confidence": 0.9}) for i in range(10)))
    assert [r["text_length"] for r in load_log(str(f), last=3)] == [7, 8, 9]
