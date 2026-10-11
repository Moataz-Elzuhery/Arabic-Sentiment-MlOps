"""Prometheus metrics, drift monitoring (PSI) and prediction logging for the API.

Privacy: only lengths, confidences and labels are recorded, never the review text.
Monitoring must never break serving, so every hook swallows its own errors.
"""
import bisect
import csv
import json
import logging
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Histogram,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily

log = logging.getLogger("monitoring")

FEATURES = ("text_length", "confidence")
PSI_ALERT_THRESHOLD = 0.25  # rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 significant

# ---------------------------------------------------------------- PSI (pure python)


def make_edges(values, bins: int = 10) -> list[float]:
    """Interior bin edges at the reference quantiles (unique and increasing)."""
    xs = sorted(values)
    n = len(xs)
    edges: list[float] = []
    for i in range(1, bins):
        q = xs[min(n - 1, int(i * n / bins))]
        if not edges or q > edges[-1]:
            edges.append(q)
    return edges


def bin_proportions(values, edges, eps: float = 1e-4) -> list[float]:
    counts = [0] * (len(edges) + 1)
    for v in values:
        counts[bisect.bisect_right(edges, v)] += 1
    total = max(len(values), 1)
    return [max(c / total, eps) for c in counts]


def psi_from_proportions(expected, actual) -> float:
    return sum((a - e) * math.log(a / e) for e, a in zip(expected, actual))


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index of `current` against `reference` (reference-quantile bins)."""
    edges = make_edges(reference, bins)
    return psi_from_proportions(bin_proportions(reference, edges),
                                bin_proportions(current, edges))


class DriftMonitor:
    """Sliding window of recent requests compared with a fixed reference sample."""

    def __init__(self, reference: dict[str, list[float]], window: int = 300,
                 min_samples: int = 100, bins: int = 10):
        self.edges = {f: make_edges(reference[f], bins) for f in FEATURES}
        self.ref_props = {f: bin_proportions(reference[f], self.edges[f]) for f in FEATURES}
        self.window = {f: deque(maxlen=window) for f in FEATURES}
        self.min_samples = min_samples
        self._lock = threading.Lock()

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "DriftMonitor":
        ref: dict[str, list[float]] = {f: [] for f in FEATURES}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for feat in FEATURES:
                    ref[feat].append(float(row[feat]))
        if not ref["text_length"]:
            raise ValueError(f"empty reference file: {path}")
        return cls(ref, **kwargs)

    def observe(self, text_length: float, confidence: float) -> None:
        with self._lock:
            self.window["text_length"].append(float(text_length))
            self.window["confidence"].append(float(confidence))

    def snapshot(self) -> tuple[int, dict[str, float]]:
        """(window size, PSI per feature). PSI is empty until `min_samples` requests arrived."""
        with self._lock:
            size = len(self.window["text_length"])
            data = {f: list(self.window[f]) for f in FEATURES}
        if size < self.min_samples:
            return size, {}
        return size, {f: psi_from_proportions(self.ref_props[f],
                                              bin_proportions(data[f], self.edges[f]))
                      for f in FEATURES}


# ---------------------------------------------------------------- Prometheus metrics

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency",
                         ["method", "path"],
                         buckets=(.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10))
PREDICTIONS = Counter("sentiment_predictions_total", "Predictions by label",
                      ["label", "model_version"])
CONFIDENCE = Histogram("sentiment_prediction_confidence", "Top-class probability",
                       buckets=(.4, .5, .6, .7, .8, .9, .95, .99, 1.0))
TEXT_LENGTH = Histogram("sentiment_input_text_length_chars", "Input length in characters",
                        buckets=(5, 10, 20, 40, 80, 160, 320, 640, 1280))


class _DriftCollector:
    """Computes PSI at scrape time so the gauges are always current."""

    def collect(self):
        psi_gauge = GaugeMetricFamily("sentiment_drift_psi",
                                      "Population Stability Index vs the reference sample",
                                      labels=["feature"])
        size_gauge = GaugeMetricFamily("sentiment_drift_window_size",
                                       "Requests in the drift window")
        enabled_gauge = GaugeMetricFamily("sentiment_drift_enabled",
                                          "1 if a drift reference is loaded")
        monitor = STATE.drift
        enabled_gauge.add_metric([], 1.0 if monitor else 0.0)
        if monitor:
            size, values = monitor.snapshot()
            size_gauge.add_metric([], float(size))
            for feature, value in values.items():
                psi_gauge.add_metric([feature], value)
        else:
            size_gauge.add_metric([], 0.0)
        yield psi_gauge
        yield size_gauge
        yield enabled_gauge


class _State:
    def __init__(self):
        self.drift: DriftMonitor | None = None
        self.log_path: Path | None = None
        self._log_lock = threading.Lock()


STATE = _State()
REGISTRY.register(_DriftCollector())


def configure(reference_path: str | None = None, log_path: str | None = None,
              window: int = 300, min_samples: int = 100) -> None:
    STATE.drift = None
    STATE.log_path = None
    if reference_path and Path(reference_path).exists():
        STATE.drift = DriftMonitor.from_csv(reference_path, window=window,
                                            min_samples=min_samples)
        log.info("Drift monitoring on (reference: %s)", reference_path)
    elif reference_path:
        log.warning("Drift reference not found: %s (drift monitoring is off)", reference_path)
    if log_path:
        STATE.log_path = Path(log_path)
        STATE.log_path.parent.mkdir(parents=True, exist_ok=True)


def configure_from_env() -> None:
    configure(os.getenv("MONITORING_REFERENCE"), os.getenv("PREDICTION_LOG"),
              window=int(os.getenv("DRIFT_WINDOW", "300")),
              min_samples=int(os.getenv("DRIFT_MIN_SAMPLES", "100")))


def record_prediction(text_length: int, result: dict) -> None:
    try:
        confidence = float(result["confidence"])
        PREDICTIONS.labels(result["label"], result.get("model_version", "?")).inc()
        CONFIDENCE.observe(confidence)
        TEXT_LENGTH.observe(text_length)
        if STATE.drift:
            STATE.drift.observe(text_length, confidence)
        if STATE.log_path:
            line = json.dumps({"ts": round(time.time(), 3), "text_length": text_length,
                               "confidence": round(confidence, 4), "label": result["label"],
                               "model_version": result.get("model_version", "?")})
            with STATE._log_lock, open(STATE.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        log.exception("could not record prediction")


def record_http(method: str, path: str, status: int, seconds: float) -> None:
    try:
        HTTP_REQUESTS.labels(method, path, str(status)).inc()
        HTTP_LATENCY.labels(method, path).observe(seconds)
    except Exception:
        log.exception("could not record http metrics")


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
