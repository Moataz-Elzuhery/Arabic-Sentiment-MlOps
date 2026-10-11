# Monitoring and drift detection

The FastAPI service exports Prometheus metrics at `/metrics`, compares live traffic with a
reference sample using the Population Stability Index (PSI), and alerts when the shift is large.

## What is monitored

| Signal | Metric | Why |
|---|---|---|
| Traffic and errors | `http_requests_total{path,status}` | request rate, 5xx rate |
| Latency | `http_request_duration_seconds` (histogram) | p50 / p95 of `/predict` |
| Predictions | `sentiment_predictions_total{label,model_version}` | label mix over time |
| Model confidence | `sentiment_prediction_confidence` (histogram) | a model that becomes unsure is a drift symptom |
| Input length | `sentiment_input_text_length_chars` (histogram) | proxy for a change in what users send |
| **Drift** | `sentiment_drift_psi{feature}` | PSI of `text_length` and `confidence` vs the reference |
| Drift window | `sentiment_drift_window_size` | PSI exists only once the window is large enough |

Only lengths, confidences and labels are recorded (also in `monitoring/data/predictions.jsonl`).
The review text is never logged.

## How drift is computed

- **Reference**: `scripts/build_reference.py` scores 2,000 random validation reviews with the
  production model and stores `text_length`, `confidence`, `label` in `monitoring/reference.csv`.
- **Live window**: the last `DRIFT_WINDOW` requests (default 300). PSI is exported only when the
  window holds at least `DRIFT_MIN_SAMPLES` (default 100).
- **PSI**: 10 bins whose edges are the reference deciles; `PSI = sum((live% - ref%) * ln(live% / ref%))`.
  Rule of thumb: below 0.1 stable, 0.1-0.25 moderate shift, above 0.25 significant shift.
- The gauge is computed when Prometheus scrapes, so it is always current. With no new traffic the
  window does not change, so the value (and a firing alert) stays as it was.

## Alerts (`monitoring/alerts.yml`)

| Alert | Condition | For |
|---|---|---|
| `DataDriftHigh` | `sentiment_drift_psi > 0.25` (per feature) | 1 min |
| `APIDown` | Prometheus cannot scrape the API | 1 min |
| `HighErrorRate` | more than 5% of requests answer 5xx | 2 min |
| `HighLatencyP95` | p95 of `/predict` above 1 s | 2 min |

Prometheus sends firing and resolved alerts to Alertmanager, which posts them to a small webhook
receiver (`monitoring/alert_sink.py`) that prints them: `docker compose --profile monitoring logs alert_sink`.
Swap the receiver in `monitoring/alertmanager.yml` for Slack or e-mail for real notifications.

## Run the demo

```powershell
# 1. reference sample from the production model (about 2-3 minutes on a CPU)
python -m scripts.build_reference --model-dir models/v1 --val-path data/processed/val.csv --n 2000 --output monitoring/reference.csv

# 2. start the API and the monitoring stack
docker compose --profile monitoring up -d --build
```

Open Grafana at http://localhost:3001 (user `admin`, password from `GRAFANA_ADMIN_PASSWORD`),
Prometheus alerts at http://localhost:9090/alerts and Alertmanager at http://localhost:9093.

```powershell
# 3. normal traffic: PSI stays low
python -m scripts.simulate_traffic --url http://localhost:8000 --mode normal --n 300 --rate 5

# 4. drifted traffic (only the first 1-3 words of each review): PSI rises, the alert fires after ~1 min
python -m scripts.simulate_traffic --url http://localhost:8000 --mode short --n 500 --rate 5
docker compose --profile monitoring logs alert_sink

# 5. back to normal: the alert resolves
python -m scripts.simulate_traffic --url http://localhost:8000 --mode normal --n 400 --rate 5

# 6. Evidently report (HTML) on the most recent 300 requests
python -m scripts.drift_report --last 300 --output reports/drift_report.html
```

The drifted traffic is **synthetic**: it shows that the pipeline detects a shift, not that real users
changed. Evidently uses its own binning, so its PSI values differ from the gauge's; both should agree
on which side of the threshold the traffic is.

## Limitations

- Only two features are monitored (input length and model confidence). There is no ground truth in
  production, so model accuracy itself cannot be monitored; PSI on inputs and outputs is a proxy.
- A shift in confidence depends on how the real model behaves on the shifted inputs; the input-length
  PSI is the more reliable trigger in the demo.
- Single API instance; the window and the prediction log live in that process and on a bind mount.
  The log file grows without rotation.
- Only the FastAPI service is instrumented (not the BentoML service or the canary router).
- The Grafana dashboard and the container images are provisioned from files; they were written to
  the documented formats but are only confirmed once you open them.
