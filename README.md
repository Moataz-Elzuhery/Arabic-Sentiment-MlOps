# Arabic Sentiment MLOps

Fine-tuned Arabic BERT (AraBERT) for 3-class sentiment (positive / negative / neutral),
served with FastAPI + Docker. Built session by session toward a full MLOps system.

## Status
- [x] Session 1: Model + FastAPI + Docker
- [ ] Session 2: MLflow + DVC + GitHub Actions
- [ ] Session 3: BentoML + Locust + Canary
- [ ] Session 4: Distillation + INT8 + TensorRT + Benchmark
- [ ] Session 5: Drift + Prometheus + Grafana + Alert

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
make test

# 1) put a raw reviews file in data/raw/, then:
python -m scripts.prepare_data --input data/raw/reviews.csv --text-col review --rating-col rating
#    (or --label-col label if it already has positive/negative/neutral)

# 2) smoke test, then the real run
make smoke
make train            # writes models/v1 + metrics.json

# 3) serve
docker compose up --build
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"text": "المنتج ممتاز جدًا"}'
curl localhost:8000/health
```

## Results
| Model | Accuracy | F1 macro |
|-------|----------|----------|
| AraBERT v02 (v1) | 0.898 | 0.847 |
