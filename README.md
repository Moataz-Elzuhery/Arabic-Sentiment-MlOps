# Arabic Sentiment MLOps

[![CI](https://github.com/Moataz-Elzuhery/Arabic-Sentiment-MlOps/actions/workflows/ci.yml/badge.svg)](https://github.com/Moataz-Elzuhery/Arabic-Sentiment-MlOps/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange)](https://docs.astral.sh/ruff/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/data%20versioning-DVC-945DD6)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![BentoML](https://img.shields.io/badge/serving-BentoML-lightgrey)](https://www.bentoml.com/)
[![Docker](https://img.shields.io/badge/container-Docker-2496ED)](https://www.docker.com/)
[![AraBERT](https://img.shields.io/badge/model-AraBERT%20v02-FFD21E)](https://huggingface.co/aubmindlab/bert-base-arabertv02)

**Arabic hotel-review sentiment classification (positive / negative / neutral) with an end-to-end
MLOps pipeline.** AraBERT fine-tuned on the HARD dataset (Modern Standard Arabic and dialects),
served with FastAPI and BentoML, with MLflow tracking, DVC, a CI/CD quality gate, load testing
and a canary release.

The point of the project is the **system around the model**, not a record-breaking score.

## Status

| Area | What is in the repo | State |
|---|---|---|
| Model + API + Docker | Fine-tuned AraBERT, FastAPI (`/predict`, `/health`), Docker Compose | Done |
| Experiment tracking | MLflow: 5 runs, Model Registry (`baseline` / `production`) | Done |
| Data & pipeline versioning | DVC (`prepare` -> `train` -> `evaluate`) | Done (smoke-sized params) |
| CI/CD | GitHub Actions: lint, tests, Docker build, quality gate, push to GHCR | Done |
| Serving & load testing | Locust baseline, BentoML with adaptive batching | Done |
| Canary release | nginx 95/5 split, staged rollout, instant rollback | Done |
| Monitoring | Drift detection (PSI), Prometheus, Grafana, alerts | Planned |

## Architecture

```
HARD reviews --DVC--> prepare --> train --> evaluate           (reproducible pipeline)
                                    |
                           MLflow (5 runs) --> Model Registry (v1 baseline, v2 production)
                                    |
                                  models/
                                    |
        +---------------------------+----------------------------+
        |                           |                            |
   FastAPI :8000              BentoML :3000                nginx router :8080
   (single requests)       (adaptive batching)          95% -> api_v1 / 5% -> api_v2

GitHub Actions: Lint -> Tests -> Build Docker -> Quality gate (F1 vs baseline) -> Push image
```

## Results

### Data
[HARD](https://github.com/elnagara/HARD-Arabic-Dataset) (Hotel Arabic-Reviews Dataset; reviews collected
from Booking.com in June/July 2016, in Modern Standard Arabic and dialects), unbalanced version. Ratings were mapped to labels: 1-2 -> negative, 3 -> neutral, 4-5 -> positive.
After removing duplicate texts: **train 320,064 / val 40,008 / test 40,009** (stratified 80/10/10,
about 67% positive, 20% neutral, 13% negative).


### Model
`aubmindlab/bert-base-arabertv02` fine-tuned for 3-class classification (max length 128,
AdamW, 10% warmup, weight decay 0.01, seed 42, fp16 on GPU). Text is normalised before tokenising
(diacritics, tatweel, elongated letters, URLs, whitespace) in `src/preprocessing.py`; the same
function runs at training and serving time.

### Hyperparameter runs (MLflow experiment `arabic-sentiment`)
Each run trained on a random 60,000-review subset of the training set, on a Colab T4.
Model selection used **validation** F1; the test set is only reported.

| Run | LR | Batch | Epochs | Val F1 (macro) | Test F1 (macro) | Test acc. | F1 neutral | Train time |
|---|---|---|---|---|---|---|---|---|
| **run4** (selected) | 3e-5 | 16 | 2 | **0.8529** | 0.8474 | 0.8979 | 0.758 | 15 min |
| run2 | 3e-5 | 8 | 2 | 0.8513 | 0.8470 | 0.8971 | 0.757 | 21 min |
| run1 | 2e-5 | 8 | 2 | 0.8494 | 0.8468 | 0.8975 | 0.756 | 21 min |
| run5 | 3e-5 | 16 | 3 | 0.8489 | 0.8460 | 0.8960 | 0.758 | 23 min |
| run3 | 5e-5 | 8 | 3 | 0.8423 | 0.8381 | 0.8918 | 0.745 | 32 min |

Registered model `arabic-sentiment`: **v1** (alias `baseline`) and **v2** (alias `production`,
= run4). v1 and v2 use the same configuration and seed, so their test scores are identical
(F1 macro 0.8474). Per class for v1: negative 0.834, neutral 0.758, positive 0.950.

How to read this: the differences between the first four runs are small and each run used a single
seed, so I do not claim any of them is better than another. The higher learning rate (run3) was
clearly the worst. Neutral is the hardest class in every run (3-star reviews often mix praise and
criticism).

## Quickstart

Requirements: Python 3.12, Docker Desktop, Git.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest -q
```

The trained weights are not in git (they are large). Put a trained model folder in `models/v1`
(it needs `model.safetensors`, `config.json`, tokenizer files and `metrics.json`), then:

```powershell
Copy-Item .env.example .env        # local settings, see "Configuration"
docker compose up -d --build api
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body '{"text":"good product"}'
```

API contract:

```
POST /predict   {"text": "المنتج ممتاز جدًا"}
             -> {"label": "positive", "confidence": 0.94, "model_version": "v1"}
GET  /health -> {"status": "healthy"}        (503 while the model is not loaded)
```

## MLOps

### MLflow
Five runs are tracked (parameters, validation/test metrics, training time). The runs were produced
on Colab (`notebooks/train_sweep_colab.ipynb`) and the tracking database was copied back. Models are
registered with `scripts/register_model.py`:

```powershell
python -m scripts.register_model --model-dir models/v1 --alias baseline
python -m scripts.register_model --model-dir models/v2 --alias production
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```

### DVC pipeline
`dvc.yaml` defines `prepare` -> `train` -> `evaluate`; the raw dataset is tracked with
`data/raw/unbalanced-reviews.txt.dvc`. Stages are cached and re-run only when their inputs change.

```powershell
dvc repro            # whole pipeline
dvc dag              # show the graph
dvc metrics show
```

Important: the default `params.yaml` is **smoke-sized** (200 samples, 1 epoch) so `dvc repro`
finishes in minutes on a CPU. Its metrics (`metrics/eval.json`, F1 ~0.25) are therefore not the
model's quality. The registered v1/v2 models were trained on Colab with the same `src.train` code
at 60,000 samples. The full-scale parameters are noted in `params.yaml`. The dataset itself is not
stored in a DVC remote, so download `unbalanced-reviews.tsv.rar` from the `data/` folder of the HARD repository, extract it
(it is a RAR archive; use 7-Zip or WinRAR) and place the extracted file at
`data/raw/unbalanced-reviews.txt` (rename it if needed) before `dvc repro`.

### CI/CD (GitHub Actions)
On every push and pull request (`.github/workflows/ci.yml`):

1. **Lint + Tests** - `ruff` and `pytest`.
2. **Build Docker image** - builds the API image and checks that the container boots (without a
   model mounted, `/health` must answer 503).
3. **Quality gate** - `scripts/quality_gate.py` compares the candidate metrics
   (`metrics/production.json`) with the baseline (`metrics/baseline.json`) and fails if macro F1
   drops by more than 0.005, or if any class falls below an F1 of 0.70.
4. **Push image** - only on the default branch and only after the gate passes; publishes to GitHub
   Container Registry.

The gate was verified on a pull request: with F1 lowered to 0.8074 it failed, and the push job was
skipped. Note what the gate is: it checks the **metrics files committed to git**. CI has no GPU and
no model, so it does not retrain; it blocks committing results worse than the baseline.

## Serving and load testing

Method: Locust, 10 concurrent users with 0.1-0.5 s think time, 90 s per run, ramp-up discarded
(`--reset-stats`), each service run alone. Hardware: Intel Core i9-10885H (8 cores / 16 threads),
16 GB RAM, Windows, Docker Desktop (containers get 4 CPUs and about 5.8 GiB of RAM, as reported by
`docker info`). Locust and the service share the same machine, so absolute numbers are specific to
this setup and CPU-only inference.

| Service | Req/s | p50 | p95 | p99 | Failures |
|---|---|---|---|---|---|
| FastAPI (one request at a time) | 19.06 | 200 ms | 340 ms | 400 ms | 0 |
| BentoML (adaptive batching) | 21.06 | 160 ms | 290 ms | 380 ms | 0 |

Raw data and per-run summaries: `benchmarks/`.

Notes:
- Each number comes from **one run**. BentoML is lower on p50 and mean latency (~20-27%); the p99
  and requests/s differences are small and I would not call them improvements without repeats.
- With think time, requests/s is mostly determined by latency (10 users / (0.3 s + latency)), so it
  is not a capacity measurement.
- A first FastAPI run that included the ramp-up had p99 = 6,600 ms: the first requests after
  start-up are slow (likely model warm-up), which is why the table uses `--reset-stats` and why a
  warm-up before taking traffic is worth having.
- BentoML's `max_latency_ms` must be larger than the model's latency under load. With 50 ms, about
  half of the requests were rejected with HTTP 503 in my tests; the default here is 2000 ms.

```powershell
docker compose --profile bento up -d --build bento     # BentoML on :3000, POST /predict {"texts": ["..."]}
```

## Canary release

`docker compose --profile canary up -d --build` starts two copies of the API (`api_v1` stable,
`api_v2` candidate) behind an nginx router on port 8080. Each request is assigned independently, so
shares are statistical. A check with 400 requests at `CANARY_PCT=5` gave 95.8% v1 / 4.2% v2.

```powershell
python -m scripts.canary_check --url http://localhost:8080 --n 400 --expected 5
```

Rollout stages (5% -> 25% -> 50% -> 100%), promotion criteria and the measured per-version latency
are in [`docs/canary_rollout.md`](docs/canary_rollout.md). Rollback is instant: set
`CANARY_ENABLED=0` and recreate the router (nginx rejects a 0% split, hence the separate switch).

Here v1 and v2 are the same model configuration, so the canary demonstrates the **routing and
rollout mechanics**; model quality is controlled separately by the quality gate. Because both versions run the same
weights, any latency difference between them in the rollout log is noise (small samples at low
shares, one run per stage), not a model effect.

## Configuration

Docker Compose reads `.env` (copy it from `.env.example`; `.env` is git-ignored).

| Variable | Default | Used by |
|---|---|---|
| `API_MODEL_DIR` | `/app/models/v1` | `api`, `bento` |
| `STABLE_MODEL_DIR` / `CANARY_MODEL_DIR` | `/app/models/v1` / `/app/models/v2` | canary services |
| `MAX_BATCH_SIZE` / `MAX_LATENCY_MS` | `32` / `2000` | BentoML batching |
| `CANARY_PCT` | `5` | share of traffic to the canary (1-100) |
| `CANARY_ENABLED` | `1` | `0` = rollback to the stable model |

A variable set in the shell (`$env:CANARY_PCT=25`) overrides `.env`.

## Project structure

```
src/                  preprocessing, model, training, evaluation, inference, FastAPI app, MLflow helper
scripts/              prepare_data, register_model, export_metrics, quality_gate, locust_summary, canary_check
configs/train.yaml    default training configuration
params.yaml           DVC parameters (smoke-sized by default)
dvc.yaml / dvc.lock   pipeline definition and lock file
service.py            BentoML service (adaptive batching)
canary/               nginx config template for the traffic split
loadtest/             Locust file
benchmarks/           load-test summaries and raw Locust CSVs
metrics/              eval.json (DVC), production.json and baseline.json (CI gate)
notebooks/            Colab notebooks used for GPU training
docs/                 canary rollout plan and log
tests/                pytest suite
.github/workflows/    CI
Dockerfile, Dockerfile.bento, docker-compose.yml, requirements*.txt, .env.example
```

## Reproducing the training

Fine-tuning needs a GPU. On a Colab T4, `notebooks/train_sweep_colab.ipynb` runs the five
configurations (about 15-32 minutes each at 60,000 samples) and logs them to MLflow;
`notebooks/train_colab.ipynb` runs a single configuration. Both copy the project from Google Drive
(see the notebook header for the expected folder layout). Training locally on a CPU measured about
2 samples per second, which is not practical for this dataset.

## Limitations

- The model was trained on **hotel** reviews. It has not been evaluated on product reviews, so
  accuracy on e-commerce text is unknown and likely lower (domain shift, dialect).
- Neutral is the weakest class (F1 about 0.76). Class weights or more data for that class were not tried.
- Single seed per configuration; no confidence intervals.
- Trained on a 60,000-review subset of the 320,064 training reviews.
- Load-test numbers are from one machine, one run each, CPU-only.
- v1 and v2 are the same configuration, so the canary does not compare two different models.
- The DVC pipeline's default parameters are a smoke test, and the dataset is not in a DVC remote.

## Roadmap

- Monitoring: drift detection on input text length and model confidence (PSI), Prometheus metrics,
  Grafana dashboard and an alert when PSI exceeds 0.25.

## Dataset citation

Elnagar A., Khalifa Y.S., Einea A. (2018). *Hotel Arabic-Reviews Dataset Construction for Sentiment
Analysis Applications.* In: Shaalan K., Hassanien A., Tolba F. (eds) Intelligent Natural Language
Processing: Trends and Applications. Studies in Computational Intelligence, vol 740, pp. 35-52.
Springer International Publishing.
https://doi.org/10.1007/978-3-319-67056-0_3