# Arabic Sentiment MLOps

Fine-tuned Arabic BERT ([AraBERT v02](https://huggingface.co/aubmindlab/bert-base-arabertv02)) for **3-class sentiment** (positive / negative / neutral) on Arabic product reviews, served with **FastAPI + Docker**, and wrapped in an MLOps workflow: data/experiment versioning, CI with a model-quality gate, load testing, and (planned) drift monitoring.

Built session by session toward a full MLOps system.

![CI](https://github.com/Moataz-Elzuhery/Arabic-Sentiment-MlOps/actions/workflows/ci.yml/badge.svg)

---

## Table of Contents

1. [Status](#status)
2. [How it works](#how-it-works)
3. [Results](#results)
4. [Quickstart](#quickstart)
5. [CI/CD and the quality gate](#cicd-and-the-quality-gate)
6. [Load testing (Locust)](#load-testing-locust)
7. [Project structure](#project-structure)
8. [Problems I hit and how I solved them](#problems-i-hit-and-how-i-solved-them)
9. [Roadmap: Session 4 (Drift & Monitoring)](#roadmap-session-4-drift--monitoring)
10. [Limitations](#limitations)

---

## Status

- [x] **Session 1:** Model + FastAPI + Docker
- [x] **Session 2:** MLflow + DVC + GitHub Actions
- [ ] **Session 3:** BentoML + Locust + Canary
  - [x] Locust load tests (baseline + saturation, FastAPI vs BentoML)
  - [x] BentoML serving with batching (`docker compose --profile bento`)
  - [ ] Canary rollout
- [ ] **Session 4:** Drift + Prometheus + Grafana + Alert

---

## How it works

```
raw reviews CSV ──► prepare_data ──► train / val / test splits   (tracked with DVC, params.yaml)
                                          │
                                          ▼
                              src.train  (AraBERT fine-tuning, Hugging Face Trainer)
                                          │
                       ┌──────────────────┴──────────────────┐
                       ▼                                     ▼
              models/v1  (weights + tokenizer)        metrics.json  (accuracy, F1 macro)
                       │                                     │
                       ▼                                     ▼
            FastAPI  (/predict, /health)           CI quality gate  (F1 vs baseline)
                       │
                       ▼
                Docker image  ──►  GHCR (pushed only from main)
```

**Training.** `src.train` loads `aubmindlab/bert-base-arabertv02`, adds a fresh 3-class classification head, and fine-tunes it with the Hugging Face `Trainer`. Hyperparameters live in `configs/train.yaml`. Evaluation runs every epoch and the best checkpoint (by macro-F1) is kept. A `--max-samples` flag allows a fast smoke run on a small subset.

**Serving.** `src/api.py` is a FastAPI app. The model directory is chosen with the `MODEL_DIR` environment variable.

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | `{"text": "..."}` → predicted label + confidence |
| `/health` | GET | Liveness. Returns **503** if no model is loaded (used by the CI smoke test) |

**BentoML.** The same model is also served by a BentoML service (compose profile `bento`, port 3000) with request batching, so FastAPI and BentoML can be load-tested side by side (see [Load testing](#load-testing-locust)). Its request format differs from the FastAPI one, which is why the load test takes an `API_STYLE=bento` switch.

**Data and experiments.** DVC tracks the data pipeline (`dvc.yaml`, `params.yaml`, `dvc.lock`), so a run is reproducible from the committed pointers. MLflow is used for experiment tracking.

**Quality gate.** CI compares the candidate model's F1 (`metrics/production.json`) against a committed baseline (`metrics/baseline.json`) and fails the pipeline if F1 dropped.

---

## Results

| Model            | Accuracy | F1 macro |
| ---------------- | -------- | -------- |
| AraBERT v02 (v1) | 0.898    | 0.847    |

> The `neutral` class is the minority class in review datasets, so macro-F1 is the metric to watch rather than accuracy alone.

---

## Quickstart

**Requirements:** Python **3.12**, Docker, Git.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
make test

# 1) put a raw reviews file in data/raw/, then:
python -m scripts.prepare_data --input data/raw/reviews.csv --text-col review --rating-col rating
#    (or --label-col label if it already has positive/negative/neutral)

# 2) smoke test, then the real run
make smoke            # 200 samples, 1 epoch: checks the pipeline end to end
make train            # writes models/v1 + metrics.json

# 3) serve
docker compose up --build
```

Call the API:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"text": "المنتج ممتاز جدًا"}'
curl localhost:8000/health
```

**On Windows PowerShell** use `curl.exe` (plain `curl` is an alias for `Invoke-WebRequest`) or:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/predict `
  -ContentType "application/json" -Body '{"text": "المنتج ممتاز جدًا"}'
Invoke-RestMethod http://localhost:8000/health
```

**BentoML service** (run one service at a time when benchmarking, they compete for the same CPU):

```bash
docker compose stop api
docker compose --profile bento up -d bento     # http://localhost:3000
```

**Make targets:** `test`, `lint`, `smoke`, `train`, `serve`.

---

## CI/CD and the quality gate

`.github/workflows/ci.yml` runs on every push to `main` and on every pull request. Jobs run in sequence (`needs:`):

| Job (check name) | What it does |
|---|---|
| `Lint + Tests` | `ruff check .` then `pytest -q` |
| `Build Docker image` | Builds the image and smoke-tests it (no model mounted, so `/health` must answer 503) |
| `Quality gate (F1 vs baseline)` | `scripts/quality_gate.py` fails if F1 dropped below the baseline |
| `Push image to GHCR` | Only on pushes to the default branch; not run on pull requests |

**Branch protection.** `main` is protected by a ruleset (`protect-main`): changes go through a pull request, and the three checks above must pass before the merge button unlocks.

All three checks are marked *required*, not only the quality gate. Because the jobs are chained with `needs:`, a failing `Lint + Tests` makes the later jobs **skipped**, and GitHub treats a skipped check as passing. Requiring only the quality gate would let a lint failure through.

---

## Load testing (Locust)

All runs: PyTorch model on **CPU only**, Docker, one local machine (Dell Precision 7550, i9-10885H, 16 GB RAM), Locust running in Docker on the same host. Because Locust competes for the same CPU, treat the numbers as a comparison between variants on this machine, not as production capacity.

### 1. Moderate load (users wait between requests)

| Run | Duration | Req/s | p50 (ms) | p95 (ms) | p99 (ms) | Failures |
|---|---|---|---|---|---|---|
| baseline_fastapi | 60 s | 12.7 | 300 | 710 | 6600 | 0 |
| bento_batched | 90 s | 21.1 | 160 | 290 | 380 | 0 |

10 users in both. These two rows were **not** run with identical settings (duration and wait time differ) and the FastAPI baseline contains the slow first requests (see problem 12), so do not compare them with each other. The saturation test below is the fair comparison.

### 2. Saturation (no wait time, 10 users, 60 s, after a 30 s warm-up)

| Service | Requests | Req/s | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) | Failures |
|---|---|---|---|---|---|---|---|
| FastAPI (:8000) | 1223 | 21.1 | 460 | 640 | 760 | 963 | 0 |
| BentoML, batched (:3000) | 1495 | 25.9 | 390 | 470 | 540 | 594 | 0 |

Sanity check (Little's law): req/s x mean latency is about 10 for both services (21.1 x 0.473 s and 25.9 x 0.386 s), i.e. all 10 users were always waiting on the server, so this really is saturation.

**Reading the result.** On this CPU both services top out at roughly 21 to 26 req/s. BentoML with batching was about 23 % higher in throughput and had lower tail latency (p95 470 vs 640 ms), but this is **one run per service**, and an earlier discarded FastAPI run reached about 24 to 25 req/s in its final window. Run-to-run variation on a laptop is therefore of the same order as the gap. The honest conclusion for now: batching helps tail latency a little and does not change capacity dramatically, which is plausible when the CPU is the bottleneck. A conclusive claim needs several alternating runs and the median.

### How to reproduce

Wait for the model to be loaded (HTTP 200 on `/health`), warm up, then measure. `docker compose up -d` returns immediately, long before the model is loaded, so skipping the wait makes Locust report connection failures.

```powershell
function Wait-Http($url) {
  for ($i=0; $i -lt 180; $i++) {
    try { $r = Invoke-WebRequest -UseBasicParsing $url -TimeoutSec 3; if ($r.StatusCode -eq 200) { return } } catch {}
    Start-Sleep 1
  }
  throw "$url never became ready"
}
function Run-Locust($port, $prefix, $style) {
  $s = if ($style) { @("-e","API_STYLE=$style") } else { @() }
  # 30 s warm-up (output discarded), then the measured 60 s run
  docker run --rm @s -e WAIT_MIN=0 -e WAIT_MAX=0 -v "${PWD}\loadtest:/mnt/locust" -v "${PWD}\benchmarks\raw:/out" locustio/locust -f /mnt/locust/locustfile.py --host "http://host.docker.internal:$port" --headless -u 2 -r 2 -t 30s --csv /out/warmup_tmp | Out-Null
  docker run --rm @s -e WAIT_MIN=0 -e WAIT_MAX=0 -v "${PWD}\loadtest:/mnt/locust" -v "${PWD}\benchmarks\raw:/out" locustio/locust -f /mnt/locust/locustfile.py --host "http://host.docker.internal:$port" --headless -u 10 -r 5 -t 60s --reset-stats --csv /out/$prefix | Out-Null
}

docker compose stop bento
docker compose up -d api
Wait-Http "http://localhost:8000/health"
Run-Locust 8000 sat_fastapi $null
docker compose stop api

docker compose --profile bento up -d bento
Wait-Http "http://localhost:3000/readyz"      # BentoML's built-in readiness endpoint
Run-Locust 3000 sat_bento "bento"
docker compose stop bento

Remove-Item benchmarks\raw\warmup_tmp* -ErrorAction SilentlyContinue
python -m scripts.locust_summary --csv benchmarks/raw/sat_fastapi_stats.csv --label sat_fastapi --users 10 --duration 60
python -m scripts.locust_summary --csv benchmarks/raw/sat_bento_stats.csv --label sat_bento --users 10 --duration 60
```

Acceptance rule for a run: **0 failures** and no multi-second outlier in `Max`. Keep users, duration and payload identical between variants. Raw Locust CSVs are kept in `benchmarks/raw/`.

---

## Project structure

```
.
├── src/                   # train, evaluate, API (FastAPI), metrics helpers
├── scripts/               # prepare_data, quality_gate, locust_summary
├── configs/               # train.yaml (hyperparameters)
├── tests/                 # pytest suite
├── loadtest/              # locustfile.py
├── benchmarks/raw/        # raw Locust CSVs (evidence for the table above)
├── metrics/               # baseline.json / production.json (used by the quality gate)
├── data/                  # DVC-tracked, not committed
├── models/                # trained checkpoints, not committed
├── .github/workflows/     # ci.yml
├── .dvc/ · dvc.yaml · dvc.lock · params.yaml
├── Dockerfile · docker-compose.yml · .dockerignore
├── requirements*.txt      # dev / ci / serve / lock
├── check_env.py           # verifies the environment imports cleanly
└── Makefile
```

---

## Problems I hit and how I solved them

Most of these came from developing on a locked-down Windows machine with Smart App Control / Application Control enabled.

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | `TypeError: TrainingArguments got an unexpected keyword argument 'warmup_ratio'` | `transformers` v5 removed `warmup_ratio` | Use `warmup_steps` with a float below 1, which is read as a fraction of total steps |
| 2 | `ImportError: Using the Trainer with PyTorch requires accelerate>=1.1.0` | `accelerate` missing | Install `accelerate>=1.1.0` and add it to `requirements.txt` |
| 3 | Commands ran on Python 3.14, not the project's 3.12 | The global `python` pointed to a different interpreter | Recreated the venv with `py -3.12 -m venv`, pinned `3.12` in `.python-version`, and ran everything through the venv's interpreter |
| 4 | `WinError 4551: An Application Control policy has blocked this file` (`torch\lib\c10.dll`) | Policy blocked DLLs inside the venv | `Get-ChildItem -Recurse .venv \| Unblock-File` |
| 5 | `DLL load failed while importing murmurhash` (scikit-learn) | Smart App Control blocks the unsigned compiled `.pyd`; `Unblock-File` does not help. Confirmed in the Code Integrity event log (event 3077) | Dropped the scikit-learn dependency for metrics: accuracy and macro-F1 are computed with NumPy in `src/metrics.py` |
| 6 | Locust failed to start (`gevent` DLL blocked) | Same policy blocking gevent's compiled extensions | Run Locust from the official Docker image instead |
| 7 | `curl` asked for confirmation in PowerShell | `curl` is an alias for `Invoke-WebRequest` | Use `curl.exe` or `Invoke-RestMethod` |
| 8 | PowerShell profile "running scripts is disabled" | Execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| 9 | Locust summary script: `FileNotFoundError` | Locust had never run, so no CSV existed; Locust appends `_stats.csv` to the `--csv` prefix | Run Locust first with `--csv /out/baseline_fastapi`, then summarize |
| 10 | A failing lint could still allow a merge | Jobs are chained with `needs:`, downstream jobs become *skipped*, and skipped counts as passing | Marked all three CI checks as required in the branch ruleset |
| 11 | Locust reported ~2,000 `HTTP 0` failures at the start of the FastAPI run, which made the numbers useless | `docker compose up -d` returns immediately and the port answers before the app has loaded the model; a plain TCP port check was not enough either | Wait until `/health` returns 200 before starting Locust (`Wait-Http`) |
| 12 | p99 of several seconds (max up to 36 s) in early runs | The first requests after start are slow (model warm-up) | Added a 30 s warm-up run whose stats are discarded; the final runs have a max of about 0.6 to 1 s |
| 13 | `ValueError: I/O operation on closed file` printed at the end of the warm-up run | Locust's CSV writer thread outlives the shutdown of the discarded warm-up run | Harmless: it only affects the warm-up CSV, which is deleted; the measured runs finish cleanly |

---

## Roadmap: Session 4 (Drift & Monitoring)

Planned, not implemented yet:

- **Drift detection.** Compare production traffic against a reference sample (input features such as text length and out-of-vocabulary rate, plus prediction/class-mix drift) using PSI.
- **Prometheus.** Expose request count, latency histogram and prediction distribution from the API on `/metrics`.
- **Grafana.** Dashboard for latency, throughput, class distribution and drift scores.
- **Alerts.** Alert rules for drift above threshold, error-rate and latency regressions.

---

## Limitations

- Neutral is the weakest class (minority class); overall accuracy hides this, so report per-class metrics.
- The load-test numbers are for one machine, CPU-only, with the load generator on the same host, and the FastAPI vs BentoML comparison is a single run per service; they are for comparison, not production capacity.
- The canary rollout (rest of Session 3) and Session 4 (drift and monitoring) are not done yet.
- The model directory is not committed; see the Quickstart for training it locally.

---

## License

MIT (add a `LICENSE` file to the repo).