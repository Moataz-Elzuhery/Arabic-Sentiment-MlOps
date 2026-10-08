# Canary rollout: v1 (stable) -> v2 (candidate)

Traffic is split by an nginx router (`canary/default.conf.template`). Each request is
assigned independently (random `$request_id`), so shares are statistical, not exact.

## Stages and promotion criteria

| Stage | Canary share | Promote to next stage only if (measured at this stage) |
|---|---|---|
| 0 | 0% (`CANARY_ENABLED=0`) | v2 container is healthy, `/predict` answers |
| 1 | 5% | 0 failed requests; canary p95 <= 1.2 x stable p95 |
| 2 | 25% | same |
| 3 | 50% | same |
| 4 | 100% | v2 becomes the new stable; keep v1 image for rollback |

The F1 gate is separate and runs before any rollout: `python -m scripts.quality_gate`
(CI job "Quality gate").

## Commands (PowerShell)

```powershell
$env:CANARY_PCT=5; docker compose --profile canary up -d --build
python -m scripts.canary_check --url http://localhost:8080 --n 400 --expected 5

# per-version latency at this stage (60 s, 10 users)
docker run --rm -e TAG_VERSION=1 -v "${PWD}\loadtest:/mnt/locust" -v "${PWD}\benchmarks\raw:/out" locustio/locust -f /mnt/locust/locustfile.py --host http://host.docker.internal:8080 --headless -u 10 -r 2 -t 60s --reset-stats --csv /out/canary_5pct
python -m scripts.locust_summary --csv benchmarks/raw/canary_5pct_stats.csv --label canary_5pct --users 10 --duration 60 --per-name

# next stage
$env:CANARY_PCT=25; docker compose --profile canary up -d router

# rollback (instant): everything back to v1
$env:CANARY_ENABLED=0; docker compose --profile canary up -d router
```

Note: nginx rejects `0%`, so rollback uses `CANARY_ENABLED=0`, not `CANARY_PCT=0`.

Alternative that avoids leftover shell variables: edit `CANARY_PCT` / `CANARY_ENABLED` in `.env`
and run `docker compose --profile canary up -d router`. (A variable set with `$env:` in the
current terminal overrides `.env`; remove it with `Remove-Item Env:CANARY_PCT`.)

## Rollout log (fill in with measured numbers)

nginx splits traffic between the stable model (v1) and the candidate (v2). The share is set with `CANARY_PCT`, and each stage is checked with `scripts/canary_check.py` (routing) and a Locust run (10 users, per-version stats) before moving on.

**Promotion rule:** promote when failures = 0, v2 p95 ≤ 1.1 × v1 p95, and the observed v2 share is within 3σ of the target. Otherwise roll back (`CANARY_ENABLED=0`).

| Stage | Share | Observed v2 share | v1 p95 (ms) | v2 p95 (ms) | Failures | Decision |
|---|---|---|---|---|---|---|
| 1 | 5% | 5.1% (111 / 2157) | 390 | 260 | 0 | Promote |
| 2 | 25% | 25.4% (280 / 1102) | 370 | 310 | 0 | Promote |
| 3 | 50% | 48.8% (522 / 1070) | 370 | 350 | 0 | Promote |
| 4 | 100% | 100% (961 / 961) | n/a | 510 | 0 | Promoted |

| Stage | v1 req/s | v2 req/s | v1 p50 (ms) | v2 p50 (ms) | v1 p99 (ms) | v2 p99 (ms) |
|---|---|---|---|---|---|---|
| 5% | 17.91 | 0.97 | 210 | 100 | 500 | 470 |
| 25% | 15.12 | 5.15 | 180 | 140 | 420 | 410 |
| 50% | 10.08 | 9.60 | 170 | 180 | 430 | 420 |
| 100% | n/a | 17.81 | n/a | 230 | n/a | 620 |

Notes:
- At 5% v2 served only 111 requests, so its p95 is less stable than in the other stages.
- At 100% v2 carries the whole load alone, which explains the higher p95 (510 ms). There is no v1-only run under identical settings, so this is not a like-for-like comparison with v1.
- All numbers are from one run per stage on a single CPU-only machine, with Locust on the same host.