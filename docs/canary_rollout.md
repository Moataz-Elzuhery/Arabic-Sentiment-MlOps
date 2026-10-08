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

| Stage | Share | Observed v2 share | v1 p95 (ms) | v2 p95 (ms) | Failures | Decision |
|---|---|---|---|---|---|---|
| 1 | 5% | | | | | |
| 2 | 25% | | | | | |
| 3 | 50% | | | | | |
| 4 | 100% | | | | | |
