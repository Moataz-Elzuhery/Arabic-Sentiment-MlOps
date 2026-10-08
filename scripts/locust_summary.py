"""Turn a Locust *_stats.csv into a small JSON + a markdown table row.

python -m scripts.locust_summary --csv benchmarks/raw/baseline_fastapi_stats.csv \
    --label baseline_fastapi --users 10 --duration 60
"""
import argparse
import csv
import json
from pathlib import Path


def summarize(csv_path: str, label: str, users=None, duration=None) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if r["Name"] == "/predict"), None) or \
        next(r for r in rows if r["Name"] == "Aggregated")
    return {
        "label": label,
        "users": users,
        "duration_s": duration,
        "requests": int(row["Request Count"]),
        "failures": int(row["Failure Count"]),
        "rps": round(float(row["Requests/s"]), 2),
        "avg_ms": round(float(row["Average Response Time"]), 1),
        "p50_ms": float(row["50%"]),
        "p95_ms": float(row["95%"]),
        "p99_ms": float(row["99%"]),
    }


def summarize_all(csv_path: str, label: str, users=None, duration=None) -> list[dict]:
    """One summary per request name starting with /predict (e.g. '/predict [v1]')."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["Name"].startswith("/predict")]
    out = []
    for row in rows:
        out.append({
            "label": label, "name": row["Name"], "users": users, "duration_s": duration,
            "requests": int(row["Request Count"]), "failures": int(row["Failure Count"]),
            "rps": round(float(row["Requests/s"]), 2),
            "avg_ms": round(float(row["Average Response Time"]), 1),
            "p50_ms": float(row["50%"]), "p95_ms": float(row["95%"]), "p99_ms": float(row["99%"]),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--users", type=int)
    p.add_argument("--duration", type=int)
    p.add_argument("--outdir", default="benchmarks")
    p.add_argument("--per-name", action="store_true",
                   help="one row per '/predict [version]' name (canary runs)")
    a = p.parse_args()

    if a.per_name:
        rows = summarize_all(a.csv, a.label, a.users, a.duration)
        out = Path(a.outdir) / f"{a.label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))
        print("| Run | Version | Requests | Req/s | p50 (ms) | p95 (ms) | p99 (ms) | Failures |")
        print("|---|---|---|---|---|---|---|---|")
        for s in rows:
            print(f"| {s['label']} | {s['name']} | {s['requests']} | {s['rps']} | "
                  f"{s['p50_ms']:.0f} | {s['p95_ms']:.0f} | {s['p99_ms']:.0f} | {s['failures']} |")
        return

    s = summarize(a.csv, a.label, a.users, a.duration)
    out = Path(a.outdir) / f"{a.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
    print("\n| Run | Users | Req/s | p50 (ms) | p95 (ms) | p99 (ms) | Failures |")
    print("|---|---|---|---|---|---|---|")
    print(f"| {s['label']} | {s['users']} | {s['rps']} | {s['p50_ms']:.0f} | "
          f"{s['p95_ms']:.0f} | {s['p99_ms']:.0f} | {s['failures']} |")


if __name__ == "__main__":
    main()
