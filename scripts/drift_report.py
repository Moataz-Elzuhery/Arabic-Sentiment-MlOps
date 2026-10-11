"""Evidently drift report: the reference sample vs the most recent logged predictions.

python -m scripts.drift_report --reference monitoring/reference.csv \
    --log monitoring/data/predictions.jsonl --last 300 --output reports/drift_report.html

The PSI shown here uses Evidently's own binning, so its values are not identical to the
`sentiment_drift_psi` gauge (reference-quantile bins); both should tell the same story.
"""
import argparse
import json
from pathlib import Path

FEATURES = ["text_length", "confidence"]


def load_log(path: str, last: int | None = None) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return rows[-last:] if last else rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="monitoring/reference.csv")
    ap.add_argument("--log", default="monitoring/data/predictions.jsonl")
    ap.add_argument("--last", type=int, default=300, help="use only the last N logged requests")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--output", default="reports/drift_report.html")
    a = ap.parse_args()

    import pandas as pd
    from evidently import Report
    from evidently.presets import DataDriftPreset

    reference = pd.read_csv(a.reference)[FEATURES]
    current = pd.DataFrame(load_log(a.log, a.last))[FEATURES]
    print(f"reference rows: {len(reference)}   current rows: {len(current)}")

    report = Report([DataDriftPreset(columns=FEATURES, method="psi", threshold=a.threshold)])
    snapshot = report.run(reference_data=reference, current_data=current)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out))

    for metric in snapshot.dict()["metrics"]:
        name, value = metric.get("metric_name", ""), metric.get("value")
        if name.startswith("ValueDrift"):
            print(f"{name}: PSI = {value:.3f}  ->  {'DRIFT' if value > a.threshold else 'ok'}")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
