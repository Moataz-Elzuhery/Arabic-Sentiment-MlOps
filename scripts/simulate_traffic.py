"""Send review-like traffic to /predict so the monitoring stack has something to show.

  normal : real validation reviews (what the model was built for)
  short  : only the first 1-3 words of a review (like "تحفة", "مش بطال", "وحش")
  long   : three reviews concatenated

This is SYNTHETIC drift for demonstration, not a real change in user behaviour.

python -m scripts.simulate_traffic --url http://localhost:8000 --mode normal --n 200 --rate 5
"""
import argparse
import csv
import json
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path

FALLBACK = [
    "المنتج ممتاز جدًا والخامة كويسة وسعره مناسب",
    "الخدمة سيئة والمنتج وصل مكسور ومحدش رد عليا",
    "المنتج وصل في معاده ومطابق للوصف تقريبًا",
    "الفندق جميل والموظفين متعاونين لكن الموقع بعيد شوية عن وسط البلد",
    "تجربة سيئة للغاية ولن أكررها أبدًا والغرفة كانت غير نظيفة",
    "الإطلالة خيالية والفطار متنوع والاستقبال سريع في إنهاء الإجراءات",
]


def load_base(path: str) -> list[str]:
    if Path(path).exists():
        with open(path, newline="", encoding="utf-8") as f:
            return [r["text"] for r in csv.DictReader(f)][:5000]
    print(f"{path} not found, using a tiny built-in sample")
    return FALLBACK


def make_text(base: list[str], mode: str, rng: random.Random) -> str:
    if mode == "short":
        return " ".join(rng.choice(base).split()[:rng.randint(1, 3)])
    if mode == "long":
        return " ".join(rng.choices(base, k=3))[:2000]
    return rng.choice(base)[:2000]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--mode", choices=["normal", "short", "long"], default="normal")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rate", type=float, default=5.0, help="requests per second")
    ap.add_argument("--source", default="data/processed/val.csv")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    base = load_base(a.source)
    labels, errors, latencies = Counter(), 0, []
    for i in range(a.n):
        text = make_text(base, a.mode, rng) or "جيد"
        req = urllib.request.Request(a.url + "/predict", data=json.dumps({"text": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                labels[json.loads(r.read())["label"]] += 1
        except Exception:  # noqa: BLE001
            errors += 1
        latencies.append(time.perf_counter() - start)
        time.sleep(max(0.0, 1 / a.rate - latencies[-1]))
    ok = a.n - errors
    mean_ms = 1000 * sum(latencies) / max(len(latencies), 1)
    print(f"mode={a.mode} sent={a.n} ok={ok} errors={errors} mean latency={mean_ms:.0f} ms")
    print("labels:", dict(labels))
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
