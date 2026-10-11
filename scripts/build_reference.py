"""Build the drift reference sample: validation reviews scored by the trained model.

Only lengths, confidences and labels are stored, never the review text.

python -m scripts.build_reference --model-dir models/v1 --val-path data/processed/val.csv \
    --n 2000 --output monitoring/reference.csv
"""
import argparse
import csv
import random
from pathlib import Path


def build_rows(texts, predict_fn, batch_size: int = 32) -> list[dict]:
    rows = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for text, result in zip(batch, predict_fn(batch)):
            rows.append({"text_length": len(text), "confidence": round(result["confidence"], 4),
                         "label": result["label"]})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models/v1")
    p.add_argument("--val-path", default="data/processed/val.csv")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--output", default="monitoring/reference.csv")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    with open(a.val_path, newline="", encoding="utf-8") as f:
        texts = [row["text"] for row in csv.DictReader(f)]
    texts = random.Random(a.seed).sample(texts, min(a.n, len(texts)))

    from src.inference import (
        Predictor,  # heavy import (torch), kept out of module import time
    )
    predictor = Predictor(a.model_dir)
    print(f"scoring {len(texts)} validation reviews with {a.model_dir} ...")
    rows = build_rows(texts, predictor.predict)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text_length", "confidence", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
