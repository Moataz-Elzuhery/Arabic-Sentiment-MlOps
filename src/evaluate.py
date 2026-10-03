"""Evaluate a trained model folder on the test split and write a small metrics JSON.

python -m src.evaluate --model-dir models/pipeline --test-path data/processed/test.csv \
    --output metrics/eval.json --max-samples 500
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .preprocessing import LABEL2ID, LABELS, normalize_arabic


def evaluate(model_dir, test_path, max_samples=None, batch_size=32, max_length=128, seed=42):
    import torch

    from .model import load_model

    df = pd.read_csv(test_path)
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=seed)  # same sampling rule as train.py
    df = df.reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(model_dir)
    model.to(device).eval()

    texts = [normalize_arabic(t) for t in df["text"]]
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to(device)
            preds.extend(model(**enc).logits.argmax(-1).tolist())

    y_true = [LABEL2ID[label] for label in df["label"]]
    per_class = f1_score(y_true, preds, labels=list(range(len(LABELS))), average=None)
    return {
        "n_samples": len(df),
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "f1_macro": round(float(f1_score(y_true, preds, average="macro")), 4),
        "per_class_f1": {label: round(float(x), 4) for label, x in zip(LABELS, per_class)},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--test-path", default="data/processed/test.csv")
    p.add_argument("--output", default="metrics/eval.json")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()

    metrics = evaluate(a.model_dir, a.test_path, a.max_samples, a.batch_size)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
