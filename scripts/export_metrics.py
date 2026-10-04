"""Copy the headline metrics of a trained model folder into a small JSON tracked by git.

python -m scripts.export_metrics --model-dir models/v2 --output metrics/production.json
python -m scripts.export_metrics --model-dir models/v1 --output metrics/baseline.json
"""
import argparse
import json
from pathlib import Path

KEYS = ["model_version", "base_model", "accuracy", "f1_macro", "val_f1_macro", "per_class_f1"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    meta = json.loads((Path(a.model_dir) / "metrics.json").read_text())
    out = {k: meta[k] for k in KEYS if k in meta}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {a.output}: {out}")


if __name__ == "__main__":
    main()
