"""Log a trained model folder to MLflow and register it in the Model Registry.

python -m scripts.register_model --model-dir models/v1 --alias baseline
python -m scripts.register_model --model-dir models/v2 --alias production
"""
import argparse
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--name", default="arabic-sentiment")
    p.add_argument("--alias", help="e.g. production / baseline / candidate")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    a = p.parse_args()

    model_dir = Path(a.model_dir)
    meta = json.loads((model_dir / "metrics.json").read_text())
    version_tag = meta.get("model_version", model_dir.name)

    mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_experiment("model-registry")
    with mlflow.start_run(run_name=f"register_{version_tag}") as run:
        mlflow.log_params({**meta["hyperparameters"], "base_model": meta["base_model"]})
        mlflow.log_metrics({"test_accuracy": meta["accuracy"], "test_f1_macro": meta["f1_macro"],
                            **{f"f1_{k}": v for k, v in meta.get("per_class_f1", {}).items()}})
        for f in model_dir.iterdir():
            if f.is_file() and f.name != "training_args.bin":
                mlflow.log_artifact(str(f), artifact_path="model")
        run_id = run.info.run_id
        source = f"{run.info.artifact_uri}/model"

    client = MlflowClient()
    try:
        client.get_registered_model(a.name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(a.name)
    mv = client.create_model_version(a.name, source=source, run_id=run_id)
    if a.alias:
        client.set_registered_model_alias(a.name, a.alias, mv.version)
    print(f"Registered {a.name} version {mv.version}" + (f" with alias '{a.alias}'" if a.alias else ""))


if __name__ == "__main__":
    main()
