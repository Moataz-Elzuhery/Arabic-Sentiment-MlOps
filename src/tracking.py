"""MLflow logging for training runs (params + metrics). Imported lazily by train.py."""


def log_run(metrics: dict, run_name: str, experiment: str, tracking_uri: str) -> str:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({**metrics["hyperparameters"], "base_model": metrics["base_model"]})
        values = {
            "test_accuracy": metrics["accuracy"],
            "test_f1_macro": metrics["f1_macro"],
            "val_f1_macro": metrics["val_f1_macro"],
            "train_seconds": metrics["train_seconds"],
        }
        for label, score in metrics["per_class_f1"].items():
            values[f"f1_{label}"] = score
        mlflow.log_metrics(values)
        return run.info.run_id
