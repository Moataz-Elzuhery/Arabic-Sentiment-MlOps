"""Fine-tune an Arabic BERT for 3-class sentiment.

python -m src.train --config configs/train.yaml
python -m src.train --max-samples 200 --epochs 1   # smoke test
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

from .model import build_model
from .preprocessing import LABEL2ID, LABELS, normalize_arabic


class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        texts = [normalize_arabic(t) for t in df["text"]]
        self.enc = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = [LABEL2ID[l] for l in df["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def load_split(path, max_samples, seed):
    df = pd.read_csv(path)
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=seed)
    return df.reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--output-dir")
    p.add_argument("--model-version")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--run-name", help="MLflow run name")
    p.add_argument("--experiment", default="arabic-sentiment", help="MLflow experiment name")
    p.add_argument("--tracking-uri", help="e.g. sqlite:///mlflow.db (omit to disable MLflow)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for k in ["learning_rate", "batch_size", "epochs", "output_dir", "model_version", "max_samples"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    set_seed(cfg["seed"])
    tokenizer, model = build_model(cfg["base_model"])
    ds = {
        s: ReviewsDataset(load_split(cfg[f"{s}_path"], cfg["max_samples"], cfg["seed"]),
                          tokenizer, cfg["max_length"])
        for s in ["train", "val", "test"]
    }

    out = Path(cfg["output_dir"])
    # warmup_ratio was removed in newer transformers; compute explicit steps (works everywhere)
    steps_per_epoch = -(-len(ds["train"]) // cfg["batch_size"])
    warmup_steps = int(cfg["warmup_ratio"] * steps_per_epoch * cfg["epochs"])
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        num_train_epochs=cfg["epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
        seed=cfg["seed"],
        report_to="none",
        logging_steps=50,
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["val"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    val_metrics = trainer.evaluate(ds["val"])

    pred = trainer.predict(ds["test"], metric_key_prefix="test")
    test_metrics = pred.metrics
    per_class = f1_score(pred.label_ids, pred.predictions.argmax(-1),
                         labels=list(range(len(LABELS))), average=None)
    metrics = {
        "model_version": cfg["model_version"],
        "base_model": cfg["base_model"],
        "labels": LABELS,
        "accuracy": test_metrics["test_accuracy"],
        "f1_macro": test_metrics["test_f1_macro"],
        "val_f1_macro": val_metrics["eval_f1_macro"],
        "train_seconds": round(train_result.metrics["train_runtime"], 1),
        "per_class_f1": {l: round(float(x), 4) for l, x in zip(LABELS, per_class)},
        "hyperparameters": {k: cfg[k] for k in
                            ["learning_rate", "batch_size", "epochs", "max_length", "seed", "max_samples"]},
    }
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    shutil.rmtree(out / "checkpoints", ignore_errors=True)  # keep the output folder small
    shutil.rmtree(out / "checkpoints", ignore_errors=True)  # keep the output folder small (DVC / registry)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.tracking_uri:
        from .tracking import log_run
        run_id = log_run(metrics, args.run_name or cfg["model_version"],
                         args.experiment, args.tracking_uri)
        print("MLflow run:", run_id)


if __name__ == "__main__":
    main()
