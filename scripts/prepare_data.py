"""Convert a raw reviews file into train/val/test CSVs with columns: text,label.

Either the file already has a sentiment label column, or a 1-5 rating column
(1-2 -> negative, 3 -> neutral, 4-5 -> positive).

python -m scripts.prepare_data --input data/raw/reviews.csv --text-col review --rating-col rating
# HARD (utf-16 TSV):
python -m scripts.prepare_data --input data/raw/unbalanced-reviews.txt --encoding utf-16 \\
    --text-col review --rating-col rating
python -m scripts.prepare_data --input data/raw/reviews.csv --text-col text --label-col label
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import LABELS


def rating_to_label(r):
    r = float(r)
    return "negative" if r <= 2 else "neutral" if r == 3 else "positive"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--text-col", required=True)
    p.add_argument("--label-col")
    p.add_argument("--rating-col")
    p.add_argument("--sep", default=None, help="default: ',' for .csv else tab")
    p.add_argument("--encoding", default="utf-8", help="HARD files are utf-16")
    p.add_argument("--outdir", default="data/processed")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    assert bool(a.label_col) != bool(a.rating_col), "give exactly one of --label-col / --rating-col"

    sep = a.sep.replace("\\t", "\t") if a.sep else ("," if a.input.endswith(".csv") else "\t")
    # quoting=3 (QUOTE_NONE): review text contains stray quotes
    df = pd.read_csv(a.input, sep=sep, encoding=a.encoding, quoting=3, on_bad_lines="skip")
    out = pd.DataFrame({"text": df[a.text_col].astype(str).str.strip()})
    out["label"] = (df[a.label_col].astype(str).str.lower().str.strip()
                    if a.label_col else df[a.rating_col].map(rating_to_label))
    out = out[out["label"].isin(LABELS) & (out["text"].str.len() > 0)].drop_duplicates("text")

    Path(a.outdir).mkdir(parents=True, exist_ok=True)  # DVC deletes the output folder before each run
    counts = out["label"].value_counts().to_dict()
    missing = [l for l in LABELS if l not in counts]
    if missing:
        raise SystemExit(f"Missing classes {missing}. Class counts: {counts}")
    train, rest = train_test_split(out, test_size=0.2, stratify=out["label"], random_state=a.seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=a.seed)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(f"{a.outdir}/{name}.csv", index=False)
        print(name, len(part), part["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
