"""Aggregate evaluation utilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hearcue.evaluation.compute_f1 import macro_f1, per_class_f1
from hearcue.evaluation.confusion_matrix import plot_confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HearCue predictions from a CSV file.")
    parser.add_argument("predictions", help="CSV with columns y_true,y_pred")
    parser.add_argument("--output", default="evaluation.png")
    args = parser.parse_args()
    df = pd.read_csv(args.predictions)
    y_true = df["y_true"].astype(int).tolist()
    y_pred = df["y_pred"].astype(int).tolist()
    macro = macro_f1(y_true, y_pred)
    per_class = per_class_f1(y_true, y_pred)
    fig = plot_confusion_matrix(y_true, y_pred, normalize=True)
    fig.savefig(args.output)
    print(json.dumps({"macro_f1": macro, "per_class": per_class}, indent=2))
    print(f"Confusion matrix figure saved to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
