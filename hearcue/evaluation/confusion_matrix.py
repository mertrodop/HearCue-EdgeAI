"""Confusion matrix plotting."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from hearcue.utils.constants import MODEL


def confusion_matrix(y_true, y_pred):
    labels = list(MODEL.class_labels)
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def plot_confusion_matrix(y_true, y_pred, normalize: bool = False, title: str | None = None) -> plt.Figure:
    matrix = confusion_matrix(y_true, y_pred)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
            matrix = np.nan_to_num(matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", ax=ax,
                xticklabels=MODEL.class_labels, yticklabels=MODEL.class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title or "HearCue Confusion Matrix")
    fig.tight_layout()
    return fig
