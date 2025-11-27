"""F1 score helpers."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from hearcue.utils.constants import MODEL


def per_class_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int | None = None) -> Dict[str, float]:
    num_classes = num_classes or len(MODEL.class_labels)
    scores: Dict[str, float] = {}
    for idx in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == idx and p == idx)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != idx and p == idx)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == idx and p != idx)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        denom = precision + recall
        score = 2 * precision * recall / denom if denom else 0.0
        label = MODEL.class_labels[idx] if idx < len(MODEL.class_labels) else f"class_{idx}"
        scores[label] = score
    return scores


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    scores = per_class_f1(y_true, y_pred)
    return float(np.mean(list(scores.values())))
