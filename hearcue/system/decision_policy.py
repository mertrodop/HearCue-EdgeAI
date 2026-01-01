"""Decision policy that mimics MCU behavior."""
from __future__ import annotations

import time
from typing import Deque, Optional

import numpy as np
from collections import deque
from dataclasses import dataclass, field

from hearcue.utils.constants import MODEL, POLICY

# Per-class gates and temporal smoothing parameters
CLASS_THRESHOLDS = {
    "dog": 0.60,
    "car": 0.65,
    "speech": 0.70,
    "ringtone": 0.75,
    "alarm": 0.80,
    "other": 0.0,
}

WINDOW = 5

REQUIRED_HITS = {
    "dog": 3,
    "car": 3,
    "speech": 4,
    "ringtone": 4,
    "alarm": 4,
}

MIN_MARGIN_GLOBAL = 0.25


@dataclass
class DecisionPolicy:
    threshold: float = POLICY.confidence_threshold
    window: int = WINDOW
    refractory_period: float = 0.0
    class_thresholds: dict[str, float] = field(default_factory=lambda: CLASS_THRESHOLDS.copy())
    required_hits: dict[str, int] = field(default_factory=lambda: REQUIRED_HITS.copy())
    hist: Deque[Optional[str]] = field(init=False)
    last_trigger_time: float = field(default=0.0, init=False)


    def __post_init__(self) -> None:
        self.hist = deque(maxlen=self.window)

    def should_alert(self, probs: np.ndarray) -> tuple[Optional[str], float]:
        labels = list(MODEL.class_labels)

        top_i = int(np.argmax(probs))
        top_label = labels[top_i]
        top_conf = float(probs[top_i])
        # margin between top1 and top2
        sorted_idx = np.argsort(probs)
        top2_i = int(sorted_idx[-2]) if len(sorted_idx) > 1 else top_i
        margin = float(probs[top_i] - probs[top2_i])

        # Per-class threshold gate
        thr = self.class_thresholds.get(top_label, self.threshold)
        if top_conf < thr:
            top_label = None
        # Margin gate for tonal / prone classes
        if top_label is not None and margin < MIN_MARGIN_GLOBAL:
            top_label = None

        self.hist.append(top_label)

        # Temporal smoothing: K out of N
        if top_label is not None:
            need = self.required_hits.get(top_label, self.window)
            hits = sum(1 for x in self.hist if x == top_label)
            if hits >= need:
                now = time.monotonic()
                if now - self.last_trigger_time < self.refractory_period:
                    return None, top_conf
                self.last_trigger_time = now
                return top_label, top_conf

        return None, top_conf

    def reset(self) -> None:
        self.hist.clear()
        self.last_trigger_time = 0.0
