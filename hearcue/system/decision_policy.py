"""Decision policy that mimics MCU behavior."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from hearcue.utils.constants import POLICY
from hearcue.utils.helpers import rolling_mean


@dataclass
class DecisionPolicy:
    threshold: float = POLICY.confidence_threshold
    smoothing_window: int = POLICY.smoothing_window
    refractory_period: float = POLICY.refractory_period_s
    _history: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    _last_alert: float = field(default=0.0)

    def should_alert(self, confidence: float, label: str) -> Optional[str]:
        if confidence < self.threshold:
            self._history.append(confidence)
            return None
        self._history.append(confidence)
        smoothed = rolling_mean(list(self._history), self.smoothing_window)
        now = time.monotonic()
        if smoothed < self.threshold:
            return None
        if now - self._last_alert < self.refractory_period:
            return None
        self._last_alert = now
        return label

    def reset(self) -> None:
        self._history.clear()
        self._last_alert = 0.0
