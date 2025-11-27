"""Haptic driver simulation for vibration motor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class HapticDriver:
    pattern_definitions: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "short": [0.1],
            "triple": [0.05, 0.05, 0.05],
            "modulated": [0.1, 0.05, 0.2],
            "medium": [0.2],
        }
    )
    history: List[str] = field(default_factory=list)

    def emit(self, pattern: str) -> None:
        if pattern not in self.pattern_definitions:
            raise ValueError(f"Unknown haptic pattern: {pattern}")
        self.history.append(pattern)
        # Real hardware API call would happen here.

    def last_pattern(self) -> str | None:
        return self.history[-1] if self.history else None
