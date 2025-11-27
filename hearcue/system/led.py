"""LED driver simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class LEDDriver:
    color_map: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: {
            "doorbell": (0, 255, 0),
            "fire_alarm": (255, 0, 0),
            "knock": (0, 0, 255),
            "speech": (255, 255, 0),
            "vacuum": (255, 80, 0),
        }
    )
    state: Dict[str, str] = field(default_factory=dict)

    def set_state(self, label: str, mode: str = "steady") -> None:
        if label not in self.color_map:
            raise ValueError(f"Unknown LED label {label}")
        if mode not in {"steady", "blink", "pulse"}:
            raise ValueError("Invalid LED mode")
        self.state = {"label": label, "mode": mode, "color": self.color_map[label]}

    def clear(self) -> None:
        self.state = {}
