"""System Usability Scale calculator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SUSSurvey:
    responses: List[int]

    def score(self) -> float:
        if len(self.responses) != 10:
            raise ValueError("SUS requires exactly 10 responses")
        adjusted = []
        for idx, value in enumerate(self.responses):
            if idx % 2 == 0:
                adjusted.append(value - 1)
            else:
                adjusted.append(5 - value)
        return sum(adjusted) * 2.5
