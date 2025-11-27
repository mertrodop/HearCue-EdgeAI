"""NASA TLX scoring utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class NASATLXSurvey:
    scores: Dict[str, float]

    def score(self) -> float:
        required = [
            "mental",
            "physical",
            "temporal",
            "performance",
            "effort",
            "frustration",
        ]
        for key in required:
            if key not in self.scores:
                raise ValueError(f"Missing NASA TLX metric: {key}")
        return sum(self.scores[key] for key in required) / len(required)
