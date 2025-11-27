"""False alarm stress tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from hearcue.simulation.event_player import EventPlayer, Scenario
from hearcue.system.device_controller import DeviceController
from hearcue.utils.constants import AUDIO


@dataclass
class FalseAlarmMetrics:
    false_positives: int
    duration_s: float

    @property
    def per_minute(self) -> float:
        minutes = self.duration_s / 60.0
        return self.false_positives / minutes if minutes else 0.0


class FalseAlarmTester:
    def __init__(self, controller: DeviceController, scenario: Scenario, tolerance_s: float = 1.5) -> None:
        self.controller = controller
        self.player = EventPlayer(scenario)
        self.tolerance = tolerance_s

    def run(self) -> FalseAlarmMetrics:
        schedule = {label: sorted(times) for label, times in self.player.event_schedule().items()}
        consumed: Dict[str, List[bool]] = {label: [False] * len(times) for label, times in schedule.items()}
        processed_samples = 0
        false_positives = 0
        for chunk in self.player.stream():
            processed_samples += len(chunk)
            timestamp = processed_samples / AUDIO.sample_rate
            detection = self.controller.process_chunk(chunk)
            if detection is None:
                continue
            if detection not in schedule:
                false_positives += 1
                continue
            matched = False
            for idx, (t, used) in enumerate(zip(schedule[detection], consumed[detection])):
                if used:
                    continue
                if abs(timestamp - t) <= self.tolerance:
                    consumed[detection][idx] = True
                    matched = True
                    break
            if not matched:
                false_positives += 1
        duration_s = len(self.player._timeline) / AUDIO.sample_rate
        return FalseAlarmMetrics(false_positives=false_positives, duration_s=duration_s)
