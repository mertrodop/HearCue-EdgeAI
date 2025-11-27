"""Latency benchmarking utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from hearcue.simulation.event_player import EventPlayer, Scenario
from hearcue.system.device_controller import DeviceController
from hearcue.utils.constants import AUDIO


@dataclass
class LatencyResult:
    latencies: Dict[str, List[float]]
    misses: Dict[str, int]

    def summary(self) -> Dict[str, float]:
        summary = {}
        for label, values in self.latencies.items():
            if values:
                summary[label] = float(np.mean(values))
        return summary


class LatencyTester:
    def __init__(self, controller: DeviceController, scenario: Scenario) -> None:
        self.controller = controller
        self.player = EventPlayer(scenario)

    def run(self) -> LatencyResult:
        schedule = {label: sorted(times) for label, times in self.player.event_schedule().items()}
        detection_state = {label: [False] * len(times) for label, times in schedule.items()}
        latencies: Dict[str, List[float]] = {label: [] for label in schedule}
        processed_samples = 0
        for chunk in self.player.stream():
            processed_samples += len(chunk)
            timestamp = processed_samples / AUDIO.sample_rate
            detected = self.controller.process_chunk(chunk)
            if detected is None or detected not in schedule:
                continue
            for idx, (event_time, seen) in enumerate(zip(schedule[detected], detection_state[detected])):
                if not seen and timestamp >= event_time:
                    latency = max(0.0, timestamp - event_time)
                    latencies.setdefault(detected, []).append(latency)
                    detection_state[detected][idx] = True
                    break
        misses = {label: statuses.count(False) for label, statuses in detection_state.items()}
        return LatencyResult(latencies=latencies, misses=misses)
