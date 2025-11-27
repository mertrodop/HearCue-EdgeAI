"""Scenario and event simulation utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np

from hearcue.audio.mic_stream import MicStream
from hearcue.utils.constants import AUDIO


@dataclass
class AudioEvent:
    label: str
    start_s: float
    duration_s: float
    amplitude: float = 0.8


@dataclass
class Scenario:
    name: str
    length_s: float
    noise_level: float
    events: List[AudioEvent] = field(default_factory=list)
    difficulty: float = 1.0


class EventPlayer:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.sample_rate = AUDIO.sample_rate
        self._timeline = self._synthesize()

    def _synthesize(self) -> np.ndarray:
        num_samples = int(self.scenario.length_s * self.sample_rate)
        noise = np.random.normal(scale=self.scenario.noise_level, size=num_samples).astype(np.float32)
        timeline = noise
        for event in self.scenario.events:
            start = int(event.start_s * self.sample_rate)
            length = int(event.duration_s * self.sample_rate)
            waveform = self._event_waveform(event.label, length, event.amplitude)
            end = min(start + length, num_samples)
            segment = waveform[: end - start]
            timeline[start:end] += segment
        return timeline

    def _event_waveform(self, label: str, length: int, amplitude: float) -> np.ndarray:
        freq_map = {
            "doorbell": 700.0,
            "fire_alarm": 1000.0,
            "knock": 200.0,
            "speech": 350.0,
            "vacuum": 120.0,
        }
        freq = freq_map.get(label, 300.0)
        t = np.arange(length) / self.sample_rate
        waveform = amplitude * np.sin(2 * np.pi * freq * t)
        envelope = np.hanning(length)
        return (waveform * envelope).astype(np.float32)

    def stream(self, chunk_size: int = AUDIO.chunk_size, realtime: bool = False) -> Iterable[np.ndarray]:
        mic = MicStream(self.sample_rate, chunk_size, realtime=realtime)
        return mic.from_array(self._timeline)

    def event_schedule(self) -> Dict[str, List[float]]:
        schedule: Dict[str, List[float]] = {}
        for event in self.scenario.events:
            schedule.setdefault(event.label, []).append(event.start_s)
        return schedule
