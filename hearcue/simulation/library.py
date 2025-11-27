"""Library quiet scenario emphasizing false alarms."""
from __future__ import annotations

from hearcue.simulation.event_player import AudioEvent, Scenario


def build_scenario() -> Scenario:
    events = [
        AudioEvent("knock", start_s=8.0, duration_s=0.8, amplitude=0.7),
        AudioEvent("doorbell", start_s=20.0, duration_s=1.0, amplitude=0.6),
    ]
    return Scenario(name="library", length_s=35.0, noise_level=0.005, events=events, difficulty=0.8)
