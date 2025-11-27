"""Outdoor street scenario."""
from __future__ import annotations

from hearcue.simulation.event_player import AudioEvent, Scenario


def build_scenario() -> Scenario:
    events = [
        AudioEvent("fire_alarm", start_s=10.0, duration_s=3.0, amplitude=1.0),
        AudioEvent("speech", start_s=30.0, duration_s=2.0, amplitude=0.5),
    ]
    return Scenario(name="outdoor", length_s=50.0, noise_level=0.05, events=events, difficulty=1.5)
