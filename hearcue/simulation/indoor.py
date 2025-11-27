"""Indoor apartment scenario."""
from __future__ import annotations

from hearcue.simulation.event_player import AudioEvent, Scenario


def build_scenario() -> Scenario:
    events = [
        AudioEvent("doorbell", start_s=5.0, duration_s=1.5, amplitude=0.9),
        AudioEvent("speech", start_s=15.0, duration_s=2.5, amplitude=0.5),
        AudioEvent("knock", start_s=25.0, duration_s=1.0, amplitude=0.8),
    ]
    return Scenario(name="indoor", length_s=40.0, noise_level=0.02, events=events, difficulty=1.0)
