"""High-level orchestrator connecting audio, model, and actuators."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from hearcue.audio.logmelspec import log_mel_spectrogram
from hearcue.audio.mic_stream import MicStream
from hearcue.audio.ring_buffer import RingBuffer
from hearcue.model.infer import TFLiteAudioClassifier
from hearcue.system.decision_policy import DecisionPolicy
from hearcue.system.haptic import HapticDriver
from hearcue.system.led import LEDDriver
from hearcue.utils.constants import AUDIO


@dataclass
class DeviceController:
    classifier: TFLiteAudioClassifier = field(default_factory=TFLiteAudioClassifier)
    policy: DecisionPolicy = field(default_factory=DecisionPolicy)
    haptics: HapticDriver = field(default_factory=HapticDriver)
    leds: LEDDriver = field(default_factory=LEDDriver)
    ring_buffer: RingBuffer = field(default_factory=lambda: RingBuffer(AUDIO.ring_buffer_size))

    def process_chunk(self, chunk: np.ndarray) -> Optional[str]:
        self.ring_buffer.write(chunk)
        window_size = AUDIO.frame_length * 10
        window = self.ring_buffer.read(window_size)
        if window is None:
            return None
        features = log_mel_spectrogram(window)
        result = self.classifier.classify(features)
        label = self.policy.should_alert(result.confidence, result.label)
        if label is None:
            self.leds.clear()
            return None
        pattern = self._pattern_for_label(label)
        self.haptics.emit(pattern)
        self.leds.set_state(label, mode="blink")
        return label

    def _pattern_for_label(self, label: str) -> str:
        mapping = {
            "doorbell": "short",
            "fire_alarm": "modulated",
            "knock": "triple",
            "speech": "short",
            "vacuum": "medium",
        }
        return mapping.get(label, "short")

    def run_from_wav(self, wav_path: str | Path, realtime: bool = False) -> None:
        stream = MicStream(AUDIO.sample_rate, AUDIO.chunk_size, realtime=realtime)
        for chunk in stream.from_wav(wav_path):
            self.process_chunk(chunk)

    def run_stream(self, chunks: Iterable[np.ndarray]) -> None:
        for chunk in chunks:
            self.process_chunk(chunk)
