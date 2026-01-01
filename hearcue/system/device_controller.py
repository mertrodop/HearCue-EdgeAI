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

    # how many audio frames to include in the inference window
    window_frames: int = 10

    # per-window normalization (helps in real-time environments)
    normalize_features: bool = True

    ring_buffer: RingBuffer = field(init=False)

    def __post_init__(self) -> None:
        window_size = AUDIO.frame_length * self.window_frames
        # Make sure the ring buffer can always serve the requested window.
        # Use 2x window for safety so reads keep working smoothly.
        rb_size = max(AUDIO.ring_buffer_size, window_size * 2)
        self.ring_buffer = RingBuffer(rb_size)

    def process_chunk(self, chunk: np.ndarray) -> Optional[str]:
        chunk = np.asarray(chunk, dtype=np.float32)
        self.ring_buffer.write(chunk)

        window_size = AUDIO.frame_length * self.window_frames
        window = self.ring_buffer.read(window_size)
        if window is None:
            return None

        # Feature extraction (explicit params so it always matches constants)
        features = log_mel_spectrogram(
            window,
            sample_rate=AUDIO.sample_rate,
            frame_length=AUDIO.frame_length,
            hop_length=AUDIO.hop_length,
            n_mels=AUDIO.n_mels,
            fmin=AUDIO.fmin,
            fmax=AUDIO.fmax,
        )

        # Optional per-window standardization for stable live outputs
        if self.normalize_features:
            mu = float(features.mean())
            sigma = float(features.std())
            features = (features - mu) / (sigma + 1e-6)

        result = self.classifier.classify(features)
        label, _conf = self.policy.should_alert(result.logits)

        if label is None:
            # Don’t spam clear if you don’t want flicker; but safe to clear.
            self.leds.clear()
            return None

        # Actuate
        pattern = self._pattern_for_label(label)
        self.haptics.emit(pattern)

        # Only set LED if this label exists in current LED color_map
        # (prevents ValueError if LEDDriver isn’t updated for new labels)
        if hasattr(self.leds, "color_map") and label in self.leds.color_map:
            self.leds.set_state(label, mode="blink")
        else:
            self.leds.clear()

        return label

    def _pattern_for_label(self, label: str) -> str:
        # Your model labels: alarm, car, dog, explosion, knock, other, speech
        mapping = {
            "alarm": "modulated",
            "explosion": "triple",
            "knock": "triple",
            "dog": "short",
            "car": "medium",
            "speech": "short",
            "other": "short",
        }
        return mapping.get(label, "short")

    def run_from_wav(self, wav_path: str | Path, realtime: bool = False) -> None:
        stream = MicStream(AUDIO.sample_rate, AUDIO.chunk_size, realtime=realtime)
        for chunk in stream.from_wav(wav_path):
            self.process_chunk(chunk)

    def run_stream(self, chunks: Iterable[np.ndarray]) -> None:
        for chunk in chunks:
            self.process_chunk(chunk)
