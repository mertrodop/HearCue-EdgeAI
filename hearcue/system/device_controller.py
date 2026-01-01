"""High-level orchestrator connecting audio, model, and actuators with debug-style UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from hearcue.audio.mic_stream import MicStream
from hearcue.model.infer import TFLiteAudioClassifier
from hearcue.system.haptic import HapticDriver
from hearcue.system.symbolic_ui import SymbolicUI
from hearcue.system.audio_frontend import AudioFrontend, DebugDecisionConfig
from hearcue.utils.constants import AUDIO


@dataclass
class DeviceController:
    classifier: TFLiteAudioClassifier = field(default_factory=TFLiteAudioClassifier)
    haptics: HapticDriver = field(default_factory=HapticDriver)
    ui: SymbolicUI = field(default_factory=SymbolicUI)
    frontend_cfg: DebugDecisionConfig = field(default_factory=DebugDecisionConfig)
    frontend: AudioFrontend = field(init=False)

    allowed_haptics: dict[str, set[str]] = field(
        default_factory=lambda: {
            "library": {"speech", "ringtone", "alarm"},
            "outdoors": {"alarm", "car", "dog", "ringtone", "speech"},
            "home": {"ringtone", "alarm", "speech"},
        }
    )

    def __post_init__(self) -> None:
        def predict_fn(spec: np.ndarray) -> np.ndarray:
            return self.classifier.predict_proba(spec)

        self.frontend = AudioFrontend(predict_fn=predict_fn, config=self.frontend_cfg)

    def process_chunk(self, chunk: np.ndarray) -> Optional[str]:
        res = self.frontend.process_chunk(chunk)
        if res is None:
            return None
        if "error" in res:
            print(res["error"])
            return None

        self.ui.show(
            top_label=res["top_label"],
            top_conf=res["top_conf"],
            margin=res["margin"],
            rms=res["rms"],
            spec_min=res["spec_min"],
            spec_max=res["spec_max"],
            spec_std=res["spec_std"],
            triggered=res["triggered"],
        )

        if res["triggered"] is None or res["triggered"] == "other":
            return None

        allowed = self.allowed_haptics.get(self.ui.mode, set())
        if res["triggered"] in allowed:
            pattern = self._pattern_for_label(res["triggered"])
            self.haptics.emit(pattern)
        return res["triggered"]

    def _pattern_for_label(self, label: str) -> str:
        mapping = {
            "alarm": "modulated",
            "dog": "short",
            "car": "medium",
            "speech": "short",
            "other": "short",
            "ringtone": "short",
        }
        return mapping.get(label, "short")

    def run_from_wav(self, wav_path: str | Path, realtime: bool = False) -> None:
        stream = MicStream(AUDIO.sample_rate, AUDIO.chunk_size, realtime=realtime)
        for chunk in stream.from_wav(wav_path):
            self.process_chunk(chunk)

    def run_stream(self, chunks: Iterable[np.ndarray]) -> None:
        for chunk in chunks:
            self.process_chunk(chunk)
