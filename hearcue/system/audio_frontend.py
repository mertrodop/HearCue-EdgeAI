from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from hearcue.utils.constants import AUDIO, MODEL
from hearcue.utils.helpers import normalize_signal
from hearcue.audio.logmelspec import log_mel_spectrogram
from hearcue.audio.ring_buffer import RingBuffer


@dataclass
class DebugDecisionConfig:
    train_mean: float = -10.733114242553711
    train_std: float = 5.043337821960449
    frames_expected: int = 197
    rms_silence: float = 0.03
    min_margin: float = 0.25
    class_thresholds: Dict[str, float] | None = None
    smooth_window: int = 5
    required_hits: Dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.class_thresholds is None:
            self.class_thresholds = {
                "dog": 0.60,
                "car": 0.65,
                "speech": 0.70,
                "ringtone": 0.75,
                "alarm": 0.80,
                "other": 0.0,
            }
        if self.required_hits is None:
            self.required_hits = {
                "dog": 3,
                "car": 3,
                "speech": 4,
                "ringtone": 4,
                "alarm": 4,
            }


class AudioFrontend:
    """
    Converts streaming audio chunks into model probabilities and debug stats,
    matching realtime_pc_debug.py behavior.
    """

    def __init__(self, predict_fn, config: DebugDecisionConfig | None = None):
        self.predict_fn = predict_fn
        self.cfg = config or DebugDecisionConfig()
        self.labels: List[str] = list(MODEL.class_labels)

        self.window_samples = AUDIO.frame_length + AUDIO.hop_length * (self.cfg.frames_expected - 1)
        self.rb = RingBuffer(size=max(getattr(AUDIO, "ring_buffer_size", 8192), self.window_samples * 2))
        self.history: List[Optional[str]] = []

    def process_chunk(self, chunk: np.ndarray) -> Optional[dict]:
        chunk = np.asarray(chunk, dtype=np.float32)
        self.rb.write(chunk)

        wave = self.rb.read(self.window_samples)
        if wave is None:
            return None

        rms = float(np.sqrt(np.mean(wave * wave)))

        if rms < self.cfg.rms_silence:
            probs = np.zeros(len(self.labels), dtype=np.float32)
            probs[self.labels.index("other")] = 1.0
            return {
                "probs": probs,
                "rms": rms,
                "top_label": "other",
                "top_conf": 1.0,
                "margin": 1.0,
                "spec_min": None,
                "spec_max": None,
                "spec_std": None,
                "triggered": None,
            }

        wave = normalize_signal(wave.astype(np.float32))
        spec = log_mel_spectrogram(
            wave,
            sample_rate=AUDIO.sample_rate,
            frame_length=AUDIO.frame_length,
            hop_length=AUDIO.hop_length,
            n_mels=AUDIO.n_mels,
            fmin=AUDIO.fmin,
            fmax=AUDIO.fmax,
        )
        spec = (spec - self.cfg.train_mean) / (self.cfg.train_std + 1e-6)

        if spec.shape[0] != self.cfg.frames_expected or spec.shape[1] != AUDIO.n_mels:
            return {"error": f"BAD SHAPE: {spec.shape}", "rms": rms}

        probs = np.asarray(self.predict_fn(spec), dtype=np.float32)

        sorted_idx = np.argsort(probs)
        top_i = int(sorted_idx[-1])
        top2_i = int(sorted_idx[-2]) if len(sorted_idx) > 1 else top_i
        top_label = self.labels[top_i]
        top_conf = float(probs[top_i])
        margin = float(probs[top_i] - probs[top2_i])

        if margin < self.cfg.min_margin or top_conf < self.cfg.class_thresholds.get(top_label, 0.0):
            label = None
        else:
            label = top_label

        self.history.append(label)
        if len(self.history) > self.cfg.smooth_window:
            self.history.pop(0)

        triggered = None
        if label is not None:
            hits = sum(1 for l in self.history if l == label)
            need = self.cfg.required_hits.get(label, self.cfg.smooth_window)
            if hits >= need:
                triggered = label

        return {
            "probs": probs,
            "rms": rms,
            "top_label": top_label,
            "top_conf": top_conf,
            "margin": margin,
            "spec_min": float(spec.min()),
            "spec_max": float(spec.max()),
            "spec_std": float(spec.std()),
            "triggered": triggered,
        }
