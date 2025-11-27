"""Global constants shared across HearCue modules."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConstants:
    sample_rate: int = 16000
    frame_length: int = 512
    hop_length: int = 160
    n_mels: int = 40
    fmin: int = 20
    fmax: int = 7600
    pre_emphasis: float = 0.97
    chunk_size: int = 1024
    ring_buffer_size: int = 4096


@dataclass(frozen=True)
class ModelConstants:
    num_classes: int = 5
    class_labels: tuple[str, ...] = (
        "doorbell",
        "fire_alarm",
        "knock",
        "speech",
        "vacuum",
    )
    model_path: str = "hearcue/model/hearcue_cnn.tflite"
    quant_scale: float = 0.05
    quant_zero_point: int = -2


@dataclass(frozen=True)
class PolicyConstants:
    confidence_threshold: float = 0.6
    smoothing_window: int = 5
    refractory_period_s: float = 4.0


AUDIO = AudioConstants()
MODEL = ModelConstants()
POLICY = PolicyConstants()
