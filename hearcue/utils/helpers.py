"""Utility helpers shared by multiple HearCue subsystems."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import soundfile as sf


FloatArray = np.ndarray


def ensure_mono(signal: FloatArray) -> FloatArray:
    """Ensure waveform is mono by averaging channels if necessary."""
    if signal.ndim == 1:
        return signal
    return signal.mean(axis=1)


def load_audio(path: str | Path, target_sr: int) -> Tuple[FloatArray, int]:
    """Load an audio file and optionally resample using librosa."""
    data, sr = sf.read(str(path), always_2d=False)
    data = ensure_mono(data.astype(np.float32))
    if sr == target_sr:
        return data, sr
    # Lazy import to avoid librosa dependency unless resampling needed.
    import librosa

    resampled = librosa.resample(y=data, orig_sr=sr, target_sr=target_sr)
    return resampled.astype(np.float32), target_sr


def softmax(logits: FloatArray) -> FloatArray:
    """Compute numerically stable softmax."""
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def normalize_signal(signal: FloatArray) -> FloatArray:
    """Scale signal to unit range for consistent processing."""
    max_val = np.max(np.abs(signal)) + 1e-10
    return signal / max_val


def rolling_mean(values: Sequence[float], window: int) -> float:
    """Return rolling mean of last *window* values."""
    if not values:
        return 0.0
    window = min(len(values), window)
    return float(np.mean(values[-window:]))


def chunk_iterable(items: Sequence, chunk_size: int) -> Iterable[List]:
    """Yield slices of size chunk_size from sequence."""
    for idx in range(0, len(items), chunk_size):
        yield list(items[idx : idx + chunk_size])


def sliding_windows(signal: FloatArray, window: int, hop: int) -> Iterable[FloatArray]:
    """Generate overlapping windows from signal."""
    for start in range(0, len(signal) - window + 1, hop):
        yield signal[start : start + window]


def pairwise(iterable: Sequence[int]) -> Iterable[Tuple[int, int]]:
    """Yield neighbor pairs."""
    for idx in range(len(iterable) - 1):
        yield iterable[idx], iterable[idx + 1]


def write_npz(path: str | Path, **arrays: FloatArray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
