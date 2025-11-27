"""Simulated microphone stream that feeds data to a ring buffer."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional

import numpy as np

from hearcue.utils.helpers import load_audio, normalize_signal


@dataclass
class MicStream:
    sample_rate: int
    chunk_size: int
    realtime: bool = False
    sleep_factor: float = 1.0

    def from_array(self, data: np.ndarray) -> Generator[np.ndarray, None, None]:
        data = normalize_signal(data.astype(np.float32))
        total = len(data)
        for idx in range(0, total, self.chunk_size):
            chunk = data[idx : idx + self.chunk_size]
            if len(chunk) < self.chunk_size:
                pad = np.zeros(self.chunk_size - len(chunk), dtype=np.float32)
                chunk = np.concatenate((chunk, pad))
            if self.realtime:
                time.sleep(self.chunk_size / self.sample_rate * self.sleep_factor)
            yield chunk

    def from_wav(self, path: str | Path) -> Generator[np.ndarray, None, None]:
        data, sr = load_audio(path, self.sample_rate)
        return self.from_array(data)

    def stream(self, source: Iterable[np.ndarray]) -> Generator[np.ndarray, None, None]:
        for chunk in source:
            yield chunk


def feed_stream(stream: Iterable[np.ndarray], ring_buffer: "RingBuffer") -> None:
    """Continuously write chunks into ring buffer."""
    from hearcue.audio.ring_buffer import RingBuffer

    if not isinstance(ring_buffer, RingBuffer):
        raise TypeError("ring_buffer must be a RingBuffer instance")
    for chunk in stream:
        ring_buffer.write(chunk)
