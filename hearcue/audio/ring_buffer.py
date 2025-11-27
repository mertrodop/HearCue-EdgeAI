"""Circular buffer for streaming audio."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RingBuffer:
    size: int
    dtype: type = np.float32
    buffer: np.ndarray = field(init=False)
    write_pos: int = field(init=False, default=0)
    is_full: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.buffer = np.zeros(self.size, dtype=self.dtype)

    def write(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=self.dtype)
        n = len(data)
        end = self.write_pos + n
        if end < self.size:
            self.buffer[self.write_pos:end] = data
        else:
            first = self.size - self.write_pos
            self.buffer[self.write_pos:] = data[:first]
            self.buffer[: end % self.size] = data[first:]
        self.write_pos = end % self.size
        if n >= self.size:
            self.is_full = True
        else:
            self.is_full = self.is_full or self.write_pos == 0

    def read(self, length: int, offset: int = 0) -> Optional[np.ndarray]:
        if not self.is_full and self.write_pos < length:
            return None
        start = (self.write_pos - offset - length) % self.size
        if start + length <= self.size:
            return self.buffer[start : start + length].copy()
        first = self.size - start
        return np.concatenate((self.buffer[start:], self.buffer[: length - first])).copy()

    def clear(self) -> None:
        self.buffer.fill(0)
        self.write_pos = 0
        self.is_full = False
